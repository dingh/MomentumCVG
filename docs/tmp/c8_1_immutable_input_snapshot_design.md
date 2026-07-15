# Sprint 004 — C8.1 immutable input snapshot design

**Task:** C8.1B — final corrections to immutable snapshot design
**Mode:** Design correction only
**Starting HEAD:** `d4f05f5089d847992d89e19dc6bc3a8bdf40e1be`
**Grounded in:** accepted C8.0 and C8.0A reality maps.

> This memo specifies the smallest safe C8 implementation that packages and
> revalidates the accepted C4/C5 bundle, rebuilds spot and weekly A1/A2, and
> publishes one immutable input snapshot for Sprint 005. It supersedes the
> earlier recommendations in this file that rebuilt C4/C5 from raw.

---

## 1. Decision summary

| Decision | Final recommendation |
|---|---|
| Isolation boundary | The snapshot root is the transaction boundary. Individual C4/C5/surface files do not need independent transaction protocols. |
| Upstream scope | **Copy the accepted, matching C4/C5 bundle together**; do not rerun liquidity, split fetch, or split adjustment. |
| Derived scope | Rebuild hardened spot and weekly A1/A2 from the copied adjusted chains. |
| Lifecycle | Build under `<build_id>.building`; publish only by same-volume rename to `<build_id>` after acceptable status. |
| Full range | Production copies the entire accepted C4/C5 history. Resolve actual dates from the copied adjusted inventory before spot/surface. |
| Bounded evidence | Build a small isolated date/ticker fixture; every stage uses one `effective_ticker_universe`; mark it non-production. |
| Surface pair atomicity | Not required at producer level; root isolation plus A1/A2 completeness gates is sufficient. |
| Manifest | Keep schema version 1; use final-root `cache_dir`, root-relative paths, expanded identity parameters, and truthful PASS/WARN/FAIL handling. |
| Validation outcome | `production_accepted`, status, warnings, failures, and publication result are execution outcomes, never `snapshot_id` inputs. |
| Handoff | Sprint 005 receives `--input-snapshot-manifest`; canonical feature input is joined A1 plus A2 ATM call and put rows. |
| Build identity | Human-readable UTC timestamp with microseconds plus UUID suffix. |
| Warning policy | Only explicitly allowlisted informational warnings may publish. |
| Implementation | C8.2 spot; C8.3A lifecycle/identity/gates; C8.3B orchestration; C8.4 bounded evidence; C8.5 production evidence/closeout. |

**Boundary statement:** C8 is packaging, revalidating, and extending the
accepted C4/C5 input bundle with newly rebuilt spot and surface artifacts. It
is not re-proving C4/C5 from raw during every snapshot build.

---

## 2. Goal and non-goals

**Goal:** publish one complete, immutable, auditable historical input snapshot
that Sprint 005 can use as the sole upstream input for a trusted full-history
momentum/CVG feature backfill.

**Non-goals:** weekly append, incremental refresh, watermark operation, split
delta detection, selected-ticker repair, spot/surface merge, changed-only
audits, feature formulas, feature generation, strategy research, backtesting,
shadow operation, scheduling, notifications, dashboards, distributed
execution, remote backup, and a `current`/`latest` pointer.

Preserved decisions:

- the root is the isolation and publication boundary;
- producer-level A1/A2 transaction support is unnecessary;
- C8 supports no incremental or repair operation;
- the manifest path is the Sprint 005 handoff;
- Sprint 005 owns feature formulas, feature generation, and feature audits;
- `run_surface_search.py` is not on the C8 critical path.

---

## 3. Sprint 004 → Sprint 005 handoff

**Primary interface:**

```text
--input-snapshot-manifest
C:/MomentumCVG_env/snapshots/<build_id>/manifests/input_snapshot_<snapshot_id>.json
```

Sprint 005 resolves every artifact relative to `manifest.cache_dir`. It must
not rediscover or copy over mutable global paths.

The handoff exposes:

- `snapshot_id`, `build_id`, repository commit SHA, scope, production
  acceptance flag, and requested/resolved range;
- liquidity daily, weekly, panel, and historical ticker superset;
- split history and adjusted-chain root;
- spot-price parquet;
- option-surface A1 and A2;
- audit/completeness reports and aggregate status;
- `upstream_bundle_id` and its component identifiers.

### Canonical feature-source contract

The canonical input for momentum and CVG is:

> the joined A1 metadata row plus the A2 ATM body call and put rows for the
> same `(ticker, entry_date, expiry_date, body_strike)`.

A1 supplies entry/expiry/spot/body-strike and processing status. A2 supplies
the call/put bid, ask, mid, spread, volume, open interest, and Greeks needed
for straddle economics. **A1 and A2 are both required** for the canonical
Sprint 005 path; A2 is not optional supporting detail.

Adjusted chains and spot remain validated lower-level inputs. The liquidity
panel is the PIT eligibility input and `liquid_tickers.csv` is the historical
precompute superset. C8 does not define momentum or CVG formulas.

Sprint 005 copies `snapshot_id`, `build_id`, repository SHA,
`upstream_bundle_id`, resolved range, and consumed artifact paths into its
feature manifest.

---

## 4. Successful snapshot definition

> C8 is complete when the repository can copy and revalidate the accepted
> C4/C5 input bundle into a fresh immutable snapshot root, rebuild and validate
> spot and weekly A1/A2 surface artifacts from that bundle, publish the
> snapshot only after all blocking gates pass, and hand Sprint 005 a manifest
> that correctly resolves the complete A1+A2, liquidity, adjusted-chain, and
> spot inputs needed for a full-history momentum/CVG feature backfill.

- **Complete:** all gates in §10 pass for one internally consistent scope and
  denominator.
- **Accepted:** publication rename succeeded and status is PASS or an
  explicitly allowlisted WARN.
- **Immutable:** no code writes into an existing final root; every retry uses
  a new build root.
- **Usable:** the published manifest resolves all required paths inside its
  final root, and the manifest-resolution smoke passes.

Full-denominator processing completeness means every expected ticker-date was
processed and emitted an A1 row. It does not mean every ticker-date is feature
eligible: valid processed failure rows are complete infrastructure records but
remain ineligible for feature calculation.

---

## 5. Snapshot lifecycle

```text
C:/MomentumCVG_env/snapshots/
    <build_id>.building/
        input/
            liquidity/
            adjusted_liquid/
        cache/
            spot_prices_adjusted.parquet
            option_surface_meta_weekly_<S>_<E>.parquet
            option_surface_quotes_weekly_<S>_<E>.parquet
        reports/
        manifests/
    <build_id>/
    <build_id>.failed/
```

- Generate `build_id` before writes.
- Refuse an existing staging, final, or failed destination; never reuse a
  non-empty root.
- Require staging and final roots on the same local volume.
- Use same-volume directory rename as the only publication point.
- A previously accepted final root is never modified.
- Interruption leaves `.building`; ordinary producer/gate failure is converted
  to `.failed` with a truthful FAIL manifest.
- Retry starts a new empty root with a new `build_id`.
- The same logical inputs may produce the same `snapshot_id`, but every
  execution receives a different `build_id`.
- Do not create a `current` or `latest` pointer.

### Collision-resistant `build_id`

C8.3A replaces the second-resolution, command-derived generator for snapshot
builds with:

```text
YYYYMMDDTHHMMSSffffffZ_<8 lowercase UUID hex>
```

This is human-readable, preserves chronological sorting, and remains unique
for identical invocations in the same microsecond. Tests inject time and UUID
sources and prove two identical commands cannot collide.

---

## 6. Accepted C4/C5 upstream bundle

### Source bundle

Copy these accepted roots:

```text
C:/MomentumCVG_env/input/liquidity/
    ticker_liquidity_daily_observations.parquet
    ticker_liquidity_weekly_observations.parquet
    ticker_liquidity_panel.parquet
    liquid_tickers.csv

C:/MomentumCVG_env/input/adjusted_liquid/
    splits_hist_liquid.parquet
    YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet
```

They are one accepted upstream bundle. Never combine a newly generated
`liquid_tickers.csv` with the older accepted split/adjusted files: absence
from the split table cannot distinguish “queried with no split” from “never
queried.”

### Copy mechanism and post-copy policy

- Before copying, compute source identities for `liquid_tickers.csv`, the split
  file, and the accepted adjusted inventory.
- Full build uses recursive, metadata-preserving file copies into an empty
  `.building` destination (`copy2` semantics; no links, references, or
  in-place reads after copy).
- Refuse any pre-existing destination file.
- Preserve the exact relative paths.
- After copying, check all required C4 files exist and adjusted year
  directories contain only expected daily parquet names plus the split file.
- Run schema checks for C4 artifacts and split history.
- Build the adjusted inventory and reject duplicate dates, malformed names,
  empty inventory, or unexpected non-parquet payloads.
- After copying, recompute the same component identities from the staging
  root and require exact source-to-copy equality before accepting the bundle.
- Rerun C5 `audit_adjusted_liquid.py` against the copied adjusted root and
  copied split/universe files, using the explicitly supplied read-only raw
  root and years derived from the copied adjusted inventory.
- Rerun C7 `audit_pit_universe.py` against the copied liquidity files.
- Do not redesign or rerun C4/C5 producers.

Lineage records the accepted evidence identifiers:

- C4: `docs/sprint_memos/004_c4_liquidity_panel.md` and its accepted production
  evidence (2017–2026-02-20 backfill);
- C5: `docs/sprint_memos/004_c5_adjusted_liquid.md`, path-wiring commit
  `0d2357381e373f217e21ef2213749a5880f195a9`, and C5.10B/C5.10D evidence.

### Source, copied, and verified upstream identities

Before copying, compute:

```text
source_liquid_tickers_sha256 = sha256(source liquid_tickers.csv bytes)
source_splits_sha256         = sha256(source splits_hist_liquid.parquet bytes)

source_adjusted_inventory_digest =
    sha256(canonical JSON of sorted [
        [relative_adjusted_parquet_path, file_size_bytes], ...
    ])
```

After copying, compute independently from the staging root:

```text
copied_liquid_tickers_sha256
copied_splits_sha256
copied_adjusted_inventory_digest
```

Require exact equality:

```text
source_liquid_tickers_sha256 == copied_liquid_tickers_sha256
source_splits_sha256 == copied_splits_sha256
source_adjusted_inventory_digest == copied_adjusted_inventory_digest
```

Only after equality passes, compute:

```text
upstream_bundle_id =
    sha256(canonical JSON {
        c4_evidence_id,
        c5_evidence_id,
        source_liquid_tickers_sha256,
        source_splits_sha256,
        source_adjusted_inventory_digest
    })[:16]
```

Canonical JSON uses sorted keys and compact separators. Source and copied
inventory paths use the same accepted-root-relative, forward-slashed form and
exclude the split file.

This proves the accepted bundle identity and adjusted-file inventory, **not
byte-level identity of every large parquet**. The accepted source roots are
treated as immutable inputs by policy. Full-byte hashes are required only for
the small `liquid_tickers.csv` and split-history files.

The completeness report records both source and copied component values and
their equality verdicts. The manifest may expose only the verified canonical
source values plus `upstream_bundle_id`. `upstream_bundle_id` is
identity-relevant to `snapshot_id`.

---

## 7. Full and bounded scope contracts

### Full production scope

- Copy the **entire** accepted C4/C5 bundle.
- Do not accept start/end arguments that truncate production scope.
- Discover actual resolved start/end from the copied adjusted inventory
  before spot or surface execution.
- Record each artifact's actual coverage independently. Supporting artifacts
  may begin earlier than the surface (for example, 2017 liquidity history may
  provide rolling PIT warm-up for 2018 surface entries).
- Derive the weekly surface schedule and a dependency-aware,
  **calendar-level** `feature_ready_range` only after all artifact ranges are
  known:

  ```text
  feature_ready_start =
      max(first entry with a valid strict-prior PIT snapshot,
          first entry supported by complete spot/chain inputs,
          first complete A1/A2 entry date)

  feature_ready_end =
      min(last entry with valid PIT coverage,
          last entry whose entry/expiry inputs are complete,
          last complete A1/A2 entry date)
  ```

  Identical artifact start dates are not required. This range means the
  infrastructure is ready: strict-prior PIT schedule support exists, required
  adjusted and spot entry/expiry dates exist, and the complete expected A1
  processing schedule was emitted. It does **not** mean every ticker-date is
  feature-valid.
- `effective_ticker_universe` is the complete, normalized copied
  `liquid_tickers.csv` universe.
- Set `scope="full"` and `production_accepted=true` only after successful
  publication.

The manifest records the actual accepted bundle range. An optional operator
label may describe an intended range, but it cannot alter the production
denominator.

### Bounded evidence scope

Use one concrete method:

1. Create a bounded isolated fixture under `.building`.
2. Copy the four C4 artifacts needed for path/lifecycle validation, while
   clearly labeling them as fixture provenance rather than a production C4
   bundle.
3. Copy the accepted split file.
4. Copy a **small date window** of adjusted daily files and filter each copied
   parquet to a validated small ticker list when preparing the fixture.
5. Run hardened spot and surface only against that fixture.

Fixture preparation is C8.4 evidence tooling, not a production producer and
not an incremental/merge capability. It does not invoke
`apply_split_adjustment.py`, which has year-only scope and no arbitrary date
filter.

For bounded evidence:

```text
effective_ticker_universe = normalized validated bounded ticker list
scope = "bounded"
production_accepted = false
```

The same effective universe controls copied adjusted rows, spot denominator,
surface ticker input, expected A1 keys, A2 checks, reports, and manifest
params. A five-ticker artifact is never compared with the 2,783-ticker full
denominator. Bounded publication may exercise rename mechanics, but it cannot
be presented to Sprint 005 as the accepted production input.

`production_accepted` is a validation/publication outcome. It does not
participate in `snapshot_id`; bounded and full logical identities differ
through scope, universe hash, ranges, bundle identity, and artifact
definitions.

---

## 8. Producer policies

| Producer | C8 policy |
|---|---|
| `build_liquidity_panel.py` | Do not run. Copy accepted C4 and revalidate. |
| `fetch_splits.py` | Do not run. Copy the matching accepted C5 split file. |
| `apply_split_adjustment.py` | Do not run. Copy accepted C5 adjusted chains. |
| `extract_spot_prices.py` | Harden in C8.2 and rebuild against copied/fixture chains. |
| `precompute_option_surface.py` | Use with explicit copied-chain, rebuilt-spot, output-root, date, and effective-universe arguments; gate outputs. |

Producer-level A1/A2 transaction support remains unnecessary. Failure between
A1 and A2 writes leaves an unpublished staging root; Gate SF requires both
artifacts and the final root is never created until the pair passes.

### C8.2 spot contract

Before `groupby("ticker").first()`, for every `(date, ticker)` verify that all
source rows agree on both `stkPx` and `adj_stkPx` using:

```text
math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-8)
```

Reject nonfinite values before comparison and fail closed on inconsistency.
The tolerance is deliberately much tighter than market tick size: repeated
underlying spot fields in one daily chain should be numerically identical
apart from serialization noise.

Then require:

- one row per expected `(date, ticker)`;
- unique `(date, ticker)` grain;
- finite, positive adjusted and raw spot;
- exact expected date set;
- exact per-date ticker equality with the adjusted fixture/full daily file;
- no failed or empty dates;
- temporary output followed by same-directory `os.replace`;
- nonzero exit on any failure and no final success artifact.

C8.2 tests cover happy path, empty output, missing date, duplicate key,
null/nonpositive/nonfinite value, per-date ticker mismatch, read failure,
atomic replacement, and inconsistent repeated source `stkPx`/`adj_stkPx`
inside one ticker-date.

---

## 9. Orchestration

`scripts/build_input_snapshot.py` performs:

1. Parse arguments and validate path syntax.
2. Generate collision-resistant `build_id`.
3. Resolve staging, final, and failed roots.
4. On `--dry-run`, print paths, sources, scope, commands, and gates and exit
   before any write.
5. Compute source C4/C5 component identities.
6. Create a fresh empty `.building` root.
7. Copy the accepted C4 liquidity bundle.
8. Copy the accepted C5 split history and adjusted chains.
9. Compute copied component identities.
10. Require exact source-to-copy identity equality.
11. Compute verified `upstream_bundle_id`.
12. Run copied-bundle schema and inventory gates.
13. Derive adjusted years and the resolved date range from copied inventory.
14. Run C5 adjusted audit with read-only `--raw-root`, copied adjusted/split/
    universe paths, derived `--years`, and a staging-root report path.
15. Run C7 PIT audit against the copied liquidity bundle.
16. Resolve `effective_ticker_universe`.
17. Run hardened spot extraction against copied chains.
18. Run Gate SP.
19. Run weekly surface production with resolved schedule and effective
    universe.
20. Run Gate SF and C6 audit.
21. Derive calendar-level `feature_ready_range`.
22. Record observation-level feature-eligibility semantics.
23. Run cross-layer Gate X.
24. Aggregate warnings through the explicit allowlist.
25. Write the completeness report.
26. Write the manifest atomically with **planned final root** as `cache_dir`
    and root-relative paths.
27. Publish `.building → final` only when status is acceptable.
28. On any failure, retain `.failed` with a truthful FAIL manifest.

Bounded evidence replaces steps 5–10 with fixture source/copy identity and
preparation described in §7,
sets the bounded effective universe before denominator computation, and can
never set `production_accepted=true`.

---

## 10. Completeness gates

Each gate independently checks producer/copy success, existence, internal
consistency, completeness, and cross-layer compatibility.

### Gate L — copied liquidity

- four required files exist and have expected schemas;
- daily/weekly/panel dates and watermarks are internally consistent;
- normalized `liquid_tickers.csv` is non-empty and unique;
- copied `liquid_tickers.csv` sha256 exactly equals the independently computed
  source sha256;
- C7 PIT audit passes;
- source and copied hashes plus evidence identifiers are recorded.

### Gate S — copied split history

- file and required schema exist;
- divisor values are valid;
- no conflicting `(ticker, split_date)` records;
- all split tickers are in the **full accepted** liquid superset;
- copied split-file sha256 exactly equals the independently computed source
  sha256;
- source and copied hashes plus evidence lineage are recorded.

No requirement says every liquid ticker appears in the split table. Matching
C4/C5 provenance establishes that no-split tickers were in the accepted fetch
scope.

### Gate A — copied adjusted chains

- inventory is non-empty and date filenames are unique and parseable;
- no adjusted file is missing or additional relative to the accepted source
  inventory;
- each copied adjusted file size equals its source inventory size, expressed
  by exact equality of source and copied adjusted inventory digests;
- required adjusted columns exist;
- derive the audited year list from copied adjusted inventory;
- C5 adjusted audit passes with `--raw-root` (read-only), copied
  `--adj-root`, copied `--splits`, copied `--ticker-universe`, derived
  `--years`, and a report path under staging.

The production resolved range is computed from this inventory **before** Gates
SP/SF. Gate A does not depend on a not-yet-resolved requested range.
Raw ORATS is read only for source inventory comparison and sampled
raw-vs-adjusted math; raw files are neither copied nor modified.

### Gate SP — rebuilt spot

- hardened producer exits zero;
- date set exactly equals adjusted inventory dates;
- each date's ticker set exactly equals the tickers in that adjusted file
  (bounded files have already been restricted to the effective universe);
- `(date, ticker)` is unique;
- values are finite and positive;
- no failed dates and no intra-ticker source inconsistencies.

### Gate SF — rebuilt surface

```text
expected_A1_keys =
    effective_ticker_universe
    × resolved_weekly_entry_date_schedule
```

- actual A1 keys exactly equal expected keys;
- missing or unexpected A1 keys fail;
- exactly one A1 row per key;
- a documented `surface_valid=false` row with a valid `failure_reason` counts
  as processed and complete;
- that valid processed failure row requires no A2 body pair and is not
  feature eligible;
- a key omitted because it was never processed is incomplete and fails;
- A1 grain is unique;
- A2 `(ticker, entry_date, expiry_date, strike, side)` grain is unique;
- every A2 row joins to A1;
- every `surface_valid=true` A1 has both ATM body rows in A2:
  `side=call` and `side=put` at the A1 `(expiry_date, body_strike)`;
- weekly entry/expiry policy passes;
- C6 audit passes.

Expected no-trade failures do not reduce completeness. Coverage and validity
rate are distinct metrics.

Define observation-level eligibility without defining feature formulas:

```text
feature_eligible_observation =
    A1 row exists
    and surface_valid == true
    and documented A1/A2 contract passes
    and ATM call body row exists
    and ATM put body row exists
    and required prices are valid
    and PIT eligibility is satisfied
```

A valid processed failure is complete for infrastructure coverage but excluded
from this observation set.

### Gate X — lineage and scope

- all copied/produced artifacts and reports resolve inside one staging root;
- path resolution rejects absolute artifact/report paths and any `..` escape;
- one `build_id`, scope, production flag, repository SHA, evidence IDs,
  source roots (including read-only raw root), commands, exits, source and
  copied hashes, equality results, and `upstream_bundle_id` are recorded;
- spot reads copied/fixture chains and surface reads those chains plus rebuilt
  spot;
- all denominators use the same effective universe and resolved schedule;
- artifact-specific ranges are recorded and the dependency-aware
  `feature_ready_range` is non-empty;
- every expected key inside the resolved schedule has an A1 row;
- every `surface_valid=true` A1 row has the required canonical A2 body pair;
- valid processed failure rows remain complete but are excluded from the
  feature-eligible observation set;
- each calendar date in `feature_ready_range` has strict-prior PIT support,
  required adjusted and spot entry/expiry date infrastructure, and a complete
  A1 processing schedule;
- full scope alone may set `production_accepted=true`.

---

## 11. Manifest and identity

### Schema and paths

Retain C1 schema version `"1"`. `input_snapshot.py::_parse_artifacts` accepts
arbitrary string keys; add:

```text
liquidity_daily
liquidity_weekly
liquidity_panel
liquid_tickers
splits
adjusted_chains_root
spot_prices
option_surface_meta
option_surface_quotes
```

Report keys include copied-bundle validation, C5 audit, C7 PIT audit, spot
gate, C6 surface audit, and snapshot completeness.

All artifact/report values are forward-slashed paths relative to the snapshot
root. A resolver must reject absolute paths and normalized paths outside the
root.

The PASS manifest is physically written under `.building`, but:

```text
manifest.cache_dir = C:/MomentumCVG_env/snapshots/<build_id>
```

That is the **planned final root**, never `.building`. Therefore every
published relative path resolves after rename.

### `snapshot_id` semantics and inputs

```text
snapshot_id = logical data definition
build_id    = specific execution
repo SHA    = implementation lineage
overall_status / production_accepted = validation and publication outcome
```

Repository SHA remains non-identity lineage. Materially different logical
builds must have different `snapshot_id`s.

Identity-relevant `params` include:

- `scope`;
- requested range (if any), artifact-specific resolved ranges, and
  `feature_ready_start`/`feature_ready_end`;
- `effective_ticker_universe_sha256`;
- `upstream_bundle_id`, verified `source_liquid_tickers_sha256`, and verified
  `source_splits_sha256`;
- frequency, `dte_target`, `min_abs_delta`, `max_abs_delta`,
  `delta_buckets`, and `keep_zero_bid_quotes`;
- liquidity/PIT eligibility parameters copied from C4
  (`lookback_weeks`, `min_valid_quote_weeks`, `dte_min`, `dte_max`,
  `dvol_top_pct`, `spread_bot_pct`, and PIT strict-prior policy identifier);
- explicit surface schedule/expiry/contract policy version identifiers;
- the snapshot-root-relative artifact paths.

`as_of_resolved_trading_day` is the resolved end date. Relative paths alone
never establish logical identity.

Explicitly excluded from identity: `production_accepted`, `overall_status`,
blocking failures, warning results, and publication result. A failed or
unpublished execution may compute the same logical `snapshot_id` as a
successful execution with the same data definition; `build_id` and outcome
fields distinguish those executions. A bounded build still differs from a
full build because scope, effective-universe hash, ranges, and artifact
definitions differ.

### Atomic write and publication failure

- Write manifest to a same-directory temporary file and `os.replace`.
- Before rename, read it back and resolve all paths against the planned final
  root.
- A PASS/WARN manifest may exist only transiently in `.building` immediately
  before the attempted publication.
- If publication rename fails, do **not** leave that manifest claiming PASS:
  rewrite status to FAIL, add a publication failure, set `cache_dir` to the
  actual retained failed root, and rename/retain the root as `.failed`.
- If even the `.failed` rename fails, rewrite the manifest in `.building`
  with FAIL and the actual `.building` `cache_dir`; exit nonzero.
- A FAIL manifest always uses the actual retained root as `cache_dir`.

---

## 12. Failure and warning behavior

### Warning allowlist

Status aggregation is:

```text
PASS                         -> publish
allowlisted informational WARN -> publish with overall_status=WARN
non-allowlisted WARN         -> block pending review
FAIL                         -> block
```

Initial allowlist:

1. `bounded_scope_non_production` — bounded evidence only; necessarily
   `production_accepted=false`;
2. `runtime_advisory` — runtime or storage advisory with no correctness,
   coverage, or lineage impact;
3. `redundant_provenance_note` — an optional legacy provenance note only when
   stronger `upstream_bundle_id` evidence has already passed.

Warnings involving coverage, lineage identity, schema, missing artifacts,
unexpected dates/tickers, path escape, audit correctness, spot consistency,
surface joins/body rows, or publication are never allowlisted and block.

Sprint 005 may consume PASS or explicitly allowlisted WARN only.

### Failure/retry

- Producer/copy failure: no final root; FAIL manifest under `.failed`.
- Blocking gate or non-allowlisted WARN: no final root; exit nonzero.
- Ctrl+C/crash: `.building` remains unpublished; next run uses a new build ID.
- Retry never resumes or reuses a partial root.
- Existing final roots are never opened for write.

---

## 13. CLI contract

New entrypoint:

```powershell
python scripts/build_input_snapshot.py `
  --snapshots-root C:/MomentumCVG_env/snapshots `
  --c4-source C:/MomentumCVG_env/input/liquidity `
  --c5-source C:/MomentumCVG_env/input/adjusted_liquid `
  --raw-root C:/ORATS/data/ORATS_Data `
  --scope full `
  --frequency weekly `
  [--workers N] `
  [--dry-run]
```

Bounded evidence additionally requires an explicit ticker list and fixture
date range. Those arguments are rejected with `--scope full`.

`--raw-root` is required and read-only. It is used only by the copied-C5
inventory comparison and sampled raw-vs-adjusted audit. Raw ORATS files are
not copied into the snapshot and are never modified. The path is recorded as
implementation lineage but is not a snapshot artifact and is not exposed to
Sprint 005. The orchestrator derives `--years` from copied adjusted inventory
and passes all required arguments to `audit_adjusted_liquid.py`:
`--raw-root`, `--adj-root`, `--splits`, `--ticker-universe`, `--years`, and
`--report-path`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Published with PASS or explicitly allowlisted WARN |
| 1 | Copy/producer/subprocess failure |
| 2 | Usage, collision, path, or root-lifecycle error |
| 3 | Unsupported incremental/repair request |
| 4 | Blocking gate, FAIL, or non-allowlisted WARN |
| 5 | Publication rename failure |

`--dry-run` performs validation and planning that requires no artifact reads
beyond argument/path syntax, prints exact roots, copy sources, commands, scope,
and gates, then exits before any filesystem write.

---

## 14. Sprint 005 consumption contract

Sprint 005:

1. loads the explicit manifest;
2. rejects non-production scope for the production feature backfill;
3. accepts only PASS or explicitly allowlisted WARN;
4. resolves and validates all required relative paths inside
   `manifest.cache_dir`;
5. uses calendar-level `feature_ready_range` only to bound dates with complete
   infrastructure and expected A1 processing;
6. excludes valid processed failure rows from observation-level feature
   eligibility without treating them as missing infrastructure;
7. for `surface_valid=true` observations, joins A1 with A2 ATM call and put rows on
   `(ticker, entry_date, expiry_date, body_strike)`;
8. requires valid prices and PIT eligibility before feature calculation;
9. records the input lineage in its feature manifest.

C8 does not patch legacy feature scripts or `run_surface_search.py`. Sprint
005 owns the canonical feature-backfill entrypoint and formula validation.

---

## 15. Evidence plan

### C8.4 bounded evidence

- create the bounded fixture specified in §7;
- prove one effective universe controls every denominator;
- run hardened spot, surface, gates, warning aggregation, manifest creation,
  rename, and post-rename resolution;
- mark `scope=bounded`, `production_accepted=false`;
- never present this as Sprint 005 production acceptance.

### C8.5 full production evidence

Copy and revalidate the complete accepted C4/C5 bundle, rebuild full spot and
surface, and record:

- repository SHA, build ID, snapshot ID, upstream bundle ID, root, and
  manifest path;
- accepted evidence IDs, source and copied component hashes/digests, and
  exact equality verdicts;
- read-only raw-root lineage and audited inventory-derived years;
- resolved range and effective universe hash/count;
- calendar-level `feature_ready_range` and feature-eligible observation count;
- copied/produced file and row counts;
- stage commands, exit codes, runtimes, and storage;
- every gate and warning decision;
- post-publication manifest-resolution smoke.

No feature generation or backtest is part of C8 closeout.

---

## 16. Revised implementation commits

```text
C8.1A  Correct and finalize immutable snapshot design

C8.2   Harden full spot extraction
       - fail-closed behavior
       - intra-ticker spot consistency
       - coverage/uniqueness/value checks
       - atomic write
       - focused tests

C8.3A  Snapshot lifecycle, identity, manifest, and validation primitives
       - collision-resistant build_id
       - staging/final/failed root lifecycle
       - source-to-copy identity verification and upstream bundle identity
       - safe artifact resolver
       - corrected snapshot_id semantics and post-rename manifest correctness
       - warning allowlist
       - gates L/S/A/SP/SF/X as callable/testable components
       - calendar readiness vs observation-level eligibility primitives

C8.3B  Snapshot orchestrator wiring
       - copy accepted C4/C5 bundle
       - read-only raw-root C5 audit wiring with inventory-derived years
       - rebuild spot
       - rebuild surface
       - ordered gates and dry-run
       - truthful failure behavior
       - publication rename

C8.4   Bounded evidence
       - bounded effective universe
       - lifecycle and manifest-resolution smoke
       - explicitly non-production

C8.5   Full production snapshot evidence and C8 closeout
```

### Required isolated tests

- build-ID collision resistance;
- dry-run performs no writes;
- existing final root is never modified;
- producer failure never publishes;
- gate failure creates no final root;
- non-allowlisted WARN blocks;
- allowlisted WARN publishes with WARN status;
- PASS paths resolve after `.building → final`;
- rename failure cannot leave misleading PASS;
- FAIL manifest resolves inside `.failed`;
- relative paths cannot escape the root;
- copied C4/C5 bundle identity is recorded;
- changed copied `liquid_tickers.csv` blocks;
- changed copied split file blocks;
- missing/additional copied adjusted file blocks;
- changed copied adjusted file size blocks;
- identical source/copy bundle passes;
- effective universe controls every denominator;
- dependency range derivation rejects a surface entry without strict-prior PIT
  or required entry/expiry chain and spot coverage;
- bounded build cannot be production accepted;
- changing `production_accepted` or status does not change `snapshot_id`;
- changing scope or effective-universe hash changes `snapshot_id`;
- spot rejects inconsistent repeated source spot values;
- surface gate catches a missing A1 key;
- valid A1 failure row counts as processed;
- valid A1 failure row requires no A2 body rows and is not feature eligible;
- `surface_valid=true` A1 requires ATM call and put A2 rows.

All tests use temporary roots and synthetic fixtures; none reads production.

---

## 17. Explicitly deferred work

| Work | Owner |
|---|---|
| Momentum/CVG formulas, feature backfill, feature correctness/audits | Sprint 005 |
| Canonical feature entrypoint and legacy feature-script cleanup | Sprint 005 |
| Weekly append, incremental refresh, split delta, selected-ticker repair | Later operational sprint |
| Spot/surface merge, watermark and changed-only audits | Later operational sprint |
| `current`/`latest` pointer, scheduling, notifications, dashboards, backup | Later operational sprint if required |
| `run_surface_search.py` cleanup and backtest wiring | Later backtest sprint |
| Strategy evaluation and alpha research | Later research sprint |
| Shadow/broker operation | Later trading sprint |

---

## 18. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Full surface build has only bounded prior evidence | C8.4 first; tune existing full-overwrite producer, do not introduce merge. |
| Copying a mutable source during concurrent modification | Accepted C4/C5 roots are immutable by policy; compute component identity after copy and rerun C5/C7 audits. |
| File-size inventory digest misses in-place same-size corruption | Limitation is explicit; accepted-source policy plus C5 audit provides correctness evidence without hashing every byte. |
| Exact spot denominator exposes benign source anomalies | Fail closed; change the denominator only through reviewed C8.2 evidence, never silently. |
| Bounded fixture confused with production | Identity includes scope/universe hash; manifest sets `production_accepted=false`; Sprint 005 rejects it. |
| Rename failure after writing planned-final paths | Rewrite truthful FAIL manifest with actual retained root before returning nonzero. |
| Surface scale pressures C8 toward chunk/merge | Treat as runtime tuning; escalate rather than adding incremental infrastructure. |

**Remaining unresolved blocker:** none at design level. Full-scale surface
runtime is an evidence risk, not a reason to expand C8 before C8.4/C8.5.

---

## 19. Acceptance checklist

- [ ] Accepted matching C4/C5 bundle is copied, not rebuilt.
- [ ] Source and copied ticker/split hashes and adjusted inventory digests
      match exactly before `upstream_bundle_id` is accepted.
- [ ] `upstream_bundle_id` and verified component evidence/digests are
      recorded and identity-relevant.
- [ ] Required read-only `--raw-root` is recorded as lineage, and C5 audit
      receives copied paths plus inventory-derived years.
- [ ] Full range is resolved from copied adjusted inventory before spot/surface.
- [ ] Bounded scope uses one effective universe and cannot be production accepted.
- [ ] Hardened spot contract and focused tests pass.
- [ ] Every expected key has one A1 row; only `surface_valid=true` rows require
      ATM A2 call/put body pairs.
- [ ] Calendar-level `feature_ready_range` and observation-level feature
      eligibility are separately derived and tested.
- [ ] C5, C6, and C7 audits pass against snapshot paths.
- [ ] `production_accepted`, status, warnings, failures, and publication result
      do not affect `snapshot_id`.
- [ ] Warning allowlist is enforced.
- [ ] Manifest paths resolve after publication and cannot escape the root.
- [ ] Rename failure leaves no misleading PASS.
- [ ] Bounded evidence passes as explicitly non-production.
- [ ] One full snapshot publishes and passes manifest-resolution smoke.
- [ ] Sprint 005 receives the explicit production manifest path.

---

## 20. Inspected files

**Design/evidence:** `AGENTS.md`, `docs/agenda/current_sprint.md`,
`docs/tmp/c8_0_refresh_pipeline_reality_map.md`,
`docs/tmp/c8_0a_backfill_snapshot_reality_map.md`,
`docs/sprint_memos/004_c4_liquidity_panel.md`,
`004_c5_adjusted_liquid.md`, `004_c6_option_surface.md`,
`004_c7_pit_universe.md`.

**Implementation:** `src/data/input_snapshot.py`,
`src/data/split_adjuster.py`,
`src/features/option_surface_analyzer.py`,
`scripts/build_liquidity_panel.py`,
`scripts/apply_split_adjustment.py`,
`scripts/extract_spot_prices.py`,
`scripts/precompute_option_surface.py`,
`scripts/audit_adjusted_liquid.py`,
`scripts/audit_option_surface_artifacts.py`,
`scripts/audit_pit_universe.py`.

**Tests checked through C8.0/C8.0A and this correction:**
`tests/unit/test_input_snapshot.py`,
`test_build_liquidity_panel.py`,
`test_apply_split_adjustment_cli.py`,
`test_split_adjuster.py`,
`test_split_adjuster_filtered_zip.py`,
`test_precompute_option_surface_cli.py`,
`test_option_surface_contract.py`,
`test_audit_adjusted_liquid.py`,
`test_audit_option_surface_artifacts.py`,
`test_pit_universe_audit.py`; there remains no
`test_extract_spot_prices.py`, which C8.2 adds.
