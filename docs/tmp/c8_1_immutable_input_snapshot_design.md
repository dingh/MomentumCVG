# Sprint 004 — C8.1 immutable input snapshot design

**Task:** C8.1 — immutable input snapshot design
**Mode:** Design only (no code/test/artifact changes; no external data access)
**Starting HEAD:** `a100220872ee593b91012b3d0194dd5127b1970f`
**Grounded in:** C8.0 (`docs/tmp/c8_0_refresh_pipeline_reality_map.md`) and C8.0A (`docs/tmp/c8_0a_backfill_snapshot_reality_map.md`), both accepted.

> This is a concrete, decisive design for the **smallest safe** system that produces one complete, immutable, auditable historical input snapshot for Sprint 005 to consume. It deliberately excludes weekly/incremental/repair work. Every recommendation states the tradeoff, the choice, and why it is the smallest safe choice.

---

## 1. Decision summary

| Decision | Recommendation | Why smallest-safe |
|----------|----------------|-------------------|
| Isolation boundary | The snapshot **root directory** is the only isolation unit; producers are not made individually transactional | Root-level staging + publish-by-rename subsumes per-producer atomicity |
| Root lifecycle | Build into `…/snapshots/<build_id>.building/`, publish by **rename** to `…/snapshots/<build_id>/` only after all blocking gates PASS | Directory rename is atomic on local NTFS; one boundary for the whole snapshot |
| Build scope | **Rebuild** liquidity, adjusted chains, spot, surface from the immutable raw ORATS source; **copy** an explicit accepted split-history file (with recorded source + hash + validation) | Keeps the ORATS **network/token off the critical path**; everything else is deterministic from raw |
| Split fetch | **Not** in the C8 critical path; consumed as an explicit `--splits-source` file | Avoids network/token dependency for the Sprint 005 handoff |
| Spot producer | **Harden** in C8.2 (fail-closed, completeness, uniqueness, value checks, atomic write, tests) | It is the only producer currently unsafe for batch acceptance |
| Surface A1/A2 atomicity | **Not required at producer level** | Staging root + "both exist + join + completeness" gate + publish-after-PASS makes a mid-write failure only an unpublished failed root |
| Adjusted-chain producer | Use as-is; rely on empty-root construction + raw↔adjusted inventory gate + C5 audit | Empty root eliminates stale-file/skip-existing risk; no ticker repair needed |
| Liquidity producer | Use as-is (backfill into snapshot); reuse C7 PIT audit | C4 already accepted, tested, and writes to an arbitrary `--cache-dir` |
| Manifest | **Reuse C1 schema version 1, no migration**; add artifact keys; `cache_dir` = snapshot root; snapshot-root-relative paths | `_parse_artifacts` already accepts arbitrary keys; identity excludes `cache_dir` |
| Orchestrator | **New entrypoint `scripts/build_input_snapshot.py`**; leave `refresh_weekly_inputs.py` untouched | Accurate name, no misleading weekly/`--mode` baggage |
| Sprint 005 handoff | Primary interface = **manifest path** (`--input-snapshot-manifest`) | Manifest resolves all artifacts + carries identity + status |
| Same-generation lineage | One `build_id`, one root, recorded source paths + producer commands + repo SHA, sha256 of the two small upstream files (`liquid_tickers.csv`, split history) | Cheap lineage; no large-parquet content hashing |
| Implementation | C8.1 (design) → C8.2 (spot) → C8.3 (orchestrator+gates+manifest+publish) → C8.4 (evidence+closeout) | Four commits, no refresh-platform creep |

---

## 2. C8 goal and non-goals

**Goal:** Produce one complete, immutable, auditable historical input snapshot inside an isolated versioned root that Sprint 005 can consume as the *sole* upstream input for a trusted momentum/CVG feature backfill.

**In scope:** fresh-empty-root full-backfill construction; post-build completeness + correctness gates reusing C5/C6/C7 audits; spot hardening; an accurate manifest; publish-by-rename; a manifest-based Sprint 005 handoff.

**Explicit non-goals (do not design or build in C8):** weekly append, incremental refresh, watermark operation, split-history delta detection, selected-ticker repair, spot/surface row-level merge, changed-only audits, `current/latest` pointer, `run_surface_search.py` repair, legacy feature-script cleanup, momentum/CVG formulas, feature backfill/audits, scheduling/notifications/dashboards/distributed/backup/broker/shadow. These belong to later milestones (§16).

**Decision rule applied throughout:** *a capability belongs in C8 only when Sprint 005 cannot safely generate and validate momentum/CVG features without it.*

---

## 3. Sprint 004 → Sprint 005 handoff contract

**Primary interface (recommendation): a manifest path.**

```
Sprint 005 entrypoint  --input-snapshot-manifest  C:/MomentumCVG_env/snapshots/<build_id>/manifests/input_snapshot_<snapshot_id>.json
```

**Tradeoff:** a bare snapshot-root path is simpler but forces Sprint 005 to re-derive filenames and cannot carry identity/status; a manifest path resolves every artifact and carries `snapshot_id`, `overall_status`, and lineage. **Choose the manifest** — it is the single source of truth and reuses the existing `read_manifest` reader. **Deferred alternative:** also accept `--input-snapshot-root` as a convenience that locates the single manifest under `<root>/manifests/`; only add if Sprint 005 asks for it.

The manifest (via `cache_dir` = snapshot root + relative artifact paths) exposes at least:

| Field | Source | Sprint 005 consumes directly? |
|-------|--------|-------------------------------|
| `snapshot_id`, `build_id` | manifest identity + execution id | Yes (record in feature manifest) |
| repository commit SHA | manifest `notes` + lineage | Yes (record) |
| requested historical range | manifest `params.requested_start/end` | Yes |
| resolved historical range | manifest `params.resolved_start/end` | Yes (feature date bounds) |
| liquidity panel | `artifacts.liquidity_panel` | Yes (S1 PIT eligibility) |
| liquid_tickers superset | `artifacts.liquid_tickers` | Yes (universe superset) |
| scoped split history | `artifacts.splits` | Rarely (provenance only) |
| adjusted-chain root | `artifacts.adjusted_chains_root` | Optional (low-level fallback) |
| spot-price artifact | `artifacts.spot_prices` | Optional (low-level fallback) |
| option-surface A1 | `artifacts.option_surface_meta` | **Yes (primary momentum/CVG source)** |
| option-surface A2 | `artifacts.option_surface_quotes` | **Yes (supporting quotes/greeks)** |
| audit reports | `reports.*` | Optional (evidence) |
| overall PASS/WARN/FAIL | `overall_status` | Yes (must be PASS/WARN to consume) |

Sprint 005 must **not** rediscover mutable global paths and must **not** copy snapshot files over `C:/MomentumCVG_env/input/liquidity`, `…/input/adjusted_liquid`, or `…/cache`. It resolves everything relative to the snapshot root recorded in the manifest.

---

## 4. Definition of a successful snapshot (C8 definition of done)

> **C8 is complete when a single command (`scripts/build_input_snapshot.py`) can build a complete historical input snapshot inside a new empty `.building` staging root, run the required completeness and correctness gates, write an accurate manifest, and publish the root — by renaming it to its final immutable name — only if every blocking gate passes; and the published snapshot is directly consumable by a Sprint 005 momentum/CVG feature entrypoint through the manifest handoff.**

Precise meanings:

- **Complete** — every blocking completeness gate in §9 passes: all four liquidity artifacts present and PIT-valid; a validated split file; one adjusted parquet for every resolved trading date; a spot artifact covering exactly the adjusted date/ticker denominator; A1 covering the full expected `(ticker, entry_date)` schedule (valid failure rows included) with A2 joining correctly; and cross-layer same-generation lineage recorded.
- **Accepted** — the final root `<build_id>/` exists (rename succeeded), which by construction happens only after all blocking gates returned PASS (WARN allowed, see §12).
- **Immutable** — no C8 code path writes into an existing final `<build_id>/` root; it is treated as read-only by convention (optional read-only file attributes). Re-running never mutates an accepted snapshot — it creates a new `build_id`.
- **Usable by Sprint 005** — the manifest resolves all artifacts against the snapshot root, and a manifest-resolution smoke (read manifest → resolve each artifact path → assert existence + expected row/file counts) passes without touching any global path.

---

## 5. Snapshot-root layout and lifecycle

**Chosen layout:**

```
C:/MomentumCVG_env/snapshots/
    <build_id>.building/            # unpublished; being built
        input/
            liquidity/              # ticker_liquidity_{daily,weekly}_observations.parquet,
                                    #   ticker_liquidity_panel.parquet, liquid_tickers.csv
            adjusted_liquid/        # splits_hist_liquid.parquet + {YYYY}/ORATS_SMV_Strikes_*.parquet
        cache/                      # spot_prices_adjusted.parquet,
                                    #   option_surface_{meta,quotes}_weekly_<S>_<E>.parquet
        reports/                    # C5/C6/C7 + completeness reports
        manifests/                  # input_snapshot_<snapshot_id>.json
    <build_id>/                     # published immutable snapshot (same tree, renamed)
    <build_id>.failed/              # (optional) retained failed build for debugging
```

Rationale: this mirrors the accepted production sub-roots (`input/liquidity`, `input/adjusted_liquid`, `cache`) so producer path arguments map 1:1 with no new conventions, and it keeps reports + manifest inside the same root for same-generation lineage.

**Publication boundary — directory rename.** On local Windows/NTFS a same-volume directory rename (`os.replace(staging, final)`) is atomic and cheap. **Confirmed appropriate**: the snapshots parent (`C:/MomentumCVG_env/snapshots`) is one volume, so `.building → <build_id>` is a single atomic metadata operation. This is the whole-snapshot commit point. *Constraint to enforce in C8.3:* staging and final roots must be on the **same volume** (assert before build).

**Lifecycle policy (decided):**

- **Staging-root naming:** `<build_id>.building`. `build_id` from `generate_build_id()` (`{UTC ts}_{6-hex}`), unique per execution.
- **Final-root naming:** `<build_id>` (drop the `.building` suffix).
- **Staging root already exists:** refuse, exit 2 (should never happen given unique `build_id`; a collision signals a clock/logic fault). Never reuse a non-empty staging root.
- **Final root already exists:** refuse, exit 2 (never overwrite an accepted snapshot).
- **Failed roots:** **retained**, renamed `<build_id>.failed` (not auto-deleted) so evidence is inspectable; operator deletes manually. Interrupted builds leave `<build_id>.building`.
- **Rerun with identical logical inputs:** creates a **new `build_id`** (new root) but computes the **same `snapshot_id`** (identity is range + params + relative artifact paths; §10). This satisfies "same logical inputs → same snapshot_id, new build root, never mutate accepted."
- **No `current`/`latest` pointer.** Sprint 005 receives an explicit manifest path. (Deferred; add only if a real need appears.)

---

## 6. Build-scope decision

**Recommendation: rebuild all derived layers from the immutable raw ORATS source; copy the split-history file as an explicit input.**

This is a deliberate blend of Option A (reconstruct) and Option B (consume accepted upstream), chosen per layer:

| Layer | Decision | Reason |
|-------|----------|--------|
| Liquidity (daily/weekly/panel/liquid_tickers) | **Rebuild** from raw into snapshot | Deterministic, **no token**, self-contained, `liquid_tickers.csv` generated *within* the snapshot guarantees true same-generation scoping of everything downstream; C4 accepted + tested |
| Split history | **Copy** an explicit `--splits-source` file into `input/adjusted_liquid/splits_hist_liquid.parquet` | `fetch_splits` needs **network + ORATS token**; keeping it off the critical path is an explicit goal; the file is small and stable |
| Adjusted chains | **Rebuild** from raw + copied split file (filtered mode, explicit `--adj-root`, `--overwrite` into empty root) | Deterministic; empty root removes skip-existing/stale-factor risk; C5 accepted + tested |
| Spot | **Rebuild** with the hardened extractor (C8.2) from snapshot adjusted chains | No accepted evidence exists; must be produced and gated |
| Surface A1/A2 | **Rebuild** with `precompute_option_surface` from snapshot adjusted chains + snapshot spot | No full-history evidence exists; must be produced and gated |

**Why not full Option A:** running `fetch_splits` in the critical path adds a network/token dependency and a non-deterministic external call for no Sprint-005 benefit — the split file is a stable small input.

**Why not lighter Option B (copy liquidity too):** copying accepted C4 liquidity is *permitted* (with lineage) and faster, but rebuilding from raw makes the snapshot fully self-contained and removes any reference to the mutable global `input/liquidity`. Since raw ORATS is local (no token) and C4 is deterministic + tested, rebuild is the smaller *risk* even if larger *runtime*.

**Copied-artifact rules (apply to the split file, and to liquidity if the fallback in §17 is taken):**
- **Which may be copied:** only the scoped split history (`splits_hist_liquid.parquet`). Nothing else by default.
- **How source is recorded:** absolute source path + `sha256` + row/ticker counts recorded in the manifest (`params.splits_source_path`, `params.splits_sha256`) and the completeness report.
- **Validation after copying:** schema check; no conflicting `(ticker, split_date)` divisors; every split ticker ∈ the snapshot's `liquid_tickers` superset. Fail closed on violation.
- **Does copying preserve immutability?** Yes — the copy lands inside the immutable snapshot root and is fingerprinted, so the snapshot is self-describing and reproducible; the external source is only provenance.

---

## 7. Producer-by-producer policy

| Producer | C8 policy | Notes |
|----------|-----------|-------|
| `build_liquidity_panel.py` | **Use as-is** | `--mode backfill --data-root <raw> --cache-dir <staging>/input/liquidity`; orchestrator validates 4 artifacts + runs C7 PIT audit. Its staged per-file `os.replace` publication is sufficient inside the snapshot root; **do not redesign C4.** |
| `fetch_splits.py` | **Defer** (not in critical path) | Split history supplied via `--splits-source` and copied+validated. Network fetch remains available for operators outside C8. |
| `apply_split_adjustment.py` | **Use as-is** | Filtered mode: `--raw-root <raw> --adj-root <staging>/input/adjusted_liquid --splits <staging>/…/splits_hist_liquid.parquet --ticker-universe <staging>/…/liquid_tickers.csv --overwrite`. Empty root ⇒ direct per-file writes are acceptable; gated by inventory + C5 audit. **No ticker repair.** |
| `extract_spot_prices.py` | **Small hardening (C8.2)** | The one producer unsafe for batch acceptance; contract below. |
| `precompute_option_surface.py` | **Use as-is + orchestrator-side gates** | Producer-level A1/A2 pair-transaction **not required** (staging-root model, below). Write A1/A2 into `<staging>/cache`; gate both-exist + join + completeness before publish. |

### 7.1 Spot producer hardening (C8.2)

Bounded strictly to full-history snapshot generation — **no append/merge.**

- **Exit-code contract:** exit `0` only if the output covers the full expected denominator and passes all value/uniqueness checks; exit **nonzero** on any failed date, empty output, or validation failure. (Never `return`/exit 0 on partial output — the current behavior.)
- **Per-date failure tracking:** accumulate every date that raised or produced zero rows; if the failed set is non-empty, fail closed and write **no** success artifact.
- **Empty-output behavior:** refuse to write a success artifact when the result is empty or short of the denominator; exit nonzero.
- **`(date, ticker)` uniqueness:** assert unique; fail on any duplicate.
- **Null/nonfinite rejection:** reject rows with null or ≤0 `adj_spot_price` or `spot_price`; treat as a data fault (fail closed), not a silent drop.
- **Expected-date coverage:** expected dates = the set of trading dates present in the snapshot adjusted-chain inventory over the resolved range; require **exact set equality**.
- **Expected-ticker coverage (denominator, decided):** for each adjusted date, the spot artifact must contain **exactly** the ticker set present in that adjusted daily parquet (equality after value validation). *Tradeoff:* exact per-date equality is stricter than "≥ some floor," but it is the natural output of `groupby(ticker).first()` over the day file and gives Sprint 005 a guaranteed spot for every chain ticker; a null/absent spot then correctly fails the build rather than silently shrinking coverage. **Choose exact equality.**
- **Temporary output + atomic replace:** write to `spot_prices_adjusted.parquet.tmp` in the same directory, then `os.replace` to the final name. (Inside staging this is belt-and-suspenders; the root rename is the real commit.)
- **Tests (required for C8.2):** unit tests for each failure mode — missing date, duplicate `(date,ticker)`, null/nonpositive value, empty output, and a small happy path proving exit 0 + coverage. This closes the "no test at all" gap from C8.0A.

### 7.2 Surface producer — is producer-level A1/A2 atomicity required?

**No.** Under the staging-root model: A1 and A2 are written into `<staging>/cache`; a **blocking gate** requires both to exist, share a valid grain, and satisfy the A1↔A2 join + completeness (§9); publication renames the root only after PASS. A failure between the A1 and A2 writes therefore leaves only an **unpublished** `.building` (later `.failed`) root that is never referenced by any manifest handoff. Hence producer-level pair transactions are **unnecessary for C8**, and no surface rewrite is warranted. *Deferred alternative:* add producer-level pairing only if a future non-staged use appears (it will not in C8).

### 7.3 Adjusted-chain producer — direct writes and `split_factor` recomputation

- Direct per-file writes into an **empty** staging `adjusted_liquid` are acceptable; there is no pre-existing file to corrupt or stale-skip.
- **`split_factor` recomputation gate: not required for C8.** Building into an empty root from the single copied split file computes each factor fresh during this build, so the stale-factor risk that motivated recomputation (C8.0 §8) cannot occur here. **Choose: empty-root construction + raw↔adjusted inventory gate + C5 sampled-math audit is sufficient.** *Deferred alternative:* add factor recomputation-vs-current-split audit if a future non-empty-root path is introduced.

---

## 8. Batch orchestration sequence

`scripts/build_input_snapshot.py` executes, in order, into `<build_id>.building/`:

1. **Preflight:** assert staging + final parents on same volume; assert `--splits-source` exists; create empty `.building` tree; resolve `build_id`.
2. **Liquidity:** run `build_liquidity_panel.py --mode backfill` → `input/liquidity/*`.
3. **Gate L** (liquidity completeness + C7 PIT audit). Abort on FAIL.
4. **Split copy + validate:** copy `--splits-source` → `input/adjusted_liquid/splits_hist_liquid.parquet`; record sha256; run **Gate S**. Abort on FAIL.
5. **Adjusted chains:** run `apply_split_adjustment.py` (filtered, `--overwrite`) → `input/adjusted_liquid/{YYYY}/…`.
6. **Gate A** (raw↔adjusted inventory + C5 audit). Abort on FAIL.
7. **Resolve historical range** from the adjusted-chain inventory (requested vs resolved start/end).
8. **Spot:** run hardened `extract_spot_prices.py` → `cache/spot_prices_adjusted.parquet`.
9. **Gate SP** (spot completeness/uniqueness/values, §9). Abort on FAIL.
10. **Surface:** run `precompute_option_surface.py --frequency weekly` → `cache/option_surface_{meta,quotes}_weekly_<S>_<E>.parquet`.
11. **Gate SF** (surface completeness + C6 audit + A1/A2 join). Abort on FAIL.
12. **Gate X** (cross-layer same-generation lineage, §9).
13. **Write completeness report** + **manifest** into `reports/` and `manifests/` with aggregated `overall_status`.
14. **Publish:** if all blocking gates PASS → `os.replace(<build_id>.building, <build_id>)`. Else → rename to `<build_id>.failed`, exit nonzero, no publication.

"Abort on FAIL" means: still write the completeness report + a `overall_status=FAIL` manifest into the staging root for debugging, then stop before publication.

---

## 9. Completeness and audit gates

Each gate separates five concerns: **producer exit success · artifact existence · internal consistency · completeness · cross-layer compatibility.** All listed gates are **blocking** unless marked WARN.

### Gate L — Liquidity
- Producer exit 0; all four artifacts exist (`daily`, `weekly`, `panel`, `liquid_tickers.csv`).
- Schema valid; panel `month_date` range within requested range; weekly/panel watermarks consistent.
- `liquid_tickers.csv` non-empty.
- **C7 `audit_pit_universe` PASS** (strict prior-snapshot semantics; full-history superset coverage).

### Gate S — Split history
- Split artifact exists; schema valid (`ticker, split_date, divisor`).
- No conflicting `(ticker, split_date)` divisors.
- Every split ticker ∈ `liquid_tickers` superset.
- Source path + sha256 + row/ticker counts recorded.
- *No requirement that every liquid ticker appear* (most legitimately have no split). Because C8 **copies** an accepted file rather than fetching, "successful completion of no-split tickers" is established by provenance (the accepted file was produced by an accepted C5 fetch) — not re-derived here.

### Gate A — Adjusted chains
- Producer exit 0.
- **Every** resolved trading date in range has exactly one adjusted parquet; **no** adjusted date outside the requested range.
- Required adjusted columns present.
- **C5 `audit_adjusted_liquid` PASS** (inventory full; sampled `adj == raw / stored_factor` math). `split_factor` recomputation not required (§7.3).

### Gate SP — Spot
- Producer exit 0 (hardened, so exit 0 already implies coverage — but re-assert independently).
- **All expected dates present** = adjusted-chain trading dates over resolved range.
- **Per-date ticker equality** with the adjusted daily file's ticker set (§7.1).
- Unique `(date, ticker)`; finite positive `adj_spot_price` and `spot_price`.
- No silently failed dates (failed-date set empty).

### Gate SF — Surface
- Both A1 and A2 exist.
- **Expected A1 denominator = `liquid_tickers` superset × resolved weekly entry-date schedule** (`weekly_trade_dates_in_range` per `precompute_option_surface`/`trading_day`). Confirmed against producer semantics: the producer emits one A1 row per processed `(ticker, entry_date)`, including a valid `failure_reason` row when no trade is possible.
- **One A1 row for every expected processed `(ticker, entry_date)`**, counting valid failure rows. Distinguish: *a processed ticker-date with a valid `failure_reason`* (**complete**) vs *a ticker-date the producer never emitted* (**incomplete → FAIL**). Expected no-trade failure rows are **not** treated as incompleteness.
- Unique A1 grain `(ticker, entry_date)`; valid A2 grain `(ticker, entry_date, expiry_date, strike, side)`.
- Every A2 row joins to an A1 row; every `surface_valid` A1 row satisfies its required A2 conditions.
- Weekly entry/expiry policy passes (strict calendar-paired weekly expiry).
- **C6 `audit_option_surface_artifacts` PASS.**

### Gate X — Cross-layer same-generation
- All artifacts live under the one `<build_id>.building` root; all reports were generated from paths inside it.
- Recorded: one `build_id`, repository SHA, producer commands + exit codes, source paths.
- `sha256` recorded for the two small upstream files: `liquid_tickers.csv` and the split history. (Large parquets are **not** content-hashed — cost without Sprint-005 benefit.)
- Adjusted/spot/surface built strictly from paths inside this root (verified by construction: producer `--data-root`/`--spot-db-path` point into the staging root).

**Minimal lineage justification:** because a single orchestrator execution creates every artifact into one fresh root and publishes only after all gates pass, same-generation is guaranteed structurally; the two small hashes + recorded commands + SHA are sufficient to *prove* it after the fact without hashing multi-GB parquet sets.

**WARN vs FAIL:** any audit-reported WARN (e.g., a bounded-sample provenance warning) is **non-blocking** — it is recorded in `overall_status`/`notes` and the snapshot still publishes. Only FAIL (or a missing/short artifact) blocks publication.

---

## 10. Manifest and lineage design

**Recommendation: reuse `input_snapshot.py` schema version `"1"` with no migration.** `_parse_artifacts` already accepts arbitrary string keys, `cache_dir` is excluded from `snapshot_id`, and artifact path *values* feed identity — exactly what a snapshot-root-relative manifest needs.

- **`cache_dir`** = the snapshot root absolute path (e.g., `…/snapshots/<build_id>`). Excluded from `snapshot_id`, so identity is independent of where the snapshot lives.
- **Artifact paths** = **relative to the snapshot root**, forward-slashed (the parser already normalizes `\`→`/`). Two rebuilds of the same logical range produce identical relative paths → identical `snapshot_id`.

**Artifact keys (added; no schema change):**
```
liquidity_daily        input/liquidity/ticker_liquidity_daily_observations.parquet
liquidity_weekly       input/liquidity/ticker_liquidity_weekly_observations.parquet
liquidity_panel        input/liquidity/ticker_liquidity_panel.parquet
liquid_tickers         input/liquidity/liquid_tickers.csv
splits                 input/adjusted_liquid/splits_hist_liquid.parquet
adjusted_chains_root   input/adjusted_liquid            (directory; presence + inventory gated, not a single file)
spot_prices            cache/spot_prices_adjusted.parquet
option_surface_meta    cache/option_surface_meta_weekly_<S>_<E>.parquet
option_surface_quotes  cache/option_surface_quotes_weekly_<S>_<E>.parquet
```

**Report keys:**
```
liquidity_report / pit_universe_audit / adjusted_audit / surface_audit / snapshot_completeness_report
```

**Identity (`snapshot_id`) inputs:** `schema_version`, `as_of_resolved_trading_day` (= resolved **end** date), `data_source` (e.g., `"orats_snapshot_backfill"`), `artifacts` (relative paths above), and `params`. **Identity-relevant `params`:** `requested_start`, `requested_end`, `resolved_start`, `resolved_end`, `frequency` (`weekly`), and the key liquidity build params already persisted by C4 (`lookback_weeks`, `min_valid_quote_weeks`, `dte_min/max`, `dvol_top_pct`, `spread_bot_pct`). These make the snapshot identity reflect the logical build definition.

**Repository SHA:** recorded in `notes` and the completeness report — **not** in identity `params`. *Tradeoff:* including SHA in identity would make two builds at different code versions produce different `snapshot_id`s (content-reproducibility flavor), but `snapshot_id` hashes *paths*, not content, so it can never truly certify content anyway; keeping SHA out preserves the stated policy "same logical inputs → same `snapshot_id`." **Choose: SHA as non-identity lineage.** *Deferred alternative:* promote SHA into identity only if a future requirement makes code-version part of snapshot identity.

**Other fields:**
- `build_id` — per execution; `created_at_utc` — timestamp.
- `overall_status` — aggregate PASS/WARN/FAIL from §9; `blocking_failures` — list of failed gates; `notes` — repo SHA, producer commands, `--splits-source` path, `sha256`s.
- **When a manifest may be written:** always at end of build (even on FAIL, into the staging root, with the true status). **A PASS manifest exists in a final root only because publication renames the root after PASS** — never write a PASS manifest into a root that will not be published.
- **FAIL manifest:** retained inside the `.failed` staging root; never inside a final `<build_id>/`.
- **Atomic manifest write:** write to `manifests/input_snapshot_<snapshot_id>.json.tmp` then `os.replace`. (Largely subsumed by the root rename, but cheap and makes staging reads consistent.) This is a 2-line hardening of `write_manifest`'s call site; the C1 module can stay as-is.

---

## 11. Failure, retry, and immutability behavior

- **Producer failure (nonzero exit):** orchestrator stops, writes FAIL manifest + report into staging, renames staging → `<build_id>.failed`, exits nonzero. No final root created.
- **Gate FAIL:** same as above (distinct exit code, §12).
- **Interruption (Ctrl+C / crash):** leaves `<build_id>.building` incomplete and unpublished. No final root is touched. Recovery = delete the `.building` dir and re-run (new `build_id`). **No resume/checkpoint** — full rebuild is the simple, safe path (aligned with the design philosophy).
- **Retry:** always a fresh `build_id` + fresh empty staging root; no partial reuse.
- **Immutability enforcement:** no C8 code path opens an existing `<build_id>/` for writing; the orchestrator refuses if the final root exists (exit 2). Optional: set final-tree files read-only after rename (Windows attrib) — *recommended but non-blocking*; convention + no-write-path is the minimum.
- **A previously accepted snapshot is never modified or invalidated** by any failed or subsequent build.

---

## 12. CLI and exit-code contract

**New entrypoint (recommended): `scripts/build_input_snapshot.py`.** Leave `refresh_weekly_inputs.py` untouched (its plan/dry-run stub stays; cleanup deferred). *Reason:* accurate name; no misleading `--mode incremental/backfill/repair` display semantics from C2; smaller blast radius.

```powershell
python scripts/build_input_snapshot.py `
    --snapshots-root C:/MomentumCVG_env/snapshots `
    --start-date 2017-01-01 `
    --end-date   2026-02-20 `
    --raw-root   C:/ORATS/data/ORATS_Data `
    --splits-source <accepted-split-file.parquet> `
    --frequency weekly `
    [--scope full | --tickers-file <bounded.txt>] `
    [--workers N] `
    [--dry-run]
```

Only **full backfill** is supported. There is no `--mode`; incremental/repair are rejected if passed via any alias.

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Snapshot built, all blocking gates PASS (WARN allowed), published (final root created) |
| 1 | Producer/subprocess failure — no publication |
| 2 | Usage/argument error: missing/invalid `--splits-source`, staging or final root collision, cross-volume roots, bad dates |
| 3 | Unsupported operation requested (any incremental/repair alias) — explicit message, no writes |
| 4 | Build ran but a **blocking gate FAILED** — no publication; staging retained as `.failed` |

**Behavior by situation:**
- **plan/dry-run:** print resolved staging/final roots, resolved (best-effort) date range, exact producer commands, and the ordered gate list; **no writes**; exit 0.
- **real backfill:** §8 sequence.
- **incremental / repair:** exit 3 with "C8 supports full backfill only."
- **subprocess failure:** exit 1.
- **audit WARN:** publish; `overall_status=WARN`; exit 0.
- **audit FAIL / missing artifact:** exit 4; no publication.
- **existing staging root:** exit 2 (never reuse).
- **existing final root:** exit 2 (never overwrite).
- **Ctrl+C:** leave `.building`; no final root touched.

---

## 13. Sprint 005 consumption contract

C8 does **not** implement the feature layer. It defines what Sprint 005 consumes.

- **Interface:** `--input-snapshot-manifest <path>` (primary). Sprint 005 uses `read_manifest` + a small resolver (`snapshot_root = manifest.cache_dir`; `artifact_abs = snapshot_root / manifest.artifacts[key]`).
- **Sprint 005 must resolve:** adjusted chains (`adjusted_chains_root`), spot prices (`spot_prices`), option surfaces (`option_surface_meta`/`option_surface_quotes`), liquidity panel (`liquidity_panel`), liquid-ticker superset (`liquid_tickers`), snapshot identity (`snapshot_id`/`build_id`), and resolved date range (`params.resolved_start/end`).
- **Artifact → feature role mapping (contract, not formulas):**
  - **Momentum — primary source: option-surface A1 (`option_surface_meta`).** The per-`(ticker, entry_date)` surface/straddle economics are the momentum input series.
  - **CVG — primary source: option-surface A1, with A2 (`option_surface_quotes`) as supporting detail** for finer strike/greek information.
  - **Eligibility / PIT inputs:** `liquidity_panel` (S1 strict prior-snapshot universe) + `liquid_tickers` (superset).
  - **Low-level fallback:** `adjusted_chains_root` + `spot_prices` are available if Sprint 005 chooses to recompute straddle history rather than consume A1/A2 directly. C8 evidences A1/A2, so A1/A2 are the recommended canonical source.
- **Lineage Sprint 005 copies into its feature manifest:** `snapshot_id`, `build_id`, repository SHA, resolved range, and the exact artifact paths consumed.
- C8 does **not** patch existing feature scripts (`precompute_straddle_history.py`, `build_features.py`, `run_surface_search.py`). Sprint 005 owns creating/hardening the canonical feature-backfill entrypoint against this contract.

---

## 14. Bounded and full evidence plan

### 14.1 Bounded evidence (fast, before the full run)
A small isolated snapshot build with `--tickers-file` (e.g., 5–10 known tickers: AAPL, MSFT, NVDA, SPY, QQQ) over a short window (e.g., Q1 2024) that exercises the **entire** path: staging-root creation → liquidity → split copy+validate → adjusted → **hardened spot** → surface → **all gates** → manifest → **publish (rename)** → **manifest-based resolution smoke**. Explicitly labeled bounded; **not** production acceptance.

### 14.2 Full production evidence (C8 acceptance)
One complete accepted snapshot for the intended Sprint 005 range. Record:
```
repository SHA · build_id · snapshot_id · snapshot root
requested + resolved dates
per-artifact row/file counts (liquidity, split, adjusted file count, spot rows, A1 rows, A2 rows)
producer commands + exit codes
audit verdicts (L, S, A, SP, SF, X)
runtime per stage · storage per stage
final manifest path
```
Then a **manifest/path-resolution smoke** (read manifest → resolve every artifact → assert existence + counts). **No backtest or feature calculation is required** for C8 closeout.

---

## 15. Implementation commit sequence

```
C8.1  Immutable input snapshot design                     (this memo)
C8.2  Harden full spot extraction
      - fail-closed exit contract; per-date failure tracking; empty-output refusal
      - (date,ticker) uniqueness; finite-positive value checks
      - expected date + per-date ticker coverage
      - temp-file + os.replace atomic write
      - unit tests for each failure mode + happy path
C8.3  Batch snapshot orchestrator  (scripts/build_input_snapshot.py)
      - same-volume preflight; empty .building staging root
      - explicit producer path wiring (liquidity, split copy, adjusted, spot, surface)
      - blocking gates L/S/A/SP/SF/X reusing C5/C6/C7 audits
      - manifest (schema v1, snapshot-root-relative, added keys) written atomically
      - publish by root rename on PASS; .failed on gate FAIL
      - unsupported incremental/repair → exit 3
C8.4  Bounded evidence, full snapshot evidence, C8 closeout memo
```
No additional implementation commit unless C8.3 review reveals a clearly separable blocking capability that cannot be reviewed safely inline. **Do not** let C8 grow into a refresh platform.

---

## 16. Explicitly deferred work

| Deferred capability | Owner |
|---------------------|-------|
| weekly append · incremental refresh · watermark operation | later operational sprint |
| split-history delta detection · selected-ticker repair | later operational sprint |
| spot row-level merge · surface row-level merge | later operational sprint |
| changed-only audits · continuous scheduling · notifications · dashboards · distributed execution · remote backup | later operational sprint |
| `current`/`latest` snapshot pointer | later (only if a consumer needs it) |
| `run_surface_search.py` cleanup (incl. the `contract_multiplier` / `SurfaceDataPaths` mismatch found in C8.0A) · legacy feature-script cleanup | Sprint 005 (feature entrypoint) or later |
| momentum/CVG **formulas**, feature-layer backfill, feature-layer audits | **Sprint 005** |
| shadow-trading update workflow · broker integration | later trading sprint |
| strategy evaluation / alpha research | later research/backtest sprint |

Category ownership:
- **Sprint 005:** momentum/CVG feature backfill + feature correctness + canonical feature entrypoint.
- **Later operational sprint:** weekly/incremental/repair behavior.
- **Later research/backtest sprint:** strategy evaluation and alpha research.

---

## 17. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Full-history rebuild runtime** (liquidity + adjusted + spot + surface from raw over ~2017–2026; ~2,299 adjusted files) could be long and delay Sprint 005 | Could inflate C8.4 | Run **bounded evidence first**; parallel `--workers`; **sanctioned fallback:** copy accepted C4 liquidity into the snapshot with recorded source + sha256 + C7 PIT audit (removes one full raw pass) if the full liquidity rebuild dominates runtime |
| **Surface full-history has no prior evidence** (C6 is bounded 5×13) | First full run may hit scale/memory/failure-rate surprises | Bounded evidence exercises the producer + gates first; tune workers; A1 completeness gate catches silent gaps |
| **Exact per-date spot ticker equality** may fail on adjusted files containing tickers with legitimately absent/bad spot | Could block otherwise-usable snapshots | Treat null/absent spot as a genuine data fault (correct fail-closed); if bounded evidence shows systemic benign gaps, revisit the denominator in C8.2 *before* the full run (documented, not silently loosened) |
| **Snapshot storage size** (adjusted chains + surfaces duplicated per snapshot) | Disk pressure with multiple snapshots | One accepted snapshot for Sprint 005; retain `.failed` roots only transiently; deletion is a manual operator action |
| **Resolved-range determination** depends on raw/adjusted inventory | Wrong bounds → wrong denominators | Resolve range from the built adjusted-chain inventory (step 7) and record requested vs resolved explicitly in the manifest |
| **Cross-volume staging/final** would break atomic rename | Non-atomic publish | Same-volume preflight assertion (exit 2 if violated) |
| **Scope-creep pressure** (adding merge/pointer/factor-recompute "while we're here") | Turns C8 into a platform | Apply the §2 decision rule; this memo lists each as deferred |

**Design risk that could materially increase C8 scope:** if bounded evidence shows the surface producer cannot complete the full range within acceptable runtime/memory, C8.3 might be tempted to add chunked/merged surface production — which is exactly the incremental-merge complexity C8 is avoiding. **Mitigation/decision:** treat surface scale as a *producer-tuning* problem (workers, date batching within a single overwrite build), **not** a merge problem; if genuinely infeasible, escalate as a scope decision rather than silently adding merge.

---

## 18. Acceptance checklist

C8 is accepted when:

- [ ] `scripts/build_input_snapshot.py` exists and supports full backfill only (incremental/repair → exit 3).
- [ ] Hardened `extract_spot_prices.py` fails closed, enforces coverage/uniqueness/values, writes atomically, and has unit tests (C8.2).
- [ ] A build creates an empty `<build_id>.building` root, wires all producer paths into it, and touches no global path.
- [ ] Blocking gates L, S, A, SP, SF, X are implemented and reuse C5/C6/C7 audits.
- [ ] A manifest (schema v1, `cache_dir`=snapshot root, relative artifact paths, added keys, `overall_status`) is written atomically.
- [ ] Publication is a root **rename** that occurs only on all-PASS; FAIL leaves `.failed` and no final root.
- [ ] Bounded evidence build passes end-to-end incl. manifest-resolution smoke.
- [ ] One full production snapshot is accepted with all recorded evidence (§14.2).
- [ ] The full snapshot is consumable via `--input-snapshot-manifest` (resolution smoke), with no copying over global roots.
- [ ] A C8 closeout memo records the above (C8.4).

---

## 19. Inspected files

**Docs:** `AGENTS.md`, `docs/agenda/current_sprint.md`, `docs/agenda/sprint004_execution_guardrails.md`, `docs/tmp/c8_0_refresh_pipeline_reality_map.md`, `docs/tmp/c8_0a_backfill_snapshot_reality_map.md`, `docs/sprint_memos/004_c4_liquidity_panel.md`, `004_c5_adjusted_liquid.md`, `004_c6_option_surface.md`, `004_c7_pit_universe.md`, `docs/tmp/c1_manifest_design_plan.md`, `docs/tmp/c2_cli_design_plan.md`, `docs/repo_map.md`, `docs/v1_weekly_runbook.md`, `docs/v1_universe_protocol.md`, `docs/surface_engine_data_contract.md`.

**Code (this task + carried from C8.0/C8.0A):** `src/data/input_snapshot.py` (full re-read — arbitrary artifact keys, identity excludes `cache_dir`, `write_manifest` non-atomic, `default_manifest_path`), `src/data/paths.py`, `trading_day.py`, `split_adjuster.py`, `spot_price_db.py`, `orats_provider.py`; `scripts/refresh_weekly_inputs.py`, `build_liquidity_panel.py`, `fetch_splits.py`, `apply_split_adjustment.py`, `extract_spot_prices.py`, `precompute_option_surface.py`, `audit_adjusted_liquid.py`, `audit_option_surface_artifacts.py`, `audit_pit_universe.py`; `src/backtest/surface_run_config.py`, `surface_runner.py`.

**Tests referenced for feasibility:** `tests/unit/test_input_snapshot.py`, `test_build_liquidity_panel.py`, `test_apply_split_adjustment_cli.py`, `test_split_adjuster.py`, `test_precompute_option_surface_cli.py`, `test_audit_adjusted_liquid.py`, `test_audit_option_surface_artifacts.py`, `test_pit_universe_audit.py`; and the confirmed **absence** of any `test_extract_spot_prices*` (the gap C8.2 closes).
