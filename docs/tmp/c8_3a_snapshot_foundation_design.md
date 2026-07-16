# C8.3A — Snapshot foundation design

| Field | Value |
|-------|-------|
| Expected starting HEAD | `e99c091ce7d5e4d3d1ab17645813991da2811701` |
| Actual inspected HEAD | `e99c091ce7d5e4d3d1ab17645813991da2811701` |
| Status | **Design only** — no production code |
| Allowed file | `docs/tmp/c8_3a_snapshot_foundation_design.md` |
| Grounded in | accepted C8.1, C8.2 spot/inventory, C5/C6/C7 audits, C1 `input_snapshot.py` |

---

## 1. Decision summary

C8.3A builds the **smallest reusable foundation** so C8.3B can publish one complete historical snapshot and later weekly operation can reuse the same contracts without partial-snapshot consumption.

| Decision | Choice |
|----------|--------|
| Isolation | Snapshot root is the transaction boundary (`.building` → final / `.failed`) |
| Module shape | One new `src/data/snapshot_foundation.py` + narrow edits to C1/C5/spot |
| Inventory | Lift C8.2 `discover_adjusted_dates` semantics into a shared resolver |
| C5 freeze | Add optional `--expected-dates`; omit → preserve today’s year-wide raw compare |
| Spot gate | Dates = resolved trading inventory; keys = retained after documented ambiguous drops |
| Identity | Unchanged C1 shape; logical config lives in `params`; no full-byte hash of large adj parquets |
| Published scope | `params.scope ∈ {"full","bounded"}` only — every published snapshot is self-contained |
| Weekly seam | Future `update_mode` may be `delta_refresh`; published artifact set remains complete |
| Spot summary | Compact diagnostics + small ambiguous-exclusion list; no retained-key dump |

**Production-shaped, not production-complete.**

---

## 2. Current repository findings

- **C8.1** defines lifecycle, copy identity, gates L/S/A/SP/SF/X, manifest handoff, and C8.3A/B split. Still authoritative for layout and Sprint 005 consumption.
- **C8.2** (`extract_spot_prices.py`) already implements resolved inventory: year membership before weekend exclusion; empty weekend → exclude; nonempty weekend → fail; ambiguous repeated spots → drop with warning.
- **C5 audit** (`audit_inventory`) still diffs **all raw ZIP dates in `--years`** vs adj. Newer raw files outside a frozen adj bundle incorrectly FAIL. No `--expected-dates`.
- **C1** (`input_snapshot.py`): identity = `schema_version`, `as_of_resolved_trading_day`, `data_source`, `artifacts`, `params`; `build_id` is second-resolution + 6-hex (insufficient for snapshot roots); non-atomic write; incomplete artifact keys; no path-under-root resolver; no inventory/`upstream_bundle_id` helpers; reader keeps only required top-level keys (unknown extras are discarded today).
- **Surface** (`precompute_option_surface.py`): explicit roots/dates/tickers; every key emits A1; valid failure rows need no A2; A1/A2 pair not atomic (root isolation + Gate SF covers this).
- **PIT** (`audit_pit_universe.py`): explicit panel/weekly/liquid paths + required report; reusable as Gate L subprocess.
- **C8.1 Gate SP “exact ticker set = adjusted file”** is superseded by accepted C8.2 drop semantics; do not reopen C8.2.

---

## 3. Build-now versus defer

| Item | Now? | Why |
|------|------|-----|
| Lifecycle roots, collision-resistant `build_id`, refuse reuse, same-volume rename precondition | **Now** | Wrong root = corrupt/replace published snapshot |
| Canonical digests, source/copy identity, `upstream_bundle_id` | **Now** | Changing after 005 artifacts forces remanifest |
| Adjusted physical vs resolved inventory + weekend rules | **Now** | Shared by spot, surface schedule, gates, manifest, future refresh |
| C5 `--expected-dates` | **Now** | Frozen bundle + later refresh audits need same boundary |
| Schema-v1 manifest compatibility + path escape rejection | **Now** | Sprint 005 consumes one manifest, not mutable globals |
| Gate result structs + basic gate callables (no orchestration) | **Now** | Prevents silent incompleteness; cheap to unit-test |
| Compact spot exclusion summary | **Now** | Gate SP reconciliation without dumping all keys |
| `params.scope` = `full` \| `bounded` | **Now** | Published artifact scope only |
| Future `update_mode` (`full_rebuild` \| `delta_refresh`) | **Defer** (document seam) | Execution lineage, not incomplete scope |
| Copy C4/C5, run producers, publish, one-command orchestrator | **C8.3B** | Assembly, not contracts |
| Auto date discovery, watermarks, retries, schedule, alerts, `latest` pointer | **Weekly later** | Attach at inventory date-set + new complete publish |

---

## 4. Exact C8.3A scope

```text
lifecycle primitives · build/snapshot identity · digests
source↔copy identity · adjusted inventory resolution
C5 frozen-date audit · basic gate structs/callables
manifest schema/path contract · path/immutability safety
producer-facing I/O contracts for C8.3B · weekly seam fields
```

Does **not** copy bundles, invoke full producer pipelines, write a populated production manifest, or publish.

---

## 5. C8.3B and weekly-shadow handoff

**C8.3B:** `build_input_snapshot.py` uses these primitives to copy → verify → audit → spot → surface → gates → manifest → rename. Publishes one **complete** `scope=full` (or non-production `bounded`) snapshot.

**Published contract:** every snapshot Sprint 005 (or later feature generation) consumes is complete, coherent, and self-contained under one manifest. Consumers never chain delta manifests, join snapshot roots, or rediscover prior artifacts.

**Weekly later (execution only):**

```text
discover new dates
+ process only required new work where supported
+ combine with the prior accepted snapshot through a controlled producer/publish step
+ validate the resulting complete artifact set
+ publish a new complete immutable snapshot
```

Merge/composition mechanism is **not** designed here. Attach points: inventory accepts an explicit date set for processing; producers already take roots/dates/tickers; future non-identity `update_mode=delta_refresh` records how the complete snapshot was produced. No new artifact schema; no partial published scope.

---

## 6. Proposed files and responsibilities

| File | New/mod | Responsibility | Durable contract | Failure prevented | Why 3A not 3B |
|------|---------|----------------|------------------|-------------------|---------------|
| `src/data/snapshot_foundation.py` | **new** | Roots, `build_id`, digests, inventory, identity helpers, path resolve, gate types | Layout, identity, inventory | Collision, path escape, inventory drift | Pure primitives |
| `src/data/input_snapshot.py` | mod | Expand artifact constants; atomic write; preserve schema-v1 identity; round-trip any optional outcome fields used | Manifest v1 | Non-atomic/partial receipt; silent field loss | Shared with 3B write |
| `scripts/audit_adjusted_liquid.py` | mod | Optional `--expected-dates` inventory boundary | Frozen/refresh audit | Later raw dates FAIL frozen bundle | Audit hardening |
| `scripts/extract_spot_prices.py` | mod | Call shared inventory; emit compact summary for 3B | Spot exclusion report | Gate SP cannot reconcile drops | Contract surface |
| `tests/unit/test_snapshot_foundation.py` | **new** | High-value foundation tests | — | Regress durable behavior | — |
| `tests/unit/test_audit_adjusted_liquid.py` | mod | Frozen-date cases | — | — | — |
| `tests/unit/test_extract_spot_prices.py` | mod | Compact summary + shared inventory | — | — | — |
| `tests/unit/test_input_snapshot.py` | mod | Old v1 load; new C8 round-trip; identity shape | — | — | — |

Keep one foundation module; do not add a workflow engine.

---

## 7. CLI and producer contracts

### C5 audit (3A change)

```text
--expected-dates <path>   # optional; UTF-8, one ISO date YYYY-MM-DD per line, # comments ok
```

When set: inventory denominator = that date set (physical filename dates). Prefer **derive years from the file**; treat `--years` as optional compatibility when the flag is present. Without flag: **unchanged** year-wide raw↔adj behavior.

### Spot (3A contract; compact emit)

Inputs unchanged: `--data-root`, `--output`, year bounds.  
Durable output for 3B: compact `SpotProducerSummary` JSON (path under `reports/` or returned to orchestrator) — **not** a full key dump:

```text
resolved_date_count, resolved_date_min, resolved_date_max
weekend_excluded_dates
source_ticker_date_key_count, source_ticker_date_key_digest
output_ticker_date_key_count, output_ticker_date_key_digest
ambiguous_exclusion_count, ambiguous_exclusions[(date, ticker)]
output_row_count, producer_status, warnings
```

`ambiguous_exclusions` may list the small unusual drop set for reconciliation. Gate SP reads adjusted source and spot parquet for actual key validation.

### Surface / PIT (no 3A CLI change)

3B argv: copied adj root, rebuilt spot, snapshot `cache/`, weekly frequency, effective universe, inventory-derived date window; PIT against copied liquidity + report under `reports/`.

---

## 8. Lifecycle and identity

```text
<snapshots-root>/<build_id>.building/   staging only
<snapshots-root>/<build_id>/            published final
<snapshots-root>/<build_id>.failed/     retained failure
```

Layout under root (unchanged from C8.1): `input/liquidity/`, `input/adjusted_liquid/`, `cache/`, `reports/`, `manifests/`.

**`build_id`:** `YYYYMMDDTHHMMSSffffffZ_<8 lowercase uuid hex>`; inject clock + UUID in tests. Refuse if staging/final/failed exists. Require staging and final on same volume before publish helper returns OK.

**Digests:** `canonical_json` (sorted keys, compact separators); `sha256_file` for small files; `adjusted_inventory_digest` = sha256 of canonical `[[rel_path, size], ...]` (forward-slash, adj-root-relative, **exclude** splits file, include all physical daily parquets).

**`upstream_bundle_id`:** 16-hex of `{c4_evidence_id, c5_evidence_id, liquid_tickers_sha256, splits_sha256, adjusted_inventory_digest}` — stored in **`params`**.

**`snapshot_id` identity shape (unchanged):**

```text
schema_version
as_of_resolved_trading_day   # resolved end of inventory
data_source
artifacts                    # root-relative paths
params                       # all identity-relevant configuration
```

**Identity-relevant `params` (examples):** `scope` (`full`\|`bounded`), `upstream_bundle_id`, component digests, `effective_ticker_universe_sha256`, surface/PIT policy ids, `frequency`, date-inventory digest, `feature_ready_*` when known.

**Excluded from `snapshot_id`:** `build_id`, `created_at_utc`, repository SHA, `overall_status`, `production_accepted`, warnings, failures, gate outcomes, publication outcome, runtime info, absolute machine paths, future `update_mode`.

---

## 9. Manifest v1

Keep `schema_version: "1"`. Do **not** introduce schema v2 or a second identity shape.

**Compatibility with `input_snapshot.py`:**

- Keep all existing required top-level fields.
- Expand artifact key constants (`liquidity_daily`, `liquidity_weekly`, `liquidity_panel`, `liquid_tickers`, `splits`, `adjusted_chains_root`, `spot_prices`, `option_surface_meta`, `option_surface_quotes`).
- Place new logical metadata in `params` (including `scope`).
- Place report path references in `reports` (C5/C7/C6 audits, spot summary, completeness).
- Place concise validation/publication text in `overall_status`, `blocking_failures`, `notes`.
- Optional outcome fields (e.g. `production_accepted`, later `update_mode`) only if reader/writer round-trip is extended explicitly so fields are not silently discarded.
- Old manifests still load; new manifests round-trip without field loss; `snapshot_id` computation shape unchanged; Sprint 005 needs one parser.

`cache_dir` = **planned final root** (never `.building`) on the success path.

| Concern | Where it lives |
|---------|----------------|
| Published scope | `params.scope` = `full` \| `bounded` |
| Upstream / policy identity | `params` |
| Artifact paths | `artifacts` (root-relative) |
| Audit/summary paths | `reports` |
| Status / failures / notes | existing outcome fields |
| Future how-built lineage | optional non-identity `update_mode` (`full_rebuild`\|`delta_refresh`) — not in C8 |

`full` / `bounded` define published artifact scope. Future `update_mode` describes production method only; the published snapshot remains self-contained either way.

---

## 10. Gate semantics

Callable pure checks (3A); orchestration order is 3B.

| Gate | Pass requires |
|------|----------------|
| Copy ID | source == copied liquid/splits sha256 and adj inventory digest |
| Exists | required copied C4/C5 paths present |
| Inv | physical inventory valid; year membership; weekend rules |
| C5 | audit exit PASS/(allowlisted WARN) with `--expected-dates` = physical adj dates |
| C7 | PIT audit PASS |
| SP | spot dates == resolved trading dates; unique keys; compact summary digests/counts reconcile with source∖ambiguous and output parquet; excluded keys absent from output |
| SF | A1 keys == universe × weekly schedule; valid failure OK without A2; `surface_valid` needs call+put body |
| Path | all artifact/report paths resolve under snapshot root; no abs/`..`; no mutable external roots |

---

## 11. C5 frozen-date audit

| Topic | Spec |
|-------|------|
| Source of dates | 3B writes expected-dates file from **physical** copied adj inventory (all `YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet` filename dates, including verified-empty weekends) |
| Format | one `YYYY-MM-DD` per line |
| Compare | for each expected date: require matching raw ZIP; **raw dates ∉ expected → ignore** |
| Missing raw inside expected | **FAIL** |
| Extra adj vs expected | **FAIL** if adj physical set ≠ expected set when flag set |
| Years | derive from expected dates; validate year dirs exist |
| Math sample | sample only within expected ∩ adj paths |
| Without flag | legacy all-raw-in-year behavior preserved |
| Weekly | same flag against the **complete** published physical inventory (or a processing subset during build); published snapshot stays complete |

---

## 12. Adjusted and spot inventory semantics

```text
physical_inventory  = all well-formed adj parquet filename dates (year-checked)
resolved_trading    = physical minus verified-empty weekends
weekend_excluded    = verified-empty Sat/Sun only
```

Rules (preserve C8.2): wrong-year dir → fail before weekend skip; nonempty weekend → fail; empty weekend → exclude from resolved.

**Spot:** dates = `resolved_trading`; expected keys = retained after ambiguous drops; exclusions in compact summary only; do not require excluded keys in output.

Reusable by spot, surface schedule, coverage gates, manifest, future refresh date selection.

---

## 13. High-value tests

- `build_id` uniqueness with injected time+UUID  
- canonical digest stability  
- existing-root refusal; path escape rejection  
- source/copy identity mismatch  
- C5: frozen expected-dates; later raw ignored; missing expected raw FAIL  
- inventory: empty weekend excluded; nonempty weekend fail; wrong-year weekend fail  
- existing schema-v1 manifest still loads  
- new C8 manifest round-trips without field loss  
- `scope` is identity-relevant through `params`; execution outcomes do not affect `snapshot_id`  
- `full` and `bounded` published manifests use the same resolver contract  
- future `update_mode=delta_refresh` metadata does not imply a partial snapshot  
- compact spot summary does not contain all retained keys  
- spot source/output key digests reconcile after ambiguous exclusions  
- manifest paths valid after staging→final rename (temp dirs)

---

## 14. Implementation order

1. Digests + path resolve + collision-resistant `build_id` + root lifecycle helpers  
2. Adjusted inventory resolver (lift from spot) + wire spot to it + compact summary  
3. C5 `--expected-dates` + tests  
4. Manifest artifact keys + atomic write + schema-v1 round-trip for any optional outcome fields  
5. Gate result types + SP/path/copy unit gates  
6. Pytest green on focused suites  

---

## 15. Acceptance criteria and deferred work

**Accept when:** 3A establishes a schema-v1-compatible foundation; 3B can publish one complete trusted historical snapshot; Sprint 005 consumes one self-contained manifest; weekly operation can later optimize processing to new dates without requiring consumers to combine partial snapshots; spot summary stays compact; no new architecture phase.

**Deferred:** 3B orchestrator/copy/publish; `update_mode` population; merge/composition; auto discovery; watermarks; retries; scheduling; alerts; retention; `latest` pointer; remote store; feature/strategy checks.

**No blocking ambiguity** beyond pinning evidence IDs for `upstream_bundle_id` at 3B from C4/C5 memos. C8.2 spot drop policy is accepted as-is.
