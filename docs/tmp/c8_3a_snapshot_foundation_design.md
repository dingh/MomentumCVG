# C8.3A — Snapshot foundation design

| Field | Value |
|-------|-------|
| Expected starting HEAD | `a909b77fe689414b993100d408ac45d433c8abb1` |
| Actual inspected HEAD | `a909b77fe689414b993100d408ac45d433c8abb1` |
| Status | **Design only** — no production code |
| Allowed file | `docs/tmp/c8_3a_snapshot_foundation_design.md` |
| Grounded in | accepted C8.1, C8.2 spot/inventory, C5/C6/C7 audits, C1 `input_snapshot.py` |

---

## 1. Decision summary

C8.3A builds the **smallest reusable foundation** so C8.3B can publish one full historical snapshot and Sprint 005 / later weekly delta runs can reuse the same contracts.

| Decision | Choice |
|----------|--------|
| Isolation | Snapshot root is the transaction boundary (`.building` → final / `.failed`) |
| Module shape | One new `src/data/snapshot_foundation.py` + narrow edits to C1/C5/spot |
| Inventory | Lift C8.2 `discover_adjusted_dates` semantics into a shared resolver |
| C5 freeze | Add optional `--expected-dates`; omit → preserve today’s year-wide raw compare |
| Spot gate | Dates = resolved trading inventory; keys = retained after documented ambiguous drops |
| Identity | Logical params + upstream digests; no full-byte hash of large adjusted parquets |
| Scope field | `scope ∈ {"full","bounded","delta"}`; only `full`/`bounded` executed in C8; `delta` reserved |
| Weekly seam | Same producers + same manifest + date-set inventory; discovery/orchestration deferred |

**Production-shaped, not production-complete.**

---

## 2. Current repository findings

- **C8.1** defines lifecycle, copy identity, gates L/S/A/SP/SF/X, manifest handoff, and C8.3A/B split. Still authoritative for layout and Sprint 005 consumption.
- **C8.2** (`extract_spot_prices.py`) already implements resolved inventory: year membership before weekend exclusion; empty weekend → exclude; nonempty weekend → fail; ambiguous repeated spots → drop with warning.
- **C5 audit** (`audit_inventory`) still diffs **all raw ZIP dates in `--years`** vs adj. Newer raw files outside a frozen adj bundle incorrectly FAIL. No `--expected-dates`.
- **C1** (`input_snapshot.py`): canonical JSON + 16-hex `snapshot_id`; `build_id` is second-resolution + 6-hex (insufficient for snapshot roots); non-atomic write; incomplete artifact keys; no path-under-root resolver; no inventory/`upstream_bundle_id` helpers.
- **Surface** (`precompute_option_surface.py`): explicit `--data-root` / `--spot-db-path` / `--output-root` / dates / tickers; every key emits A1; valid failure rows need no A2; A1/A2 pair not atomic (root isolation + Gate SF covers this).
- **PIT** (`audit_pit_universe.py`): explicit panel/weekly/liquid paths + required report; reusable as Gate L subprocess.
- **C8.1 Gate SP “exact ticker set = adjusted file”** is superseded by accepted C8.2 drop semantics; do not reopen C8.2 — update gate wording in this foundation.

---

## 3. Build-now versus defer

| Item | Now? | Why |
|------|------|-----|
| Lifecycle roots, collision-resistant `build_id`, refuse reuse, same-volume rename precondition | **Now** | Wrong root = corrupt/replace published snapshot |
| Canonical digests, source/copy identity, `upstream_bundle_id` | **Now** | Changing after 005 artifacts forces remanifest |
| Adjusted physical vs resolved inventory + weekend rules | **Now** | Shared by spot, surface schedule, gates, manifest, future delta |
| C5 `--expected-dates` | **Now** | Frozen bundle + later delta audits need same boundary |
| Manifest v1 paths/fields + path escape rejection | **Now** | Sprint 005 consumes manifest, not mutable globals |
| Gate result structs + basic gate callables (no orchestration) | **Now** | Prevents silent incompleteness; cheap to unit-test |
| Spot exclusion summary schema | **Now** | Gate SP must reconcile without reopening C8.2 |
| `scope` including reserved `"delta"` | **Now** (schema only) | Avoid assuming every snapshot rebuilds all history |
| Copy C4/C5, run producers, publish, one-command orchestrator | **C8.3B** | Assembly, not contracts |
| Auto date discovery, watermarks, retries, schedule, alerts, `latest` pointer | **Weekly later** | Attach at inventory date-set + new `build_id` publish |

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

**C8.3B:** `build_input_snapshot.py` uses these primitives to copy → verify → audit → spot → surface → gates → manifest → rename.

**Weekly later:**

```text
discover new dates  →  same producers(date_set)  →  same gates  →  new build_id publish
```

Attach points only: inventory accepts an explicit date set; manifest `scope="delta"` + delta date list in `params`; producers already take roots/dates/tickers. No new artifact schema.

---

## 6. Proposed files and responsibilities

| File | New/mod | Responsibility | Durable contract | Failure prevented | Why 3A not 3B |
|------|---------|----------------|------------------|-------------------|---------------|
| `src/data/snapshot_foundation.py` | **new** | Roots, `build_id`, digests, inventory, identity, path resolve, gate types | Layout, identity, inventory | Collision, path escape, inventory drift | Pure primitives |
| `src/data/input_snapshot.py` | mod | Artifact key constants; atomic manifest write; identity hash reuse | Manifest v1 | Non-atomic/partial receipt | Shared with 3B write |
| `scripts/audit_adjusted_liquid.py` | mod | Optional `--expected-dates` inventory boundary | Frozen/delta audit | Later raw dates FAIL frozen bundle | Audit hardening |
| `scripts/extract_spot_prices.py` | mod | Call shared inventory; emit small JSON summary sidecar or return struct for 3B | Spot exclusion report | Gate SP cannot reconcile drops | Contract surface |
| `tests/unit/test_snapshot_foundation.py` | **new** | High-value foundation tests | — | Regress durable behavior | — |
| `tests/unit/test_audit_adjusted_liquid.py` | mod | Frozen-date cases | — | — | — |
| `tests/unit/test_extract_spot_prices.py` | mod | Summary + shared inventory import | — | — | — |

Keep one foundation module; do not add a workflow engine.

---

## 7. CLI and producer contracts

### C5 audit (3A change)

```text
--expected-dates <path>   # optional; UTF-8, one ISO date YYYY-MM-DD per line, # comments ok
```

When set: inventory denominator = that date set (physical filename dates). Years must equal `sorted({d.year for d in expected})` or be a validated superset derived from the file (prefer **derive years from the file** and treat `--years` as optional compatibility when flag present). Without flag: **unchanged** year-wide raw↔adj behavior.

### Spot (3A contract; light emit)

Inputs unchanged: `--data-root`, `--output`, year bounds.  
Additional durable output for 3B: `SpotProducerSummary` JSON (path next to output or returned to orchestrator):

```text
resolved_dates, weekend_excluded_dates,
retained_keys[(date,ticker)],
ambiguous_exclusions[(date,ticker)],
row_count, exit semantics
```

### Surface / PIT (no 3A CLI change)

Document required 3B argv: copied adj root, rebuilt spot, snapshot `cache/`, weekly frequency, effective universe, inventory-derived date window; PIT against copied liquidity + report under `reports/`.

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

**`upstream_bundle_id`:** 16-hex of `{c4_evidence_id, c5_evidence_id, liquid_tickers_sha256, splits_sha256, adjusted_inventory_digest}`.

**`snapshot_id`:** 16-hex over identity dict only:

```text
schema_version, data_source, scope,
as_of_resolved_trading_day (= resolved end of inventory),
artifacts (root-relative),
params: upstream_bundle_id, component digests, effective_ticker_universe_sha256,
        surface/PIT policy ids, frequency, date inventory digest,
        feature_ready_* when known, delta_dates when scope=delta
```

**Excluded from `snapshot_id`:** `build_id`, timestamps, `production_accepted`, PASS/WARN/FAIL, warnings, failures, publish result, duration, absolute machine paths.

---

## 9. Manifest v1

Keep `schema_version: "1"`. `cache_dir` = **planned final root** (never `.building`) on success path.

| Section | Fields (minimum) |
|---------|------------------|
| Identity | `snapshot_id`, `schema_version`, `data_source`, identity `params` |
| Lineage | `build_id`, `created_at_utc`, repo SHA (non-identity), `upstream_bundle_id`, evidence IDs, read-only `raw_root` note |
| Scope | `scope` (`full`\|`bounded`\|`delta`), resolved ranges, optional `delta_dates` |
| Artifacts | root-relative: `liquidity_daily/weekly/panel`, `liquid_tickers`, `splits`, `adjusted_chains_root`, `spot_prices`, `option_surface_meta/quotes` |
| Coverage | physical/resolved date counts, weekend exclusions, A1 processed vs feature-eligible counts (3B fills) |
| Producer summaries | spot summary ref; surface exit/coverage refs |
| Warnings/exclusions | allowlisted codes + spot ambiguous list ref |
| Gates | named results `{name, status, metrics, failures, warnings}` |
| Publication | `overall_status`, `production_accepted`, `blocking_failures`, `notes` |

Do not add fields that cannot be populated truthfully in 3A/3B.

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
| SP | spot dates == resolved trading dates; unique keys; exclusions ⊆ source keys; retained ∪ excluded == source ticker-dates; excluded absent from output |
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
| Weekly | same flag with delta date list; no updater in C8 |

---

## 12. Adjusted and spot inventory semantics

```text
physical_inventory  = all well-formed adj parquet filename dates (year-checked)
resolved_trading    = physical minus verified-empty weekends
weekend_excluded    = verified-empty Sat/Sun only
```

Rules (preserve C8.2): wrong-year dir → fail before weekend skip; nonempty weekend → fail; empty weekend → exclude from resolved.

**Spot:** dates = `resolved_trading`; expected keys = retained after ambiguous drops; exclusions recorded in summary; do not require excluded keys in output.

Reusable by spot, surface schedule, coverage gates, manifest, future delta selection.

---

## 13. High-value tests

- `build_id` uniqueness with injected time+UUID  
- canonical digest stability  
- existing-root refusal; path escape rejection  
- source/copy identity mismatch  
- C5: frozen expected-dates; later raw ignored; missing expected raw FAIL  
- inventory: empty weekend excluded; nonempty weekend fail; wrong-year weekend fail  
- spot exclusions reconcile with coverage  
- manifest paths valid after staging→final rename (temp dirs)  
- `scope=full` and `scope=delta` parse same manifest schema (delta execution not run)

---

## 14. Implementation order

1. Digests + path resolve + collision-resistant `build_id` + root lifecycle helpers  
2. Adjusted inventory resolver (lift from spot) + wire spot to it + summary struct  
3. C5 `--expected-dates` + tests  
4. Identity/`upstream_bundle_id` + manifest artifact keys + atomic write  
5. Gate result types + SP/path/copy unit gates  
6. Pytest green on focused suites  

---

## 15. Acceptance criteria and deferred work

**Accept when:** 3A primitives let 3B build one trusted full snapshot; Sprint 005 resolves artifacts only via manifest; weekly delta can later set `scope=delta` + date-set without changing layout, producer I/O, or manifest resolution.

**Deferred:** 3B orchestrator/copy/publish; auto discovery; watermarks; retries; scheduling; alerts; retention; `latest` pointer; remote store; feature/strategy checks.

**No blocking ambiguity** beyond operators choosing evidence IDs for `upstream_bundle_id` (pin from C4/C5 memos at 3B). C8.2 spot drop policy is accepted as-is.
