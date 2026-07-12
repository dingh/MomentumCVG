# C7.1 — Point-in-time universe design memo

**Sprint:** 004 · **Task:** C7  
**Repository HEAD:** `86013504a5a0ec0a8a5d648649e7ab632cd38d01`  
**Created:** 2026-07-11  
**Updated:** 2026-07-11 (C7.0/C7.1 follow-up — rolling provenance, envelope, discovery)  
**Prerequisite:** [c7_0_pit_universe_reality_map.md](c7_0_pit_universe_reality_map.md)  
**Status:** Design only — **no code in this commit**

---

## Objective

Implement an **input-trust gate** that answers:

> Can we prove that the weekly trading universe is selected point-in-time from the accepted C4 rolling liquidity panel, without future-data leakage, stale-snapshot ambiguity, nondeterminism, or silent acceptance of missing liquidity?

C7 validates **A3 panel + S1 membership only**. It does not evaluate signals, profitability, Sharpe, trade construction, execution, or portfolio performance.

---

## Reality-map findings driving this design

| Finding | Design response |
|---------|-----------------|
| S1 uses **global** snapshot; protocol doc says per-ticker | **Global cross-section is policy**; doc drift noted for C9 |
| Same-day `trade_date == week_end` in HEAD code | **FAIL** under strict `<`; implement in C7.2 |
| Superset built at 0.20/1.0 vs wider S1 configs | **Supported artifact envelope** enforced before coverage checks |
| Prior rolling design was tautological | **Independent C4 recomputation** from weekly grid (§7) |
| Sample discovery used `trade_date = S` | Map target snapshot **S → trade date T** with `resolve(T)=S` (§6) |
| No PIT harness exists | New module + CLI |
| Tests synthetic only | Real-artifact sample + full-history coverage |
| Row order non-canonical | Membership hash on sorted tickers |
| Missing liquidity silent in S1 | Explicit exclusion counts |
| `surface_engine_data_contract.md` says `<=` | Update in **same C7.2 commit** as S1 change |

---

## Locked boundaries

### C7 may validate

- A3 liquidity-panel schema and grain
- PIT snapshot selection
- Rolling-window provenance (bounded samples)
- S1 universe membership
- Eligibility and ranking behavior
- Reproducibility and deterministic membership
- Missing/new ticker handling (classification)
- Coverage against precompute superset
- Real production-panel sample dates

### C7 must not

- Change S2–S8, signal logic, momentum/CVG, sizing, execution, fills, portfolio logic
- Run a backtest or Sharpe evaluation
- Add A4 feature validation
- Wire into `refresh_weekly_inputs.py` (C3 after C8)
- Modify accepted C4/C5/C6 evidence
- Rebuild full liquidity panel unless confirmed artifact defect + separate approval

### Approved single S1 change (C7.2)

C7 may make **exactly one** approved production behavior change:

```text
S1 snapshot resolution:  <=  →  <
```

No other S1 eligibility, ranking, threshold, schema, or output behavior may change without separate evidence and HD approval.

---

## 1. Canonical snapshot semantics

### HD decision (2026-07-11): prior completed weekly snapshot

At trade date `t`, the universe must use liquidity from the **last completed weekly snapshot before** `t` — not the snapshot whose `month_date` equals `t`.

**Why:** At the trade decision, that day's dollar volume is not yet observable (market open or day incomplete). Panel `atm_straddle_dollar_vol` is built from **completed weekly** EOD ORATS data. Universe ranking must use the **prior completed week**, not same-day liquidity.

**Canonical rule (Sprint 004 C7 implementation target):**

```text
global_snapshot_date = max(month_date where month_date < trade_date)
```

- Cross-section taken at `month_date == global_snapshot_date` only (still **global**, not per-ticker carry-forward).
- Aligns with the C4 operator model: Saturday build after week-end → snapshot available for the **following** trade week, not the same calendar day as the rolling window close.
- For typical Friday `trade_date` and Friday `week_end_date` snapshots, this selects the **prior week's** panel row (≈7 calendar days lag).

**Current code at HEAD** still uses `<=` (reality map §2). C7.2 **changes** `step1_get_universe` to `<` and updates source-of-truth contracts in the **same commit** (§12).

### Edge cases

| Case | Behavior |
|------|----------|
| `trade_date` on or before earliest snapshot | No row with `month_date < trade_date` → **empty universe** (correct columns) |
| `trade_date` is Monday after first Friday snapshot | Uses that Friday snapshot (3-day lag) — still strictly prior to trade date |
| Non-Friday `trade_date` | Latest snapshot **strictly before** that calendar day (no special weekly calendar in S1) |

### Reporting

- Always emit `snapshot_lag_days = (trade_date - resolved_snapshot_date).days` (integer calendar days).
- **FAIL** if `resolved_snapshot_date >= trade_date` (includes same-day / future).
- **WARN** if `snapshot_lag_days > 14` (stale snapshot — possible data gap; optional threshold).

### Out of scope for this rule change

- Per-ticker carry-forward — not implemented; global cross-section is policy.
- Changing C4 panel build or `month_date` grain.

---

## 1.5 Supported artifact parameter envelope

`BacktestRunConfig` allows any legal `(0, 1]` values. The **current precomputed data layer** certifies only configurations within the envelope implied by accepted production artifacts.

### Read from panel stamp (production baseline)

```text
superset build dvol_top_pct   = 0.20   (column stamped on panel rows)
superset build spread_bot_pct = 1.0
```

### Supported S1 request against current `liquid_tickers.csv`

A run is **structurally supported** when:

```text
requested dvol_top_pct        <= superset build dvol_top_pct
requested spread_bottom_pct   <= 1.0
```

Spread nuance: superset build has spread filter fully open. Any S1 spread filter in `(0, 1]` only narrows. Requesting `dvol_top_pct > 0.20` can introduce tickers never precomputed.

### Audit behavior

1. Read superset build params from panel stamp (must be homogeneous — FAIL if mixed).
2. Report: requested S1 params, superset build params, `supported: true/false`.
3. **Blocking FAIL** when `requested dvol_top_pct > superset build dvol_top_pct` unless a broader accepted superset artifact is supplied.
4. Superset-coverage and full-history checks run **only** when configuration is supported.

**Canonical C7 audit baseline:** `dvol_top_pct=0.20`, `spread_bottom_pct=1.0`.

---

## 2. Proposed implementation files

### C7.2 behavior-changing commit (single atomic unit)

| File | Responsibility |
|------|----------------|
| `src/backtest/pipeline.py` | Change S1 snapshot line: `<=` → `<` only |
| `tests/contract/test_step1_universe_contract.py` | Update PIT contract for strict prior snapshot |
| `docs/surface_engine_data_contract.md` | S1 invariant I1: `max(month_date < trade_date)` |
| `docs/agenda/current_sprint.md` | PIT acceptance: `snapshot date < trade_date` |
| `src/data/pit_universe_audit.py` | Pure audit functions (reference, rolling recompute, envelope, coverage) |
| `tests/unit/test_pit_universe_audit.py` | Synthetic tests §10 |

`docs/v1_universe_protocol.md` — **C9** (temporarily stale; must not remain implementation contract).

### C7.3+ files

| File | Responsibility |
|------|----------------|
| `scripts/audit_pit_universe.py` | Standalone CLI, markdown report, exit codes |
| `tests/unit/test_audit_pit_universe_cli.py` | CLI tests |
| `docs/tmp/c7_pit_universe_audit.md` | Generated report (archived on C7.4) |
| `docs/sprint_memos/004_c7_pit_universe.md` | Closeout memo (C7.6) |

---

## 3. Independent reference calculation

The audit module implements `compute_reference_universe(trade_date, panel, config) -> ReferenceUniverseResult` with these steps:

1. **Validate required columns** — `month_date`, `ticker`, `atm_straddle_dollar_vol`, `atm_spread_pct`, `has_valid_atm_pair`. FAIL if missing.
2. **Normalize dates** — see §3.1 date parsing contract.
3. **Select global snapshot** — `resolved = max(month_date where month_date < trade_date)`. Empty → `ReferenceEmptyReason.BEFORE_FIRST_SNAPSHOT`.
4. **Reject duplicate grain** — if any duplicate `(month_date, ticker)` at `resolved`, FAIL artifact check (also scan full panel at load).
5. **Eligibility filter** — same predicates as S1 (`has_valid_atm_pair == True`, finite dvol/spread).
6. **Compute ranks independently** — pandas `rank(ascending=True, pct=True, method="average")` for dvol; `ascending=False` for spread.
7. **Apply thresholds independently** — `dvol_rank_pct >= 1 - dvol_top_pct` AND `spread_rank_pct >= 1 - spread_bottom_pct`.
8. **Canonical sort** — sort selected rows by `ticker` ascending (stable tie-break).
9. **Compare to production S1** — call `step1_get_universe(trade_date, panel, config)` **once** as production path; compare sets and rank columns.

### Numerical comparison tolerances

| Field | Tolerance |
|-------|-----------|
| `dvol_rank_pct`, `spread_rank_pct` | `abs(a - b) <= 1e-9` (float64 rank pct) |
| Ticker set | Exact string match (case-sensitive) |
| Row count | Exact |

Mismatch on any ticker or rank beyond tolerance → **FAIL** for that sample.

### 3.1 Date parsing contract

**Locked rule (no ambiguity):**

| Input | Behavior |
|-------|----------|
| Parseable heterogeneous date values (`date`, `datetime`, ISO strings, `Timestamp`) | Normalize to **timezone-naive** `pandas.Timestamp` at date precision |
| Timezone-aware parsed values | Convert to UTC, then drop tz → naive date (document in module docstring) |
| Any non-null unparseable value | **FAIL** artifact check |
| Null `month_date` on required rows | **FAIL** |

Do not silently drop unparseable rows.

---

## 4. Audit result model

Prefer immutable dataclasses in `src/data/pit_universe_audit.py`:

```python
@dataclass(frozen=True)
class ArtifactCheckResult:
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    message: str
    details: dict[str, object]

@dataclass(frozen=True)
class PitResolutionResult:
    target_snapshot_date: date | None  # discovery intent; must equal resolved_snapshot_date
    trade_date: date
    resolved_snapshot_date: date | None
    snapshot_lag_days: int | None
    eligible_count: int
    selected_count: int
    dvol_threshold: float
    spread_threshold: float
    window_start_date: date | None
    window_end_date: date | None
    window_shortfall: int | None
    exclusions: dict[str, int]
    membership_hash: str
    production_reference_match: bool
    mismatch_tickers: tuple[str, ...]
    status: Literal["PASS", "WARN", "FAIL"]
    notes: tuple[str, ...]

@dataclass(frozen=True)
class RollingProvenanceResult:
    target_snapshot_date: date
    tickers_checked: tuple[str, ...]
    expected_week_ends: tuple[date, ...]
    recomputed_matches_panel: bool
    field_mismatches: tuple[tuple[str, str, object, object], ...]  # ticker, field, expected, actual
    future_invariance_pass: bool
    status: Literal["PASS", "WARN", "FAIL"]

@dataclass(frozen=True)
class ArtifactEnvelopeResult:
    requested_dvol_top_pct: float
    requested_spread_bottom_pct: float
    superset_build_dvol_top_pct: float
    superset_build_spread_bot_pct: float
    supported: bool
    status: Literal["PASS", "WARN", "FAIL"]

@dataclass(frozen=True)
class FullHistorySupersetCoverageResult:
    snapshots_checked: int
    unique_selected_tickers: int
    missing_ticker_count: int
    sample_missing_tickers: tuple[str, ...]
    canonical_params: tuple[float, float]
    status: Literal["PASS", "WARN", "FAIL"]

@dataclass(frozen=True)
class SupersetCoverageResult:
    trade_date: date
    selected_tickers: tuple[str, ...]
    missing_from_superset: tuple[str, ...]
    status: Literal["PASS", "WARN", "FAIL"]

@dataclass(frozen=True)
class PitUniverseAuditReport:
    artifact_checks: tuple[ArtifactCheckResult, ...]
    artifact_envelope: ArtifactEnvelopeResult
    samples: tuple[PitResolutionResult, ...]
    rolling_provenance: tuple[RollingProvenanceResult, ...]
    sample_superset_coverage: tuple[SupersetCoverageResult, ...]
    full_history_superset_coverage: FullHistorySupersetCoverageResult | None
    overall_status: Literal["PASS", "WARN", "FAIL"]
    blocking_failures: tuple[str, ...]
    warnings: tuple[str, ...]
```

Helper: `render_audit_report_markdown(report) -> str` for CLI output.

---

## 5. PASS / WARN / FAIL policy

### FAIL (blocking)

- Missing/unreadable panel or weekly artifact path
- Required columns absent; unparseable date values (§3.1)
- Duplicate `(month_date, ticker)` keys in panel
- `resolved_snapshot_date >= trade_date` (same-day or future — strict prior-week rule)
- `requested dvol_top_pct > superset build dvol_top_pct` (unsupported artifact envelope)
- Stored panel row differs materially from **independent rolling recomputation** (§7)
- Stored window metadata inconsistent with expected global week calendar
- Historical recomputation at snapshot `S` **changes** when weekly rows with `week_end_date > S` are present (future-invariance failure)
- A future weekly observation is included in the **expected historical window** for `S`
- Production S1 membership differs from independent reference
- Selected ticker with invalid liquidity appears in S1 output
- Non-deterministic canonical membership hash on repeat
- Mixed artifact build parameters within one panel file
- Selected trading-universe ticker absent from `liquid_tickers.csv` (supported config only)
- Full-history superset coverage missing any selected ticker (canonical config)
- Automatic discovery: `target_snapshot_date != resolved_snapshot_date`

### WARN (non-blocking unless `--strict`)

- `window_shortfall > 0` on sample snapshot
- Empty universe with explicit valid reason
- Material membership-size movement between samples (optional >25% threshold)
- New ticker excluded due to insufficient `valid_quote_weeks` (explicit classification)
- Global-vs-per-ticker protocol doc drift (informational; C9 fix)

### PASS

- All artifact checks PASS
- All sample comparisons match
- Rolling provenance PASS for checked tickers on samples
- Sample + full-history superset coverage PASS (supported envelope)

**Rule:** Missing liquidity or unreadable snapshot must **never** yield silent PASS.

### Exit codes (CLI)

| Code | Meaning |
|------|---------|
| 0 | PASS (WARN allowed without `--strict`) |
| 1 | FAIL or WARN with `--strict` |
| 2 | Usage/config error |

---

## 6. Real-artifact sample plan

### Discovery invariant (strict `<` semantics)

For every automatically discovered case:

1. Select target panel snapshot **S** (a `month_date` in the panel).
2. Resolve canonical trade date **T** such that:

```text
T > S
resolved_snapshot(T) == S
```

3. Prefer the project's accepted weekly trade-date schedule when available (`weekly_trade_dates_in_range` / ORATS entry dates). Otherwise choose the **first valid intended weekly entry date strictly after S** and document the fallback in the report.

**Do not** set `trade_date = S` under strict `<` — that resolves to the snapshot **before** S.

Manual `--sample-date` supplies an operator **trade date T**, not a target snapshot.

### Report fields (every sample)

```text
target_snapshot_date
trade_date
resolved_snapshot_date
snapshot_lag_days
```

Assert: `target_snapshot_date == resolved_snapshot_date`.

### Three discovery cases

| Case | Target snapshot S | Trade date T |
|------|-------------------|--------------|
| **Normal** | Snapshot in middle tertile with median eligible count | Next scheduled trade date after S where `resolve(T)=S` |
| **Boundary/holiday** | Snapshot near month boundary or gap (`month(S) != month(prev)` or `(S-prev).days > 7`) | Next trade date T consuming S |
| **Missing/new liquidity** | Snapshot where ∃ ticker failing eligibility or first-seen with short history | Later trade date T with `resolve(T)=S` |

### Per-sample report fields (additional)

```text
eligible_count
selected_count
dvol threshold
spread threshold
membership_hash
production/reference match
rolling window start/end (from panel + recomputation)
window_shortfall summary
explicit exclusions
liquid_tickers.csv coverage (sample-level)
```

### Default production paths

```text
--panel-path            C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet
--weekly-path           C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet
--liquid-tickers-path   C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
```

---

## 7. Independent rolling provenance recomputation

**Not** a tautological filter. For each audited snapshot **S** and bounded ticker set:

### Algorithm

1. Read global sorted distinct `week_end_date` values from `ticker_liquidity_weekly_observations.parquet`.
2. Expected window = last `lookback_weeks` distinct global weeks with `week_end_date <= S`.
3. Build Cartesian grid: `checked_tickers × expected_week_end_dates`.
4. Left-join weekly observations.
5. Apply C4 missing-week conventions:
   - missing ticker-week → `weekly_atm_straddle_dollar_vol = 0`
   - missing ticker-week → not a valid-quote week
6. Independently recompute per ticker:

```text
atm_straddle_dollar_vol = sum(weekly_atm_straddle_dollar_vol) / lookback_weeks
atm_spread_pct          = mean(weekly_atm_spread_pct where weekly_has_valid_quote)
valid_quote_weeks       = count(weeks with weekly_has_valid_quote)
zero_volume_weeks       = count(weeks with zero/missing vol)
has_valid_atm_pair      = valid_quote_weeks >= min_valid_quote_weeks
window_start_date       = min(expected_week_end_dates)
window_end_date         = S
window_shortfall        = max(0, lookback_weeks - len(expected_week_end_dates))
```

7. Compare to stored panel row at `(S, ticker)`.

### Tolerances

| Field | Tolerance |
|-------|-----------|
| `atm_straddle_dollar_vol` | `abs(a-b) <= 1e-9 * max(1, abs(a))` or absolute `1e-9` |
| `atm_spread_pct` | same float64 rule |
| Integer/boolean fields | exact |
| Dates | exact after normalization |

### Future-invariance check

> Adding or retaining weekly rows with `week_end_date > S` must not alter the recomputed historical snapshot at `S`.

Procedure:

1. Recompute snapshot `S` from full weekly artifact.
2. Recompute snapshot `S` from weekly artifact restricted to `week_end_date <= S`.
3. **FAIL** if recomputed values differ.

Later weekly rows are normal inventory — their existence alone is **not** a defect. **FAIL** only if they change historical snapshot `S`, if stored panel differs from recomputation, if window metadata is wrong, or if a future week appears **inside** the expected historical window.

### Bounded ticker scope

Per sample: all **selected** tickers + deterministic bounded set of excluded/new/invalid tickers (capped by `--max-examples`). Mathematically complete for each checked `(S, ticker)` — not a partial join proof.

---

## 8. Audit scope: full artifact vs bounded recomputation

### Full artifact checks ( inexpensive on complete panel )

- Schema and required columns
- Duplicate grain
- Date parsing (§3.1)
- Build-parameter homogeneity
- Supported parameter envelope (§1.5)
- Full-history superset coverage (§9)

### Bounded rolling recomputation

- Selected tickers per sample + deterministic excluded/new/invalid examples
- Future-invariance test on at least one sample snapshot
- Does **not** recompute all 2.4M weekly rows for all tickers

---

## 9. Full-history supported-envelope superset coverage

Distinct artifact-level check (not sample-only).

For **canonical supported configuration** (`dvol_top_pct=0.20`, `spread_bottom_pct=1.0`):

1. Iterate or vectorize across all production snapshots `S`.
2. For each `S`, derive trade date `T` = next canonical weekly entry after `S` with `resolve(T)=S` (same mapping rule as discovery; document if vectorized shortcut uses `T = min(date > S)` from schedule table).
3. Independently compute S1 membership at `T`.
4. Assert every selected ticker ∈ `liquid_tickers.csv`.
5. Report: snapshots checked, unique selected tickers, missing count, sample missing tickers (capped by `--max-examples`).

Any missing selected ticker → **FAIL**.

Does **not** certify unsupported wider configurations (e.g. `dvol_top_pct=0.50`).

Reads existing artifacts only — no panel or surface rebuild.

---

## 10. Determinism contract

| Item | Specification |
|------|----------------|
| Canonical ticker ordering | Sort selected tickers ascending lexicographically |
| Stable rank representation | Float64 six decimal places in hash input |
| Membership hash | `sha256` of canonical JSON: `{trade_date, resolved_snapshot, dvol_top_pct, spread_bottom_pct, members: [{ticker, dvol_rank_pct, spread_rank_pct}]}` sorted by ticker; first 16 hex chars displayed |
| Repeated-run equality | Two audit runs same paths/params → identical hash per sample |
| Shuffled panel rows | Membership hash and rank values identical; report notes row-order independence |

Production S1 output order is **not** part of the contract — comparison uses canonical sort.

---

## 11. CLI design

### Command

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_pit_universe.py `
  --panel-path C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet `
  --weekly-path C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet `
  --liquid-tickers-path C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv `
  --sample-date 2024-06-28 `
  --output-report docs/tmp/c7_pit_universe_audit.md
```

### Flags

| Flag | Purpose |
|------|---------|
| `--sample-date` | Repeatable explicit ISO date(s) |
| `--discover-samples` | Auto-pick normal + boundary + missing/new cases |
| `--dvol-top-pct` | Default `0.20` — canonical **supported** baseline; FAIL if above superset stamp |
| `--spread-bottom-pct` | Default `1.0` — spread filter must be `<= 1.0` for current artifacts |
| `--strict` | WARN → exit 1 |
| `--output-report` | Markdown path (required on real runs) |
| `--max-examples` | Cap exclusion/offending ticker listings (default 20) |

**Not wired** into `refresh_weekly_inputs.py` during C7.

---

## 12. Test plan

Synthetic tests in `tests/unit/test_pit_universe_audit.py`:

| # | Case |
|---|------|
| T1 | Global snapshot selection (`month_date < trade_date`) |
| T2 | No same-day snapshot (`trade_date == month_date` → prior week) |
| T3 | Before-first-snapshot / on-first-snapshot empty universe |
| T4 | Duplicate grain → FAIL |
| T5 | Missing columns → FAIL |
| T6 | Parseable mixed date types → normalize; unparseable → FAIL |
| T7 | Invalid ATM pair exclusion |
| T8 | Missing volume/spread exclusion |
| T9 | Rank direction (dvol asc, spread desc) |
| T10 | AND filtering |
| T11 | Ties at boundary (`method=average`) |
| T12 | Shuffled-row determinism (hash stable) |
| T13 | Independent reference mismatch detection |
| T14 | Mixed build parameters in panel → FAIL |
| T15 | Sample superset coverage failure |
| T16 | Independent rolling recompute matches panel |
| T17 | Future-invariance: later weekly rows do not change snapshot S |
| T18 | Future week inside expected window → FAIL |
| T19 | Early-history `window_shortfall` → WARN |
| T20 | Missing/new ticker explicit classification |
| T21 | Same-day snapshot (`>= trade_date`) → FAIL |
| T22 | Discovery maps S → T with `target == resolved` |
| T23 | Unsupported envelope (`dvol > stamp`) → FAIL |
| T24 | Full-history superset coverage (synthetic mini panel) |
| T25 | Timezone-aware dates normalize to naive |

CLI tests (`tests/unit/test_audit_pit_universe_cli.py`):

| # | Case |
|---|------|
| C1 | Exit 0 on PASS fixture |
| C2 | Exit 1 on FAIL fixture |
| C3 | Exit 2 on bad paths |
| C4 | `--strict` elevates WARN |
| C5 | Report file created |

Reuse patterns from `tests/unit/test_audit_adjusted_liquid.py` and C6 surface audit CLI tests.

---

## 13. Acceptance criteria

C7 closes only when:

- [ ] C7.0/C7.1 follow-up design accepted
- [ ] Strict prior-snapshot S1 rule implemented (`<`)
- [ ] `surface_engine_data_contract.md` and sprint PIT acceptance updated in same C7.2 commit
- [ ] Independent rolling recomputation matches stored panel on accepted samples
- [ ] Historical snapshots invariant to later weekly rows (future-invariance PASS)
- [ ] Supported artifact parameter envelope explicit and enforced
- [ ] Full-history superset coverage PASS for canonical configuration (`0.20` / `1.0`)
- [ ] Automatic samples correctly consume intended target snapshots (`target == resolved`)
- [ ] No silent missing-liquidity PASS
- [ ] Unit, contract, and CLI tests pass
- [ ] Substantive real production-panel audit report archived
- [ ] No S2–S8, A4, backtest, Sharpe, or portfolio changes
- [ ] Sprint blocker **#6** closed; Sprint 004 active; C3 deferred until C8

---

## 14. Commit plan

| Commit | Scope |
|--------|-------|
| **C7.0** | Reality map (initial) |
| **C7.1** | Design memo (initial) |
| **C7.0/C7.1 follow-up** | Documentation-only corrections (this revision) |
| **C7.2** | Strict prior-snapshot S1; `surface_engine_data_contract.md` + sprint PIT wording; `pit_universe_audit.py`; unit + contract tests |
| **C7.3** | Standalone audit CLI; CLI tests; markdown rendering |
| **C7.4** | Real production audit: three mapped samples; bounded rolling recompute; full-history superset coverage; archived evidence |
| **C7.5** | Narrow repairs only if evidence shows another defect |
| **C7.6** | Closeout memo + sprint-status update |

Do **not** combine implementation with documentation commits.

---

## Default canonical C7 policy (summary)

| Topic | Policy |
|-------|--------|
| Snapshot rule | Global `max(month_date < trade_date)` — implement C7.2 |
| Ranking | dvol asc pct rank; spread desc pct rank; `method=average`; AND filter |
| Supported envelope | `requested dvol <= stamp dvol (0.20)`; `requested spread <= 1.0` |
| Superset | Engineering precompute scope; `trading_universe ⊆ liquid_tickers.csv` when supported |
| Same-day / future snapshot | **FAIL** if `resolved_snapshot >= trade_date` |
| Rolling proof | Independent C4 recomputation + future-invariance — not filter/assert |
| Per-ticker protocol | Doc drift only (C9); global cross-section is policy |
| Proof method | Independent reference + rolling recompute vs stored panel |
| Missing liquidity | Explicit exclusion; never silent PASS |

---

## References

- [c7_0_pit_universe_reality_map.md](c7_0_pit_universe_reality_map.md)
- [004_c4_liquidity_panel.md](../sprint_memos/004_c4_liquidity_panel.md)
- [v1_universe_protocol.md](../v1_universe_protocol.md)
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) § A3, § S1
- Sprint agenda blocker #6: [current_sprint.md](../agenda/current_sprint.md)
