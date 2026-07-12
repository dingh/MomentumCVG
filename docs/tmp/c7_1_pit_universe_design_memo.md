# C7.1 — Point-in-time universe design memo

**Sprint:** 004 · **Task:** C7  
**Repository HEAD:** `609c15196ca1ab3644cabcabde8b96213a57bf01`  
**Created:** 2026-07-11  
**Updated:** 2026-07-11 (HD: prior-week snapshot rule — `month_date < trade_date`)  
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
| S1 uses **global** snapshot; protocol doc says per-ticker | Keep global cross-section; WARN on doc drift (G8) |
| Same-day `trade_date == week_end` in current code | **HD change:** use prior completed snapshot (`<` not `<=`); implement in C7.2 |
| Production panel carries full cross-section | Default audit may use stamp values | C7 validates mechanics for **caller-supplied** params |
| No PIT harness exists | New module + CLI |
| Tests synthetic only | Real-artifact sample plan required for closeout |
| Panel alone insufficient for rolling proof | Join weekly observations (G2) |
| Row order non-canonical | Membership hash on sorted tickers (G6) |
| Missing liquidity silent in S1 | Explicit exclusion counts in report (G12) |

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

- Change S2–S8 (except the **narrow S1 snapshot-selection line** in `step1_get_universe` per §1)
- Change signal logic, momentum/CVG, sizing, execution, fills, portfolio logic
- Run a backtest or Sharpe evaluation
- Add A4 feature validation
- Wire into `refresh_weekly_inputs.py` (C3 consumes report later; C8 owns orchestration)
- Modify accepted C4/C5/C6 evidence
- Rebuild full liquidity panel unless confirmed artifact defect + separate approval

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

**Current code at HEAD** still uses `<=` (reality map §2). C7.2 **changes** `step1_get_universe` to `<` and updates contract tests in the same commit as the audit module.

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

- Per-ticker carry-forward (protocol doc wording) — still not implemented; informational WARN only.
- Changing C4 panel build or `month_date` grain.
- **Locking** `dvol_top_pct` / `spread_bottom_pct` — backtest-tunable via `BacktestRunConfig`; C7 validates mechanics only (reality map §5C).

---

## 2. Proposed implementation files

| File | Responsibility |
|------|----------------|
| `src/data/pit_universe_audit.py` | Pure functions: artifact checks, independent reference, rolling provenance, comparison, verdict aggregation. Typed dataclasses for all results. **Must not** call `step1_get_universe` twice to "prove" correctness — production path calls S1 once; reference is separate. |
| `scripts/audit_pit_universe.py` | Standalone CLI: load artifacts, run audit, write markdown report, exit codes 0/1/2. |
| `tests/unit/test_pit_universe_audit.py` | Synthetic panel/weekly fixtures covering gap matrix (§10). |
| `tests/unit/test_audit_pit_universe_cli.py` | CLI argv, exit codes, report file creation. |
| `docs/tmp/c7_pit_universe_audit.md` | Generated report output (gitignored or archived per run). |
| `docs/sprint_memos/004_c7_pit_universe.md` | Closeout memo after implementation + production audit. |

**Not in C7.0/C7.1:** no file creation beyond design docs.

---

## 3. Independent reference calculation

The audit module implements `compute_reference_universe(trade_date, panel, config) -> ReferenceUniverseResult` with these steps:

1. **Validate required columns** — `month_date`, `ticker`, `atm_straddle_dollar_vol`, `atm_spread_pct`, `has_valid_atm_pair`. FAIL if missing.
2. **Normalize dates** — coerce `month_date` to timezone-naive `Timestamp`; reject mixed unparseable types (FAIL).
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
    exclusions: dict[str, int]  # e.g. invalid_atm_pair, nan_metrics, below_dvol, below_spread
    membership_hash: str
    production_reference_match: bool
    mismatch_tickers: tuple[str, ...]
    status: Literal["PASS", "WARN", "FAIL"]
    notes: tuple[str, ...]

@dataclass(frozen=True)
class RollingProvenanceResult:
    trade_date: date
    snapshot_date: date
    tickers_checked: tuple[str, ...]
    max_week_end_in_window: date
    offending_weeks: tuple[tuple[str, date], ...]  # (ticker, week_end_date) with week_end > snapshot
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
    samples: tuple[PitResolutionResult, ...]
    rolling_provenance: tuple[RollingProvenanceResult, ...]
    superset_coverage: tuple[SupersetCoverageResult, ...]
    overall_status: Literal["PASS", "WARN", "FAIL"]
    blocking_failures: tuple[str, ...]
    warnings: tuple[str, ...]
```

Helper: `render_audit_report_markdown(report) -> str` for CLI output.

---

## 5. PASS / WARN / FAIL policy

### FAIL (blocking)

- Missing/unreadable panel or weekly artifact path
- Required columns absent
- Duplicate `(month_date, ticker)` keys in panel
- `resolved_snapshot_date >= trade_date` (same-day or future snapshot — violates prior-week rule)
- Any weekly row with `week_end_date > snapshot_date` used in recomputed window for checked tickers
- Production S1 membership differs from independent reference (tickers or ranks beyond tolerance)
- Selected ticker with `has_valid_atm_pair=False` or NaN metrics appears in production S1 output
- Same saved inputs + params produce different **canonical** membership hash on repeat
- Mixed artifact build parameters (`lookback_weeks`, `dvol_top_pct`, etc.) within one panel file
- Selected trading-universe ticker absent from `liquid_tickers.csv` superset

### WARN (non-blocking unless `--strict`)

- `window_shortfall > 0` on sample snapshot
- Empty universe with explicit valid reason (before first snapshot, or zero eligible)
- Material membership-size movement vs prior sample (report counts; threshold: >25% relative change optional)
- New ticker excluded due to `valid_quote_weeks < min_valid_quote_weeks` (document explicitly)
- Global-vs-per-ticker protocol drift (informational)

### PASS

- All artifact checks PASS
- All sample comparisons match
- Rolling provenance PASS for checked samples
- Superset coverage PASS

**Rule:** Missing liquidity or unreadable snapshot must **never** yield silent PASS.

### Exit codes (CLI)

| Code | Meaning |
|------|---------|
| 0 | PASS (WARN allowed without `--strict`) |
| 1 | FAIL or WARN with `--strict` |
| 2 | Usage/config error |

---

## 6. Real-artifact sample plan

### Minimum samples (closeout)

| Sample | Purpose | Discovery |
|--------|---------|-----------|
| Normal trade date | Baseline ranks/membership | `--discover-samples` picks median-eligible-count Friday in central history |
| Month/holiday boundary | Snapshot edge | Latest snapshot where `month_date` month ≠ prior snapshot month, or Jan/Jul boundary |
| Missing/new liquidity | Eligibility exclusion | Ticker with `has_valid_atm_pair=False` at snapshot or first appearance with `< min_valid_quote_weeks` |

### CLI selection logic (`--discover-samples`)

Deterministic algorithm on loaded panel (no invented dates):

1. Build sorted unique `snapshot_dates = unique(month_date)`.
2. **Normal:** among snapshots in middle tertile by date, pick snapshot `S` with eligible count closest to median; set `trade_date = S` (date of snapshot).
3. **Boundary:** pick snapshot where `(month(S) != month(prev))` or `(S - prev).days > 7`; `trade_date = S`.
4. **Missing/new:** find snapshot where ∃ ticker with `has_valid_atm_pair=False` and high raw dvol (optional) OR ticker first seen within last 4 snapshots with exclusion; `trade_date = that snapshot`.

Manual override: repeatable `--sample-date YYYY-MM-DD` (multiple allowed).

### Per-sample report fields

```text
trade_date
resolved_snapshot_date
snapshot_lag_days
eligible_count
selected_count
dvol threshold
spread threshold
membership_hash
production/reference match
rolling window start/end
window_shortfall summary
explicit exclusions
liquid_tickers.csv coverage
```

### Default production paths

```text
--panel-path       C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet
--weekly-path      C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet
--liquid-tickers-path C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
```

---

## 7. Rolling provenance check

For each sample `(trade_date, resolved_snapshot, ticker_subset)`:

1. Read panel row provenance: `window_start_date`, `window_end_date`, `window_shortfall`.
2. From weekly artifact, select rows with `week_end_date <= resolved_snapshot` and `week_end_date >= window_start_date`.
3. For each ticker in sample (bounded: all selected + `--max-examples` excluded tickers):
   - Recompute eligible week set: last `lookback_weeks` distinct `week_end_date` values ≤ `resolved_snapshot`.
   - **Assert** no joined week has `week_end_date > resolved_snapshot`.
4. Optional light recompute: mean of `weekly_atm_straddle_dollar_vol` over that week set vs panel `atm_straddle_dollar_vol` within tolerance `1e-6` relative for checked tickers (WARN on drift, FAIL if future week would change mean).

**Scope:** bounded samples only — do not recompute full 2.4M-row panel in C7.

---

## 8. Determinism contract

| Item | Specification |
|------|----------------|
| Canonical ticker ordering | Sort selected tickers ascending lexicographically |
| Stable rank representation | Float64 six decimal places in hash input |
| Membership hash | `sha256` of canonical JSON: `{trade_date, resolved_snapshot, dvol_top_pct, spread_bottom_pct, members: [{ticker, dvol_rank_pct, spread_rank_pct}]}` sorted by ticker; first 16 hex chars displayed |
| Repeated-run equality | Two audit runs same paths/params → identical hash per sample |
| Shuffled panel rows | Membership hash and rank values identical; report notes row-order independence |

Production S1 output order is **not** part of the contract — comparison uses canonical sort.

---

## 9. CLI design

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
| `--dvol-top-pct` | Required for audit identity; **default `0.20`** (example baseline — backtest may use any valid value) |
| `--spread-bottom-pct` | Required for audit identity; **default `1.0`** (example baseline — backtest may tune independently of panel superset stamp) |
| `--strict` | WARN → exit 1 |
| `--output-report` | Markdown path (required on real runs) |
| `--max-examples` | Cap exclusion/offending ticker listings (default 20) |

**Not wired** into `refresh_weekly_inputs.py` during C7.

---

## 10. Test plan

Synthetic tests in `tests/unit/test_pit_universe_audit.py`:

| # | Case |
|---|------|
| T1 | Global snapshot selection (`month_date < trade_date`) |
| T2 | No same-day snapshot (`trade_date == month_date` → uses prior week) |
| T3 | Before-first-snapshot / on-first-snapshot empty universe |
| T4 | Duplicate grain → FAIL |
| T5 | Missing columns → FAIL |
| T6 | Mixed date types → FAIL or normalized |
| T7 | Invalid ATM pair exclusion |
| T8 | Missing volume/spread exclusion |
| T9 | Rank direction (dvol asc, spread desc) |
| T10 | AND filtering |
| T11 | Ties at boundary (`method=average`) |
| T12 | Shuffled-row determinism (hash stable) |
| T13 | Independent reference mismatch detection |
| T14 | Mixed build parameters in panel → FAIL |
| T15 | Superset coverage failure |
| T16 | Future weekly row in window → FAIL |
| T17 | Early-history `window_shortfall` → WARN |
| T18 | Missing/new ticker explicit classification |
| T19 | Same-day snapshot (`>= trade_date`) → FAIL |
| T20 | Friday trade uses prior Friday snapshot (typical 7-day lag) |

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

## 11. Acceptance criteria

C7 closes only when:

- [ ] This design is accepted
- [ ] Unit and CLI tests pass
- [ ] Substantive real production-panel audit report archived (`docs/tmp/c7_pit_universe_audit.md` or sprint memo attachment)
- [ ] No blocking future-data or reproducibility FAIL on accepted samples
- [ ] Prior-week snapshot rule implemented in S1 and validated on production samples
- [ ] Sprint blocker **#6** closed
- [ ] Sprint 004 remains active; C3 deferred until C8 complete
- [ ] No strategy, S2–S8, A4, Sharpe, or backtest claims introduced

---

## 12. Commit plan

| Commit | Scope |
|--------|-------|
| **C7.0** | Reality map (`c7_0_pit_universe_reality_map.md`) |
| **C7.1** | Design memo (`c7_1_pit_universe_design_memo.md`) |
| **C7.2** | `step1_get_universe` prior-week rule (`<`) + `pit_universe_audit.py` + unit/contract tests |
| **C7.3** | `audit_pit_universe.py` + CLI tests |
| **C7.4** | Bounded real production-panel audit evidence + archived report |
| **C7.5** | Narrow repairs **only** if audit finds defect beyond the S1 rule change |
| **C7.6** | Closeout memo `004_c7_pit_universe.md` + sprint-status update |

Do **not** combine implementation with C7.0/C7.1 documentation commits.

---

## Default canonical C7 policy (summary)

| Topic | Policy |
|-------|--------|
| Snapshot rule | Global `max(month_date < trade_date)` — **implement in C7.2** (prior completed week) |
| Universe thresholds | **Not locked** — `dvol_top_pct` and `spread_bottom_pct` are per-run `BacktestRunConfig` fields; C7 validates AND rank mechanics for supplied params |
| Superset build (C4) | Panel stamp 0.20 / 1.0 contracts `liquid_tickers.csv` only — separate from backtest tuning |
| Same-day Friday | **Rejected** — FAIL if `resolved_snapshot_date >= trade_date` |
| Per-ticker protocol | WARN documentation drift (unchanged) |
| Superset | **Engineering scope only** — narrows precompute (surface, signals, adjusted chains); trading universe must always be **⊆** `liquid_tickers.csv` |
| Proof method | Independent rank/threshold reference vs single S1 call |
| Missing liquidity | Explicit exclusion counts; never silent PASS |

---

## References

- [c7_0_pit_universe_reality_map.md](c7_0_pit_universe_reality_map.md)
- [004_c4_liquidity_panel.md](../sprint_memos/004_c4_liquidity_panel.md)
- [v1_universe_protocol.md](../v1_universe_protocol.md)
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) § A3, § S1
- Sprint agenda blocker #6: [current_sprint.md](../agenda/current_sprint.md)
