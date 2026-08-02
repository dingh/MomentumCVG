# Sprint 005 D1 — Feature specification + bounded Momentum/CVG audit

**Status:** Implementation-ready plan (planning only; not yet implemented)  
**Sprint mode:** Build (agenda authorized D1–D5; D2 accepted; D1 next)  
**Repository commit reviewed:** `3f598eb558f157dc84ef0a85eb512fb18f39552a`  
**Working tree at review:** clean (`main...origin/main`)  
**Canonical D2 artifact:** `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/`  
**D2 implementation lineage:** `repo_sha` `6f0d570727ce7979d7e1222466879c62ab8ba89a`

---

# Part A — Owner decision brief

Overall verdict: READY FOR IMPLEMENTATION

## 1. D1 purpose and boundaries

* **Decision:** D1 freezes one versioned weekly Momentum/CVG contract, proves it with literal production-path tests, and records a bounded D2 audit plus a short correctness memo. It does not emit the 281-window backfill.
* **Current gap:** Window grid, minimum counts, and zero-gap sign rules are not pinned in one authoritative production specification; helper/CLI defaults disagree.
* **High-level design:** Lock semantics first; run focused calculator tests; audit a small real D2 sample; correct only defects proven by failing production-path tests.
* **Input:** Accepted D2 weekly observations (`e2c1f8fd44d72176`), settled Sprint 005 constraints, and current production calculators.
* **Output:** Versioned feature spec, golden tests, bounded audit evidence, correctness memo, and at most one minimal calculator fix.
* **Owner action:** None.

## 2. Weekly feature specification

* **Decision:** Create `configs/feature_backfill_v1.json` as the sole authority for feature semantics: the 281-window grid, Momentum/CVG minimum counts, required output column names, and the predeclared `42:8` Sprint 006 baseline. It does not hard-code D2 paths or snapshot IDs.
* **Current gap:** `build_features.py` helper defaults, CLI defaults, and calculator constructor defaults can silently disagree.
* **High-level design:** D3/D4 load this file for calculator construction; runtime inputs and publication receipts record path, snapshot identity, digest, and repository SHA.
* **Input:** Settled grid and naming rules below.
* **Output:** One small, flat, versioned JSON specification consumed by tests, the D1 memo, and later D3.
* **Owner action:** None.

## 3. Momentum definition

* **Decision:** Freeze Momentum as the simple mean of available `return_pct` values over an inclusive row-based lag window; not a compounded return; units remain percentage points; publish `mom_<max>_<min>_mean` and `mom_<max>_<min>_count`.
* **Current gap:** None material in the bulk mean/count path once minimum counts are pinned by the versioned spec.
* **High-level design:** Null returns keep their scheduled week slot, reduce the count, and are excluded from the mean. Partial history is published when count meets the configured minimum.
* **Input:** D2 `return_pct` on the complete scheduled key grid.
* **Output:** Per-window mean and independent observation count.
* **Owner action:** None.

## 4. CVG definition

* **Decision:** Freeze the paper-supported formula: sum raw formation-window `vol_gap`, then subtract the feature-date cross-sectional median. Use date-median-adjusted gaps only for positive/negative proportions. Freeze `cgap == 0` as `DVG = 0`, `CVG = 1`, and median-adjusted gap `== 0` as neutral.
* **Current gap:** Sum-of-raw cumulative construction and `cgap == 0 → CVG = 1` already match production; zero adjusted gaps are still counted positive (`>= 0`).
* **High-level design:** Keep the frozen cumulative rule. Correct only the zero-neutral classification via a failing literal test.
* **Input:** D2 `vol_gap`, including usable values retained on spread-ineligible rows.
* **Output:** `cvg_<max>_<min>` and `cvg_count_<max>_<min>`, with CVG in `[0, 2]` when defined.
* **Owner action:** None.

## 5. Window grid and configuration authority

* **Decision:** Freeze `min_lag = 2..24` step 2, `max_lag = 6..60` step 2, `max_lag > min_lag`, exactly 281 windows, ordered max-lag outer / min-lag inner; baseline window `42:8`.
* **Current gap:** `generate_momentum_windows()` function default starts `max_lag` at 12 and yields 272 windows; CLI starts at 6 and yields 281.
* **High-level design:** The versioned spec stores the ranges, step, ordering rule, and baseline; tests assert exact expansion to 281 unique ordered pairs including `42:8`.
* **Input:** Settled Sprint 005 grid decision.
* **Output:** Deterministic window list used by D1 tests and D3 emit.
* **Owner action:** None.

## 6. Missing-data and partial-history behavior

* **Decision:** Freeze `momentum_min_periods = 1` and `cvg_min_periods = 1`. Publish partial-history values with counts. A signal is null only when its own count is below its minimum.
* **Current gap:** Calculator constructors default to 1, but `build_features.py` CLI defaults Momentum to 3 and CVG to 5.
* **High-level design:** Counts report available observations; Sprint 005 applies no eligibility threshold. Ranking thresholds remain Sprint 006.
* **Input:** Scheduled D2 rows, including null economics.
* **Output:** Non-null early-history features with low counts, rather than forced nulls from higher CLI defaults.
* **Owner action:** None.

## 7. Cross-sectional universe and sparse-history participation

* **Decision:** Run canonical CVG on the complete D2 cross-section before any Sprint 006 filter. Both medians use all finite participants, including spread-ineligible `vol_gap`. With `cvg_min_periods = 1`, sparse finite cumulative gaps participate in the second median.
* **Current gap:** Prefiltering tickers before CVG changes other names’ results — a D3 call-site hazard.
* **High-level design:** Calculate on the full panel; select reporting rows afterward.
* **Input:** Full D2 panel.
* **Output:** Reproducible CVG independent of later reporting subsets.
* **Owner action:** None.

## 8. Point-in-time safety

* **Decision:** Freeze `min_lag >= 2` and require every contributing observation in a feature window to satisfy `expiry_date < feature_date`.
* **Current gap:** Calculators enforce lag by row position only; PIT must be proven with expiries, especially for `42:8`.
* **High-level design:** Permanent unit tests use synthetic panels with explicit expiries. Real-data PIT checks belong only in the bounded D2 audit memo.
* **Input:** Synthetic `entry_date` / `expiry_date` in unit tests; D2 expiries in the audit.
* **Output:** CI-safe PIT unit coverage plus audit evidence for `min_lag = 2` and `42:8`.
* **Owner action:** None.

## 9. Golden-test approach

* **Decision:** Add one focused D1 test module calling real `calculate_bulk()` with tiny hand-calculated fixtures; no second implementation. All unit fixtures are synthetic and CI-local.
* **Current gap:** Existing tests do not yet gate the versioned 281-window contract, zero-neutral gaps, or synthetic PIT/`42:8` cases.
* **High-level design:** Eighteen mandatory literals; update zero-as-positive expectations only when that fix lands.
* **Input:** Inline synthetic fixtures only.
* **Output:** Failing-before / passing-after proof for any required fix; green suite without the Windows D2 artifact.
* **Owner action:** None.

## 10. Bounded real-data audit

* **Decision:** Run a small deterministic audit on the canonical D2 parquet through the real bulk calculators for a few dates, tickers, and representative windows including `42:8`, including real-data PIT expiry verification.
* **Current gap:** No D1 correctness memo yet ties calculator semantics to the accepted D2 lineage.
* **High-level design:** Verify receipt/lineage, units, counts, nulls, finite CVG in `[0, 2]`, complete-cross-section CVG, and D2 PIT expiries; report selected rows only after full-cross-section calculation.
* **Input:** Accepted D2 artifact and receipt under `derived/e2c1f8fd44d72176/` (runtime path chosen by the audit, not by the feature spec).
* **Output:** Evidence section in `docs/sprint_memos/005_feature_correctness_audit.md`.
* **Owner action:** None.

## 11. Proposed implementation scope

* **Decision:** Limit D1 to the versioned spec, focused tests, correctness memo, and—only if proven—a minimal zero-neutral fix in `cvg_calculator.py`. No D2, SurfaceRunner, backfill, ranking, or Sprint 004 changes.
* **Current gap:** None beyond what D1 is meant to close.
* **High-level design:** Spec → contract tests → optional minimal fix → bounded audit → correctness memo.
* **Input:** This approved plan.
* **Output:** The deliverables listed below.
* **Owner action:** None.

## 12. Acceptance criteria and next step

* **Decision:** Accept D1 when the versioned spec controls production choices, the 281 grid and literal Momentum/CVG/synthetic-PIT tests pass, the bounded D2 audit (including real-data PIT) is recorded, any calculator fix is test-proven, the full suite stays green, and D3 can consume the spec without reopening semantics.
* **Current gap:** D1 not yet implemented.
* **High-level design:** Implement D1 from this plan; then proceed to D3 backfill.
* **Input:** Approved plan.
* **Output:** D1 acceptance evidence; D3 authorized to emit from the frozen spec.
* **Owner action:** None.

### Decisions requiring owner approval

None. The proposed D1 contract can proceed to implementation.

### What D1 will produce

* `configs/feature_backfill_v1.json` — versioned weekly feature specification.
* Focused literal production-path tests for the mandatory Momentum/CVG/config/synthetic-PIT cases (CI-local; no D2 artifact required).
* At most one minimal `CVGCalculator` correction for neutral zero median-adjusted gaps, only if proven by a failing literal test.
* `docs/sprint_memos/005_feature_correctness_audit.md` — decision ledger and bounded D2 audit evidence.
* Enough frozen authority for D3 to implement the 281-window backfill without reopening feature semantics.

### What D1 will not do

* Full 281-window feature backfill (D3/D4).
* Backfill orchestration, publication, resume, or incremental refresh.
* Strategy ranking, joint count thresholds, or PIT liquidity filtering.
* Economic evaluation or backtesting.
* Sprint 006 modeling or trading decisions.
* Changes to D2, the accepted Sprint 004 snapshot, SurfaceRunner, or legacy chain precompute paths.

---

# Part B — Technical implementation appendix

## 1. Purpose, scope, and non-goals

**Purpose.** Give a later implementation task exact financial definitions, fixtures, file touches, and acceptance gates for Sprint 005 D1 without rediscovering economics.

**In scope.** Versioned feature specification; literal tests through `MomentumCalculator.calculate_bulk` / `CVGCalculator.calculate_bulk`; bounded audit on canonical D2; correctness memo; minimal calculator fix only when a D1 literal fails.

**Non-goals.** Full 281 emit; `scripts/backfill_features.py`; D2 republish; SurfaceRunner changes; ranking/eligibility; economic backtests; raw ORATS reads; second feature implementation; general config/audit frameworks; broad calculator cleanup.

**Blocker test used for inclusion.** Work is included only if omitting it would risk incorrect Sprint 006 feature values, PIT leakage, inconsistent configuration, broken lineage, lost scheduled coverage, incorrect CVG cross-section membership, or D3/D4 inability to consume a deterministic specification.

## 2. Repository state and evidence inspected

### 2.1 Git and agenda

| Check | Result |
|-------|--------|
| Expected HEAD | `3f598eb558f157dc84ef0a85eb512fb18f39552a` |
| Actual HEAD | `3f598eb558f157dc84ef0a85eb512fb18f39552a` (match) |
| Working tree | Clean at planning start |
| Ahead of expected HEAD? | No |
| Sprint status | `ACCEPTED`, Build mode |
| D2 status | `ACCEPTED` (2026-08-01) |
| Next deliverable | D1 |

### 2.2 Canonical D2 receipt vs artifact

Path: `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/`

| Field | Receipt / measured |
|-------|--------------------|
| `repo_sha` | `6f0d570727ce7979d7e1222466879c62ab8ba89a` (matches accepted D2 lineage) |
| `snapshot_id` | `e2c1f8fd44d72176` |
| `build_id` | `20260724T045049097520Z_40b16886` |
| `a1_key_count` / output rows | `1063995` / `1063995` |
| `a1_key_digest` | `faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` |
| digest matches manifest & output | `true` |
| `output.file_sha256` | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` (recomputed match) |
| `return_pct` non-null | 294145 |
| `vol_gap` non-null | 313946 |
| `entry_iv` non-null | 314385 |
| status counts | ok 294145; body_spread_ineligible 20240; surface_invalid 749610 |
| units (receipt) | `return_pct_units=percentage_points`, `volatility_units=annualized_decimal` |
| sample magnitudes | `return_pct` like `-59.5`, `+33.8`; `entry_iv`/`realized_volatility`/`vol_gap` annualized decimals |

**Planning blockers:** None material. D2 is closed; source identity agrees.

### 2.3 Files inspected

`AGENTS.md`; `docs/agenda/current_sprint.md`; `docs/v1_spec_pins.md`; `docs/surface_straddle_observation_transform_design.md`; D2 receipt + bounded parquet read; `src/features/{momentum_calculator,cvg_calculator,base,straddle_observations}.py`; `scripts/build_features.py`; `tests/unit/test_{momentum,cvg}_calculator.py`; `tests/unit/test_straddle_observations_compat.py`; `src/backtest/surface_run_config.py` / SurfaceRunner feature path; `docs/surface_engine_data_contract.md` A4; `configs/momentum_30_4.json` (legacy example only); existing fixtures referenced by calculator tests.

Academic PDFs were not present under the repository or `MomentumCVG_env`. Paper mapping below uses the task research anchors plus repository Fix #1 commentary/tests.

## 3. Confirmed current calculator behavior

### 3.1 Momentum bulk path (`MomentumCalculator.calculate_bulk`)

* Sorts by `ticker`, `entry_date`.
* For each `(max_lag, min_lag)`: `window_size = max_lag - min_lag + 1`.
* Rolling sum/count on `return_pct` with `min_periods=1`, then per-ticker `.shift(min_lag)`.
* Mean = sum / count; null returns excluded from sum/count but occupy grid rows.
* Output signals nulled where `count < self.min_periods`.
* Constructor default `min_periods=1` (docstring Args text incorrectly says default 3).
* Emits `mom_{max}_{min}_{mean,sum,count,std}`.

### 3.2 CVG bulk path (`CVGCalculator.calculate_bulk`)

* Optional `tickers=` filter applied **before** both cross-sectional medians.
* Stage 1: `vol_gap_adjusted = vol_gap - median(vol_gap)` per `entry_date` over the in-memory panel.
* Rolling count/mean/pos/neg on adjusted gaps; rolling sum on **raw** `vol_gap`; then `cgap = raw_sum - median(raw_sum)` per feature date.
* Positive adjusted gaps currently counted with `(x >= 0)`; negatives with `(x < 0)`.
* `cgap > 0` → `DVG = pct_neg - pct_pos`; `cgap < 0` → opposite; `cgap == 0` → `DVG = 0`; `CVG = 1 - DVG`.
* Rolling raw sum uses `min_periods=1`, so sparse finite cumulative gaps enter the second median **before** the output mask `count < min_periods`.
* Emits `cvg_*`, `dvg_*`, `cgap_*`, `pct_pos_*`, `pct_neg_*`, `volgap_mean_*`, `cvg_count_*`.

### 3.3 Window helper / CLI

```text
generate_momentum_windows(short_range=(2,24), long_range=(12,60), step=2)  # function defaults → 272
generate_momentum_windows(short_range=(2,24), long_range=(6,60), step=2)   # CLI long-min=6 → 281
```

Measured 281-list prefix/suffix:

```text
first: (6,2), (6,4), (8,2), (8,4), (8,6)
last:  (60,16), (60,18), (60,20), (60,22), (60,24)
(42,8) present: yes
order: for max_lag in long_lags: for min_lag in short_lags: if max_lag > min_lag
```

CLI defaults: `--min-periods-momentum 3`, `--min-periods-cvg 5`.

### 3.4 SurfaceRunner / A4 consumer contract

* Loads `features_{max}_{min}.parquet`.
* Requires configurable `momentum_col` (e.g. `mom_42_8_mean`), `cvg_col` (`cvg_42_8`), `count_col` (today typically `mom_42_8_count`).
* D1 freezes both momentum and CVG counts in the future emit contract; Sprint 006 eligibility policy remains out of scope.

### 3.5 Inconsistency register (verify-only in planning)

| # | Current behavior | Economic / reproducibility consequence | D1 decision | Smallest distinguishing literal | Preserve or correct? |
|---|------------------|----------------------------------------|-------------|----------------------------------|----------------------|
| 1 | Calculators default `min_periods=1`; CLI defaults mom=3, cvg=5 | Same code can emit different null masks | Spec pins both to 1; production ignores CLI defaults | Build calculators from spec; assert `min_periods` attributes equal spec; assert a 1-obs window yields non-null mean/CVG | Preserve calculator ability to accept explicit min_periods; **do not** inherit CLI defaults |
| 2 | Helper default `long_range=(12,60)` → 272 windows; docs/CLI say start at 6 → 281 | Silent wrong grid | Spec pins `max_lag.start=6`; test expects 281 | Expand from spec; assert `len==281` and set equality vs oracle list | Correct via spec authority, not by changing unrelated callers in D1 |
| 3 | Zero median-adjusted gap counted positive (`>= 0`) | Inflates `%pos`, can flip DVG when zeros present | Weekly-v1: zero is neutral | Fixture with adjusted gap exactly 0; expect not in pos or neg numerators; still in count/denominator | **Minimal correction** in bulk + single-date helpers if literal fails |
| 4 | Second median includes all finite raw cumulative gaps from rolling `min_periods=1`, then output mask applies | If `cvg_min_periods>1`, sparse names move the median without publishing their own CVG | With frozen `cvg_min_periods=1`, current order matches the intended participation rule | Two-ticker fixture where ticker B has count=1; show B changes A’s `cgap`/`cvg` | **Preserve** under min_periods=1; do not reorder mask/median in D1 |
| 5 | `tickers=` subset changes CVG medians | Prefiltering before CVG silently changes economics | Canonical calculation = complete D2 cross-section; subset only after | Full-panel bulk vs prefiltered bulk; same reported ticker differs | Preserve calculator math; **enforce call-site contract** in tests/spec/memo for D3 |

## 4. Settled project decisions

Carry forward as frozen for D1:

* D2 observations are the only feature input; complete scheduled key grid preserved; null weeks occupy slots.
* Inclusive row-based lag endpoints; 281-window grid as above; `42:8` first Sprint 006 baseline.
* Momentum = simple mean of available percentage-point returns.
* `vol_gap = realized_volatility - entry_iv` (annualized decimals).
* CVG cumulative gap = sum of **raw** formation-window `vol_gap`, then subtract feature-date cross-sectional median; date-median-adjusted gaps used only for `%pos` / `%neg`.
* Null returns / null vol gaps excluded from their own stats but retained in the grid.
* Independent Momentum and CVG counts; no Sprint 005 eligibility threshold.
* Spread-ineligible rows may retain usable IV/RV/`vol_gap` (D2 Decision D-1).
* Weekly 1×1 surface-body straddles are deliberate; monthly paper construction is not a defect target.
* Feature-spec JSON carries semantics only; lineage (path, snapshot id, digests, repo SHA) is recorded by D3 runtime inputs and publication receipts.
* Ranking, joint thresholds, PIT liquidity filtering, economics → Sprint 006.

## 5. Proposed frozen feature specification

Create `configs/feature_backfill_v1.json` approximately as:

```json
{
  "spec_version": "feature_backfill_v1",
  "spec_id": "sprint005_d1",
  "description": "Weekly Momentum/CVG feature semantics for Sprint 005 D3/D4",
  "input_schema": {
    "artifact_name": "straddle_observations_weekly",
    "required_columns": [
      "ticker", "entry_date", "return_pct",
      "entry_iv", "realized_volatility", "vol_gap", "expiry_date"
    ]
  },
  "windows": {
    "min_lag_start": 2,
    "min_lag_end": 24,
    "max_lag_start": 6,
    "max_lag_end": 60,
    "step": 2,
    "require_max_gt_min": true,
    "order": "max_lag_outer_min_lag_inner",
    "expected_count": 281
  },
  "baseline_window": { "max_lag": 42, "min_lag": 8 },
  "momentum": {
    "min_periods": 1,
    "statistic": "simple_mean_return_pct",
    "return_units": "percentage_points"
  },
  "cvg": {
    "min_periods": 1,
    "vol_gap_rule": "realized_volatility_minus_entry_iv",
    "volatility_units": "annualized_decimal",
    "first_cross_section": "all_finite_vol_gap_on_observation_date_complete_reference_panel",
    "cumulative_gap": "sum_raw_vol_gap_over_inclusive_window",
    "second_cross_section": "all_finite_raw_cumulative_gaps_on_feature_date_complete_reference_panel",
    "adjusted_gaps_used_for": "positive_negative_proportions_only",
    "zero_adjusted_gap": "neutral",
    "zero_cgap": { "dvg": 0.0, "cvg": 1.0 },
    "cross_section_timing": "calculate_before_any_eligibility_or_liquidity_filter"
  },
  "output_columns_per_window": [
    "ticker",
    "date",
    "mom_{max}_{min}_mean",
    "mom_{max}_{min}_count",
    "cvg_{max}_{min}",
    "cvg_count_{max}_{min}"
  ],
  "notes": {
    "calculator_may_emit_diagnostics": "sum/std/dvg/cgap/pct_* allowed upstream; D3/D4 required publish set is output_columns_per_window",
    "no_feature_eligibility_in_sprint005": true,
    "lineage_not_in_this_file": "D3 runtime args and publication receipts record observation path, snapshot_id, digests, and repo SHA"
  }
}
```

**Authority rule.** Any production backfill path must read this file for **feature semantics**. Constructing `MomentumCalculator` / `CVGCalculator` without passing `windows` and `min_periods` from the spec is non-conformant for Sprint 005. Observation artifact location and snapshot identity are **not** fields of this file.

## 6. Paper-to-weekly-v1 decision ledger

| Topic | Paper / research-anchor | Weekly-v1 | Classification |
|-------|-------------------------|-----------|----------------|
| Momentum average of formation returns | Monthly straddle return average `t-12..t-2` | Weekly simple mean over inclusive row lags on 281 windows | DELIBERATE ADAPTATION |
| Momentum compounding | Not used in the stated simple-average signal | Not used | MATCH (to simple-average intent) |
| `return_pct` units | Paper monthly returns | D2 percentage points | DELIBERATE ADAPTATION (units/contract from D2) |
| `vol_gap` direction | RV − IV | D2/calculator `realized_volatility - entry_iv` | MATCH |
| First cross-sectional median on gaps | Yes | Yes, complete reference-panel finite `vol_gap` | MATCH intent / DELIBERATE ADAPTATION to weekly panel + spread-ineligible retention |
| Cumulative gap | Sum raw formation-window gaps, then feature-date cross-sectional median | Same (production Fix #1); date-median-adjusted gaps used only for `%pos`/`%neg` | MATCH |
| Second median on cumulative gap | Yes | Yes, complete reference-panel finite raw cumulative gaps | MATCH intent |
| DVG sign branches | `sign(cgap) × (...); cgap==0 → 0` | Same; `CVG=1-DVG` | MATCH |
| Zero median-adjusted gap | Paper discusses pos/neg; exact zero not separately prescribed | Neutral: in count/denominator; in neither numerator | DELIBERATE ADAPTATION (explicit weekly-v1 freeze); current code REQUIRED FIX |
| Formation calendar | Monthly | Scheduled weekly surface straddles | DELIBERATE ADAPTATION |
| Partial history / counts | Not the Sprint 005 publish contract | Publish with `min_periods=1` and independent counts | DELIBERATE ADAPTATION |

## 7. Configuration shape and exact 281-window definition

Oracle expansion (to embed in tests):

```python
windows = [
    (max_lag, min_lag)
    for max_lag in range(6, 61, 2)
    for min_lag in range(2, 25, 2)
    if max_lag > min_lag
]
assert len(windows) == 281
assert windows[0] == (6, 2)
assert (42, 8) in windows
assert windows[-1] == (60, 24)
```

D3 must use this list (or the identical generator parameters), not `generate_momentum_windows()` defaults.

## 8. Golden-test fixtures and literal expected results

All cases call production `calculate_bulk` unless noted. Expected current status assumes HEAD calculators and recommended freezes.

### G1 — Exact 281-window expansion

* **Setup:** Expand from spec fields / oracle loop above.
* **Expected:** `len==281`; unique; order equals oracle; contains `(42,8)`.
* **Protects:** Grid authority.
* **Current:** PASS only when `max_lag_start=6`; FAIL for function default `(12,60)`.

### G2 — Configuration authority

* **Setup:** Load spec JSON; construct both calculators with `windows=spec_windows`, `min_periods=spec_*`.
* **Expected:** `calc.min_periods` equals spec; a one-observation Momentum window is non-null (contrast with CLI default 3).
* **Protects:** Silent CLI/constructor defaults.
* **Current:** PASS once constructed from spec; documents that CLI defaults are non-authoritative.

### G3 — Momentum inclusive endpoints (small window)

```text
tickers: AAA
dates: 2020-01-03 .. 2020-02-07 weekly (6 rows, positions 0..5)
return_pct: [10, 20, NaN, 30, 40, 50]
window: (4, 2)  # size 3
feature date: 2020-02-07 (pos 5)
rows used: pos 1..3 → 20, NaN, 30
expected: mom_4_2_mean = 25.0
          mom_4_2_count = 2
```

* **Current:** PASS.

### G4 — Momentum inclusive endpoints `42:8`

```text
ticker BBB; 50 weekly rows; return_pct = 1.0 everywhere
window (42, 8); size 35
feature date at position 42
rows used: pos 0..34 → thirty-five 1.0 values
expected: mom_42_8_mean = 1.0
          mom_42_8_count = 35
```

* **Current:** PASS.

### G5 — Retained null week

Use G3: null at pos 2 occupies the slot, reduces count to 2, mean ignores it.

* **Current:** PASS.

### G6 — Partial-history Momentum `min_periods=1`

```text
window (4, 2); feature at pos 2 → rows pos 0..0 → [10]
expected: mean=10.0, count=1 (non-null)
```

* **Current:** PASS at `min_periods=1`; would be null at CLI default 3.

### G7 — CVG first-date cross-sectional median

```text
dates d1,d2,d3; tickers AAPL=2, TSLA=6, ADP=10 each date → median 6
adjusted: -4, 0, +4
window (4,0) at d3, min_periods=1
expected after zero-neutral freeze:
  AAPL pct_pos=0, pct_neg=1
  ADP  pct_pos=1, pct_neg=0
  TSLA pct_pos=0, pct_neg=0   # zero neutral; count=3
```

* **Current:** FAIL on TSLA (`pct_pos=1` today). Supports REQUIRED FIX.

### G8 — Raw cumulative + second median

Reuse repository Fix #1 fixture (rank reversal):

```text
d1: AAPL=10, TSLA=2, ADP=4
d2: AAPL=2,  TSLA=10, ADP=4
window (2,0) at d2
raw sums: 12, 12, 8 → median 12
expected cgap(AAPL)=0; cvg(AAPL)=1
cgap(AAPL) != 4  # confirms the feature-date cross-sectional median is applied
```

* **Current:** PASS (characterizes frozen cumulative construction).

### G9 — Positive final `cgap`

Shared panel with G10. Call `CVGCalculator(windows=[(3, 0)], min_periods=1).calculate_bulk(...)`.

```text
dates: 2020-01-03, 2020-01-10, 2020-01-17   (d1, d2, d3)
vol_gap:
  A: 10, 10, 2
  B:  6,  6, 6
  C:  2,  2, 10
feature date: d3; window (3, 0) uses all three rows

Per-date median of vol_gap = 6 on every date
Adjusted gaps:
  A: +4, +4, -4
  B:  0,  0,  0
  C: -4, -4, +4
Raw sums: A=22, B=18, C=14 → feature-date median = 18
cgap: A=4, B=0, C=-4

Expected for A at d3:
  cvg_count_3_0 = 3
  pct_pos_3_0   = 2/3
  pct_neg_3_0   = 1/3
  cgap_3_0      = 4.0          # > 0
  dvg_3_0       = (1/3)-(2/3) = -1/3
  cvg_3_0       = 1 - (-1/3) = 4/3
```

* **Protects:** Positive `cgap` DVG branch under sum-of-raw cumulative rule.
* **Current:** PASS (after zero-neutral fix, B’s zeros do not affect A’s numerators).

### G10 — Negative final `cgap`

Same panel and calculator call as G9; assert ticker C.

```text
Expected for C at d3:
  cvg_count_3_0 = 3
  pct_pos_3_0   = 1/3
  pct_neg_3_0   = 2/3
  cgap_3_0      = -4.0         # < 0
  dvg_3_0       = (1/3)-(2/3) = -1/3
  cvg_3_0       = 4/3
```

* **Protects:** Negative `cgap` DVG branch (`DVG = pct_pos - pct_neg`).
* **Current:** PASS.

### G11 — Exactly zero median-adjusted gap is neutral

Dedicated bulk fixture: one adjusted value `0.0`, one `+1.0`, one `-1.0` inside the window after stage 1 (construct raw gaps so stage-1 medians produce these).

```text
count = 3
pos_count = 1
neg_count = 1
pct_pos = 1/3
pct_neg = 1/3
```

* **Current:** FAIL (`pos_count` would be 2 with `>=0`).

### G12 — Exact `cgap == 0 → DVG=0 → CVG=1`

Use Fix #2 inline panel where one ticker’s raw sum equals the cross median. Optionally also assert G9/G10 ticker B: `cgap_3_0 = 0.0`, `dvg_3_0 = 0.0`, `cvg_3_0 = 1.0`.

* **Current:** PASS.

### G13 — Sparse-history participation in the second CVG median

Isolates sparse **history** (null slots on a ticker that remains in the panel). Does **not** omit tickers from the calculation universe (that is G14).

```text
dates: 2020-01-03 (d1), 2020-01-10 (d2)
vol_gap panel (all three tickers always present):
  A: 10, 2
  B:  4, 4
  S: NaN, 6          # sparse: only one finite gap in the window
window: (2, 0) at d2; min_periods=1; calculate_bulk on the full panel

d1 finite median = median(10, 4) = 7
d2 finite median = median(2, 4, 6) = 4
Raw cumulative gaps: A=12, B=8, S=6
Second median = median(12, 8, 6) = 8
cgap: A=4, B=0, S=-2

Hand counterfactual if S were wrongly dropped from the second median only:
  median(12, 8) = 10 → cgap(A) = 2  (must NOT be the production result)

Expected production at d2:
  A: cgap_2_0 = 4.0, cvg_count_2_0 = 2,
     pct_pos_2_0 = 0.5, pct_neg_2_0 = 0.5,
     dvg_2_0 = 0.0, cvg_2_0 = 1.0
  S: cgap_2_0 = -2.0, cvg_count_2_0 = 1,
     pct_pos_2_0 = 1.0, pct_neg_2_0 = 0.0,
     dvg_2_0 = 1.0, cvg_2_0 = 0.0
```

* **Protects:** Sparse finite raw cumulative gaps participate in the feature-date median and move other tickers’ `cgap`.
* **Current:** PASS.

### G14 — Complete reference universe vs prefiltered subset

Same economic panel as G9/G10. Compare one `calculate_bulk` over `{A,B,C}` versus one over `tickers=['A','B']` only; report ticker A at d3.

```text
Full panel {A,B,C} (from G9):
  A cgap_3_0 = 4.0, cvg_3_0 = 4/3

Prefiltered tickers=['A','B'] only:
  d1/d2 median(10,6)=8; d3 median(2,6)=4
  Adjusted A: +2, +2, -2
  Raw sums A=22, B=18 → median 20
  A cgap_3_0 = 2.0
  pct_pos=2/3, pct_neg=1/3, cgap>0 → dvg=-1/3, cvg_3_0=4/3

Exact asserts:
  full_A.cgap_3_0    == 4.0
  subset_A.cgap_3_0  == 2.0
  full_A.cgap_3_0    != subset_A.cgap_3_0
```

* **Protects:** Canonical CVG must use the complete reference cross-section before any reporting subset.
* **Current:** PASS as characterization of the call-site hazard.

### G15 — Independent Momentum and CVG counts

```text
dates: 2020-01-03 .. 2020-01-31 (five Fridays; positions 0..4)
tickers X, Y
X return_pct: [10, NaN, 20, NaN, NaN]
X vol_gap:    [NaN, 0.10, 0.20, 0.30, 0.40]
Y return_pct: [1, 1, 1, 1, 1]
Y vol_gap:    [0.05, 0.05, 0.05, 0.05, 0.05]

MomentumCalculator(windows=[(4, 0)], min_periods=1)
CVGCalculator(windows=[(4, 0)], min_periods=1)
feature date: 2020-01-31 (pos 4); window uses positions 0..4

Expected for X:
  mom_4_0_count = 2
  mom_4_0_mean  = 15.0
  cvg_count_4_0 = 4
  cgap_4_0      = 0.375
  pct_pos_4_0   = 1.0
  pct_neg_4_0   = 0.0
  dvg_4_0       = -1.0
  cvg_4_0       = 2.0

CVG hand calc:
  raw_cgap(X)=1.0, raw_cgap(Y)=0.25 → median=0.625 → cgap(X)=0.375
  X adjusted gaps (vs per-date medians with Y) are all positive
```

* **Protects:** Missing returns and missing vol gaps reduce only their own counts/signals.
* **Current:** PASS.

### G16 — Synthetic PIT at `min_lag=2` (unit test)

Permanent CI unit test; **no** D2 artifact.

```text
One ticker T; six weekly Fridays starting 2020-01-03
expiry_date = entry_date + 7 days on every row
return_pct = 1.0 on every row (vol_gap unused for this assert)
window (4, 2) at feature date = last Friday (pos 5)
Contributing positions for returns: start=1, end=3
Assert every contributing row has expiry_date < feature_date
(by construction: latest contributing expiry is entry[3]+7,
 feature is entry[5], two weeks later)
```

* **Current:** PASS on synthetic panel.

### G17 — Synthetic PIT for `42:8` (unit test)

Permanent CI unit test; **no** D2 artifact.

```text
One ticker T; 50 weekly Fridays; expiry_date = entry_date + 7
return_pct = 1.0 everywhere
window (42, 8) at position 42
Contributing positions: 0..34 (35 rows)
Assert all contributing expiry_date < feature_date
Also assert mom_42_8_mean == 1.0 and mom_42_8_count == 35
```

* **Current:** PASS on synthetic panel.

### G18 — Units and CVG range

* Unit-test portion: Momentum means from percentage-point fixtures remain on that scale (G3/G4/G15); CVG finite values from G9–G15 satisfy `0 <= cvg <= 2`.
* Audit portion (not a unit test): confirm D2 `entry_iv` / `realized_volatility` / `vol_gap` are annualized decimals on the real artifact.

* **Current:** PASS for calculator range math; audit records real-data units.

**Test module recommendation:** `tests/unit/test_feature_backfill_v1_contract.py` (new; synthetic only), plus update the existing zero-as-positive assertion in `test_cvg_calculator.py` only when G11 lands. Real-data PIT stays in the bounded audit memo, not in this unit module.

## 9. Bounded real-data audit procedure

**Artifact.** Read-only:

```text
C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.parquet
C:/MomentumCVG_env/derived/e2c1f8fd44d72176/straddle_observations_weekly.lineage.json
```

**Pre-checks.** Assert receipt `repo_sha`, `snapshot_id`, `a1_key_digest`, `output.file_sha256`, and row count match §2.2. Do not rewrite the artifact.

**Deterministic sample rules.**

1. Sort unique `entry_date` ascending; select feature dates at indices **60, 220, 400**  
   → `2019-03-01`, `2022-03-25`, `2025-09-05`.
2. Tickers (fixed symbols):  
   - Mature: `AAPL`  
   - Sparse return history: first ticker alphabetically among those with `0 < return_pct_count <= 5` on the full panel (planning probe saw `AFSI` with 1)  
   - Spread-ineligible volatility retention: first row by (`entry_date`,`ticker`) with `observation_status==body_spread_ineligible` and finite `vol_gap` (probe: `A` @ `2020-03-13`)
3. Windows: `(6,2)`, `(12,2)`, `(42,8)` only.
4. CVG/Momentum calculation input: **all tickers** in the D2 panel for the date span needed by the largest window (full cross-section). Report only the selected tickers/dates afterward.
5. Implementation vehicle: a short pytest or notebook-style script inside the D1 test/memo workflow — **not** a new audit CLI or PASS/WARN/FAIL framework.

**Report.**

* Observation-count distributions for selected windows on selected dates (full cross-section): min/median/p90/max of `mom_*_count` and `cvg_count_*` among rows with count>0.
* Null rates for `mom_*_mean` and `cvg_*` on the selected report rows.
* Finite CVG within `[0, 2]` where non-null.
* Unit spot checks: sample `|return_pct|` often ≫ 1; `entry_iv`/`vol_gap` typically ≪ 1 in absolute annualized decimal space.
* **Real-data PIT evidence** (not a unit test): table of contributing rows of `AAPL` at `2022-03-25` for `(6,2)` and `(42,8)`, asserting every contributing `expiry_date < feature_date`. Planning probe already saw 35 contributors for `(42,8)` with max expiry `2022-02-04`.
* Explicit note that sparse and spread-ineligible cases were included without applying eligibility filters.
* Lineage fields recorded in the memo/receipts (path, `snapshot_id`, digests, `repo_sha`) — not copied into `feature_backfill_v1.json`.

**Forbidden.** Full-history independent recomputation of all windows; all 281 windows; coverage gates; raw ORATS; mutating D2; duplicating D2/004 validation; requiring the Windows D2 artifact for unit-test green.

## 10. Minimal proposed file changes

| File | Change |
|------|--------|
| `configs/feature_backfill_v1.json` | **Add** versioned **semantics-only** spec (no hard-coded D2 path or snapshot id) |
| `tests/unit/test_feature_backfill_v1_contract.py` | **Add** G1–G18 synthetic literals (no D2 dependency) |
| `tests/unit/test_cvg_calculator.py` | **Update** only expectations that encode zero-as-positive, after G11 |
| `src/features/cvg_calculator.py` | **Only if G11 fails:** change positive count from `>= 0` to `> 0` in bulk rolling apply and `_calculate_window_features` |
| `docs/sprint_memos/005_feature_correctness_audit.md` | **Add** decision ledger + bounded D2 audit evidence including real-data PIT |
| `docs/README.md` | **Optional** one-line index entry for the memo when D1 completes — not required for `docs/tmp` plans |

**Explicitly do not change:** D2 transform/publication; Momentum financial logic unless a literal fails (none expected); `scripts/backfill_features.py`; `scripts/build_features.py` beyond optional later non-D1 cleanup; SurfaceRunner; snapshot code; `refresh_weekly_inputs.py`; legacy precompute; dependencies.

## 11. Dependency-aware implementation sequence

1. Add `configs/feature_backfill_v1.json` with the frozen semantic fields (sum-of-raw CVG; no path/snapshot hard-coding).
2. Add G1–G2 (spec/grid/authority) and G3–G6 (Momentum); run them.
3. Add G8–G10, G12–G15 (CVG) under the frozen cumulative rule; run them.
4. Add G11 (neutral zero); confirm FAIL; apply minimal `cvg_calculator.py` patch; update conflicting legacy expectation; confirm PASS.
5. Add G16–G17 synthetic PIT unit tests and G18 unit-test range/units asserts; run them (no D2).
6. Run bounded D2 audit (including real-data PIT); write `docs/sprint_memos/005_feature_correctness_audit.md` with MATCH / DELIBERATE ADAPTATION / REQUIRED FIX ledger.
7. Run focused D1 tests, then full suite; confirm no D2/004/SurfaceRunner diffs.
8. Hand off to D3 with the semantics spec; D3 records runtime path/lineage in its own inputs and receipts.

## 12. Objective D1 acceptance criteria

1. `configs/feature_backfill_v1.json` is the sole authority for windows, min periods, baseline `42:8`, CVG cumulative rule, and required publish columns — and does not hard-code D2 path or snapshot id.
2. Grid expands to exactly 281 unique deterministically ordered windows including `42:8`.
3. Momentum literals prove mean, inclusive lags, null-slot retention, counts, and partial history at `min_periods=1`.
4. CVG literals prove both medians, both nonzero sign branches, neutral zero adjusted gaps, `cgap==0→CVG=1`, sparse second-median participation, and full-universe vs subset hazard.
5. Synthetic `min_lag=2` and `42:8` PIT unit tests pass without the Windows D2 artifact.
6. Independent Momentum/CVG counts preserved when inputs differ.
7. Bounded audit uses the canonical D2 artifact, records receipt lineage digests/SHA, and documents real-data PIT for `(6,2)` and `(42,8)`.
8. Any calculator fix has failing-before / passing-after literal proof.
9. Existing relevant tests and the full repository suite remain green after intentional expectation updates.
10. No D2, Sprint 004, ranking, or backtesting behavior changes.
11. Correctness memo records MATCH / DELIBERATE ADAPTATION / REQUIRED FIX for each material definition.
12. D3 can implement backfill from the semantics spec without reopening feature economics, while supplying its own runtime lineage fields.

## 13. Risks and genuine unresolved owner decisions

**Owner decisions.**

None. The proposed D1 contract can proceed to implementation.

**Non-blocking risks.**

* `calculate()` vs `calculate_bulk()` still differ on collapsed-window count `0` vs `NaN`; D1 bulk literals should prefer bulk semantics; do not expand into a consistency rewrite unless a bulk literal fails.
* Bounded audit still requires local access to `MomentumCVG_env` derived data; unit tests must not.
* Existing `MomentumCVGStrategy.min_count_pct` remains a Sprint 006 concern; D1 must not silently reintroduce CLI min_periods=3/5 as a substitute eligibility policy.

## 14. Evidence-to-design mapping

| Design element | Evidence |
|----------------|----------|
| D2-only input + full grid | D2 design §1; receipt key digest 1:1; calculator row-positional lookback |
| 281 windows / not 272 | Measured `generate_momentum_windows`; agenda frozen grid; inconsistency #2 |
| Spec authority over defaults | Agenda “do not silently inherit”; CLI vs constructor vs helper mismatch |
| Semantics vs lineage split | Spec omits path/snapshot; D2 receipt already carries lineage fields for D3 to mirror |
| Momentum simple mean | `calculate_bulk` sum/count; settled constraints |
| CVG two medians + DVG branches | `cvg_calculator.py` bulk path; Fix #2 tests |
| Cumulative = sum raw | Paper-supported freeze; Fix #1 tests reject sum-of-adjusted for `%`-independent cgap |
| Zero adjusted gap neutral | Weekly-v1 freeze; current `>=0` mismatch; G11 |
| min_periods=1 | Settled baseline; avoids CLI 3/5; publishes counts |
| Full cross-section CVG | G14 exact fixture; inconsistency #5; D2 spread-ineligible vol retention |
| Sparse second-median participation | G13 exact fixture; bulk median before mask |
| Compact publish columns | A4 consumer needs mom mean, cvg, counts; extras optional upstream |
| Synthetic PIT unit tests | G16/G17 constructed expiry panels; CI-local |
| Real-data PIT audit | D2 AAPL `42:8` probe; memo §9 |
| Minimal fix surface | Only neutral-zero requires code change under frozen freezes |
| Audit sample indices 60/220/400 | Deterministic measured dates with early/mid/late coverage |

---

**End of plan.** Part A is sufficient for owner approval to implement. Part B is the implementation contract for the later D1 build task.
