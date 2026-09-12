# Sprint 008 D1 cost diagnosis — corrected evidence report

- Evidence directory: `C:\MomentumCVG_env\runs\sprint008_d1_cost_diagnosis_20260912T211530Z`
- Command: `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d1_cost_diagnosis.py`
- Runtime (s): 51.39
- Code SHA: `e1248f00105add7299e3870d057ecb926c57ae37`
- Working tree: `dirty`
- Environment: `{"python": "3.13.7", "numpy": "2.4.1", "pandas": "3.0.0", "scipy": "1.17.1", "matplotlib": "3.10.8"}`
- Historical D1 gate preserved: `STOP_NO_THRESHOLDS`
- Decision status: awaiting review (no automatic recommendation)

**Review annotation (2026-09-12):** Reporting and implementation accepted in the D1 documentation closeout (commit `870d4b7`). This annotation does not change the run record below: HEAD `e1248f0`, dirty tree, evidence directory, or the numerical results. Original D1 gate remains `STOP_NO_THRESHOLDS`. Strategy decision for any later evaluation remains pending a separately accepted D2 design.
- Post-hoc disclosure: Designed after seeing D1 and within-date follow-up results; exploratory, not independent confirmation.

**Status:** corrected evidence awaiting review. Prior artifacts preserved at `C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260911T162501Z/`. This rerun does not close the research direction and does not start another experiment. Historical D1 `STOP_NO_THRESHOLDS` is unchanged.

### What this correction changed

- Drawdown now uses a running peak of \(\max(0,\text{cumulative P\&L so far})\). Regression tests cover an initially losing path (cumulated \(-\$10{,}000\), then \(-\$20{,}000\) must report \(-\$20{,}000\)) and a path that recovers.
- On **this** development path the published drawdown dollars are **unchanged** (\(-\$63{,}649.91\) baseline). The path starts at about \(-\$2{,}973\), so the omitted zero did matter for the early prefix, but a later peak-to-trough of \(-\$63{,}649.91\) remains the binding decline. Filtered drawdowns are also unchanged versus the prior run.
- Half-period **exclusion** results, weekly uplift distribution, and concentration versus total positive and total absolute negative weekly contributions are now reported. L−U half-period return gaps are not used as evidence of filter stability.
- The automatic next-experiment recommendation has been removed. Cost savings are treated as an intended mechanism, not as evidence against usefulness.
- Tests: `pytest` on D0, D1, within-date follow-up, and cost-diagnosis — **57 passed**.

Core economic results (group membership, L−U means, baseline and filtered dollar P&L, winner retention) match the prior run.

### What remains uncertain

The four Bonferroni-adjusted intervals still include zero. That does not establish no benefit. Fees are unmodeled. Quote fills are not claimed achievable. The strategy decision is awaiting review.

---

Return differences below are in **percentage points** (pp) unless labeled as dollars.
Cumulative totals cover the **complete development period** under the fixed-budget convention ($B per entry date; cash return 0).

## Weighting

Total dollar P&L equals B times the sum of date-level returns on B, so its sign follows the date-equal-weighted mean of R_t. The pooled trade mean weights each trade equally, so dates with larger N receive more weight. When N varies, a slightly negative pooled trade mean can coexist with positive total dollar P&L.

| Quantity | Pooled trade-weighted | Date-weighted return on original B (includes cash) |
|---|---:|---:|
| Gross | 3.49 pp | 3.98 pp |
| Spread drag | 3.60 pp | 3.66 pp |
| Net | -0.11 pp | 0.32 pp |
| Sample | 3585 trades | 209 dates |

Date-level identity \(G_t - A_t = R_t\): **True**.
Total baseline dollar P&L: $6,628.20.

## Reconciliation

Within-date L−U net means vs prior follow-up:

- M1: eligible 209, mean d_net 0.052389, prior 0.052389, match=True
- M2: eligible 209, mean d_net 0.075514, prior 0.075514, match=True

Core P&L vs reviewed cost-diagnosis evidence (drawdown expected to change):

- Prior evidence: `C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260911T162501Z/`
- Prior peak omitted the initial $0. Corrected peak is max(0, cumulative P&L so far).
- M1 core P&L match: **True**
  - Corrected baseline drawdown $-63,649.91 (prior $-63,649.91)
  - Corrected filtered drawdown $-43,314.51 (prior $-43,314.51)
- M2 core P&L match: **True**
  - Corrected baseline drawdown $-63,649.91 (prior $-63,649.91)
  - Corrected filtered drawdown $-41,965.84 (prior $-41,965.84)

Drawdown definition: running peak = maximum of 0 and cumulative P&L so far. This is a drawdown of cumulative fixed-budget dollar P&L, not compounded equity and not intraholding-period risk.

## Decomposition (date-weighted L minus U)

Identity: d_net = d_gross + spread_saving. Cost savings are an intended mechanism, not evidence against usefulness.

| | M1 | M2 |
|---|---:|---:|
| d_net | 5.24 pp | 7.55 pp |
| d_gross | -0.68 pp | 2.86 pp |
| spread_saving | 5.92 pp | 4.69 pp |

## Fixed U exclusion — full development

| | M1 | M2 |
|---|---:|---:|
| Dates | 209 | 209 |
| Baseline $ P&L | $6,628.20 | $6,628.20 |
| Filtered $ P&L | $20,823.08 | $25,088.18 |
| Improvement $ | $14,194.88 | $18,459.98 |
| Mean weekly uplift | 0.68 pp | 0.88 pp |
| Losses avoided $ | $121,603.70 | $124,638.89 |
| Winning profits sacrificed $ | $107,408.82 | $106,178.91 |
| Winning-profit retention | 83.8% | 84.0% |
| Top-5 winner-profit retention | 80.0% | 51.0% |
| Top-10 winner-profit retention | 88.0% | 62.9% |
| Baseline drawdown $ | $-63,649.91 | $-63,649.91 |
| Filtered drawdown $ | $-43,314.51 | $-41,965.84 |
| Half-period $ reconcile | True | True |

Winning-profit retention is retention of **baseline winners**. It is not concentration of the filter's incremental improvement.

## Fixed U exclusion — descriptive halves (no new tests)

### M1

| Half | Dates | Baseline $ | Filtered $ | Mean weekly uplift | Losses avoided $ | Winning profits sacrificed $ | Winning-profit retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020-2021 | 105 | $12,737.05 | $11,769.07 | -0.09 pp | $62,112.07 | $63,080.05 | 82.2% |
| 2022-2023 | 104 | $-6,108.85 | $9,054.01 | 1.46 pp | $59,491.63 | $44,328.77 | 85.6% |

Weekly uplift (M1), complete calendar (n=209):
- Mean 0.68 pp; median 2.39 pp; std 9.65 pp.
- Fractions positive / zero / negative: 63.6% / 0.0% / 36.4%.
- Five largest positive weeks' share of **total positive** dollar contributions: 8.7%.
- Five largest negative weeks' share of **total absolute negative** dollar contributions: 23.2%.

Largest positive weeks:

- 2020-03-27: $1,549.20 (15.49 pp)
- 2020-05-01: $1,463.38 (14.63 pp)
- 2020-10-16: $1,455.91 (14.56 pp)
- 2021-05-21: $1,395.38 (13.95 pp)
- 2023-10-13: $1,394.49 (13.94 pp)

Largest negative weeks:

- 2020-02-21: $-3,724.59 (-37.25 pp)
- 2020-02-14: $-3,395.93 (-33.96 pp)
- 2022-11-04: $-3,235.21 (-32.35 pp)
- 2020-03-06: $-2,947.33 (-29.47 pp)
- 2021-12-31: $-2,699.65 (-27.00 pp)

### M2

| Half | Dates | Baseline $ | Filtered $ | Mean weekly uplift | Losses avoided $ | Winning profits sacrificed $ | Winning-profit retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020-2021 | 105 | $12,737.05 | $15,930.95 | 0.30 pp | $63,696.97 | $60,503.08 | 82.9% |
| 2022-2023 | 104 | $-6,108.85 | $9,157.24 | 1.47 pp | $60,941.92 | $45,675.83 | 85.2% |

Weekly uplift (M2), complete calendar (n=209):
- Mean 0.88 pp; median 2.83 pp; std 10.36 pp.
- Fractions positive / zero / negative: 64.1% / 0.0% / 35.9%.
- Five largest positive weeks' share of **total positive** dollar contributions: 8.5%.
- Five largest negative weeks' share of **total absolute negative** dollar contributions: 29.9%.

Largest positive weeks:

- 2022-10-21: $1,776.36 (17.76 pp)
- 2020-05-01: $1,463.38 (14.63 pp)
- 2023-06-09: $1,416.55 (14.17 pp)
- 2022-05-06: $1,407.61 (14.08 pp)
- 2021-07-16: $1,358.77 (13.59 pp)

Largest negative weeks:

- 2021-02-19: $-5,726.63 (-57.27 pp)
- 2020-02-14: $-4,112.97 (-41.13 pp)
- 2020-02-21: $-3,749.02 (-37.49 pp)
- 2020-05-29: $-3,572.88 (-35.73 pp)
- 2023-12-08: $-3,481.49 (-34.81 pp)

## Frozen inference (family size 4)

HAC: maxlags=3, Bartlett kernel, small-sample correction, Student-t with T−1 df. Adjusted p = min(1, 4 × raw p). Adjusted interval is 98.75%.

| Contrast | n | Point estimate | HAC SE | Ordinary 95% CI | Raw p | Adjusted p | Adjusted 98.75% CI |
|---|---:|---:|---:|---|---:|---:|---|
| M1_mean_uplift | 209 | 0.68 pp | 0.72 pp | [-0.74 pp, 2.10 pp] | 0.3464 | 1.0000 | [-1.13 pp, 2.49 pp] |
| M1_winrate_LU | 209 | 1.08 pp | 2.56 pp | [-3.97 pp, 6.12 pp] | 0.6744 | 1.0000 | [-5.37 pp, 7.52 pp] |
| M2_mean_uplift | 209 | 0.88 pp | 0.73 pp | [-0.55 pp, 2.32 pp] | 0.2273 | 0.9092 | [-0.95 pp, 2.72 pp] |
| M2_winrate_LU | 209 | 4.31 pp | 2.63 pp | [-0.89 pp, 9.50 pp] | 0.1036 | 0.4143 | [-2.33 pp, 10.94 pp] |

An interval that includes zero does not establish no effect. An insignificant gross difference does not establish equivalence.

## Interpretation (facts only; decision awaiting review)

Lower execution cost can improve net economics without predicting a better gross payoff. A cost contribution is not evidence against usefulness.

- An insignificant gross-return difference does not establish equivalence.
- An insignificant uplift does not establish no benefit.
- An interval that includes zero does not establish a zero effect.
- No automatic next-experiment recommendation is issued.
- Final strategy decision awaits review.

## Limitations

- Fees remain unmodeled.
- Quote-based full-cross results do not establish achievable fills or dependable income.
- Post-hoc relative to D1 and the within-date follow-up; not independent confirmation.
- Evaluation-period outcomes remain closed.
- Broader threshold search remains unauthorized. Historical D1 STOP_NO_THRESHOLDS is preserved.

Fees remain unmodeled; quote-based results do not establish achievable fills or dependable income. No automatic close of this research direction.
