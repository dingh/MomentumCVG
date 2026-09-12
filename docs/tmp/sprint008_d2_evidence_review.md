# Sprint 008 D2 — evidence review

**Status:** `EXECUTED — AWAITING REVIEW`  
**Executed:** 2026-09-12  
**Design:** [`sprint008_d2_design.md`](sprint008_d2_design.md) — accepted 2026-09-12, reviewed commit `c2ba972`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint008_d2_20260912T232144Z/`  
**Characterization:** Retrospective validation of already frozen rules. Not a pristine holdout or independent confirmation. Historical D1 findings and `STOP_NO_THRESHOLDS` are unchanged. This run does not promote a filter.

---

## Question answered

> Do the two existing exclusion rules improve the long book during the later period, and what valuable winners do they sacrifice?

| Rule | Adjusted 97.5% interval | Label |
|---|---|---|
| M1 exclude-U (`H/M`) | entirely above 0 | `relative_benefit` |
| M2 exclude-U (`H/S0`) | includes 0 | `inconclusive` |

These labels are relative to the unfiltered baseline on the original budget \(B\). They are not a decision to trade either book, and they do not reopen threshold search.

Separate facts:

- **Relative improvement:** M1’s mean weekly uplift is positive and its Bonferroni-adjusted interval excludes 0. M2’s point uplift is smaller and inconclusive.
- **Absolute historical profitability:** The unfiltered later-period book loses money. Both filtered books also lose money over the full evaluation window. A relative gain on a losing book is not a profitable book.
- **Reliable advantage:** Not established. The window was inspected in earlier sprints. Fees are 0. Quote crosses are not attainable fills. Significance does not promote a filter.

An inconclusive M2 result completes that contrast. No cutoff was tuned after seeing results. D3 remains the subsequent closeout.

---

## Calendar and validation

Authority: official Sprint 006 `date_status_sprint006_baseline_v1_mid.parquet`, joined one-to-one to `funnel_summary_sprint006_baseline_v1_mid.parquet`, reconciled to reconstructed `in_N`. Mid and cross `date_status` agreed on the evaluation window. Metadata row count 403 on both status files. Dates after `2026-07-10`: 0.

| Check | Result |
|---|---|
| Window | `2024-01-01` through `2026-07-10` inclusive |
| Calendar dates | 132; both uplift series have these 132 dates |
| Verified \(N=0\) | 0 (no missing-data date was converted to cash) |
| Dates with long candidates | 132 |
| Valid score split | 132 of 132 (no date retained baseline for lack of a split) |
| `in_N` rows | 2305; D0 required-input failures 0 |
| Executed (association-valid) | 2304 |
| Crossed-quote cash | 1; stayed in \(N\); stake cash; not analysis-eligible |
| Successive-date gaps | median 7 days, max 8 days |
| Period dollar reconcile | True for M1 and M2 |
| P&L identity | incremental P&L = losses avoided − winning profits sacrificed |

---

## Primary inference

Family size **2**. HAC maxlags 3, Bartlett, small-sample correction, Student-\(t\) with \(T-1=131\). Adjusted \(p=\min(1, 2\times p_{\mathrm{raw}})\). Adjusted interval 97.5%. One observation per calendar date, including any zero-return dates (none were verified \(N=0\)).

| Rule | Mean uplift | HAC SE | Ordinary 95% CI | Raw p | Adjusted p | 97.5% CI | Label |
|---|---:|---:|---|---:|---:|---|---|
| M1 | 2.36 pp | 0.61 pp | [1.15 pp, 3.56 pp] | 0.0002 | 0.0003 | [0.98 pp, 3.74 pp] | `relative_benefit` |
| M2 | 0.84 pp | 0.89 pp | [−0.92 pp, 2.59 pp] | 0.3465 | 0.6930 | [−1.17 pp, 2.85 pp] | `inconclusive` |

---

## Economics

Budget \(B=\$10{,}000\) per date. Fees \(=0\), disclosed. Full cross \(h=1\). Original \(N\) unchanged; excluded stakes cash.

| | Baseline | M1 exclude-U | M2 exclude-U |
|---|---:|---:|---:|
| Absolute P&L | \$-39,244.98 | \$-8,150.05 | \$-28,194.16 |
| Incremental P&L | — | \$31,094.93 | \$11,050.82 |
| Mean return on \(B\) | −2.97 pp | −0.62 pp | −2.14 pp |
| Losses avoided | — | \$80,881.80 | \$75,448.80 |
| Winning profits sacrificed | — | \$49,786.86 | \$64,397.98 |
| Winning-profit retention | — | 87.0% (778/914 winners) | 83.2% (761/914) |
| Top-5 winner-profit retention | — | 100% (5/5) | 76.8% (4/5) |
| Top-10 winner-profit retention | — | 100% (10/10) | 86.7% (9/10) |
| Drawdown | \$-53,311.48 | \$-37,901.86 | \$-46,184.30 |
| Dates exclusion applied | — | 132/132 | 132/132 |
| Executed retained | 2304 | 1906 (398 excluded) | 1906 (398 excluded) |
| Mean invested / cash (filtered) | 100.0% invested | 82.7% / 17.3% | 82.7% / 17.3% |

Actual exclusion is about 17.3% of executed trades (\(398/2304\)), consistent with \(k=\lfloor n/5\rfloor\) plus the one crossed-quote cash name remaining in \(N\).

Drawdown is peak-to-trough of cumulative fixed-budget dollar P&L. The running peak includes the initial zero. It is not compounded equity and not intraholding-period risk.

### Weekly uplift

| | M1 | M2 |
|---|---:|---:|
| Median | 3.52 pp | 2.20 pp |
| Std | 7.35 pp | 8.27 pp |
| Positive / zero / negative | 70.5% / 0% / 29.5% | 63.6% / 0% / 36.4% |
| Top-5 share of **positive** dollar contributions | 11.6% | 14.1% |
| Top-5 share of **absolute negative** dollar contributions | 30.4% | 28.8% |

Shares are of positive contributions and of absolute negative contributions separately, not of the net improvement.

Largest positive weeks (M1): 2025-05-02 \$1,530.54; 2025-03-21 \$1,366.27; 2025-12-05 \$1,328.35; 2025-04-11 \$1,286.26; 2025-01-17 \$1,240.29.

Largest negative weeks (M1): 2026-06-12 \$-1,723.76; 2024-08-30 \$-1,707.05; 2025-07-25 \$-1,683.06; 2024-08-09 \$-1,649.38; 2024-07-05 \$-1,412.76.

Largest positive weeks (M2): 2025-05-02 \$1,651.56; 2024-08-23 \$1,379.28; 2026-02-06 \$1,299.59; 2025-04-11 \$1,294.57; 2024-03-01 \$1,181.15.

Largest negative weeks (M2): 2025-09-12 \$-4,364.21; 2025-09-26 \$-1,723.80; 2025-11-21 \$-1,666.29; 2026-06-12 \$-1,551.50; 2025-04-17 \$-1,415.44.

---

## Period slices

Descriptive only. 2026 is a **partial year** ending at the pinned boundary `2026-07-10`.

| Rule | Slice | Dates | Baseline $ | Filtered $ | Mean uplift | Losses avoided | Sacrificed | Winner retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 2024 full | 52 | \$-3,832.74 | \$5,969.72 | 1.89 pp | \$29,222.96 | \$19,420.50 | 87.3% |
| M1 | 2025 full | 52 | \$-18,054.31 | \$-4,374.36 | 2.63 pp | \$33,130.66 | \$19,450.71 | 87.2% |
| M1 | 2026 partial | 28 | \$-17,357.93 | \$-9,745.41 | 2.72 pp | \$18,528.18 | \$10,915.65 | 86.1% |
| M2 | 2024 full | 52 | \$-3,832.74 | \$-3,521.72 | 0.06 pp | \$27,165.29 | \$26,854.27 | 82.5% |
| M2 | 2025 full | 52 | \$-18,054.31 | \$-17,021.67 | 0.20 pp | \$28,849.65 | \$27,817.01 | 81.7% |
| M2 | 2026 partial | 28 | \$-17,357.93 | \$-7,650.77 | 3.47 pp | \$19,433.86 | \$9,726.70 | 87.6% |

M1’s point uplift is positive in each slice. Only the 2024 filtered book has positive absolute P&L. 2025 and partial-2026 filtered books still lose money. M2’s point uplift is near zero in 2024 and 2025 and larger in partial 2026; that slice is not a new test.

---

## Comparison with published development findings

Copied from the reviewed cost-diagnosis evidence. Periods are not pooled. Development inference used family size 4 and is not a D2 p-value.

| | Development 2020–2023 | Evaluation 2024–2026-07-10 |
|---|---:|---:|
| Baseline P&L | \$6,628.20 | \$-39,244.98 |
| M1 filtered P&L | \$20,823.08 | \$-8,150.05 |
| M1 incremental P&L | \$14,194.88 | \$31,094.93 |
| M1 mean weekly uplift | 0.68 pp (adj. p 1.00, inconclusive) | 2.36 pp (`relative_benefit`) |
| M2 filtered P&L | \$25,088.18 | \$-28,194.16 |
| M2 incremental P&L | \$18,459.98 | \$11,050.82 |
| M2 mean weekly uplift | 0.88 pp (adj. p 0.91, inconclusive) | 0.84 pp (`inconclusive`) |

The later baseline is not a continuation of the development book’s modest positive dollar result. Do not read the two periods as one significance test.

---

## Tests and provenance

Focused tests, then existing D0 / D1 / within-date / cost-diagnosis tests, before the official run:

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint008_d2_fixed_exclusion_validation.py tests/unit/test_sprint008_d1_cost_diagnosis.py tests/unit/test_sprint008_d0_input_readiness.py tests/unit/test_sprint008_d1_measurement_validation.py tests/unit/test_sprint008_d1_within_date_followup.py -q
```

**67 passed.**

Official command:

```text
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d2_fixed_exclusion_validation.py
```

Wall time about 54 seconds. Stage times: `build_panel` 46.15s, `calendar` 0.17s, `filter_eval` 0.01s, `d0_checks` 0.15s, `economics` 0.03s, `M1` 0.89s, `M2` 0.86s, `inference` 0.00s.

| Provenance | Value |
|---|---|
| HEAD | `c2ba9726a36d92407961648563ae9d5e325b461f` |
| Working tree | dirty (implementation not yet committed at execution) |
| Diff SHA-256 | `830e8ba867adeea74df91b31b87355f814e65cce6b5a4c89c4ff453b1673dc9f` |
| Source run | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Python / numpy / pandas / scipy | 3.13.7 / 2.4.1 / 3.0.0 / 1.17.1 |

---

## Deviations and limitations

No research-rule changes after seeing results. No threshold, measurement, sizing, signal, or structure change.

- Executing tree was dirty. Receipt records source hashes and the diff hash. Historical evidence directories were not mutated.
- The evaluation window contained no verified zero-long date. The calendar still has every official entry date. Absence of a long-trade row was not treated as \(N=0\).
- Fees remain 0. Full-cross quotes are not claimed attainable fills.
- This is retrospective validation, not independent confirmation.

D3 is not started.
