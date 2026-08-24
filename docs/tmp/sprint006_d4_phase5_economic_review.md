# Sprint 006 D4 Phase 5 — Economic review checkpoint

**Status:** `PROPOSED — AWAITING REVIEW`  
**Phase 4 evidence verdict:** `ACCEPTED` (through independent source audit `326e13d`)  
**Review basis:** frozen `decision_report.json` from the official run only; `decision_report.md` spot-checked for agreement  
**Baseline rerun:** none  
**Smoke economics:** not cited

---

## 1. Official run identity

| Field | Value |
|-------|-------|
| `RUN_DIR` | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| `experiment_id` / `contract_id` | `sprint006_baseline_v1` |
| `repo_sha` | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| `result_complete` | `true` |
| `has_unresolved_failures` | `false` |
| Primary fill | `sprint006_baseline_v1_cross` |
| Diagnostic fill | `sprint006_baseline_v1_mid` |
| Full-history window | `2018-10-26` … `2026-07-10` |
| Primary window | `2020-01-01` … `2026-07-10` |

Report identity matches the accepted Phase 3/4 evidence memo. JSON and Markdown agree on all headline fields checked below.

---

## 2. Primary cross results (economic result)

### 2.1 Headline metrics — cross (primary economic view)

| Window | View | mean cycle CAR *(conditional on traded dates)* | compounded | annualized return | Sharpe | drawdown | n traded |
|--------|------|-----------------------------------------------|------------|-------------------|--------|----------|----------|
| primary | A conditional | −0.0271 | n/a | n/a | −1.303 | −1.000 | 341 |
| primary | B calendar | n/a | −1.000 | −0.867 | −1.303 | −1.000 | complete |
| full_history | A conditional | −0.0356 | n/a | n/a | −1.584 | −1.000 | 403 |
| full_history | B calendar | n/a | −1.000 | −0.929 | −1.584 | −1.000 | complete |

View A numbers are **conditional on traded dates** only. View B treats every expected date as a calendar week (`valid_no_trade` = 0 here).

### 2.2 Yearly cross results — primary window

| Year | n expected | n traded | View B compounded | View B annualized | View A Sharpe *(conditional)* | View B drawdown |
|------|------------|----------|-------------------|-------------------|-------------------------------|-----------------|
| 2020 | 53 | 53 | −0.314 | −0.310 | 0.501 | −0.843 |
| 2021 | 52 | 52 | −0.975 | −0.975 | −2.854 | −0.973 |
| 2022 | 52 | 52 | −0.547 | −0.547 | −0.495 | −0.704 |
| 2023 | 52 | 52 | −0.665 | −0.665 | −0.856 | −0.527 |
| 2024 | 52 | 52 | −0.801 | −0.801 | −1.062 | −0.892 |
| 2025 | 52 | 52 | −0.942 | −0.942 | −2.751 | −0.941 |
| 2026 | 28 | 28 | −0.940 | −0.995 | −5.041 | −0.937 |

Every primary-window year is negative on View B compounded return. Only 2020 shows a positive View A mean cycle CAR (+0.0157) and Sharpe (+0.501); its calendar-aligned compounded return is still negative (−0.314).

### 2.3 Long / short attribution — cross, primary window

| Side | n traded rows | mean cycle return *(conditional)* | pnl_total | capital_at_risk_dollars |
|------|---------------|-----------------------------------|-----------|---------------------------|
| long | 5890 | −0.00964 | −16,993 | 2,701,558 |
| short | 3322 | −0.04302 | −146,280 | 3,400,000 |

Both sides lose under cross fills. Short iron-fly losses dominate total P&L magnitude.

### 2.4 Top-five concentration — primary cross

| Rank | Ticker | abs_pnl | share |
|------|--------|---------|-------|
| 1 | GME | 61,003 | 0.0187 |
| 2 | DASH | 40,407 | 0.0124 |
| 3 | ZM | 34,291 | 0.0105 |
| 4 | AVGO | 34,070 | 0.0105 |
| 5 | VZ | 33,048 | 0.0102 |

**top5_share_sum:** 0.0623 (6.2% of total |PnL|). Concentration is modest; the aggregate loss is not driven by a single name.

### 2.5 Activity, turnover, and weekly diagnostics — cross, primary window

| Metric | Value |
|--------|-------|
| avg included names / traded date | 27.01 |
| avg long / traded date | 17.27 |
| avg short / traded date | 9.74 |
| turnover complete | true |
| win_rate | 0.370 |
| profit_factor | 0.632 |
| no_trade_frequency | 0.0 |

### 2.6 Funnel coverage and structure failures — cross

| Window | n expected | joint_coverage_rate | mean jointly eligible | sum included | structure failures (metadata / missing body / wing-liq / other) |
|--------|------------|---------------------|----------------------|--------------|------------------------------------------------------------------|
| primary | 341 | 1.0 | 332.81 | 9212 | 239 / 306 / 2096 / 0 |
| full_history | 403 | 1.0 | 326.18 | 10486 | 283 / 362 / 2622 / 0 |

Every expected date traded (`n_valid_no_trade=0`, `n_failed=0`). Structure failures are frequent at the candidate level but do not suppress trading on any calendar date.

---

## 3. Diagnostic mid results (fill-assumption sensitivity only)

Mid is **not** the economic result. It bounds how much outcomes move when fills change from mid to cross.

### 3.1 Headline metrics — mid (diagnostic)

| Window | View | mean cycle CAR *(conditional on traded dates)* | compounded | annualized return | Sharpe | drawdown |
|--------|------|-----------------------------------------------|------------|-------------------|--------|----------|
| primary | A conditional | +0.0240 | n/a | n/a | +1.046 | −0.847 |
| primary | B calendar | n/a | +48.855 | +0.815 | +1.046 | −0.847 |
| full_history | A conditional | +0.0157 | n/a | n/a | +0.649 | −0.974 |
| full_history | B calendar | n/a | +0.269 | +0.0312 | +0.649 | −0.974 |

### 3.2 Fill-assumption sensitivity

| Window | n dates both traded | n dates mid-only | n dates cross-only | n candidates both included | n candidates mid-only | n candidates cross-only | mean cross−mid CAR | mean cross−mid PnL | mean spread_cost_ratio cross | mean spread_cost_ratio mid |
|--------|---------------------|------------------|--------------------|----------------------------|-----------------------|-------------------------|--------------------|--------------------|------------------------------|----------------------------|
| primary | 341 | 0 | 0 | 9212 | 0 | 0 | −0.0510 | −945.9 | 0.0530 | 0.0 |
| full_history | 403 | 0 | 0 | 10486 | 0 | 0 | −0.0513 | −946.9 | 0.0530 | 0.0 |

No unmatched traded dates and no unmatched included candidates between fills. The cross-minus-mid gap is therefore a pure within-set fill comparison, not a selection artifact. Cross-minus-mid CAR is about −5.1 percentage points per traded date on average; mid shows positive economics while cross is deeply negative.

---

## 4. Plain-language interpretation

**Return and risk (primary cross).** Under conservative cross fills—the designated primary economic view—the frozen `42:8` baseline loses money on both reporting windows. View B compounded return is essentially total loss (−1.000 primary; −1.000 full history). View A mean cycle CAR is negative (−2.7% primary; −3.6% full history). Sharpe is negative (−1.30 primary; −1.58 full history) with drawdown near −100%. View A and View B agree on direction: this is not a View-definition disagreement.

**Yearly stability.** Losses are persistent. After the primary window begins, every calendar year is negative on View B compounded return. 2020 is the least bad year but still negative on a calendar basis; later years compound large losses (2021, 2025, 2026 especially severe).

**Side dependence.** Both long straddles and short iron flies lose under cross fills. Short-side mean cycle return (−4.3%) and total P&L (−146k) are worse than long-side (−1.0%; −17k).

**Fill sensitivity.** Mid-fill diagnostics are materially positive while cross is strongly negative, with no unmatched dates or candidates. The strategy’s sign flips with fill assumption. Mean cross-minus-mid CAR ≈ −5.1% per traded date indicates that bid/ask conservatism—not a small spread tweak—dominates the economic conclusion.

**Concentration.** Top-five |PnL| share is 6.2%; losses are broad-based across names, not concentrated in one ticker.

**Activity and coverage.** The pipeline trades every expected date (403/403 full history; 341/341 primary). Mean ~27 names per date. Win rate 37% and profit factor 0.63 confirm weak per-date economics. Joint feature coverage is complete (`joint_coverage_rate=1.0`). Candidate-level structure failures are common but never block a date from trading.

---

## 5. Limitations

### 5.1 Report limitations (`limitations[]`, verbatim)

- Hold-to-expiry; positions are not managed intra-week.
- No earnings filter.
- Iron-fly wings use below-nearest 0.15-delta selection.
- Tier A sizing is not integer lots.
- Long-only fallback dates are possible.
- Mid is a fill-assumption diagnostic, not a pure transaction-cost attribution.
- `robust_score` is not a decision metric and is not used for go/no-go.
- Post-signal candidate means the name already passed the Momentum-tail and within-side CVG filters; these artifacts cannot support full-universe Momentum IC or CVG increment tests.

### 5.2 Accepted Phase 4 limitations (additional)

- Phase 3 shell `EXIT_CODE`, stdout, and stderr were not retained; completion is evidenced by artifacts and receipt only (accepted as non-blocking).
- Input artifact digests were recorded pre/post run but are not pinned in code (PG-2).
- Receipt still reports `deliverable=sprint006_d3` with D4 deferred in producer metadata (expected; not a defect).
- S3 source audit is N/A because `n_valid_no_trade_dates=0`.
- No pinned input-digest identity enforcement in the adapter.
- Iron-condor alternative untested while KB-001 remains open.
- Tier A fractional sizing; no earnings PIT artifact.

---

## 6. Proposed conclusions *(pending human review)*

### 6.1 Proposed economic characterization

**`WEAK/NEGATIVE`**

**Direct reasons (primary cross, frozen report fields):**

1. View B compounded return ≈ −1.000 on both primary and full-history windows (`by_fill.cross.primary.view_b_calendar.compounded`, `…full_history…`).
2. View A mean cycle CAR negative on both windows (−0.027 primary; −0.036 full history), Sharpe negative (−1.30; −1.58).
3. Every primary-window year negative on View B compounded return; no sustained profitable sub-period.
4. Both long and short sides negative on mean cycle return and total P&L; short side dominates losses.
5. Win rate 0.370 and profit factor 0.632 on primary cross weekly diagnostics.
6. Fill sensitivity is extreme: mid diagnostics positive while cross is deeply negative, with mean cross−mid CAR ≈ −0.051 per traded date and zero unmatched dates/candidates.

No profitability cutoff was invented. This characterization describes **only** the frozen `42:8` configuration on the accepted snapshot/derived dataset under hold-to-expiry, Tier A equal-max-loss sizing, and cross-fill conservatism. It is not a general statement about momentum or CVG.

### 6.2 Proposed recommendation

**Reject or defer the economic hypothesis.**

**Direct reasons:**

1. Evidence is trusted (`ACCEPTED`); the negative cross result is therefore interpretable, not blocked by a technical defect.
2. Primary cross economics are uniformly weak/negative across views, years, and both sides—consistent with deferring further capital or Sprint 007 robustness spend on this exact baseline.
3. Mid-positive / cross-negative split shows the hypothesis does not survive conservative fills; retuning after this exposure would violate D0 §6.
4. No named correctness or data defect is indicated by the report given Phase 4 acceptance.

**What would need to change to revisit (not designed here):** a new preregistered experiment contract—not parameter edits to this frozen baseline.

If Sprint 007 were authorized later, bounded questions it **may** consider (not designed here):

- Whether a different preregistered feature window or structure passes the same trust/evidence bar.
- Whether integer-lot Tier B sizing materially changes conclusions under the same fills.
- Whether pinned input-digest identity or earnings filtering changes eligibility enough to alter economics.
- Iron-condor comparison once KB-001 is resolved.
- Walk-forward or sub-period robustness on a **new** frozen contract, not this one.

---

## 7. Scope statement

This checkpoint opens and interprets the official frozen report only. It does not close Sprint 006, does not start Sprint 007, does not rerun the baseline, and does not modify production code, tests, configuration, or run artifacts. Characterization and recommendation remain **proposed** until reviewed and accepted.
