# Sprint 009 — closeout

**Status:** `CLOSED — DIAGNOSTIC SCOPE COMPLETED`  
**Submitted:** 2026-09-14  
**Closeout authorization:** Sprint 009 scope amendment and D5 diagnostic closeout (this memo)  
**D2 reviewed / accepted through:** prior review carried forward; evidence recorded through `3a2b3e0`; implementation `52be6254ef877a791fcf476e0a82fc663804ffa5`  
**Design of record for D2:** [`docs/tmp/sprint009_d2_design.md`](../tmp/sprint009_d2_design.md) — accepted at `a34f21e`

Sprint 006 remains **accepted and unchanged**. Sprint 007 remains **closed** with `EXECUTION_CALIBRATION_REQUIRED`. Sprint 008 remains **closed** through `61cbf30`: not an income-generating long filter; historical `STOP_NO_THRESHOLDS` preserved. This memo does not replace those conclusions, retune `42:8`, authorize uncovered trading, or design Sprint 010.

This closeout uses committed evidence only. No experiment was rerun.

---

## 1. Verdict

**Central diagnostic question answered by D0–D2:** Where does the selected short book lose its economic edge, and what protection do the existing wings provide at fixed official quantities?

**Answer:** On development history, the body-only cross book is profitable at about **+$37.3k**, but the wings turn that into a combined iron-fly loss of about **−$69.8k**. Net wing contribution is about **−$107.1k**. Wings reduce some observed expiry losses and still lose money after purchase cost. Body profitability depends on 2022; excluding that year leaves about **−$2.5k**. These are fixed-quantity expiry outcomes on the iron-fly-selected population. They are not deployment readiness.

| Conclusion | Result |
|---|---|
| **D0–D2 evidence** | Accepted. D2 **REVIEWED / ACCEPTED**; implementation `52be625`; output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/` |
| **Original D3 / D4** | **SUPERSEDED — NOT EXECUTED**. No filter was tested or frozen. Not `STOP_NO_RULE` |
| **Amended D5** | This diagnostic closeout and research handoff |
| **Income-generating short book / uncovered body** | **Not established** |
| **Sprint 007 handoff** | Remains open: `EXECUTION_CALIBRATION_REQUIRED` |
| **This memo** | Closes Sprint 009 as diagnostic scope completed |

---

## 2. Scope amendment (2026-09-14)

Accepted D0–D2 evidence shows substantial negative net contribution from the existing wings on development history, while positive aggregate body-cross P&L depended on 2022. These findings motivate prioritizing sizing and protection architecture as the next research direction. The potential benefit of entry filtering remains untested because D3/D4 were not executed. Intraperiod mark-to-market, margin, liquidation, and unseen-tail risks remain unmeasured.

Under that evidence, **sizing and protection architecture take priority over entry filtering**. Original D3 and D4 remain documented as historical methodology under **SUPERSEDED — NOT EXECUTED**. They are not failed experiments. Closing does not require executing them or producing a later-period companion.

---

## 3. What the sprint did

D0 built a matched short iron-fly body/wing panel against the official cross book, including the short-book calendar and the verified zero-short date `2020-03-13`. Verdict `READY`. Accepted through `82e3b46`. Implementation `004ba80`. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/`.

D1 attributed development dollars into body midpoint profit, body concession, wing midpoint premium, wing concession, and wing expiry payout. The five-term identity reconciled. Accepted through `28f5ea4`. Implementation `5669773`. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/`.

D2 compared the same positions with and without wings at fixed official cross quantities. Gross payout, purchase cost, net contribution, and loss avoided stayed distinct. Corrected implementation `52be625`. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/`. Older D2 runs remain superseded and preserved. Evidence **REVIEWED / ACCEPTED**.

Original D3/D4 entry-filter work was not executed. Amended D5 is this closeout.

---

## 4. Findings (copied from accepted evidence)

Development population: **2,087** selected short trades across **209** dates in **2020–2023**, including `2020-03-13` at zero. Fees = 0. \(Q\) is already share-equivalent; do not ×100. Midpoint results are diagnostic at the same quantities; this analysis does not establish whether midpoint fills are attainable.

### 4.1 Body, wing, and iron-fly dollars

| Book | Body P&L | Wing contribution | Iron-fly P&L |
|---|---:|---:|---:|
| Cross, primary | 37,345.69 | −107,122.42 | −69,776.72 |
| Midpoint, diagnostic, same quantities | 120,527.85 | −68,737.46 | 51,790.39 |

Cross purchase cost is **$401,339.13** (`w_mid + h_wing`). Gross expiry payout is **$294,216.71**. Removing wings would have added **$107,122.42** of development expiry P&L at cross. That is costs saved minus payouts forgone. It is not a decision to drop wings.

### 4.2 When wings paid

| Event | Trades | Dates |
|---|---:|---:|
| Gross payout positive | 360 / 2,087 (17.2%) | 155 / 209 (74.2%) |
| Payout exceeded purchase cost | 271 / 2,087 (13.0%) | 46 / 209 (22.0%) |

### 4.3 Observed loss reduction

Loss avoided is not gross payout. Trade-level loss avoided sums to **$120,740.25**. Date-level loss avoided, after netting within each date, sums to **$64,055.26**. Those totals are not added. Date-level loss avoided on **2020-02-21** is **$36,506** of the **$64,055** date-level total, about **57%** of aggregate net date-level loss avoided.

Body-only cross ending cumulative **+$37,345.69**, max drawdown **−$59,351.85**. Iron-fly cross ending cumulative **−$69,776.72**, max drawdown **−$73,405.21**. Those drawdowns are expiry-accounting paths with an initial peak of zero. They are not intraperiod mark-to-market, margin, or liquidation paths.

### 4.4 Annual concentration and 2022 dependence

| Year | Body cross | Fly cross | Net wing |
|---|---:|---:|---:|
| 2020 | −5,881.12 | −17,202.68 | −11,321.55 |
| 2021 | −6,980.26 | −36,644.46 | −29,664.20 |
| 2022 | 39,864.75 | −601.37 | −40,466.12 |
| 2023 | 10,342.33 | −15,328.22 | −25,670.54 |

Body cross excluding 2022 is **−$2,519.05** (−5,881.12 − 6,980.26 + 10,342.33). Positive body profitability in this sample depends on 2022. The advantage from removing wings is positive every year and largest in 2022 at **$40,466**; the wing contribution itself is **−$40,466** that year.

---

## 5. Limitations

- Results are conditional on the **iron-fly-selected** population. Names that never entered the official short book stay out.
- Fees remain **0**. Concession is not deducted twice.
- Quantities are official cross quantities, already share-equivalent. Do not rescale by 100.
- Midpoint is diagnostic and does not establish attainable fills.
- Positive body-only expiry P&L is **not** deployment readiness and does not authorize uncovered trading.
- Expiry-accounting drawdown is not intraperiod equity, margin, liquidation, or unseen-tail risk.
- Gross wing payout, net wing contribution, and loss avoided remain distinct quantities.
- Sprint 008’s long-filter conclusion and Sprint 007’s `EXECUTION_CALIBRATION_REQUIRED` handoff are unchanged and unresolved here.

---

## 6. Research handoff (not Sprint 010)

**Next research question:**

> Does the short-body strategy retain an economic edge under explicit portfolio stress limits, and which feasible protection architecture improves its remaining risk at an acceptable cost?

The first questions to scope, not design here, are **candidate recovery** and **research sizing**. Do not select stress budgets, hedges, or trading rules in this closeout. Do not start Sprint 010 planning or implementation from this memo.

---

## 7. Evidence chain

| Deliverable | Status | Evidence |
|---|---|---|
| D0 | **ACCEPTED** through `82e3b46` | [`sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md); `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/` |
| D1 | **ACCEPTED** through `28f5ea4` | [`sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md); `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/` |
| D2 | **REVIEWED / ACCEPTED**; implementation `52be625` | [`sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md); `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/` |
| D3 | **SUPERSEDED — NOT EXECUTED** | Original methodology retained in [`sprint9_short_body_wing_plan.md`](../agenda/sprint9_short_body_wing_plan.md) §10 |
| D4 | **SUPERSEDED — NOT EXECUTED** | Original methodology retained in [`sprint9_short_body_wing_plan.md`](../agenda/sprint9_short_body_wing_plan.md) §11 |
| D5 | **COMPLETED** as diagnostic closeout | this memo |

Official Sprint 006 artifacts were not mutated. Code, tests, v1 pins, and the frozen baseline configuration were not changed by this closeout.
