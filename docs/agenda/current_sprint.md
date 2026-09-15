# Current sprint — 009

**Updated:** 2026-09-14

**Status:** `CLOSED — DIAGNOSTIC SCOPE COMPLETED`

**Mode:** **Audit.** Sprint 009 is closed. Do not start Sprint 010 planning or implementation from this agenda. Do not execute superseded D3/D4.

**Closeout:** [`docs/sprint_memos/009_closeout.md`](../sprint_memos/009_closeout.md)  
**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md) — closed with the 2026-09-14 scope amendment; do not duplicate under `docs/tmp/`.  
**D0 design:** [`docs/tmp/sprint009_d0_design.md`](../tmp/sprint009_d0_design.md) — **ACCEPTED** at `5329726`.  
**D0 evidence:** [`docs/tmp/sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md) — **ACCEPTED** through `82e3b46`. Run `READY` at `004ba80`. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z`. Supersedes `546d3e6` / `sprint009_d0_20260913T212939Z`.  
**D1 design:** [`docs/tmp/sprint009_d1_design.md`](../tmp/sprint009_d1_design.md) — **ACCEPTED** at `e109a9e`. Implementation `5669773`. Run verdict `READY`.  
**D1 evidence:** [`docs/tmp/sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md) — **ACCEPTED** through `28f5ea4`. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z`.  
**D2 design:** [`docs/tmp/sprint009_d2_design.md`](../tmp/sprint009_d2_design.md) — **ACCEPTED** at `a34f21e`. Implementation `52be625`. Run verdict `READY`.  
**D2 evidence:** [`docs/tmp/sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md) — **REVIEWED / ACCEPTED**. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z`. Supersedes `e7a1108` / `sprint009_d2_20260914T151216Z`.

**Previous:** Sprint 008 — [`CLOSED — D3 ACCEPTED`](../sprint_memos/008_closeout.md) through `61cbf30`. Findings unchanged: not an income-generating long filter; historical `STOP_NO_THRESHOLDS` preserved.  
**Prior diagnosis:** Sprint 007 — [`CLOSED — D3 ACCEPTED; D4 EXECUTION_CALIBRATION_REQUIRED`](../sprint_memos/007_closeout.md). Remains open and unresolved by Sprint 009.  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint.

---

## 1. Sprint intent (diagnostic scope completed)

Answered, for the frozen `42:8` short iron-fly book, on development history:

> Where does the selected short book lose its economic edge, and what protection do the existing wings provide at fixed official quantities?

D0–D2 are **accepted**. Original D3/D4 entry filtering is **SUPERSEDED — NOT EXECUTED**. Amended D5 is the diagnostic closeout. No filter was tested or frozen. Closing does not require a later-period companion. This sprint does not authorize uncovered trading, retune the signal window, search new wings, claim attainable fills, or resolve Sprint 007.

---

## 2. Central conventions (unchanged for the executed work)

| Item | Convention |
|---|---|
| Side | Official short iron-fly population. Body-only is a counterfactual on those names, not a new selection |
| Selection | Frozen `42:8` / CVG / liquidity / structure / name cap / weekly hold-to-expiry |
| Wings | Current `0.15` below-nearest rule. No strike search |
| Reference quantities | Official cross book, fixed. Midpoint at those quantities is a diagnostic, not the official midpoint run |
| Calendar | Whole-book `date_status` is not short-book status. Verified zero-short dates stay at zero. Missing short rows are a blocker |
| Primary measure | Paired dollar P&L on expiry settlement |
| Fees | 0, matching the official book. Not deducted twice through the spread |
| Development | `2020-01-01` through `2023-12-31` |
| Later period | `2024-01-01` through `2026-07-10`. Not required for this closeout |

Full protocol and the 2026-09-14 amendment: [`sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md).

---

## 3. Deliverables

| ID | Question | Status |
|---|---|---|
| **D0** | Can the accepted artifacts support a matched body/wing dataset that reproduces every selected short iron fly and correctly accounts for every trading date? | **ACCEPTED** through `82e3b46`. Implementation `004ba80`. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z`. `546d3e6` superseded |
| **D1** | Where does the short book lose economic margin? | **ACCEPTED** through `28f5ea4` ([`sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md)). Design accepted at `e109a9e`. Implementation `5669773`. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z` |
| **D2** | What protection do the wings provide? | **REVIEWED / ACCEPTED** ([`sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md)). Design accepted at `a34f21e`. Implementation `52be625`. Run verdict `READY`. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z`. Supersedes `e7a1108` / `sprint009_d2_20260914T151216Z` |
| **D3** | Can entry measurements identify unattractive trades? | **SUPERSEDED — NOT EXECUTED**. No filter tested or frozen. Not `STOP_NO_RULE` |
| **D4** | Does the frozen rule improve later-period economics? | **SUPERSEDED — NOT EXECUTED**. No later-period companion required for closeout |
| **D5** | What does the evidence justify? | **COMPLETED** as diagnostic closeout ([`009_closeout.md`](../sprint_memos/009_closeout.md)) |

---

## 4. Definition of done (amended closeout)

Sprint 009 is **complete** when:

- [x] D0 records `READY` or a named blocker, including the short-book calendar classification.
- [x] D1 reconciles the five-term identity on development history, or stops on that identity.
- [x] D2 separates gross protection payout from net contribution and does not claim unmeasured path risks were measured.
- [x] Original D3/D4 are recorded **SUPERSEDED — NOT EXECUTED** under the 2026-09-14 amendment (no requirement to freeze a filter or run a later-period companion).
- [x] Amended D5 states historical results and one next investigation only. It does not authorize uncovered trading, production use, or a Sprint 007 resolution.
- [x] Sprint 006/007/008 accepted results remain unreinterpreted.
- [x] Hypothetical fills are not claimed attainable.
- [x] Focused tests passed for D0–D2 financial calculation code.
- [x] Official evidence directories and the frozen contract are unchanged.

---

## 5. Explicitly out of scope

- Signal-window or universe expansion.
- Long-side changes, including promotion of Sprint 008 M1.
- Alternative wing-strike or delta searches.
- Sizing or leverage optimization (handed off as a next research theme, not executed here).
- Brokerage margin, live execution, paper plumbing, or the Sprint 007 observer.
- Intraday hedging or a new exit policy.
- Iron condor and KB-001, unless a later amendment says otherwise.
- Editing the frozen Sprint 006 contract or mutating official evidence directories.
- Sprint 010 planning or implementation.

---

## 6. Authorization and next action

**Plan status:** Sprint 009 **CLOSED — DIAGNOSTIC SCOPE COMPLETED**. D0–D2 accepted. D3/D4 superseded, not executed. Amended D5 completed in [`009_closeout.md`](../sprint_memos/009_closeout.md).

**Next action:** none inside Sprint 009. Research handoff only: candidate recovery and research sizing under explicit portfolio stress limits. Do not design Sprint 010 here. Sprint 007 `EXECUTION_CALIBRATION_REQUIRED` remains open.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-14 | Sprint 009 **CLOSED — DIAGNOSTIC SCOPE COMPLETED**. D2 evidence **REVIEWED / ACCEPTED**. Original D3/D4 **SUPERSEDED — NOT EXECUTED**. Amended D5 closeout [`009_closeout.md`](../sprint_memos/009_closeout.md). No Sprint 010 planning. |
| 2026-09-14 | D2 **corrected** at `52be625`. Run verdict `READY`. Evidence awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z`. Supersedes `e7a1108` / `sprint009_d2_20260914T151216Z`. Headline dollars unchanged. D3–D5 not started. |
| 2026-09-14 | D2 **executed** at `e7a1108`. Run verdict `READY`. Evidence [`sprint009_d2_evidence_review.md`](../tmp/sprint009_d2_evidence_review.md) awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d2_20260914T151216Z`. D3–D5 not started. |
| 2026-09-14 | D2 design at `a34f21e` **accepted**, with the midpoint-attainability wording correction. Mode set to Build, restricted to D2. Implementation **IN PROGRESS**. Official comparison evidence not yet attached. D3–D5 not started. |
| 2026-09-14 | D1 evidence **accepted** through `28f5ea4`. D2 design **drafted** ([`sprint009_d2_design.md`](../tmp/sprint009_d2_design.md)). `DESIGN DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. D3–D5 not started. |
| 2026-09-14 | D1 correction **executed** at `5669773`. Run verdict `READY`. Evidence awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z`. Supersedes `0d63293` / `sprint009_d1_20260914T001504Z`. Headline dollars unchanged. D2–D5 not started. |
| 2026-09-14 | D1 correction of `0d63293` **in progress**. Date-ratio truthiness, boolean pairing, exact unit quantities, and official D0 receipt checks. Prior run `sprint009_d1_20260914T001504Z` is not the corrected record. D2–D5 not started. |
| 2026-09-14 | D1 **executed** at `0d63293`. Run verdict `READY`. Evidence [`sprint009_d1_evidence_review.md`](../tmp/sprint009_d1_evidence_review.md) awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z`. D2–D5 not started. |
| 2026-09-13 | D1 design at `e109a9e` **accepted**. Mode set to Build, restricted to D1. Implementation **IN PROGRESS**. Official decomposition evidence not yet attached. D0 remains accepted. D2–D5 not started. |
| 2026-09-13 | D1 design **corrected** from `5d33055`. Stored ORATS mid is not the fill-model midpoint; that difference is a diagnostic, not a gate. Still `DRAFT — AWAITING REVIEW`. Implementation not started. |
| 2026-09-13 | D1 design **drafted** ([`sprint009_d1_design.md`](../tmp/sprint009_d1_design.md)). `DRAFT — AWAITING REVIEW`. Implementation not started. D0 evidence **accepted** through `82e3b46`. |
| 2026-09-13 | D0 validation **corrected** at `004ba80` and rerun. Verdict `READY`. Official accounting unchanged. Evidence awaiting review at that commit. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z`. Supersedes `546d3e6` / `sprint009_d0_20260913T212939Z`. D1 not started. |
| 2026-09-13 | D0 official readiness **executed** at `546d3e6`. Verdict `READY`. Evidence [`sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md) awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z`. D1 not started. |
| 2026-09-13 | D0 design at `5329726` **accepted**. Mode set to Build for the approved helper, runner, tests, and documentation. Implementation **in progress**. Official readiness evidence not yet attached. D1 not started. |
| 2026-09-13 | D0 design **corrected** from `b813f5e`. Direct mid/cross pairing and saved-row contract added. Design still `DRAFT — AWAITING REVIEW`. Implementation not started. |
| 2026-09-13 | Sprint scope **accepted for D0 planning**. D0 design drafted ([`sprint009_d0_design.md`](../tmp/sprint009_d0_design.md)), `DRAFT — AWAITING REVIEW`. Implementation not started. |
| 2026-09-13 | Sprint 009 draft **corrected** from `5a14348`. Tie-break is dollar uplift, not exposure-scaled bounds. Still `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. |
| 2026-09-12 | Sprint 009 draft **revised** from `cbd3f23`. M1 filters the body-only book; M2 filters the iron fly. Still `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. |
| 2026-09-12 | Sprint 009 plan **drafted** for review. Status `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. No D0, no new economic run. |
| 2026-09-12 | Sprint 008 D3 closeout **accepted** through `61cbf30`. Findings unchanged. See [`008_closeout.md`](../sprint_memos/008_closeout.md). |
