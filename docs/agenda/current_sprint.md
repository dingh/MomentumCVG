# Current sprint — 009

**Updated:** 2026-09-13

**Status:** `D0 EXECUTED — EVIDENCE AWAITING REVIEW; D1 NOT STARTED`

**Mode:** **Build**, restricted to the approved D0 helper, runner, tests, and documentation. Official input artifacts remain read-only. Do not start D1. The official run returned `READY`; that evidence is awaiting review and is not an accepted D0 closeout.

**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md) — scope accepted for D0 planning; do not duplicate under `docs/tmp/`.
**D0 design:** [`docs/tmp/sprint009_d0_design.md`](../tmp/sprint009_d0_design.md) — **ACCEPTED** at `5329726`.  
**D0 evidence:** [`docs/tmp/sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md) — `AWAITING REVIEW`. Run `READY` at `546d3e6`. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z`.

**Previous:** Sprint 008 — [`CLOSED — D3 ACCEPTED`](../sprint_memos/008_closeout.md) through `61cbf30`. Findings unchanged: not an income-generating long filter; historical `STOP_NO_THRESHOLDS` preserved.  
**Prior diagnosis:** Sprint 007 — [`CLOSED — D3 ACCEPTED; D4 EXECUTION_CALIBRATION_REQUIRED`](../sprint_memos/007_closeout.md). This draft does not implement that execution-observation handoff and does not cancel it.  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint.

---

## 1. Sprint intent

Answer, for the frozen `42:8` short iron-fly book:

> Where does the selected short book lose its economic edge, and can better trade selection improve it while accounting for the value of protection?

Sprint scope is accepted. The D0 design at `5329726` is accepted. Implementation `546d3e6` has been run; the official readiness verdict is `READY` and the evidence is awaiting review. This sprint does not rescue the frozen cross book, retune the signal window, search new wings, or claim that historical quote scenarios are attainable fills. D1 has not started.

Proposed method order:

1. Trust a matched body/wing dataset against the official cross book (D0).
2. Attribute development-history dollars into body economics, execution concession, protection price, wing spread, and wing payout (D1).
3. Compare the same positions with and without wings at fixed body quantities (D2).
4. Only then test two predeclared pairs and freeze at most one, or stop (D3). M1 filters the body-only cross book. M2 filters the cross iron fly. Same population. No cross-combinations.
5. If a pair is frozen, evaluate it once on the later period against its matching unfiltered expression and its matching exposure benchmark (D4).
6. Close with historical results and one next investigation (D5). Do not authorize uncovered trading or resolve Sprint 007.

---

## 2. Central conventions (accepted for D0 planning)

| Item | Convention |
|---|---|
| Side | Official short iron-fly population. Body-only is a counterfactual on those names, not a new selection |
| Selection | Frozen `42:8` / CVG / liquidity / structure / name cap / weekly hold-to-expiry |
| Wings | Current `0.15` below-nearest rule. No strike search |
| Reference quantities | Official cross book, fixed. Midpoint at those quantities is a diagnostic, not the official midpoint run |
| Candidates | M1 score filters the body-only cross book. M2 score filters the cross iron fly. Freeze at most one pair. If both qualify, the tie-break is the larger adjusted lower bound on mean date-level dollar uplift, not an exposure-scaled ranking |
| Exposure | M2: official iron-fly capital at risk. M1: \(\sum Q S_0\), labeled notional, not margin. Filtered quantities stay unscaled |
| Calendar | Whole-book `date_status` is not short-book status. Verified zero-short dates stay at zero. Missing short rows are a blocker |
| Primary measure | Paired dollar P&L. A normalized companion is a diagnostic, not return on capital. Do not put an uncovered body on the iron fly’s max-loss denominator |
| Fees | 0, matching the official book. Not deducted twice through the spread |
| Development | `2020-01-01` through `2023-12-31` |
| Later period | `2024-01-01` through `2026-07-10`. Retrospective. Used only after a freeze or a recorded skip |

Full protocol: [`sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md).

---

## 3. Deliverables

| ID | Question | Status |
|---|---|---|
| **D0** | Can the accepted artifacts support a matched body/wing dataset that reproduces every selected short iron fly and correctly accounts for every trading date? | Executed at `546d3e6`. Official run `READY`. Evidence [awaiting review](../tmp/sprint009_d0_evidence_review.md). Not an accepted closeout. D1 not started |
| **D1** | Where does the short book lose economic margin? | **Not started** |
| **D2** | What protection do the wings provide? | **Not started** |
| **D3** | Can entry measurements identify unattractive trades? | **Not started** |
| **D4** | Does the frozen rule improve later-period economics? | **Not started.** Skipped if D3 does not freeze a rule |
| **D5** | What does the evidence justify? | **Not started** |

D0 implementation and the official readiness run are done. Evidence is awaiting review. Do not start D1. D1–D5 have no design files.

---

## 4. Definition of done

Sprint 009 is not complete while this plan is a draft. After a later execution authorization, it is complete when:

- [ ] D0 records `READY` or a named blocker, including the short-book calendar classification.
- [ ] D1 reconciles the five-term identity on development history, or stops on that identity.
- [ ] D2 separates gross protection payout from net contribution and does not claim unmeasured path risks were measured.
- [ ] D3 freezes at most one measurement/expression pair, or records `STOP_NO_RULE`. Family size stays 2.
- [ ] D4 reports both matching comparisons, or is skipped with the D3 reason. The pair is not revised on the later period.
- [ ] Dollar-profit retention is reported separately from winner-count retention.
- [ ] D5 states historical results and one next investigation only. It does not authorize uncovered trading, production use, or a Sprint 007 resolution.
- [ ] Sprint 006/007/008 accepted results remain unreinterpreted.
- [ ] Hypothetical fills are not claimed attainable.
- [ ] Focused tests pass for any new financial calculation code.
- [ ] Official evidence directories and the frozen contract are unchanged.

An inconclusive measurement or a skipped D4 is a valid completion.

---

## 5. Explicitly out of scope

- Signal-window or universe expansion.
- Long-side changes, including promotion of Sprint 008 M1.
- Alternative wing-strike or delta searches.
- Sizing or leverage optimization.
- Brokerage margin, live execution, paper plumbing, or the Sprint 007 observer.
- Intraday hedging or a new exit policy.
- Iron condor and KB-001, unless a later amendment says otherwise.
- Editing the frozen Sprint 006 contract or mutating official evidence directories.

---

## 6. Authorization and next action

**Plan status:** sprint scope accepted. D0 design accepted at `5329726`. D1–D5 are not started and are not redesigned here.

**D0 design:** [`sprint009_d0_design.md`](../tmp/sprint009_d0_design.md) — **ACCEPTED**.

**Implementation:** `546d3e6`. Official run returned `READY`. Evidence is awaiting review. Do not start D1.

**Next action:** review [`sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md). Do not start D1 from this execution.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-13 | D0 official readiness **executed** at `546d3e6`. Verdict `READY`. Evidence [`sprint009_d0_evidence_review.md`](../tmp/sprint009_d0_evidence_review.md) awaiting review. Output `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z`. D1 not started. |
| 2026-09-13 | D0 design at `5329726` **accepted**. Mode set to Build for the approved helper, runner, tests, and documentation. Implementation **in progress**. Official readiness evidence not yet attached. D1 not started. |
| 2026-09-13 | D0 design **corrected** from `b813f5e`. Direct mid/cross pairing and saved-row contract added. Design still `DRAFT — AWAITING REVIEW`. Implementation not started. |
| 2026-09-13 | Sprint scope **accepted for D0 planning**. D0 design drafted ([`sprint009_d0_design.md`](../tmp/sprint009_d0_design.md)), `DRAFT — AWAITING REVIEW`. Implementation not started. |
| 2026-09-13 | Sprint 009 draft **corrected** from `5a14348`. Tie-break is dollar uplift, not exposure-scaled bounds. Still `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. |
| 2026-09-12 | Sprint 009 draft **revised** from `cbd3f23`. M1 filters the body-only book; M2 filters the iron fly. Still `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. |
| 2026-09-12 | Sprint 009 plan **drafted** for review. Status `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. No D0, no new economic run. |
| 2026-09-12 | Sprint 008 D3 closeout **accepted** through `61cbf30`. Findings unchanged. See [`008_closeout.md`](../sprint_memos/008_closeout.md). |
