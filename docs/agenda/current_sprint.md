# Current sprint — 009

**Updated:** 2026-09-12

**Status:** `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`

**Mode:** **Audit.** Planning only. Do not begin D0, run new economic analyses, or treat this draft as accepted.

**Working plan:** [`docs/agenda/sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md) — canonical draft; do not duplicate under `docs/tmp/`.

**Previous:** Sprint 008 — [`CLOSED — D3 ACCEPTED`](../sprint_memos/008_closeout.md) through `61cbf30`. Findings unchanged: not an income-generating long filter; historical `STOP_NO_THRESHOLDS` preserved.  
**Prior diagnosis:** Sprint 007 — [`CLOSED — D3 ACCEPTED; D4 EXECUTION_CALIBRATION_REQUIRED`](../sprint_memos/007_closeout.md). This draft does not implement that execution-observation handoff and does not cancel it.  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint.

---

## 1. Sprint intent

Answer, for the frozen `42:8` short iron-fly book:

> Where does the selected short book lose its economic edge, and can better trade selection improve it while accounting for the value of protection?

This is a separately scoped research draft. It does not rescue the frozen cross book, retune the signal window, search new wings, or claim that historical quote scenarios are attainable fills.

Proposed method order:

1. Trust a matched body/wing dataset against the official cross book (D0).
2. Attribute development-history dollars into body economics, execution concession, protection price, wing spread, and wing payout (D1).
3. Compare the same positions with and without wings at fixed body quantities (D2).
4. Only then test two predeclared entry measurements and freeze at most one rule, or stop (D3).
5. If a rule is frozen, evaluate it once on the later period against the unfiltered book and a same-exposure cash benchmark (D4).
6. Close with what the evidence justifies, including a stop (D5).

---

## 2. Central conventions (proposed)

| Item | Convention |
|---|---|
| Side | Official short iron flies only |
| Selection | Frozen `42:8` / CVG / liquidity / structure / name cap / weekly hold-to-expiry |
| Wings | Current `0.15` below-nearest rule. No strike search |
| Reference quantities | Official cross book, fixed. Midpoint at those quantities is a diagnostic, not the official midpoint run |
| Population | Conditional on names that passed iron-fly construction, including wing availability |
| Primary measure | Paired dollar P&L. Ratios need an explicit denominator. Do not put an uncovered body on the iron fly’s max-loss denominator |
| Fees | 0, matching the official book. Not deducted twice through the spread |
| Development | `2020-01-01` through `2023-12-31` |
| Later period | `2024-01-01` through `2026-07-10`. Retrospective. Used only after a freeze or a recorded skip |

Full protocol: [`sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md).

---

## 3. Deliverables

| ID | Question | Status |
|---|---|---|
| **D0** | Can we trust the body/wing comparison? | **Not started** |
| **D1** | Where does the short book lose economic margin? | **Not started** |
| **D2** | What protection do the wings provide? | **Not started** |
| **D3** | Can entry measurements identify unattractive trades? | **Not started** |
| **D4** | Does the frozen rule improve later-period economics? | **Not started.** Skipped if D3 does not freeze a rule |
| **D5** | What does the evidence justify? | **Not started** |

No design, runner, or evidence file exists yet. Do not create empty ones before the relevant step is authorized.

---

## 4. Definition of done

Sprint 009 is not complete while this plan is a draft. After a later execution authorization, it is complete when:

- [ ] D0 records `READY` or a named blocker.
- [ ] D1 reconciles the five-term identity on development history, or stops on that identity.
- [ ] D2 separates gross protection payout from net contribution and does not claim unmeasured path risks were measured.
- [ ] D3 freezes at most one predeclared rule, or records `STOP_NO_RULE`.
- [ ] D4 reports both comparisons, or is skipped with the D3 reason. The rule is not revised on the later period.
- [ ] Dollar-profit retention is reported separately from winner-count retention.
- [ ] D5 answers the central question without assuming wing removal or a production filter.
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

**Plan status:** draft, awaiting review. Not accepted.

**Implementation:** not started. Do not begin D0.

**Next action:** review [`sprint9_short_body_wing_plan.md`](sprint9_short_body_wing_plan.md), especially the kickoff decisions in that plan’s §16. Acceptance of this draft, if given later, still does not start code.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-12 | Sprint 009 plan **drafted** for review. Status `DRAFT — AWAITING REVIEW; IMPLEMENTATION NOT STARTED`. No D0, no new economic run. |
| 2026-09-12 | Sprint 008 D3 closeout **accepted** through `61cbf30`. Findings unchanged. See [`008_closeout.md`](../sprint_memos/008_closeout.md). |
