# Current sprint — 007

**Updated:** 2026-08-29

**Status:** `ACTIVE — D0 IMPLEMENTED — AWAITING REVIEW`

**Mode:** **Audit** — D0 readiness evidence produced; D1–D4 not authorized.

**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](sprint7_shortfall_plan.md) — canonical path; do not duplicate under `docs/tmp/`.

**D0 design:** [`docs/tmp/sprint007_d0_design.md`](../tmp/sprint007_d0_design.md) — `PROPOSED — AWAITING ACCEPTANCE`

**D0 evidence:** `C:/MomentumCVG_env/runs/sprint007_d0_20260829T233453Z/` (outside repo)
**Previous:** Sprint 006 — [`CLOSED — EVIDENCE ACCEPTED; FROZEN 42:8 ECONOMICS WEAK/NEGATIVE`](../sprint_memos/006_closeout.md)  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable evidence, not a Sprint 007 starting configuration to edit.

---

## 1. Sprint intent

Diagnose the implementation shortfall exposed by Sprint 006 before conducting any signal-window search or strategy retuning.

Sprint 006 established a trusted result for the frozen `42:8` Momentum+CVG baseline:

- midpoint economics were materially positive;
- conservative full-cross economics were deeply negative;
- the same 9,212 primary-window date/ticker/direction keys were included under both fills;
- fill changes affected both trade economics and portfolio quantities;
- short iron-fly losses dominated the primary cross result;
- the official twin-fill full-history run required approximately three hours.

The Sprint 006 result remains accepted and unchanged. Sprint 007 is not an attempt to rescue it. Sprint 007 asks whether the loss between apparent gross economics and implementable economics is selectively avoidable, structurally reducible, fundamental to the current implementation, or unresolved without execution evidence.

---

## 2. Central question

> For the frozen `42:8` selected option book, where does the midpoint-to-cross implementation shortfall come from, how much implementation cost can the apparent gross margin tolerate, and what single next action is justified by the evidence?

The analysis follows this economic chain:

> **signal → trade selection → instrument/payoff expression → execution → realized economics**

Sprint 007 may diagnose the frozen selected trade expression. It must not claim that post-signal artifacts alone identify the intrinsic quality of Momentum or CVG as signal families.

---

## 3. Sprint outcome

By sprint end, the project must have a reproducible and evidence-bounded answer to four questions:

1. **Gross-expression margin:** Does the frozen selected option book possess broad enough midpoint economic margin to justify additional implementation work, and on which side?
2. **Shortfall mechanism:** Which economically distinct mechanisms reconcile the midpoint-to-cross gap?
3. **Required execution:** What effective execution quality must the current expression achieve, and how much headroom remains for unmodeled frictions?
4. **Decision:** Does the evidence justify one preregistered redesign, an execution-shadow experiment, stopping the current implementation, or resolving a specific evidence gap first?

Success is a trustworthy diagnosis and justified next decision. A positive backtest or identified cure is not required.

---

## 4. Starting point and evidence authority

### Accepted evidence

| Item | Authority |
|---|---|
| Sprint 006 closeout | [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md) |
| Official run | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Execution commit | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Repository closeout baseline | `9535c3a` |
| Frozen contract | [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) |
| Primary reporting period | `2020-01-01` → `2026-07-10` |
| Full-history period | `2018-10-26` → `2026-07-10` |

The official run contains paired mid/cross trade, leg, candidate, funnel, date-status, date-summary, run-summary, and decision-report artifacts. These are the default Sprint 007 inputs.

### Interpretation boundary

- Midpoint is an optimistic gross-expression reference.
- Full cross is the accepted Sprint 006 conservative execution case.
- Neither is automatically expected real execution.
- The observed mid/cross difference is fill-assumption sensitivity, not a pure transaction-cost number, because fill prices also change sizing and capital at risk.
- Historical quote artifacts can estimate required execution quality; they cannot establish package-order fill probability or actual attainable execution.

---

## 5. Architecture and efficiency principle

Sprint 007 is **artifact-first**.

Use the accepted Sprint 006 outputs for D0–D3 wherever they contain the required evidence. Do not rerun `SurfaceRunner`, invoke `scripts/run_surface_search.py`, or optimize the full engine merely to reproduce information already present in the official artifacts.

Any enabling implementation must be a narrow, read-only post-pass over accepted artifacts and must reuse existing report/reconciliation logic where practical. It must not become:

- a second economic engine;
- a generic research framework;
- a new storage contract;
- a strategy-search system;
- or a broad performance-refactoring project.

Full-engine efficiency work is deferred until an accepted next experiment actually requires new structures, selections, or full-history reruns.

---

## 6. In scope

- Verify that accepted Sprint 006 artifacts can support a trusted, fast investigation.
- Evaluate midpoint economics of the **current selected trade expression**, including side, breadth, stability, and concentration only as needed to answer the sprint questions.
- Reconcile the mid-to-cross gap into economically distinct mechanisms, including direct quote concession and fill-dependent sizing/capital feedback.
- Attribute the dominant mechanism further only when doing so discriminates among competing explanations.
- Determine the execution quality required by the current expression while clearly separating requirement from attainability.
- Classify the evidence and specify exactly one justified next action.
- Add only the minimum deterministic analysis code and focused tests required for trusted evidence.

---

## 7. Explicitly out of scope

- Changing or replacing the accepted Sprint 006 result.
- Retuning `42:8`, searching the 281 feature windows, or broad hyperparameter search.
- Selecting a spread/liquidity threshold by historical P&L, Sharpe, or best retained subset.
- Testing multiple filters, sides, structures, maturities, holding frequencies, or execution policies and selecting a winner.
- Treating same-sample exploratory diagnostics as validation of a cure.
- Running an alternative structure solely because the frozen result was disappointing.
- Full-universe Momentum IC, CVG incremental-value testing, or new feature engineering.
- Broker integration, order placement, live trading, or execution-shadow implementation.
- Broad `SurfaceRunner` optimization, a new engine, or unrelated refactoring.
- Iron-condor evaluation while KB-001 remains open unless a later separately accepted experiment explicitly resolves that dependency.

---

## 8. High-level deliverables

Detailed methods are designed immediately before each deliverable and accepted before its new granular outputs are generated. The sprint-level questions and inference boundaries below are fixed.

### D0 — Investigation readiness and evidence contract

**Question:** Can the accepted Sprint 006 artifacts answer the diagnostic questions reproducibly and efficiently without a full economic rerun?

D0 establishes artifact identity, schema sufficiency, reconciliation prerequisites (including primary unit = dollar P&L), the minimal analysis path, and the rules that separate descriptive evidence from new hypotheses.

### D1 — Gross economics of the frozen trade expression

**Question:** At midpoint, where—if anywhere—does the frozen selected option book contain broad enough gross economic margin to justify further implementation work?

D1 evaluates the current selection plus current payoff expression against the sprint-level continue/stop gate. It must not label its result as pure Momentum/CVG signal quality.

### D2 — Implementation-shortfall mechanism

**Question:** What mechanisms account for the entire midpoint-to-cross gap?

D2 first performs a complete dollar-P&L reconciliation with a fixed-position reference. It then examines only the dominant terms needed to distinguish selective tradability, side/structure concentration, sizing feedback, or diffuse unavoidable friction, in that discrimination order.

### D3 — Required execution envelope

**Question:** What effective execution quality must the current expression achieve to preserve economically meaningful margin?

D3 estimates the requirement and remaining headroom. It states requirement and unknown attainability only; it does not claim that the required package execution is attainable from historical end-of-day quote data.

### D4 — Diagnosis, next hypothesis, and closeout

**Question:** Which explanation best fits D1–D3, and what single next action is justified?

D4 synthesizes existing Sprint 007 evidence. It does not run a collection of cure backtests. It selects one of the following outcomes:

- `SELECTIVE_FRICTION_HYPOTHESIS`
- `STRUCTURE_OR_SIZING_HYPOTHESIS`
- `EXECUTION_CALIBRATION_REQUIRED`
- `CURRENT_IMPLEMENTATION_NOT_VIABLE`
- `EVIDENCE_INCONCLUSIVE`

---

## 9. Evidence and inference boundaries

1. **Frozen evidence remains frozen.** No Sprint 007 result silently replaces or reinterprets the accepted Sprint 006 cross result.
2. **Current-expression economics are not pure signal economics.** The accepted artifacts already embed selection, structure, holding period, and sizing.
3. **Primary reconciliation unit is dollar P&L.** The mid→cross bridge reconciles aggregate `pnl_total` on the matched included key set. Capital-at-risk / quantity change is a required companion. View A mean cycle CAR is a secondary portfolio view, not the attribution residual target.
4. **D1 continue/stop is gated before metrics are opened.** Midpoint margin is “economically meaningful” only if predeclared sign, breadth, location, and stability criteria all hold (see working plan). Exact concentration/year formulas are frozen in the D1 design; the four-part gate is not reinvented after output.
5. **Competing explanations are discriminated in order.** Reconcile the dollar bridge first; attribute by side/structure role next; test ex-ante package tradability only after that. Structure/side concentration alone is not selective friction.
6. **Required execution is not attainable execution.** Historical quotes can establish a break-even requirement; shadow orders are needed to estimate fill opportunities and no-fill behavior. Language claiming recoverability, likely package fills, or ORATS-implied attainability is forbidden without shadow evidence. Break-even strictly between mid and full cross maps to `EXECUTION_CALIBRATION_REQUIRED` when D1 continues.
7. **Attribution is not a counterfactual cure.** Wing cost does not prove that a wingless payoff would be superior; short-side damage does not validate a long-only strategy.
8. **Same-sample diagnostics generate hypotheses.** Any cost-aware filter or expression change motivated by Sprint 007 requires a separately preregistered evaluation.
9. **No threshold winner.** Descriptive cost relationships may be examined only under rules frozen before output; Sprint 007 must not select the historically best cutoff.

---

## 10. Definition of done

Sprint 007 is complete when:

- [ ] The official Sprint 006 artifacts remain unchanged and are identity-checked.
- [ ] D0 confirms a trusted artifact-first path or records a specific blocker.
- [ ] D1 states whether the current selected expression has gross margin worth investigating and where that margin resides.
- [ ] D2 reconciles the observed implementation shortfall with no material unexplained residual.
- [ ] D3 states the execution quality required and what cannot be inferred about attainability.
- [ ] D4 records one evidence classification and exactly one next action.
- [ ] Every new diagnostic rule is frozen before its granular output is opened.
- [ ] Exploratory findings are clearly separated from accepted evidence.
- [ ] No signal window, spread threshold, structure, side, or execution policy is selected because it improved the same-sample backtest.
- [ ] Relevant focused tests pass and evidence is reproducible.
- [ ] Remaining limitations and stop conditions are documented.

---

## 11. Authorization sequence

Sprint 007 agenda and working plan were **accepted 2026-08-26**. That acceptance authorized D0 design only.

D0 design is **proposed** at [`docs/tmp/sprint007_d0_design.md`](../tmp/sprint007_d0_design.md) (`PROPOSED — AWAITING ACCEPTANCE`). D0 implementation is **not** authorized until the design is reviewed and accepted.

For each deliverable:

1. Inspect the current repository, accepted artifacts, and prior accepted evidence.
2. Write a short one-page design (question, frozen decisions, reuse, footprint, tests, acceptance evidence, inference boundaries, non-goals, stop rule). Do not rewrite the sprint contract between deliverables unless a stop condition fires.
3. Wait for review and acceptance.
4. Implement or execute only the accepted deliverable.
5. Present evidence and request acceptance before designing the next deliverable.

D1–D4 are not authorized. D0 implementation is not authorized until D0 design acceptance.

Pause and request rescoping if proposed work:

- changes a P&L-sensitive rule;
- requires a full-history rerun not justified by D0;
- evaluates several potential cures;
- introduces a general framework or second economic path;
- or cannot support a clear inference about the sprint questions.

---

## 12. Initial next action

D0 implementation complete — **await D0 review**. Evidence: `C:/MomentumCVG_env/runs/sprint007_d0_20260829T233453Z/`. Do not begin D1 design until D0 is accepted.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-08-29 | D0 implemented: artifact validation helper, unit tests, readiness notebook; all gates passed (`READY_WITH_NARROW_ENABLING_CHANGE`). |
| 2026-08-26 | Sprint 007 agenda and working plan **accepted**. D0 design proposed — `PROPOSED — AWAITING REVIEW`. |
| 2026-08-26 | Sprint 007 agenda and working plan written (`PROPOSED — AWAITING ACCEPTANCE`). |
