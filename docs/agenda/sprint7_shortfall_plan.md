# Sprint 007 — Implementation-shortfall diagnostic plan

**Status:** `PROPOSED — AWAITING ACCEPTANCE`  
**Updated:** 2026-08-26  
**Agenda:** [`docs/agenda/current_sprint.md`](current_sprint.md)  
**Prior evidence:** [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)  
**Canonical path:** `docs/agenda/sprint7_shortfall_plan.md` — do not duplicate under `docs/tmp/`.  
**Purpose:** Cursor-executable sprint-level plan for D0–D4. This plan defines questions, evidence boundaries, gates, and required answers. It intentionally defers each deliverable's exact metrics, charts, calculation conventions, file footprint, and tests until that deliverable is designed and accepted.

---

## 1. Human-readable summary

| Item | Sprint 007 decision |
|---|---|
| **Problem** | Sprint 006 found a very large difference between optimistic midpoint economics and conservative full-cross economics for the frozen `42:8` book. |
| **Goal** | Explain where the apparent gross margin is lost, determine how much implementation cost the current expression can tolerate, and select one justified next action. |
| **Not the goal** | Make the backtest positive, rescue `42:8`, search parameters, or validate a cure on the same sample. |
| **Frozen evidence** | Sprint 006 contract, official run, trade identities, structures, dates, fill definitions, and accepted results remain untouched. |
| **Primary inputs** | Existing paired mid/cross artifacts from `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`. |
| **Primary method** | Read-only post-pass analysis over accepted artifacts. |
| **Minimum implementation** | Reuse existing report/reconciliation code; add only narrowly scoped deterministic analysis code when required evidence cannot otherwise be produced. |
| **Efficiency decision** | Do not optimize or rerun the full engine before D0 proves it necessary. Artifact analysis should support short diagnostic iterations rather than multi-hour runs. |
| **Inference boundary** | Diagnose gross economics of the current selected expression, not intrinsic Momentum/CVG quality. Estimate required execution, not actual fill attainability. Primary bridge unit is dollar P&L; CAR is secondary. |
| **Decision architecture** | D1 continue/stop gate, discrimination order, and attainability forbid-list are frozen at sprint level (§6.4–§6.7); exact metric formulas freeze in each deliverable design. |
| **Approval boundary** | Agenda acceptance authorizes D0 design only. Every deliverable requires its own accepted one-page design before implementation or new granular output. |
| **Final output** | One diagnosis, one next action, and—when applicable—one preregistered hypothesis specification. |

---

## 2. Why two documents

`docs/agenda/current_sprint.md` is the stable sprint contract: intent, scope, deliverable questions, Definition of Done, and authorization.

This working plan explains how Cursor should reason about the deliverables and what evidence permits each conclusion. It may be refined through review before acceptance, while the agenda remains concise. Deliverable-specific designs will be created only when their deliverable begins.

---

## 3. Preserved Sprint 006 result

Sprint 006 is accepted evidence, not a draft to correct.

### Frozen experiment identity

- Contract: [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json)
- Feature window: `max_lag=42`, `min_lag=8`
- Long expression: ATM straddle
- Short expression: `0.15`-delta-wing iron fly
- Frequency/holding: weekly, approximately 7 DTE, held to expiry
- Sizing: Tier A `equal_max_loss`; short-side budget `$10,000`
- Diagnostic fill: midpoint
- Primary economic fill: full cross
- Full history: `2018-10-26` → `2026-07-10`
- Primary window: `2020-01-01` → `2026-07-10`
- Official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- Official execution commit: `e205b9acc5d0400aa38169de721acb7fb8268f29`

### Accepted motivation for Sprint 007

- Primary cross mean cycle CAR was approximately `-2.708%`.
- Mean cross-minus-mid CAR was approximately `-5.105` percentage points per traded date.
- The primary mid/cross included key sets matched: 9,212 date/ticker/direction keys and no mid-only or cross-only included candidates.
- Cross-fill short-side P&L was approximately `-$146,280`, versus approximately `-$16,993` for the long side.
- Fill changes affect quantities and capital at risk; the aggregate gap is not a pure transaction-cost subtraction.
- The official twin-fill run required approximately three hours.

These observations motivate diagnosis. They do not authorize changing any rule.

---

## 4. Economic model for the investigation

Sprint 007 reasons about distinct layers:

> **signal → trade selection → instrument/payoff expression → execution → realized economics**

The available Sprint 006 artifacts begin after much of the signal and selection process has already occurred. Therefore:

- they can show whether the **frozen selected trade expression** has midpoint economic margin;
- they can show how quoted fills and sizing transform that margin;
- they cannot independently identify full-universe signal IC or the incremental value of CVG;
- they cannot prove that an alternative payoff expression would preserve the same signal economics;
- and they cannot observe actual complex-order fill probability.

All Sprint 007 conclusions must respect these boundaries.

---

## 5. Competing explanations

The sprint should discriminate among explanations rather than execute a predetermined list of analyses.

| Explanation | Evidence that would support it | Evidence that would weaken it |
|---|---|---|
| **Current-expression gross weakness** | Midpoint margin is absent, unstable, or dependent on a small number of trades, dates, or one fragile component. | Broad, persistent midpoint margin with adequate economic headroom. |
| **Selectively avoidable friction** | Most shortfall is associated with an ex-ante-observable package-cost or tradability characteristic, while economically meaningful gross margin remains away from the expensive region. | Shortfall is diffuse across clearly tradable and expensive opportunities alike. |
| **Structure or sizing friction** | Shortfall is systematically concentrated in one current expression, side, leg role, or fill-dependent sizing mechanism after simple tradability differences are considered. | The same implementation loss appears broadly across sides and structure components. |
| **Full cross is severe relative to attainable package execution** | The current expression has gross margin and can tolerate an effective package fill meaningfully worse than midpoint, but historical data cannot determine whether that quality is attainable. | Break-even requires midpoint-or-better execution or leaves no room for commissions, missed fills, or adverse selection. |
| **Fundamental current-implementation mismatch** | Gross margin is too small relative to widespread unavoidable quoted cost, or required execution is economically implausible even for clearly tradable opportunities. | Loss is concentrated in an identifiable and economically avoidable mechanism. |

These explanations are not mutually exclusive at the trade level. D4 must identify the explanation that best controls the decision and disclose material secondary findings.

---

## 6. Cross-deliverable rules

### 6.1 What is frozen now

- The four sprint questions and their order.
- The accepted Sprint 006 evidence authority and official artifact directory.
- No modification or rerun of `configs/sprint006_baseline_v1.json`.
- No search over feature windows, thresholds, structures, or execution policies.
- Artifact-first analysis and minimal enabling implementation.
- Separation of accepted evidence, exploratory evidence, and future hypotheses.
- Final decision categories and the requirement to choose exactly one next action.
- One-deliverable-at-a-time design, authorization, execution, review, and acceptance.
- Primary reconciliation unit, D1 continue/stop gate architecture, competing-explanation discrimination order, and attainability forbid-list (§6.4–§6.7).

### 6.2 What is deliberately deferred

Before each deliverable, its short design must freeze:

- exact calculations and bridge conventions;
- exact metrics required to answer that deliverable's question;
- any segmentation or descriptive comparison rules;
- output schemas and presentation;
- minimum code/files changed;
- focused tests and tolerances;
- acceptance evidence and stop conditions.

This is disciplined progressive design, not permission to choose methods after viewing the deliverable's new output. Exact concentration ceilings, year-stability formulas, break-even algebra, and headroom buffer percentages belong in the accepted deliverable designs — not reinvented after granular output.

### 6.3 New-output boundary

Sprint 006 aggregate economics are already known. Sprint 007 cannot restore blindness to those results. It can still prevent additional researcher degrees of freedom by freezing each diagnostic method before opening its new granular result.

### 6.4 Primary accounting units

| Role | Unit | Use |
|---|---|---|
| **Primary bridge** | Aggregate `pnl_total` mid → cross on the matched included key set | D2 must reconcile this exactly within a predeclared tolerance |
| **Required companion** | Capital-at-risk / quantity change on the same keys | Isolates fill-dependent sizing feedback |
| **Secondary portfolio view** | View A mean cycle CAR (primary window) | Communicates portfolio impact; not the attribution residual target |
| **Fixed-position reference** | Reprice under a frozen quantity convention (e.g. mid quantities at cross prices and/or cross quantities at mid) | Separates pure quote concession from resize; exact convention frozen in D2 design |

D0 acceptance evidence must confirm that the official artifacts support this unit hierarchy.

### 6.5 D1 continue/stop gate (decision architecture)

Before D1 opens new granular midpoint diagnostics, the D1 design freezes exact formulas for the statistics below. The **four-part gate itself is frozen at sprint level** and must not be redefined after output.

On midpoint fills, primary reporting window, all of the following must hold to **continue** implementation diagnosis:

1. **Sign:** Aggregate mid `pnl_total` > 0 **and** View A mean cycle CAR > 0.
2. **Breadth:** Positive mid P&L is not a tiny-tail artifact (exact concentration/date-breadth statistic frozen in D1 design).
3. **Location:** At least one declared component (long book, short book, or both) has positive mid aggregate P&L with non-trivial trade count.
4. **Stability:** Margin is not a single-year artifact (exact year rule frozen in D1 design).

If any part fails, lean toward `CURRENT_IMPLEMENTATION_NOT_VIABLE` for the current expression without rejecting Momentum/CVG generally. Skip D3. D2 may still perform the minimum reconciliation needed to show why mid/cross is not decision-relevant.

### 6.6 Competing-explanation discrimination order

When attributing shortfall, follow this order and do not skip ahead:

1. Reconcile the dollar P&L bridge (direct price concession vs sizing/capital vs trade-set difference vs residual).
2. Attribute the dominant residual by **side / structure role** (long vs short; body vs wing or equivalent current-expression roles) under the frozen bridge.
3. Only then examine **ex-ante observable package tradability** (e.g. quoted width, package cost vs credit/debit) as a selective-friction candidate.

**Tie-break rules:**

- Structure/side concentration alone → prefer `STRUCTURE_OR_SIZING_HYPOTHESIS`, not selective friction.
- `SELECTIVE_FRICTION_HYPOTHESIS` requires: after conditioning on side/structure role, shortfall still concentrates in an ex-ante cost/tradability slice, **and** meaningful mid margin remains outside that slice.
- If cost and structure are collinear (e.g. short wings always wide), classify structure/sizing as primary; disclose cost collinearity as secondary; do not invent a filter winner.
- Descriptive cost relationships only: full relationship or predeclared contrast bins — no best-cutoff search.

### 6.7 Attainability forbid-list and forced mapping

Without execution-shadow evidence, Sprint 007 must not use language such as:

- “recoverable,”
- “likely fillable,”
- “patient execution would capture midpoint,”
- or “ORATS / historical quotes imply attainable package fills.”

D3 states **requirement**, **headroom**, and **unknown attainability** only. There is no `EXECUTION_RECOVERABLE` outcome.

If break-even effective execution lies strictly between midpoint and full cross **and** the D1 continue gate passed, the primary D4 outcome is `EXECUTION_CALIBRATION_REQUIRED` unless a stronger primary explanation (absent/fragile margin, structure/sizing, selective friction, or evidence gap) controls the decision. Any D3 unmodeled-friction buffer percentage is frozen in the D3 design before output; headroom ≤ 0 under that buffer supports not-viable or calibration-with-stop lean, not “probably fine.”

### 6.8 Procedural ceremony

Keep one-deliverable authorization. Between deliverables, use the one-page design template only; do not rewrite this sprint plan or the agenda unless a stop condition fires.

---

## 7. D0 — Investigation readiness and evidence contract

### Question

> Can the accepted Sprint 006 artifacts support a trusted, fast investigation of gross-expression economics, implementation-shortfall mechanisms, and required execution quality without a full economic rerun?

### Required answer

D0 must return one of:

- `READY_ARTIFACT_FIRST`
- `READY_WITH_NARROW_ENABLING_CHANGE`
- `BLOCKED_BY_SPECIFIC_EVIDENCE_GAP`

### Required behavior

D0 must:

1. Confirm the official run identity and that its accepted artifacts are unchanged.
2. Inspect actual schemas and determine which D1–D3 questions they can answer directly.
3. Confirm paired mid/cross identity and the fields needed for the §6.4 unit hierarchy (dollar P&L primary; capital/quantity companion; CAR secondary; fixed-position reference feasible).
4. Identify evidence that cannot be obtained from the artifacts, especially actual package fill opportunities and alternative-expression counterfactuals.
5. Determine the narrowest analysis path and an operationally useful runtime expectation.
6. Define how later outputs will distinguish accepted calculation, exploratory description, and future hypothesis.

### Expected reuse

Inspect and reuse, where applicable:

- `src/backtest/surface_decision_report.py`
- `src/backtest/sprint006_baseline.py`
- `src/backtest/surface_runner.py`
- official `trade_log_*`, `leg_log_*`, `candidate_view_*`, `date_summary_*`, `date_status_*`, `funnel_summary_*`, `run_summary_*`, and `decision_report.json`

Do not assume a new module, CLI, or artifact contract is required before inspection.

### Minimum implementation rule

If existing code can answer the questions safely, D0 should authorize analysis without production changes. If code is necessary, it must be a small read-only post-pass with deterministic inputs and outputs. It must not invoke or duplicate signal selection, structure construction, settlement, or the full `SurfaceRunner` path.

### D0 acceptance evidence

- Official artifact identity/schema check.
- A question-to-field sufficiency matrix for D1–D3.
- Confirmation that primary unit = dollar P&L is supportable from artifacts, with CAR as companion only.
- A reproducible dry calculation or smoke proving the proposed post-pass works without opening unplanned granular results.
- Expected file footprint and focused tests for any enabling change.
- Explicit list of unanswerable questions and why.

### D0 non-goals

- No economic interpretation beyond already accepted Sprint 006 facts.
- No full-history rerun.
- No performance optimization of `SurfaceRunner`.
- No filter, fill, structure, side, or signal experiment.
- No generic diagnostic framework.

### Stop rule

Stop after readiness is demonstrated or a specific evidence blocker is documented. D1 design is not authorized until D0 evidence is reviewed and accepted.

---

## 8. D1 — Gross economics of the frozen trade expression

### Question

> At midpoint, where—if anywhere—does the frozen selected option book contain broad enough gross economic margin to justify further implementation work?

### Required answer

D1 must state:

- whether the current expression has economically meaningful midpoint margin under the §6.5 continue/stop gate;
- whether that margin is broad or concentrated;
- which portfolio side or current component supplies or destroys it;
- and whether implementation diagnosis should continue.

### Interpretation

D1 measures the economics of:

> frozen `42:8` selection + frozen CVG rule + current long/short expression + current holding period + current midpoint sizing.

It does **not** measure pure Momentum/CVG signal quality.

### Method boundary

The D1 design should choose only the smallest set of breadth, stability, side, and concentration evidence required to answer the §6.5 gate. Exact metrics and segmentation must be frozen before producing new D1 outputs. The four-part gate architecture itself is already frozen and must not be replaced after looking.

D1 must not:

- search for the best-performing subgroup;
- introduce a spread or liquidity cutoff;
- calculate alternative-structure P&L;
- retune signal selection;
- or interpret midpoint as expected executable return.

### D1 acceptance evidence

- Reconciliation to the accepted midpoint aggregate.
- Explicit pass/fail on each §6.5 gate part, plus a direct answer about gross-expression margin and its economic location.
- Evidence that the answer is not merely a handful of names or dates, or an explicit finding that it is.
- Clear limits on what can be inferred about the underlying signal.

### Gate

If the §6.5 continue rule fails, D3 should normally be skipped. D2 may still perform the minimum reconciliation needed to establish why the mid/cross comparison is not decision-relevant. D4 should then consider `CURRENT_IMPLEMENTATION_NOT_VIABLE` without rejecting Momentum/CVG generally.

---

## 9. D2 — Implementation-shortfall mechanism

### Question

> What economically distinct mechanisms account for the entire midpoint-to-cross gap?

### Required answer

D2 must first produce a complete, auditable bridge between accepted midpoint and cross economics using **dollar P&L as the primary residual target** (§6.4). The bridge must distinguish:

- direct quoted entry-price concession on the same contracts;
- quantity and capital-at-risk changes caused by fill-dependent sizing;
- any trade-set or opportunity difference, recorded explicitly even when it is zero in the frozen run;
- and any unexplained residual.

Because attribution can depend on bridge order when sizing changes, D2's design must freeze and justify its attribution convention before output. A fixed-position reference must be available so direct execution-price effect is not silently mixed with resizing. CAR may be reported as a companion portfolio view; it must not replace the dollar residual.

### Adaptive diagnosis rule

Only after the aggregate dollar bridge reconciles may D2 examine the dominant mechanism further, following §6.6. The follow-up should be limited to evidence that distinguishes among competing explanations, such as:

- side;
- current structure or leg role;
- turnover/frequency;
- or ex-ante-observable package tradability (only after side/structure conditioning).

Do not automatically run every attribution. The D2 design should explain why each proposed breakdown can change the final decision.

### Inference boundaries

- Body/wing cost attribution is not the P&L of a different structure.
- Side attribution is not validation of a one-sided strategy.
- A relationship between quoted cost and P&L is not validation of a filter.
- Historical artifacts do not observe the opportunity cost of orders that would not fill under patient execution.

### D2 acceptance evidence

- Exact aggregate reconciliation under the frozen convention, within a predeclared numerical tolerance.
- Direct price-concession and sizing/capital effects shown separately.
- Dominant mechanism identified with evidence.
- Any material residual or unobservable component explicitly bounded or classified.
- No cure P&L and no selected threshold.

### Stop rule

If the bridge does not reconcile, stop and resolve the calculation/evidence problem. Do not continue to interpretation or D3.

---

## 10. D3 — Required execution envelope

### Question

> What effective execution quality must the current expression achieve to retain economically meaningful margin, and what execution facts remain unknown?

### Required answer

D3 must state:

- the break-even or decision-relevant execution requirement for the current expression;
- how fill-dependent sizing changes that requirement;
- a fixed-position reference that isolates price concession;
- how much headroom remains for commissions, no-fills, timing, and adverse selection;
- and whether the requirement is impossible, clearly tolerant, or requires empirical calibration.

### Method boundary

The D3 design must choose and freeze the smallest economically interpretable representation of execution quality. It may use an effective-spread or net-package-price requirement, but it must not label a mechanically interpolated fill rule as expected real execution.

Avoid an arbitrary dense fill grid when a direct break-even calculation or bounded sensitivity can answer the question. Do not add a generic slippage number that collapses distinct mechanisms already identified in D2.

### Attainability boundary

Existing end-of-day quote artifacts cannot show:

- whether a complex/package order would fill;
- at what limit and after how long;
- how often the trade would be skipped;
- or what adverse selection follows a fill.

Apply §6.7: if current economics depend on an execution quality between midpoint and full cross, D3 should conclude toward `EXECUTION_CALIBRATION_REQUIRED`, not recoverability. Forbidden language includes “recoverable,” “likely fillable,” “patient execution would capture midpoint,” and “historical quotes imply attainable package fills.”

### D3 acceptance evidence

- Requirement reconciles to D1–D2 economics under the §6.4 unit hierarchy.
- Fixed-position and actual-sizing effects are not conflated.
- Result includes unmodeled-friction headroom (buffer frozen in D3 design) or clearly states that none exists.
- Requirement and attainability are explicitly separated.
- No claim that historical ORATS snapshots validate complex-order execution.

---

## 11. D4 — Diagnosis, next hypothesis, and closeout

### Question

> Which explanation best fits D1–D3, and what single next action is justified by the evidence?

### D4 behavior

D4 synthesizes already accepted Sprint 007 evidence. It should not create a new family of P&L analyses or test multiple potential cures.

It must assign one primary outcome:

| Outcome | Required evidence pattern | Authorized handoff |
|---|---|---|
| `SELECTIVE_FRICTION_HYPOTHESIS` | Gross margin passes §6.5; after §6.6 side/structure conditioning, shortfall still concentrates in an ex-ante-observable cost/tradability characteristic, with meaningful mid margin outside that slice. | Write one preregistered cost-aware selection hypothesis. Do not select the historically best threshold. |
| `STRUCTURE_OR_SIZING_HYPOTHESIS` | Gross margin passes §6.5; dominant loss is systematically linked to the current payoff expression, side, leg burden, or sizing feedback (including when cost is collinear with structure). | Write one preregistered expression or sizing hypothesis. Do not backtest it in D4. |
| `EXECUTION_CALIBRATION_REQUIRED` | Gross margin passes §6.5; required execution lies between accepted mid and cross bounds; attainability cannot be inferred from historical snapshots (§6.7). | Specify an execution-shadow measurement experiment. |
| `CURRENT_IMPLEMENTATION_NOT_VIABLE` | §6.5 continue gate fails, or required execution leaves no credible room for unavoidable costs, or friction remains diffuse after discrimination. | Stop pursuing the current implementation. Do not respond with a 281-window search. |
| `EVIDENCE_INCONCLUSIVE` | A material evidence or reconciliation gap prevents discrimination. | Resolve the named gap before redesign or shadow work. |

### Required next-action specification

Exactly one of the following should be produced:

#### A. Preregistered redesign hypothesis

State:

- economic reason it should work;
- exactly what would change;
- what remains frozen;
- evidence that would support or reject it;
- permitted sensitivity range;
- and why it is not selected merely because of same-sample P&L.

Do not execute the redesign in Sprint 007.

#### B. Execution-shadow experiment

Specify measurement of:

- arrival package midpoint and natural price;
- submitted net limit price;
- fill/no-fill and time to fill;
- skipped trades;
- implementation shortfall;
- and post-fill adverse movement.

Do not add broker connectivity or place orders in Sprint 007.

#### C. Stop decision

State precisely which implementation is being stopped, which conclusions do not generalize to the underlying signal family, and what evidence would be required to reopen it.

#### D. Evidence-gap resolution

Name the missing evidence, why it blocks the decision, and the narrowest next task that can obtain it.

### D4 acceptance evidence

- One primary classification.
- Material secondary finding disclosed without creating a second next action.
- Exactly one next-action specification.
- No new cure backtest.
- Sprint closeout memo and clean status update.

---

## 12. Post-hoc optimization controls

The following are prohibited within Sprint 007:

1. Trying multiple spread/liquidity cutoffs and retaining the best result.
2. Trying long-only, short-only, alternative wings, maturities, structures, or holding frequencies and selecting the winner.
3. Reporting a same-sample filtered Sharpe as accepted strategy evidence.
4. Treating an effective-spread interpolation as observed execution.
5. Inferring a counterfactual payoff from the cost of selected legs.
6. Revising a diagnostic definition after seeing its output without versioning it as a new exploratory analysis.
7. Searching the 281 feature windows while the implementation mechanism remains unresolved.

When an exploratory relationship is necessary to discriminate explanations:

- freeze its definition before output;
- prefer a complete relationship or economically motivated contrast over a best cutoff;
- label it exploratory;
- and use it only to motivate a later preregistered test.

---

## 13. Efficiency and implementation limits

### What should be fast now

The official artifacts contain approximately ten thousand included trade keys, not the full raw options universe. Reading and analyzing these artifacts should be operationally small relative to a Surface backtest.

D0 should therefore prioritize:

- column-pruned Parquet reads;
- paired mid/cross joins performed once and reused within the accepted deliverable;
- deterministic post-pass calculations;
- and focused synthetic/reconciliation tests.

### What is deferred

- Profiling or optimizing surface loading and option construction.
- Reworking the full weekly date loop.
- Parallel or cached multi-configuration execution.
- Search-runner repair or redesign.
- Alternative-structure generation.

If D4 authorizes a later experiment requiring repeated full-history reruns, backtest efficiency can be scoped then against that experiment's actual bottleneck.

---

## 14. Cursor execution protocol

### Before each deliverable

Cursor must:

1. Read `AGENTS.md`, `docs/agenda/current_sprint.md`, this plan, the accepted prior deliverable evidence, and relevant source files.
2. Inspect the actual artifacts and schemas required for the question.
3. Write a short one-page deliverable design with a human-readable summary first. Do not rewrite this sprint plan unless a stop condition fires.
4. State:
   - question and required answer;
   - decisions being frozen;
   - existing code/artifacts to reuse;
   - minimum file/code footprint;
   - focused tests and numerical tolerances;
   - acceptance evidence;
   - inference boundaries;
   - explicit non-goals and stop rule.
5. Wait for approval.

### During an accepted deliverable

- Change only the files authorized by its accepted design.
- Do not expand into another deliverable.
- Do not alter the Sprint 006 contract or official artifacts.
- Add tests for new financial calculation behavior.
- Run focused tests first; run broader regression tests only when the changed dependency surface justifies them.
- Do not open or report granular results beyond the accepted design.

### After implementation or execution

Cursor must report:

- files changed;
- commands/tests run and results;
- evidence produced;
- reconciliation result and residual where applicable;
- limitations and inference boundary;
- whether the deliverable meets its accepted criteria;
- and the exact next authorized action.

Update `docs/agenda/current_sprint.md` only after the deliverable is reviewed and accepted. Do not start the next deliverable in the same commit unless explicitly authorized.

---

## 15. Sprint stop conditions

Pause and request review before proceeding when:

- accepted artifact identities do not match;
- the proposed calculation cannot reconcile accepted economics;
- a required field is absent;
- an analysis requires changing a P&L-sensitive strategy rule;
- the work requires a new economic engine or general framework;
- the proposed evidence cannot distinguish competing explanations;
- or the work begins evaluating multiple potential cures.

A negative or fundamental diagnosis is not a blocker. It may be the correct Sprint 007 result.

---

## 16. Sprint-level Definition of Done

Sprint 007 is complete only when all accepted deliverables jointly establish:

1. Whether the current selected expression has gross economic margin worth preserving.
2. A reconciled explanation of the midpoint-to-cross shortfall.
3. The execution quality required by the current expression and the limits of historical evidence about attainability.
4. One evidence-supported classification.
5. Exactly one next action: preregistered redesign, execution-shadow measurement, stop, or resolve a named evidence gap.
6. No strategy rescue, parameter search, or same-sample cure validation.

---

## 17. Authorization from this plan

Acceptance of this plan together with `docs/agenda/current_sprint.md` authorizes **D0 design only**.

It does not authorize:

- D0 implementation or execution;
- D1–D4 design or work;
- a full-history rerun;
- a new economic configuration;
- or any strategy, filter, structure, fill, or sizing change.

