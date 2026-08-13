# Current sprint — 006

**Updated:** 2026-08-12
**Status:** `PROPOSED — AWAITING ACCEPTANCE`
**Mode:** Proposed. Acceptance authorizes D0 only. D1–D4 work is authorized one deliverable at a time after a short deliverable design is reviewed and accepted.
**Previous:** Sprint 005 — [`CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`](../sprint_memos/005_closeout.md) (closeout baseline `1517b1b`; HEAD verified unchanged at proposal)

---

## 1. Sprint intent

Bridge from trusted historical features to the **first economic backtest result we can genuinely trust**.

**Central question:** Does the frozen `42:8` Momentum+CVG signal produce believable economic results on the accepted real dataset after conservative transaction costs?

A weak or negative strategy result is still a successful Sprint 006 outcome if the evidence is correct and complete. This is not another general infrastructure sprint.

---

## 2. Starting point (Sprint 005)

Sprint 005 closed with accepted, lineaged artifacts and a one-date `SurfaceRunner` consumability smoke — not an economic evaluation.

| Input | Identity |
|-------|----------|
| Snapshot | `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Derived root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |
| Features | Frozen 281-window grid; baseline `(42,8)` ready interval `2018-10-26` → `2026-07-10` |
| Existing execution path | `SurfaceRunner` (`scripts/run_surface_search.py` → S1→S8 pipeline + surface metrics); to be validated and minimally repaired in D1 before acceptance as the baseline runner |

Do not reopen or redesign accepted Sprint 004/005 work.

```text
Sprint 004: trusted immutable input snapshot          ← CLOSED
Sprint 005: trusted full-history weekly Mom/CVG        ← CLOSED
Sprint 006: first trusted real-data economic backtest  ← THIS SPRINT (proposed)
Sprint 007: bounded robustness (only after 006 trusted)
```

---

## 3. Agenda-level decisions (fixed for this sprint)

These are locked at the agenda level. Exact numeric values and detailed economic rules are frozen in **D0** (not chosen here) before any new P&L is inspected:

* One fixed `42:8` Momentum+CVG baseline — not a parameter search
* Accepted Sprint 005 feature and surface artifacts as inputs
* Weekly option positions held to expiration
* Long ATM straddles; short iron flies
* Diagnostic mid-price results **and** conservative cross-price results, with **cross fills as the primary economic view**
* Full accepted historical coverage, with a clearly identified primary reporting period
* Joint Momentum and CVG eligibility and data-coverage checks
* Reproducible inputs, configuration, code identity, outputs, and evidence
* Decision-oriented reporting: returns, risk, stability, costs, side attribution, concentration, trading activity, and data availability
* Manual verification of a small trade sample before accepting the full result
* No retuning after observing the result

---

## 4. What must be achieved

By sprint end: a reproducible, trustworthy answer to the central question — positive, negative, or inconclusive — with enough evidence to recommend the next step without changing parameters in response to observed P&L.

---

## 5. In scope / out of scope

### In scope

* Freeze and run one fixed baseline on accepted real artifacts via the existing Surface path
* Joint eligibility / coverage correctness so failed or no-trade dates cannot disappear silently
* Diagnostic mid + conservative cross evaluation; decision-quality reporting
* Small-sample manual verification, full-history execution, reproducibility evidence, and closeout recommendation

### Explicitly out of scope

* Searching or ranking all 281 feature windows
* Broad hyperparameter optimization
* Walk-forward model or parameter selection
* Tier B integer-lot portfolio construction or a production dollar capital budget
* Live or shadow trading integration
* Broker connectivity, scheduling, monitoring, or order management
* A generic multi-strategy research platform
* New feature engineering unless a defect directly invalidates baseline correctness
* PIT earnings filtering without a trusted accepted earnings artifact
* Iron-condor comparison while KB-001 remains open
* Unrelated refactoring, cleanup, or known-bug fixes that do not block the baseline

---

## 6. High-level deliverables

Design each deliverable immediately before implementing it. **D0** must freeze every choice capable of materially changing P&L before any new P&L is inspected. **D1–D4** may defer engineering decisions only (CLI, manifests, module boundaries, tests, presentation). This agenda does not choose the exact experiment values.

### D0 — Baseline experiment contract

Freeze the exact baseline configuration, input identities, evaluation periods, assumptions, exclusions, and evidence expectations before reviewing full economic results.

### D1 — Trusted baseline runner

Provide one supported, reproducible way to run the fixed baseline from accepted artifacts, with enough identity and configuration recording to reproduce the result. Reuse existing backtest machinery; do not create a new generalized framework.

### D2 — Eligibility and coverage correctness

Jointly enforce Momentum and CVG eligibility; handle missing or ineligible observations explicitly; ensure failed or no-trade dates cannot disappear silently. Add only the focused tests and diagnostics needed.

### D3 — Decision-quality evaluation report

Produce the economic and operational evidence needed to judge the baseline: overall and yearly behavior, transaction-cost impact, long/short attribution, concentration, drawdown, trading activity, and data coverage. Support a decision — not a general analytics platform.

### D4 — Verification, full execution, and closeout

Small real-data smoke; independent inspection of a limited trade sample; frozen full-history baseline under required fill assumptions; reproducibility evidence; documented conclusion; clean closeout.

---

## 7. Definition of done (outcomes)

Success is evidence quality, not a required Sharpe or positive-return threshold.

- [ ] One frozen baseline runs reproducibly from accepted inputs
- [ ] The accepted result can be reproduced through one documented command using the recorded inputs and frozen configuration
- [ ] Input, configuration, code, and output identities are recorded
- [ ] Joint feature eligibility and complete date handling are verified
- [ ] Every expected decision date is classified as `traded`, `valid_no_trade`, or `failed`; no date is silently absent, and unresolved failures block acceptance
- [ ] A small sample of trades is independently checked from signal through settlement and aggregation
- [ ] Full-history diagnostic (mid) and conservative-fill (cross) runs complete
- [ ] Final report has enough economic and operational evidence to judge credibility
- [ ] Result and limitations are documented **without** changing parameters in response to observed P&L
- [ ] Sprint records a clear recommendation: proceed to bounded Sprint 007 robustness work, investigate a specific defect, or reject/defer the hypothesis

---

## 8. Sprint 007 handoff

Sprint 007 begins only after the Sprint 006 baseline is trusted. Its possible role is bounded robustness testing, a small preregistered candidate set, walk-forward correctness, Tier B integer lots, and a realistic capital budget.

Sprint 006 does **not** pre-build those capabilities.

---

## 9. Scope-control principles

* Prefer the narrowest change that answers the sprint question
* Reuse accepted artifacts and existing backtest code (`SurfaceRunner` / pipeline / surface metrics)
* Design each deliverable immediately before implementing it
* Do not add speculative abstractions or flexibility
* Treat non-blocking imperfections as follow-up items
* Block only on economic correctness, look-ahead bias, reproducibility, silent data loss, or inability to interpret the result
* Evidence and decision quality matter more than architectural polish

---

## 10. Decision boundaries

### Must be frozen in D0 before any new P&L is viewed

* Signal-selection and CVG-retention thresholds
* Momentum and CVG count-eligibility thresholds
* Liquidity and spread-selection rules
* Per-side position limits, ranking, and tie handling
* Option-selection and iron-fly wing rules
* Entry timing, holding period, and settlement treatment
* Sizing and capital-at-risk normalization
* Mid and cross fill definitions
* Exact full-history and primary reporting dates
* Manual trade-sample selection method
* Required decision metrics and evidence expectations

Before D0 is accepted, only coverage- and schema-level preflight may be inspected. Do not inspect aggregate P&L, Sharpe, strategy rankings, or side-level performance.

### May be decided in the relevant deliverable design

* CLI interface and command shape
* Configuration and manifest representation
* Report and output-file layout
* Module boundaries and internal implementation
* Focused test organization
* Diagnostic presentation details

---

## 11. Implementation authorization and stop rules

Sprint 006 is implementation-light, not implementation-free. A proposed code change is authorized only when all three conditions hold:

1. Without it, a Definition-of-Done outcome cannot be achieved or trusted.
2. It is the narrowest practical change that reuses the existing backtest path.
3. Its acceptance evidence is defined before implementation begins.

If a change does not pass this test, record it as follow-up work rather than implementing it.

Each deliverable starts with a short design covering the required behavior, minimum expected changes, focused tests, acceptance evidence, and explicit non-goals. Stop the deliverable once that evidence passes.

Pause and request rescoping before implementation if the work:

* Introduces a new framework, generalized abstraction, data product, or storage contract
* Pre-builds a Sprint 007 capability
* Expands into a subsystem not identified in the accepted deliverable design
* Is expected to push total enabling implementation materially beyond the planned 12–18 focused hours
* Requires changing the frozen experiment after P&L has been viewed

The 12–18 hour budget is a review trigger, not an acceptance condition. A correctness-driven change after P&L exposure must be documented and versioned as a new experiment run; it must not silently replace the original result.

---

## Progress log

| Date | Notes |
|------|-------|
| 2026-08-09 | Sprint 005 closed — [`005_closeout.md`](../sprint_memos/005_closeout.md). Sprint 006 not started. |
| 2026-08-12 | Proposed Sprint 006 agenda written into this document (`PROPOSED — AWAITING ACCEPTANCE`). HEAD at proposal: `1517b1b`. Refined with D0 experiment-freeze boundaries, deliverable-level implementation authorization, and minimal-implementation stop rules. |
