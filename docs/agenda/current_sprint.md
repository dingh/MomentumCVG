# Current sprint — 006

**Updated:** 2026-08-22
**Status:** `ACTIVE — D0/D1/D2/D3 COMPLETE; D4 BLOCKED — AMENDED PLAN AWAITING ACCEPTANCE`
**Mode:** Build. D0 accepted and frozen. D1 accepted (`241b0d3` + review fix `c6b1735`). D2 accepted (`9224068`; design `aa72a86`). D3 design `b924330` **accepted**; **D3 implementation accepted through `10133f6`** (`361b333` → `bb40864` + `f009684` → `6c7e44f` → `eaa8421` → `10133f6`). **D4 is blocked pending acceptance of an amended plan.** Unaccepted D4 activity did occur at `5c31e49`: Phase 1 checks ran but failed §1.7's literal parent-directory criterion and are **not** accepted as the official gate; the original single-date smoke was **blocked** because `BacktestRunConfig` rejects equal start/end dates; a provisional two-date smoke later passed plumbing checks but ran **before** the plan was amended and accepted, so it is **not** official D4 evidence. **Phase 3 and the official full-history run have not started**, and no aggregate P&L has been reviewed. Phase 1 and the amended Phase 2 must be **rerun in full** from the final accepted plan commit. Smoke economics are not evidence and must not be cited. Do not mark the sprint closed.
**Previous:** Sprint 005 — [`CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`](../sprint_memos/005_closeout.md) (closeout baseline `1517b1b`)
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (design commit `1cdfad7`; SHA-256 of committed LF bytes `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715`)
**D0 plan:** [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](../tmp/sprint006_d0_baseline_experiment_contract_plan.md) (`ACCEPTED — D0 COMPLETE`)

**Architecture (006):** end-to-end acceptance of the canonical **single-configuration** Surface path (`SurfaceRunner.run_single_config`) on the frozen D0 baseline—not a search platform. D1 hardens that path (any new command = thin contract adapter only). Sprint 007 may reuse the accepted runner/result schema for a bounded preregistered study; it must not retune the 006 baseline or add a separate economic execution path.

---

## 1. Sprint intent

Bridge from trusted historical features to the **first economic backtest result we can genuinely trust**, by accepting the canonical single-configuration Surface execution and evaluation path on the frozen D0 baseline.

**Central question:** Does the frozen `42:8` Momentum+CVG signal produce believable economic results on the accepted real dataset after conservative transaction costs?

A weak or negative strategy result is still a successful Sprint 006 outcome if the evidence is correct and complete. This is not another general infrastructure sprint, and it is **not** a parameter-search sprint.

---

## 2. Starting point (Sprint 005)

Sprint 005 closed with accepted, lineaged artifacts and a one-date `SurfaceRunner` consumability smoke — not an economic evaluation.

| Input | Identity |
|-------|----------|
| Snapshot | `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Derived root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |
| Features | Frozen 281-window grid; baseline `(42,8)` ready interval `2018-10-26` → `2026-07-10` |
| Existing execution path | Canonical: `SurfaceRunner.run_single_config()` → S1→S8 pipeline + surface metrics. The grid-search CLI (`scripts/run_surface_search.py` / `SurfaceSearch`) is **not** the Sprint 006 acceptance path |
| Sprint 006 D0 baseline | Frozen in `configs/sprint006_baseline_v1.json` — do not retune after P&L |

Do not reopen or redesign accepted Sprint 004/005 work.

```text
Sprint 004: trusted immutable input snapshot          ← CLOSED
Sprint 005: trusted full-history weekly Mom/CVG        ← CLOSED
Sprint 006: first trusted real-data economic backtest  ← THIS SPRINT (D0/D1/D2/D3 complete; D4 blocked, amended plan awaiting acceptance)
Sprint 007: bounded robustness (only after 006 trusted)
```

---

## 3. Agenda-level decisions (fixed for this sprint)

These are locked at the agenda level. Exact numeric values and detailed economic rules are frozen in **D0** (see contract above) before any new P&L is inspected:

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

Design each deliverable immediately before implementing it. **D0** freezes every choice capable of materially changing P&L before any new P&L is inspected. **D1–D4** may defer engineering decisions only (CLI, manifests, module boundaries, tests, presentation).

### D0 — Baseline experiment contract — **COMPLETE**

Frozen in [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json). Plan: [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](../tmp/sprint006_d0_baseline_experiment_contract_plan.md).

### D1 — Trusted baseline runner — **ACCEPTED — COMPLETE**

Design and implementation record: [`docs/tmp/sprint006_d1_trusted_baseline_runner_plan.md`](../tmp/sprint006_d1_trusted_baseline_runner_plan.md) (`ACCEPTED — D1 COMPLETE`). Accepted implementation: `241b0d3` + review fix `c6b1735`. Scope remains D1 only; D2–D4 limitations stay deferred. No accepted real-data economic backtest or aggregate P&L was run or reviewed in D1.

Exercise and harden the existing `SurfaceRunner.run_single_config()` path so the frozen D0 twin mid/cross configs run reproducibly from accepted snapshot/derived artifacts, with identity/config recording and fill-pricing verification (no stacked `cost_model` deduction). Any new command must be only a **thin frozen-contract adapter**—not a separate backtest implementation or generalized framework. Repairing or redesigning the parameter-search workflow is deferred unless a shared defect blocks fixed-contract execution.

### D2 — Eligibility and coverage correctness — **ACCEPTED — COMPLETE**

Plan / implementation record: [`docs/tmp/sprint006_d2_eligibility_coverage_correctness_plan.md`](../tmp/sprint006_d2_eligibility_coverage_correctness_plan.md) (`ACCEPTED — D2 COMPLETE`). Accepted implementation: `9224068` (design `aa72a86`). Joint Mom+CVG count eligibility (`ceil` → 28 for `(42,8)`), A1 expected-date `date_status` accounting, iron-fly all-leg `max_leg_spread_pct`, and thin adapter persistence of `date_status`. Scope remains D2 only; D3–D4 deferred. No accepted real-data economic backtest or aggregate P&L.

### D3 — Decision-quality evaluation report — **ACCEPTED — COMPLETE**

Plan: [`docs/tmp/sprint006_d3_decision_diagnostic_report_plan.md`](../tmp/sprint006_d3_decision_diagnostic_report_plan.md) (accepted design `b924330`). **Accepted implementation through `10133f6`** (`361b333` → `bb40864` + `f009684` → `6c7e44f` → `eaa8421` → `10133f6`): dual-view metrics; funnel/leg log/integrity checks; candidate view; deterministic `decision_report.json` / `.md`; D3 receipt digests. Accepted scope remains D3 only; D4 (real-data smoke, manual sample, full-history execution, conclusion) stays deferred. No accepted real-data economic backtest or aggregate P&L in D3.

### D4 — Verification, full execution, and closeout — **BLOCKED — AMENDED PLAN AWAITING ACCEPTANCE**

Plan: [`docs/tmp/sprint006_d4_execution_acceptance_plan.md`](../tmp/sprint006_d4_execution_acceptance_plan.md) (`BLOCKED — AMENDED PLAN AWAITING ACCEPTANCE`). Small real-data smoke; independent inspection of a limited trade sample; frozen full-history baseline under required fill assumptions; reproducibility evidence; documented conclusion; clean closeout.

Truthful execution record — **nothing below is accepted D4 evidence**:

* At `5c31e49`, Phase 1 pre-execution checks were attempted. They did **not** satisfy the literal §1.7 criterion (`C:/MomentumCVG_env/runs` did not exist), so Phase 1 is **not accepted** as the official gate.
* The original single-date smoke was **blocked**: `BacktestRunConfig` intentionally requires `start_date < end_date`, so equal start/end dates abort in preflight and write nothing.
* A provisional two-date smoke (`2022-09-02` … `2022-09-09`) later passed plumbing checks, but it ran **before** the plan was amended and accepted, so it is **not** official D4 evidence.
* **Phase 3 and the official full-history run have not started.** No official run directory exists.
* Once the amended plan is accepted, **Phase 1 and Phase 2 are rerun in full** from the final accepted plan commit.
* **Smoke economics are not evidence and must not be cited.** Provisional artifacts stay outside the repository under `C:/MomentumCVG_env/runs/`.

The plan's Phase 2 is now the minimal valid **two-date** window (median `2022-09-02` → next A1 date `2022-09-09`), still exactly four changed contract fields in an outside-repository copy. Adding single-date CLI support or changing the validator remains out of scope.

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

Sprint 007 begins only after the Sprint 006 baseline is trusted. It may **reuse** the accepted single-config runner and result schema for a bounded, preregistered candidate study (and related robustness items such as walk-forward correctness or Tier B sizing). It must **not** retrospectively retune the Sprint 006 baseline or introduce a separate economic execution path.

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

**D0 status:** frozen in `configs/sprint006_baseline_v1.json`. Before any new P&L is inspected, only coverage- and schema-level preflight may run. Do not inspect aggregate P&L, Sharpe, strategy rankings, or side-level performance.

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
| 2026-08-15 | D0 accepted at design commit `1cdfad7` (including §13). Contract frozen as `configs/sprint006_baseline_v1.json` (SHA-256 of committed LF bytes `3cd57f4d…ef715`). Sprint status → `ACTIVE — D0 COMPLETE; D1 AWAITING DESIGN`. No runtime changes; no economic backtest; no P&L inspected. |
| 2026-08-16 | D1 design accepted and implemented in commit `241b0d3` (parent `b380d38`): thin frozen-contract adapter + one CLI over `SurfaceRunner.run_single_config` (both mid and cross), light run receipt, overwrite refusal, S5 cap tie-break pinned to `ticker` ascending. Tests: 33 focused, 238 regression subset, full suite 1528 passed / 1 skipped. Frozen contract unchanged; no real-data economic run; no P&L inspected. D1 → `IMPLEMENTED — AWAITING REVIEW`. Review fix `c6b1735`: run output directory must also be outside the repo and outside the mutable producer cache. |
| 2026-08-16 | D1 reviewed and accepted (`241b0d3` + `c6b1735`). Sprint status → `ACTIVE — D0/D1 COMPLETE; D2 AWAITING DESIGN`. Accepted scope remains D1 only; D2–D4 deferred. No accepted real-data economic backtest or aggregate P&L in D1. Next authorized activity: D2 design only (not implementation). |
| 2026-08-16 | D2 design accepted at `aa72a86` and implemented: joint Mom+CVG ceil eligibility, A1 `date_status` on `SurfaceRunResult`, iron-fly body spread gate, adapter persistence/receipt fields. Frozen JSON untouched; no real-data economic run; no P&L inspected. Sprint status → `ACTIVE — D0/D1 COMPLETE; D2 IMPLEMENTED — AWAITING REVIEW`. |
| 2026-08-16 | D2 reviewed and accepted (`9224068`). Correctness review found no D2 blockers. Sprint status → `ACTIVE — D0/D1/D2 COMPLETE; D3 AWAITING DESIGN`. Scope remains D2 only; D3–D4 deferred. No accepted real-data economic backtest or aggregate P&L. Next authorized activity: D3 design only (not implementation). |
| 2026-08-17 | D3 design proposed: [`sprint006_d3_decision_diagnostic_report_plan.md`](../tmp/sprint006_d3_decision_diagnostic_report_plan.md) (`PROPOSED — AWAITING ACCEPTANCE`). HEAD at proposal: `62bdf38`. Sprint status → `ACTIVE — D0/D1/D2 COMPLETE; D3 DESIGN UNDER REVIEW`. No D3 code, frozen-config change, real-data run, or P&L inspection. Implementation is not authorized until the plan is accepted. |
| 2026-08-17 | D3 design corrected in-place (follow-up to `688c2a3`): short-iron-fly `abs(quantity)` scaling; abort on broken traded-date economics; remove `outcome_status` and drop-ticker/drop-week; funnel null vs zero; fill-assumption labeling; small structure-failure classes. Status remains `D3 DESIGN UNDER REVIEW` / `PROPOSED — AWAITING ACCEPTANCE`. No implementation, frozen-config change, real-data run, or P&L inspection. |
| 2026-08-18 | D3 design `b924330` accepted. Sprint status → `ACTIVE — D0/D1/D2 COMPLETE; D3 IMPLEMENTATION IN PROGRESS (COMMIT 1)`. Commit 1 authorizes pure `surface_decision_report` calculations and synthetic tests only. No real-data run or aggregate P&L inspection. Commits 2–3 deferred. |
| 2026-08-19 | D3 Commit 1 (`361b333`) accepted. Commit 2 implemented: shared S2 eligibility helper, funnel_summary, constructable leg log, included-trade reconciliation checks. Status → `D3 IMPLEMENTATION IN PROGRESS (COMMIT 2)`. D3 not complete. No real-data run or aggregate P&L inspection. |
| 2026-08-19 | D3 Commit 3 implemented: candidate view, dual-fill decision report JSON/MD, adapter persistence, D3 receipt (`deliverable=sprint006_d3`), deferred list = D4 only. Status → `ACTIVE — D0/D1/D2 COMPLETE; D3 IMPLEMENTED — AWAITING REVIEW`. Frozen contract unchanged; no real-data economic run; no aggregate P&L inspected. Sprint remains open; D4 deferred. |
| 2026-08-22 | **D3 reviewed and accepted** through `10133f6` (`361b333`, `bb40864` + `f009684`, `6c7e44f`, `eaa8421`, `10133f6`); accepted scope remains D3 only. **D4 planning started**: [`sprint006_d4_execution_acceptance_plan.md`](../tmp/sprint006_d4_execution_acceptance_plan.md) written as `PROPOSED — AWAITING ACCEPTANCE`. HEAD at proposal: `10133f6`. Status → `ACTIVE — D0/D1/D2/D3 COMPLETE; D4 PLANNING — PLAN PROPOSED, AWAITING ACCEPTANCE`. **No D4 execution has started**: no real-data economic run, no smoke run, no official run directory, no aggregate P&L inspected; frozen JSON untouched; no production code changed. Plan records four planning gaps (median-date convention, no single-date smoke path in the CLI, unpinned input digests, no run-duration field) for human review. |
| 2026-08-22 | **Unaccepted D4 activity at `5c31e49`, then plan amended.** Phase 1 checks ran but failed §1.7's literal parent-directory criterion → **not accepted** as the official gate. The original single-date smoke was **blocked**: `BacktestRunConfig` requires `start_date < end_date`, so the adapter aborted in preflight and wrote nothing. A provisional two-date smoke (`2022-09-02` … `2022-09-09`) then passed plumbing checks, but ran **before** the plan was amended and accepted, so it is **not** official D4 evidence. A copy of those smoke artifacts was mistakenly committed as `docs/tmp/sprint006_d4_phase12_review/` and is **deleted** in this commit; outside-repository originals under `C:/MomentumCVG_env/runs/` are untouched and are never cited. Plan Phase 2 corrected to the minimal valid **two-date** window (four contract fields: start → `MEDIAN_DATE`, end → `SMOKE_END_DATE`); validator and CLI unchanged; single-date support stays out of scope. Status → `ACTIVE — D0/D1/D2/D3 COMPLETE; D4 BLOCKED — AMENDED PLAN AWAITING ACCEPTANCE`. **Phase 3 and the official full-history run have not started**; no aggregate P&L reviewed. Phase 1 and amended Phase 2 must be rerun in full from the final accepted plan commit. Frozen JSON untouched; no production code or test changed. |
