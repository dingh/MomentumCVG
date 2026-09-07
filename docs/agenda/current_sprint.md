# Current sprint — 008

**Updated:** 2026-09-07

**Status:** `D0 EXECUTION COMPLETE — EVIDENCE AWAITING REVIEW` (`READY_WITH_NARROW_ENABLING_CHANGE`)

**Mode:** **Build/Audit.** Sprint-level plan accepted. D0 design amended (`sprint008_d0_crossed_quote_v1`) and re-executed; evidence under review. D1–D3 pending.

**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](sprint8_long_filter_plan.md) — accepted detailed scope; canonical path; do not duplicate under `docs/tmp/`.

**D0 design:** [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md) — `ACCEPTED` (+ crossed-quote amendment v1)  
**D0 evidence:** [`docs/tmp/sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md) — verdict `READY_WITH_NARROW_ENABLING_CHANGE`; official run `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` (SHA `af24f50`, clean tree)

**Previous:** Sprint 007 — [`CLOSED — D3 ACCEPTED; D4 EXECUTION_CALIBRATION_REQUIRED`](../sprint_memos/007_closeout.md)  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable evidence; not edited by this sprint.

---

## 1. Sprint intent

Answer, for the frozen `42:8` long ATM-straddle book only:

> Among the current long-straddle candidates, do entry-time measurements reliably distinguish net profitability per trade, and, if they do, can a simple threshold improve the long book while retaining valuable winners?

Sprint 008 is a separately motivated research experiment. It preserves accepted Sprint 006/007 results and conclusions. It does not rescue the frozen cross book, retune the signal window, or claim that historical quote scenarios are attainable fills.

Method order is fixed:

1. Define measurements.
2. Validate measurement–profitability relationships on an independent equal-dollar long baseline (required gate).
3. Only if supported, run conditional threshold testing, tracking losses avoided and winning profits retained (including largest contributors).

---

## 2. Central conventions (accepted)

| Item | Convention |
|---|---|
| Side | Long ATM straddles only |
| Selection | Frozen `42:8` / CVG / liquidity / structure / name cap / weekly hold-to-expiry |
| Research baseline | Equal stake \(B/N\) per eligible pre-filter candidate; rejected capital stays cash |
| Sizing | Scenario all-in cost: \(q_i(h)=(B/N)/(M_i + h H_i + \mathrm{fees}_i)\); freeze quantities within each \(h\) |
| Dependence | Consecutive-date block resampling; block protocol freezes in D1 before association output |
| Primary scenario | Full cross \(h=1\); midpoint diagnostic; limited intermediates as sensitivity only |

Full protocol: [`sprint8_long_filter_plan.md`](sprint8_long_filter_plan.md).

---

## 3. Deliverables

| ID | Deliverable | Status |
|---|---|---|
| **D0** | Freeze research protocol details as needed and confirm input readiness | **Execution complete — evidence awaiting review** (`READY_WITH_NARROW_ENABLING_CHANGE`) |
| **D1** | Validate measurements; record `supported` / `unsupported` / `inconclusive` gate | Pending (awaiting D0 evidence review) |
| **D2** | Conditional threshold study, only if D1 supports it | Pending (conditional) |
| **D3** | Closeout: conclusions, limitations, implications for later work | Pending |

D0 design: [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md).  
D0 evidence: [`docs/tmp/sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md).

Do not start D1 association/profitability/threshold work until the D0 evidence is reviewed and any required protocol amendment is accepted.

---

## 4. Definition of done

Sprint 008 is complete when:

- [ ] D0 confirms protocol freeze / input readiness (or records a specific blocker).
- [ ] D1 records an explicit measurement gate decision.
- [ ] D2 runs the conditional threshold study if justified, or records a stop without threshold search.
- [ ] D3 closes with a defensible answer to the central question.
- [ ] Equal-dollar baseline, within-\(h\) quantity freeze, unused-cash treatment, and winning-profit retention metrics are honored.
- [ ] Sprint 006/007 accepted results remain unreinterpreted.
- [ ] No signal-window, short-side, structure, or execution-policy winner is selected from this sprint.
- [ ] Hypothetical fills are not claimed attainable.
- [ ] Focused tests pass for any new financial calculation code.
- [ ] Remaining risks and required future/unused confirmation are documented.

An unsupported measurement, inconclusive relationship, or ineffective threshold is a valid completion.

---

## 5. Explicitly out of scope

- Short-side measurements, wings, protection, and margin (Sprint 009).
- Signal-window search or `42:8` retuning.
- New trade structures; broker / live / paper execution infrastructure.
- Sizing-optimization search beyond the pinned equal-dollar convention.
- Editing the frozen Sprint 006 contract or mutating official Sprint 006/007 evidence directories.
- Using future outcomes as entry features, or \(H/\lvert S_0-K\rvert\).

---

## 6. Authorization and next action

**Plan status:** accepted and frozen.

**D0 status:** **execution complete — evidence awaiting review.** Computed verdict `READY_WITH_NARROW_ENABLING_CHANGE` under policy `sprint008_d0_crossed_quote_v1` (1 exclusion: `2025-04-04|MU` held as cash; `in_N` unchanged). Rerun SHA `af24f50`.

Reviewers should accept readiness + cash treatment, then authorize D1 design. Do not run association, profitability, or threshold analysis yet.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-07 | D0 **rerun** under crossed-quote policy v1; evidence `sprint008_d0_20260907T204449Z`; SHA `af24f50` (clean); verdict `READY_WITH_NARROW_ENABLING_CHANGE` (MU cash-excluded). Evidence awaiting review. |
| 2026-09-07 | D0 **rerun** after validation fixes; evidence `sprint008_d0_20260907T202025Z`; SHA `5023fe7` (clean); verdict still `BLOCKED_BY_SPECIFIC_INPUT_GAP` (MU preserved). Evidence awaiting review. |
| 2026-09-07 | D0 **executed**; evidence `sprint008_d0_20260907T193835Z`; verdict `BLOCKED_BY_SPECIFIC_INPUT_GAP`. Evidence awaiting review. |
| 2026-09-07 | D0 design revised (midpoint authority, M3 full-cross rolling hurdle, deterministic missing-data gates). |
| 2026-09-06 | D0 design **drafted** (`docs/tmp/sprint008_d0_design.md`). |
| 2026-09-06 | Sprint 008 plan **accepted**. Agenda switched to Sprint 008 **Build/Audit**. |
| 2026-09-06 | Sprint 007 closed (`EXECUTION_CALIBRATION_REQUIRED`). See [`007_closeout.md`](../sprint_memos/007_closeout.md). |
