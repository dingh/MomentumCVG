# Current sprint — 008

**Updated:** 2026-09-07

**Status:** `ACCEPTED — PLAN FROZEN; D0 DESIGN DRAFTED — AWAITING REVIEW`

**Mode:** **Build/Audit.** Sprint-level plan accepted. **D0 design drafted — awaiting review; implementation not started.**

**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](sprint8_long_filter_plan.md) — accepted detailed scope; canonical path; do not duplicate under `docs/tmp/`.

**D0 design:** [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md) — `PROPOSED — AWAITING REVIEW` (implementation not started)

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

All deliverables remain pending for **implementation/evidence**. D0 **design** is drafted and awaiting review; D0 implementation has **not** started. D1–D3 have not started.

| ID | Deliverable | Status |
|---|---|---|
| **D0** | Freeze research protocol details as needed and confirm input readiness | **Design drafted — awaiting review; implementation not started** |
| **D1** | Validate measurements; record `supported` / `unsupported` / `inconclusive` gate | Pending (after D0) |
| **D2** | Conditional threshold study, only if D1 supports it | Pending (conditional) |
| **D3** | Closeout: conclusions, limitations, implications for later work | Pending |

D0 design: [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md). Do not implement D0 or open association/profitability/threshold results until the design is accepted and implementation is authorized.

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

**D0 status:** **design drafted — awaiting review; implementation not started.**

Design: [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md).

Do not implement analysis code, execute the D0 study, or produce association, profitability, or threshold results until the D0 design is accepted and implementation is explicitly authorized.

Threshold work requires an accepted D1 `supported` gate. Pause and rescope if proposed work changes frozen selection rules, expands into short-side research, searches many cutoffs after seeing results, or claims fill attainability from quotes alone.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-07 | D0 design revised (midpoint authority, M3 full-cross rolling hurdle, deterministic missing-data gates). Agenda: distinguish drafted D0 design from pending implementation. |
| 2026-09-06 | D0 design **drafted** (`docs/tmp/sprint008_d0_design.md`) — awaiting review; implementation not started. |
| 2026-09-06 | Sprint 008 plan **accepted**. Agenda switched to Sprint 008 **Build/Audit**. |
| 2026-09-06 | Sprint 007 closed (`EXECUTION_CALIBRATION_REQUIRED`). See [`007_closeout.md`](../sprint_memos/007_closeout.md). |
