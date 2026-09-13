# Current sprint — 008

**Updated:** 2026-09-12

**Status:** `D3 SUBMITTED — AWAITING FINAL REVIEW`

**Mode:** **Build/Audit.** Sprint-level plan accepted. **D0 accepted.** **D1 closed and reviewed.** Findings unchanged: M1/M2 `inconclusive`, M3 `unsupported`, gate **`STOP_NO_THRESHOLDS` preserved.** **D2 reviewed and accepted** through `c9b0a6a`. **D3 closeout submitted** for final review; not yet reviewer-accepted.

**Working plan:** [`docs/agenda/sprint8_long_filter_plan.md`](sprint8_long_filter_plan.md) — accepted detailed scope; canonical path; do not duplicate under `docs/tmp/`.

**D0 design:** [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md) — `ACCEPTED` (+ crossed-quote amendment v1)  
**D0 evidence:** [`docs/tmp/sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md) — **accepted** `READY_WITH_NARROW_ENABLING_CHANGE`; `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` (SHA `af24f50`)  
**D1 design:** [`docs/tmp/sprint008_d1_design.md`](../tmp/sprint008_d1_design.md) — `ACCEPTED`  
**D1 evidence:** [`docs/tmp/sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md) — **reviewed / D1 closed** 2026-09-12; gate `STOP_NO_THRESHOLDS` unchanged; `C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/` (exec SHA `72629a0`, clean tree)
**D1 within-date follow-up protocol:** [`docs/tmp/sprint008_d1_within_date_followup_protocol.md`](../tmp/sprint008_d1_within_date_followup_protocol.md) — frozen; **reviewed** 2026-09-12
**D1 within-date follow-up evidence:** [`docs/tmp/sprint008_d1_within_date_followup_evidence.md`](../tmp/sprint008_d1_within_date_followup_evidence.md) — **reviewed** 2026-09-12; `C:/MomentumCVG_env/runs/sprint008_d1_within_date_20260908T195615Z/` (HEAD `c23c364`, dirty tree at run)
**D1 cost-diagnosis protocol:** [`docs/tmp/sprint008_d1_cost_diagnosis_protocol.md`](../tmp/sprint008_d1_cost_diagnosis_protocol.md) — bounded amendment (two fixed U-exclusions); **reviewed** 2026-09-12
**D1 cost-diagnosis evidence:** [`docs/tmp/sprint008_d1_cost_diagnosis_evidence.md`](../tmp/sprint008_d1_cost_diagnosis_evidence.md) — **reviewed** (correction accepted, commit `870d4b7`); `C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260912T211530Z/` (HEAD `e1248f0`, dirty at run). Prior run preserved: `sprint008_d1_cost_diagnosis_20260911T162501Z`.
**D2 design:** [`docs/tmp/sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) — **`ACCEPTED`** 2026-09-12 (reviewed commit `c2ba972`)
**D2 evidence:** [`docs/tmp/sprint008_d2_evidence_review.md`](../tmp/sprint008_d2_evidence_review.md) — **reviewed / accepted** through `c9b0a6a`; `C:/MomentumCVG_env/runs/sprint008_d2_20260912T232144Z/` (HEAD `c2ba972`, dirty at run). Historical `STOP_NO_THRESHOLDS` preserved.
**D3 closeout:** [`docs/sprint_memos/008_closeout.md`](../sprint_memos/008_closeout.md) — **submitted for final review**; not yet reviewer-accepted.

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
| Crossed quotes | Policy `sprint008_d0_crossed_quote_v1`: exclude from execution/analysis; keep in \(N\); stake cash |

Full protocol: [`sprint8_long_filter_plan.md`](sprint8_long_filter_plan.md).

---

## 3. Deliverables

| ID | Deliverable | Status |
|---|---|---|
| **D0** | Freeze research protocol details as needed and confirm input readiness | **Accepted** (`READY_WITH_NARROW_ENABLING_CHANGE`) |
| **D1** | Validate measurements; record `supported` / `unsupported` / `inconclusive` gate | **Closed / reviewed** (`STOP_NO_THRESHOLDS` **preserved**; M1/M2 inconclusive, M3 unsupported) |
| **D1-FU** | Within-date lowest vs highest 20% (M1/M2); exploratory | **Complete and reviewed** (does not amend the gate) |
| **D1-CD** | Cost diagnosis + fixed U-exclusion (M1, M2); bounded amendment | **Complete and reviewed** (correction `870d4b7` accepted; not D2) |
| **D2** | Accepted amendment: one retrospective evaluation of unchanged M1 and M2 exclude-U rules on `2024-01-01` through `2026-07-10` | **Reviewed / accepted** through `c9b0a6a`. Not a threshold search and not a change to historical `STOP_NO_THRESHOLDS` |
| **D3** | Closeout: conclusions, limitations, implications for later work | **Submitted for final review** ([`008_closeout.md`](../sprint_memos/008_closeout.md)). Not yet reviewer-accepted |

D0: [`sprint008_d0_design.md`](../tmp/sprint008_d0_design.md), [`sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md).  
D1: [`sprint008_d1_design.md`](../tmp/sprint008_d1_design.md), [`sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md).  
D1-FU: [`sprint008_d1_within_date_followup_protocol.md`](../tmp/sprint008_d1_within_date_followup_protocol.md), [`sprint008_d1_within_date_followup_evidence.md`](../tmp/sprint008_d1_within_date_followup_evidence.md).  
D1-CD: [`sprint008_d1_cost_diagnosis_protocol.md`](../tmp/sprint008_d1_cost_diagnosis_protocol.md), [`sprint008_d1_cost_diagnosis_evidence.md`](../tmp/sprint008_d1_cost_diagnosis_evidence.md).
D2: [`sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) — design accepted (`c2ba972`); evidence **reviewed / accepted** through `c9b0a6a`.
D3: [`008_closeout.md`](../sprint_memos/008_closeout.md) — submitted for final review.

D1 is closed. Historical `STOP_NO_THRESHOLDS` is preserved. The accepted D2 amendment replaced threshold search with the specified retrospective evaluation. It is not a threshold search and does not change D1 findings.

---

## 4. Definition of done

Sprint 008 is complete when:

- [x] D0 confirms protocol freeze / input readiness (or records a specific blocker).
- [x] D1 records an explicit measurement gate decision (`STOP_NO_THRESHOLDS`; follow-ups reviewed; not a D2 completion).
- [x] D2 is accepted and executed, or a stop is recorded without threshold search. Reviewed / accepted through `c9b0a6a`. The amendment replaced threshold search; historical `STOP_NO_THRESHOLDS` unchanged.
- [x] D3 memo answers the central question and is **submitted for final review** ([`008_closeout.md`](../sprint_memos/008_closeout.md)). Not yet reviewer-accepted.
- [x] Equal-dollar baseline, within-\(h\) quantity freeze, unused-cash treatment, and winning-profit retention metrics are honored. Dollar retention is reported separately from winner-count retention.
- [x] Sprint 006/007 accepted results remain unreinterpreted.
- [x] No signal-window, short-side, structure, or execution-policy winner is selected from this sprint. M1 is a candidate for later validation only.
- [x] Hypothetical fills are not claimed attainable.
- [x] Focused tests pass for any new financial calculation code (67 passed before the D2 run; recorded in the D2 evidence).
- [x] Remaining risks and required future/unused confirmation are documented in the closeout.

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

**D0 status:** **accepted** — `READY_WITH_NARROW_ENABLING_CHANGE` under `sprint008_d0_crossed_quote_v1`.

**D1 status:** **closed and reviewed** (2026-09-12). Labels unchanged: M1/M2 `inconclusive`, M3 `unsupported`. Gate **`STOP_NO_THRESHOLDS` preserved.** Original evidence `sprint008_d1_20260907T223037Z` (exec SHA `72629a0`, clean tree). Within-date follow-up and corrected cost-diagnosis follow-up are accepted as D1 extensions, not as a change to those labels.

**D2 status:** **reviewed and accepted** through `c9b0a6a`. [`docs/tmp/sprint008_d2_evidence_review.md`](../tmp/sprint008_d2_evidence_review.md); `C:/MomentumCVG_env/runs/sprint008_d2_20260912T232144Z/`. Retrospective validation only. M1 label `relative_benefit`; M2 label `inconclusive`. Neither filtered book is profitable over the full window. Do not promote a filter or retune cutoffs. Historical `STOP_NO_THRESHOLDS` is unchanged.

**D3 status:** **submitted for final review.** [`docs/sprint_memos/008_closeout.md`](../sprint_memos/008_closeout.md). Not yet reviewer-accepted. Sprint 009 is not designed here. The user’s stated intention is a separately scoped short-side sprint.

---

## Changelog

| Date | Event |
|------|-------|
| 2026-09-12 | D3 closeout **submitted for final review** ([`008_closeout.md`](../sprint_memos/008_closeout.md)). Not yet reviewer-accepted. D2 accepted through `c9b0a6a`. No experiment rerun. |
| 2026-09-12 | D2 **executed.** Evidence `sprint008_d2_20260912T232144Z` awaiting review. M1 `relative_benefit`; M2 `inconclusive`. No filter promoted. D3 not started. |
| 2026-09-12 | D2 design **accepted** (reviewed commit `c2ba972`). Amendment replaces threshold search with one retrospective evaluation of unchanged M1 and M2 exclude-U rules on `2024-01-01` through `2026-07-10`. Historical `STOP_NO_THRESHOLDS` preserved. Implementation authorized. |
| 2026-09-12 | D2 design **revised in place** after review of `269fde0` (evaluation-window portfolio interface; authoritative calendar). Still `DRAFT — AWAITING REVIEW`. Not implemented. |
| 2026-09-12 | D2 design **drafted** (`sprint008_d2_design.md`) — awaiting review. Bounded amendment only; D1 gate unchanged; not implemented. |
| 2026-09-12 | D1 **documentation closeout.** Original D1, within-date follow-up, and corrected cost-diagnosis (`870d4b7`) reviewed. Findings unchanged: M1/M2 inconclusive, M3 unsupported, `STOP_NO_THRESHOLDS`. Follow-ups did not complete D2. Proposed eval-window amendment recorded as pending design only. |
| 2026-09-12 | D1 cost-diagnosis **corrected** (drawdown peak includes $0; half-period exclusion and weekly concentration reported; no automatic recommendation). Evidence `sprint008_d1_cost_diagnosis_20260912T211530Z`. Core P&L unchanged vs prior run. |
| 2026-09-11 | D1 cost-diagnosis / fixed U-exclusion **executed**; evidence `sprint008_d1_cost_diagnosis_20260911T162501Z`; HEAD `db0e859` (dirty). M1 net L−U mostly mechanical spread; U-exclusion point uplift not HAC-significant. Historical `STOP_NO_THRESHOLDS` preserved. |
| 2026-09-08 | D1 within-date L vs U follow-up **executed** (M1/M2); evidence `sprint008_d1_within_date_20260908T195615Z`; HEAD `c23c364` (dirty). Point mean \(d_t\) positive; not adj.-significant. Original D1 `STOP_NO_THRESHOLDS` preserved. |
| 2026-09-07 | D1 **executed**; evidence `sprint008_d1_20260907T223037Z`; SHA `72629a0` (clean); labels M1/M2 inconclusive, M3 unsupported; gate `STOP_NO_THRESHOLDS`. Evidence awaiting review. |
| 2026-09-07 | D1 design **accepted**; agenda → `D1 IMPLEMENTATION IN PROGRESS`. |
| 2026-09-07 | D1 design **revised** (Bonferroni \(\Delta\) gate, gross/drag decomposition, development-only firewall, bootstrap/grouping protocol, exhaustive labels). Still `DRAFT — AWAITING REVIEW`. |
| 2026-09-07 | D0 **accepted**; D1 design **drafted** (`sprint008_d1_design.md`). Agenda → `D1 DESIGN AWAITING REVIEW`. |
| 2026-09-07 | D0 **rerun** under crossed-quote policy v1; evidence `sprint008_d0_20260907T204449Z`; SHA `af24f50` (clean); verdict `READY_WITH_NARROW_ENABLING_CHANGE` (MU cash-excluded). |
| 2026-09-07 | D0 **rerun** after validation fixes; evidence `sprint008_d0_20260907T202025Z`; SHA `5023fe7` (clean); verdict still `BLOCKED_BY_SPECIFIC_INPUT_GAP` (MU preserved). |
| 2026-09-07 | D0 **executed**; evidence `sprint008_d0_20260907T193835Z`; verdict `BLOCKED_BY_SPECIFIC_INPUT_GAP`. |
| 2026-09-07 | D0 design revised (midpoint authority, M3 full-cross rolling hurdle, deterministic missing-data gates). |
| 2026-09-06 | D0 design **drafted** (`docs/tmp/sprint008_d0_design.md`). |
| 2026-09-06 | Sprint 008 plan **accepted**. Agenda switched to Sprint 008 **Build/Audit**. |
| 2026-09-06 | Sprint 007 closed (`EXECUTION_CALIBRATION_REQUIRED`). See [`007_closeout.md`](../sprint_memos/007_closeout.md). |
