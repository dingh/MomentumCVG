# Sprint 007 — closeout

**Status:** `CLOSED — D3 ACCEPTED; D4 EXECUTION_CALIBRATION_REQUIRED`
**Closed:** 2026-09-06
**D4 design:** [`docs/tmp/sprint007_d4_design.md`](../tmp/sprint007_d4_design.md) — `ACCEPTED`

Sprint 006 remains **accepted and unchanged**. This closeout does not replace the official cross result, retune `42:8`, or authorize live orders.

---

## 1. Verdict

Sprint 007 diagnosed the midpoint-to-cross implementation shortfall on the frozen Sprint 006 selected option book. D0–D3 are accepted. D4 assigns one primary outcome from already accepted evidence. No new P&L analysis was run.

| Conclusion | Result |
|------------|--------|
| **D0–D3 evidence** | `ACCEPTED` |
| **Primary D4 outcome** | `EXECUTION_CALIBRATION_REQUIRED` |
| **Next sprint** | **Unauthorized** |

Gross midpoint margin exists. Entry-price concession dominates the mid-to-cross gap. Sizing feedback is not material. Real package-fill attainability remains unknown. The required Path R \(h\) lies strictly between midpoint and full cross; this closeout does **not** claim that required \(h\) is achievable.

---

## 2. Sprint objective

**Central question:** For the frozen `42:8` selected option book, where does the midpoint-to-cross implementation shortfall come from, how much implementation cost can the apparent gross margin tolerate, and what single next action is justified?

Sprint 007 answered that as a diagnosis, not as a rescue of Sprint 006. Success was a trustworthy classification and one next action. A positive backtest or identified cure was not required.

---

## 3. Why `EXECUTION_CALIBRATION_REQUIRED`

Working plan §6.7 / §11: D1 continue passed; required Path R execution lies strictly between midpoint and full cross; historical end-of-day quotes cannot show package-fill attainability. That mapping is the primary outcome unless a stronger explanation controls the decision. None does.

| Competing outcome | Why it is not primary |
|---|---|
| `SELECTIVE_FRICTION_HYPOTHESIS` | Expensive-package concentration is real but secondary. It does not replace the book-level requirement between mid and cross, and it does not validate a filter |
| `STRUCTURE_OR_SIZING_HYPOTHESIS` | Short share of \(\Delta_{\mathrm{price}}\) is 63.6% (< 70%); \(\Delta_{\mathrm{size}}\) is not material |
| `CURRENT_IMPLEMENTATION_NOT_VIABLE` | D1 continue passed; Path R headroom 50→25 and 25→P0 is positive |
| `EVIDENCE_INCONCLUSIVE` | D0–D3 reconciliations passed. The remaining gap is attainability, which this outcome names |

Authorized handoff (working plan §11.B, specialized here): a future **manual-first execution-observation** project. Not a redesign. Not a stop. Not a filter.

---

## 4. Accepted D0–D3 chain

| Deliverable | Verdict / class | Official evidence (outside repo) | Commit |
|---|---|---|---|
| D0 | `READY_WITH_NARROW_ENABLING_CHANGE` | `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/` | `8a59474` |
| D1 | `D1_CONTINUE_TO_D2` | `C:/MomentumCVG_env/runs/sprint007_d1_20260903T013933Z/` | `516235f` |
| D2A | price dominant; size not material | `C:/MomentumCVG_env/runs/sprint007_d2a_20260904T043124Z/` | — |
| D2B | final class `D3_EXECUTION_FOCUSED` | `C:/MomentumCVG_env/runs/sprint007_d2b_20260904T045019Z/` | `ab53e26` |
| D3 | `D3_ENVELOPE` (not blocked) | `C:/MomentumCVG_env/runs/sprint007_d3_20260906T024837Z/` | `5450740` |

Sprint 006 official run remains `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` at execution SHA `e205b9acc5d0400aa38169de721acb7fb8268f29`. Frozen contract [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) was not edited.

### D1 — gross margin exists

Primary window, midpoint, official included book:

| Metric | Value |
|---|---|
| \(P_{\mathrm{mid}}\) | \(+159{,}283.23\) |
| View A mean cycle CAR | \(+2.396\%\) |
| Trades / traded dates | 9,212 / 341 |
| Long P&L / short P&L | \(+101{,}326\) / \(+57{,}957\) |

All four continue-gate parts passed. Both sides are positive at midpoint. This is current-expression economics, not a pure Momentum/CVG quality claim.

### D2A — entry-price concession is dominant; sizing is immaterial

| Term | Dollars | Share of \(\lvert G\rvert\) | Material? |
|---|---|---|---|
| \(G = P_{\mathrm{cross}} - P_{\mathrm{mid}}\) | \(-322{,}556.06\) | 100% | — |
| \(\Delta_{\mathrm{price}}\) | \(-343{,}367.92\) | 106.5% | **yes** |
| \(\Delta_{\mathrm{size}}\) | \(+20{,}811.85\) | 6.45% | **no** |
| \(\Delta_{\mathrm{set}}\) | \(0\) | 0% | no |
| Residual | \(\sim 0\) | — | no |

Hybrid \(P(Q_{\mathrm{mid}}, p_{\mathrm{cross}}) = -184{,}084.69\). Short share of \(\Delta_{\mathrm{price}}\) is 63.6%; long 36.4%. Neither side meets the 70% structure predicate.

### D3 — required envelope (accepted)

Path R (resized Tier A at each \(h\)). \(h=0\) is midpoint; \(h=1\) is full cross. There is no `h_req` and no selected live fill.

| Bound | Meaning |
|---|---|
| \(h < 0.2236\) | retains 50% of midpoint P&L |
| \(h < 0.3407\) | retains 25% of midpoint P&L |
| \(h < 0.4616\) | remains dollar-profitable **before unmodeled costs** |

Companion: Path R View A CAR first reaches zero at \(h_{R,\mathrm{CAR}0} = 0.4546\).

Headroom (additional \(h\) between Path R dollar-margin marks): 50→25 = 0.1171; 25→P0 = 0.1209. Both positive. Distances from break-even to full cross (0.5384 P0→1; 0.5454 CAR0→1) are **not** headroom.

Path F diagnostics are slightly later than Path R and do not bind the envelope. Historical quotes state **requirement** only. They do not show whether any package order would fill inside that envelope.

---

## 5. Secondary findings (not a second next action)

These are disclosed because they are material. They do **not** set the primary outcome, a filter threshold, or a long-only strategy.

**Expensive-package concentration (D2B).** Book tercile 3 is 3,070 of 9,212 trades (33.3%) and holds 57.7% of \(\lvert\Delta_{\mathrm{price}}\rvert\). T1+T2 retain midpoint P&L \(+105{,}035\). That concentration is meaningful but not overwhelming. It does not validate a cutoff, a filtered book, or a profitable implementation. No trade was removed and no threshold was searched.

**Short-side fragility (D3 Path R sides, descriptive).** Short-side Path R P&L is already negative by \(h_{R,25}\). Book-level dollar break-even is later because longs stay positive. Side dollars are not a long-only authorization and do not prove a wingless or short-only alternative.

---

## 6. Exactly one next action

A **future** sprint — separately designed and authorized — should run a **manual-first execution-observation** project on the frozen selected book:

1. Export weekly order tickets (identity, structure, and preregistered hypothetical net limits).
2. Record, without placing live orders:
   - arrival package midpoint / natural price;
   - whether a preregistered hypothetical net limit was touched (touch / no-touch);
   - time-to-touch;
   - skipped observations;
   - post-touch quote movement.
3. After observation is defined, validate order plumbing with **paper** trading only.

A quote touch is **not** a fill. Paper trading validates order plumbing only. Real fill probability, time-to-fill, implementation shortfall, and adverse selection remain unresolved until separately authorized live-order observation is available.

Live orders are **not** authorized. This closeout does not add broker code, place orders, or implement the observer.

This is the working-plan §11.B execution-shadow handoff, specialized to quote observation first. It is not a redesign hypothesis, not a stop of the current implementation, and not a filter search.

---

## 7. What this closeout does not claim

- That the required Path R \(h\) is achievable in live or paper markets.
- That midpoint, \(h_{R,50}\), \(h_{R,25}\), or \(h_{R,P0}\) is a live limit.
- That a quote touch is a fill, or that paper plumbing tests establish fill probability, time-to-fill, implementation shortfall, or adverse selection.
- That historical quotes imply attainable package fills.
- That a spread/liquidity filter or long-only book is validated.
- That Sprint 006 cross economics are revised.
- That Momentum or CVG as signal families are accepted or rejected.

Forbidden language remains unused: recoverable; likely fillable; patient execution would capture midpoint; historical quotes imply attainable package fills.

---

## 8. Definition of done

| Outcome | Status | Evidence |
|---------|--------|----------|
| Official Sprint 006 artifacts unchanged | ✓ | Official run identity above; frozen contract not edited |
| D0 trusted artifact-first path | ✓ | `READY_WITH_NARROW_ENABLING_CHANGE` |
| D1 gross-margin location | ✓ | `D1_CONTINUE_TO_D2`; both sides positive at mid |
| D2 shortfall reconciled | ✓ | Residual ~0; \(\Delta_{\mathrm{price}}\) dominant |
| D3 required execution and unknown attainability | ✓ | Path R envelope above; review [`docs/tmp/sprint007_d3_evidence_review.md`](../tmp/sprint007_d3_evidence_review.md) **accepted** |
| D4 one classification and one next action | ✓ | `EXECUTION_CALIBRATION_REQUIRED`; §6 |
| Diagnostic rules frozen before granular output | ✓ | D0–D3 designs accepted before their official runs |
| Exploratory vs accepted separated | ✓ | D2B concentration and D3 side splits labeled secondary / descriptive |
| No same-sample winner (window, threshold, side, structure, fill) | ✓ | No filter or \(h\) selected |
| Focused tests recorded on implementation commits | ✓ | D3: 29 focused / 1687 full at `5450740` (no new suite at closeout) |
| Limitations and stop conditions documented | ✓ | §7, §9 |

---

## 9. Accepted limitations

- Midpoint is an optimistic gross-expression reference, not expected live execution.
- Full cross is the accepted Sprint 006 conservative case, not proven real execution.
- \(h\) is a mechanical half-spread fraction, not a fill probability.
- Commissions, missed fills, timing, and adverse selection are unmodeled; the 50% / 25% buffers exist because of them.
- Historical end-of-day quotes cannot validate complex-order execution.
- Path F remaining positive is not executable return.
- Hold-to-expiry; no earnings filter; iron-fly wings remain the frozen 0.15-delta selection.
- Iron-condor remains untested while KB-001 is open.
- Sprint 007 diagnoses the current selected expression, not intrinsic signal quality.

---

## 10. Post-Sprint-007 handoff

Sprint 007 is **closed**. The next sprint is **not** authorized by this closeout.

Any later execution-observation work requires a **new preregistered design**. Do not retune `configs/sprint006_baseline_v1.json`, search feature windows, or pick a spread cutoff from these same-sample results.

Preserve outside-repository artifacts:

- Sprint 006 official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- D0: `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/`
- D1: `C:/MomentumCVG_env/runs/sprint007_d1_20260903T013933Z/`
- D2A: `C:/MomentumCVG_env/runs/sprint007_d2a_20260904T043124Z/`
- D2B: `C:/MomentumCVG_env/runs/sprint007_d2b_20260904T045019Z/`
- D3: `C:/MomentumCVG_env/runs/sprint007_d3_20260906T024837Z/`
