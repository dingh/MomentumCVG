# Sprint 007 D4 — Diagnosis, next action, and closeout

**Status:** `ACCEPTED`  
**Updated:** 2026-09-06  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md) §6.7, §11  
**Closeout:** [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md)

D0–D3 are **accepted**. D4 synthesizes those results only. No new P&L, no code, no filter, no strategy change.

---

## Question

> Which explanation best fits D1–D3, and what single next action is justified by the evidence?

---

## Frozen decisions

| Item | Decision |
|---|---|
| Primary outcome | `EXECUTION_CALIBRATION_REQUIRED` |
| Why this outcome | D1 continue gate passed; required Path R \(h\) lies strictly between midpoint and full cross; historical quotes cannot show package-fill attainability (working plan §6.7) |
| Path R envelope (accepted D3) | \(h < 0.2236\) retains 50% of midpoint P&L; \(h < 0.3407\) retains 25%; \(h < 0.4616\) remains dollar-profitable before unmodeled costs |
| Companion CAR | Path R CAR first reaches zero at \(h_{R,\mathrm{CAR}0} = 0.4546\) |
| Secondary findings | Expensive-package concentration and short-side fragility are disclosed only. They do not set a filter, a long-only book, or a second next action |
| Next action | One future **manual-first execution-observation** project. Not a redesign. Not live trading |
| Forbidden | Claiming the required \(h\) is achievable; selecting a cutoff; treating a quote touch as a fill; treating paper fills as live proof |

---

## Evidence consumed (do not reopen)

| Source | Accepted fact |
|---|---|
| D0 | Official artifacts support a trusted artifact-first path (`READY_WITH_NARROW_ENABLING_CHANGE`) |
| D1 | Mid-primary \(P_{\mathrm{mid}} = +159{,}283.23\); View A CAR \(= +2.396\%\); 9,212 trades; 341 dates; `D1_CONTINUE_TO_D2` |
| D2A | \(G = -322{,}556.06\); \(\Delta_{\mathrm{price}} = -343{,}367.92\) (dominant); \(\Delta_{\mathrm{size}} = +20{,}811.85\) (not material); residual ~0 |
| D2B | Final class `D3_EXECUTION_FOCUSED`; expensive tercile is 33.3% of trades and 57.7% of \(\lvert\Delta_{\mathrm{price}}\rvert\); T1+T2 mid P&L \(=+105{,}035\) |
| D3 | Path R envelope above; positive headroom 50→25 and 25→P0; attainability unknown |

Gross midpoint margin exists. Entry-price concession dominates the mid-to-cross gap. Sizing feedback is not material. Real package-fill attainability remains unknown.

---

## Why not the other outcomes

| Outcome | Why it is not primary |
|---|---|
| `SELECTIVE_FRICTION_HYPOTHESIS` | D2B concentration is real but secondary. It does not replace the book-level execution requirement between mid and cross |
| `STRUCTURE_OR_SIZING_HYPOTHESIS` | Short share of \(\Delta_{\mathrm{price}}\) is 63.6% (< 70%); \(\Delta_{\mathrm{size}}\) is not material |
| `CURRENT_IMPLEMENTATION_NOT_VIABLE` | D1 continue passed; Path R headroom 50→25 and 25→P0 is positive |
| `EVIDENCE_INCONCLUSIVE` | D0–D3 reconciliations passed; the remaining gap is attainability, which this outcome names |

---

## One next action

A **future** sprint, separately designed and authorized, should run a **manual-first execution-observation** project on the frozen selected book:

1. Export weekly order tickets (identity, structure, and preregistered hypothetical net limits).
2. Record, without placing live orders:
   - arrival package midpoint / natural price;
   - whether a preregistered hypothetical net limit was touched (touch / no-touch);
   - time-to-touch;
   - skipped observations;
   - post-touch quote movement.
3. After observation is defined, validate order plumbing with **paper** trading only.

A quote touch is **not** a fill. Paper trading validates order plumbing only. Real fill probability, time-to-fill, implementation shortfall, and adverse selection remain unresolved until separately authorized live-order observation is available.

Live orders are **not** authorized. This sprint does not add broker code, place orders, or implement the observer.

---

## Non-goals

- New P&L, `SurfaceRunner`, or official-artifact mutation
- Filter / long-only / wing / window search
- Selecting one \(h\) as a live limit
- Claiming midpoint or any Path R mark is attainable
- Treating a quote touch as a fill, or paper plumbing as live-order evidence
- Starting the next sprint in this commit

---

## Acceptance

- [x] One primary classification: `EXECUTION_CALIBRATION_REQUIRED`
- [x] Secondary findings disclosed without a second next action
- [x] Exactly one next-action specification
- [x] No new cure backtest
- [x] Closeout memo and closed agenda; next sprint unauthorized
