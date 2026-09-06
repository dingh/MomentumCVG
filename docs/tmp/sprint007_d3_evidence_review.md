# Sprint 007 D3 — Evidence Review

**Date:** 2026-09-05  
**Repo HEAD at execution:** `545074042d0c9cdbbd88b1a0504fc2d224d2246b`  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint007_d3_20260906T024837Z/` (outside repo)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`  
**Window:** `2020-01-01` → `2026-07-10`  
**Notebook:** `notebooks/sprint007/d3_execution_envelope.ipynb` — committed copy remains unexecuted; one fresh `momentumcvg` kernel produced the executed `.ipynb` and `.html` in the evidence dir.  
**Status:** **Accepted.** No `h_req`. No preferred fill. D4 closeout: `EXECUTION_CALIBRATION_REQUIRED` ([`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md)).

---

## Execution identity

| Item | Value |
|---|---|
| Implementation commit | `5450740` (`545074042d0c9cdbbd88b1a0504fc2d224d2246b`) |
| Sprint 006 execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| D0 prerequisite | `READY_WITH_NARROW_ENABLING_CHANGE` (all gates passed) |
| D1 prerequisite | `D1_CONTINUE_TO_D2` |
| D2 prerequisite | final class `D3_EXECUTION_FOCUSED` |
| Official book | 9,212 included trades; 341 traded dates |
| Wall time | 412 s (~6.9 min); 325 unique `(path, h)` evaluations |
| Executed notebook | `d3_execution_envelope.executed.ipynb` |
| HTML export | `d3_execution_envelope.html` |
| Tables | `d3_envelope.json`, `d3_curves.csv`, `d3_crossings.csv`, `d3_side_snapshot.csv` |
| Receipt | `execution_receipt.json` |
| Executed notebook SHA256 | `e65c218b29216662d28b6a9618f34ffcc36ae4109510a2747adc3a7ef3dbcfba` |
| HTML SHA256 | `ed6f833ade1c6e664e81bfee8ce76c5c5742cfad5832550fd05ab8fe5a4fb357` |
| Verdict | `D3_ENVELOPE` (not blocked) |

### Tests run before this official export

```
pytest tests/unit/test_sprint007_d3_execution_envelope.py -q
29 passed in 8.20s

pytest -q
1687 passed, 1 skipped in 61.44s
```

---

## Endpoint reconciliation

All required identities passed. Path F \(h=1\) CAR was **not** checked against official cross CAR.

| Metric | Recomputed | Reference | Δ | Tolerance | Pass |
|---|---|---|---|---|---|
| `F_h0_pnl` | 159283.22635664625 | 159283.22635664625 | 0.0 | $0.01 | yes |
| `F_h1_pnl` | −184084.68947301456 | −184084.68947301456 | 0.0 | $0.01 | yes |
| `R_h0_pnl` | 159283.22635664628 | 159283.22635664625 | 2.91×10⁻¹¹ | $0.01 | yes |
| `R_h1_pnl` | −163272.83609432433 | −163272.83609432433 | 0.0 | $0.01 | yes |
| `F_h0_car` | 0.023961666655473095 | 0.023961666655473095 | 0.0 | 1e−9 | yes |
| `R_h0_car` | 0.02396166665547309 | 0.023961666655473095 | −6.94×10⁻¹⁸ | 1e−9 | yes |
| `R_h1_car` | −0.027083577148278994 | −0.027083577148278994 | 0.0 | 1e−9 | yes |
| `R_h0_Q` (worst \|ΔQ\|) | 3.18×10⁻¹² | 0.0 | 3.18×10⁻¹² | 7.14×10⁻⁶ | yes |
| `R_h1_Q` (worst \|ΔQ\|) | 9.09×10⁻¹³ | 0.0 | 9.09×10⁻¹³ | 6.45×10⁻⁶ | yes |

Every `H_vis` row has `n_trades=9212` and `n_dates=341`.

---

## Path R envelope (primary)

Targets use \(M = P_{\mathrm{mid}} = 159283.23\).

| Mark | \(h\) | Target | Method |
|---|---|---|---|
| \(h_{R,50}\) | 0.22359375 | \(P_R \le 79641.61\) | `H_det` + midpoint miss-check + bisection |
| \(h_{R,25}\) | 0.340703125 | \(P_R \le 39820.81\) | same |
| \(h_{R,P0}\) | 0.46156250 | \(P_R \le 0\) | same |
| \(h_{R,\mathrm{CAR}0}\) | 0.454609375 | Path R View A CAR \(\le 0\) | same (companion) |

All four Path R marks lie strictly inside \((0,1)\). There is no `h_req` and no single selected fill.

---

## Path F diagnostics and resize gaps

Path F does not bind Path R.

| Mark | Path F \(h\) | Method | \(h_R - h_F\) |
|---|---|---|---|
| 50% of \(M\) | 0.23194250 | closed form | −0.008349 |
| 25% of \(M\) | 0.34791375 | closed form | −0.007211 |
| dollar break-even | 0.46388500 | closed form | −0.002323 |
| CAR break-even | 0.45570313 | bracket-bisection | −0.001094 |

Negative \(h_R - h_F\) means resizing reaches each threshold at a **slightly smaller** \(h\) than frozen \(Q_{\mathrm{mid}}\). The gaps are small. Path R remains the envelope.

---

## Headroom and distances to full cross

| Quantity | Value | Meaning |
|---|---|---|
| `headroom_50_to_25` | 0.117109 | additional \(h\) between 50% and 25% of midpoint P&L |
| `headroom_25_to_P0` | 0.120859 | additional \(h\) between 25% and dollar break-even |
| `distance_P0_to_cross` | 0.538437 | how far \(h=1\) lies beyond dollar break-even; **not** headroom |
| `distance_CAR0_to_cross` | 0.545391 | how far \(h=1\) lies beyond Path R CAR break-even; **not** headroom |

Both headroom gaps are positive. Distances to full cross are large because break-even is near \(h \approx 0.46\), not because unused commission capacity remains after 25%.

---

## Long / short P&L at exact Path R marks

Path R only. Descriptive. Not a side-only strategy.

| Mark | \(h\) | Long P&L | Short P&L | Book P&L |
|---|---|---|---|---|
| \(h=0\) | 0.000000 | 101326.38 | 57956.84 | 159283.23 |
| \(h_{R,50}\) | 0.223594 | 70532.98 | 9099.84 | 79632.82 |
| \(h_{R,25}\) | 0.340703 | 55498.59 | −15693.69 | 39804.90 |
| \(h_{R,P0}\) | 0.461563 | 40717.95 | −40742.14 | −24.19 |
| \(h=1\) | 1.000000 | −16992.99 | −146279.85 | −163272.84 |

Short-side Path R P&L is already negative by \(h_{R,25}\). Book-level dollar break-even is later because longs are still positive.

---

## Monotonicity and selected curve values

Path R P&L and Path R View A mean cycle CAR are both `monotonic_nonincreasing` on `H_det` (true). First-adverse roots are therefore also the only roots on that grid.

`H_vis` is the 21-point chart grid, not the root. Every row has 9,212 trades and 341 dates.

| \(h\) | \(P_R\) | CAR\(_R\) | \(P_F\) | CAR\(_F\) | \(\sum\|Q\|_R\) | capital\(_R\) |
|---|---|---|---|---|---|---|
| 0.00 | 159283.23 | 0.023962 | 159283.23 | 0.023962 | 3,533,108 | 6,502,881 |
| 0.20 | 87809.46 | 0.013252 | 90609.64 | 0.013293 | 3,457,799 | 6,416,620 |
| 0.25 | 70542.97 | 0.010616 | 73441.25 | 0.010663 | 3,439,478 | 6,395,570 |
| 0.35 | 36695.98 | 0.005394 | 39104.46 | 0.005448 | 3,403,414 | 6,354,057 |
| 0.45 | 3731.98 | 0.000235 | 4767.66 | 0.000291 | 3,368,096 | 6,313,304 |
| 0.50 | −12430.17 | −0.002322 | −12400.73 | −0.002266 | 3,350,708 | 6,293,203 |
| 1.00 | −163272.84 | −0.027084 | −184084.69 | −0.027086 | 3,186,035 | 6,101,558 |

---

## Required execution quality

On the resized book, \(h < 0.2236\) retains at least 50% of midpoint dollar P&L; \(h < 0.3407\) retains at least 25%; \(h < 0.4616\) remains dollar-profitable. Path R CAR first reaches zero at \(h = 0.4546\) (companion).

Path R `headroom_50_to_25` is 0.1171 and `headroom_25_to_P0` is 0.1209 (positive = additional execution-cost capacity between those marks). Separately, `distance_P0_to_cross` is 0.5384 and `distance_CAR0_to_cross` is 0.5454; those distances say how far full cross lies beyond break-even and are not headroom.

Unknown: whether any package order would fill inside that envelope; commissions, missed fills, timing, and adverse selection; counterfactual structures. Historical end-of-day quotes do not validate complex-order execution.

Not claimed: that midpoint is attainable; that any one of \(h_{R,50}\), \(h_{R,25}\), or \(h_{R,P0}\) is a live limit price; that Path F is the executable book; or that a filter / side / structure change would preserve the midpoint book.

The working-plan shape is a requirement strictly between midpoint and full cross, recorded here as a range. D4 assigned `EXECUTION_CALIBRATION_REQUIRED`. This review does not claim the required \(h\) is achievable.

---

## Limits

- This states **required** interpolated execution quality for the frozen 9,212-trade book. It does not state real-world attainability.
- \(h\) is a mechanical fraction of the quoted half-spread, not expected live execution.
- Path F remaining positive is not executable return.
- Side dollars are not a long-only or short-only test.
- No filter, alternative structure, commission model, fill probability, or `SurfaceRunner` was added.
- Forbidden language is absent: recoverable, likely fillable, patient execution would capture midpoint, historical quotes imply attainable package fills.

---

## Stop

D3 evidence is **accepted**. Sprint 007 is closed at [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md). The next sprint is unauthorized.
