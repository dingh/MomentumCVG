# Sprint 007 D2B — Evidence Review

**Date:** 2026-09-03  
**Repo HEAD:** `ab53e26c13abf2f98f791651c27553d8d9b81c7c`  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint007_d2b_20260904T045019Z/` (outside repo)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`  
**Window:** `2020-01-01` → `2026-07-10`  
**Notebook:** `notebooks/sprint007/d2b_package_tradability.ipynb` — committed copy remains unexecuted; a fresh `momentumcvg` kernel ran a JSON-valid copy through the D2B sections. Executed `.ipynb` and `.html` are in the evidence dir.  
**Status:** **Accepted.** Final class `D3_EXECUTION_FOCUSED`. No threshold search. No filtered book. D3 not started.

---

## Execution identity

| Item | Value |
|---|---|
| Implementation commit | `ab53e26` (`ab53e26c13abf2f98f791651c27553d8d9b81c7c`) |
| Sprint 006 execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| D2A prerequisite | accepted; branch `package_tradability`; provisional class `D3_EXECUTION_FOCUSED` |
| Executed notebook | `d2b_package_tradability.executed.ipynb` |
| HTML export | `d2b_package_tradability.html` |
| Tables | `d2b_tradability.json`, `d2b_group_terciles.csv`, `d2b_book_terciles.csv` |
| Receipt | `execution_receipt.json` |
| Threshold search | no |
| Filtered strategy | no |

### Focused tests

```
pytest tests/unit/test_sprint007_d2_shortfall_bridge.py tests/unit/test_sprint007_d2b_package_tradability.py -q
28 passed in 0.40s
```

---

## Reconciliations

All three D2B checks passed against accepted D2A \(\Delta_{\mathrm{price}}\) and \(P_{mid}\).

| Metric | Recomputed | Reference | Δ | Tolerance | Pass |
|---|---|---|---|---|---|
| per-trade \(\Delta_{\mathrm{price}}\) | −343367.9158296608 | −343367.91582966084 | 5.82×10⁻¹¹ | $0.01 | yes |
| per-trade \(P_{mid}\) | 159283.22635664628 | 159283.22635664625 | 2.91×10⁻¹¹ | $0.01 | yes |
| tercile + skipped \(\Delta_{\mathrm{price}}\) | −343367.9158296608 | −343367.91582966084 | 5.82×10⁻¹¹ | $0.01 | yes |

Shared mid/cross quotes were used. No D2B blocker. Verdict is **not** `D2_BLOCKED`.

---

## Ranked and skipped trades

| Count | n |
|---|---|
| Intersection keys | 9,212 |
| Ranked (nonzero midpoint cashflow) | 9,212 |
| Skipped zero-cashflow | 0 |

No zero-cashflow packages were disclosed because none occurred.

---

## Group terciles (`direction`, `instrument_type`)

Tercile 3 is expensive **within that group**.

| Direction | Instrument | Tercile | n | Mid P&L | \(\Delta_{\mathrm{price}}\) |
|---|---|---|---|---|---|
| long | long_straddle | 1 | 1964 | 29881.52 | −15734.75 |
| long | long_straddle | 2 | 1963 | 36666.48 | −32148.45 |
| long | long_straddle | 3 | 1963 | 34778.38 | −77061.99 |
| short | iron_fly | 1 | 1108 | 9245.85 | −34287.82 |
| short | iron_fly | 2 | 1107 | 29241.12 | −63107.00 |
| short | iron_fly | 3 | 1107 | 19469.87 | −121027.91 |

Within-group share of that side’s \(\Delta_{\mathrm{price}}\) in tercile 3: long 61.7%; short 55.4%.

---

## Book-level terciles

Within-group ranks stacked. Accepted D2A \(\Delta_{\mathrm{price}} = -343367.92\).

| Tercile | n | Mid P&L | \(\Delta_{\mathrm{price}}\) | \(\lvert\Delta_{\mathrm{price}}\rvert / \lvert\Delta_{\mathrm{price,book}}\rvert\) |
|---|---|---|---|---|
| 1 (cheap) | 3072 | 39127.37 | −50022.56 | 14.6% |
| 2 | 3070 | 65907.60 | −95255.45 | 27.7% |
| 3 (expensive) | 3070 | 54248.26 | −198089.90 | **57.7%** |

- Expensive-tercile share of \(\Delta_{\mathrm{price}}\): **0.576903**
- Midpoint P&L outside expensive (T1+T2): **105034.97**
- Positive mid margin remains outside the expensive tercile: **yes**

Tercile 3 itself also has positive midpoint P&L (+54,248).

---

## Concentrated or diffuse?

The expensive tercile is **33.3% of trades** (3,070 of 9,212) and contains **57.7%** of \(\lvert\Delta_{\mathrm{price}}\rvert\). That is **meaningful but not overwhelming** concentration. T1+T2 retain **+$105,035** of midpoint P&L.

This **supports a selective-friction hypothesis for D4**. It does **not** validate a filter, a cutoff, or a profitable implementation. Branch 3 does not set `structure`. The 70% side-concentration rule is a D2A/D3 `structure` predicate only; it is **not** a tercile gate.

---

## Final D3 class

Exactly one class, from frozen precedence  
`D2_BLOCKED → D3_MIXED_MECHANISM → D3_SIZING_AWARE → D3_STRUCTURE_CONDITIONED → D3_EXECUTION_FOCUSED`:

| Predicate | Value |
|---|---|
| `blocked` | false |
| `mixed` (price and size material, or order-sensitive) | false |
| `sizing` | false |
| `structure` (one side ≥ 70% of \(\Delta_{\mathrm{price}}\)) | false |

**Final class: `D3_EXECUTION_FOCUSED`**

This is calculated, not hardcoded. It matches the D2A provisional class. D3 is not designed here.

---

## Conclusion

**Supports:** Direct entry-price concession remains the D2A-dominant mechanism. 33.3% of trades contain 57.7% of price concession, and T1+T2 retain +$105,035 of midpoint P&L. That is enough to carry a **selective-friction hypothesis into D4**.

**Does not prove:** A filter, a threshold, or a profitable implementation. No trade was removed, no cutoff was searched, and no alternative book was priced. Midpoint fills are not shown to be attainable.

**Unknown:** Package fill probability; whether the book-level execution requirement implied by `D3_EXECUTION_FOCUSED` is attainable; commissions and other unmodeled friction; counterfactual structures.

---

## Stop

D2 is **accepted**. D3 **design** is authorized. Do not implement or execute D3 until that design is accepted.
