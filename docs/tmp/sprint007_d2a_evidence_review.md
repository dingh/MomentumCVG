# Sprint 007 D2A — Evidence Review

**Date:** 2026-09-03  
**Repo HEAD:** `fe5c6c0592fe28286c19b140340bb15d079640dc`  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint007_d2a_20260904T043124Z/` (outside repo)  
**Official artifacts:** `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`  
**Window:** `2020-01-01` → `2026-07-10`  
**Notebook:** `notebooks/sprint007/d2_shortfall_bridge.ipynb` — committed copy remains unexecuted; a fresh-kernel copy was executed through Section 7 (`momentumcvg`) and exported to `.ipynb` and `.html` in the evidence dir.  
**Status:** Awaiting human checkpoint. D2B not run. No final D3 class.

---

## Execution identity

| Item | Value |
|---|---|
| Implementation commit | `fe5c6c0` (`fe5c6c0592fe28286c19b140340bb15d079640dc`) |
| Sprint 006 execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| D0 prerequisite | `READY_WITH_NARROW_ENABLING_CHANGE` (all gates passed) |
| D1 prerequisite | `D1_CONTINUE_TO_D2` |
| Executed notebook | `d2_shortfall_bridge.executed.ipynb` |
| HTML export | `d2_shortfall_bridge.html` |
| Receipt | `execution_receipt.json` |
| D2B executed | no |
| Final D3 class | none |

### Focused tests

```
pytest tests/unit/test_sprint007_d2_shortfall_bridge.py -q
16 passed in 0.28s
```

---

## Reconciliation

All six D2A checks passed. Residual is inside `max($0.01, 1e-9 × (|P_mid| + |P_cross|))`. Unmatched keys = 0.

| Metric | Recomputed | Reference | Δ | Tolerance | Pass |
|---|---|---|---|---|---|
| `P_mid` | 159283.22635664625 | 159283.22635664628 | −2.91×10⁻¹¹ | $0.01 | yes |
| `P_cross` | −163272.83609432433 | −163272.83609432433 | 0.0 | $0.01 | yes |
| residual `R` | 5.82×10⁻¹¹ | 0.0 | 5.82×10⁻¹¹ | $0.01 | yes |
| `Δ_set` keys | 0 | 0 | 0 | exact | yes |
| `Δ_set` dollars | 0.0 | 0.0 | 0.0 | $0.01 | yes |
| interaction identity `\|I\| = \|Δ_price − Δ_price_Paasche\|` | 41462.58394320729 | 41462.58394320728 | 7.28×10⁻¹² | $0.01 | yes |

Included intersection keys = 9,212; mid-only = 0; cross-only = 0.

---

## Aggregate bridge (Laspeyres / \(Q_{mid}\))

Identity \(G = \Delta_{\mathrm{price}} + \Delta_{\mathrm{size}} + \Delta_{\mathrm{set}} + R\) holds.

| Term | Dollars |
|---|---|
| \(P_{mid}\) | 159283.22635664625 |
| \(P(Q_{mid}, p_{\mathrm{cross}})\) | −184084.68947301456 |
| \(P_{cross}\) | −163272.83609432433 |
| \(G = P_{cross} - P_{mid}\) | −322556.0624509706 |
| \(\Delta_{\mathrm{price}}\) | −343367.91582966084 |
| \(\Delta_{\mathrm{size}}\) | 20811.853378690226 |
| \(\Delta_{\mathrm{set}}\) | 0.0 |
| residual \(R\) | 5.820766091346741×10⁻¹¹ |

### Shares of \(\lvert G\rvert\) (frozen materiality ≥ 25%)

| Component | \(\lvert\Delta\rvert / \lvert G\rvert\) | Material? |
|---|---|---|
| \(\Delta_{\mathrm{price}}\) | 1.064521 (106.5%) | yes |
| \(\Delta_{\mathrm{size}}\) | 0.064521 (6.45%) | no |

Dominant term: **price**. \(\Delta_{\mathrm{size}} > 0\): at cross unit P&L, the cross size vector lost less than the mid size vector, so sizing **offsets** part of the entry concession (that is why the price share exceeds 100% of \(G\)).

---

## Long / short dollar decomposition

| Side | \(P_{mid}\) | \(P_{cross}\) | \(G\) | \(\Delta_{\mathrm{price}}\) | \(\Delta_{\mathrm{size}}\) | \(\Delta_{\mathrm{set}}\) | \(R\) | n |
|---|---|---|---|---|---|---|---|---|
| long | 101326.38 | −16992.99 | −118319.37 | −124945.19 | 6625.82 | 0.0 | ~0 | 5890 |
| short | 57956.84 | −146279.85 | −204236.69 | −218422.73 | 14186.04 | 0.0 | 0.0 | 3322 |

Share of aggregate \(\Delta_{\mathrm{price}}\): short 63.6%; long 36.4%. Neither side reaches 70%, so `structure = false`.

---

## Yearly dollar decomposition

Not a return/CAR table.

| Year | \(P_{mid}\) | \(P_{cross}\) | \(G\) | \(\Delta_{\mathrm{price}}\) | \(\Delta_{\mathrm{size}}\) | n |
|---|---|---|---|---|---|---|
| 2020 | 85544.71 | 14895.49 | −70649.22 | −65705.07 | −4944.15 | 1226 |
| 2021 | −8360.62 | −52362.49 | −44001.87 | −52417.91 | 8416.03 | 1400 |
| 2022 | 40000.25 | −6683.04 | −46683.29 | −48521.54 | 1838.25 | 1621 |
| 2023 | 26052.24 | −12483.54 | −38535.78 | −39884.98 | 1349.19 | 1425 |
| 2024 | 21891.85 | −19505.43 | −41397.28 | −42169.15 | 771.87 | 1417 |
| 2025 | 2295.97 | −42941.76 | −45237.73 | −51399.74 | 6162.01 | 1350 |
| 2026 | −8141.17 | −44192.06 | −36050.88 | −43269.53 | 7218.65 | 773 |

\(\Delta_{\mathrm{price}}\) is negative in every calendar year. \(\Delta_{\mathrm{size}}\) is negative only in 2020.

---

## Order-sensitivity

| Statistic | Value |
|---|---|
| \(S_{\mathrm{order}} = \lvert I\rvert / \lvert G\rvert\) | 0.128544 |
| \(I = \sum \Delta p \cdot \Delta Q\) | 41462.58394320729 |
| \(\Delta_{\mathrm{price,Paasche}}\) | −301905.33188645355 |
| \(\Delta_{\mathrm{size,dual}}\) | −20650.730564517056 |
| Dual price material? | yes (\(\lvert-301905.33\rvert / \lvert G\rvert = 93.6\%\)) |
| Dual size material? | no (\(\lvert-20650.73\rvert / \lvert G\rvert = 6.40\%\)) |
| Dual dominant | price |
| `order_sensitive` | **false** |

\(S_{\mathrm{order}} > 0.10\) but the dual order does **not** change materiality or dominance, so it does not trigger `mixed` and does not change the D2B branch.

---

## Checkpoint (rules, not a final decision)

| Item | Result |
|---|---|
| D2B branch | `package_tradability` |
| Provisional D3 class | `D3_EXECUTION_FOCUSED` |
| Final D3 class | **not assigned** |

Rule path: not blocked; \(\Delta_{\mathrm{size}}\) not material → not sizing-chain; \(\Delta_{\mathrm{price}}\) dominant and no side ≥ 70% of \(\Delta_{\mathrm{price}}\) → tradability branch; not mixed; not sizing; not structure → `D3_EXECUTION_FOCUSED`.

---

## Conclusion

The dollar identity **supports** direct entry-price concession as the dominant mechanism: \(\Delta_{\mathrm{price}} = -\$343{,}367.92\) is 106.5% of \(\lvert G\rvert\), material, and larger than \(\Delta_{\mathrm{size}}\).

The same identity **weakens** a trade-set explanation (\(\Delta_{\mathrm{set}} = 0\), 9,212 matched keys), fill-dependent sizing as a material loss channel (\(\Delta_{\mathrm{size}}\) is +\$20,811.85, 6.45% of \(\lvert G\rvert\), and not material), and side/structure concentration (short is 63.6% of \(\Delta_{\mathrm{price}}\), below 70%).

**Unknown:** package fill probability; whether any required execution quality is attainable; counterfactual structures; and any D2B refinement of the tradability slice. Midpoint remaining positive is not executable return.

**Provisional D3 class:** `D3_EXECUTION_FOCUSED`. **Selected D2B branch:** `package_tradability`. The final class is written only after D2B.

---

## Stop

Human review of this D2A checkpoint is required before D2B, a final D3 class, or D3 design.
