# Sprint 007 D1 — Evidence Review

**Date:** 2026-09-03  
**Repo HEAD:** `b39985fc172a61107cf8745fdeb593ffbbdf9aeb`  
**Review run dir:** `C:/MomentumCVG_env/runs/sprint007_d1_review_20260902T190148Z/`  
**Implementation evidence:** `C:/MomentumCVG_env/runs/sprint007_d1_20260903T013933Z/` (commit `516235f`)  
**Notebook:** `notebooks/sprint007/d1_gross_margin.ipynb` — executed from fresh `momentumcvg` kernel; no code, gates, or assumptions changed.  
**Status:** Awaiting acceptance. D2 design not authorized until D1 is accepted.

---

## D0 prerequisite

Verdict: `READY_WITH_NARROW_ENABLING_CHANGE` — all 8 gates passed.

| Gate | Result | Detail |
|---|---|---|
| G1 | PASS | inventory=17; all receipt hashes matched |
| G1b_mid | PASS | n_expected=403; n_failed=0; n_valid_no_trade=0 |
| G1b_cross | PASS | n_expected=403; n_failed=0; n_valid_no_trade=0 |
| G2 | PASS | required columns present |
| G3 | PASS | unique trade keys per fill |
| G4 | PASS | primary included keys=9,212; mid-only=0; cross-only=0 |
| G5 | PASS | no unit-quantity, fill, quote, settlement, or exit mismatches |
| G7 | PASS | scope limited to readiness checks |

---

## Reconciliation

All five checks passed against `decision_report.json → by_fill.mid.primary`.

| Metric | Recomputed | Reference | Δ | Tolerance | Pass |
|---|---|---|---|---|---|
| `total_pnl` | 159283.22635664625 | 159283.22635664628 | −2.91×10⁻¹¹ | $0.01 | yes |
| `view_a_mean_cycle_car` | 0.023961666655473095 | 0.023961666655473095 | 0.0 | 1e−9 | yes |
| `n_included_trades` | 9212 | 9212 | 0 | exact | yes |
| `n_traded_dates` | 341 | 341 | 0 | exact | yes |
| `expected_included_trades` | 9212 | 9212 | 0 | exact | yes |

---

## Four D1 gates

All four frozen parts passed. Formulas are fixed in [`docs/tmp/sprint007_d1_design.md`](sprint007_d1_design.md) and were not changed.

| Gate | Pass | Exact values |
|---|---|---|
| **G-Sign** | yes | `total_pnl` = 159283.22635664625; View A mean cycle CAR = 0.023961666655473095 |
| **G-Breadth** | yes | excl top-5 dates = 95719.47118030259; excl top-5 tickers = 108224.41308642241 |
| **G-Location** | yes | qualifying_sides = [long, short]; both have positive P&L and ≥10% trade share |
| **G-Stability** | yes | positive years = 5; best year = 2020; P&L excl best year = 73738.51724225555 |

Excluded dates: 2020-02-21, 2020-11-06, 2020-03-06, 2021-03-05, 2020-10-23  
Excluded tickers: AMC, FDX, GME, DLTR, VZ

---

## Long / short attribution

| Side | P&L ($) | Trades | Share | Capital at risk ($) |
|---|---|---|---|---|
| long (straddle) | 101326.38183966067 | 5890 | 63.9% | 3102880.76 |
| short (iron fly) | 57956.84451698560 | 3322 | 36.1% | 3400000.00 |

---

## Yearly P&L

| Year | P&L ($) | Trades |
|---|---|---|
| 2020 | 85544.71 | 1226 |
| 2021 | −8360.62 | 1400 |
| 2022 | 40000.25 | 1621 |
| 2023 | 26052.24 | 1425 |
| 2024 | 21891.85 | 1417 |
| 2025 | 2295.97 | 1350 |
| 2026 | −8141.17 | 773 |

**P&L excluding best year (2020):** $73,738.52

---

## Portfolio metrics (primary window)

| Metric | Value |
|---|---|
| Total P&L | $159,283.23 |
| n included trades | 9,212 |
| n traded dates | 341 |
| View A mean cycle CAR | +2.396% |
| View A annualized Sharpe | 1.046 |
| View A max drawdown | −84.7% |
| Total capital at risk | $6,502,881 |

---

## Focused tests

```
pytest tests/unit/test_sprint007_d1_gross_margin.py tests/unit/test_sprint007_artifact_validation.py -q
33 passed in 1.11s
```

Tests include: sign pass/fail, breadth date/ticker exclusions, side 10% threshold, two-year stability, best-year exclusion, reconciliation tolerance, and regression confirming a failed D0 cannot produce `D1_CONTINUE_TO_D2`.

---

## Verdict

**`D1_CONTINUE_TO_D2`**

All four frozen gate parts passed. This is the rule-based sprint gate result, not D1 acceptance.

---

## Conclusion

The frozen `42:8` midpoint book earns about **$159k** over the primary window (2020-01-01 → 2026-07-10) with a **+2.4% mean cycle CAR**. That profit is broad: removing the five best dates or the five best tickers each still leaves more than $95k. **Both sides are profitable**: long straddles contribute $101k (64% of trades), short iron flies $58k (36%). Margin is concentrated in 2020, but five of seven calendar years are positive, and dropping 2020 entirely still leaves $74k.

This is sufficient **optimistic gross-expression margin** to justify D2: reconciling the full dollar bridge between midpoint and full-cross economics and attributing the dominant shortfall mechanism. It does not mean the expression is tradable at midpoint. The mid→cross gap, execution assumptions, filters, and alternative structures are out of scope for D1 and belong to D2–D3.

---

## Limits of this evidence

- Midpoint is an **optimistic gross-expression reference**, not expected executable return.
- This measures the frozen `42:8` selection + CVG rule + current long/short expression + current holding period + midpoint sizing. It is **not** pure Momentum/CVG signal quality.
- No spread/liquidity filter, subgroup winner, or alternative structure was evaluated or selected here.
- A `D1_CONTINUE_TO_D2` verdict clears the sprint gate to diagnose **why** cross economics diverge; it does not validate the expression or claim midpoint fills are attainable.
