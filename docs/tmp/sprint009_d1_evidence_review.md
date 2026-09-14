# Sprint 009 D1 — evidence review

**Status:** `EXECUTED — EVIDENCE AWAITING REVIEW`  
**Executed:** 2026-09-14  
**Design:** [`sprint009_d1_design.md`](sprint009_d1_design.md) — **ACCEPTED** at `e109a9e`. Formulas unchanged.  
**Implementation:** `0d63293d80fda7bede869b9205880c94906d25d4`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z/`  
**Input:** accepted D0 panel `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/` (`004ba80`, verdict `READY`). The superseded D0 directory was not read.  
**Characterization:** Development-history attribution of the frozen short iron fly. Not a trading decision, not a protection study, and not a filter result. This file does not accept the evidence.

---

## Verdict

`READY`

Every gate passed. No check was weakened. A profitable book was not required. The development official cross result is negative. That does not fail the run.

The primary-window anchor, −$146,279.85, is not this result and is not the residual reference.

---

## Invocation

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d1_decomposition.py
```

| Item | Value |
|---|---|
| Code revision | `0d63293d80fda7bede869b9205880c94906d25d4` |
| D0 directory | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z` |
| D0 code SHA | `004ba80052f6586f6a230ad207e8151e695e156e` |
| Generated | `2026-09-14T00:15:04.592974+00:00` |
| Output directory | `C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z` |
| D0 directory writes | none |

Input hashes of the files read:

| File | SHA-256 |
|---|---|
| `matched_short_iron_flies.parquet` | `80974323d48e5bc8133122031d6b8da7cf9a787aa379d9778191d3d086547451` |
| `short_calendar.parquet` | `ae1cd74c0b4ebf7f211d7303a83c91dd2c552d39e24543fd60a612599042dcca` |

Generated tables, JSON, the Markdown report, and `waterfall_development.png` stay in that external directory. They are not in the repository.

---

## Tests

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint009_d1_body_wing_decomposition.py -q
```

6 passed before the official run. Coverage is hand-calculated: sold-body cash sign, concession formulas, appendix identity, quantity ×100 against official $6.00, signed wing payoff used once, a zero body-credit trade kept with a null ratio, a verified zero-short date retained at zero dollars, later-period isolation, a stored-mid discrepancy that still passes, and a saved-P&L mismatch that stays `BLOCKED`. A valid control uses the same `readiness_verdict` as the runner.

---

## Coverage and gates

| Check | Result |
|---|---|
| Development trades | 2,087, all `window_label = development`, all `input_ok` |
| Development dates | 209: 208 `verified_positive_short`, one `verified_zero_short` |
| Zero-short date | `2020-03-13`: 0 trades, every dollar field 0, ratio reason `zero body midpoint credit` |
| Pairing failures | 0 |
| Null trade-level ratios | 0 |
| Later-period rows in the decomposition tables | 0 |
| Stored vs arithmetic mid | 0 legs above 1e-6. Maximum absolute difference 4.55e-13. Diagnostic only; not a gate |

| Gate | Result |
|---|---|
| Provenance | PASS. Coverage matches the accepted D0 evidence |
| Inputs | PASS |
| Body cross | PASS |
| Wing cross | PASS |
| Identity | PASS |
| Midpoint | PASS |
| Calendar | PASS. Zero-short date retained |
| Scope | PASS. No later-period economic field |

Largest absolute trade residual versus saved D0 is 1.01e-11, inside max($0.01, 1e-9 × |official|). Development identity residual versus the development sum of `pnl_cross_official` is 1.46e-11.

---

## Five-term development dollars

Development official cross P&L, the sum of `pnl_cross_official` on the 2,087 trades: **−$69,776.72**.

| Term | Dollars |
|---|---:|
| \(B_{\mathrm{mid}}\) body midpoint P&L | 120,527.85 |
| \(H_{\mathrm{body}}\) body execution concession | 83,182.15 |
| \(P_{\mathrm{body,cross}} = B_{\mathrm{mid}} - H_{\mathrm{body}}\) | 37,345.69 |
| \(W_{\mathrm{mid}}\) wing midpoint premium | 362,954.17 |
| \(H_{\mathrm{wing}}\) wing execution concession | 38,384.96 |
| \(W_{\mathrm{pay}}\) gross wing expiry payout | 294,216.71 |
| \(P_{\mathrm{fly,cross}}\) | −69,776.72 |
| \(C_{\mathrm{body}}\) body midpoint credit | 2,129,831.44 |

Fees = 0. Concession is the quote gap already embedded in the cross fill. It is not deducted a second time. Midpoint fills are not claimed attainable. \(W_{\mathrm{pay}}\) is gross expiry payout, not net protection value.

Ratios are ratios of summed dollars. The denominator is body midpoint credit, not capital at risk.

| Ratio | Value |
|---|---:|
| \(\sum H_{\mathrm{body}} / \sum C_{\mathrm{body}}\) | 0.0391 |
| \(\sum W_{\mathrm{mid}} / \sum C_{\mathrm{body}}\) | 0.1704 |
| \(\sum H_{\mathrm{wing}} / \sum C_{\mathrm{body}}\) | 0.0180 |
| Descriptive \(\sum H_{\mathrm{wing}} / \sum W_{\mathrm{mid}}\) | 0.1058 — not a headline |

---

## Answers

**D1-A.** Development body midpoint P&L is positive, about $120,528. After body execution concession of about $83,182, body cross P&L is still positive, about $37,346. Concession is about 3.9% of body midpoint credit. That remaining body margin is not a trading decision and is not true in every year: 2020 and 2021 body cross P&L are negative.

**D1-B.** The wings’ midpoint premium, about $362,954, is much larger than the wing execution concession, about $38,385. Against body midpoint credit those burdens are 17.0% and 1.8%. The descriptive wing spread is about 10.6% of wing midpoint premium. The dollar cost of the wings is mostly the midpoint premium, not the concession.

**D1-C.** The five terms explain the development iron-fly result. Identity residual versus −$69,776.72 is 1.46e-11. Net wing contribution, \(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}}\), is about −$107,122. That more than offsets the positive body cross result. Every development year has a negative official cross result: 2020 −$17,203, 2021 −$36,644, 2022 −$601, 2023 −$15,328. 2021 is the largest loss and also has a negative body cross result. 2022’s body cross result is about +$39,865 and nearly offsets that year’s wing drag. This is a description of the reconciled terms. It is not a finding that wings should be dropped and not a filter result.

---

## What this does not do

- It does not accept this evidence.
- It does not start D2–D5.
- It does not compute later-period economics.
- It does not change signals, wings, sizing, or the frozen contract.
- It does not measure path risk, payout frequency, or the value of protection.
- It does not claim midpoint fills are attainable.
