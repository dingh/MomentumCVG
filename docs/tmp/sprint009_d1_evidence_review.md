# Sprint 009 D1 — evidence review

**Status:** `REVIEWED / ACCEPTED` through `28f5ea4`  
**Executed:** 2026-09-14  
**Review annotation (2026-09-14):** Accepted through `28f5ea4`. Findings unchanged. Implementation remains `5669773`. Output remains `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/`. The earlier D1 run stays superseded. This annotation does not change the results below. It authorizes D2 planning only. It does not start D2 implementation.  
**Design:** [`sprint009_d1_design.md`](sprint009_d1_design.md) — **ACCEPTED** at `e109a9e`. Formulas, population, quantities, and windows unchanged.  
**Implementation:** `5669773f356f6c33cef86bd0da30ce4051709a6b`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/`  
**Supersedes:** `0d63293` / `C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z/`, previously recorded through `4a9a073`. That directory is not the current D1 record.  
**Input:** accepted D0 panel `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/` (`004ba80`, receipt verdict `READY`). The superseded D0 directory was not read. D0 and the baseline were not rerun.  
**Characterization:** Development-history attribution of the frozen short iron fly. Not a trading decision, not a protection study, and not a filter result. The review annotation above accepts this evidence. It does not reinterpret it as a protection result.

---

## Verdict

`READY`

Every gate passed. No reconciliation check was weakened. A profitable book was not required. The development official cross result is negative. That does not fail the run.

The primary-window anchor, −$146,279.85, is not this result and is not the residual reference.

---

## Correction from `0d63293`

The accepted design is unchanged. The correction does three things:

- A finite zero date-level numerator with a positive body-credit denominator stays a defined ratio of zero. Missing values are explicit, not truthiness. A zero-credit date stays null with a reason. Null date-ratio counts and reasons are reported.
- `pairing_ok` must be an actual true boolean. A missing value or a truthy string fails. Leg unit quantity must equal `+1` or `−1` exactly; a fractional value is not truncated before validation. Those failures reach `BLOCKED` through `readiness_verdict`.
- The official runner checks the accepted D0 directory, receipt code SHA `004ba80052f6586f6a230ad207e8151e695e156e`, and receipt verdict `READY` before accepting a result. A mismatch is a named blocker. Input hashes are still recorded. Stored-midpoint diagnostics are unchanged and are still not a gate.

---

## Comparison with the superseded run

Input hashes are unchanged.

| File | SHA-256 | Versus `sprint009_d1_20260914T001504Z` |
|---|---|---|
| `matched_short_iron_flies.parquet` | `80974323d48e5bc8133122031d6b8da7cf9a787aa379d9778191d3d086547451` | unchanged |
| `short_calendar.parquet` | `ae1cd74c0b4ebf7f211d7303a83c91dd2c552d39e24543fd60a612599042dcca` | unchanged |

Headline development dollar totals are unchanged, including `development_official_pnl` −$69,776.72498250673, the five components, body cross, iron-fly cross, and body midpoint credit. Annual dollar columns match exactly. On this panel, no date had a finite zero numerator with a positive denominator, so date-ratio null counts stayed at one: the zero-short date. The corrected report now records `null_date_ratio_counts` as `{"zero body midpoint credit": 1}`.

The economic answers below are therefore the same numbers as the superseded run. They are restated from the corrected directory, not from the old one.

---

## Invocation

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d1_decomposition.py
```

| Item | Value |
|---|---|
| Code revision | `5669773f356f6c33cef86bd0da30ce4051709a6b` |
| D0 directory | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z` |
| D0 receipt code SHA | `004ba80052f6586f6a230ad207e8151e695e156e` |
| D0 receipt verdict | `READY` |
| Provenance problems | none |
| Generated | `2026-09-14T02:51:42.539880+00:00` |
| Output directory | `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z` |
| D0 directory writes | none |

Generated tables, JSON, the Markdown report, and `waterfall_development.png` stay in that external directory. They are not in the repository.

---

## Tests

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint009_d1_body_wing_decomposition.py -q
```

9 passed before the official rerun: the original six plus regressions for a zero date-level numerator, non-boolean `pairing_ok`, a fractional unit that must not be truncated, and a D0 receipt mismatch that blocks acceptance of a `READY` economic result.

---

## Coverage and gates

| Check | Result |
|---|---|
| Development trades | 2,087, all `window_label = development`, all `input_ok` |
| Development dates | 209: 208 `verified_positive_short`, one `verified_zero_short` |
| Zero-short date | `2020-03-13`: 0 trades, every dollar field 0, ratio reason `zero body midpoint credit` |
| Pairing failures | 0 |
| Null trade-level ratios | 0 |
| Null date-level ratios | 1, reason `zero body midpoint credit` |
| Later-period rows in the decomposition tables | 0 |
| Stored vs arithmetic mid | 0 legs above 1e-6. Maximum absolute difference 4.55e-13. Diagnostic only; not a gate |

| Gate | Result |
|---|---|
| Provenance | PASS. Accepted directory, receipt SHA, receipt `READY`, and development coverage |
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

## Annual decomposition

Source: `annual_decomposition.parquet` in the corrected evidence directory. Dollars are rounded to cents for review. Identity residuals versus each year’s official sum remain within 1e-11. 2020 includes the zero-short date. Empty `ratio_reason` means the year’s ratio is defined.

| Year | Dates | Zero-short | Trades | \(B_{\mathrm{mid}}\) | \(H_{\mathrm{body}}\) | \(W_{\mathrm{mid}}\) | \(H_{\mathrm{wing}}\) | \(W_{\mathrm{pay}}\) | Body cross | Iron-fly cross | Body credit | Body concession ratio | Wing premium ratio | Wing concession ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 53 | 1 | 437 | 18,050.01 | 23,931.13 | 95,085.67 | 11,152.93 | 94,917.05 | −5,881.12 | −17,202.68 | 556,841.24 | 0.0430 | 0.1708 | 0.0200 |
| 2021 | 52 | 0 | 531 | 16,621.96 | 23,602.22 | 87,957.28 | 9,291.08 | 67,584.16 | −6,980.26 | −36,644.46 | 517,495.66 | 0.0456 | 0.1700 | 0.0180 |
| 2022 | 52 | 0 | 611 | 59,834.66 | 19,969.91 | 86,473.98 | 8,725.97 | 54,733.82 | 39,864.75 | −601.37 | 510,030.72 | 0.0392 | 0.1695 | 0.0171 |
| 2023 | 52 | 0 | 508 | 26,021.22 | 15,678.89 | 93,437.24 | 9,214.99 | 76,981.68 | 10,342.33 | −15,328.22 | 545,463.82 | 0.0287 | 0.1713 | 0.0169 |

---

## Answers

**D1-A.** Development body midpoint P&L is positive, about $120,528. After body execution concession of about $83,182, body cross P&L is still positive, about $37,346. Concession is about 3.9% of body midpoint credit. That remaining body margin is not a trading decision and is not true in every year: 2020 and 2021 body cross P&L are negative.

**D1-B.** The wings’ midpoint premium, about $362,954, is much larger than the wing execution concession, about $38,385. Against body midpoint credit those burdens are 17.0% and 1.8%. The descriptive wing spread is about 10.6% of wing midpoint premium. The dollar cost of the wings is mostly the midpoint premium, not the concession.

**D1-C.** The five terms explain the development iron-fly result. Identity residual versus −$69,776.72 is 1.46e-11. Net wing contribution, \(W_{\mathrm{pay}} - W_{\mathrm{mid}} - H_{\mathrm{wing}}\), is about −$107,122. That more than offsets the positive body cross result. Every development year has a negative official cross result: 2020 −$17,203, 2021 −$36,644, 2022 −$601, 2023 −$15,328. 2021 is the largest loss and also has a negative body cross result. 2022’s body cross result is about +$39,865 and nearly offsets that year’s wing drag. This is a description of the reconciled terms. It is not a finding that wings should be dropped and not a filter result.

---

## What this does not do

- The review annotation accepts this evidence. It does not start D2 implementation, and it does not start D3–D5.
- It does not rerun D0 or the baseline.
- It does not compute later-period economics.
- It does not change signals, wings, sizing, or the frozen contract.
- It does not measure path risk, payout frequency, or the value of protection.
- It does not claim midpoint fills are attainable.
