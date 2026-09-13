# Sprint 009 D0 — evidence review

**Status:** `REVIEWED / ACCEPTED` through `82e3b46`  
**Executed:** 2026-09-13  
**Review annotation (2026-09-13):** Accepted through `82e3b46`. Findings unchanged. Implementation remains `004ba80`. Output remains `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/`. The earlier run stays superseded. This annotation does not change the results below and does not start D1.  
**Design:** [`sprint009_d0_design.md`](sprint009_d0_design.md) — accepted at `5329726`. Population, accounting, and official anchors unchanged.  
**Implementation:** `004ba80052f6586f6a230ad207e8151e695e156e`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z/`  
**Supersedes:** `546d3e6` / `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z/`. That run is not the current readiness record. Its accounting figures are unchanged by the stricter checks.  
**Characterization:** Readiness of a matched short iron-fly body/wing panel. Not a profitability, filter, or protection result. Later-period rows are stored only as row-level readiness. D1 has not started.

---

## Verdict

`READY`

Every gate passed under the stricter checks. The stricter checks did not expose an official input inconsistency. The baseline was not rerun and no check was weakened.

Official accounting is unchanged from the superseded run: 3,322 included short iron flies, official P&L −$146,279.84743525038, reconstructed P&L −$146,279.84743525033, residual sum 5.58e-11.

---

## What the correction changed

The correction does not change the accepted design, population, quantity convention, or anchors.

- Missing or non-finite `entry_spot`, `capital_at_risk_dollars`, `portfolio_quantity`, and logged `pnl_total_leg` now fail with the trade key, field, and reason. Reconstructed values no longer skip those checks.
- Short quantity must be negative and have positive magnitude. Each leg settlement spot must agree with the trade settlement spot.
- Duplicate midpoint legs fail pairing before set or dictionary construction. The cross row is kept and the reason is named.
- `n_included_short` must be a finite nonnegative integer. Fractional counts are not truncated. A positive short book requires `traded`. A `valid_no_trade` date cannot contain included positions. Verified long-only and zero-short dates remain `verified_zero_short`.

---

## Invocation

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d0_readiness.py
```

| Item | Value |
|---|---|
| Code revision | `004ba80052f6586f6a230ad207e8151e695e156e` |
| Official run | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Official execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Receipt SHA-256 | `4499ee89707fc0b514ad5276e5e5cfa4db5d829482cfec1c635e594a1ec35461` |
| Output directory | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z` |
| Official directory writes | none |

---

## Tests

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint009_d0_body_wing_readiness.py -q
```

17 passed, including the prior synthetic cases and regression tests that propagate each invalid case to `BLOCKED` while a valid control stays `READY`.

---

## Gates

| Gate | Result |
|---|---|
| Receipt | PASS. Inventory 17; all receipt hashes matched |
| Columns | PASS |
| Reconstruction | PASS. Independent cash, settlement, and structure checks |
| Pairing | PASS. No missing, extra, or duplicate selected-short keys |
| Reconciliation | PASS. Trade residuals failed = 0 |
| Primary anchor | PASS. n = 3,322; official and reconstructed P&L match the closeout and `decision_report.json` |
| Calendar | PASS. Blocked dates = 0 |

---

## Pairing, trades, and residuals

| Check | Result |
|---|---|
| Matched cross rows | 3,684 |
| `pairing_ok` false | 0 |
| Missing from mid | 0 |
| Unmatched midpoint keys | 0 |
| Max \|body+wing − four-leg\| | 9.09e-13 |
| Max \|four-leg − official\| | 1.01e-11 |
| Primary residual sum | 5.58e-11 |
| Primary official P&L | −146,279.84743525038 |
| Primary reconstructed P&L | −146,279.84743525033 |
| Decision-report short | n = 3,322; pnl = −146,279.84743525038 |

Trade counts by window: pre-study 362, development 2,087, later period 1,235. Primary = 3,322. Pre-study rows are labeled and are not part of the 3,322 anchor.

---

## Calendar

Authoritative dates: 403. Dates after 2026-07-10: 0.

| Window | Class | Dates |
|---|---|---:|
| pre_study | verified_positive_short | 62 |
| development | verified_positive_short | 208 |
| development | verified_zero_short | 1 |
| later_period | verified_positive_short | 132 |

The verified zero-short date is 2020-03-13. Whole-book status is `traded`, funnel `n_included_short` is 0, and there are no included short iron-fly rows.

---

## Outputs

Stored outside the repo, under the output directory above:

- `input_inventory.json`
- `matched_short_iron_flies.parquet`
- `short_calendar.parquet`
- `d0_report.json`
- `d0_report.md`
- `execution_receipt.json`

`d0_report.json` has no development-versus-later P&L, filter result, or protection summary.

---

## Not done

D1 has not started. This run does not attribute body economics, score protection, or filter entries.
