# Sprint 009 D0 — evidence review

**Status:** `AWAITING REVIEW`  
**Executed:** 2026-09-13  
**Design:** [`sprint009_d0_design.md`](sprint009_d0_design.md) — accepted at `5329726`  
**Implementation:** `546d3e6f6c79549d389e14e5ad60bc9520407d09`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z/`  
**Characterization:** Readiness of a matched short iron-fly body/wing panel. Not a profitability, filter, or protection result. Later-period rows are stored only as row-level readiness. D1 has not started.

---

## Verdict

`READY`

Every gate passed. The primary-window short book is 3,322 included iron flies and official P&L −$146,279.84743525038, within $0.01 of the accepted −$146,279.85. The baseline was not rerun. No check was weakened.

---

## Invocation

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d0_readiness.py
```

| Item | Value |
|---|---|
| Code revision | `546d3e6f6c79549d389e14e5ad60bc9520407d09` |
| Official run | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Official execution SHA | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Contract SHA-256 | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` (receipt integrity gate) |
| Receipt SHA-256 | `4499ee89707fc0b514ad5276e5e5cfa4db5d829482cfec1c635e594a1ec35461` |
| Output directory | `C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z` |
| Official directory writes | none |

---

## Tests

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint009_d0_body_wing_readiness.py -q
```

12 passed. Covers cash signs, share-equivalent quantity versus an extra ×100, swapped wing, non-body strike, duplicate leg key, reconciliation failure, missing and extra midpoint keys, quote mismatch, paired settlement mismatch, long-only and missing funnel dates, null short count, and the saved-row contract. Official artifacts were not used as fixtures.

---

## Gates

| Gate | Result |
|---|---|
| Receipt | PASS. Inventory 17; all receipt hashes matched |
| Columns | PASS |
| Reconstruction | PASS. Independent cash, settlement, and structure checks |
| Pairing | PASS. No missing or extra selected-short keys |
| Reconciliation | PASS. Trade residuals failed = 0 |
| Primary anchor | PASS. n = 3,322; official and reconstructed P&L match the closeout and `decision_report.json` |
| Calendar | PASS. Blocked dates = 0 |

---

## Pairing, trades, and residuals

| Check | Result |
|---|---|
| Matched cross rows | 3,684, including pairing-failure rows if any; none failed |
| `pairing_ok` false | 0 |
| Missing from mid | 0 |
| Unmatched midpoint keys | 0 |
| Max \|body+wing − four-leg\| | 9.09e-13 |
| Max \|four-leg − official\| | 1.01e-11 |
| Primary residual sum | 5.58e-11 |
| Primary official P&L | −146,279.84743525038 |
| Primary reconstructed P&L | −146,279.84743525033 |
| Decision-report short | n = 3,322; pnl = −146,279.84743525038 |

Trade counts by window: pre-study 362, development 2,087, later period 1,235. Primary = development + later period = 3,322. Pre-study rows are labeled and are not part of the 3,322 anchor.

---

## Calendar

Authoritative dates: 403. Dates after 2026-07-10: 0.

| Window | Class | Dates |
|---|---|---:|
| pre_study | verified_positive_short | 62 |
| development | verified_positive_short | 208 |
| development | verified_zero_short | 1 |
| later_period | verified_positive_short | 132 |

The verified zero-short date is 2020-03-13. Whole-book status is `traded`, funnel `n_included_short` is 0, and there are no included short iron-fly rows. It is not converted to cash and is not dropped.

---

## Outputs

Stored outside the repo, under the output directory above:

- `input_inventory.json`
- `matched_short_iron_flies.parquet`
- `short_calendar.parquet`
- `d0_report.json`
- `d0_report.md`
- `execution_receipt.json` (invocation metadata; not a sixth economic output)

`d0_report.json` has no development-versus-later P&L, filter result, or protection summary.

---

## Not done

D1 has not started. This run does not attribute body economics, score protection, or filter entries.
