# Sprint 005 D1 — Feature correctness audit

**Status:** `PASS / D1 ACCEPTED`  
**Accepted:** 2026-08-03  
**Audited repository SHA:** `ff68a3d98a444558f3294fbcb0c6cfd99e197c1a`  
**Spec:** [`configs/feature_backfill_v1.json`](../../configs/feature_backfill_v1.json)

---

## 1. Verdict

**PASS / D1 ACCEPTED.** Sprint 005 D1 freezes the weekly Momentum/CVG contract, proves it with synthetic production-path literals (including zero-neutral CVG), and records a bounded real-data audit on the accepted D2 artifact. D3 may consume the versioned spec without reopening feature semantics.

D1 did not emit the 281-window backfill. That remains D3/D4.

---

## 2. What D1 accomplished

D1 pinned one authoritative weekly feature specification (`feature_backfill_v1`), locked Momentum inclusive-lag / null-slot / partial-history behavior and CVG cross-sectional construction through literal calculator tests, and corrected zero-adjusted gaps to neutral (`ff68a3d`). A deterministic bounded audit on accepted D2 confirmed lineage, units, count behavior, CVG range, sparse/spread-ineligible participation without eligibility prefiltering, and real-data PIT for `AAPL` at `2022-03-25`.

---

## 3. Frozen contract summary

| Element | Frozen value |
|---------|--------------|
| Spec | `feature_backfill_v1` / `sprint005_d1` |
| Grid | `min_lag` 2..24 step 2; `max_lag` 6..60 step 2; `max_lag > min_lag`; 281 windows; max-outer / min-inner |
| Baseline | `42:8` (first Sprint 006 baseline) |
| Momentum | Simple mean of available `return_pct` (percentage points); `min_periods=1`; publish `mom_*_mean`, `mom_*_count` |
| CVG | Sum raw `vol_gap` then feature-date median; adjusted gaps for `%pos`/`%neg` only; zero adjusted gap **neutral**; `cgap==0 → DVG=0, CVG=1`; `min_periods=1` |
| Cross-section | Complete D2 panel before any eligibility/liquidity filter |
| Lineage | Not in the feature spec; D3 receipts record path / snapshot / digests / repo SHA |

---

## 4. Lineage

### D2 artifact (read-only)

| Field | Value |
|-------|-------|
| Path | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |
| `snapshot_id` | `e2c1f8fd44d72176` |
| `build_id` | `20260724T045049097520Z_40b16886` |
| D2 `repo_sha` | `6f0d570727ce7979d7e1222466879c62ab8ba89a` |
| Rows / `a1_key_count` | `1063995` |
| `a1_key_digest` | `faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` |
| `output.file_sha256` | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` (recomputed match) |
| Units (receipt `transform_config`) | `return_pct_units=percentage_points`, `volatility_units=annualized_decimal` |

### Audited repository

| Field | Value |
|-------|-------|
| HEAD at audit | `ff68a3d98a444558f3294fbcb0c6cfd99e197c1a` |
| Zero-neutral fix | `ff68a3d` |

---

## 5. Bounded audit method and results

Followed plan §9. Windows audited only: `(6,2)`, `(12,2)`, `(42,8)`. Minimum periods read from the versioned spec (`1` / `1`). Momentum and CVG used `calculate_bulk` on the **complete** in-memory D2 panel; report rows filtered afterward.

### Deterministic sample

| Rule | Observed |
|------|----------|
| Feature-date indices 60, 220, 400 | `2019-03-01`, `2022-03-25`, `2025-09-05` |
| Mature ticker | `AAPL` |
| Sparse returns (`0 < return_pct_count ≤ 5`, first alphabetical) | `ADGI` (count = 5) |
| First `body_spread_ineligible` + finite `vol_gap` by (`entry_date`,`ticker`) | `AET` @ `2018-01-05`, `vol_gap ≈ -0.0455` |

### Count distributions (full cross-section, selected dates, count > 0)

| Window | mom count min/median/p90/max | cvg count min/median/p90/max |
|--------|------------------------------|------------------------------|
| `(6,2)` | 1 / 2 / 5 / 5 (n=4182) | 1 / 2 / 5 / 5 (n=4528) |
| `(12,2)` | 1 / 3 / 11 / 11 (n=4519) | 1 / 3 / 11 / 11 (n=4788) |
| `(42,8)` | 1 / 8 / 35 / 35 (n=4984) | 1 / 8 / 35 / 35 (n=5171) |

### Selected-row null rates (`AAPL`, `ADGI`, `AET` × 3 dates = 9 rows)

| Window | `mom_*_mean` null rate | `cvg_*` null rate |
|--------|------------------------|-------------------|
| `(6,2)` | 5/9 | 5/9 |
| `(12,2)` | 5/9 | 5/9 |
| `(42,8)` | 4/9 | 4/9 |

Sparse `ADGI` and spread-ineligible `AET` appear in calculator outputs without eligibility prefiltering. Nulls on those report rows reflect sparse/history-limited economics, not panel exclusion.

### Finite CVG range (full cross-section, selected dates)

All finite CVG values for the three windows lie in `[0, 2]` (mins 0.0, maxes 2.0).

### Units spot checks

- `|return_pct|` median ≈ 54.2, p90 ≈ 100 → percentage-point scale.
- `|entry_iv|` median ≈ 0.39; `|vol_gap|` median ≈ 0.11 → annualized-decimal scale.
- Receipt `transform_config` units match the observed magnitudes.

---

## 6. Real-data PIT evidence

Feature date: `2022-03-25` (`AAPL` panel position 220). Calculator selects by scheduled-week lag; expiry is checked only for PIT proof.

### `(6,2)` — 5 contributing rows (positions 214..218)

| entry_date | expiry_date |
|------------|-------------|
| 2022-02-11 | 2022-02-18 |
| 2022-02-18 | 2022-02-25 |
| 2022-02-25 | 2022-03-04 |
| 2022-03-04 | 2022-03-11 |
| 2022-03-11 | 2022-03-18 |

Max expiry `2022-03-18` < feature date. **All contributing expiries strictly before feature date.**

### `(42,8)` — 35 contributing rows (positions 178..212)

Entry span `2021-06-04` → `2022-01-28`; max expiry `2022-02-04` < `2022-03-25`. **All 35 contributing expiries strictly before feature date.**

---

## 7. Decision ledger

| Topic | Classification | Notes |
|-------|----------------|-------|
| Momentum simple mean of formation returns | DELIBERATE ADAPTATION | Weekly inclusive row lags on 281 windows |
| No Momentum compounding | MATCH | Simple-average intent |
| `return_pct` percentage points | DELIBERATE ADAPTATION | D2 contract |
| `vol_gap = RV − IV` | MATCH | |
| First / second cross-sectional medians | MATCH intent / DELIBERATE ADAPTATION | Weekly panel; spread-ineligible `vol_gap` retained |
| Cumulative = sum of raw gaps | MATCH | Production Fix #1 |
| DVG branches / `cgap==0 → CVG=1` | MATCH | Fix #2 |
| Zero adjusted gap neutral | DELIBERATE ADAPTATION; **REQUIRED FIX resolved** | `ff68a3d` (`>=0` → `>0`) |
| Weekly scheduled formation | DELIBERATE ADAPTATION | Surface weekly straddles |
| `min_periods=1` + independent counts | DELIBERATE ADAPTATION | Sprint 005 publishes counts; ranking later |

---

## 8. Tests, residual risks, D3 handoff

### Tests (at acceptance)

- `tests/unit/test_feature_backfill_v1_contract.py` — G1–G18 unit portion green
- `tests/unit/test_momentum_calculator.py`, `tests/unit/test_cvg_calculator.py` — green
- Full repository suite green after `ff68a3d`

### Residual risks

- Bounded audit covers three windows and three dates, not the full 281 emit.
- `calculate()` vs `calculate_bulk()` collapsed-window count `0` vs `NaN` inconsistency remains; D3 should prefer bulk semantics.
- Prefiltering tickers before CVG still changes medians (G14); D3 must calculate on the complete panel.
- Sprint 006 ranking / joint count thresholds are intentionally unset.

### D3 handoff

D3 must load `configs/feature_backfill_v1.json` for windows, min periods, baseline, and publish columns; expand the 281-grid from those explicit bounds (do not inherit helper/CLI defaults); calculate CVG on the full D2 cross-section; and record observation path, `snapshot_id`, digests, and repo SHA in its own runtime inputs / publication receipts.
