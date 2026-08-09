# Sprint 005 D4 — Feature quality audit plan

**Status:** DRAFT — awaiting owner approval  
**Mode:** Audit planning only (no D4 implementation in this step)  
**Date:** 2026-08-08

---

## 1. Accepted D3 checkpoint

| Item | Value |
|------|-------|
| Evidence commit | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| Production code SHA (D3 receipt) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Verdict | `PASS / PUBLISHED` |
| Snapshot / build | `e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886` |
| Features | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` (281 files) |
| Receipt | `…/features_backfill_v1.lineage.json` (`status=complete`) |
| Staging | `features.building/` absent |
| D4 code today | **Not implemented** (no audit script / D4 evidence file) |

D3 already proved generation, schema, key equality, digests, and atomic publication. D4 must not repeat that work.

---

## 2. D4 objective and acceptance question

**Objective.** Produce lean, reproducible coverage / missingness / PIT evidence on the **accepted D3 output** so Sprint 006 can choose a credible sample without guessing why features are null or whether lookbacks leak.

**Acceptance question.**

> Are the frozen Momentum and CVG features usable for Sprint 006, with understood coverage, explainable missingness, and credible point-in-time protection?

---

## 3. Frozen inputs and identities

| Input | Path / value |
|-------|----------------|
| Feature root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` |
| D3 receipt | `…/features_backfill_v1.lineage.json` |
| D2 observations | `…/straddle_observations_weekly.parquet` |
| D2 lineage | `…/straddle_observations_weekly.lineage.json` |
| Config | `configs/feature_backfill_v1.json` |
| Baseline window | `(42, 8)` (`window_size = 35`) |
| Expected ids | snapshot `e2c1f8fd44d72176`, build `20260724T045049097520Z_40b16886` |
| Expected D3 repo SHA | `131d0ac05e1e57749d3095923927a394fdcbc25b` |

CLI must take these paths/ids explicitly. No production defaults that hide identity.

---

## 4. D3 evidence reused (do not re-audit)

Trust from the accepted D3 receipt + evidence memo without re-hashing or re-proving publication:

* Exactly 281 ordered window files; staging absent; receipt `status=complete`.
* Per-file SHA-256 list already recorded (D4 may spot-check receipt fields; **not** re-hash all 281).
* Six-column schema, row count `1,063,995`, canonical D2 key equality, deterministic `(ticker, date)` order.
* Snapshot / build / config digest / Git / D2 digests / key digest identities.

**Startup gate (lightweight):** load receipt; require `status=complete`, matching expected snapshot/build/repo SHA, `window_count=281`, features dir present, staging absent, config digest match. Then proceed to quality metrics. Fail closed if identities disagree.

---

## 5. Coverage and count metrics (minimum set)

Process **one feature file at a time**; discard the frame after aggregating scalars. Never retain multiple full feature frames.

For each window `(max_lag, min_lag)` with columns `mom_*_mean`, `mom_*_count`, `cvg_*`, `cvg_count_*`:

| Metric | Definition | Sprint 006 failure prevented |
|--------|------------|------------------------------|
| `n_rows` | Row count (must equal receipt/D2 count; identity check only) | Wrong artifact / silent truncate |
| `mom_nonnull_{n,rate}` | Finite `mom_*_mean` | Ranking on mostly-empty momentum |
| `cvg_nonnull_{n,rate}` | Finite `cvg_*` | Ranking on mostly-empty CVG |
| `both_nonnull_{n,rate}` | Finite mom **and** cvg | Joint-signal sample overstated |
| `mom_count` summary | min / median / max; share with `count == window_size` | Ignoring partial-history dilution |
| `cvg_count` summary | same, independently | Same for CVG |
| By-date series (baseline + optional short/mid/long sentinels) | Daily `both_nonnull_rate` (and mom/cvg rates) over calendar | Missing warm-up / late collapse |

**Cross-window rollup:** one table of the per-window scalars above for all 281, ordered as in the receipt. No dashboards, plots, or extra distributional catalogs.

Sentinel windows for deeper date series: `(6,2)`, `(42,8)`, `(60,24)`.

---

## 6. Missingness attribution (calculator-faithful, independent)

Calculators are **row-lag** on the scheduled D2 panel (`rolling(window_size).shift(min_lag)` / equivalent positions `[i-max_lag, i-min_lag]`). `min_periods=1`. Null feature ⇒ `count < 1` (i.e. count 0). Counts ignore null economics inside the lag window.

Attribute **Momentum** and **CVG** separately using only:

* panel position vs `max_lag` / `min_lag` (from D2 date index per ticker), and  
* published `*_count` (from the feature file).

| Category | Rule | Meaning |
|----------|------|---------|
| `structural_warmup` | `end_idx < start_idx` after clamp, or `target_position < min_lag` (empty lag window) | Not enough scheduled history behind the feature date |
| `no_finite_inputs` | Valid lag window exists and `count == 0` | All contributing `return_pct` (mom) or `vol_gap` (cvg) were null |
| `partial_window` | `0 < count < window_size` | Some null economics; feature may still be non-null |
| `full_window` | `count == window_size` | Every lag slot contributed a finite input |

Report per-window counts of rows in each category (mom and cvg independently). Do **not** invent broader taxonomies (`observation_status` trees, liquidity reasons, etc.). Optional one-line note that D2 retains null economics on the complete key grid; D4 does not re-open D2 missingness design.

---

## 7. Feature-ready interval (baseline `(42, 8)`)

**Recommendation (requires approval before implementation):**

Define the **structural feature-ready interval for `(42, 8)`** as:

```text
ready_start = the earliest panel feature date whose 0-based per-ticker
              position index i satisfies i >= max_lag (= 42)
              on the sorted unique entry_date index of the D2 panel
              (equivalently: first date that can form a non-clamped
              lag window [i-42, i-8])
ready_end   = last D2 entry_date present on the published key grid
```

Properties:

* Uses only panel availability / lag geometry — **not** returns, Sharpe, or coverage maximization.
* Matches production membership: contributing slots are exactly `[i-max_lag, i-min_lag]` when `i >= max_lag`.
* Within `[ready_start, ready_end]`, **report** mom/cvg/both non-null rates and count summaries; do **not** shrink the interval to chase a coverage target.
* Sprint 006’s `min_count_pct` filter remains a **backtest eligibility** knob; D4 records the share of baseline rows with `count >= ceil(min_count_pct × 35)` only as an informational footnote if the owner wants it later — **not** as the ready-boundary rule.

If approval rejects structural readiness, the only alternate allowed without redesign is: same end date, with `ready_start` taken from the Sprint 004 manifest `feature_ready_start` (`2018-01-12`) **intersected** with structural readiness (max of the two). Still no performance-based trimming.

---

## 8. Exhaustive PIT evidence

**Invariant to prove for every contribution:**

```text
contributing observation.expiry_date < feature_date
```

**Method (no feature recompute, no second producer):**

1. Load D2 once: `ticker`, `entry_date`, `expiry_date` (plus row order).
2. For each ticker, sort by `entry_date` (same as calculators).
3. For each window `(max_lag, min_lag)` in receipt order:
   * For every feature position `i` with `end_idx = i - min_lag >= start_idx = max(0, i - max_lag)`:
     * Membership rows = `iloc[start_idx : end_idx + 1]` (all scheduled slots in the lag window, including null-economics slots — same positions D1 used for PIT).
     * Fail if any membership row has null `expiry_date` or `expiry_date >= feature_date`.
   * Aggregate: contributions checked, violations (must be 0), max expiry gap summary.
4. Process windows sequentially; do not hold feature frames for PIT (D2 only).

This reconstructs production membership from retained D2 + frozen lag rules. Calculators do not filter on expiry; D1 already established expiry is a **proof** concern. Exhaustive D4 proof closes the gap between D1’s bounded sample and the full panel × 281 windows.

**Planning blocker?** None for reconstruction. D2 retains `expiry_date` on the full key grid; D3 features are not required for membership. If a future change dropped `expiry_date` from D2, D4 would be blocked — that is not the current state.

---

## 9. Exact future files and CLI

| File | Role |
|------|------|
| `scripts/audit_feature_quality.py` | **Add** — read-only D4 audit CLI + helpers |
| `tests/unit/test_audit_feature_quality.py` | **Add** — synthetic focused tests |
| `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json` | **Produce at production run** — machine-readable metrics + PIT summary + ready interval |
| `docs/tmp/sprint005_d4_quality_audit_evidence.md` | **Produce after production run** — short acceptance memo |

**CLI (explicit args, no hidden defaults):**

```text
python scripts/audit_feature_quality.py \
  --features-dir <…/features> \
  --d3-receipt <…/features_backfill_v1.lineage.json> \
  --observations <…/straddle_observations_weekly.parquet> \
  --d2-lineage <…/straddle_observations_weekly.lineage.json> \
  --config configs/feature_backfill_v1.json \
  --output-json <…/features_quality_audit_v1.json> \
  --expected-snapshot-id e2c1f8fd44d72176 \
  --expected-build-id 20260724T045049097520Z_40b16886 \
  --expected-repo-sha 131d0ac05e1e57749d3095923927a394fdcbc25b
```

Refuse to overwrite an existing `--output-json`. Read-only w.r.t. features/D2/receipt.

**Reuse only:** receipt identity fields, `sha256_file` for config/D2 digest checks at startup, D2 columns already present, lag geometry identical to `MomentumCalculator` / `CVGCalculator` bulk membership. Do **not** call `calculate_bulk` for the audit.

---

## 10. Two-block implementation sequence

1. **Implement + test** — `audit_feature_quality.py` + unit tests (coverage aggregations, missingness categories, ready-interval helper, exhaustive PIT on a tiny synthetic panel). Full suite green. No production read required in CI.
2. **Production read-only run + evidence** — run CLI once against accepted D3 paths; write JSON beside derived outputs; write `docs/tmp/sprint005_d4_quality_audit_evidence.md`; commit evidence only after checks pass.

---

## 11. Focused synthetic test strategy

| ID | Case |
|----|------|
| T1 | Startup refuses wrong snapshot/build/repo SHA or incomplete receipt |
| T2 | Known warm-up panel: structural_warmup vs no_finite_inputs vs partial/full counts match hand expectations (mom and cvg independent) |
| T3 | Coverage rates and count summaries match oracle on a tiny frame |
| T4 | Ready interval for `(42,8)` equals structural rule on a long synthetic date index |
| T5 | PIT: planted `expiry_date >= feature_date` in a contributing slot → fail; clean panel → pass; checks every membership row, not a sample |
| T6 | Audit does not import/call feature calculators’ `calculate_bulk` and does not write under `features/` |

No million-row CI test; no production artifact required in unit tests.

---

## 12. Production execution and evidence procedure

1. Confirm clean tree at the D4 implementation SHA; confirm D3 receipt still `complete` at expected identities.
2. Run the CLI once; capture log under `C:/MomentumCVG_env/ops_logs/`.
3. Require: JSON written; PIT violations = 0 for all 281 windows; baseline ready interval recorded; 281-window coverage table present.
4. Write the short evidence memo linking receipt SHA, JSON digest, ready interval, headline coverage for `(42,8)`, PIT pass, and “D3 publication not re-validated.”
5. Commit memo (and only memo) unless the owner also wants the JSON path recorded solely by digest reference (JSON stays outside the repo).

---

## 13. Acceptance criteria

D4 is accepted when:

1. Read-only audit completes against the accepted D3 output + D2 without modifying them.
2. Lightweight receipt/identity gate passes.
3. Coverage + independent count summaries exist for all 281 windows; date series exist for baseline (and the two sentinels).
4. Missingness categories in §6 are reported for mom and cvg independently.
5. Approved feature-ready interval for `(42, 8)` is recorded and justified by the structural rule (not performance).
6. Exhaustive PIT: **zero** violations of `expiry_date < feature_date` for every lag-membership contribution on every window.
7. Evidence memo ties to D3 receipt identities and the audit JSON digest.
8. No feature recompute, no D5, no Sprint 006 backtest.

---

## 14. Explicit non-goals

* Recompute / republish features; change calculators, config, D2, or D3 artifacts.
* Re-hash all 281 files or repeat D3 publication validation.
* Rank/select windows; compute returns/Sharpe; add eligibility to “improve” coverage.
* D5 `SurfaceRunner` smoke; Sprint 006 backtest.
* Resume, multiprocessing, audit frameworks, notebooks, dashboards, new dependencies.
* Generalized data-quality platform or second feature producer.

---

## 15. Approval decisions (only these two)

1. **Feature-ready rule for `(42, 8)`** — approve §7 structural rule (`ready_start` at first date with panel index `i >= 42`, `ready_end` = last D2 date), with coverage reported inside the interval but not used to trim it.  
2. **PIT membership source** — approve reconstructing lag membership from D2 + frozen lag geometry (no `calculate_bulk`, no new retained contribution lists). Exhaustive over all windows.

No other forks. After approval, implement Block 1; do not update `docs/agenda/current_sprint.md` until this plan is accepted.
