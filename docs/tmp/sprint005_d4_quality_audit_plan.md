# Sprint 005 D4 — Feature quality audit plan

**Status:** DRAFT — awaiting owner approval of this corrected plan  
**Mode:** Audit planning only (no D4 implementation in this step)  
**Date:** 2026-08-08  
**Baseline:** D3 evidence `816e28f`; D3 producer SHA `131d0ac`

---

## 1. Accepted D3 checkpoint

| Item | Value |
|------|-------|
| Evidence commit | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| D3 producer SHA (receipt `repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Verdict | `PASS / PUBLISHED` |
| Snapshot / build | `e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886` |
| Features | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` (281 files) |
| Receipt | `…/features_backfill_v1.lineage.json` (`status=complete`) |
| Staging | `features.building/` absent |
| D4 code today | **Not implemented** |

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
| Expected snapshot / build | `e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886` |
| Expected D3 producer SHA | `131d0ac05e1e57749d3095923927a394fdcbc25b` |

**Provenance (two distinct SHAs):**

| Identity | Source | Role |
|----------|--------|------|
| `d3_producer_repo_sha` | D3 receipt `repo_sha`; CLI `--expected-d3-repo-sha` | Validates the frozen feature producer |
| `d4_audit_repo_sha` | Clean Git `HEAD` obtained once by the audit before reading inputs | Records the audit implementation that produced the JSON/memo |

The operator supplies only `--expected-d3-repo-sha`. The audit must require a clean worktree, resolve its own D4 SHA from Git, and write both fields into the machine-readable result and evidence memo. Do not confuse this planning commit with the future D4 implementation commit.

CLI paths/ids are explicit. No production defaults that hide identity.

---

## 4. D3 evidence reused (do not re-audit)

Trust from the accepted D3 receipt + evidence memo without re-hashing or re-proving publication:

* Exactly 281 ordered window files; staging absent; receipt `status=complete`.
* Per-file SHA-256 list already recorded (D4 may spot-check receipt fields; **not** re-hash all 281).
* Six-column schema, row count `1,063,995`, canonical D2 key equality, deterministic `(ticker, date)` order.
* Snapshot / build / config digest / D3 Git / D2 digests / key digest identities.

**Startup gate (lightweight):** clean Git worktree; resolve `d4_audit_repo_sha`; load receipt; require `status=complete`, matching expected snapshot/build/`--expected-d3-repo-sha`, `window_count=281`, features dir present, staging absent, config digest match. Then proceed to quality metrics. Fail closed if identities disagree.

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
| `mom_count` summary | min / median / max over **finite** counts; share with `count == window_size` | Ignoring partial-history dilution |
| `cvg_count` summary | same, independently | Same for CVG |
| By-date series (baseline + sentinels) | Daily `both_nonnull_rate` (and mom/cvg rates) | Missing warm-up / late collapse |

**Cross-window rollup:** one table of the per-window scalars above for all 281, ordered as in the receipt. No dashboards, plots, or extra distributional catalogs.

Sentinel windows for deeper date series: `(6,2)`, `(42,8)`, `(60,24)`.

Note: calculator counts are `NaN` before `min_lag` (not zero). Summaries that need numeric counts must ignore those NaNs or report them separately as structural `no_slots`.

---

## 6. Missingness attribution (two dimensions)

Calculators are **row-lag** on the scheduled D2 panel. For each ticker, zero-based feature position `i`, and window:

```text
window_size = max_lag - min_lag + 1
available_slots = min(window_size, max(0, i - min_lag + 1))
```

Report **structural geometry** once per feature row (shared), then **per-signal economic availability** only where `available_slots > 0`, independently for Momentum and CVG using published counts.

### Structural geometry

| Label | Rule |
|-------|------|
| `no_slots` | `available_slots == 0` |
| `truncated_window` | `0 < available_slots < window_size` |
| `full_window` | `available_slots == window_size` |

Calculator counts are `NaN` when `no_slots` (before `min_lag`); they are **not** zero and must not be labeled as missing economics.

### Per-signal economic availability (`available_slots > 0` only)

| Label | Rule |
|-------|------|
| `zero_finite` | count equals 0 |
| `partial_finite` | `0 < count < available_slots` |
| `all_available_finite` | count equals `available_slots` |

Make explicit:

* A truncated window can still be `all_available_finite`.
* Truncated history (`truncated_window`) and missing economics (`zero_finite` / `partial_finite`) are different causes.
* Momentum and CVG attribution remain independent.
* No broader missingness taxonomy.

Report per-window row counts for each structural label and each mom/cvg economic label.

---

## 7. Feature-ready interval (baseline `(42, 8)`) — frozen

**Approved structural rule (no alternatives):**

```text
ready_start = 43rd common ordered D2 entry date
ready_end   = final D2 entry date
```

Reason:

```text
window_size = 42 - 8 + 1 = 35
first full window occurs at zero-based position 42
→ 43rd date in the common ordered date index (1-based)
```

D2 is an exact `2,391 × 445` ticker/date cross-product, so the common ordered-date index is sufficient.

The audit must compute and report the actual `ready_start` / `ready_end` calendar dates. Inside that interval it must report observed Momentum, CVG, and both-feature coverage — and **must not** move the interval based on coverage, returns, Sharpe, or model performance.

Sprint 006’s `min_count_pct` remains a backtest eligibility knob, not the D4 ready boundary.

---

## 8. Exhaustive PIT evidence (compact proof)

**Invariant for every numerical contribution under the frozen grid** (global `min_lag` minimum is 2):

```text
expiration_date of a contributing observation < feature date
```

**Compact exhaustive method** (production path; no nested enumeration of ~7.32B membership pairs):

1. Load D2 once: `ticker`, `entry_date`, `expiry_date`, `return_pct`, `vol_gap`.
2. For each ticker, order by `entry_date` (strictly increasing scheduled dates).
3. For each economic observation at position `j` that has a possible future feature row at `j + 2`, require:

```text
expiration_date[j] < entry_date[j + 2]
```

Because dates are strictly increasing, this proves `expiration_date[j] < entry_date[i]` for every configured contribution with `i >= j + 2` (including all windows whose `min_lag ≥ 2`).

4. Apply the proof **independently**:
   * Momentum: rows with finite `return_pct`.
   * CVG: rows with finite `vol_gap`.
5. Do **not** require expiration evidence for null-economics slots (they make no numerical contribution).
6. A finite economic observation with missing or invalid `expiry_date` **fails** the audit.
7. Report independently for Momentum and CVG:
   * finite observations eligible for the proof;
   * number checked;
   * number of violations (must be 0);
   * bounded violation diagnostics (a few sample keys/dates only);
   * the **minimum** expiration safety gap (`entry_date[j+2] - expiry_date[j]`), not the maximum.

No feature recompute; no `calculate_bulk`; D3 feature files are not required for PIT.

**Synthetic equivalence test (required):** on a tiny multi-ticker panel with several lag windows (including `min_lag=2` and larger), null mom/cvg economics, and safe + unsafe expiries, the compact proof must agree with explicit brute-force membership enumeration. Production execution uses **only** the compact proof.

---

## 9. Exact future files and CLI

| File | Role |
|------|------|
| `scripts/audit_feature_quality.py` | **Add** — read-only D4 audit CLI + helpers |
| `tests/unit/test_audit_feature_quality.py` | **Add** — synthetic focused tests |
| `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json` | **Produce at production run** |
| `docs/tmp/sprint005_d4_quality_audit_evidence.md` | **Produce after production run** |

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
  --expected-d3-repo-sha 131d0ac05e1e57749d3095923927a394fdcbc25b
```

Refuse to overwrite an existing `--output-json`. Read-only w.r.t. features/D2/receipt. Record `d3_producer_repo_sha` and `d4_audit_repo_sha` in the JSON and evidence memo.

**Reuse only:** receipt identity fields, `sha256_file` for config/D2 digest checks at startup, D2 columns already present, lag geometry for missingness/ready-interval. Do **not** call `calculate_bulk`.

---

## 10. Two-block implementation sequence

1. **Implement + test** — audit script + unit tests (coverage, two-dimension missingness, ready-interval dates, compact PIT vs brute-force equivalence on a tiny panel, dual SHA provenance). Full suite green. No production read in CI.
2. **Production read-only run + evidence** — run CLI once against accepted D3 paths; write JSON; write evidence memo; commit memo after checks pass.

---

## 11. Focused synthetic test strategy

| ID | Case |
|----|------|
| T1 | Startup refuses wrong snapshot/build/`--expected-d3-repo-sha` or incomplete receipt; dirty tree refuses; clean tree records `d4_audit_repo_sha` |
| T2 | Structural `no_slots` / `truncated_window` / `full_window` match `available_slots`; mom vs cvg economic labels independent; truncated + all_available_finite supported; `no_slots` not labeled as missing economics; counts NaN before `min_lag` |
| T3 | Coverage rates and count summaries match oracle on a tiny frame |
| T4 | Ready interval for `(42,8)` equals 43rd → last common ordered date |
| T5 | Compact PIT vs brute-force membership agree on a tiny multi-ticker, multi-window panel with null economics and safe/unsafe expiries; finite row with null expiry fails; production path uses compact only |
| T6 | Audit does not call `calculate_bulk` and does not write under `features/` |

No million-row CI test; no production artifact required in unit tests.

---

## 12. Production execution and evidence procedure

1. Confirm clean tree at the **D4 implementation** SHA; confirm D3 receipt still `complete` with `repo_sha = 131d0ac…`.
2. Run the CLI once; capture log under `C:/MomentumCVG_env/ops_logs/`.
3. Require: JSON written; mom and cvg PIT violations = 0; baseline ready dates recorded; 281-window coverage table present; both SHAs recorded.
4. Write the short evidence memo linking D3 receipt identity, `d3_producer_repo_sha`, `d4_audit_repo_sha`, JSON digest, ready interval, headline `(42,8)` coverage, PIT pass (eligible/checked/violations/min gap), and “D3 publication not re-validated.”
5. Commit the memo only (JSON stays outside the repo; record its digest in the memo).

---

## 13. Acceptance criteria

D4 is accepted when:

1. Read-only audit completes against accepted D3 output + D2 without modifying them.
2. Lightweight receipt/identity gate passes; `d3_producer_repo_sha` and `d4_audit_repo_sha` both recorded.
3. Coverage + independent count summaries exist for all 281 windows; date series for baseline and sentinels.
4. Structural + per-signal economic missingness (§6) reported; mom/cvg independent.
5. Feature-ready interval for `(42, 8)` equals the frozen 43rd→last common date rule; coverage reported inside without moving bounds.
6. Compact exhaustive PIT: zero violations for mom and cvg; min safety gap reported; synthetic compact≡brute-force test green.
7. Evidence memo ties to D3 receipt identities and the audit JSON digest.
8. No feature recompute, no D5, no Sprint 006 backtest.

---

## 14. Explicit non-goals

* Recompute / republish features; change calculators, config, D2, or D3 artifacts.
* Re-hash all 281 files or repeat D3 publication validation.
* Nested enumeration of all window×row×lag membership pairs in production.
* Rank/select windows; compute returns/Sharpe; add eligibility to “improve” coverage.
* D5 `SurfaceRunner` smoke; Sprint 006 backtest.
* Resume, multiprocessing, audit frameworks, notebooks, dashboards, new dependencies.
* Generalized data-quality platform or second feature producer.

---

## 15. Remaining approval

Ready-interval rule, compact PIT method, two-dimension missingness, and dual SHA provenance are **frozen by this corrected plan**.

**One remaining decision:** accept this corrected plan so Block 1 implementation may begin.

Do not update `docs/agenda/current_sprint.md` until this plan is accepted.
