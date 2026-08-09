# Sprint 005 — closeout

**Status:** `CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`
**Closed:** 2026-08-09
**Closeout baseline SHA:** `38920791de89a65b05a20985461b0eb1f37317d9`

---

## 1. Verdict

**CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS.**

Sprint 005 delivered trusted, reproducible, full-history weekly Momentum/CVG features from the accepted Sprint 004 surface snapshot, and proved that the canonical `SurfaceRunner` can consume the baseline `(42,8)` artifact with snapshot surfaces and PIT liquidity.

Sprint 005 does **not** claim window ranking, strategy profitability, Sharpe survival, trading-eligibility policy, or a full-history economic backtest.

---

## 2. Accepted lineage

| Identity | Value |
|----------|-------|
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Feature config | `configs/feature_backfill_v1.json` |
| Derived root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |

### D1

| Field | Value |
|-------|-------|
| Acceptance commit | `0dec69ef964d0182a8c570d812ec976e1f61a4ba` |
| Spec freeze | `221c730ab48a50c34afb91490a822573c4c50339` |
| Zero-neutral CVG fix / audited SHA | `ff68a3d98a444558f3294fbcb0c6cfd99e197c1a` |
| Correctness memo | [`005_feature_correctness_audit.md`](005_feature_correctness_audit.md) |

### D2

| Field | Value |
|-------|-------|
| Acceptance commit | `3f598eb558f157dc84ef0a85eb512fb18f39552a` |
| Implementation | `d0c8e9d07b6da5358a9782da05eb1d324f2e0418` |
| Publication safeguards / lineage `repo_sha` | `6f0d570727ce7979d7e1222466879c62ab8ba89a` |
| Observations | `…/straddle_observations_weekly.parquet` |
| Lineage | `…/straddle_observations_weekly.lineage.json` |
| Rows / key count | `1063995` / `1063995` |
| `output.file_sha256` | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` |
| `output.output_key_digest` | `faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` |
| Design of record | [`surface_straddle_observation_transform_design.md`](../surface_straddle_observation_transform_design.md) |

### D3

| Field | Value |
|-------|-------|
| Producer SHA (receipt `repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Evidence commit | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| Evidence memo | [`sprint005_d3_production_backfill_evidence.md`](sprint005_d3_production_backfill_evidence.md) |
| Features | `…/features/` — exactly **281** windows `(6,2)` … `(60,24)` |
| Receipt | `…/features_backfill_v1.lineage.json` (`status=complete`; SHA-256 `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5`) |

### D4

| Field | Value |
|-------|-------|
| Implementation | `22a8375d2d6c3b2dbd661697d9524548ea6def9a` |
| Evidence commit | `3c59f05ed971b0d56afd39937113a4f55e0880a1` |
| Evidence memo | [`sprint005_d4_quality_audit_evidence.md`](sprint005_d4_quality_audit_evidence.md) |
| Audit JSON | `…/features_quality_audit_v1.json` |

### D5

| Field | Value |
|-------|-------|
| Plan commit | `b19e9c8869664bf1ebc9e0b796f8045dd900a196` |
| Evidence commit / closeout baseline | `38920791de89a65b05a20985461b0eb1f37317d9` |
| Evidence memo | [`sprint005_d5_surface_runner_smoke_evidence.md`](sprint005_d5_surface_runner_smoke_evidence.md) |

---

## 3. D1–D5 completion

| Deliverable | What it proved |
|-------------|----------------|
| **D1** | Frozen weekly Momentum/CVG contract (`feature_backfill_v1`); bounded production-path correctness on D2, including zero-neutral CVG. |
| **D2** | Canonical surface→straddle observation table preserving the full A1 key grid with snapshot lineage. |
| **D3** | Standalone 281-window backfill published under the derived root with a complete receipt. |
| **D4** | Coverage, missingness, and exhaustive PIT evidence on the published grid without re-emitting features. |
| **D5** | `SurfaceRunner` consumability of accepted surfaces/liquidity + baseline `(42,8)` features on one real trade date. |

---

## 4. D4 evidence (accepted findings)

From [`sprint005_d4_quality_audit_evidence.md`](sprint005_d4_quality_audit_evidence.md):

| Item | Value |
|------|-------|
| `(42,8)` ready interval | `2018-10-26` → `2026-07-10` |
| Rows inside interval | `963573` |
| Joint (both non-null) coverage in interval | `658917` / rate `0.6838267572877198` (~68.38%) |
| PIT (Momentum / CVG) | `292470` / `312033` checked; **0** violations; min safety gap `6` days |

Structural warm-up (`no_slots` / truncated window geometry) is distinct from economic missingness (`zero_finite` / `partial_finite`), which is consistent with D2 null `return_pct` / `vol_gap` on the full A1 key grid.

---

## 5. D5 consumer evidence

From [`sprint005_d5_surface_runner_smoke_evidence.md`](sprint005_d5_surface_runner_smoke_evidence.md):

```text
trade date:          2022-09-02
feature rows:        2,391
PIT-universe rows:   3,869
scored-signal rows:  911
```

`SurfaceRunner.run_single_config()` completed successfully for the one-date interval. Structure/trade-log counts were recorded as diagnostics only and were not acceptance gates.

---

## 6. Final test gate

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q
```

| Field | Value |
|-------|-------|
| Result | **1494 passed**, 1 skipped |
| Runtime | `44.14s` |
| Exit code | `0` |
| Closeout baseline SHA | `38920791de89a65b05a20985461b0eb1f37317d9` |
| Date | 2026-08-09 |

---

## 7. Residual limitations

Sprint 005 did **not**:

* Rank or select feature windows among the 281.
* Evaluate returns, Sharpe, drawdown, or predictive performance.
* Establish trading-eligibility or count-threshold policies.
* Validate an economically viable strategy.
* Perform a full-history or multi-date economic backtest.
* Start shadow trading.
* Scope or implement Sprint 006.

---

## 8. Next step

Sprint 006 requires a **separate planning and authorization** step. This closeout does not authorize Sprint 006 work.
