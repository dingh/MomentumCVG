# Sprint 004 — closeout

**Status:** `CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`
**Closed:** 2026-07-26
**Functional baseline SHA:** `2127638e9860cd76edb75e0d5d25a17c7baffe20`
**C8.6:** not planned

---

## 1. Goal and answer

**Goal:** Build a trustworthy weekly input layer (liquidity → adjusted → spot → surface) for future real-data backtesting and trade-decision generation.

**Answer:** Yes — a full-scope cold backfill published an immutable production snapshot with `production_accepted=true`. Spot and Surface gates returned accepted `WARN` statuses; Liquidity and Adjusted returned `PASS`. This closeout constitutes operator acceptance of those documented warnings.

Sprint 004 does **not** claim strategy profitability, Sharpe survival, or that every ticker-week surface is economically valid.

---

## 2. Accepted production snapshot

| Field | Value |
|-------|-------|
| Build ID | `20260724T045049097520Z_40b16886` |
| Snapshot ID | `e2c1f8fd44d72176` |
| Final root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Manifest | `…/manifests/input_snapshot_e2c1f8fd44d72176.json` |
| Requested output start | `2018-01-01` |
| Resolved as-of | `2026-07-17` |
| Feature-ready interval | `2018-01-12` → `2026-07-10` (from manifest `params`) |
| Scope | `full` |
| `production_accepted` | `true` |
| Manifest overall | `WARN` (accepted Spot + Surface warnings; `blocking_failures=[]`) |

---

## 3. Repository SHA lineage

| Label | SHA | Role |
|-------|-----|------|
| Freeze (`repo_sha_at_freeze`) | `007322a56e72f33691dc590f4e95e458df9b387c` | Identity frozen when the building root was prepared |
| Completion / functional baseline HEAD | `2127638e9860cd76edb75e0d5d25a17c7baffe20` | `--dictionary-only` Core mode; accepted closeout baseline |
| Liquidity marker `producer_repo_sha` | `2127638…` | Stage executed under the dictionary-only resume |

Producer markers may stamp **later recovery commits** than freeze when stages were resumed after orchestrator/reuse/parallelization fixes. Freeze remains authoritative for run identity; marker SHAs record which tree actually produced each stage.

---

## 4. Four-stage results

| Stage | Gate | Key evidence |
|-------|------|--------------|
| Liquidity | **PASS** | 2,391 liquid tickers; 8,932 classified; 2,200 panel days |
| Adjusted | **PASS** | 2,217 / 2,217 physical ZIPs; ~122 GB (`output_total_bytes` 121,886,028,614) |
| Spot | **WARN** (accepted) | 3,817,322 output keys; 2,209 inconsistent repeated spot keys excluded |
| Surface | **WARN** (accepted) | Exactly 1,063,995 A1 keys (expected=actual); 314,385 valid / 749,610 invalid metadata rows |

Prior component memos: [004_c4](004_c4_liquidity_panel.md), [004_c5](004_c5_adjusted_liquid.md), [004_c6](004_c6_option_surface.md), [004_c7](004_c7_pit_universe.md), [004_c8_4](004_c8_4_bounded_evidence.md).

---

## 5. Accepted limitations

1. **`--dictionary-only`:** Core classification made **zero** API calls and excluded **450** known-unresolved candidates from the equity filter (same set previously exhausted against Core).
2. **Spot ambiguity:** exclusions were fail-closed (inconsistent repeated spot values dropped; dominated by XSP-class cases).
3. **29.5% Surface-valid rate** uses the full liquidity-superset × 445-week denominator. It is **not** a missing-output rate — every expected A1 key exists.
4. **29,166** invalid metadata rows still carry partial quote rows; they remain `surface_valid=False` (informational WARN).
5. **Sprint 005 must** filter `surface_valid=True` and measure usable coverage inside its PIT-selected universe — do not treat the full-superset validity rate as economic coverage.

---

## 6. Raw inventory

| Item | Value |
|------|-------|
| Digest | `d581b9e7c72d79a35eb7c3b0bc762f87994e12a4481c39b62d39c84e9d31a736` |
| Physical / resolved | 2217 / 2215 |
| Range | `2017-09-25` → `2026-07-17` |

---

## 7. Publication evidence

- Process exit code **0**
- Atomic rename: `.building` → final root `20260724T045049097520Z_40b16886`
- All four stage markers present under `markers/`
- Manifest readback confirms `snapshot_id=e2c1f8fd44d72176`, `params.scope="full"`, `production_accepted=true`
- CLI printed `build_id`, `final_root`, and `snapshot_id`

---

## 8. Closeout test gate

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest
```

| Item | Value |
|------|-------|
| Result | **1321 passed**, 1 skipped |
| Duration | 31.68s |
| Exit code | 0 |
| `git diff --check` | clean (recorded at commit) |

---

## 9. Sprint 005 handoff

Use snapshot **`e2c1f8fd44d72176`** as the **immutable** input for:

```text
Surface (A1/A2, filter surface_valid=True)
  → straddle history
  → Momentum / CVG feature generation
```

Resolve artifacts from the published manifest under
`C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886`.

Do not reopen C8.5 producers for feature work. No **C8.6** is planned.

---

## 10. Explicit close statement

**Sprint 004 is closed.** Remaining Stage A polish (umbrella `validate`, CLI plan-copy cleanup historically labeled C3/C9) is absorbed into Sprint 005 planning if still needed — it does not keep Sprint 004 open.
