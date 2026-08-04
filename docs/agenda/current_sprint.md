# Current sprint — 005

**Updated:** 2026-08-03
**Status:** `ACCEPTED`
**Mode:** Build — implementation authorized for deliverables D1–D5 below (scope-limited)
**Previous:** Sprint 004 — [`CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`](../sprint_memos/004_closeout.md)

---

## 1. Sprint goal

Produce trusted, reproducible, full-history weekly Momentum/CVG features from the accepted Sprint 004 surface snapshot so Sprint 006 can run a first credible real-data economic backtest.

---

## 2. Context and role in the project

```text
Sprint 004: trusted immutable input snapshot (done)
Sprint 005: trusted full-history weekly Momentum/CVG features  ← this sprint
Sprint 006: first credible real-data economic backtest
Later:     shadow signals and execution observation
```

**Accepted input:** snapshot `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886` at `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886`. Resolve A1/A2 from the published manifest. Do not reopen C8.5 producers or mutate the snapshot.

**Canonical bridge:** Surface (A1/A2) → one surface-derived straddle history (complete A1 key grid) → frozen 281-window Momentum/CVG grid → A4 files that `SurfaceRunner` / `SurfaceDataPaths` load as `features_{max}_{min}.parquet`.

**Reuse with explicit config:** Calculators and helpers (`MomentumCalculator`, `CVGCalculator`, `build_straddle_from_surface`, and functions inside `scripts/build_features.py`) may be reused, but Sprint 005 must pin an explicit, versioned configuration for the frozen 281-window grid and feature semantics. Do **not** silently inherit from inconsistent `build_features.py` helper vs CLI defaults. Legacy chain-based `precompute_straddle_history.py` is **not** the Sprint 005 source of truth.

### Lesson from Sprint 004

Do **not** repeat Sprint 004 over-engineering: duplicate audit logic, global blockers from minor irregularities, or destructive recovery that discards completed work.

**Blocker test:** *What specific incorrect Sprint 006 result could occur if this work is not completed now?*
Defer unless the answer involves incorrect feature values, leakage, broken lineage, lost coverage, irreproducibility, or Sprint 006 inability to consume the artifact. Validate correctness without a second orchestration system or destructive full-rebuild audits.

---

## 3. Required deliverables

### D1 — Feature specification + bounded Momentum/CVG audit — `ACCEPTED`
**Produces:** Versioned weekly feature spec and a bounded audit on the real production calculators (focused literal / independently calculated cases through production code—no second implementation), using the canonical straddle history from D2. Settles before full emit: minimum histories, missing-week behavior, CVG cross-sectional membership, sparse-history participation, zero-gap handling, units/signs, and PIT alignment (every contributing straddle expires before the feature date).
**Why / 004→006:** Pins Momentum/CVG decisions before Sprint 006 ranks on them; depends on D2 because features are defined on straddle observations.
**Areas:** `src/features/momentum_calculator.py`, `cvg_calculator.py`, existing unit tests; short audit note.
**Accept when:** Those decisions are explicit and settled; focused production-path cases pass on D2 observations; paper differences from weekly/project windows are not treated as defects.
**Status:** Accepted 2026-08-03. Spec `configs/feature_backfill_v1.json`; correctness memo [`005_feature_correctness_audit.md`](../sprint_memos/005_feature_correctness_audit.md); zero-neutral fix `ff68a3d`. Next: D3.

### D2 — Canonical surface → straddle observation transform — `ACCEPTED`
**Produces:** One full-history weekly straddle observation table that preserves exactly the complete accepted A1 `(ticker, entry_date)` key grid (no duplicates, no drop/filter of keys). Populate economics from valid A1/A2 rows; retain unavailable/invalid keys as rows with null economic fields and a missingness reason. Do not substitute another week or strike. Simple surface-to-straddle builder only—Momentum windows and CVG rules do not affect D2. Preserve `realized_volatility` and `entry_iv`; define `vol_gap = realized_volatility - entry_iv`. Also carry `return_pct` (and related economics) when available.
**Why / 004→006:** Closeout path is Surface → straddle → features; chain history would diverge from the trusted surface. Full key grid keeps missingness explicit for D1/D4.
**Areas:** Thin transform via `build_straddle_from_surface` / surface settlement—not a `refresh_weekly_inputs.py` stage.
**Accept when:** One reproducible, snapshot-lineaged straddle history matches the A1 key grid 1:1; coverage/missingness is measurable without collapsing invalid keys.
**Status:** Accepted 2026-08-01. Artifact at `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` (lineage `repo_sha` `6f0d570`); design § 11.2 checks A1–A10 all pass. Code: `d0c8e9d` + publication safeguards `6f0d570`.

### D3 — Standalone full-history feature backfill script
**Produces:** `scripts/backfill_features.py` that reads the accepted snapshot, uses D2, applies the explicit versioned config from D1, and emits the frozen 281-window set.
**Why / 004→006:** Explicit, repeatable path from published snapshot to Sprint 006–ready A4.
**Areas:** New standalone script; may reuse calculator functions; must not silently inherit `build_features.py` helper/CLI defaults; **do not** extend `refresh_weekly_inputs.py`.
**Accept when:** Completes on the accepted snapshot with minimal lineage (snapshot_id, build_id, repo SHA, versioned config / window grid).

### D4 — Complete Momentum/CVG outputs + coverage evidence
**Produces:** All 281 window files in `features/features_{max}_{min}.parquet` form, each including both `momentum_count` and `cvg_count` (alongside the configured mom/cvg signal columns), plus short lineage / coverage / missingness evidence.
**Why / 004→006:** Sprint 006 loads one file per config; incomplete grids or missing counts block trusted evaluation.
**Areas:** Emit per-window files under the versioned config; brief evidence next to artifacts or in a sprint memo.
**Accept when:** Full grid for the feature-ready interval; both counts present; coverage reported; lineage tied to `e2c1f8fd44d72176`; PIT expiry-before-feature-date rule held for contributing observations.

### D5 — Sprint 006 consumer smoke + acceptance evidence
**Produces:** Focused smoke that `SurfaceRunner` loads real feature files with snapshot surfaces/liquidity and scores ≥1 trade date; short acceptance record.
**Why / 004→006:** Unconsumable features are not done; proves the bridge before economic backtests.
**Areas:** `SurfaceRunner`, `SurfaceDataPaths`; pattern from existing synthetic data-flow tests.
**Accept when:** Smoke passes; evidence lists commands, snapshot id, pass/fail—no Sharpe/ranking required.

---

## 4. Rough dependency-aware execution order

1. ~~**D2** — Surface→straddle transform~~ **done**
2. ~~**D1** — Spec + bounded Momentum/CVG audit; settle decisions before full emit~~ **done**
3. **D3** — Standalone backfill wired to D2 + versioned config + calculators *(next)*
4. **D4** — Full 281-window emit + coverage/lineage
5. **D5** — Consumer smoke + acceptance evidence

Prefer small literal tests and one successful full emit over repeated infrastructure rebuilds.

---

## 5. Definition of done

- [ ] D1–D5 done under the blocker test
- [x] Straddle history preserves the full accepted A1 key grid; features lineaged to the accepted snapshot (not mutable `cache/` / `input/` stand-ins) — D2 done; feature lineage still pending D3/D4

- [ ] Frozen 281-window grid in `SurfaceRunner`-consumable form with `momentum_count` and `cvg_count`
- [x] D1 decisions settled; focused production-path Momentum/CVG cases pass (no second implementation)
- [ ] PIT: contributing straddles expire before the feature date
- [ ] Sprint 006 smoke loads artifacts (no economic ranking)
- [ ] Short acceptance evidence recorded (commands, ids, results, residual risks)

---

## 6. Non-goals and scope-control rules

**Out of scope:** economic backtesting / parameter ranking / best-window selection; strategy optimization, portfolio construction, execution modeling; exact monthly paper replication; rebuilding or modifying the accepted Sprint 004 snapshot; changes to the Sprint 004 backfill state machine; generalized feature store or new orchestration; broad legacy cleanup; production scheduling, incremental refresh, dashboards, broker integration; duplicate audit implementations, arbitrary coverage gates, or repeated full builds solely for infrastructure proof.

**Rules:** Apply the blocker test before expanding work. Do not redesign existing feature semantics unless repo evidence plus a focused correctness test show a material defect. Papers motivate economics; they do not specify this weekly implementation.

---

## Progress log

| Date | Notes |
|------|-------|
| 2026-07-26 | Sprint 004 closed; Sprint 005 marked scope under review |
| 2026-07-26 | Proposed Sprint 005 scope/plan written into this agenda (awaiting acceptance) |
| 2026-07-26 | Execution order: D2 (surface→straddle) before D1 (feature audit) |
| 2026-07-26 | Surgical corrections: full A1 key grid; D2 economics/`vol_gap`; versioned backfill config; D1 settle list; counts + PIT expiry |
| 2026-08-01 | Sprint 005 scope accepted; mode set to Build. D2 implementation authorized against the reviewed design [`surface_straddle_observation_transform_design.md`](../surface_straddle_observation_transform_design.md) (rev 3) |
| 2026-08-01 | **D2 accepted.** Canonical straddle observations published under `derived/e2c1f8fd44d72176/`; receipt `repo_sha` `6f0d570`; design § 11.2 A1–A10 pass. Next: D1 (feature spec + bounded Momentum/CVG audit) |
| 2026-08-03 | **D1 accepted.** Versioned feature contract frozen; zero-neutral CVG fix `ff68a3d`; bounded D2 audit + PIT recorded in [`005_feature_correctness_audit.md`](../sprint_memos/005_feature_correctness_audit.md). Next: D3 (standalone 281-window backfill) |
