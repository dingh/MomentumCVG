# Current sprint — 005

**Updated:** 2026-08-09
**Status:** `CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`
**Mode:** Closed (no active Build authorization)
**Closeout:** [`005_closeout.md`](../sprint_memos/005_closeout.md)
**Previous:** Sprint 004 — [`CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`](../sprint_memos/004_closeout.md)

---

## 1. Sprint goal

Produce trusted, reproducible, full-history weekly Momentum/CVG features from the accepted Sprint 004 surface snapshot so a future Sprint 006 can run a first credible real-data economic backtest.

**Outcome:** Goal met under documented limitations. See the closeout memo.

---

## 2. Context and role in the project

```text
Sprint 004: trusted immutable input snapshot (done)
Sprint 005: trusted full-history weekly Momentum/CVG features  ← CLOSED
Sprint 006: first credible real-data economic backtest (not started; requires separate authorization)
Later:     shadow signals and execution observation
```

**Accepted input:** snapshot `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886` at `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886`.

**Canonical bridge delivered:** Surface (A1/A2) → D2 straddle observations → frozen 281-window Momentum/CVG features → D5 `SurfaceRunner` consumability smoke on baseline `(42,8)`.

---

## 3. Required deliverables

### D1 — Feature specification + bounded Momentum/CVG audit — `ACCEPTED`
**Status:** Accepted 2026-08-03. Spec `configs/feature_backfill_v1.json`; correctness memo [`005_feature_correctness_audit.md`](../sprint_memos/005_feature_correctness_audit.md); zero-neutral fix `ff68a3d`; acceptance `0dec69e`.

### D2 — Canonical surface → straddle observation transform — `ACCEPTED`
**Status:** Accepted 2026-08-01. Artifact under `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` (lineage `repo_sha` `6f0d570`); design [`surface_straddle_observation_transform_design.md`](../surface_straddle_observation_transform_design.md); acceptance `3f598eb`.

### D3 — Standalone full-history feature backfill — `ACCEPTED`
**Status:** Accepted. Producer SHA `131d0ac`; evidence [`sprint005_d3_production_backfill_evidence.md`](../sprint_memos/sprint005_d3_production_backfill_evidence.md) (`816e28f`); 281 windows + complete receipt.

### D4 — Complete Momentum/CVG outputs + coverage evidence — `ACCEPTED`
**Status:** Accepted. Implementation `22a8375`; evidence [`sprint005_d4_quality_audit_evidence.md`](../sprint_memos/sprint005_d4_quality_audit_evidence.md) (`3c59f05`).

### D5 — Sprint 006 consumer smoke + acceptance evidence — `ACCEPTED`
**Status:** Accepted. Plan `b19e9c8`; evidence [`sprint005_d5_surface_runner_smoke_evidence.md`](../sprint_memos/sprint005_d5_surface_runner_smoke_evidence.md) (`3892079`).

---

## 4. Execution order (completed)

1. ~~**D2** — Surface→straddle transform~~ **done**
2. ~~**D1** — Spec + bounded Momentum/CVG audit~~ **done**
3. ~~**D3** — Standalone 281-window backfill~~ **done**
4. ~~**D4** — Coverage / missingness / PIT evidence~~ **done**
5. ~~**D5** — SurfaceRunner consumer smoke~~ **done**

---

## 5. Definition of done

- [x] D1–D5 done under the blocker test
- [x] Straddle history preserves the full accepted A1 key grid; features lineaged to the accepted snapshot (not mutable `cache/` / `input/` stand-ins)
- [x] Frozen 281-window grid in `SurfaceRunner`-consumable form with `momentum_count` and `cvg_count`
- [x] D1 decisions settled; focused production-path Momentum/CVG cases pass (no second implementation)
- [x] PIT: contributing straddles expire before the feature date
- [x] Consumer smoke loads artifacts (no economic ranking) — D5
- [x] Short acceptance evidence recorded (commands, ids, results, residual risks)

---

## 6. Non-goals (unchanged; still out of scope for this sprint)

Economic backtesting / parameter ranking / best-window selection; strategy optimization; rebuilding the accepted Sprint 004 snapshot; generalized feature store; production scheduling; Sprint 006 implementation.

---

## Progress log

| Date | Notes |
|------|-------|
| 2026-07-26 | Sprint 004 closed; Sprint 005 marked scope under review |
| 2026-07-26 | Proposed Sprint 005 scope/plan written into this agenda (awaiting acceptance) |
| 2026-07-26 | Execution order: D2 (surface→straddle) before D1 (feature audit) |
| 2026-07-26 | Surgical corrections: full A1 key grid; D2 economics/`vol_gap`; versioned backfill config; D1 settle list; counts + PIT expiry |
| 2026-08-01 | Sprint 005 scope accepted; mode set to Build. D2 implementation authorized against the reviewed design [`surface_straddle_observation_transform_design.md`](../surface_straddle_observation_transform_design.md) (rev 3) |
| 2026-08-01 | **D2 accepted.** Canonical straddle observations published under `derived/e2c1f8fd44d72176/`; receipt `repo_sha` `6f0d570`; design § 11.2 A1–A10 pass. Next: D1 |
| 2026-08-03 | **D1 accepted.** Versioned feature contract frozen; zero-neutral CVG fix `ff68a3d`; bounded D2 audit + PIT in [`005_feature_correctness_audit.md`](../sprint_memos/005_feature_correctness_audit.md). Next: D3 |
| 2026-08-09 | **D3 accepted.** 281-window publication + receipt; evidence [`sprint005_d3_production_backfill_evidence.md`](../sprint_memos/sprint005_d3_production_backfill_evidence.md) |
| 2026-08-09 | **D4 accepted.** Coverage / missingness / PIT audit; evidence [`sprint005_d4_quality_audit_evidence.md`](../sprint_memos/sprint005_d4_quality_audit_evidence.md) |
| 2026-08-09 | **D5 accepted.** One-date `SurfaceRunner` consumer smoke; evidence [`sprint005_d5_surface_runner_smoke_evidence.md`](../sprint_memos/sprint005_d5_surface_runner_smoke_evidence.md) |
| 2026-08-09 | **Sprint 005 closed** — `CLOSED — ACCEPTED WITH DOCUMENTED LIMITATIONS`. Closeout [`005_closeout.md`](../sprint_memos/005_closeout.md). Sprint 006 not started. |
