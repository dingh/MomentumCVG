# Current sprint — 005

**Updated:** 2026-07-26
**Status:** Open — feature pipeline (straddle history → Momentum/CVG)
**Mode:** Audit (inspect and plan before Build)
**Previous:** Sprint 004 — [CLOSED](../sprint_memos/004_closeout.md)

---

## Goal

Generate trusted **Stage A4 features** (straddle history, momentum, CVG) from the accepted Sprint 004 production snapshot.

**Sprint question:**

> Given immutable snapshot `e2c1f8fd44d72176`, can we build straddle history and Momentum/CVG features that are point-in-time correct and filtered to `surface_valid=True` within the PIT universe?

---

## Immutable input (do not rebuild)

| Field | Value |
|-------|-------|
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Final root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Feature-ready | `2018-01-12` → `2026-07-10` |
| Closeout | [sprint_memos/004_closeout.md](../sprint_memos/004_closeout.md) |

**Mandatory:** filter `surface_valid=True`; measure coverage inside the PIT-selected universe (not the full liquidity-superset validity rate).

---

## In scope

- Straddle history precompute + audit
- `build_features` / momentum / CVG (A4) trust
- Feature paths and schedule alignment to surface `entry_date`
- Incremental / watermark hardening for the feature branch (as needed)

## Out of scope

- Re-running C8.5 / ORATS cold backfill
- Strategy backtests / S1→S8 smoke → Sprint **006**
- Tier B go/no-go → Sprint **007**
- C8.6 (not planned)

---

## Source of truth

- [004_closeout.md](../sprint_memos/004_closeout.md)
- [v1_universe_protocol.md](../v1_universe_protocol.md)
- [surface_engine_data_contract.md](../surface_engine_data_contract.md)
- [v1_spec_pins.md](../v1_spec_pins.md)

---

## Progress log

| Date | Notes |
|------|-------|
| 2026-07-26 | Sprint 004 closed; Sprint 005 opened on snapshot `e2c1f8fd44d72176` |
