# Known bugs and spec drift

Registry of **confirmed bugs not yet fixed**. Remove an entry when fixed and covered by tests.

Agents: read this file when touching the listed modules. See also [agenda/current_sprint.md](agenda/current_sprint.md) § Known bugs.

---

## KB-001 — Iron condor `body_credit_per_share` uses condor short legs, not ATM straddle

| Field | Value |
|-------|-------|
| **Status** | Open — fix deferred (post–Sprint 003 or next `option_surface.py` review) |
| **File** | `src/backtest/option_surface.py` → `build_ironcondor_from_surface` (~line 792) |
| **Symptom** | M3 (`return_on_atm_straddle`) compares iron fly vs iron condor on **different denominators**. |
| **Spec** | [surface_engine_portfolio_metrics_design.md](surface_engine_portfolio_metrics_design.md) § M3 — both structures should use **ATM body call + put** at `body_strike` (`is_body` quotes), same as iron fly. |
| **Current (wrong)** | `fill.sell_price(short_call) + fill.sell_price(short_put)` — condor short OTM strikes. |
| **Expected** | `fill.sell_price(body_call) + fill.sell_price(body_put)` from `is_body` rows (mirror iron fly). |
| **Impact** | Cross-structure “alpha lost to wing protection” (M3 vs M1) is **not comparable** between iron fly and iron condor until fixed. `pipeline.py` only pass-throughs S3; no bug there. |
| **Fix checklist** | (1) Load `is_body` call + put in condor builder. (2) Set `body_credit_per_share` from body legs. (3) Unit test: condor M3 denominator = ATM straddle, ≠ condor short-leg sum. (4) Remove inline `BUG KB-001` comment. |

**Reported:** 2026-06-14 (Sprint 003 review).
