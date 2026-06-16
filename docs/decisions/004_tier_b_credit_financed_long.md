# Decision 004: Tier B credit-financed integer sizing

**Status:** Accepted  
**Date:** 2026-06-14 (amended 2026-06-14 — symmetric short fair-share pass)  
**Supersedes:** Tier B per-name `max_loss_budget_per_trade` short sizing and total-CAR `deployable_capital` binding in [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) § Tier B (partial — see Consequences)

---

## Context

Sprint 003 Tier B sizes the vol book in two passes: shorts collect premium and bounded max loss; long straddles are financed from realized short credit.

Initial ADR 004 (same day) used `max_loss_budget_per_trade` as an independent per-short guardrail. HD review clarified that real-world deployment sets a **strategy-level short max-loss budget** and fits as many short names as possible — symmetric with the long-side iterative fair-share pass already in ADR 004.

Tier A (`conceptual`) is **unchanged**.

---

## Decision

### Config (Tier B)

| Field | Role |
|-------|------|
| `tier_b_short_max_loss_budget` | **Required** when `sizing_mode='integer_lots'`. Total max-loss dollars to deploy across the short book this cycle. May deploy less due to integer slack; never exceeds budget. |
| `deployable_capital` | **Not used** in Tier B S5 sizing (reserved on config for future / other layers). Long budget = collected short credit only. |
| `max_loss_budget_per_trade` | **Not used** for Tier B short sizing (retained on config for legacy / other uses). |

### Tier B (`integer_lots`) — two-pass credit-financed sizing

**Pass 1 — Shorts (iterative fair share on max-loss dollars)**

```text
short_budget = tier_b_short_max_loss_budget

Iterative filter (repeat until all survivors fit, or pool empty):

- `fair_share = short_budget / n_shorts_remaining`
- If any short cannot afford 1 contract (`cost > fair_share`), drop the **single worst offender** (highest 1-contract max-loss dollars) and recalculate
- `exclusion_reason = 'max_loss_exceeds_fair_share'` for dropped shorts

Dropping one expensive name raises `fair_share` for the rest (better capital utilization than bulk-removing all unaffordable names in one pass).

Integer sizing on survivors:
  fair_share = short_budget / n_survivors
  contracts = floor(fair_share / (max_loss_per_share × contract_multiplier))
  quantity = sign × (contracts × contract_multiplier)   # share-equivalent units, same as Tier A

**Quantity units (Tier B):** stored in **share-equivalent units** (`contracts × contract_multiplier`), not raw contract count. `quantity / contract_multiplier` is always an integer. Settlement matches Tier A: `pnl_total = quantity × pnl_per_share` (no separate multiplier in simulate).
```

Total deployed short max-loss ≤ `tier_b_short_max_loss_budget` (integer slack may leave budget unspent — no second pass to spend slack).

**Long premium budget**

```text
long_budget = collected_credit
  where collected_credit = Σ (|quantity_short| × net_credit_per_share)
```

(`quantity_short` is share-equivalent units; no extra `× contract_multiplier` in the credit sum.)

**Pass 2 — Longs (unchanged)**

When `long_budget <= 0` or no shorts funded → exclude longs with `no_short_credit`. Short-only cycles allowed.

1. Iterative fair-share filter on premium (`premium_exceeds_fair_share`) — drop the **worst** unaffordable long per round until all survivors fit
2. `contracts = floor(fair_share / (premium_paid_per_share × contract_multiplier))`; `quantity = sign × (contracts × contract_multiplier)`

Leftover long budget after integer sizing is not reallocated.

### Exclusion vocabulary

| String | When |
|--------|------|
| `max_loss_exceeds_fair_share` | Short 1-contract max-loss dollars > iterative `fair_share` |
| `premium_exceeds_fair_share` | Long 1-contract premium > iterative `fair_share` |
| `no_short_credit` | Long candidate but `long_budget <= 0` |
| `invalid_max_loss` | Cannot place ≥ 1 contract after floor sizing |

Existing reasons unchanged: `no_tradeable_structure`, `earnings_exclusion`, `max_names_cap`.

---

## Consequences

- `pipeline._apply_tier_b_sizing`: symmetric iterative fair-share on short max-loss, then credit-financed longs.
- `BacktestRunConfig` gains `tier_b_short_max_loss_budget` (required for `integer_lots`).
- ADR amended in place (no ADR 005).
- Design doc § Tier B glossary update deferred to Sprint 003 closeout (Deliverable 7).

---

## References

- [surface_engine_portfolio_metrics_design.md](../surface_engine_portfolio_metrics_design.md) § Tier A `equal_max_loss`
- [003_position_cap_per_side.md](003_position_cap_per_side.md)
- `src/backtest/pipeline.py` — `_apply_tier_b_sizing`
