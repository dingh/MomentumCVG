# Decision 003: Per-side position cap (preserve `max_names_per_side`)

**Status:** Accepted  
**Date:** 2026-06-07  
**Supersedes:** [002_position_cap_semantics.md](002_position_cap_semantics.md)

---

## Context

[Decision 002](002_position_cap_semantics.md) required a **50 total** cap across long and short combined, with selection policy (global rank vs side quotas) left open (design doc Q1).

HD preference (2026-06-07): keep selection **simpler** — do not mix long and short into one ranked pool. Preserve the existing config field `max_names_per_side` and apply the cap **independently per direction**.

The runner already selects this way; aligning the spec removes implementation drift and closes Q1.

---

## Decision

For v1 backtest selection (S5 Phase 1):

1. **Cap is per side**, not a single combined long+short pool.
2. Use **`max_names_per_side`** in `BacktestRunConfig` — same limit applied separately to long and short candidate pools.
3. **Rank within each side** by `signal_rank_pct` (long: descending; short: ascending); take top `max_names_per_side` per direction.
4. **Total concurrent positions** on a rebalance date is `n_long_included + n_short_included`, with maximum `2 × max_names_per_side` when both sides fill.

**Example (≈50-name book):** `max_names_per_side = 25` → up to 25 long + 25 short = 50 total when both pools have enough eligible names.

Asymmetric long vs short caps (e.g. 20 long / 30 short) are **out of scope for v1** — use the same `max_names_per_side` for both directions unless a future decision adds `max_names_long` / `max_names_short`.

---

## Consequences

- S5 contract and `pipeline.step5` should document per-side cap semantics; no global long+short rank merge.
- `SurfaceRunner._select_size_and_settle` cap behavior is **aligned** with v1 target (modulo S4 handoff cleanup).
- Config search scripts should set `max_names_per_side` deliberately — e.g. 25 for a ~50-name book, not 50 per side (which would allow 100 total).
- Decision 002 total-cap semantics are **withdrawn**.
- Contract tests for S5 should assert per-side cap independently (long overflow does not steal short slots).

---

## References

- `docs/v1_spec_pins.md`
- `docs/surface_engine_portfolio_metrics_design.md` § S5 Phase 1
- `src/backtest/surface_runner.py` — `_select_size_and_settle`
