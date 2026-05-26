# Decision 001: Canonical backtest path

**Status:** Accepted (provisional until repo audit completes)  
**Date:** 2026-05-23

---

## Context

MomentumCVG has three backtest implementations:

1. **Legacy** — `BacktestEngine` + `run_backtest.py` + JSON configs; straddle-focused; loads ORATS at runtime.
2. **Surface** — `SurfaceRunner` + `run_surface_search.py`; precomputed surface; iron fly/condor; equal max-loss; `FillAssumption`.
3. **Engine V2** — `BacktestEngineV2` + `pipeline.py`; intended unified engine but `run()` is still a skeleton.

v1 spec requires: momentum/CVG signal, short iron fly or iron condor (per run), long straddle on high-momentum names, max-loss sizing, conservative fills, weekly rebalance, dynamic liquidity universe.

---

## Decision

**Use the SurfaceRunner path as the canonical v1 backtest engine.**

Primary entry: `scripts/run_surface_search.py` (single config or controlled search)  
Core modules: `src/backtest/surface_runner.py`, `src/backtest/option_surface.py`, `src/backtest/pipeline.py` (steps 1–2)

Legacy engine remains for historical comparison only. Engine V2 is not a blocker for v1 unless audit finds SurfaceRunner cannot meet spec.

---

## Rationale

| Criterion | SurfaceRunner | Legacy | Engine V2 |
|-----------|---------------|--------|-----------|
| Iron fly / iron condor economics | Native | Straddle proxy | Planned |
| Max-loss integer sizing | Yes | Notional-based | Planned |
| Fill assumptions | `FillAssumption` | Limited | TBD |
| Runnable today | Yes | Yes | No |
| Aligns with v1 ops model | Yes | No | TBD |

`SurfaceRunner` docstring explicitly targets live-plausible configs and future weekly execution sheets.

---

## Consequences

- Go/no-go backtests (Tier B, 2020 → latest) run on surface path with conservative fills.
- Verification tests should target surface assembly, max-loss, and settlement before large backtest matrix.
- Legacy straddle results are not used for capital allocation decisions.
- Engine V2 consolidation is deferred until after v1 baseline unless audit falsifies this decision.

---

## What would falsify this decision

- Surface path cannot enforce PIT universe or momentum/CVG without lookahead
- Iron fly assembly systematically wrong vs builder unit tests
- Missing cache coverage makes 2020+ window non-reproducible
- Equal max-loss + 50-name cap cannot be implemented cleanly in `SurfaceRunner`

Sprint 001 repo audit should explicitly check these.

---

## Known gaps (SurfaceRunner not complete)

**Status as of Week 0 review** — canonical path is chosen; implementation is still in progress.

| Gap | Notes |
|-----|-------|
| Weekly rebalance wiring | Confirm end-to-end schedule vs grid search modes |
| 50-name cap | Enforce max concurrent positions in runner |
| Conservative fills default | Go/no-go must use harsh fills; confirm default in CLI |
| Short-structure comparison | Fly vs condor + wing params via config grid; may need small implementation fixes |
| Portfolio caps | Global max-loss budget, per-name cap, sector cap — pinned Sprint 002 |
| Integration smoke test | No automated E2E backtest in pytest yet |
| `BacktestEngineV2` | Skeleton only; not required for v1 if SurfaceRunner passes audit |

Sprint 001 `repo_audit.md` should expand this list with file-level references.

---

## References

- [v1_spec_pins.md](../v1_spec_pins.md)
- [repo_map.md](../repo_map.md)
- Archived: `docs/archive/backtest_engine_redesign.md`
