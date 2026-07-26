# Repository map

**Status:** Active  
**Last updated:** 2026-07-26 (Sprint 004 closeout — snapshot-first handoff)

---

## Top-level layout

```
MomentumCVG/
├── src/              Core package
├── scripts/          CLI: data prep, precompute, backtests
├── configs/          JSON configs for legacy backtest
├── docs/             Active docs (+ docs/archive/)
├── tests/unit/       Unit tests
├── AGENTS.md         Agent operating rules
├── setup.py
├── pytest.ini
└── requirements.txt
```

## External paths (not in Git)

| Path | Purpose |
|------|---------|
| `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` | **Accepted production snapshot** (`e2c1f8fd44d72176`) — resolve Stage A via its manifest |
| `C:/ORATS/data/ORATS_Data` | ORATS raw daily ZIPs (liquidity panel producer input) |
| `C:/MomentumCVG_env/input/adjusted_liquid` | Producer / legacy working root for split-adjusted chains (C5) — **not** interchangeable with a published snapshot |
| `C:/MomentumCVG_env/input/liquidity` | Producer liquidity panel artifacts |
| `C:/ORATS/data/ORATS_Adjusted` | Legacy full-universe adjusted mirror (maintenance only; not active default) |
| `C:/MomentumCVG_env/cache/` | Mutable producer cache (spot, surfaces, features) — not the accepted handoff |
| `C:/MomentumCVG_env/venv/` | Python virtual environment |

Trusted downstream consumption uses the immutable published snapshot + manifest ([sprint_memos/004_closeout.md](sprint_memos/004_closeout.md)). Mutable `input/` and `cache/` roots remain valid **producer** locations only.

Activate venv:

```powershell
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1
```

---

## Source modules (`src/`)

| Module | Role |
|--------|------|
| `core/models.py` | Domain types: legs, strategies, signals, positions |
| `data/orats_provider.py` | Load split-adjusted ORATS parquet chains |
| `data/paths.py` | Canonical raw / adjusted-liquid / legacy path constants |
| `data/spot_price_db.py` | Spot prices |
| `features/momentum_calculator.py` | Momentum features |
| `features/cvg_calculator.py` | CVG (vol gap) features |
| `strategy/momentum_cvg.py` | Momentum + CVG signal |
| `strategy/builders.py` | Straddle, iron fly, iron condor builders |
| `strategy/universe_filter.py` | Universe filters |
| `portfolio/optimizer.py` | Equal-notional optimizer (legacy path) |
| `execution/backtest_executor.py` | Simulated fills/settlement (legacy) |
| `backtest/engine.py` | Legacy `BacktestEngine` (JSON config, runtime ORATS) |
| `backtest/engine_v2.py` | `BacktestEngineV2` skeleton |
| `backtest/pipeline.py` | 6-step pipeline (step1 universe, step2 signals, …) |
| `backtest/option_surface.py` | Surface DB + iron fly/straddle assembly + `FillAssumption` |
| `backtest/surface_runner.py` | **Canonical v1 runner** — one config, trade log |
| `backtest/surface_search.py` | Grid search over surface configs |
| `backtest/run_config.py` | `BacktestRunConfig` dataclass |
| `visualization/` | Plotly chain diagnostics |

---

## Three backtest paths

| Path | Entry | Maturity | v1 use |
|------|-------|----------|--------|
| **Surface runner** | `scripts/run_surface_search.py` → `SurfaceRunner` | Runnable; iron fly / iron condor, max-loss, fills | **Canonical for v1** |
| **Legacy engine** | `scripts/run_backtest.py` + `configs/*.json` | Runnable; straddle-centric | Research / comparison only |
| **Engine V2** | `BacktestEngineV2` + `pipeline.py` | Skeleton (`run()` not implemented) | Future; not v1 blocker |

See [decisions/001_canonical_backtest_path.md](decisions/001_canonical_backtest_path.md).

---

## Data flow (v1 target)

```
ORATS raw ZIPs (ORATS_Data)
    → build_liquidity_panel.py → ticker_liquidity_panel.parquet (C4 producer)
    → apply_split_adjustment.py (filtered) → input/adjusted_liquid/*.parquet (C5 producer)
    → extract_spot_prices / precompute_option_surface → mutable cache (producer)
    → publish immutable snapshot + manifest  → accepted Stage A handoff (Sprint 004)

Published snapshot (manifest-resolved paths)
    → build_features.py        → momentum + CVG features (Sprint 005 — scope under review)

At rebalance date t:
    step1_get_universe (liquidity panel, PIT)
    → step2_score_signals (momentum/CVG)
    → build_ironfly_from_surface (FillAssumption)
    → equal max-loss sizing, integer contracts
    → hold-to-expiry PnL in trade log
```

---

## Scripts (typical order)

| Script | Output |
|--------|--------|
| `fetch_splits.py` | Scoped split history (`splits_hist_liquid.parquet`) |
| `apply_split_adjustment.py` | Filtered adjusted chains → producer `input/adjusted_liquid` |
| `audit_adjusted_liquid.py` | PASS/WARN/FAIL audit report |
| `extract_spot_prices.py` | Spot DB |
| `build_liquidity_panel.py` | Liquidity panel (raw ORATS only) |
| `build_features.py` | Feature parquet |
| `precompute_option_surface.py` | Surface artifacts |
| `run_surface_search.py` | Backtest / search results |

---

## Tests

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest c:\MomentumCVG\tests\ -q
```

326+ unit tests under `tests/unit/` — run `python -m pytest tests/ -q` for current count (see [baseline_status.md](baseline_status.md)).

---

## Configs

- `configs/*.json` — legacy engine only
- `BacktestRunConfig` / `SurfaceDataPaths` — surface path (Python dataclasses)

---

## Read first (for humans)

1. [v1_spec_pins.md](v1_spec_pins.md)
2. `src/backtest/surface_runner.py`
3. `src/backtest/pipeline.py` and `src/strategy/momentum_cvg.py`
4. [decisions/001_canonical_backtest_path.md](decisions/001_canonical_backtest_path.md)
