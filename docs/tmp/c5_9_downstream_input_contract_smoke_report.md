# C5.9 Downstream Adjusted-Liquid Input Contract Smoke Report

> **Historical (pre-C5.10/C5.11A).** Path-default gap noted below was resolved in **C5.11A** (`src/data/paths.py`, closeout [004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md)). Do not treat § path/config gap as current state.

**Date/time:** 2026-07-04  
**Commit SHA:** `a99468f15443d7a268a34229f1de0f0f0ff5cfd5`  
**Mode:** Operational smoke only — no code changes

## Goal

Validate that the narrowest downstream ORATS parquet consumer can load and use the C5.8B scoped-split adjusted smoke root **before** a full adjusted-liquid production backfill or any strategy backtest.

This is **not** PIT universe validation. `liquid_tickers.csv` is the historical precompute/storage universe, not the rebalance-date trading universe.

## Downstream layer inspected

**Primary consumer:** `src/data/orats_provider.py` — `ORATSDataProvider`

This is the narrowest shared loader for adjusted ORATS wide-format parquets. It is used by:

| Consumer | Role |
|----------|------|
| `scripts/extract_spot_prices.py` | Spot DB extraction |
| `src/features/option_surface_analyzer.py` | Stage A surface precompute |
| `src/features/straddle_analyzer.py` | Straddle history precompute |
| `src/backtest/engine.py` | Legacy backtest path |
| `src/backtest/config.py` | Factory wiring (default path) |

Higher layers (`strategy/builders.py`, `src/backtest/option_surface.py`) consume `OptionQuote` objects built by `ORATSDataProvider`; they do not read raw parquet columns directly.

**Out of scope for this smoke:** `scripts/build_liquidity_panel.py` reads **raw** ORATS ZIPs from `ORATS_Data` and intentionally uses raw `stkPx` / `strike` for ATM liquidity scoring (C4 design).

## Commands run

### 1. Regression / audit tests

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py -q
```

**Result:** 70 passed in 1.91s

### 2. Downstream input-contract smoke (inline Python)

Operational script invoking `ORATSDataProvider`, `load_ticker_universe`, parquet schema checks, and functional calls (`get_spot_price`, `get_available_expiries`, `get_option_chain`) with relaxed liquidity filters (`min_volume=0`, `min_open_interest=0`).

Artifacts written (not committed):

- `C:/MomentumCVG_env/cache_c4_smoke/c5_9_downstream_input_smoke/c5_9_loaded_file_summary.parquet`
- `C:/MomentumCVG_env/cache_c4_smoke/c5_9_downstream_input_smoke/c5_9_downstream_sample_rows.parquet`
- `C:/MomentumCVG_env/cache_c4_smoke/c5_9_downstream_input_smoke/c5_9_smoke_results.json`

## Input paths

| Path | Role |
|------|------|
| `C:/MomentumCVG_env/cache_c4_smoke/adjusted_liquid_scoped_split_smoke/2020` | C5.8B adjusted smoke parquets |
| `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` | Storage universe (2,783 tickers) |
| `C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet` | C5.7 scoped splits (read-only context; not modified) |

## Output / cache paths

| Path | Written? |
|------|----------|
| `C:/MomentumCVG_env/cache_c4_smoke/c5_9_downstream_input_smoke/` | Yes (smoke artifacts only) |
| Production ORATS / input paths | **Not modified** |

---

## Check 1 — Reader compatibility

`ORATSDataProvider(data_root=...adjusted_liquid_scoped_split_smoke)` successfully loaded all 10 smoke parquets via `_load_day_data()`.

File pattern confirmed: `ORATS_SMV_Strikes_YYYYMMDD.parquet` under `{data_root}/2020/`.

Functional smoke on **AAPL / 2020-01-02**:

| Call | Result |
|------|--------|
| `get_spot_price('AAPL', 2020-01-02)` | `74.8925` (from `adj_stkPx`) |
| `get_available_expiries(...)` | 17 expiries |
| `get_option_chain(..., first_expiry)` | 104 quotes (52 strikes × call+put) |

**Result:** PASS

---

## Check 2 — Date inventory contract

Expected 10 C5.8B smoke dates:

```text
20200102  20200103  20200106  20200630  20200701
20200702  20200706  20201229  20201230  20201231
```

Found dates (sorted): **exact match** — 10/10.

**Result:** PASS

---

## Check 3 — Required downstream schema contract

Required columns checked per file (downstream trust layer + provider needs):

```text
ticker, trade_date, expirDate, strike, stkPx, adj_stkPx, adj_strike, split_factor
```

Optional adjusted option columns (when raw source columns present):

```text
adj_cBidPx, adj_cAskPx, adj_pBidPx, adj_pAskPx
```

| Metric | Value |
|--------|-------|
| Files checked | 10 |
| Missing required columns | 0 |
| Missing optional adjusted columns (when raw exists) | 0 |

**Result:** PASS

---

## Check 4 — Adjusted-field usage inspection

Static review of downstream trade-construction path:

### `ORATSDataProvider` — **uses adjusted fields correctly**

| Operation | Field used |
|-----------|------------|
| Spot lookup | `adj_stkPx` |
| Option strike in `OptionQuote` | `adj_strike` |
| Call/put bid/ask/mid | `adj_cBidPx`, `adj_cAskPx`, `adj_pBidPx`, `adj_pAskPx` |
| Liquidity filters | `adj_cBidPx`, `adj_pBidPx`, spreads on adjusted mids |
| ATM selection (`find_atm_strike`) | Compares `OptionQuote.strike` (already adjusted) to spot from `adj_stkPx` |

Raw `stkPx` / `strike` are **present in parquet** (preserved from ORATS raw) but **not used** by `ORATSDataProvider` for chain construction.

### Other downstream modules

| Module | Raw vs adjusted | Notes |
|--------|-----------------|-------|
| `extract_spot_prices.py` | Reads both `adj_stkPx` and `stkPx` | Writes both columns to spot DB; primary backtest column is `adj_spot_price` |
| `option_surface_analyzer.py` | Indirect via provider | Body strike / moneyness use adjusted `OptionQuote.strike` and spot DB |
| `strategy/builders.py` | Indirect via `OptionQuote` | No direct parquet access |
| `build_liquidity_panel.py` | **Raw** `stkPx` / `strike` | By design on `ORATS_Data` ZIPs — separate from adjusted-liquid path |

**Conclusion:** The option-trade-construction path through `ORATSDataProvider` is already adjusted-field-native. No input-layer code change required for C5.9.

**Path/config gap (not a schema blocker):** Many scripts still **default** `data_root` to `C:/ORATS/data/ORATS_Adjusted` (`orats_provider.py`, `extract_spot_prices.py`, `precompute_straddle_history.py`, `src/backtest/config.py`, `src/data/trading_day.py` helpers). Production C5.10 backfill must wire the new root explicitly or update defaults — see recommendation below.

**Result:** PASS (schema/field usage) · **WARN** (hardcoded legacy paths remain)

---

## Check 5 — Universe containment

Loaded tickers from all 10 smoke files compared against `liquid_tickers.csv`.

| Metric | Value |
|--------|-------|
| Total rows | 4,113,889 |
| Unique tickers loaded | 2,047 |
| Outside-universe tickers | **0** |

This confirms storage-universe containment only. It does **not** validate PIT trading universe at rebalance date `t`.

**Result:** PASS

---

## Check 6 — Adjusted field validity

| Metric | Value |
|--------|-------|
| Non-finite / null `adj_stkPx` rows | 0 |
| Non-finite / null `adj_strike` rows | 0 |

**Result:** PASS

---

## Code changes

**None.** Smoke passed without audit-script or provider patches.

## Safety confirmations

- No real ORATS API fetch
- No full adjusted backfill
- No strategy/backtest run
- `C:/ORATS/data/ORATS_Adjusted` not modified
- `C:/ORATS/data/ORATS_Data` read-only (C5.8B sample already on disk; not re-copied)
- `splits_hist_liquid.parquet` read-only
- Only `cache_c4_smoke/c5_9_downstream_input_smoke` written (plus reading existing C5.8B smoke)
- No generated parquet/cache files committed

## Limitations

- 10 trading days from 2020 only; 2,047 tickers vs 2,783 storage universe
- Functional chain test used one ticker (AAPL) on one date with filters disabled
- Spot DB / surface precompute / straddle scripts not end-to-end executed — only the shared reader validated
- Hardcoded default paths to legacy mirror not exercised in this smoke (explicit `data_root` override used)
- PIT universe enforcement deferred to C4 panel consumers at rebalance time

## Final verdict

**PASS WITH WARNING**

All input-contract acceptance criteria pass on real C5.8B smoke data. Warning is limited to **operational path wiring**: production scripts still default to `ORATS_Adjusted` and must be pointed at the new adjusted-liquid root during C5.10.

---

## Recommendation for C5.10 (full adjusted-liquid production backfill)

**Ready to proceed with C5.10**, subject to these pre-backfill items:

1. **Run full backfill** to the production adjusted-liquid root (not the legacy mirror), using scoped splits + `liquid_tickers.csv` filter — same pipeline as C5.6B/C5.8B but full year range.
2. **Wire downstream defaults** — update Stage A scripts / CLI to accept `--adj-root` or repoint defaults from `ORATS_Adjusted` to `C:/MomentumCVG_env/input/adjusted_liquid` (or HD-approved path). This is config/ops work, not a schema fix.
3. **Re-run C5.8 audit** on the production root after backfill completes.
4. **Do not** conflate `liquid_tickers.csv` containment with PIT universe — backtests must still filter via C4 liquidity panel at date `t`.
5. **Keep** `build_liquidity_panel.py` on raw ORATS_Data unless HD decides otherwise; it is a separate input artifact.

No blocker was found in parquet format or `ORATSDataProvider` consumption logic.
