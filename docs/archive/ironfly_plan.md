# Plan: Iron Fly History Precompute Pipeline

Build the iron fly equivalent of the straddle history pipeline. The key difference: instead of one row per (ticker, date), output **one row per (ticker, date, wing_width candidate)** — this lets backtesting sweep over every wing width/delta that was available, picking the best one in hindsight or as a strategy parameter.

**Decisions captured:**
- Frequency: both monthly and weekly via `--frequency` flag (default monthly / 30 DTE)
- Filters: permissive defaults (`max_spread=0.99, min_yield=0.0`) to capture all wings
- Return pct: store both `return_pct_on_width` and `return_pct_on_credit`
- Failures: one row per ticker/date with `is_tradeable=False, failure_reason` (traceability)

---

## Steps

### 1. Create `src/features/ironfly_analyzer.py`

Define class `IronFlyHistoryBuilder` mirroring the structure of `StraddleHistoryBuilder` in `src/features/straddle_analyzer.py`:

- **`__init__`** — params: `data_root, spot_db, dte_target=30, max_spread_pct=0.99, min_yield_on_capital=0.0, min_volume=0, min_oi=0, frequency='monthly'`. No `wing_delta` — we want ALL candidates.
- **`_init_worker_components()`** — lazy-init `ORATSDataProvider` + `IronButterflyBuilder(wing_delta=0.15, max_spread_pct=self.max_spread_pct, min_yield_on_capital=self.min_yield_on_capital)`. (wing_delta value is irrelevant for `enumerate_candidates` but still required.)
- **`_find_best_expiry(ticker, trade_date)`** — delegated from straddle pattern; `monthly` = first Friday of next month (≥25 DTE); `weekly` = closest Friday ±4 days.
- **`process_single_entry(ticker, entry_date) -> List[Dict]`** — the core method:
  1. Fetch chain via `ORATSDataProvider.get_option_chain()`
  2. Validate: no chain → return one failure row (`failure_reason="no_options_at_entry"`)
  3. Find expiry via `_find_best_expiry()` → no expiry → `failure_reason="no_expiry_found"`
  4. Find ATM body strike via `builder._find_atm_strike()`
  5. Fetch body `short_call`, `short_put` → missing/invalid → `failure_reason="no_body_call"` / `"no_body_put"` / `"invalid_body_mid"`
  6. Call `builder.enumerate_candidates(chain, body_strike, short_call, short_put)`
  7. No candidates → one failure row `failure_reason="no_candidates"`
  8. Get exit spot from `spot_db` at `expiry_date` → missing → `failure_reason="no_spot_at_expiry"`
  9. For each `IronButterflyCandidate`: assemble `OptionStrategy` from candidate legs, call `strategy.calculate_payoff({expiry_date: exit_spot})`, compute `pnl = exit_value - strategy.net_premium`, form output dict

**Output dict schema per candidate row** (all fields):
- Identity: `ticker, entry_date, expiry_date, dte_target, dte_actual, entry_spot, body_strike`
- Candidate geometry: `wing_width, call_wing_strike, put_wing_strike, avg_wing_delta`
- Entry economics: `net_credit, credit_to_width, total_spread, spread_cost_ratio`
- Entry greeks: `net_delta, net_gamma, net_vega, net_theta`
- Individual leg quotes (for post-hoc analysis): `sc_bid, sc_ask, sc_iv, sc_delta, sp_bid, sp_ask, sp_iv, sp_delta, lc_bid, lc_ask, lc_iv, lc_delta, lp_bid, lp_ask, lp_iv, lp_delta`
- Exit / P&L: `exit_spot, exit_value, pnl, return_pct_on_width, return_pct_on_credit, annualized_return_on_width, spot_move_pct, days_held`
- Status: `is_tradeable=True, failure_reason=None`

---

### 2. Create `scripts/precompute_ironfly_history.py`

Mirror structure of `scripts/precompute_straddle_history.py`:

- **Constants** (overridable via argparse): `DTE_TARGET=30`, `N_WORKERS=24`, `MAX_SPREAD_PCT=0.99`, `MIN_YIELD=0.0`, `SP500_FILE`
- **argparse args**: `--data-root`, `--start-year` (2018), `--end-year` (2024), `--frequency {monthly,weekly}` (default `monthly`), `--max-spread-pct`, `--min-yield`
- **`generate_trade_dates(start_year, end_year, frequency)`** — reuse `get_trading_fridays()` logic; for monthly, group by (year, month) and pick the first Friday → same pattern as straddle monthly sampling
- **`process_date_batch(data_root, spot_db_path, trade_date, tickers, ...)`** — top-level function (pickleable for joblib); creates `IronFlyHistoryBuilder`, calls `process_single_entry()` per ticker; returns flat `List[Dict]`
- **`Parallel(n_jobs=N_WORKERS, backend='loky')` over dates** — batch by date (all tickers per date) to maximize `ORATSDataProvider` LRU cache hits — same reasoning as straddle script
- **Checkpoint** — `save_checkpoint(results, path)` every N dates; gzip parquet
- **Output path**: `C:/MomentumCVG_env/cache/ironfly_history_{frequency}_{start_year}_{end_year}.parquet` (gzip)

---

### 3. Tests

After notebook validation, add `TestIronFlyHistoryBuilder` in `tests/` using inline chains or existing IBF fixtures from `tests/conftest.py` — `sample_ibf_chain_atm` and `sample_ibf_chain_multi_width` are already present.

Test cases:
- `process_single_entry` with valid chain returns N candidate rows (one per symmetric wing pair)
- Each candidate row has `is_tradeable=True` and correct `pnl` math
- Failure paths return exactly one row with correct `failure_reason`:
  - `"no_options_at_entry"` — empty chain
  - `"no_expiry_found"` — no expiry within DTE tolerance
  - `"no_body_call"` / `"no_body_put"` — missing body option
  - `"invalid_body_mid"` — body mid = 0
  - `"no_candidates"` — no symmetric wing pairs survive filters
  - `"no_spot_at_expiry"` — spot DB missing exit price

---

### 4. Validation Notebook

Create `MomentumCVG_env/notebook/test_ironfly_history_builder.ipynb`:
- Load small slice of real ORATS data
- Call `process_single_entry` on one (ticker, date)
- Display all candidates as DataFrame
- Sanity-check: `return_pct_on_width` at expiry matches expected butterfly payoff math
- Plot `avg_wing_delta` vs `credit_to_width` across candidates for a single date

---

## Verification

- Run `python scripts/precompute_ironfly_history.py --start-year 2023 --end-year 2023 --frequency monthly` as a smoke-test on one year
- Inspect output: `df.groupby(['ticker','entry_date']).size()` should show multiple rows per date (multiple wing widths per ticker)
- Confirm `avg_wing_delta` values span a realistic delta range (e.g. 0.05–0.45) across rows for any single (ticker, date)
- Verify `return_pct_on_width` at max profit (spot == body_strike at expiry) ≈ `net_credit / wing_width` for all rows

---

## Design Decisions

- No `wing_delta` constructor param on `IronFlyHistoryBuilder` — it's irrelevant when the goal is exhaustive enumeration; `IronButterflyBuilder` still needs one but `0.15` is a harmless placeholder
- `return_pct_on_credit` uses `abs(net_credit)` as denominator to avoid sign confusion
- `annualized_return_on_width` uses `(return_pct_on_width * 365 / days_held)`
- Failure rows have all numeric candidate fields as `None`/`NaN` (same pattern as straddle history `is_tradeable=False` rows)
- `process_date_batch` is a **module-level function** (not a method) to remain pickleable by `joblib.Parallel`
- `IronFlyHistoryBuilder` instance is created fresh inside each `process_date_batch` call — stateless per worker
