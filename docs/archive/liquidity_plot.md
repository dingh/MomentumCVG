# Chain Tradability Diagnostic Plots — Implementation Plan

**Goal:** Add a `src/visualization/` module with 6 diagnostic plot functions that visualize raw (unfiltered) ORATS option chain data across strike × DTE, helping the user understand tradability (liquidity, spreads, depth) before codifying backtest filters.

**Primary consumer:** `c:\MomentumCVG_env\notebook\chain_tradability.ipynb`

---

## 1. Column Mapping Layer — `src/visualization/__init__.py`

Define the canonical column mapping from actual ORATS parquet names to the plot-friendly names used in `docs/liquidity_plot.md`:

| Parquet column | Canonical name | Notes |
|---|---|---|
| `adj_stkPx` | `stockPrice` | split-adjusted |
| `adj_strike` | `strike` | split-adjusted |
| `adj_cBidPx` / `adj_cAskPx` | `callBidPrice` / `callAskPrice` | |
| `adj_pBidPx` / `adj_pAskPx` | `putBidPrice` / `putAskPrice` | |
| `smoothSmvVol` | `smvVol` | ORATS smoothed surface IV |
| `cMidIv` / `pMidIv` | `callMidIv` / `putMidIv` | raw mid IVs |
| `cOi` / `pOi` | `callOpenInterest` / `putOpenInterest` | |
| `cVolu` / `pVolu` | `callVolume` / `putVolume` | |
| `delta`, `gamma`, `theta`, `vega` | same | call greeks from row |
| `expirDate` | `expirDate` | parsed to `datetime` |

Store this as a `COLUMN_MAP: dict[str, str]` constant so it's maintained in one place and used by all downstream code.

---

## 2. `ChainSlice` Dataclass — `src/visualization/chain_slice.py`

A frozen dataclass holding preprocessed chain data for a single `(ticker, trade_date)` pair:

- **Fields:** `ticker: str`, `trade_date: date`, `df: pd.DataFrame` (the cleaned, renamed, derived-column-enriched DataFrame)
- **Derived columns** computed at construction time (following the `add_chain_features` pattern from the design doc):
  - `callMid`, `putMid` — `(bid + ask) / 2`
  - `callSpr`, `putSpr` — absolute spreads
  - `callSprPct`, `putSprPct` — percent spreads (with `eps=1e-9` denominator guard)
  - `mny` — `strike / stockPrice`
  - `logMny` — `ln(strike / stockPrice)`
  - `dte` — `(expirDate - trade_date).days`
- **Helper methods:**
  - `available_expiries() -> list[date]` — sorted unique expiry dates
  - `filter_expiry(expiry_date) -> pd.DataFrame` — single-expiry subset
  - `summary() -> dict` — row count, expiry count, DTE range, ticker, date

**Why a dedicated object:** All 6 plots share the same preprocessing. Compute once, plot many. Also makes the notebook API clean: `cs = load_chain("AAPL", date(2024,6,7)); plot_iv_surface(cs)`.

---

## 3. `ChainLoader` — `src/visualization/chain_loader.py`

A thin wrapper that:
1. Accepts `data_root` (defaults to `c:/ORATS/data/ORATS_Adjusted`)
2. Uses `ORATSDataProvider._load_day_data()` internally to reuse the existing LRU-cached parquet reader
3. Filters to the requested `ticker`
4. Renames columns via `COLUMN_MAP`
5. Computes derived columns
6. Returns a `ChainSlice`

Public API:
- `load_chain(ticker: str, trade_date: date) -> ChainSlice`
- `load_chain_multi(tickers: list[str], trade_date: date) -> dict[str, ChainSlice]` — batch loading (one file read)

**No liquidity filters applied here** — the whole point is to visualize the *unfiltered* chain so the user can see what gets filtered and why.

---

## 4. Plot Functions — `src/visualization/chain_diagnostics.py`

Six standalone functions, each accepting a `ChainSlice` (or its `.df` DataFrame) plus optional kwargs. All return `matplotlib.figure.Figure` so notebooks can display inline and scripts can save to file.

### Plot 1: `plot_iv_surface(cs, iv_col="smvVol", ax=None) -> Figure`
- Heatmap of IV across `strike × dte` (pivot table, `imshow` or `pcolormesh`)
- Y-axis: strike (or `logMny` for normalized view — controlled by `moneyness_axis: bool` kwarg)
- X-axis: DTE bins (unique expiry dates)
- Colorbar: IV level
- **Answers:** Is the IV surface smooth (tradable) or jagged (illiquid/noisy)? Where is IV rich/cheap?

### Plot 2: `plot_spread_heatmap(cs, side="worst", ax=None) -> Figure`
- `side` param: `"call"`, `"put"`, or `"worst"` (max of call/put spread%)
- Same strike × DTE geometry as Plot 1
- Color scale: 0% → 100%+ with a diverging colormap (green = tight, red = wide)
- Overlay: contour line at the `max_spread_pct` threshold from config (e.g., 50%) so you can see the "tradable zone"
- **Answers:** Can you realistically execute multi-leg trades in this ticker/DTE? Where do spreads explode?

### Plot 3: `plot_oi_volume(cs, expiry_date, ax=None) -> Figure`
- Two subplots (OI and Volume) for a single expiry
- X-axis: strike, Y-axis: count
- Dual lines: call vs put
- Vertical dashed line at ATM strike (`stockPrice`)
- Optional: shade the region that passes min_volume / min_oi thresholds
- **Answers:** Where is liquidity depth concentrated? Are candidate wings dead strikes?

### Plot 4: `plot_smile(cs, expiry_date, iv_col="smvVol", x_axis="logMny", ax=None) -> Figure`
- Single expiry smile curve
- `x_axis`: `"logMny"`, `"strike"`, or `"delta"`
- Plot `smvVol` as primary line; optionally overlay `callMidIv` / `putMidIv` as scatter
- Vertical line at ATM
- **Answers:** How quickly does IV/premium change as you move from ATM? Is put skew steep or flat?

### Plot 5: `plot_term_structure(cs, target_delta=0.50, iv_col="smvVol", ax=None) -> Figure`
- Interpolate IV at `target_delta` per expiry (using `np.interp` on sorted delta values)
- X-axis: DTE, Y-axis: interpolated IV
- Markers at each expiry point
- **Answers:** Which maturity is rich vs cheap? Is front-end IV elevated (event risk)?

### Plot 6: `plot_theta_per_capital(cs, expiry_date, structure="csp", ax=None) -> Figure`
- `structure` param determines capital calc: `"csp"` (cash-secured put: `strike × 100`), `"straddle"` (`2 × strike × 100`)
- `theta_$ = theta × 100` (per-contract), `theta_per_cap = theta_$ / capital`
- X-axis: `logMny`, Y-axis: `theta$/capital$ per day`
- Overlay: expected move band at `± S × IV_ATM × sqrt(T/365)`
- **Answers:** Where do you get the best theta per unit of capital committed?

---

### Plot 7: `plot_straddle_friction(cs, ax=None) -> Figure`

The single most important metric for ATM straddle tradability.

- Core metric: `straddle_friction = (callSpr + putSpr) / (callMid + putMid)` at the ATM strike for each available expiry
- Bar chart, X-axis = DTE (one bar per expiry), Y-axis = friction ratio
- Horizontal dashed line at candidate threshold (e.g., 10%, 15%) — passed as `threshold` kwarg
- Color bars green (below threshold) / red (above threshold)
- **Answers:** Which DTEs are executable for a given ticker? What's the round-trip cost of entering this straddle?

**Implementation notes:**
- For each expiry, find the ATM strike (closest to `stockPrice`) and compute friction from the call+put at that strike
- If multiple strikes are equidistant from spot, average across them
- This is the straddle-specific analog of Plot 2 (spread% heatmap) but collapsed to the one strike that matters

---

### Plot 8: `plot_tradability_distribution(friction_series, threshold=0.12, ax=None) -> Figure`

Cross-sectional view: how tradable is your universe on a given date?

- **Input:** A `pd.Series` (or dict) of `{ticker: atm_straddle_friction}` for all tickers on one trade date
  - Computed externally by calling `ChainLoader.load_chain_multi()` + ATM friction calc for each ticker
- Histogram (or CDF) of friction values across the universe
- Vertical dashed line at `threshold` — area to the right = fraction of untradable signals
- Annotate: "N of M tickers pass (X%)" in the legend
- Optional: overlay a second series for a signal-selected subset (e.g., Q10 picks) as a different color
- **Answers:** If I set `max_straddle_spread = 0.12`, how many of my trades survive? What fraction of the universe is realistically executable?

**Implementation notes:**
- The friction metric is the same as Plot 7 but computed in bulk
- A helper function `compute_universe_friction(loader, tickers, trade_date, target_dte) -> pd.Series` should live alongside this plot to batch-compute the metric
- `target_dte` finds the closest expiry per ticker (same logic as `StraddleHistoryBuilder._find_best_expiry`)

---

### Plot 9: `plot_tradability_timeseries(friction_df, threshold=0.12, vix=None, ax=None) -> Figure`

Time-series view: is liquidity stable or regime-dependent?

- **Input:** A `pd.DataFrame` with index = trade dates, columns = `["median_friction", "pct_passing"]`
  - Built by running Plot 8's friction calc across all trade dates in the backtest window
- Two-panel line chart:
  - **Top panel:** Median ATM straddle friction across the universe over time, with ±1 std shaded band
  - **Bottom panel:** Fraction of tickers passing threshold over time
- Optional: overlay VIX on secondary Y-axis (top panel) to visualize correlation between vol regime and spreads
- Horizontal reference line at threshold (top) and at a target pass rate (bottom, e.g., 80%)
- **Answers:** Does a fixed filter threshold work across 2019–2025, or do you need adaptive filtering? Did COVID / vol spikes blow out spreads?

**Implementation notes:**
- This is the most expensive plot to compute (requires loading chains for every trade date × every ticker)
- Pre-compute the friction DataFrame in a script or notebook cell and cache as parquet for reuse
- Suggested cache file: `cache/universe_atm_friction.parquet`

---

### Plot 10: `plot_signal_tradability(universe_friction, signal_friction, threshold=0.12, ax=None) -> Figure`

Does your signal systematically select less-liquid names?

- **Input:**
  - `universe_friction`: friction values for full universe on a given date (same as Plot 8 input)
  - `signal_friction`: friction values for signal-selected tickers only (e.g., Q10 picks)
- Side-by-side boxplots (or overlaid histograms) comparing distributions
- Annotate: median, pass rate for each group
- Can also be rendered as a time-series variant: "median friction of Q10 picks" vs "median friction of universe" over time
- **Answers:** If Q10 tickers have systematically worse friction than the average ticker, your backtest returns are overstated and you need a pre-signal liquidity gate.

**Implementation notes:**
- Pairs naturally with the task1b signal selection output — feed in the tickers your strategy actually picked
- Time-series variant reuses the `friction_df` from Plot 9, just split by signal membership per date

---

## 5. Dashboard Composite — `plot_tradability_dashboard(cs, expiry_date=None) -> Figure`

A single function that arranges Plots 1–7 for a quick visual assessment of one `(ticker, date)`. Layout: 4×2 grid (Plot 7 spans the bottom row). If `expiry_date` is not provided, auto-selects the expiry closest to 30 DTE.

Plots 8–10 are cross-sectional/time-series and require multi-ticker data, so they live outside the single-ticker dashboard.

---

## 6. Notebook — `c:\MomentumCVG_env\notebook\chain_tradability.ipynb`

| Cell | Purpose |
|------|---------|
| 1 | Setup (path, imports) |
| 2 | Load a single chain: `cs = loader.load_chain("AAPL", date(2024, 6, 7))`; print `cs.summary()` |
| 3 | Dashboard: `plot_tradability_dashboard(cs)` |
| 4–9 | Individual plots (1–7) with parameter exploration (vary ticker, date, expiry, delta target) |
| 10 | Plot 8: Cross-sectional tradability distribution for full universe on one date |
| 11 | Plot 9: Tradability time series (pre-compute friction, load from cache) |
| 12 | Plot 10: Signal-conditional tradability — Q10 picks vs universe |
| 13 | Multi-ticker comparison — load 5 tickers on same date, plot straddle friction (Plot 7) side-by-side |
| 14 | Notes / observations cell for documenting filter threshold insights |

---

## 7. Module Structure

```
src/visualization/
    __init__.py          # COLUMN_MAP constant, public imports
    chain_slice.py       # ChainSlice dataclass
    chain_loader.py      # ChainLoader (wraps ORATS data loading)
    chain_diagnostics.py # 10 plot functions + dashboard composite
```

Add `plotly` to `requirements.txt` if not already present.

---

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Data access:** `ChainLoader` instantiates `ORATSDataProvider` internally and calls `_load_day_data()` | Reuses existing LRU cache and path logic without exposing internals. Alternative (read parquet directly) rejected — duplicates path construction. |
| 2 | **No liquidity filtering in the visualization module** | Plots show the *unfiltered* chain. Existing `_apply_liquidity_filters()` thresholds drawn as overlay lines/contours (e.g., 50% spread threshold) so the user sees both the raw data and where the filter cuts. |
| 3 | **Column renaming at load time** | Rename to canonical names rather than keeping raw names. Makes plot code readable and matches the design doc. Mapping maintained in one constant (`COLUMN_MAP`). |
| 4 | **`plotly` for all charts** | Interactive hover/zoom/pan is essential for chain exploration. All plot functions return `plotly.graph_objects.Figure`. |

---

## Verification Plan

1. Load a known ticker/date (e.g., AAPL 2024-06-07) via `ChainLoader`, verify `ChainSlice.df` has all expected columns and derived fields
2. Render each of the 10 plots individually — confirm axes, labels, colorbars are correct
3. Render the dashboard composite — confirm layout renders without overlap
4. Compare the spread% heatmap "tradable zone" contour against the existing `ORATSDataProvider._apply_liquidity_filters()` thresholds (`max_spread_pct=0.50`) — they should visually agree
5. Test with an illiquid small-cap ticker to confirm the plots reveal the problem (wide spreads, zero OI in wings)
6. Plot 8: verify pass-rate annotation matches manual count
7. Plot 9: confirm VIX overlay lines up temporally with friction spikes (e.g., March 2020)
8. Plot 10: run on a date where Q10 picks are known — confirm the two distributions are visually distinct
