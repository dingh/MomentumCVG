## Plan: `fetch_orats.py` — Corporate Actions Fetcher (Task 2A partial)

**Scope:** API fetching of splits + earnings only. Manual CSV → parquet ingestion is deferred. The existing `OratsCorporateActionsFetcher` code is solid — this plan refines it, places it correctly in the project, and adds a proper CLI entry point.

---

**Architecture decision:** Split into two layers — a reusable class in `src/data/` and a thin CLI script in `scripts/`. This matches the pattern of `orats_provider.py` + `extract_spot_prices.py`.

---

**Steps**

1. **Create `src/data/corporate_actions.py`** — move and refine `OratsCorporateActionsFetcher` here:
   - Read API token from env var `ORATS_API_TOKEN` in `__init__` (fall back to explicit `token` kwarg) — keeps credentials out of scripts
   - Keep `_get()` retry/backoff logic as-is; it's correct
   - Keep `fetch_splits_for_ticker()` and `fetch_earnings_for_ticker()` as-is
   - **Fix** `_load_checkpoint` return type annotation — current `(List[pd.DataFrame], set)` is not a valid type hint; change to `tuple[list[pd.DataFrame], set[str]]`
   - **Fix** `_save_parquet` — add dedup before save: `df.drop_duplicates(subset=["ticker", "split_date"])` for splits and `["ticker", "earn_date"]` for earnings to handle re-runs gracefully
   - Add `DEFAULT_SPLITS_PATH` and `DEFAULT_EARNINGS_PATH` as module-level constants pointing to `cache/splits_hist.parquet` and `cache/earnings_hist.parquet`
   - Move `get_all_unique_tickers()` free function here (it's a data utility, not part of the fetcher class)

2. **Create `scripts/fetch_orats.py`** — CLI entry point with `argparse` subcommands:
   - `splits` — fetch all splits, save to `cache/splits_hist.parquet`
   - `earnings` — fetch all earnings, save to `cache/earnings_hist.parquet`
   - `all` — run both sequentially
   - Common args: `--token` (overrides env), `--data-root` (default `C:/ORATS/data/ORATS_Adjusted`), `--cache-dir` (default `C:/MomentumCVG_env/cache`), `--tickers` (JSON file path OR `"all"` to scan parquet root), `--rate-limit`, `--checkpoint/--no-checkpoint`
   - After each fetch, print a brief summary: total rows, date range, ticker count
   - Log to `C:/MomentumCVG_env/logs/fetch_orats_{date}.log` to match script logging convention

3. **Storage schema** for the two output files:
   - `cache/splits_hist.parquet` — columns: `ticker`, `split_date` (date), `divisor` (float), sorted by `(ticker, split_date)`
   - `cache/earnings_hist.parquet` — columns: `ticker`, `earn_date` (date), `annc_tod` (str), `updated_at` (timestamp), sorted by `(ticker, earn_date)`

4. **Default ticker list** — the CLI `--tickers all` option calls `get_all_unique_tickers(data_root)`. For targeted runs (e.g., just SP500), accept `--tickers configs/all_tickers.json` since that JSON already exists.

5. **Deferred — CSV ingestion** (`scripts/ingest_orats_csv.py`): Not built now, but the interface is known — scan downloaded CSV files, normalize column names to match the `adj_*` schema used by `ORATSDataProvider`, partition by `{YYYY}/ORATS_SMV_Strikes_{YYYYMMDD}.parquet`, run basic QA (expected row count, spot price sanity). Will be built when you get the ORATS subscription and start doing weekly live updates.

---

**Verification**
- Run with a single known ticker: `python scripts/fetch_orats.py splits --tickers AAPL --cache-dir ./cache_test`
- Inspect output parquet for correct schema and date parsing
- Re-run to verify checkpoint resume skips already-fetched tickers and dedup works correctly on the output file

---

**Decisions**
- Token from env `ORATS_API_TOKEN` (not hardcoded / not required CLI arg) — standard secure practice
- `get_all_unique_tickers` lives in `src/data/corporate_actions.py`, not the script — reusable for future feature pipeline filtering
- CSV ingestion deferred: it's a one-time operation and the schema is stable; not worth building until you have a live feed
