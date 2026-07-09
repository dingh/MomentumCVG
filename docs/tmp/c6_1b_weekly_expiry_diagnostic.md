# C6.1B — Weekly Expiry Policy Diagnostic

**C6.1B weekly expiry diagnostic: PASS**

No producer expiry behavior was changed in C6.1B.

---

## 1. Scope

- **Task:** C6.1B — weekly expiry policy diagnostic (read-only)
- **Repo commit reviewed:** `8aa03ea6ca06b2a8eb02412610d718bdcc64605c`
- **Data root:** `C:\MomentumCVG_env\input\adjusted_liquid`
- **Sample tickers:** AAPL, MSFT, NVDA, SPY, QQQ
- **Sample date range:** `2024-01-01` … `2024-03-31`
- **Entry date generation:** `weekly_trade_dates_in_range` on adjusted-liquid parquet presence (Friday anchor with Mon–Fri walk-back); bounded to 12 entry dates
- **Non-goal:** C6.1B is read-only — no producer expiry behavior changes, no parquet/cache writes

## 2. Sample summary

- **Observations attempted:** 60
- **Successfully diagnosed:** 60
- **Skipped:** 0

**Skip reasons:**
- (none)

**Resolved entry dates:**
- 2024-01-05, 2024-01-12, 2024-01-19, 2024-01-26, 2024-02-02, 2024-02-09, 2024-02-16, 2024-02-23, 2024-03-01, 2024-03-08, 2024-03-15, 2024-03-22

## 3. Core metrics

- **`expiry_chain_scanned == expiry_target_weekly` rate:** 100.0%
- **`target_listed_on_chain` rate:** 100.0%
- **Target body call quotable rate:** 100.0%
- **Target body put quotable rate:** 100.0%
- **Target body pair quotable rate:** 100.0%
- **C6.1C readiness (listed + body pair quotable):** 100.0% (60/60 diagnosed; threshold >= 90%)

## 4. Mismatch diagnostics

### Expiry mismatches (chain-scanned vs target weekly)

- (none)

### Missing target expiry on entry chain

- (none)

### Target listed but body pair not quotable

- (none)

### DTE delta distribution (`DTE_delta = DTE_chain - DTE_target`)

- `0`: 60

## 5. Recommendation

Proceed to C6.1C calendar-paired weekly expiry implementation. Target listed + body-pair quotable coverage is 100.0% (>= 90% threshold) on diagnosed observations.

---

## Method notes

- `expiry_chain_scanned` uses current `OptionSurfaceBuilder._find_best_expiry` with `dte_target=7` (weekly chain-scanned path).
- `expiry_target_weekly` uses `target_weekly_expiry_from_schedule(entry_date, schedule)` only.
- Target body quotability matches producer rules: ATM strike nearest spot (ties -> lower); `bid > 0`, `ask > 0`, `mid > 0` on body call/put.
- Entry-date parquets are read once per sample date with ticker-column pruning.
- `expiry_chain_scanned` still uses `OptionSurfaceBuilder._find_best_expiry` on in-memory available-expiry lists (no producer code changes).
