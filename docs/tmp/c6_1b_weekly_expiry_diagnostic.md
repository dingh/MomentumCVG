# C6.1B — Weekly Expiry Policy Diagnostic

**C6.1B weekly expiry diagnostic: PASS WITH POLICY CLARIFICATION**

No producer expiry behavior was changed in C6.1B.

C6.1C must not silently substitute a nearby expiry. No fallback to nearest-DTE expiry is allowed.

---

## 1. Scope

- **Task:** C6.1B — weekly expiry policy diagnostic (read-only)
- **Repo commit reviewed:** `6c6dcfc5a79397fd602467d2cd19bfc4ec28823b`
- **Data root:** `C:\MomentumCVG_env\input\adjusted_liquid`
- **Sample date range:** `2024-01-01` … `2024-03-31`
- **Entry date generation:** `weekly_trade_dates_in_range` on adjusted-liquid parquet presence (Friday anchor with Mon–Fri walk-back)
- **Non-goal:** C6.1B is read-only — no producer expiry behavior changes, no parquet/cache writes
- **Policy role:** C6.1B is deciding policy semantics, not proving broad weekly-option coverage.

## Policy decision for C6.1C

Proceed to C6.1C with strict calendar-paired weekly expiry.

For weekly strategy semantics, ticker-weeks without the exact next-week target expiry are skipped / not weekly-ready.

No fallback to nearest-DTE expiry is allowed.

Missing exact target weekly expiry means no weekly trade.

Missing exact target weekly expiry is expected for non-weekly-option names.

C6.1C must not silently substitute a nearby expiry.

The old chain-scanned `_find_best_expiry` behavior is permissive and should not define weekly strategy semantics.

Broad coverage affects opportunity count/capacity, not correctness.

## Sample A — known-weekly sanity check

The AAPL/MSFT/NVDA/SPY/QQQ sample is a known-weekly sanity sample. The result confirms mechanical viability on known weekly-option names. This is the main C6.1C mechanical gate.

- **Tickers (5):** AAPL, MSFT, NVDA, SPY, QQQ
- **Sampling method:** fixed known-weekly defaults (AAPL/MSFT/NVDA/SPY/QQQ) or CLI override
- **Observations attempted:** 60
- **Successfully diagnosed:** 60
- **Skipped (data/load errors):** 0

**Skip reasons (data/load):**
- (none)

**Resolved entry dates:**
- 2024-01-05, 2024-01-12, 2024-01-19, 2024-01-26, 2024-02-02, 2024-02-09, 2024-02-16, 2024-02-23, 2024-03-01, 2024-03-08, 2024-03-15, 2024-03-22

### Metrics

- **`expiry_chain_scanned == expiry_target_weekly` rate:** 100.0%
- **`target_listed_on_chain` rate:** 100.0%
- **Target body call quotable rate:** 100.0%
- **Target body put quotable rate:** 100.0%
- **Target body pair quotable rate:** 100.0%
- **Weekly-tradable ticker-week rate (listed + body pair):** 100.0% (60/60 diagnosed)
- **Missing exact target expiry rate (among diagnosed):** 0.0% (0/60)
- **Target listed but body pair not quotable rate:** 0.0% (0/60)

### Mismatch / skip examples

#### Expiry mismatches (chain-scanned vs target weekly)

- (none)

#### Missing target expiry on entry chain

- (none)

#### Target listed but body pair not quotable

- (none)

#### DTE delta distribution (`DTE_delta = DTE_chain - DTE_target`)

- `0`: 60

## Sample B — broad-universe coverage check

The broader C4 liquid-universe sample estimates how much of the universe is actually weekly-tradable. This is informational coverage, not a C6.1C correctness blocker. Missing exact target weekly expiry should be counted as expected no-trade skip behavior.

- **Tickers (50):** AAL, AAPL, ABBV, ABT, ACN, ADBE, ADSK, ALB, AMAT, AMD, AMGN, AMZN, ASML, AVGO, AXP, BA, FDC, HAIN, NFX, OTLY, ROP, RRR, STAA, AKRX, BTDR, CANO, DAVE, DLPH, EH, IWO, KITE, MMP, XP, WNR, WEC, WBT, TBPH, SC, PNT, PGNY, PAVM, OLO, ODT, OA, MSOX, MMS, MKTX, MAPS, JEPQ, HXL
- **Sampling method:** stratified top/middle/lower by snapshots_qualified from C:\MomentumCVG_env\input\liquidity\liquid_tickers.csv (max=50)
- **Observations attempted:** 600
- **Successfully diagnosed:** 430
- **Skipped (data/load errors):** 170

**Skip reasons (data/load):**
- `no_expiries_on_entry_chain`: 170

**Resolved entry dates:**
- 2024-01-05, 2024-01-12, 2024-01-19, 2024-01-26, 2024-02-02, 2024-02-09, 2024-02-16, 2024-02-23, 2024-03-01, 2024-03-08, 2024-03-15, 2024-03-22

### Metrics

- **`expiry_chain_scanned == expiry_target_weekly` rate:** 60.5%
- **`target_listed_on_chain` rate:** 60.5%
- **Target body call quotable rate:** 58.8%
- **Target body put quotable rate:** 58.6%
- **Target body pair quotable rate:** 57.2%
- **Weekly-tradable ticker-week rate (listed + body pair):** 57.2% (246/430 diagnosed)
- **Missing exact target expiry rate (among diagnosed):** 39.5% (170/430)
- **Target listed but body pair not quotable rate:** 3.3% (14/430)

**Coverage interpretation:** Missing exact target weekly expiry means no weekly trade. Missing exact target weekly expiry is expected for non-weekly-option names. Broad coverage affects opportunity count/capacity, not correctness.

### Mismatch / skip examples

#### Expiry mismatches (chain-scanned vs target weekly)

- (none)

#### Missing target expiry on entry chain

- `HAIN` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `OTLY` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `ROP` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `RRR` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `STAA` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `CANO` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `EH` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- `IWO` `2024-01-05`: chain=`None` target=`2024-01-12` listed=False body_pair=False dte_delta=None
- … and 162 more

#### Target listed but body pair not quotable

- `OTLY` `2024-01-12`: chain=`2024-01-19` target=`2024-01-19` listed=True body_pair=False dte_delta=0
- `CANO` `2024-01-12`: chain=`2024-01-19` target=`2024-01-19` listed=True body_pair=False dte_delta=0
- `PAVM` `2024-01-12`: chain=`2024-01-19` target=`2024-01-19` listed=True body_pair=False dte_delta=0
- `MAPS` `2024-01-12`: chain=`2024-01-19` target=`2024-01-19` listed=True body_pair=False dte_delta=0
- `OTLY` `2024-02-09`: chain=`2024-02-16` target=`2024-02-16` listed=True body_pair=False dte_delta=0
- `TBPH` `2024-02-09`: chain=`2024-02-16` target=`2024-02-16` listed=True body_pair=False dte_delta=0
- `PAVM` `2024-02-09`: chain=`2024-02-16` target=`2024-02-16` listed=True body_pair=False dte_delta=0
- `OLO` `2024-02-09`: chain=`2024-02-16` target=`2024-02-16` listed=True body_pair=False dte_delta=0
- … and 6 more

#### DTE delta distribution (`DTE_delta = DTE_chain - DTE_target`)

- `0`: 260

## Correct C6.1C gate

C6.1C may proceed if:

1. The strict weekly expiry policy is explicitly defined.
2. Known-weekly sanity sample (Sample A) passes.
3. C6.1C is required to skip ticker-weeks when exact target expiry is missing.
4. C6.1C is forbidden from falling back to nearest-DTE expiry.
5. Missing weekly expiry is treated as expected no-trade behavior, not as a producer bug.

Broad-universe coverage metrics may still be reported, but they should not block C6.1C unless the diagnostic cannot reliably distinguish missing target expiry from data errors.

## Recommendation

Proceed to C6.1C with strict calendar-paired weekly expiry. Sample A known-weekly sanity weekly-tradable rate is 100.0%. Ticker-weeks without the exact next-week target expiry are skipped / not weekly-ready. No fallback to nearest-DTE expiry is allowed. Sample B weekly-tradable rate is 57.2% (informational coverage only; does not block C6.1C).

---

## Method notes

- `expiry_chain_scanned` uses current `OptionSurfaceBuilder._find_best_expiry` with `dte_target=7` for comparison only.
- `expiry_target_weekly` uses `target_weekly_expiry_from_schedule(entry_date, schedule)` only.
- Target body quotability matches producer rules: ATM strike nearest spot (ties -> lower); `bid > 0`, `ask > 0`, `mid > 0` on body call/put.
- Entry-date parquets are read once per sample date with ticker-column pruning.
- Weekly-tradable = exact target listed AND body pair quotable. Otherwise not weekly-tradable.
