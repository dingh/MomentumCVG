# Near-Term Roadmap

**Goal:** Validate signal quality → build live pipeline → start weekly put selling  
**Principle:** Don't build the full backtester until you know the signals are worth it.

---

## Phase 1: Signal Validation (no executor needed)

**Input:** `cache/straddle_features_weekly_2018_2025.parquet` + `straddle_history_weekly_2018_2025.parquet`  
**Output:** Go/no-go decision on momentum + CVG before committing capital

### Non-negotiable guardrails (apply to every test)
- **No lookahead:** features at date `t` use only data up to `t` — already enforced by `(max_lag, min_lag)` shift
- **Point-in-time universe:** use only tickers with valid data at formation date `t`; don't retroactively include tickers that didn't exist or had gaps
- **Forward target must be observable after `t`:** not the same week's return

---

### Task 1A — Coverage & Stability  `notebook/signal_coverage.ipynb`
**Purpose:** Confirm the feature dataset is clean before running any signal tests

- [ ] Per-date: count of tickers with non-NaN momentum / CVG across time
- [ ] Distribution plots: median + IQR of `mom_X_Y_mean` and `cvg_X_Y` over time — flag regime shifts or obvious bugs
- [ ] Rank stability: Spearman correlation of cross-sectional ranks between adjacent dates  
  *(low stability = noisy signal or stale data)*

**Done when:** No obvious data holes, rank autocorrelation is non-trivial (> 0.3)

---

### Task 1B — Cross-Sectional Sort (momentum baseline)  `notebook/signal_sort.ipynb`
**Purpose:** Does momentum predict the forward outcome? What's the random baseline?

Forward targets (pick both — they answer different questions):
- `return_pct` shifted forward 1 week *(does it pick good straddle returns?)*
- Next-week realized vol *(more relevant for put-selling — does it predict vol regime?)*

Method:
- [ ] For each formation date: rank tickers by `mom_X_Y_mean` → top/bottom quintile
- [ ] Compute average forward outcome per quintile
- [ ] **Baseline:** random selection of same N names, same dates → same metric
- [ ] Summary: top-minus-bottom spread, hit rate (% dates spread > 0), rolling performance

**Done when:** Top quintile clearly outperforms random on hit rate and average spread

---

### Task 1C — Incremental Value of CVG  `notebook/signal_cvg_filter.ipynb`
**Purpose:** Does CVG add anything beyond momentum?

Two tests (fast):
- [ ] **Filter test:** within top momentum quintile, split high-CVG vs low-CVG → compare forward outcomes
- [ ] **2D heatmap:** momentum quintile × CVG quintile → grid of average forward outcomes  
  *(if CVG adds, you'll see it within momentum rows — not just as a correlated re-ranking)*

**Done when:** High-CVG bucket inside top momentum outperforms low-CVG bucket on at least 2 of: mean, hit rate, Sharpe proxy

---

### Task 1D — Crude Put-Write Proxy  `notebook/put_write_proxy.ipynb`
**Purpose:** Bridge from "straddle momentum signal" to "weekly put selling edge"  
**Gate:** Do this before committing real capital — signals were designed around vol gaps, not directly put P&L

Method (simplified, ignore margin/liquidity for now):
- [ ] Each week: select top-N tickers by (momentum alone) vs (momentum + high CVG)
- [ ] Assume short put at ~20–30 delta, ~7–21 DTE, filled at mid minus small haircut
- [ ] Compute P&L: premium collected − max(0, strike − spot at expiry)
- [ ] Compare: CVG-filtered vs momentum-only vs random on win rate, avg P&L, tail losses

**Done when:** CVG-filtered selection shows meaningfully better win rate or lower tail loss vs momentum-only

---

## Phase 2: ORATS Live Data Pipeline

**Trigger:** Phase 1 shows signal value worth operationalizing  
**Goal:** Weekly candidate list + audit trail → manual trade decisions

### Task 2A — Raw Data Ingestion  `scripts/fetch_orats.py`
- [ ] Script to pull latest ORATS data (API or manual download) → raw immutable storage
- [ ] Normalize to partitioned Parquet schema matching existing `ORATS_Adjusted/` format
- [ ] Basic QA: row counts, missingness, duplicates, price sanity checks

---

### Task 2B — Incremental Feature Update  `scripts/update_features.py`
- [ ] Append new weekly rows to `straddle_history_weekly.parquet`
- [ ] Re-run `build_features.py` (or an incremental version) for new dates only
- [ ] Versioned output: tag feature file with run date + code hash

---

### Task 2C — Weekly Decision Report  `scripts/weekly_report.py` or notebook
- [ ] Load latest features → apply selected window params from Phase 1 results
- [ ] Output: ranked candidate table (ticker, momentum rank, CVG, recommended delta/DTE)
- [ ] Save snapshot with date → audit trail ("what did I know when I traded?")
- [ ] Flag: ticker mapping changes, splits, new listings / delistings

---

## Decision Gates

```
Task 1A pass → proceed to 1B/1C
Task 1B pass → proceed to 1C/1D
Task 1D pass → get ORATS subscription → Phase 2
Phase 2 MVP  → start paper trading → then real capital
```

---

## What's Deferred

| Item | Why deferred |
|---|---|
| Full backtester (Tasks 7–9) | Not needed until signal quality confirmed |
| Portfolio optimizer (Task 7) | Equal-weight proxy sufficient for Phase 1 |
| Margin / liquidity constraints | Phase 1 is deliberately naive |
| Corporate actions / symbol hygiene | Phase 2B+ — needed before live trading at scale |
