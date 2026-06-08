# Backtest evaluation protocol

**Status:** Active (metrics thresholds TBD)  
**Last updated:** 2026-05-23 (Week 0 review revision)

---

## Purpose

Define **how** v1 backtests are evaluated before paper trading and capital allocation. Windows and tiers are pinned here; specific pass/fail numeric thresholds will be set after the first harsh-fill weekly baseline run (target: Sprint 004–005).

---

## Evaluation tiers

| Tier | Window | Role |
|------|--------|------|
| **A — Sanity** | Full sample available in cache (e.g. 2018 → latest) | Max drawdown, long-run stability, regime survival |
| **B — Primary go/no-go** | **2020-01-01 → latest** | Main decision window for v1 |
| **C — Stress narrative** | Sub-windows inside Tier B (2020 Q1, 2022, 2024–25) | Explain failures; not for parameter tuning |

---

## Canonical backtest configuration (v1)

All tiers use the same pinned spec unless a sensitivity memo explicitly says otherwise:

- Signal: Momentum + CVG
- Short structure: Iron fly **or** iron condor (one per run; wing rule as config)
- Long structure: Long straddle (code default on high-momentum side)
- DTE: 7
- Rebalance: Weekly
- Max concurrent: 50
- Universe: Top 20% liquidity (see [v1_universe_protocol.md](v1_universe_protocol.md))
- Exit: Hold to expiry (backtest only)
- Engine: SurfaceRunner path
- Sizing: Equal max-loss per trade with budget caps

---

## Fill assumptions

Run each tier at **three fill levels** where supported (`FillAssumption` in surface path):

| Level | Label | Use |
|-------|-------|-----|
| Optimistic | Mid | Upper bound; not used for go/no-go |
| Base | Mid + partial spread penalty | Secondary |
| **Primary** | **Conservative / cross** | **Go/no-go uses this** |

Go/no-go decisions use **conservative fills only**. If Tier B is not acceptable under conservative fills, the strategy does not advance to paper trading without a documented change to spec or economics.

---

## Metrics (to populate after first baseline run)

**Ranking for go/no-go (Tier B, conservative fills):**

1. **Sharpe ratio** — primary risk-adjusted metric for comparing runs and structure choices.
2. **Return on max-loss budget** — primary economic metric (aligns with sizing).
3. Max drawdown, concentration, and cost sensitivity — gates and diagnostics.

Record at minimum for Tier A and Tier B (conservative fills):

| Metric | Tier A | Tier B (go/no-go) | Notes |
|--------|--------|-------------------|-------|
| **Sharpe ratio** | ✓ | ✓ | **Primary risk-adjusted metric** |
| Return on max-loss budget | ✓ | ✓ | Primary economic metric |
| CAGR / annualized return | ✓ | ✓ | Secondary |
| Max drawdown | ✓ | ✓ | Hard limit TBD |
| Win rate / profit factor | ✓ | ✓ | Diagnostic |
| Avg concurrent positions | ✓ | ✓ | Should respect `max_names_per_side` per direction (e.g. ≤25+25) |
| Turnover (names/week) | ✓ | ✓ | Feeds ops model |
| Top-5 name PnL concentration | ✓ | ✓ | Reject single-name dominance |
| Harsh vs mid return delta | — | ✓ | Cost sensitivity |

**Pass/fail thresholds:** `[TBD after Sprint 004 baseline]`

---

## Go / no-go gate (paper trading)

All required before paper:

- [ ] Tier B (2020 → latest) backtest complete on canonical path
- [ ] Conservative-fill Tier B metrics recorded in sprint memo
- [ ] Pass/fail thresholds applied (or explicit waiver documented)
- [ ] Known gaps documented (hold-to-expiry vs live exit, fill model)
- [ ] Shadow mode trade counts within ops tolerance ([v1_ops_model.md](v1_ops_model.md))
- [ ] At least one verification sprint on payoff / max-loss (Sprint 001+)

---

## Sensitivity matrix (after baseline)

Do not run until weekly + conservative baseline exists:

| Dimension | Values |
|-----------|--------|
| Rebalance | Weekly (base), bi-weekly, monthly |
| Short structure | ironfly (base), ironcondor |
| Wing rule / delta | Per `BacktestRunConfig` (e.g. closest_delta, wing targets) |
| Universe cutoff | 20% (base), 15%, 25% |
| Max concurrent | 50 (base), 35 |
| Fill | Conservative (base), base, mid |

One dimension at a time; record in sprint memos.

---

## Output artifacts per evaluation run

Store under `C:/MomentumCVG_env/cache/results/` (or configured results dir):

- Run config JSON / dataclass dump
- Per-tier metrics summary (markdown or CSV)
- Sprint memo reference with run ID and date range

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-23 | Initial protocol; Tier B = 2020 → latest |
| 2026-05-23 | Week 0 review: Sharpe as primary risk-adjusted metric; short structure sensitivity |
