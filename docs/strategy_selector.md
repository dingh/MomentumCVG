# 7DTE Defined‑Risk Volatility Strategy Selector (ORATS Near‑EOD)

This document describes a **screening/selection procedure** for choosing a **defined‑risk options strategy** when:

- **DTE is fixed at 7** (weekly cadence)
- Your **signal is volatility** (directionless), i.e., it tells you to **BUY vol** or **SELL vol**
- You want the script to pick:
  - **strategy type** (iron condor / iron fly / reverse iron condor / optional long strangle)
  - **the four legs** (strikes) for the chosen strategy
  - while satisfying hard constraints:
    - **max collateral (capital)**
    - **max total combo spread you’re willing to eat**
    - **defined risk only**

The core idea is to treat this as a **constrained search**:
> Enumerate a small grid of candidate structures and leg parameters for 7DTE, compute **combo bid/ask**, **net credit/debit**, **collateral**, **max loss**, and **liquidity**, filter by your constraints, then **score** and select the best.

---

## 1) Structure menu for 7DTE

### What
For 7DTE, restrict to a small menu that is:
- **directionless** (vol-only)
- **defined risk**
- **common enough** to model reliably from quotes

**If your signal says SELL VOL**
- **Iron Condor (IC)** — short OTM put + short OTM call, each hedged by long wings (defined risk)
- **Iron Fly (IF)** — short ATM-ish straddle hedged by long wings (defined risk, higher gamma)

**If your signal says BUY VOL**
- **Reverse Iron Condor (RIC)** — long OTM put + long OTM call, capped by short further‑OTM wings (defined risk)
- *(Optional)* **Long Strangle** — long OTM put + long OTM call (defined risk, 2 legs, uncapped profit)

### How
Map your volatility signal to a binary “side” for the week:
- `vol_signal_side ∈ {SELL_VOL, BUY_VOL}`

Your candidate generator uses:
- SELL_VOL → {IC, IF}
- BUY_VOL → {RIC, (optional) long strangle}

### Why
- 7DTE concentrates **gamma risk** and **spread friction**.
- A small menu prevents “death by parameter search” and makes results auditable.
- Defined-risk only ensures your sizing and worst-case behavior are controlled.

---

## 2) Combo spread budget (multi‑leg gate)

### What
Your top execution constraint is the **total combo spread** for the multi‑leg order.

### How
For a candidate with legs \(i\):

Compute **combo bid/ask** from leg quotes:

- If leg is **SELL**, you receive bid/ask:
  - contributes `+bid` to `combo_bid` and `+ask` to `combo_ask`
- If leg is **BUY**, you pay ask/bid:
  - contributes `-ask` to `combo_bid` and `-bid` to `combo_ask`

So:
- `combo_bid = Σ(sell_bid) − Σ(buy_ask)`
- `combo_ask = Σ(sell_ask) − Σ(buy_bid)`
- `combo_mid = (combo_bid + combo_ask)/2`
- `combo_spread = combo_ask − combo_bid`

**Hard constraint**
- `combo_spread <= max_total_spread_you_eat`

**Conservative net pricing**
- Credit trades: `net_credit = combo_mid − k*combo_spread − fees`
- Debit trades:  `net_debit  = combo_mid + k*combo_spread + fees`
  - where `k ≈ 0.25–0.40` (slippage fraction)

### Why
- With 4 legs, you effectively “pay” liquidity multiple times.
- This gate prevents selecting trades that look great on mid prices but fail in reality.
- The “net” adjustment makes your selector robust to fill quality.

---

## 3) Candidate generation for 7DTE (small deterministic grids)

### What
You should not search “all strikes.” Instead, generate a finite, repeatable candidate set.

### How

#### Step A — Select the expiry
Pick the expiry nearest to 7 DTE:
- `exp = nearest_expiry(chain, target_dte=7)`

#### Step B — Choose short/long strike distance using delta targets
Use **delta grids** (stable across tickers):

**SELL VOL (Iron Condor / Iron Fly)**
- Condor short deltas: `short_delta_grid = [0.10, 0.12, 0.15, 0.18, 0.20]`
- Iron fly: ATM-ish (≈50Δ) — only if liquidity is excellent

**BUY VOL (Reverse Iron Condor)**
- Long deltas: `long_delta_grid = [0.20, 0.25, 0.30]`
- Caps (short wings): `cap_delta_grid = [0.10, 0.15]`

#### Step C — Choose wing width specification
Two options:

1) **Delta-offset wings (preferred for 7DTE)**
- If short is 15Δ, wing could be ~7Δ (further OTM)

2) **Fixed width (points or % of spot)**
- Width% grid: `[0.5%, 1.0%, 1.5%]` of spot (pick widths that fit your collateral)

This yields a manageable grid:
- IC candidates: `(short_delta, wing_spec)`
- RIC candidates: `(long_delta, cap_spec)`

### Why
- Delta-based selection is consistent across different prices/IV regimes.
- 7DTE chains can be noisy; restricting candidates improves robustness.
- Small grids are easy to debug and unit test.

---

## 4) Defined‑risk economics: collateral + max loss

### What
You need a consistent way to compute:
- **capital required** (collateral)
- **max loss** (worst case)

### How

Let:
- Put side width: `W_put = K_short_put − K_long_put`
- Call side width: `W_call = K_long_call − K_short_call`
- `W = max(W_put, W_call)`

#### Iron Condor (credit)
- Collateral: `collateral ≈ W * 100`
- Max loss: `max_loss = (W − net_credit) * 100`

#### Reverse Iron Condor (debit)
- Max loss: `max_loss = net_debit * 100`
- Capital outlay is the debit paid (also your max loss)

**Hard constraints**
- `collateral <= max_collateral` (IC)
- `max_loss <= max_loss_per_trade` (recommended even if you also cap collateral)

### Why
- These formulas align with defined-risk mechanics and common broker margin treatment.
- Max-loss budgeting is the only reliable way to control risk at 7DTE.

---

## 5) 7DTE liquidity gates (non‑negotiable)

### What
7DTE requires stricter liquidity checks:
- spreads can widen sharply
- wings can be dead
- small mids make percent spreads unreliable

### How

#### Per-leg gates (all legs)
- `leg_spread_pct <= leg_spread_pct_max`
- `mid >= min_mid_price` (avoid penny-ish options)
- `volume >= min_volume` *(optional)*
- `OI >= min_OI` *(at least on shorts; ideally on all legs)*

#### Short leg gates (stricter)
- Both short legs must have tight spreads and decent OI.

#### Combo gate
- `combo_spread <= max_total_spread_you_eat`

### Why
- A single illiquid wing can make the entire combo unfillable.
- 7DTE is sensitive to microstructure; you must force “good markets.”

---

## 6) Economics gates (ensure the trade is worth it)

### What
Even if a trade is liquid and fits collateral, it may be a bad deal after costs.

### How

#### Credit trades (IC/IF)
Require:
- `net_credit > 0`
- `net_credit / W >= min_credit_per_width`

Suggested starting values for 7DTE:
- `min_credit_per_width ≈ 0.25–0.40` (tune by universe and fees)

#### Debit trades (RIC / long strangle)
Require:
- `net_debit <= max_debit_budget` (or `<= max_loss_per_trade/100`)

### Why
- 7DTE exposes you to jump/gamma risk; you need meaningful compensation (credit quality).
- A debit that is too large relative to expected move will bleed.

---

## 7) Scoring: choose the best feasible candidate using expected move

### What
After filtering, you may have multiple candidates. Use a score aligned with your **volatility thesis**, not direction.

### How
Compute an **expected move** to expiry (choose one):
- From IV: `EM = S * IV_ATM * sqrt(T)`
- From your forecast: `EM = S * vol_signal * sqrt(T)`
where `T = 7/365`.

#### SELL VOL (Iron Condor)
Let:
- `d = min(S − K_short_put, K_short_call − S)` (closest short distance)
- `edge = net_credit / collateral`
- `safety = d / EM`
- `cost_ratio = (combo_spread + fees) / max(net_credit, eps)`

Score:
- `score = edge + a*safety − b*cost_ratio`

#### BUY VOL (Reverse Iron Condor)
Approximate breakevens using `net_debit`:
- Put BE: `K_long_put − net_debit`
- Call BE: `K_long_call + net_debit`
- `dbe = min(S − putBE, callBE − S)`

Score:
- `move = EM / max(dbe, eps)`
- `cost_ratio = (combo_spread + fees) / max(net_debit, eps)`
- `score = c*move − d*cost_ratio`

### Why
- For SELL VOL you want price to stay inside shorts: reward distance-to-shorts vs expected move.
- For BUY VOL you need move beyond breakevens: reward expected move vs breakeven distance.
- Cost penalties prevent selecting trades that only work on mid prices.

---

## 8) Practical defaults (V1 settings)

### SELL VOL default (Iron Condor)
- DTE: 7
- short delta: **0.15**
- wing: **0.07** (or width ≈ 1% of spot)
- Require:
  - `combo_spread <= budget`
  - `net_credit/width >= 0.30`
  - strict per-leg spread% caps

### BUY VOL default (Reverse Iron Condor)
- DTE: 7
- long delta: **0.25**
- caps: **0.10–0.15**
- Require:
  - `net_debit <= budget`
  - `combo_spread <= budget`
  - leg liquidity gates

### Why
These defaults are a stable starting point for 7DTE:
- not too close to ATM (reduces gamma blow-up for condors)
- not too far OTM (keeps credit/debit meaningful after costs)

---

## 9) Implementation skeleton (one function)

### What
A single function that takes a chain snapshot and returns the best tradable candidate.

### How
High-level pseudocode:

```python
def pick_trade_7dte(chain, vol_signal_side, constraints, params):
    exp = nearest_expiry(chain, target_dte=7)

    candidates = []
    for struct in structures(vol_signal_side):           # IC/IF or RIC/(strangle)
        for cfg in param_grid(struct, params):          # deltas + wing specs
            legs = build_legs_by_delta(chain, exp, struct, cfg)
            if not legs:
                continue

            cand = evaluate_combo(legs, constraints, params)  # combo bid/ask, net, collateral, max_loss
            if passes_all_constraints(cand, constraints, params):
                cand.score = score_candidate(cand, vol_signal_side, params)
                candidates.append(cand)

    return max(candidates, key=lambda x: x.score) if candidates else None
```

### Why
- Separates responsibilities: **build → evaluate → filter → score**
- Easy to unit test each component
- Produces an auditable decision trail (why skipped, why selected)

---

## 10) ORATS fields you’ll need (minimum set)

From ORATS strikes/chain snapshot (Near‑EOD):
- Keys: `ticker`, `tradeDate`, `expirDate`, `dte`, `strike`, `stockPrice`
- Quotes:
  - `callBidPrice`, `callAskPrice`, `putBidPrice`, `putAskPrice`
- Liquidity:
  - `callOpenInterest`, `putOpenInterest`, `callVolume`, `putVolume`
- Greeks / IV:
  - `smvVol` (or another IV measure), plus `delta`, `gamma`, `theta`, `vega` if available

> You can implement the full selector using just quotes + delta + dte + strike, but Greeks/IV improve expected-move scoring and risk filters.

---

## Summary
This 7DTE selector is designed to answer, mechanically:
- **Can I trade this ticker under my spread + collateral constraints?**
- **Should I buy or sell vol with defined risk?**
- **Which 4 legs (or 2 legs) should I use?**

It does so by combining:
1) a restricted strategy menu,
2) strict multi-leg liquidity gates,
3) defined-risk collateral/max-loss calculations,
4) conservative net pricing after costs,
5) expected-move scoring aligned with vol (not direction).

