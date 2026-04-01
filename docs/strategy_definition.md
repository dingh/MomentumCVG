# Strategy Definition Outline

## 0. One-line strategy summary

A cross-sectional options strategy that ranks liquid stock and ETF underlyings, selects the best candidates from a dynamically updated universe, and trades symmetric iron flies sized by max-loss budget to maximize return on capital under realistic liquidity, cost, and risk constraints.

---

## 1. Strategy objective

### 1.1 Primary objective

* Maximize expected return on max-loss budget.

### 1.2 Secondary objectives

* Preserve diversification by spreading risk across many names.
* Keep tail risk bounded through defined-risk structures.
* Maintain operational simplicity for first live deployment.
* Use a process that can be executed repeatedly with minimal discretion.

### 1.3 Non-goals for v1 live

* Maximizing raw gross return.
* Running the most complex possible model.
* Using discretionary overrides.
* Optimizing every possible hyperparameter before launch.

---

## 2. Strategy scope

### 2.1 Instruments

* US single-name stock options
* US ETF options

### 2.2 Structure traded in v1

* Symmetric iron fly only

### 2.3 Underlying universe

* Dynamically selected liquid stocks and ETFs
* Universe determined point-in-time using only information available at rebalance date

### 2.4 Tenor scope

* One fixed DTE bucket for v1 live
* Additional DTEs may be researched later but are outside initial launch scope

### 2.5 Geographic / market scope

* US-listed options only

---

## 3. Before the trade: data and signal layer

### 3.1 Data inputs

* Daily option chain data
* Underlying price data
* Option liquidity fields: volume, open interest, bid, ask, spreads
* Precomputed straddle-based features
* Any conditioning variables used by the ranking model

### 3.2 Point-in-time data policy

* All features, universe rules, and eligibility filters must be computable using only information known at decision time
* Precompute universe is an engineering superset only
* Trading universe must be dynamically rebuilt in point-in-time fashion

### 3.3 Alpha signal definition

* Primary ranking signal: [to pin down]
* Conditioning variable(s), if any: [to pin down]
* Signal output should be a cross-sectional ranking score, not necessarily a calibrated return forecast

### 3.4 Signal purpose

* Identify names where the defined-risk short-vol structure is most likely to deliver positive return on max-loss budget

### 3.5 Research assumptions to verify

* Signal remains useful on dynamic liquid stock + ETF universe
* Signal remains useful after transformation from straddle thesis to iron-fly implementation
* Signal survives realistic transaction cost assumptions

---

## 4. Before the trade: tradability and structure selection

### 4.1 Universe construction

* Start from dynamically selected liquid stock + ETF universe
* Apply liquidity requirements appropriate for the chosen DTE and structure

### 4.2 Trade eligibility

A name is eligible only if:

* Required expiry exists in target DTE bucket
* Valid ATM short call and short put exist
* Valid long wings exist under wing-selection rule
* Quotes pass basic sanity checks
* Liquidity passes minimum thresholds

### 4.3 Short strike selection

* ATM strike selection rule: [to pin down]

### 4.4 Long wing selection

* Symmetric width only
* Width chosen to be as wide as possible subject to:

  * delta cap
  * spread cap
  * quote sanity checks
  * optional OI / volume minimums

### 4.5 Trade economics captured per candidate

* Entry credit
* Width
* Max loss
* Margin / capital usage proxy
* Liquidity and spread diagnostics

---

## 5. Before the trade: risk and cost model

### 5.1 Risk unit

* Per-trade risk unit = max loss

### 5.2 Portfolio risk budget

* Total strategy budget defined in terms of max-loss budget and/or margin budget

### 5.3 Cost model

* Option execution cost modeled via effective spread assumption
* Backtest must be robust under worse-than-base cost assumptions
* Any stock hedge cost included only if hedging is part of strategy spec

### 5.4 Risk constraints to encode

* Max gross max-loss budget
* Max names
* Max position per underlying
* Max position per sector / cluster
* Liquidity-based position caps
* Event exclusions if needed

---

## 6. During the trade: portfolio construction

### 6.1 Candidate ranking

* Rank all eligible names by signal score

### 6.2 Portfolio selection

* Select top-ranked names subject to portfolio constraints

### 6.3 Position sizing

#### v1 baseline

* Equal max-loss budget per trade

#### v2 candidate

* Signal-weighted max-loss budget with clipping and caps

#### later

* Constrained optimizer maximizing expected return on max-loss budget subject to concentration, liquidity, and turnover controls

### 6.4 Portfolio construction objective

* Maximize expected portfolio return on max-loss budget while preserving diversification and respecting liquidity and concentration constraints

### 6.5 Hedging policy

* No additional systematic hedge in v1 unless explicitly specified
* Any hedge must justify its effect on expected return, cost, and complexity

---

## 7. During the trade: execution policy

### 7.1 Rebalance timing

* Signal freeze time: [to pin down]
* Strike selection time: [to pin down]
* Order submission time: [to pin down]

### 7.2 Order style

* Combo order vs legged execution: [to pin down]

### 7.3 Pricing policy

* Entry price anchor: [to pin down]
* Order adjustment / walk policy: [to pin down]
* Cancellation policy: [to pin down]

### 7.4 Exception handling

* No-fill rule
* Partial-fill rule
* Bad-data rule
* Margin-change rule
* Expiration / assignment rule

---

## 8. After the trade: evaluation and attribution

### 8.1 Core performance metrics

* Return on max-loss budget
* Sharpe ratio
* Max drawdown
* Sortino / Calmar if useful
* Worst month / worst trade cluster
* Turnover

### 8.2 Attribution

* Selection effect
* Sizing effect
* Cost / slippage effect
* Sector / cluster contribution
* Realized vs expected edge

### 8.3 Monitoring questions

* Did the signal choose the right names?
* Did sizing improve or hurt results?
* Did costs exceed assumptions?
* Did diversification behave as expected?

---

## 9. Validation protocol

### 9.1 Backtest requirements

* Dynamic point-in-time universe
* Full iron-fly instrument backtest
* Realistic cost assumptions
* Margin / max-loss-normalized returns

### 9.2 Robustness requirements

* Subperiod robustness
* Cost sensitivity
* Liquidity filter sensitivity
* DTE sensitivity
* Wing-rule sensitivity
* Concentration diagnostics

### 9.3 Shadow trading requirements

* Live signal generation without manual intervention
* Trade blotter logging
* Simulated fills
* Comparison of live shadow behavior vs backtest assumptions

---

## 10. v1 live deployment boundaries

### 10.1 What is included

* Dynamic liquid universe
* One signal
* One DTE bucket
* Symmetric iron fly only
* Equal max-loss sizing
* Hard caps
* Tiny live capital

### 10.2 What is excluded from v1 live

* Long straddle sleeve
* Regime-dependent structure switching
* Advanced optimizer
* Complex multi-signal aggregation
* Discretionary overrides

---

## 11. Open decisions to pin down next

1. Exact primary signal definition
2. Exact DTE bucket for v1
3. Exact ATM short-strike rule
4. Exact long-wing selection thresholds
5. Exact portfolio size and max-loss budget
6. Exact cost model used for research and live expectations
7. Exact execution workflow
8. Exact event exclusions and kill switches
2