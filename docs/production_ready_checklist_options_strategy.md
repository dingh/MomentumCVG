# Production-Ready Checklist for My First Live Options Strategy

Based on the current project state, the first live strategy should be a **narrow, boring, controlled version** of the bigger idea:

**dynamic liquid stock/ETF universe → rank candidates → trade only symmetric iron flies → size by max-loss budget → strict caps → tiny live deployment**

---

## A. Strategy definition is frozen

- [ ] One clear strategy spec exists in writing.
- [ ] One signal version is selected as the live candidate.
- [ ] One structure is selected for v1 live: **symmetric iron fly only**.
- [ ] One DTE bucket is selected for v1 live.
- [ ] One wing-selection rule is selected and frozen.
- [ ] One return definition is selected: **return on max-loss budget**.
- [ ] One portfolio objective is selected: **maximize return on max-loss budget, subject to caps**.

### Notes
- Freeze the protocol before more tuning.
- If the protocol changes materially, rerun the full evaluation.

---

## B. Data and research integrity are clean

- [ ] Universe is **dynamic and point-in-time**, with no leakage.
- [ ] The same universe logic used in backtest can be used in production.
- [ ] The precompute universe is only an engineering superset, not the trading universe.
- [ ] The exact live inputs are reproducible from saved data snapshots or point-in-time files.
- [ ] Feature generation is deterministic and reproducible.
- [ ] Trade selection for date `t` uses only information known at `t`.

### Notes
- Do not go live with the mocked leaking universe.
- Treat point-in-time universe construction as a hard gate.

---

## C. Iron-fly instrument layer is real, not proxied

- [ ] Backtest uses **iron-fly prices**, not straddle proxy returns.
- [ ] ATM short strike selection is explicitly defined.
- [ ] Long-wing selection rule is explicitly defined and backtested.
- [ ] Per-trade **max loss** is stored and used everywhere.
- [ ] Entry credit, width, and max loss are all saved per trade.
- [ ] If no valid symmetric fly exists, the ticker is skipped.
- [ ] Exit rule is defined clearly:
  - [ ] hold to expiry, or
  - [ ] exit at fixed DTE remaining, or
  - [ ] exit at profit / loss thresholds.

### Notes
- The edge now lives inside the defined-risk shell, not in the straddle proxy.
- Keep the structure logic explicit and reproducible.

---

## D. Portfolio construction layer exists

### Baseline portfolio layer

- [ ] **Equal max-loss per trade** is implemented.
- [ ] Total max-loss budget is fixed.
- [ ] Max number of names is fixed.
- [ ] Per-name max-loss cap is enforced.
- [ ] Sector / cluster cap is enforced.
- [ ] Liquidity cap is enforced.
- [ ] Contract rounding logic is implemented.

### Upgrade layer

- [ ] Signal-weighted max-loss allocation is implemented.
- [ ] Weight clipping is implemented so one name cannot dominate.
- [ ] Turnover penalty or turnover cap exists.

### Optional later layer

- [ ] Constrained optimizer exists for expected return on max-loss budget.
- [ ] Objective includes concentration / variance / turnover penalties.

### Notes
- Start with equal max-loss or simple clipped signal weighting.
- Do not start live with the optimizer.

---

## E. Cost model is realistic

- [ ] Backtest includes option spread costs.
- [ ] Effective spread assumption is explicitly defined.
- [ ] Sensitivity is tested at multiple spread levels.
- [ ] Spread assumptions are harsher for thinner names.
- [ ] Stock hedge costs are included if delta hedging is used.
- [ ] Margin-based performance is reported, not just premium-based performance.
- [ ] PnL is robust under worse-than-base slippage assumptions.

### Notes
- Gross alpha is not enough.
- The strategy must survive realistic cost assumptions.

---

## F. Ranking logic is good enough for production

- [ ] Signal ranking is stable out of sample.
- [ ] Ranking performance is better than a naive baseline.
- [ ] The signal is not dependent on a tiny set of names.
- [ ] Performance survives on the new liquid stock/ETF universe.
- [ ] Conditioning variables, if used, improve **cost-adjusted** performance and not just gross performance.
- [ ] Signal logic is simple enough to explain in one page.

### Notes
- For v1 live, prefer one primary ranking signal and maybe one conditioning filter.
- Do not launch with a large blended alpha stack.

---

## G. Risk framework is explicit

- [ ] Total max-loss budget cap
- [ ] Max margin usage cap
- [ ] Max number of positions
- [ ] Max loss per name
- [ ] Max loss per sector/theme
- [ ] Max live gross short-vol exposure
- [ ] Earnings / corporate-action exclusion rule
- [ ] Vol-shock reduction rule
- [ ] Drawdown-based de-risking rule
- [ ] Hard kill switch for data / order failures

### Notes
- A signal without a risk framework is not production-ready.
- Keep the first version simple and enforceable.

---

## H. Execution process is specified end to end

- [ ] What exact time are signals frozen?
- [ ] What exact time are strikes selected?
- [ ] Are orders sent as combo orders or legged?
- [ ] What price do I anchor to?
- [ ] How far from mid do I start?
- [ ] How do I walk the order if not filled?
- [ ] How long do I wait before canceling?
- [ ] What happens on partial fills?
- [ ] What happens if one side fills and the other does not?
- [ ] What happens near expiration?
- [ ] What happens on early assignment?
- [ ] What happens when margin changes intraday?

### Notes
- This is part of the strategy, not admin work.
- If these questions do not have answers yet, the strategy is not live-ready.

---

## I. Robustness tests are passed

- [ ] Leak-free backtest remains positive
- [ ] Full iron-fly backtest remains positive
- [ ] Margin-adjusted / max-loss-adjusted returns remain positive
- [ ] Strategy works in more than one subperiod
- [ ] Strategy survives worse cost assumptions
- [ ] Strategy survives different liquidity cutoffs
- [ ] Strategy survives small changes in DTE
- [ ] Strategy survives small changes in wing rule
- [ ] Results are not dominated by a handful of names
- [ ] Drawdown is tolerable for real capital

### Notes
- If it only works in one exact configuration, it is not ready.

---

## J. Attribution exists

- [ ] Can I tell whether PnL came from selection or sizing?
- [ ] Can I tell whether PnL came from signal edge or execution luck?
- [ ] Can I tell whether losses came from structure choice, timing, or bad fills?
- [ ] Do I know which names / sectors drove results?
- [ ] Do I know which trades consumed the most max-loss budget?
- [ ] Do I know whether live slippage was better or worse than assumed?

### Notes
- Without attribution, every bad week becomes a story instead of a diagnosis.

---

## K. Shadow trading is complete

- [ ] Strategy runs daily / weekly without manual intervention
- [ ] Signals are generated on schedule
- [ ] Trades are generated on schedule
- [ ] Intended orders are logged
- [ ] Simulated fills are logged
- [ ] Margin use is logged
- [ ] Failures are logged
- [ ] Shadow results are compared with backtest assumptions

### Notes
- Run this long enough to trust the plumbing.

---

## L. Live deployment plan is staged

- [ ] Start with tiny capital
- [ ] Use a reduced name count at first
- [ ] Keep sizing formula identical to backtest
- [ ] Review live vs expected fills every cycle
- [ ] Increase size only if live behavior matches expectations
- [ ] Predefine what would make me stop scaling

### Notes
- The first live trades are a systems validation, not a money-maximization exercise.

---

# Common reasons retail quants stop at backtesting

## 1. They never leave the signal stage
- They keep improving features but never build:
  - portfolio construction
  - execution
  - risk controls
  - attribution

### My defense
- [ ] Do not add more alpha features until Sections D through K are mostly complete.

## 2. They use the wrong denominator
- Premium-normalized returns look great but are often not deployable.

### My defense
- [ ] Keep everything on **max-loss budget / margin basis**.

## 3. They do not fix leakage
- They assume the leak will not matter much.

### My defense
- [ ] Treat point-in-time universe and walk-forward validation as hard gates.

## 4. They ignore costs
- Costs can wipe out unconditional profitability.

### My defense
- [ ] Force every candidate version to survive harsher spread assumptions.

## 5. They launch something too clever
- Too many moving parts means too many hidden failure modes.

### My defense
- [ ] First live version should be:
  - [ ] one signal
  - [ ] one structure
  - [ ] one DTE
  - [ ] one simple portfolio rule

## 6. They do not size like a business
- They size like a backtest.

### My defense
- [ ] Use max-loss budget first
- [ ] Enforce gross caps
- [ ] Add drawdown rules
- [ ] Ramp capital slowly

## 7. They do not build attribution
- Then every result becomes a story instead of a diagnosis.

### My defense
- [ ] Log everything from signal to fill to realized PnL.

---

# Recommended first-live version

- [ ] Dynamic liquid stock + ETF universe
- [ ] One proven ranking signal
- [ ] One fixed DTE
- [ ] Symmetric iron fly only
- [ ] Widest tradable width under delta/spread rule
- [ ] Equal max-loss per trade
- [ ] Strict name / sector / liquidity caps
- [ ] Small number of names
- [ ] No long straddle sleeve yet
- [ ] No regime switching yet
- [ ] Tiny live capital

### Notes
- The goal is not to launch the smartest strategy.
- The goal is to launch the simplest strategy that still captures the edge and can be operated reliably.

---

# Immediate next steps

1. [ ] Finish the leak-free dynamic-universe iron-fly backtest
2. [ ] Implement portfolio construction v1: equal max-loss + caps
3. [ ] Build the execution and shadow-trading workflow

### Notes
- Once these three are done, the project will be much closer to a real live strategy than most retail quant projects ever get.
