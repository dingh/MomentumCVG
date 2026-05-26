# Iron Fly vs Body: PnL Decomposition Framework

## Core idea

The gap between the body (short straddle) and the iron fly is not mysterious.

It must come from a small number of mechanical sources.

The goal is to stop asking only:

- why is the fly worse?

and instead ask:

- what exact terms create the drag?
- when do the wings actually pay for themselves?
- what can be screened or changed to improve the trade design?

---

## 1. Mechanical identity

Let:

- `K` = body strike
- `w` = wing width
- `S_T` = spot at expiry
- `C_body` = short straddle credit
- `D_wings` = cost of long put wing + long call wing
- `EC` = extra execution cost of fly relative to body

### Body PnL
`PnL_body = C_body - |S_T - K|`

### Wings PnL
`PnL_wings = -D_wings + max(K - w - S_T, 0) + max(S_T - K - w, 0)`

### Fly PnL
`PnL_fly = PnL_body + PnL_wings - EC`

### Difference between fly and body
`Delta_PnL = PnL_fly - PnL_body = -D_wings - EC + max(|S_T - K| - w, 0)`

This is the whole story.

---

## 2. Plain-English interpretation

The fly differs from the body in only three ways:

1. **Wing debit**
   - you pay for long put + long call wings

2. **Extra execution cost**
   - more legs and worse spreads can add slippage

3. **Tail recovery**
   - the wings only help if the move exceeds the wing width

So if the trade finishes **inside the wings**, the fly is worse by almost exactly:

`-(wing debit + extra cost)`

That means inside-wing trades are basically a **pure tax**.

Only when the move goes **outside the wings** do the wings start paying you back.

---

## 3. The key quantities to compute

For each matched `(ticker, date, width)` trade, compute:

### A. Wing tax
`Tax = D_wings + EC`

This is how much the fly must recover just to tie the body.

### B. Tail recovery
`Recovery = max(|S_T - K| - w, 0)`

This is how much the wings actually earn back at expiry.

### C. Net preservation
`Preservation = Recovery - Tax`

This is the true value added by the wings relative to the body.

Interpretation:
- `Preservation > 0` → the fly beat the body on this trade
- `Preservation < 0` → the fly underperformed the body

---

## 4. The three trade regions

### Region A: inside the wings
Condition:
`|S_T - K| <= w`

Then:
`Delta_PnL = -Tax`

Interpretation:
- the wings did nothing except cost money
- this is a pure insurance tax region

### Region B: outside the wings, but not enough
Condition:
`w < |S_T - K| <= w + Tax`

Then:
`-Tax < Delta_PnL <= 0`

Interpretation:
- the wings helped
- but not enough to pay for themselves

### Region C: outside the wings enough to matter
Condition:
`|S_T - K| > w + Tax`

Then:
`Delta_PnL > 0`

Interpretation:
- the fly actually outperformed the body
- the protection paid back its cost and more

This three-region split is one of the most important analyses to run.

---

## 5. What each failure mode means

## Case 1: most trades finish inside the wings
Then the fly is losing because the protection almost never pays.

Interpretation:
- the body edge may still be fine
- but the chosen wings are too expensive for the move distribution

Possible responses:
- move wings farther out
- buy cheaper protection
- use a different shell
- only use the fly on names where outside-wing moves are common enough

---

## Case 2: trades go outside the wings, but not far enough
Then the wings do help, but not enough.

Break-even condition:
`|S_T - K| - w > D_wings + EC`

Interpretation:
- outside-wing moves exist
- but not often enough or not large enough

Possible responses:
- identify names with more frequent large moves
- only use the fly when the expected move distribution can support the wing tax

---

## Case 3: the put wing is the main drag
This is very plausible in equities because downside skew is often rich.

Interpretation:
- you may be overpaying for downside protection
- the fly may be failing mainly because of the put wing

Possible responses:
- compute put-wing debit separately from call-wing debit
- compute put-wing payoff separately from call-wing payoff
- consider skew-aware filtering
- later consider condor, one-sided spreads, ETF implementation, or asymmetric structures

---

## Case 4: extra spread cost is the main drag
If raw wing debit is acceptable but spreads are large, then execution is the problem.

Interpretation:
- the economics of protection may be okay
- but the extra legs are too expensive to trade

Possible responses:
- tighter liquidity filters
- longer DTE
- ETF-heavy implementation
- structures with cleaner execution

---

## 6. Useful ratios to compute

### A. Wing cost relative to body credit
`WingCostRatio = D_wings / C_body`

If this is high, the fly starts deeply handicapped.

### B. Extra execution cost relative to body credit
`ExecCostRatio = EC / C_body`

If this is high, the issue may be execution rather than economics.

### C. Total tax relative to body credit
`TotalTaxRatio = (D_wings + EC) / C_body`

This is the total hurdle the fly must overcome.

### D. Tail recovery frequency
`RecoveryFrequency = fraction of trades where |S_T - K| > w`

This tells you how often the wings matter at all.

### E. Net preservation frequency
`PreservationFrequency = fraction of trades where Preservation > 0`

This tells you how often the fly actually beats the body.

---

## 7. What to compute for each matched trade

For every matched `(ticker, date, width)` trade, build these columns:

### Core identity
- `ticker`
- `trade_date`
- `body_strike`
- `wing_width`
- `exit_spot`

### Body terms
- `body_credit`
- `body_total_spread`
- `body_exit_value`
- `body_pnl`

### Wing terms
- `put_wing_debit`
- `call_wing_debit`
- `wing_debit_total`
- `put_wing_payoff`
- `call_wing_payoff`

### Fly terms
- `fly_total_spread`
- `fly_exit_value`
- `fly_pnl`

### Derived terms
- `move = abs(exit_spot - body_strike)`
- `extra_cost = fly_total_spread - body_total_spread`
- `tax = wing_debit_total + extra_cost`
- `tail_recovery = max(move - wing_width, 0)`
- `preservation = tail_recovery - tax`
- `inside_wings = move <= wing_width`
- `outside_wings = move > wing_width`
- `beats_body = preservation > 0`

---

## 8. Group-by analyses to run

## A. By lag window
For each `(min_lag, max_lag)`:
- average wing debit ratio
- average extra cost ratio
- outside-wing frequency
- average excess move beyond wings
- average preservation

Goal:
- identify which signal windows are structurally compatible with the fly

## B. By ticker
For each ticker:
- average preservation
- preservation frequency
- average wing cost ratio
- average outside-wing frequency

Goal:
- identify names where the fly is consistently more compatible

## C. By buckets
Group by:
- ETF vs stock
- liquidity bucket
- skew bucket
- term-structure bucket
- signal-strength bucket
- CVG / IV-HV bucket

Goal:
- identify which observable properties explain better or worse fly conversion

---

## 9. How this helps turn things around

The point of the decomposition is not just explanation.

It is to guide action.

## If tax is too high
Main issue:
- wings are too expensive
- or execution is too expensive

Possible response:
- wider wings
- cheaper structure
- skip expensive names
- longer DTE
- ETF implementation

## If recovery is too rare
Main issue:
- protection almost never pays

Possible response:
- use the fly only on names with larger-move potential
- screen on jumpiness / realized move / term structure / IV-HV / CVG

## If put wing is the problem
Main issue:
- downside skew is too rich

Possible response:
- skew-aware filtering
- one-sided alternatives later
- different shell later

## If the fly is still worse trade-by-trade but deployable
Main issue:
- edge is preserved only partially, but capital efficiency may still justify the structure

Possible response:
- judge the fly not just by trade-by-trade PnL
- also judge it by return on max-loss budget, diversification, and live deployability

---

## 10. Strategic reminder

The fly does **not** need to beat the body trade-by-trade to be useful.

Its job may be to convert:

- a high-edge but unusable body
into
- a lower-edge but actually deployable defined-risk portfolio

So the right questions are:

1. Is the gap too large to justify the capital relief?
2. Is the gap caused by something that can be screened away?
3. Is the gap concentrated in specific windows, tickers, or regimes?
4. Is the remaining fly-compatible subset large enough to support a real portfolio?

---

## 11. Strongest prior to test first

If the gap is really huge, the most likely story is:

1. most trades finish inside the wings
2. the long-wing debit is a constant tax
3. the wings only pay on a minority of observations
4. that tax is especially bad on names with expensive downside skew or weak liquidity
5. some lag windows create a body edge that is too dependent on a clean ATM expression to survive the transformation

This is the first story to try to falsify.

---

## 12. Immediate next steps

- [ ] Build the matched trade table with body and fly terms
- [ ] Compute wing debit, extra cost, tail recovery, and preservation
- [ ] Split trades into:
  - [ ] inside wings
  - [ ] outside but not enough
  - [ ] outside enough to beat body
- [ ] Summarize preservation by lag window
- [ ] Summarize preservation by ticker
- [ ] Summarize preservation by:
  - [ ] liquidity bucket
  - [ ] skew bucket
  - [ ] term-structure bucket
  - [ ] IV-HV / CVG bucket

---

## Practical principle

The useful decomposition is **not** just:

- body PnL
- wing PnL

The useful decomposition is:

- **wing tax** = wing debit + extra cost
- **tail recovery** = payoff beyond the wing width
- **net preservation** = tail recovery − wing tax

That tells you exactly what the fly is doing relative to the body, and what to change next.
