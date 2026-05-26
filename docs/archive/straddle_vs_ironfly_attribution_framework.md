# Straddle vs. Iron Fly Attribution Framework

## Goal

The goal is to understand **why a straddle outperforms an iron fly or butterfly** when both are entered from the same signal.

For each signal date and ticker, create both trades:

- the reference trade: **straddle**
- the alternative trade: **iron fly / butterfly**

Keep the following fixed:

- same entry date and exit date
- same underlying
- same DTE
- same center strike rule
- same signal
- same capital budget rule

Define the structure-switch loss for trade `i` as:

```text
L_i = PnL_straddle_i - PnL_fly_i
```

The main question becomes:

**When `L_i` is large, what explains it?**

---

## 1. Separate signal loss from instrument loss

Before decomposing Greeks or payoff geometry, first ask:

**Did the signal stop working, or did the payoff map stop monetizing the signal?**

### A. Rank correlation test
For each rebalance date, compute the cross-sectional correlation between the signal and future return:

- once using straddle returns
- once using fly returns

If the correlation is still decent for the fly, but fly portfolio PnL is much worse, then the problem is mostly **instrument/payoff compression**, not signal failure.

### B. Top-minus-bottom spread
Sort names by signal and compute:

- top bucket minus bottom bucket using straddles
- top bucket minus bottom bucket using flies

If the fly version has a much smaller spread, then the structure is compressing return dispersion.

This is the fastest way to tell whether the loss is happening because the **edge disappears** or because the **structure clips it**.

---

## 2. Tail-move / capped payoff attribution

### What to measure
For each trade, record:

- final move: `abs(S_T / S_0 - 1)`
- move in volatility units: `abs(log(S_T / S_0)) / (sigma_0 * sqrt(T))`
- whether spot ended outside the wings
- maximum excursion during the holding period

### Test
Bucket trades by move size, for example:

- small move
- medium move
- large move
- extreme move

Then compute average `L_i` in each bucket.

### Interpretation
If the fly mainly underperforms on large-move trades, especially when spot finishes outside the wings, then the lost edge is mainly **tail exposure / uncapped convexity**.

A more direct version:

- price both trades at expiry using the same final spot
- ignore transaction costs
- compare payoff difference

That isolates payoff geometry very cleanly.

---

## 3. Vega / IV repricing attribution

This tests whether the straddle wins because it captures post-entry IV changes better.

### What to measure
For each trade, record:

- change in ATM IV from entry to exit
- change in term structure
- change in skew
- entry vega of both structures

### Test
Do a **spot-frozen repricing**:

- keep spot fixed at entry spot `S_0`
- change only the IV surface from entry to exit
- reprice the straddle and the fly

Define:

```text
L_vega_i = [V_straddle(S_0, sigma_1) - V_straddle(S_0, sigma_0)]
         - [V_fly(S_0, sigma_1) - V_fly(S_0, sigma_0)]
```

### Interpretation
If this is large, then the straddle edge is partly a **vega edge** and the fly is not carrying enough of it.

This is very important for defined-risk structures because they often have much smaller and less stable vega exposure.

---

## 4. Gamma / realized-path attribution

This asks whether the straddle wins because it benefits more from the realized path of spot moves.

### What to measure
For each trade, record:

- realized volatility during the holding window
- max intraperiod excursion
- number of big daily moves
- path statistics such as realized variance and jump proxies

### Test
Do an **IV-frozen revaluation**:

- freeze the IV surface at entry
- replay the actual daily spot path
- optionally delta hedge both structures with the same hedge rule
- mark PnL through time

### Interpretation
If the straddle still beats the fly under frozen IV, then the loss is mainly about **gamma / convexity / path monetization**, not vega.

Butterflies often struggle here because their useful convexity is concentrated near the body strike.

---

## 5. Carry / theta attribution

This asks whether the hold period itself hurts the fly relative to the straddle.

### What to measure
For each trade, record:

- days held
- entry theta
- theta as a fraction of premium or risk
- whether the holding period overlaps a weekend or event

### Test
Do a **spot-and-IV frozen time roll**:

- keep spot fixed
- keep IV fixed
- move time forward from entry to exit
- reprice both structures each day

### Interpretation
This tells you how much of the performance gap comes from **carry profile**, independent of spot moves and vol changes.

---

## 6. Event-gap monetization attribution

This is a special case of tail/path, but worth isolating.

### What to measure
For each trade, flag:

- earnings overlap
- macro-event overlap
- overnight gap size
- whether a one-day jump crosses the wings

### Test
Split the sample into:

- event trades
- non-event trades

Then compare average `L_i`.

### Interpretation
If the fly only looks bad around event windows, then the lost edge is specifically **jump convexity**.

That would suggest the structure may still work away from events, but not when the original straddle edge comes from discrete jumps.

---

## 7. Wing / skew contamination attribution

This asks whether adding the wings is polluting the clean ATM signal.

### What to measure
For each trade, record:

- entry skew
- wing IV relative to body IV
- width of wings
- slope and curvature of the smile

### Tests

#### A. Regress `L_i` on skew variables
If fly underperformance is strongly related to steep skew or expensive wings, then wing pricing is part of the damage.

#### B. Sensitivity to wing choice
Rebuild the fly with several rules:

- fixed dollar width
- fixed delta wings
- fixed vol-distance wings
- wider wings
- narrower wings

If performance changes a lot with wing placement, then a big part of the loss is **wing design / skew contamination**, not the signal itself.

This is one of the most practical tests for the project.

---

## 8. Execution-cost attribution

This is very important because a fly has more legs.

### What to measure
For each trade, record:

- quoted spread per leg
- fill slippage
- open interest / volume
- effective spread paid
- time to fill

### Test
Compute PnL under two assumptions:

- **mid-to-mid**
- **realistic fill model**

Define:

```text
L_exec_i = (PnL_straddle_mid - PnL_fly_mid)
         - (PnL_straddle_real - PnL_fly_real)
```

### Interpretation
If the fly only looks bad after realistic fills, then the issue is not economics, it is **execution drag**.

For a four-leg trade, this can be large.

---

## 9. Capital-normalization attribution

Sometimes the apparent edge loss is partly a measurement artifact.

### What to test
Evaluate both structures using several denominators:

- premium paid / received
- max loss
- broker margin or equity-with-loan change
- vega target
- dollar gamma target
- equal notional center exposure

### Interpretation
If the fly looks bad under one denominator but acceptable under another, then part of the problem is **sizing / normalization**, not just raw trade quality.

This is especially important for short defined-risk structures.

---

## 10. Practical attribution workflow

### Step 1: Build the matched panel
For each matched pair, store:

- ticker, date, signal
- structure type
- body strike, wing width, DTE
- entry/exit prices
- mid and realistic fill prices
- entry Greeks
- entry IV surface summary
- exit IV surface summary
- realized move stats
- event flags

### Step 2: Compute these PnL versions
For each matched pair, compute:

1. **Actual PnL**
2. **Mid-price PnL**
3. **Expiry payoff only**
4. **Spot-frozen vol-shock PnL**
5. **IV-frozen spot-path PnL**
6. **Spot-and-IV frozen theta roll**
7. **Alternative wing constructions**

### Step 3: Build a waterfall
For the average trade, create a waterfall like:

- total straddle advantage
- minus payoff clipping
- minus vega loss
- minus gamma/path loss
- minus carry difference
- minus wing/skew contamination
- minus execution drag
- minus sizing effect

This gives a useful first attribution.

---

## 11. Important warning

These pieces are **not perfectly additive**. Vega, gamma, and skew interact.

If a rough but useful decomposition is enough, a simple waterfall is fine.

If you want a more principled decomposition, use **Shapley attribution** across the components:

- geometry
- vol shock
- path
- carry
- execution
- sizing

This is slower, but cleaner.

---

## 12. Likely major contributors for this setup

For this project, the most likely contributors to fly underperformance are:

1. **tail clipping**
2. **loss of clean vega exposure**
3. **return compression in the cross section**
4. **execution drag from extra legs**

So if prioritizing implementation, start with these five diagnostics:

- rank IC: straddle vs. fly
- average `L_i` by move-size bucket
- spot-frozen vol-shock attribution
- IV-frozen path attribution
- mid vs. realistic fill attribution

Those five will probably explain most of the gap.

---

## 13. Recommended next step

Turn this framework into a concrete implementation plan for the repo:

- exact fields to store in the matched-trade table
- functions to compute each counterfactual PnL
- plots and tables for attribution reporting
- a minimum viable first pass vs. later extensions
