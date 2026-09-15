# Sprint 009 D2 — evidence review

**Status:** `REVIEWED / ACCEPTED`  
**Executed:** 2026-09-14  
**Review annotation (2026-09-14):** Accepted under the Sprint 009 closeout. Findings unchanged from the corrected run. Implementation remains `52be6254ef877a791fcf476e0a82fc663804ffa5`. Output remains `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/`. Older D2 runs stay superseded. This annotation does not rerun the comparison. It closes the diagnostic scope with D0–D2 accepted; original D3/D4 are superseded and not executed.  
**Design:** [`sprint009_d2_design.md`](sprint009_d2_design.md) — **ACCEPTED** at `a34f21e`. Methodology unchanged. Midpoint caveat corrected before implementation.  
**Implementation:** `52be6254ef877a791fcf476e0a82fc663804ffa5`  
**Evidence:** `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/`  
**Supersedes:** `e7a1108` / `C:/MomentumCVG_env/runs/sprint009_d2_20260914T151216Z/`. Those artifacts are preserved and are not the official record.  
**Input:** accepted D1 panel `C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z/` (`5669773`, receipt `READY`). Evidence acceptance `28f5ea4`.  
**Characterization:** Fixed-quantity expiry comparison of the same iron-fly-selected names with and without the existing wings. Not a margin, liquidation, unseen-tail, or uncovered-trading result. The review annotation above accepts this evidence.

---

## Answers

Removing the existing wings, at the official cross quantities, would have added **$107,122.42** of development expiry P&L. That is minus net wing contribution. It is the purchase cost saved, $401,339.13, minus the gross expiry payout forgone, $294,216.71. The body-only book finishes at **+$37,345.69**. The iron fly finishes at **−$69,776.72**.

That gain is not free protection. The wings did pay, and on the worst body losses they reduced the realized loss. They did not pay often enough, or large enough, to cover what they cost. This is not a decision to drop the wings.

### D2-A — When did wings pay, and when did they improve net P&L?

Cross purchase cost, gross expiry payout, and net wing contribution are separate.

| Book | Body P&L | Wing contribution | Iron-fly P&L |
|---|---:|---:|---:|
| Cross, primary | 37,345.69 | −107,122.42 | −69,776.72 |
| Midpoint, diagnostic, same quantities | 120,527.85 | −68,737.46 | 51,790.39 |

Cross purchase cost is $401,339.13 (`w_mid + h_wing`). Midpoint purchase cost is $362,954.17 (`w_mid` only). Gross payout is $294,216.71 either way; settlement does not depend on the fill. Costs saved by removing wings are those purchase costs. Payouts forgone are the gross payout. Net of the two, removing wings gains $107,122.42 at cross and $68,737.46 at midpoint. The extra cross gap is the wing execution concession, already inside the cross purchase cost and not deducted again.

Frequencies are not interchangeable.

| Event | Trades | Dates, including `2020-03-13` as not positive |
|---|---:|---:|
| Gross payout positive | 360 / 2,087 (17.2%) | 155 / 209 (74.2%) |
| Payout exceeded purchase cost | 271 / 2,087 (13.0%) | 46 / 209 (22.0%) |

Exceeded purchase cost is the same event as positive net contribution. Equal to cost is not exceeded. A date can show a positive payout because one name paid, while that date's net contribution is still negative. That is why the date payout rate is high and the date net-improvement rate is not.

All-trade distributions include zeros and losses. Conditional summaries use only the 360 trades with positive payout and are not the book.

| Field | All 2,087 trades: mean / p10 / p50 / p90 | Conditional on positive payout |
|---|---|---|
| Gross payout | 141 / 0 / 0 / 323 | 817 / 73 / 422 / 1,980 |
| Cross purchase cost | 192 / 92 / 171 / 305 | 214 / 101 / 194 / 351 |
| Net contribution | −51 / −285 / −152 / 133 | 603 / −133 / 242 / 1,745 |

The unconditional median payout is zero. Among trades that did pay, the median net contribution is positive, but the 10th percentile is still negative: a payout can fail to cover purchase cost.

Midpoint diagnostic, not a second ranking: midpoint net contribution is positive on 280 / 2,087 trades (13.4%). Midpoint results are diagnostic; this analysis does not establish whether midpoint fills are attainable.

### D2-B — How much did wings reduce the worst observed losses?

Loss avoided is \(\max(-\mathrm{body\ P\&L}, 0) - \max(-\mathrm{fly\ P\&L}, 0)\). It is not gross payout. Negative values are kept. Date-level loss avoided is applied after netting trades within the date. The two sums are not added.

| Unit | Sum of loss avoided |
|---|---:|
| Trades | 120,740.25 |
| Dates | 64,055.26 |

Wings reduced the sum of realized losses and still lost money after their purchase cost. Those statements do not contradict. Loss avoided ignores the cost on trades that were not losses, and it ignores profit given up when a winning body becomes a smaller fly win. Net contribution includes both.

Worst body-only trades, paired with the iron fly on the same name. Ranking is by body cross, more negative first. The paired fly column is not a re-sort.

| Rank | Date | Name | Body cross | Fly cross | Gross payout | Purchase cost | Loss avoided |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 2020-02-21 | MCD | −9,314.52 | −833.33 | 8,888.89 | 407.71 | 8,481.18 |
| 2 | 2023-09-15 | SPLK | −7,310.76 | −833.33 | 6,660.03 | 182.60 | 6,477.42 |
| 3 | 2020-02-21 | MGM | −7,061.97 | −833.33 | 6,420.94 | 192.31 | 6,228.63 |

Those three flew to the iron fly's bounded loss while the uncovered body continued. The full list of 10 is in `worst_events.parquet`.

Worst body-only dates, after netting names:

| Rank | Date | Names | Body cross | Fly cross | Loss avoided |
|---|---|---:|---:|---:|---:|
| 1 | 2020-02-21 | 12 | −45,954.53 | −9,448.35 | 36,506.18 |
| 2 | 2023-10-27 | 11 | −12,717.04 | −7,526.99 | 5,190.05 |
| 3 | 2020-04-03 | 5 | −10,505.54 | −5,573.43 | 4,932.11 |

The iron fly's own worst trade is not the body's worst trade. On 2020-03-27 LUV, the body lost $1,070.18 and the fly lost $2,122.81. Payout was zero. Loss avoided is **−$1,052.63**. The wings increased that realized loss. The fly's own worst date is still 2020-02-21, paired with the same body-date loss above.

Loss concentration is a share of gross losing dollars, not of net P&L. A profitable date does not enter the date denominator.

| List | Worst 10 share | Single worst share | Denominator |
|---|---:|---:|---:|
| Body trades | 10.0% | 1.5% | 635,385.90 |
| Fly trades | 3.9% | 0.5% | 514,645.65 |
| Body dates | 44.4% | 16.3% | 281,452.37 |
| Fly dates | 27.4% | 4.3% | 217,397.11 |

The fly's worst trade losses are smaller and less concentrated than the body's. That is the observed bound. It is not a proof about unseen tails, and it is not a margin result.

### D2-C — Cumulative profit, drawdown, and years

Both series include `2020-03-13` as a zero step. The running peak starts at zero. Maximum drawdown is the most negative cumulative-minus-peak and is reported as a negative number.

| Book | Ending cumulative | Maximum dollar drawdown |
|---|---:|---:|
| Body-only cross | 37,345.69 | −59,351.85 |
| Iron-fly cross | −69,776.72 | −73,405.21 |
| Body-only midpoint, diagnostic | 120,527.85 | −55,205.92 |
| Iron-fly midpoint, diagnostic | 51,790.39 | −16,400.11 |

The body-only cross book ends higher and has a smaller drawdown in this sample. The midpoint diagnostic has a smaller fly drawdown than the cross series. Neither path is an intraperiod equity path or a margin path.

Annual P&L is the year's sum. Year-end cumulative is the endpoint of the continuous development series, so it carries prior years.

| Year | Dates | Zero-short | Trades | Body cross | Fly cross | Net wing | Payout-positive trades | Net-positive trades | Year-end body | Year-end fly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 53 | 1 | 437 | −5,881.12 | −17,202.68 | −11,321.55 | 79 | 62 | −5,881.12 | −17,202.68 |
| 2021 | 52 | 0 | 531 | −6,980.26 | −36,644.46 | −29,664.20 | 100 | 79 | −12,861.38 | −53,847.14 |
| 2022 | 52 | 0 | 611 | 39,864.75 | −601.37 | −40,466.12 | 87 | 58 | 27,003.37 | −54,448.51 |
| 2023 | 52 | 0 | 508 | 10,342.33 | −15,328.22 | −25,670.54 | 94 | 72 | 37,345.69 | −69,776.72 |

The dollar advantage of removing wings is positive every year. It is largest in 2022 at **$40,466**. The wing contribution itself is negative every year, and in 2022 it is **−$40,466**. It is not a one-year artifact, and 2022 is where a large positive body is almost entirely spent on wings that do not pay enough to cover their cost.

The observed protection is concentrated. Date-level loss avoided on 2020-02-21 is $36,506 of the $64,055 date-level total, about 57%. That one date is also 16.3% of body-date gross losing dollars. Protection in this sample is a few large expiry events, not a steady offset to the purchase cost.

---

## What this does not say

These are fixed-quantity expiry outcomes on the iron-fly-selected population. Removing wings does not add names, change strikes, or resize \(Q\). Fees remain zero. Midpoint results are diagnostic; this analysis does not establish whether midpoint fills are attainable.

This evidence does not measure intraperiod mark-to-market losses, margin calls, or liquidation risk. It does not estimate the probability of an unseen tail. A rare payout, including the 2020-02-21 cluster, is a historical frequency. It does not authorize uncovered trading. Under the Sprint 009 closeout, original D3/D4 are superseded and not executed.

---

## Coverage, reconciliation, tests, provenance

| Item | Value |
|---|---|
| Verdict | `READY` |
| Trades / dates | 2,087 / 209, including `2020-03-13` at zero dollars |
| Gates | Provenance, coverage, identity, reconciliation, ranking, calendar, and scope all passed |
| Identity residuals | Explicit at trade, date, annual, and development total. Development cross identity residual \(-1.46\times 10^{-11}\); midpoint identity residual \(2.91\times 10^{-11}\). Both are inside the matching-reference tolerance |
| Component residuals | Every documented saved date dollar component and saved date trade count matches the derived sum. Date dollar residuals are 0. Date trade-count residual is 0. Annual dollar residuals versus saved D1 are at most \(1.46\times 10^{-11}\) (`w_mid`). Annual count residuals are 0 |
| Comparison with `e7a1108` | Headline totals, frequencies, worst-event lists, concentration measures, and drawdowns are unchanged. Nothing in those series moved. The previous directory is preserved |
| Implementation | `52be6254ef877a791fcf476e0a82fc663804ffa5` |
| D1 code SHA | `5669773f356f6c33cef86bd0da30ce4051709a6b` |
| D1 receipt | `READY` |
| Generated | `2026-09-14T15:31:06Z` directory timestamp `20260914T153106Z` |
| Output | `C:/MomentumCVG_env/runs/sprint009_d2_20260914T153106Z/` |
| Previous output | `C:/MomentumCVG_env/runs/sprint009_d2_20260914T151216Z/` — preserved, not official |
| D1 directory writes | none |

Pinned input hashes matched:

| File | SHA-256 |
|---|---|
| `execution_receipt.json` | `6acdc0c54732766352732bab1a91c47d53b0d84e4b641614ac660f53f457283d` |
| `trade_decomposition.parquet` | `72da77c10d3e8ba403012ffad785e5338d69be1f2e180135e3414e03f3a7213b` |
| `date_decomposition.parquet` | `49db7b233ee628551ec6fffd05ac76fc07e2b115e9d03019cb55ad054c6cf611` |
| `annual_decomposition.parquet` | `62be4856076fa5485d51a945dc15b83b358fc37cf513df71fad0abd1eabec607` |

```text
$env:PYTHONPATH = "C:\MomentumCVG"
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d2_protection.py
```

```text
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_sprint009_d2_protection_comparison.py -q
```

8 passed before the official run. The valid control uses the same `readiness_verdict` as the runner. Scaling saved dollar columns still fails reconciliation and does not change \(Q\). A quantity mismatch against the source D1 row fails coverage and stays `BLOCKED`. A saved date `h_body` mismatch, which the previous body/fly-only date check would have missed, fails reconciliation. A saved date trade-count mismatch does the same. A zero gross-loss denominator stays null and does not drop trades. Drawdown starts from an initial peak of zero.

Largest absolute residuals on the official book, using the matching reference and the accepted dollar tolerance:

| Level | Largest absolute residual |
|---|---|
| Trade versus saved D1 | \(1.01\times 10^{-11}\) (`residual_fly_vs_official`) |
| Date dollar components versus saved D1 | 0 for `b_mid`, `h_body`, `w_mid`, `h_wing`, `w_pay`, `p_body_cross`, and `p_fly_cross` |
| Date trade count versus saved `n_trades` | 0 |
| Date midpoint fly versus summed trade `pnl_mid_at_cross_q` | \(3.64\times 10^{-12}\) |
| Annual dollar components versus saved D1 | \(1.46\times 10^{-11}\) (`w_mid`) |
| Annual counts versus saved D1 | 0 for `n_dates`, `n_zero_short_dates`, and `n_trades` |
| Development total, cross identity | \(-1.46\times 10^{-11}\) against fly \(-$69,776.72\) |
| Development total, midpoint identity | \(2.91\times 10^{-11}\) against fly midpoint \($51,790.39\) |

None of those residuals failed a gate. A discrepancy outside the matching tolerance, or a nonzero count residual, reaches `BLOCKED` through `readiness_verdict`.

Charts `cumulative_cross.png` and `annual_cross.png` were written only because every gate passed. They stay in the external evidence directory.
