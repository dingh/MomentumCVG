# Sprint 006 baseline decision report

This pack summarizes the frozen mid+cross Surface baseline for the contract-defined full-history and primary windows. **Cross** is the primary economic view; **mid** is a fill-assumption diagnostic. No go/no-go conclusion is declared here.

experiment_id=`sprint006_baseline_v1` · contract_id=`sprint006_baseline_v1` · repo_sha=`5c31e4903a345f496eaca90d81981f3bc6c468e7` · result_complete=`True`

## Cross (primary)

| window | view | mean_CAR / compounded | annualized_return | sharpe | drawdown | n_traded / complete |
|---|---|---|---|---|---|---|
| full_history | A conditional | 0.106147 | n/a | 30.1232 | 0 | 2 |
| full_history | B calendar | 0.223239 | 187.494 | 30.1232 | 0 | true |
| primary | A conditional | 0.106147 | n/a | 30.1232 | 0 | 2 |
| primary | B calendar | 0.223239 | 187.494 | 30.1232 | 0 | true |

## Mid (diagnostic)

| window | view | mean_CAR / compounded | annualized_return | sharpe | drawdown | n_traded / complete |
|---|---|---|---|---|---|---|
| full_history | A conditional | 0.166321 | n/a | 43.3286 | 0 | 2 |
| full_history | B calendar | 0.359922 | 2959.51 | 43.3286 | 0 | true |
| primary | A conditional | 0.166321 | n/a | 43.3286 | 0 | 2 |
| primary | B calendar | 0.359922 | 2959.51 | 43.3286 | 0 | true |

## Completeness and date classes

| fill | window | expected | traded | valid_no_trade | failed | complete |
|---|---|---|---|---|---|---|
| cross | full_history | 2 | 2 | 0 | 0 | True |
| cross | primary | 2 | 2 | 0 | 0 | True |
| mid | full_history | 2 | 2 | 0 | 0 | True |
| mid | primary | 2 | 2 | 0 | 0 | True |

## Weekly diagnostics (cross)

| window | win_rate | profit_factor | no_trade_frequency |
|---|---|---|---|
| full_history | 1 | Infinity | 0 |
| primary | 1 | Infinity | 0 |

## Yearly diagnostics (cross, primary window)

| year | n_expected | n_traded | compounded | annualized_return | sharpe | drawdown |
|---|---|---|---|---|---|---|
| 2022 | 2 | 2 | 0.223239 | 187.494 | 30.1232 | 0 |

## Long / short attribution (cross, primary)

- long mean cycle return: 0.472348; short mean cycle return: -0.163478
- long PnL: 7010.24; short PnL: -3269.55

## Activity and concentration

- primary cross activity: mean included names/traded date=31.5, turnover_complete=true, mean_turnover_names=31.5

| ticker | abs_pnl | share |
|---|---|---|
| PANW | 1230.54 | 0.0683298 |
| MGM | 1179.49 | 0.0654949 |
| SAVA | 970.919 | 0.0539135 |
| LCID | 815.104 | 0.0452613 |
| SIG | 769.231 | 0.042714 |

top-5 |PnL| aggregate share (primary cross): 0.275713

## Structure-failure counts

| fill | window | metadata_error | missing_quotes_or_body | wing_or_liquidity_selection | other_structure |
|---|---|---|---|---|---|
| cross | full_history | 0 | 3 | 10 | 0 |
| cross | primary | 0 | 3 | 10 | 0 |
| mid | full_history | 0 | 3 | 10 | 0 |
| mid | primary | 0 | 3 | 10 | 0 |

## Funnel totals (cross)

| window | expected | feature_covered | mean jointly eligible | sum included |
|---|---|---|---|---|
| full_history | 2 | 2 | 373.5 | 63 |
| primary | 2 | 2 | 373.5 | 63 |

_Selection-bias notice:_ Post-signal candidate means the name already passed the Momentum-tail and within-side CVG filters; these artifacts cannot support full-universe Momentum IC or CVG increment tests.

## Mid-versus-cross fill-assumption sensitivity

Cross-minus-mid is **not** a pure transaction-cost number: fills can also change sizing, inclusion, and selected structures.

| window | both traded | mid-only dates | cross-only dates | mid-only candidates | cross-only candidates | mean cross-minus-mid CAR | mean cross-minus-mid PnL | mean spread_cost_ratio cross | mean spread_cost_ratio mid | mean leg_spread_to_credit cross | mean leg_spread_to_credit mid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_history | 2 | 0 | 0 | 0 | 0 | -0.0601738 | -1208.99 | 0.0446099 | 0 | 0.116995 | 0.109013 |
| primary | 2 | 0 | 0 | 0 | 0 | -0.0601738 | -1208.99 | 0.0446099 | 0 | 0.116995 | 0.109013 |

## Limitations

- Hold-to-expiry; positions are not managed intra-week.
- No earnings filter.
- Iron-fly wings use below-nearest 0.15-delta selection.
- Tier A sizing is not integer lots.
- Long-only fallback dates are possible.
- Mid is a fill-assumption diagnostic, not a pure transaction-cost attribution.
- robust_score is not a decision metric and is not used for go/no-go.
- Post-signal candidate means the name already passed the Momentum-tail and within-side CVG filters; these artifacts cannot support full-universe Momentum IC or CVG increment tests.
