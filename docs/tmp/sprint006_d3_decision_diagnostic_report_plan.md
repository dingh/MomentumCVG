# Sprint 006 D3 — Decision-quality report and diagnostic observability plan

**Status:** `PROPOSED — AWAITING ACCEPTANCE`
**Mode:** Build (this document is **design only**; D3 implementation is not authorized until this plan is accepted)
**Original proposal:** `688c2a3` (`docs: propose Sprint 006 D3 decision diagnostic report plan`)
**This revision:** design-only correction of `688c2a3` (including follow-ups `22611ef` and this gap close); implementation still not authorized
**Confirmed ancestors:** D2 acceptance `62bdf38`; D2 implementation `9224068`; D1 `241b0d3` + `c6b1735`
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (unchanged; SHA-256 of committed LF bytes `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715`)
**D0 plan:** [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](sprint006_d0_baseline_experiment_contract_plan.md) (`ACCEPTED — D0 COMPLETE`)
**D1 plan:** [`docs/tmp/sprint006_d1_trusted_baseline_runner_plan.md`](sprint006_d1_trusted_baseline_runner_plan.md) (`ACCEPTED — D1 COMPLETE`)
**D2 plan:** [`docs/tmp/sprint006_d2_eligibility_coverage_correctness_plan.md`](sprint006_d2_eligibility_coverage_correctness_plan.md) (`ACCEPTED — D2 COMPLETE`)
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md`

---

## Plain-language summary

**What D3 will build.** A small, deterministic evaluation pack on top of the already-accepted single-configuration Surface path. After a frozen mid+cross run, D3 writes one machine-readable decision report and one compact Markdown rendering, plus three thin diagnostic tables: a candidate view derived from the existing `trade_log`, a per-leg reconstruction log for constructable post-signal names, and a compact stage-count funnel.

**Why it is needed.** D1/D2 make the run reproducible and stop dates from disappearing. They do not yet answer whether the baseline is economically credible, whether results are concentrated in a few names or years, or whether D4 can reconstruct a trade from signal through settlement. Existing `run_summary` metrics are search-oriented (`robust_score`, Sharpe only on dates that appear in the trade log) and are **not** the D0 decision report.

**What we will have after D3.** For both fills (cross primary, mid diagnostic) and both windows (full history `2018-10-26`→`2026-07-10`, primary `2020-01-01`→`2026-07-10`):

* Conditional-on-traded and calendar-aligned return/risk numbers using the frozen D0 formulas.
* Yearly splits, long/short attribution, drawdown, top-five ticker share of absolute P&L, activity, coverage, and a small structure-failure classification.
* A fill-assumption sensitivity block comparing mid and cross, with required overlap counts and mean spread-cost diagnostics. Cross-minus-mid is **not** a pure transaction-cost number: fills can also change sizing, inclusion, and selected structures.
* A candidate table that says, for each post-signal name, whether it traded, failed structure construction, or was excluded by portfolio rules — without treating those as failed dates.
* A leg log that lets D4 check fills, strikes, expiry payoff, and P&L arithmetic on included trades, with short-iron-fly wing/body signs preserved. Every included straddle has exactly two matching legs and every included iron fly has exactly four. Missing, incomplete, duplicate, or unexpected leg rows abort report generation.
* A funnel that counts how many names survive each existing pipeline stage. Zero means the stage ran and nobody survived; null means the stage was not evaluated (for example, `failed` / `missing_features` dates).
* Turnover that treats `valid_no_trade` as zero selected names. Failed dates, including `missing_features`, are not economic no-trades and block a complete turnover result.

If any expected date is `failed`, the pack is written but **must not** be presented as a complete result — including turnover. If a date classified as `traded` has broken or missing economics, or if any included trade has missing, incomplete, duplicate, or unexpected legs, or if entry/payoff/P&L identities fail to reconcile, report generation **aborts** — it does not drop the date or publish a quieter partial metric.

**What D3 will not answer.** It will not say whether Momentum ranks the full PIT universe, whether CVG adds incremental predictive value, or whether a different parameter set would be better. The persisted candidates have already passed the Momentum-tail and CVG filters, so those questions would be selection-biased. D3 does not run the accepted real dataset, does not inspect live P&L during this design, does not declare a go/no-go from a numeric threshold, and does not compute drop-a-ticker or drop-a-week counterfactuals.

**What remains for D4 and Sprint 007.** D4 executes the frozen command on accepted real artifacts, hand-checks the D0 sample using the new reconstruction tables, and records the conclusion. Sprint 007 is still the later bounded robustness / candidate study — not this pack.

**Implementation shape.** Keep `SurfaceRunner.run_single_config()` as the only economic engine. Add evaluation functions that consume run artifacts; serialize legs and funnel counts at existing runner boundaries (no second backtest). Extend the D1 adapter to persist the new files and receipt digests. One existing CLI; no dashboards. **Three commits:** (1) evaluation calculations and tests; (2) leg-log + funnel capture and reconciliation tests; (3) report integration, deterministic outputs, and documentation.

---

## 1. Context-read receipt

| Path | Why read | Fact used for D3 |
|------|----------|------------------|
| `AGENTS.md` | Session rules | SurfaceRunner is canonical; inspect/plan before edits |
| `docs/agenda/current_sprint.md` | Authorization | `ACTIVE — D0/D1/D2 COMPLETE; D3 DESIGN UNDER REVIEW`; design only |
| `configs/sprint006_baseline_v1.json` | Frozen metrics + windows | Cross primary; dual views; report blocks; periods; `include_diagnostics=true` |
| D0 plan §8.1 / §10 | Exact formulas | Conditional NaN vs calendar 0-fill; win rate; profit factor; Sharpe √52; no `robust_score` go/no-go |
| D1 plan / `sprint006_baseline.py` | Output path | Twin mid/cross; overwrite refusal; receipt digests; D3 still listed as deferred |
| D2 plan / `surface_runner.py` | Calendar authority | `date_status` partition; empty-signal `valid_no_trade`; `_assembly` dropped before persist |
| `surface_metrics.py` | What exists today | Date summary from trade_log only; Sharpe on finite cycle dates; `robust_score` search heuristic |
| `pipeline.py` S2–S5 | Candidate grain | S2 output = post-tail+CVG names; S3 keeps `structure_ok=False`; S5 exclusion vocabulary; settle uses `_assembly`; trade `quantity` is signed by strategy direction |
| `option_surface.py` / `src/core/models.py` | Leg economics | Fill via `buy_price`/`sell_price`; `entry_cost` signed cash; payoff = signed intrinsic; iron-fly unit signs already encode long wings / short bodies; `Position.pnl = exit_value - entry_cost` |
| `scripts/run_sprint006_baseline.py` | Command | One CLI; `--dry-run` does not execute; stdout is paths/row counts |
| `tests/contract/test_run_metrics_contract.py` | Metric tests | Pins current S8 search metrics, not D0 dual views |
| Git | Preconditions | Original D3 proposal `688c2a3`; D2 acceptance `62bdf38` |

**Git state at this revision:** design-only correction of `688c2a3`. No code changes and no economic execution.

---

## 2. Evidence-based current-state assessment

### Already correct (reuse; do not rebuild)

| Capability | Evidence |
|------------|----------|
| Twin mid/cross economic run | D1 adapter → `run_single_config` twice |
| Authoritative expected calendar | D2 `date_status` (`trade_date`, `status`, `reason`) |
| Post-signal candidates persist when diagnostics on | Frozen `include_diagnostics=true`; S5 keeps excluded rows |
| Per-date book CAR on **traded** dates | `build_date_summary` `cycle_return_on_capital_at_risk` |
| Side CAR splits | `long_cycle_*` / `short_cycle_*` on included rows |
| Drawdown helper | `_compute_max_drawdown` on a simple-return series |
| Fill model and settle | `FillAssumption`; `assembly.settle`; `pnl_total = abs(quantity) × pnl_per_share` |
| Signed trade quantity | `_signed_quantity`: long `+abs`, short `−abs`; S5 already scales P&L by `abs(quantity)` |
| Structure-failure raw text | `failure_reason` on `structure_ok=False` rows (`metadata_error:…` prefix plus builder `ValueError`/`KeyError` text) |
| Portfolio exclusion vocabulary | `no_tradeable_structure`, `max_names_cap`, `invalid_max_loss`, fair-share / no-short-credit codes |
| Receipt + overwrite refusal | Adapter writes parquet/JSON with SHA-256; refuses existing run dir |

### Gaps that actually block the D0 decision pack

| Gap | Why it blocks D3 |
|-----|------------------|
| No calendar-aligned series | `date_summary` omits `valid_no_trade` / `failed` dates; search Sharpe is not View B |
| No dual-window report | Periods exist in the JSON; nothing filters 2020-01-01..2026-07-10 |
| `robust_score` is the search rank metric | Must not be used for go/no-go; D3 must not promote it |
| `_assembly` discarded | D4 cannot reconstruct legs, fill prices, or expiry intrinsic without it |
| No candidate-stage labels | `trade_log` has the facts; D3 needs a thin derived view so D4 does not re-interpret raw columns |
| No funnel counts | Empty-signal dates have zero trade_log rows; jointly-eligible (pre-tail) counts are not persisted |
| Adapter prints no report | D0 requires `conditional_and_calendar_aligned_metrics` as a recorded output |

### Non-gaps (owned later or already decided)

* Real-data smoke, manual sample, full-history execution, written conclusion → **D4**
* Momentum IC / CVG increment / full-universe buckets → **out of scope** (selection-biased on this artifact)
* Drop-ticker / drop-week counterfactuals → **out of scope** (not in D0; ambiguous calendar/capital semantics)
* Sprint 007 study matrix → **out of scope**

---

## 3. Requirement → capability / action

| D3 requirement | Current capability | Action |
|----------------|--------------------|--------|
| Frozen dual-view decision report | Partial CAR on trade_log dates only | New evaluation functions joining `date_status` + `date_summary` + `trade_log` |
| Cross primary / mid diagnostic | Twin runs exist | One report covering both; label roles; mid-versus-cross as fill-assumption sensitivity |
| Candidate diagnostic view | `trade_log` is the grain | Derive columns; do not re-select |
| Leg log | Transient `_assembly` | Serialize before drop; scale portfolio qty by `abs(trade.quantity)`; require 2/4 matching legs on every included trade; abort on missing/incomplete/duplicate/unexpected rows |
| Funnel | Stages exist in the loop | Capture aggregates at those calls; null for unexecuted stages; do not reimplement ranking |
| Completeness | `has_unresolved_failures` | Headline/report flag; block “complete result” presentation (including turnover) |
| Broken traded-date economics | Not checked today | Abort report generation; do not drop or 0-fill the date |
| Deterministic outputs + digests | Adapter pattern | Extend `write_run_outputs` / receipt; no new CLI |

---

## 4. Proposed design

### 4.1 Execution architecture

```text
configs/sprint006_baseline_v1.json
        │
        ▼
existing run_baseline()  (unchanged economics)
        │
        ├─ SurfaceRunner.run_single_config(mid)
        │     S1→S5 as today
        │     + capture funnel counts at existing boundaries
        │     + serialize legs from _assembly before drop
        │
        └─ SurfaceRunner.run_single_config(cross)
              same
        │
        ▼
existing artifacts (trade_log, date_summary, date_status, run_summary)
+ candidate_view (derived from trade_log)
+ leg_log (from _assembly)
+ funnel_summary (from loop counts)
        │
        ▼
sprint006_evaluation.build_decision_report(mid_result, cross_result, contract periods)
        │  aborts if traded-date economics fail, included-trade legs are
        │  missing/incomplete/duplicate/unexpected, or identities fail to reconcile
        ▼
decision_report.json + decision_report.md
receipt digests updated; D3 removed from deferred (D4 remains)
```

`SurfaceRunner.run_single_config()` remains the only economic engine. Evaluation is a **pure post-pass**. `--dry-run` still runs no economics and writes nothing. CLI stdout still prints paths and row counts only — never Sharpe, CAR, or P&L.

D3 **implements the generator**. The first authorized real-data report is **D4**. D3 tests use synthetic fixtures only.

### 4.2 Frozen decision report

#### Windows (D0, closed intervals)

| Window | Start | End |
|--------|-------|-----|
| Full history | `2018-10-26` | `2026-07-10` |
| Primary | `2020-01-01` | `2026-07-10` |

Filter `date_status.trade_date` and all joined rows to the window. **`date_status` is the expected calendar.** Do not infer the calendar from `date_summary` or unique `trade_log` dates.

#### Fills

* **Primary economic view:** `sprint006_baseline_v1_cross`
* **Diagnostic view:** `sprint006_baseline_v1_mid`
* Report both. Do not rank or select by `robust_score`.

#### Completeness vs report-generation preconditions

Keep:

```text
result_complete = (n_failed_dates == 0)
has_unresolved_failures = (n_failed_dates > 0)
```

Any unresolved `failed` date **blocks Sprint 006 acceptance** and **blocks presenting that window as a complete result**. Still emit diagnostic counts and the incomplete banner. Do not 0-fill or drop `failed` dates to manufacture a complete series.

**Mandatory report-generation preconditions** (narrow; not a new status taxonomy). Before computing View A/B, win rate, profit factor, yearly, or attribution, `build_decision_report` must verify:

1. Every `date_status=traded` date has **exactly one** matching `date_summary` row with:
   * finite `cycle_pnl_total`
   * **positive** finite `cycle_capital_at_risk` (the date-summary capital column; do not accept 0 or non-finite)
   * finite `cycle_return_on_capital_at_risk`
2. Every `date_status=valid_no_trade` date has **no** `trade_log` rows with `included_in_portfolio==True`.
3. Every included trade satisfies the §4.4 leg-log completeness and identity checks (no optional subset).

If any check fails, **abort report generation** with a clear error naming the date (and ticker/direction for leg mismatches). Do **not**:

* drop the date from Sharpe/CAR
* zero-fill a broken traded date
* emit a quieter partial metric pack
* flip `result_complete` as a substitute for aborting

`result_complete` remains only “no failed dates.” Broken traded-date economics and failed included-trade leg completeness/reconciliation are **implementation defects**, not extra date-status values.

Do not add a generalized validator framework or additional completeness flags.

#### View A — Conditional deployed-capital (traded dates only)

Join `date_status` to `date_summary` on `trade_date`.

* `traded`: `cycle_return_on_capital_at_risk = Σ pnl_total / Σ capital_at_risk_dollars` (already on `date_summary`; do not recompute trade PnL). After a successful report, **every** traded date in the window has finite CAR — do not filter to “finite only” as a way to skip broken rows.
* `valid_no_trade`: **NaN**; **excluded** from mean, Sharpe, and drawdown.
* `failed`: window is incomplete.

Report:

* `n_traded_dates`, `n_valid_no_trade_dates`, `n_failed_dates`, `n_expected_dates`
* `mean_cycle_car` = mean of traded cycle returns (all of them, after the precondition)
* `annualized_sharpe` = mean/std(ddof=1)×√52 on that series; **NaN** if fewer than 2 traded dates or std=0
* `max_drawdown` on that series (existing helper; do **not** insert zeros for excluded no-trade dates)

Label every View A number **conditional on traded dates**.

#### View B — Calendar-aligned

Series length = every expected A1 date in the window, sorted.

* `traded`: that date’s book cycle CAR return
* `valid_no_trade`: **0**
* `failed`: do not present a complete View B

Report:

* `compounded_return` = Π(1 + r_t) − 1
* `annualized_return` = (1 + compounded_return)^(52 / n_expected_dates) − 1 when compounded > −1 and n_expected_dates ≥ 1; else NaN
* `annualized_sharpe` = mean/std(ddof=1)×√52 on the 0-filled series (≥2 points, std>0)
* `max_drawdown` on that same series

Pin annualization to **weekly frequency (52)** so it matches Sharpe √52. Report `n_expected_dates` and first/last date beside it. Do **not** also invent a calendar-year CAGR.

If compounded ≤ −1, annualized_return is NaN (cannot raise a non-positive wealth index).

#### Weekly outcomes (conditional traded book weeks)

* **Win rate:** fraction of **traded** book-return weeks with `cycle_return_on_capital_at_risk > 0`. After the precondition those returns are finite. Do not treat calendar 0-fill weeks as wins.
* **No-trade frequency:** `n_valid_no_trade_dates / n_expected_dates` (separate; not mixed into win rate).
* **Profit factor:** `sum(positive cycle_pnl_total) / abs(sum(negative cycle_pnl_total))` on traded dates. If denom=0 and numer>0 → `+Infinity`; if both 0 → NaN.

JSON has no `Infinity`. Serialize finite numbers as JSON numbers; `+Infinity` as the string `"Infinity"`; NaN as JSON `null`. Do not reuse adapter `_jsonable`’s silent inf→null for this field.

#### Year-by-year

Same View A / View B conventions, grouped by calendar year of `trade_date`. Each year reports return, Sharpe, drawdown, and `traded` / `valid_no_trade` / `failed` counts.

#### Long/short attribution

On included rows in the window (conditional traded dates):

* `long_n_traded_rows`, `short_n_traded_rows`
* Sum `pnl_total` and `capital_at_risk_dollars` by side
* Mean of `long_cycle_return` / `short_cycle_return` over traded dates where that side exists (NaN dates excluded from that side’s mean)

Do not 0-fill a missing side into the other side’s attribution.

#### Mid versus cross fill-assumption sensitivity

Relabel this block **Mid versus cross fill-assumption sensitivity**. It is **not** a pure transaction-cost attribution.

Cross-minus-mid date-level CAR or P&L can include changes in:

* Entry economics
* Max loss and sizing
* Portfolio inclusion
* Potentially selected structures

Keep the D0-required overlap and spread-cost diagnostics. Align on `trade_date`:

* `n_dates_both_traded`, `n_dates_cross_only`, `n_dates_mid_only`
* On **both-traded** dates: mean(cross CAR − mid CAR), mean(cross `cycle_pnl_total` − mid `cycle_pnl_total`), labeled as fill-assumption deltas
* Mean `spread_cost_ratio` and `leg_spread_to_credit_ratio` among included rows per fill (already on trade_log)
* Candidate overlap: count of `(trade_date, ticker, direction)` included in both vs one fill only

Disclose unmatched dates/candidates explicitly. Never silently inner-join away a fill’s unique dates.

Do **not** add exact leg-signature matching or a more complex attribution system in D3.

#### Concentration (primary window, primary fill = cross)

Among included rows: ticker share of `sum(|pnl_total|)`. Report top five tickers, their shares, and the top-five sum share.

This, plus yearly results, long/short attribution, and drawdown, is the D3 answer to stability and dominance. Do **not** add drop-highest-ticker or drop-best-week sensitivities.

#### Activity / data

* Average included names per **traded** date, overall and by side (`long_n_traded`, `short_n_traded` from `date_summary`)
* Turnover (complete only when `result_complete`):
  * `traded` dates contribute that date’s included-name count
  * `valid_no_trade` dates contribute **zero** selected names
  * `failed` dates, including `missing_features`, are **not** economic no-trades and **must not** be entered as zero (or any other turnover contribution)
  * If any date in the window is `failed`, do **not** present a complete turnover result; diagnostic traded/`valid_no_trade` counts may still be shown, labeled incomplete
* Joint feature coverage: from funnel (below) — `n_feature_covered_dates / n_expected_dates`; mean jointly-eligible names over dates where that count is **not null**
* Structure-failure counts: histogram of the §4.3 structure `reason_code` values (`metadata_error`, `missing_quotes_or_body`, `wing_or_liquidity_selection`, `other_structure`). Do **not** histogram full raw exception strings.
* Date-class counts (repeat from `date_status`)

#### Frozen limitations (verbatim intent)

Hold-to-expiry; no earnings filter; below-nearest 0.15-delta wings; Tier A not integer lots; long-only fallback dates possible; mid is a fill-assumption diagnostic, not a pure cost attribution; `robust_score` is not a decision metric.

#### Output files (one pair per run directory, covering both fills)

| File | Role |
|------|------|
| `decision_report.json` | Deterministic machine-readable pack |
| `decision_report.md` | Compact human rendering of the same numbers |

No charts, dashboards, or extra scores. If report generation aborts, do not write these files as a successful D3 deliverable and do not publish a `sprint006_d3` receipt claiming a complete report.

JSON skeleton (normative keys; omit invented extras):

```text
experiment_id, contract_id, repo_sha
result_complete, has_unresolved_failures
windows: { full_history, primary }
fills: { cross (primary), mid (diagnostic) }
per fill × window:
  date_class_counts
  view_a_conditional { mean_cycle_car, sharpe, drawdown, n_traded }
  view_b_calendar { compounded, annualized_return, sharpe, drawdown }
  weekly { win_rate, profit_factor, no_trade_frequency }
  yearly[]
  long_short
  activity
  structure_failure_counts
  funnel_totals
fill_assumption_sensitivity { overlap counts, labeled deltas, unmatched }
concentration_primary_cross_top5
limitations[]
```

Markdown is a short decision memo: incomplete banner (if any), headline table for both windows × both views on **cross**, then mid diagnostic, yearly, attribution, fill-assumption sensitivity, concentration, activity/funnel, limitations. No narrative go/no-go sentence in D3 (that is D4).

### 4.3 Candidate-level diagnostic view

**Grain:** one row per existing `trade_log` row. That grain is **post-signal** (Momentum tails + within-side CVG already applied). D3 does **not** emit a row for every PIT-universe name.

**Canonical record remains `trade_log`.** The candidate view is a thin derived table, not a second selection path.

Derived columns (keep all identifying trade_log keys; do not duplicate the full economics schema):

| Column | Definition |
|--------|------------|
| `run_id` | From config |
| `fill_label` | From config / trade_log |
| `trade_date`, `ticker`, `direction` | Existing |
| `decision_status` | `traded` if `included_in_portfolio==True` else `no_trade` |
| `stage` | `traded` if included; else `structure_failed` if `structure_ok!=True`; else `portfolio_excluded` |
| `reason_code` | Small normalized code (below); `null` when `stage=traded` |
| `reason_raw` | `failure_reason` if `structure_failed` else `exclusion_reason` |

Do **not** add `outcome_status` or any replacement “availability” column. Explanation of a no-trade is already `reason_code` + `reason_raw`. Economic presence is already nullable P&L fields plus `decision_status`.

**Date taxonomy stays separate.** A structure rejection or cap/sizing exclusion is a **candidate-level `no_trade`**, not a `failed` date. Dates with only such rows are `valid_no_trade` / `no_included_names` on `date_status`.

**`reason_code` for `stage=portfolio_excluded`** — existing S5 vocabulary, unchanged:

| `reason_code` | When |
|---------------|------|
| `max_names_cap` | `exclusion_reason` that value |
| `invalid_max_loss` | that value |
| `premium_exceeds_fair_share` | that value |
| `max_loss_exceeds_fair_share` | that value |
| `no_short_credit` | that value |
| `earnings_exclusion` | that value (unused on frozen earnings-off runs; still map if present) |
| `other_exclusion` | any other non-null exclusion (should not occur; still observable) |

**`reason_code` for `stage=structure_failed`** — small stable classification of existing S3 messages. Do **not** map every structure failure to `no_tradeable_structure` (that string remains the S5 `exclusion_reason` on those rows). Do **not** histogram the full raw exception, which embeds ticker, date, and numeric thresholds.

Match the **stable prefix** of `failure_reason` (`reason_raw`):

| `reason_code` | Current-code prefix / pattern |
|---------------|-------------------------------|
| `metadata_error` | starts with `metadata_error:` (S3 wrap of missing metadata or `surface_valid=False`) |
| `missing_quotes_or_body` | starts with `No quote surface rows`, `No eligible quotes`, `Missing body call/put`, or `Missing tradeable body call/put` |
| `wing_or_liquidity_selection` | starts with `No quotes with abs_delta` **or** `Iron fly spread_cost_ratio=` (threshold exceed after wing construction) |
| `other_structure` | any other `structure_failed` `failure_reason` |

This is prefix matching on messages already raised in `pipeline.py` S3 and `option_surface.py`. It is not a generalized error ontology. Retain `reason_raw` unchanged for D4.

**Combined mid/cross candidate file:** **do not** create a merged candidate parquet. Overlap lives in the report’s fill-assumption-sensitivity block (date and `(date, ticker, direction)` counts). Merging two fills into one candidate table invites false “same trade” identity. Per-run `candidate_view_<run_id>.parquet` is enough.

### 4.4 Minimal leg log

**When to capture:** in `run_single_config`, after S5, **before** dropping `_assembly`. Economic path unchanged.

**Who appears:** constructable post-signal candidates (`structure_ok==True` and `_assembly` present). Structure failures have no legs — they stay on the candidate view with `stage=structure_failed`. Non-included constructable rows may still be serialized (unit economics only).

**Included-trade completeness (mandatory; not optional).** Every `included_in_portfolio==True` trade must:

* Have `structure_ok==True`. An included row that is not constructable **aborts** report generation.
* Have matching leg-log rows keyed by `(run_id, fill_label, trade_date, ticker, direction)`:
  * included straddle (`instrument_type` `long_straddle` or `short_straddle`): **exactly 2** rows, `leg_index` `{0,1}`
  * included iron fly (`instrument_type` `iron_fly`): **exactly 4** rows, `leg_index` `{0,1,2,3}`
* Abort on **missing** (zero matching rows), **incomplete** (fewer than required, or missing `leg_index` values), **duplicate** (repeated `leg_index` or extra copies of the same key), or **unexpected** (wrong count for the instrument, extra unmatched included-trade keys, or any included instrument other than straddle/iron fly).

Do not skip an included trade because it “has no leg log.” There is no such exemption.

Then all §4.4 entry, payoff, and P&L identities must reconcile for **every** included trade. Failure **aborts** report generation. This is direct equality checking, not a generalized validation framework.

**Deterministic `leg_index`:** `enumerate(strategy.legs)` as already built:

* Long straddle: 0 call, 1 put (existing builder order)
* Iron fly: 0 long OTM put, 1 short ATM put, 2 short ATM call, 3 long OTM call

Do not invent OCC ids, quote timestamps, or fees.

S5 stores `trade.quantity` with **strategy direction** (`_signed_quantity`: short names are negative). Iron-fly `unit_quantity` (`OptionLeg.quantity`) **already** encodes long wings (+) and short bodies (−). Scaling by signed `trade.quantity` would reverse short-iron-fly legs. Use magnitude only:

```text
portfolio_quantity = abs(trade.quantity) * unit_quantity   # included; else null
pnl_total_leg      = abs(trade.quantity) * pnl_per_unit    # included; else null
```

| Column | Source | Notes |
|--------|--------|-------|
| `run_id`, `fill_label` | config | |
| `trade_date`, `ticker`, `direction` | row | |
| `expiry_date` | `assembly.expiry_date` / quote | |
| `option_type` | `leg.option.option_type` | `call` / `put` |
| `strike` | `leg.option.strike` | |
| `leg_index` | 0..n−1 | |
| `unit_quantity` | `leg.quantity` | signed strategy-unit quantity |
| `bid`, `ask`, `mid` | quote | |
| `fill_price` | `fill.buy_price` if unit_quantity>0 else `fill.sell_price` | same as `_build_strategy_entry_cost` |
| `included_in_portfolio` | S5 | |
| `portfolio_quantity` | `abs(trade.quantity) * unit_quantity` if included else **null** | no counterfactual size |
| `exit_spot` | row | needed for intrinsic |
| `expiry_payoff_per_unit` | `intrinsic(exit_spot) * unit_quantity` | signed unit payoff (exists for constructable rows) |
| `entry_cash_per_unit` | `+fill_price * |unit_quantity|` if long else `−fill_price * |unit_quantity|` | matches assembly `entry_cost` sum |
| `pnl_per_unit` | `expiry_payoff_per_unit - entry_cash_per_unit` | unit economics |
| `pnl_total_leg` | `abs(trade.quantity) * pnl_per_unit` if included else **null** | only actual trades |

An included short iron fly must retain unit and portfolio signs:

```text
long put wing      positive
short put body     negative
short call body    negative
long call wing     positive
```

**Reconciliation (included trades; per ticker/date/direction):**

```text
sum(entry_cash_per_unit)     = trade_log.entry_cost_per_share     = assembly.entry_cost
sum(expiry_payoff_per_unit)  = Position.exit_value                = strategy.calculate_payoff
sum(pnl_per_unit)            = trade_log.pnl_per_share            = exit_value - entry_cost
sum(pnl_total_leg)           = trade_log.pnl_total                = abs(quantity) * pnl_per_share
```

Tolerance: exact Decimal→float round-trip at 1e-8 relative or 1e-6 absolute, whichever is larger. Missing/incomplete/duplicate/unexpected included-trade legs, or identity mismatch, **aborts report generation** (implementation defect). It does not change `result_complete`.

Non-included constructable rows still get unit entry/payoff (needed to see whether a cap-excluded name was economically well-formed) but **null** `portfolio_quantity` and `pnl_total_leg`.

### 4.5 Compact funnel summary

**One row per expected date per fill.** Do not emit a row per rejected universe name.

| Count | Definition | Source (existing boundary) | Long/short split? |
|-------|------------|----------------------------|-------------------|
| `n_expected` | 1 | `date_status` row | no |
| `n_feature_covered` | 1 iff date ∈ feature date set else 0 | already computed in runner | no |
| `n_universe` | S1 tickers, or **null** if S1 not run | `len(universe)` after `_step1_universe` | no |
| `n_jointly_eligible` | universe ∩ feature row, finite mom+cvg, both counts ≥ `required_count`, or **null** if S2 not run | **same filter S2 already applies before ranking** — extract a shared helper used by S2 and this count | no (pre-direction) |
| `n_post_signal` | S2 output rows, or **null** if S2 not run | `len(signals)` | yes, when evaluated |
| `n_constructable` | `structure_ok==True`, or **null** if S3 not run | S3/S5 frame | yes, when evaluated |
| `n_included` | `included_in_portfolio==True`, or **null** if S5 not run | S5 | yes, when evaluated |
| `date_status`, `date_reason` | D2 taxonomy | `date_status` | no |

**Null vs zero.** Zero means the stage **ran** and no names survived. Null means the stage was **not evaluated**.

Missing-feature dates (`date_status=failed`, `reason=missing_features`; today’s fail-fast skip of S1–S5). These are **failed** dates, not `valid_no_trade`:

```text
n_expected = 1
n_feature_covered = 0
n_universe = n_jointly_eligible = n_post_signal = n_constructable = n_included = null
```

Evaluated `empty_signals` dates (S2 ran, produced no names):

```text
n_post_signal = 0
n_constructable = 0
n_included = 0
```

`n_universe` / `n_jointly_eligible` remain whatever S1/S2 actually counted on that date (typically ≥ 0, not null).

**Coverage denominator:** `n_expected_dates` in the reporting window from `date_status`.

**Joint coverage rate:** `n_feature_covered_dates / n_expected_dates`. Funnel **averages exclude null/unexecuted values**; they must not treat `failed` / `missing_features` nulls as zeros. Mean jointly-eligible names is a labeled mean over dates where `n_jointly_eligible` is not null. Funnel nulls on failed dates are coverage diagnostics; they are not turnover zeros.

**Do not** re-read features in the evaluation module to recount eligibility. The runner captures the integers (and nulls); D3 only sums/averages them.

**Shared helper (narrow):** pull S2’s pre-rank filter into something like `eligible_feature_cross_section(...)` returning the filtered slice. `step2_score_signals` ranks that slice unchanged. Funnel uses `len(slice)` when S2 ran. This is not a second eligibility implementation.

**Selection-bias notice (must appear on funnel and candidate view):** “post-signal candidate” means already in the Momentum tail and kept by the within-side CVG filter. These counts cannot support full-universe Momentum IC or CVG increment tests.

### 4.6 Integration with adapter / receipt

Extend `write_run_outputs` per `run_id`:

* existing: `trade_log_`, `date_summary_`, `date_status_`, `run_summary_`
* new: `candidate_view_`, `leg_log_`, `funnel_summary_`

Once per run directory (after both fills, and only if `build_decision_report` succeeds):

* `decision_report.json`, `decision_report.md`

Receipt: digest every new file; set `deliverable` to `sprint006_d3`; keep D2 date-status fields; add `result_complete` / `has_unresolved_failures` at report level; `deferred` = D4 only.

Do not redesign overwrite policy, output-dir location rules, twin orchestration, or `--dry-run`.

`date_summary` remains trade-log-derived. **`date_status` remains the coverage table.** Evaluation must join, not replace, them.

`SurfaceRunResult` gains `leg_log` and `funnel_summary` DataFrames (empty frames with the pinned schema when there are no rows). Candidate view may be built in the adapter from `trade_log` (no extra runner state required).

---

## 5. Alternatives considered

| Choice | Verdict |
|--------|---------|
| **Post-pass evaluation module + thin runner serialization** (recommended) | Narrowest way to get D0 metrics and D4 reconstruction without a second engine |
| Reuse `summarize_trade_log` as the decision pack | Rejected: wrong calendar, promotes `robust_score`, no primary window |
| New report CLI | Rejected: D0/D1 already require one documented command |
| Offline D3 from parquet only (no runner change) | Rejected for legs/funnel: `_assembly` is already gone; jointly-eligible counts are not in trade_log |
| Merged mid/cross candidate parquet | Rejected: identity hazards; overlap belongs in the report |
| Histogram raw `failure_reason` strings | Rejected: ticker/date/numeric fragments explode into one-off buckets |
| Map every structure failure to `no_tradeable_structure` | Rejected: hides metadata vs quotes vs wing-selection |
| `outcome_status` available/unavailable | Rejected: a reason string is not an economic outcome |
| Drop-highest-\|PnL\| ticker / drop-best-week | Rejected: not in D0; counterfactual calendar and capital semantics are ambiguous |
| Exact mid/cross leg-signature attribution | Rejected: D3 keeps overlap counts; a fuller attribution system is out of scope |

---

## 6. Proposed file-level changes

| File | Change |
|------|--------|
| `src/backtest/pipeline.py` | Extract pre-rank eligible slice helper; S2 calls it (behavior-preserving) |
| `src/backtest/surface_runner.py` | Capture funnel counts (null when unexecuted); serialize legs before `_assembly` drop; attach frames to `SurfaceRunResult` |
| `src/backtest/sprint006_evaluation.py` (**new**) | Dual-view metrics, candidate derivation, report JSON/MD, traded-date preconditions, structure `reason_code` mapping |
| `src/backtest/sprint006_baseline.py` | Persist new artifacts; build report after both runs; abort without a misleading report file; receipt/deferred updates |
| `tests/unit/test_sprint006_evaluation.py` (**new**) | View A/B NaN vs 0; win rate; profit factor Infinity; window filter; overlap disclosure; abort on broken traded dates; funnel null vs zero; turnover 0 only on `valid_no_trade`; incomplete turnover when any date is `failed` |
| `tests/unit/test_sprint006_leg_log.py` (**new** or fold into evaluation tests) | Leg/trade reconciliation; short-iron-fly sign preservation; abort on zero/incomplete/duplicate/unexpected included-trade legs; no legs for structure_failed; null portfolio qty when excluded |
| Existing runner / orchestration / adapter tests | Funnel counts; new output names; receipt keys; empty-signal and missing-feature funnel |
| `docs/surface_engine_data_contract.md` | Minimal pointer that D3 report uses `date_status` as calendar (optional, with implementation) |
| **Do not edit** | `configs/sprint006_baseline_v1.json`; iron-condor; `surface_search.py`; D4 execution; frozen thresholds |

`scripts/run_sprint006_baseline.py` stays a thin wrapper unless a one-line help-text tweak is needed.

Approximate effort: inside the 12–18h review trigger if scope stays as above.

---

## 7. Focused synthetic tests

1. View A: `valid_no_trade` excluded (NaN, not 0) from Sharpe/drawdown/mean.
2. View B: same date contributes 0; compounded matches hand Π(1+r)−1.
3. Failed date in window → `result_complete=false`; View B and turnover are not presented as complete; report still generates **if** traded-date and included-trade leg preconditions hold.
4. `traded` date missing `date_summary`, or with non-finite PnL/CAR, or with non-positive capital → **abort**; the date is not dropped and no partial metric pack is written.
5. `valid_no_trade` date with an included portfolio row → **abort**.
6. `ceil`/`joint` funnel: jointly-eligible count matches helper; post-signal count ≤ jointly-eligible when both are non-null.
7. Failed / `missing_features` date: `n_feature_covered=0`; `n_universe`…`n_included` are null; funnel averages exclude those nulls; turnover does **not** treat the date as a zero-name economic no-trade; a complete turnover result is not presented.
8. Empty S2 → `date_status=valid_no_trade/empty_signals`, funnel `n_post_signal=0` (and downstream zeros), no candidate rows; turnover contributes 0 selected names.
9. Structure fail → candidate `stage=structure_failed`, `decision_status=no_trade`, `reason_code` one of the four structure classes, `reason_raw` retained, no leg rows; date may still be `valid_no_trade`.
10. Cap exclusion → `stage=portfolio_excluded`, existing exclusion `reason_code`, legs present with null `pnl_total_leg`.
11. Included short iron fly: four legs; `unit_quantity` and `portfolio_quantity` signs are `+ − − +`; `portfolio_quantity = abs(trade.quantity) * unit_quantity`; unit sums match `entry_cost_per_share`, payoff, `pnl_per_share`, `pnl_total`.
12. Included-trade leg completeness: **abort** report generation (and do not write a partial metric pack) when an included trade has zero matching legs, fewer than 2 (straddle) or 4 (iron fly) legs, duplicate `leg_index`/extra copies, an unexpected count or instrument, or `structure_ok!=True`. Identity mismatch also aborts. `result_complete` is not used as the signal.
13. Win rate ignores calendar 0-fill weeks; profit factor `"Infinity"` when no losses.
14. Mid/cross fill-assumption sensitivity: unmatched dates listed, not dropped; block is labeled as fill-assumption, not pure cost.
15. Top-5 \|PnL\| shares sum to ≤ 1.
16. Adapter writes pinned schemas; receipt lists new files; deferred is D4 only.
17. Regression: existing S2, S5, orchestration, runner, option_surface, adapter, S8 search-metric tests still pass (`robust_score` path untouched).

**Forbidden in D3 validation:** accepted real-data run; reading aggregate P&L from snapshot/derived artifacts; proving real-data `failed` count is zero.

---

## 8. Ordered implementation steps

1. Accept this plan.
2. **Commit 1:** `sprint006_evaluation.py` dual-view calculators, traded-date preconditions, and tests on synthetic date_status/date_summary/trade_log. No runner change.
3. **Commit 2:** eligible-slice helper; runner funnel (null vs zero) + leg serialization (`abs(trade.quantity)` scaling); short-iron-fly sign and reconciliation tests. Economics of S2/S5 unchanged (diff S2 only by helper extraction).
4. **Commit 3:** adapter persistence, JSON/MD report, receipt, candidate_view derivation (no `outcome_status`), structure `reason_code` mapping, deferred-list, optional data-contract sentence.
5. Focused pytest subset; stop. Do not start D4 or real-data P&L.

---

## 9. Acceptance criteria and stop conditions

### Design acceptance (this document)

- [ ] Plain-language summary approved
- [ ] Frozen D0 formulas/windows/fills unchanged
- [ ] Candidate vs date taxonomies separated; post-signal grain explicit
- [ ] Leg log uses `abs(trade.quantity) * unit_quantity`; short-iron-fly signs specified; every included trade requires 2 or 4 matching legs
- [ ] Funnel null vs zero specified; averages skip unexecuted stages
- [ ] Traded-date economic preconditions abort rather than drop dates
- [ ] Turnover: `valid_no_trade` is zero names; `failed` / `missing_features` is not economic no-trade and blocks a complete turnover result
- [ ] No `outcome_status`; no drop-ticker / drop-week sensitivities
- [ ] D4 / Sprint 007 / real-data execution not authorized

### D3 implementation acceptance (after coding)

- [ ] One documented command still runs both fills through `run_single_config` and writes the report pack
- [ ] `date_status` is the calendar; failed dates block complete presentation
- [ ] Broken traded-date economics abort report generation
- [ ] Included trades abort on missing, incomplete, duplicate, or unexpected legs; identities reconcile
- [ ] View A NaN vs View B 0 implemented and tested
- [ ] Mid/cross block labeled as fill-assumption sensitivity; overlap disclosed
- [ ] Candidate view derived only from `trade_log`; no `outcome_status`
- [ ] Leg/trade identities reconcile on included synthetic trades; short iron fly keeps `+ − − +`
- [ ] Funnel does not reimplement ranking/CVG tails; `failed` / `missing_features` counts are null, not turnover zeros
- [ ] Complete turnover is not presented when any date is `failed`
- [ ] New files digested in the receipt
- [ ] Frozen JSON untouched; no real-data P&L inspection
- [ ] Focused tests green

**Stop** when that evidence exists. Pause if work expands into dashboards, full-universe signal research, a second runner, drop-ticker/week counterfactuals, or the 12–18h review trigger without a remaining D3 blocker.

---

## 10. Explicit out of scope

* Full-universe Momentum buckets, rank IC, or Momentum×CVG outcome grids
* Candidates rejected before Momentum-tail / CVG filters
* Counterfactual portfolio P&L or sizing, including drop-highest-ticker and drop-best-week
* Parameter search or improvement recommendations
* New fees, commissions, or fill assumptions
* Exact mid/cross leg-signature matching or a cost-attribution system
* Regime models, ML, charts, dashboards
* Numeric profitability thresholds or go/no-go from `robust_score`
* `outcome_status` or another availability abstraction
* Generalized error ontologies or validator/preflight frameworks
* Real-data economic execution and written conclusion (**D4**)
* Sprint 007 robustness / candidate studies
* Iron-condor, `SurfaceSearch`, frozen JSON edits, unrelated refactors
* Catch-and-continue for unexpected exceptions (D2 fail-fast stands)

---

## 11. Guardrails (proportional only)

| Guardrail | Failure prevented | Why the cost is justified |
|-----------|-------------------|---------------------------|
| Calendar = `date_status` | Silent date loss in Sharpe/CAR | One join; D2 already built the table |
| `result_complete` requires zero failed dates | Presenting an incomplete backtest as trusted | One flag already on the run |
| Abort if a `traded` date lacks finite positive-capital economics | Silently dropping broken traded dates from performance | Narrow precondition; no new status enum |
| Abort if `valid_no_trade` has included rows | Calendar 0-fill hiding real trades | One inclusion check |
| Abort if an included trade lacks exactly 2 (straddle) or 4 (iron fly) matching legs, or identities fail | Publishing a report D4 cannot reconstruct | Direct count + equality checks; no validator framework |
| Do not 0-fill `failed` / `missing_features` into turnover | Treating a coverage failure as an economic no-trade | Same completeness flag already on the run |
| View A NaN vs View B 0 | Misleading Sharpe from mixing no-trade with zeros or dropping zeros from calendar | Frozen D0 rule; cheap to test |
| Mid/cross overlap counts | Fill-assumption delta computed on a silent inner join | Small counter |
| Leg/trade sum checks abort the report | Wrong fill/payoff serialization that D4 would hand-check blindly | Local equality on included rows |
| `portfolio_quantity = abs(trade.quantity) * unit_quantity` | Reversed short-iron-fly wings/bodies | Matches S5’s existing `abs(quantity)` P&L scale |
| Funnel null vs zero | Averaging unexecuted stages as if they produced no names | Capture at existing skip boundaries |
| Receipt digests for new files | Unreproducible diagnostic tables | Existing adapter pattern |
| No stdout economic metrics | D3 implementation accidentally becoming P&L inspection | Keep D1 print contract |

Omit: extra preflight, schema registries, retry/publication systems, generalized diagnostic ontologies, drop-ticker/week machinery.

---

## 12. Commit recommendation

**Three commits**, matching the natural review boundaries:

1. Evaluation calculations, traded-date preconditions, and tests (no engine change).
2. Leg-log extraction (`abs(trade.quantity)`), exact 2/4 included-trade leg counts, funnel counts with null vs zero, S2 helper extract, reconciliation and abort tests.
3. Report files, adapter/receipt, candidate view without `outcome_status`, documentation.

Do not squash into one commit: commit 1 is independently reviewable without touching settle/assembly.

---

## 13. P&L firewall

This design does not authorize running accepted snapshot/derived artifacts through the economic loop or reading aggregate P&L, Sharpe, rankings, or side performance from real data. Implementation tests are synthetic. The first real-data report is D4.

---

**End of proposed D3 design.** Implementation requires acceptance of this plan. No D3 code, frozen-config, or economic execution is authorized by drafting or correcting this document alone.
