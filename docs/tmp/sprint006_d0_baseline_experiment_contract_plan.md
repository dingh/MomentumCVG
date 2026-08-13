# Sprint 006 D0 — Baseline experiment contract plan

**Status:** `PLAN — AWAITING REVIEW` (design only; not accepted; no implementation authorized by this file alone)  
**Mode:** Audit / design (no runtime edits, no economic run)  
**Repository HEAD at original design:** `f0a36f1b5ceff545cc2933c5a3c73d7a9ba891ba`  
**Plan commit:** `9ff498fdf52b335f97a194c3b132ad16edb64def`  
**Correction HEAD:** `9ff498fdf52b335f97a194c3b132ad16edb64def` (working-tree edit; uncommitted)  
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md` (matches Sprint 005 deliverable plans)

---

## Review summary

D0 freezes one Sprint 006 economic baseline **before any new P&L is inspected**: fixed `(42,8)` Momentum+CVG, no search, accepted Sprint 004/005 artifacts, weekly hold-to-expiry long ATM straddle + short iron fly, Tier A `equal_max_loss` (long financed by short premium; `10000` fallback permits long-only books), mid diagnostic + **cross primary**, history `2018-10-26`→`2026-07-10`, primary reporting from `2020-01-01`.

**Expected dates** come from sorted unique A1 `entry_date` values in that interval (**all rows, including `surface_valid=False`**), then reconciled to `features_42_8.parquet`. Feature-file absence cannot hide a calendar date. Every A1 expected date must be `traded` / `valid_no_trade` / `failed`.

**Return views (D3):** (1) **conditional deployed-capital** — existing CAR on traded dates only (`valid_no_trade` → NaN, excluded from Sharpe/DD); (2) **calendar-aligned** — full A1 calendar with `valid_no_trade=0`, any `failed` blocks a complete result. Do **not** use `robust_score` for go/no-go.

**Spread gate intent:** `max_leg_spread_pct=0.50` on **every** traded leg. Code today filters straddle bodies and iron-fly **wings only** (not iron-fly ATM shorts) — **partial; D2 corrects**. Quote gate ≠ cross-fill TC.

**Accepting this design approves every §5 `Proposed` row** (full list in §13). Remaining code gaps: joint Mom+CVG count (D2); A1 calendar + empty-date accounting (D2); all-leg spread (D2); trusted runner/CLI (D1); dual return views + metric pack (D3).

**D0 implementation later:** frozen contract JSON only (~2–4 h). **Ready for approval?** Yes as a corrected design freeze after §13 approvals — still not accepted, and not ready to run economics.

---

## 1. Goal and non-goals

| Goal | Non-goal |
|------|----------|
| Freeze every P&L-sensitive choice for one Sprint 006 baseline | Implement runner repairs (D1) |
| Pin accepted input identities and reproducibility evidence | Implement joint eligibility / date accounting / all-leg spread (D2) |
| Define expected-date taxonomy, no-trade metrics, and manual sample before results | Build decision report code (D3) |
| Bound D0 vs D1–D4 so implementation stays small | Execute mid/cross backtest or inspect new P&L (D4) |
| Prefer existing `BacktestRunConfig` / Surface path | New framework, engine, features, Sprint 007 capabilities |

---

## 2. Context read receipt

| Path | Why relevant | Key fact taken |
|------|--------------|----------------|
| `docs/agenda/current_sprint.md` | D0 freeze boundaries | Freeze all P&L knobs before new P&L; mid+cross; classify every expected date |
| `docs/backtest_evaluation_protocol.md` | Required metric families | Sharpe, CAR return, CAGR, drawdown, win rate, profit factor, concentration, harsh vs mid |
| `docs/surface_engine_data_contract.md` | A1 grain / S7–S8 | A1 grain `(ticker, entry_date)` incl. invalid rows; settle at `exit_spot`; S2 mom-count only today |
| `docs/surface_engine_portfolio_metrics_design.md` | Tier A / CAR | `equal_max_loss` + long financed by short premium; long-budget fallback edge rule; Sharpe on cycle series |
| `src/backtest/surface_runner.py` | Date loop | Trade dates from **features**; empty signals → `continue` (date disappears) |
| `src/backtest/surface_metrics.py` | Conditional CAR today | Zero CAR → NaN cycle return; Sharpe/DD on finite cycle series; `robust_score` = Sharpe×availability (search heuristic) |
| `src/backtest/run_config.py` | Required fields | `sizing_mode` required; `wing_selection_rule`, `max_loss_budget_per_trade`, `cost_model` required schema fields |
| `src/backtest/pipeline.py` | S1–S5 economics | AND universe; mom-only count; Tier A fallback to `tier_a_long_budget` when no/non-positive short credit |
| `src/backtest/option_surface.py` | Fill + structures | `mid`/`cross`; straddle filters **body** by `max_leg_spread_pct`; iron fly filters **OTM only** (body unfiltered); `_choose_below_nearest` |
| `docs/sprint_memos/005_closeout.md` | Identities | Snapshot `e2c1f8fd…`; ready `2018-10-26`→`2026-07-10` |
| `docs/sprint_memos/004_c7_pit_universe.md` | Universe envelope | Canonical `dvol_top_pct=0.20`, `spread_bottom_pct=1.0` |
| `docs/decisions/003_position_cap_per_side.md` | Caps | Independent `max_names_per_side` |
| `configs/feature_backfill_v1.json` | Window | Baseline `(42,8)`; `cvg_count_*` published |
| `scripts/run_surface_search.py` | Broken defaults | Missing `sizing_mode`; illegal `contract_multiplier` kwarg; defaults ≠ baseline |
| `tests/unit/test_option_surface_straddle.py` | Mid/cross, settle, body spread gate | Mid/cross fills; settle PnL; `max_leg_spread_pct` drops illiquid **body** legs |
| `tests/unit/test_option_surface_ironfly.py` | Wings, settle, wing-only spread gate | `_choose_below_nearest`; settle paths; `max_leg_spread_pct` applied to **wings**, not ATM body shorts |
| `tests/contract/test_step5_select_and_size_contract.py` | Tier A | `equal_max_loss` credit financing; fallback to long budget w/o shorts or zero credit; CAR/`pnl_total` |
| `tests/contract/test_run_metrics_contract.py` | Aggregation | Zero-denominator dates → NaN cycle return; Sharpe √52 on finite series; all-excluded date NaN |
| `tests/contract/test_orchestration_contract.py` | Empty-date behavior | `test_empty_signals_date_skipped_without_s5_call` — empty signals skip S5; empty summaries |
| `tests/contract/test_run_envelope_contract.py` | Config validation | `sizing_mode` required; Tier A/B field rules |
| `tests/contract/conftest.py` | Constructible defaults | Shows required fields incl. `wing_selection_rule`, `max_loss_budget_per_trade=500`, `cost_model` |
| `tests/contract/test_step1_universe_contract.py` / `test_step2_signals_contract.py` | S1/S2 | AND universe; mom/CVG pools (mom count path) |

**Research note:** Repo feature semantics (`feature_backfill_v1` + Sprint 005 audit) outrank external papers. TC model is `FillAssumption` mid/cross.

---

## 3. Accepted inputs and identities

| Identity | Value |
|----------|-------|
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Manifest | `…/manifests/input_snapshot_e2c1f8fd44d72176.json` |
| Derived root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |
| Features dir | `…/features/` |
| Baseline feature file | `…/features/features_42_8.parquet` |
| D3 receipt | `…/features_backfill_v1.lineage.json` (`status=complete`; SHA-256 `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5`) |
| D3 producer `repo_sha` | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Feature config | `configs/feature_backfill_v1.json` (SHA-256 `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd`) |
| D4 audit JSON | `…/features_quality_audit_v1.json` |
| A1 meta (**expected-date authority**) | `…/cache/surface/option_surface_meta_weekly_2018_2026.parquet` |
| A2 quotes | `…/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |
| Liquidity panel | `…/input/liquidity/ticker_liquidity_panel.parquet` |
| Earnings | **None** |
| Mutable cache | **Forbidden** as accepted input |

---

## 4. Proposed frozen experiment contract

### 4.1 Exact `BacktestRunConfig` twin (mid + cross)

Identical except `fill` / `run_id` suffix:

| Field | Frozen value | Notes |
|-------|--------------|-------|
| `run_id` | `sprint006_baseline_v1_mid` / `…_cross` | Twin runs |
| `momentum_col` / `cvg_col` / `count_col` | `mom_42_8_mean` / `cvg_42_8` / `mom_42_8_count` | Joint CVG count uses published `cvg_count_42_8` (D2) |
| `min_count_pct` | `0.80` | With joint rule → ≥28 of 35 |
| `long_top_pct` / `short_bottom_pct` | `0.10` / `0.10` | |
| `cvg_filter_pct` | `0.50` | Highest 50% CVG within side |
| `dvol_top_pct` / `spread_bottom_pct` | `0.20` / `1.0` | C7 canonical; AND |
| `short_structure` | `ironfly` | |
| `wing_selection_rule` | `"closest_delta"` | **Required config string**; surface IF actually uses `_choose_below_nearest` (`abs_delta ≤ 0.15`) |
| `wing_delta_target` | `0.15` | |
| `max_names_per_side` | `25` | Independent per side |
| `max_loss_budget_per_trade` | `500.0` | **Legacy required**; **does not** control Tier A `equal_max_loss` |
| `earnings_exclusion_days` | `0` | + `earnings_path=None` |
| `cost_model` | `"mid"` | Legacy/unused by surface economics; **`fill` authoritative** |
| `start_date` / `end_date` | `2018-10-26` / `2026-07-10` | Inclusive via runner `<= end` |
| `fill` | `FillAssumption.mid()` / `.cross()` | Cross = primary decision fill |
| `max_leg_spread_pct` | `0.50` | **Intent: all traded legs**; code partial — D2 |
| `max_spread_cost_ratio` | `None` | |
| `condor_short_delta_target` / `condor_long_delta_target` | `None` / `None` | Unused (not ironcondor) |
| `include_diagnostics` | `True` | |
| `sizing_mode` | `"conceptual"` | |
| `tier_a_mode` | `"equal_max_loss"` | |
| `tier_a_short_budget` | `10000.0` | Total short max-loss budget |
| `tier_a_long_budget` | `10000.0` | **Fallback only** (see below) |
| `tier_b_short_max_loss_budget` | `None` | Unused (not Tier B); omit/unset |
| `contract_multiplier` | `100.0` | Tier A ratios scale-invariant |
| `deployable_capital` | `None` | Unused |

**Tier A fallback (explicit):** normally longs are financed by collected short premium. If there are **no usable shorts** or collected short premium is **non-positive**, use fixed `tier_a_long_budget=10000`. This **permits a long-only book** on that date and requires approval.

**Search:** none. **Retuning after P&L:** forbidden.

### 4.2 Other frozen economic rules

| Topic | Exact rule | Support |
|-------|------------|---------|
| Primary reporting | Same run; metrics also filtered to dates in `[2020-01-01, 2026-07-10]` | D3 |
| Expected calendar | §8 — A1 `entry_date`, not feature-derived | D2 |
| No-trade metrics | §8.1 — conditional + calendar-aligned | D3 |
| ATM / expiry | A1 `body_strike` + weekly expiry/`exit_spot`; hold to expiry | Yes |
| Ranking | Momentum `rank(pct=True, method='average')`; long high / short low | Yes |
| Cap tie-break | Secondary `ticker` ascending | **Needs D1/D2 pin** |
| Missing structure | `structure_ok=False` + `failure_reason`; not included | Yes |
| TC | Cross = buy ask / sell bid; mid α=0.5; entry-only | Yes |
| Quote gate vs TC | `max_leg_spread_pct` is pre-trade quote quality; cross fills are separate execution friction | Intent vs code mismatch on IF body |

---

## 5. Decision register

| ID | Decision | Proposed exact value/rule | Source | Support | Status | If unresolved |
|----|----------|---------------------------|--------|---------|--------|---------------|
| D-01 | Feature window | `(42,8)` only | Agenda / 005 | Yes | **Fixed (agenda)** | Wrong question |
| D-02 | Search | None | Agenda | Override CLI | **Fixed** | Contaminates baseline |
| D-03 | Full history | `2018-10-26`→`2026-07-10` | D4 ready | Yes | **Fixed** | Coverage drift |
| D-04 | Primary period | Metrics filter `2020-01-01`→`2026-07-10` | Spec / eval protocol | D3 | **Proposed** | Ambiguous go/no-go window |
| D-05 | Mom tails | 10% / 10% | Working proposal | Yes | **Proposed** | Book composition |
| D-06 | CVG keep | Top 50% within side | Working proposal | Yes | **Proposed** | Book composition |
| D-07 | Count eligibility | Joint Mom **and** CVG ≥28 | Agenda “both” | Mom-only today | **Proposed; D2** | Quality bias |
| D-08 | Liquidity | PIT top 20% dvol | Spec / C7 | Yes | **Fixed** | Wrong universe |
| D-09 | Spread pct | `1.0` | C7; avoid double 20% AND | Yes; CLI≠ | **Proposed** | Shrinks universe if 0.20 |
| D-10 | Cap | 25/side independent | Decision 003 | Yes; CLI≠ | **Proposed** | Wrong book size |
| D-11 | Structures | Long straddle + short iron fly | Agenda; KB-001 | Yes | **Fixed** | Out of scope |
| D-12 | Wing economics | `_choose_below_nearest` @ 0.15; config `wing_selection_rule="closest_delta"` | Code + tests | Yes | **Proposed** | Wing/P&L change |
| D-13 | Hold model | Hold to expiry | Pins + A1 | Yes | **Fixed** | Live gap remains |
| D-14 | Sizing | Tier A `equal_max_loss`; budgets 10000/10000 fallback; long-only allowed on fallback | Portfolio metrics + step5 tests | Pipeline yes; CLI no | **Proposed** | Scale/fallback ambiguity |
| D-15 | Fills | Mid diagnostic; cross primary | Agenda | Yes | **Fixed** | Wrong decision fill |
| D-16 | Earnings | Off | Agenda | Yes | **Proposed** | Lookahead if invented |
| D-17 | Retuning | Forbidden after P&L | Agenda | Process | **Fixed** | Invalid experiment |
| D-18 | All-leg spread gate | `0.50` on straddle bodies **and** all four IF legs | Quote-quality intent | **Partial** (IF body unfiltered) | **Proposed; D2** | Inconsistent liquidity filter |
| D-19 | Expected dates | A1 unique `entry_date` in interval, any `surface_valid`; reconcile features | DoD; avoid circular calendar | Runner feature-based today | **Proposed; D2** | Silent attrition |
| D-20 | Cap tie-break | `ticker` asc secondary | Reproducibility | Not pinned | **Proposed; D1/D2** | Non-reproducible ties |
| D-21 | No-trade metrics | Conditional CAR + calendar-aligned 0-fill (§8.1) | Eval protocol completeness | Conditional≈today; calendar=D3 | **Proposed** | Misleading Sharpe |
| D-22 | Manual sample | §9 deterministic fallbacks | Agenda | Process | **Proposed** | Cherry-picking |
| D-23 | Legacy fields | `max_loss_budget_per_trade=500`, `cost_model="mid"`, multiplier 100; Tier B/condor unset | Constructible config | Schema | **Proposed** | Implicit defaults |
| D-24 | Entry point | Trusted Surface run on snapshot/derived paths | Decision 001 | CLI broken | **D1 delivers** | Cannot reproduce |

**Acceptance rule:** accepting this D0 design **approves every `Proposed` row above** (and the Fixed agenda pins). It does not authorize D1–D4 implementation by itself.

---

## 6. P&L-exposure firewall

1. No new Sprint 006 aggregate P&L, Sharpe, rankings, side returns, or strategy comparisons before contract acceptance.  
2. Read-only code/schema/coverage/identity checks allowed.  
3. Do not run a real-data economic backtest in D0 design or D0 implementation.  
4. Post-exposure correctness changes → new versioned experiment ID; no silent overwrite.  
5. No retuning because results look unattractive.

---

## 7. Reproducibility and output identity requirements

| Record | Requirement |
|--------|-------------|
| Experiment ID | `sprint006_baseline_v1` (+ `_mid` / `_cross`) |
| Code | Clean git HEAD of the run |
| Config | Frozen contract JSON SHA-256 + effective `BacktestRunConfig` dump |
| Inputs | Snapshot/build; feature + A1/A2/liquidity digests; D3 receipt |
| Outputs | Trade log, date summary, run summary, **date-status table**, both metric views |
| Command | One documented command; explicit snapshot/derived paths only |

---

## 8. Expected-date and failure-accounting contract

### 8.0 Independent calendar

**Canonical expected decision calendar:** sorted unique A1 `entry_date` values with `entry_date ∈ [2018-10-26, 2026-07-10]`, **including dates that appear only on `surface_valid=False` rows** (so an all-failure date cannot vanish).

**Reconciliation (D0/D2; not implemented now):**

1. Build A1 expected set as above.  
2. Build feature date set from `features_42_8.parquet` in the same closed interval.  
3. Every A1 expected date must be classified `traded` / `valid_no_trade` / `failed`.  
4. An A1 date **missing entirely** from the feature artifact is **not** silently absent — default class **`failed`** (missing feature coverage) unless an intentional exception is **explicit and evidenced**.  
5. Feature dates absent from A1 must be reported in reconciliation evidence (they are not members of the expected calendar).

| Class | Meaning |
|-------|---------|
| `traded` | ≥1 `included_in_portfolio=True` |
| `valid_no_trade` | Pipeline completed; zero included names for an allowed economic reason |
| `failed` | Incomplete processing, missing features vs A1, schema/identity failure, unresolved exception |

Unresolved `failed` blocks Sprint 006 acceptance. Classification + runner date loop fixes are **D2** (today: feature-derived dates + empty-signal skip).

### 8.1 No-trade return / Sharpe treatment (D3 reporting contract)

Cross = primary fill; mid = diagnostic. Report **both** views; **do not** use `robust_score` as go/no-go.

**View A — Conditional deployed-capital (preserves accepted CAR behavior)**

* `traded`: `cycle_return_on_capital_at_risk = Σ pnl_total / Σ capital_at_risk_dollars`.  
* `valid_no_trade`: zero denominator → **NaN**; **excluded** from conditional Sharpe and drawdown.  
* Label explicitly as **conditional on traded dates**.

**View B — Calendar-aligned sensitivity**

* Series length = full independent A1 expected calendar (within reporting window).  
* `traded`: book cycle CAR return.  
* `valid_no_trade`: contribute **`0`**.  
* Any `failed` → incomplete; **do not** present as a complete result.  
* Report calendar-aligned compounded return, annualized return/CAGR, Sharpe (√52 on that series), and drawdown.

Yearly splits use the **same** conventions as the parent view.

---

## 9. Manual verification sampling contract

Freeze **before** P&L. Cap ≤6 hand-checked trades. No performance-based replacement.

| Sample | Rule |
|--------|------|
| S1 | Median A1 expected date (sorted). Date-level lineage/status always. If `traded`, sample lowest-ticker included long and short **that exist** on that date |
| S2 | Earliest `traded` date with both sides. If none: earliest long-traded and/or earliest short-traded as available |
| S3 | Earliest `valid_no_trade`; if none → **`N/A`** |
| S4 | Earliest date with ≥1 `structure_ok=False`; if none → **`N/A`** |
| Shortfall | If fewer than six qualifying trade rows exist across S1/S2 picks, audit those available and document the shortfall — do not substitute a P&L-selected date |

Per included trade: universe → ranks/CVG → legs/strikes/expiry → fills → max loss → settle → date CAR contribution.

---

## 10. Required decision report and metrics

Small decision pack for **cross (primary)** and **mid (diagnostic)**. No charts, dashboards, new scores, or numeric profitability thresholds.

| Block | Required contents |
|-------|-------------------|
| Headline (both windows: full history + primary) | Mean cycle CAR (conditional); conditional traded-date annualized Sharpe + drawdown; calendar-aligned compounded return, CAGR/annualized return, Sharpe, drawdown |
| Weekly outcomes | Win rate; profit factor; no-trade frequency (separate) |
| Yearly | Per year: return, Sharpe, drawdown, `traded` / `valid_no_trade` / `failed` counts — for each view’s conventions |
| Attribution | Long vs short cycle returns and trade counts |
| Costs | Mid vs cross delta on overlapping dates; mean `spread_cost_ratio` |
| Concentration | Top-5 ticker share of \|PnL\| (primary period) |
| Activity / data | Avg names/side; turnover; joint feature coverage; structure-failure reason histogram; date-class counts |
| Limitations | Hold-to-expiry; no earnings; below-nearest wings; Tier A not integer lots; long-only fallback dates |

**Frozen definitions**

* **Weekly win rate:** fraction of **finite traded** book-return weeks with return > 0; report no-trade frequency separately (do not treat NaN/0-fill weeks as wins).  
* **Profit factor:** (sum of positive weekly book P&L) / (absolute sum of negative weekly book P&L). If denominator = 0: `+inf` when numerator > 0; `NaN` when numerator = 0. State which weekly series (conditional traded P&L weeks) is used.  
* **Sharpe:** mean/std(ddof=1)×√52 on the relevant weekly return series (≥2 finite points; else NaN).  
* **Do not** rank go/no-go by `robust_score`.

---

## 11. Minimal D0 implementation plan

**Later (~2–4 h), after acceptance:** write `configs/sprint006_baseline_v1.json` matching §4 + mark this plan accepted. No runtime/test edits; no backtest.

**Not D0:** `pipeline` / `surface_runner` / CLI fixes; eligibility; all-leg spread; metrics code; economic execution.

---

## 12. Verification and acceptance criteria

### Design acceptance (this corrected document)

- [x] P&L-sensitive choices in §5; Proposed set = approval boundary (§13)  
- [x] Independent A1 expected calendar; feature absence cannot hide dates (§8)  
- [x] No-trade treatment explicit for decision metrics (§8.1, §10)  
- [x] Metric set covers eval-protocol families without becoming a platform (§10)  
- [x] All-leg spread intent + IF body mismatch accurate (§4, §13)  
- [x] Exact config constructible without unspecified defaults (§4.1)  
- [x] Manual samples deterministic with N/A / shortfall rules (§9)  
- [x] Relevant implementation tests read (§2)  
- [x] D1–D4 not pulled into D0; P&L firewall explicit  
- [ ] User acceptance of the full Proposed contract  

### Later D0 implementation acceptance

- [ ] Approved JSON matches this contract + digests  
- [ ] No runtime/test changes; no backtest; no P&L inspection  

---

## 13. Risks, inconsistencies, and open decisions

### Documentation vs code mismatches

| Mismatch | Contract | Code today | Resolution |
|----------|----------|------------|------------|
| Expected calendar | A1 `entry_date` (any validity) | Feature dates; empty signals skipped | **D2** |
| Joint count | Mom+CVG ≥28 | Mom `count_col` only | **D2** |
| All-leg spread 0.50 | Every traded leg | Straddle bodies yes; IF **wings only** | **D2** |
| Wing label | Config `closest_delta` | `_choose_below_nearest` | Freeze both; no silent rename |
| Spread universe | `spread_bottom_pct=1.0` | CLI default 0.20 | Freeze 1.0 |
| Caps / earnings | 25 / 0 | CLI 3 / 5 | Freeze 25 / 0 |
| Trusted CLI | Snapshot/derived + sizing | Broken `sizing_mode` / kwargs | **D1** |
| Calendar metrics | Dual views | Conditional CAR only; `robust_score` exists | **D3** (no go/no-go via `robust_score`) |

### Complete user-approval boundary

Accepting this design approves **all** of the following Proposed choices (Fixed agenda pins are already locked and are not reopened):

1. Primary reporting period `2020-01-01`→`2026-07-10`  
2. Momentum tails 10% / 10%  
3. Highest 50% CVG retention within side  
4. Joint Mom+CVG counts ≥28 (D2 implements)  
5. `spread_bottom_pct=1.0`  
6. 25-name independent per-side cap  
7. Below-nearest 0.15-delta wing behavior (+ config `wing_selection_rule="closest_delta"`)  
8. Tier A `equal_max_loss` with budgets `10000` / `10000` fallback  
9. Long-side fallback (may produce long-only books)  
10. Earnings off  
11. All-leg `max_leg_spread_pct=0.50` (D2 completes IF body)  
12. Independent A1 expected-date calendar + feature reconciliation  
13. Deterministic cap tie-break (`ticker` asc)  
14. Dual no-trade metric treatment (§8.1)  
15. Manual verification sampling (§9)  
16. Legacy/unused field pins (`max_loss_budget_per_trade=500`, `cost_model="mid"`, multiplier 100, Tier B/condor unset)  

---

## 14. Explicit handoff boundaries for D1–D4

| Deliverable | Owns | Must not reopen |
|-------------|------|-----------------|
| **D0** | Contract freeze (this doc + later JSON) | Runtime behavior, P&L |
| **D1** | Trusted reproducible runner/command; config dump; tie-break pin if needed | Eligibility; all-leg spread; report pack |
| **D2** | Joint Mom+CVG eligibility; A1 expected-date status; no silent date loss; all-leg `max_leg_spread_pct`; focused tests | Parameter retune; new features |
| **D3** | Decision report: dual return views + §10 metrics | Changing frozen knobs; `robust_score` go/no-go |
| **D4** | Smoke, manual sample, full mid+cross run, reproducibility, closeout | Silent contract replacement; Sprint 007 matrix |

---

**End of corrected D0 design.** Stop here pending review.
