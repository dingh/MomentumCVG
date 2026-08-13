# Sprint 006 D0 — Baseline experiment contract plan

**Status:** `PLAN — AWAITING REVIEW` (design only; no implementation authorized by this file alone)  
**Mode:** Audit / design (no runtime edits, no economic run)  
**Repository HEAD at design:** `f0a36f1b5ceff545cc2933c5a3c73d7a9ba891ba`  
**Working tree at design:** clean  
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md` (matches Sprint 005 deliverable plans)

---

## Review summary

D0 freezes one Sprint 006 economic baseline **before any new P&L is inspected**: fixed `(42,8)` Momentum+CVG, no search, accepted Sprint 004/005 artifacts, weekly hold-to-expiry long ATM straddle + short iron fly, Tier A `equal_max_loss`, diagnostic mid + primary cross fills, full ready history with primary reporting `2020-01-01`→`2026-07-10`.

**Pinned inputs:** snapshot `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886`; derived features under `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` with D3 receipt `status=complete`; `(42,8)` file `features_42_8.parquet`; Stage A paths from the snapshot manifest (not mutable cache).

**Compact baseline:** top/bottom 10% momentum → keep highest 50% CVG per side; joint count ≥ 28/35 (80%); PIT dvol top 20% with `spread_bottom_pct=1.0` (AND semantics; matches C7); max 25 names/side; iron-fly wings via existing `_choose_below_nearest` at `|Δ|≤0.15`; earnings off; no retuning after P&L.

**D0 implementation footprint (later):** record an approved frozen contract JSON + short acceptance note only (~2–4 h). Runner repair, joint eligibility, reporting, and the economic run belong to D1–D4.

**Top risks / mismatches:** (1) code filters only Momentum count today — joint Mom+CVG ≥80% is contract-now / D2-later; (2) “closest 0.15 delta” wording ≠ `_choose_below_nearest`; (3) `run_surface_search.py` cannot construct current `BacktestRunConfig` (missing `sizing_mode`) and passes illegal `contract_multiplier` into `SurfaceDataPaths`; (4) empty-signal dates are silently skipped in `SurfaceRunner` — expected-date accounting is D2.

**Needs user approval:** spread `1.0`; wing rule as below-nearest; joint count rule; `max_leg_spread_pct=0.50`; Tier A budget scale `10000`; pre-registered manual sample rule; confirm no earnings.

**Ready for implementation?** Yes, as a **documentation/config freeze**, after the highlighted approvals. Not ready to run economics.

---

## 1. Goal and non-goals

| Goal | Non-goal |
|------|----------|
| Freeze every P&L-sensitive choice for one Sprint 006 baseline | Implement runner repairs (D1) |
| Pin accepted input identities and reproducibility evidence | Implement joint eligibility / date accounting (D2) |
| Define expected-date / failure taxonomy and manual sample rule before results | Build decision report UI/metrics code (D3) |
| Bound D0 vs D1–D4 so implementation stays small | Execute mid/cross full-history backtest or inspect new P&L (D4) |
| Prefer existing `BacktestRunConfig` / Surface path | New config framework, new engine, new features, Sprint 007 capabilities |

---

## 2. Context read receipt

| Path | Why relevant | Key fact taken |
|------|--------------|----------------|
| `docs/agenda/current_sprint.md` | Sprint 006 scope / D0 boundaries | Acceptance authorizes D0 only; freeze all P&L knobs before new P&L; `(42,8)`, mid+cross, no retuning |
| `docs/sprint_memos/005_closeout.md` | Accepted Sprint 005 lineage | Snapshot `e2c1f8fd44d72176`; `(42,8)` ready `2018-10-26`→`2026-07-10`; D5 consumability only |
| `docs/sprint_memos/005_feature_correctness_audit.md` | Feature semantics / research adaptations | `(42,8)` = 35 slots; Sprint 006 joint thresholds intentionally unset in 005; accepted adaptations beat raw papers |
| `docs/sprint_memos/sprint005_d3_production_backfill_evidence.md` | Feature publication identity | 281 files; receipt SHA-256 `c585bce…`; producer `repo_sha` `131d0ac…` |
| `docs/sprint_memos/sprint005_d4_quality_audit_evidence.md` | Coverage / ready interval | `(42,8)` joint coverage ~68.38% in ready interval; PIT 0 violations |
| `docs/sprint_memos/sprint005_d5_surface_runner_smoke_evidence.md` | Consumer path / smoke ≠ strategy pins | Explicit Stage A + `features_42_8` paths; smoke thresholds not Sprint 006 params |
| `docs/sprint_memos/004_closeout.md` | Snapshot contract | Immutable root `…/20260724T045049097520Z_40b16886`; `production_accepted=true` |
| `docs/sprint_memos/004_c6_option_surface.md` | Weekly ~7DTE surface semantics | Strict calendar weekly expiry; hold-to-expiry settlement via A1 `exit_spot` |
| `docs/sprint_memos/004_c7_pit_universe.md` | Liquidity filter envelope | Canonical `dvol_top_pct=0.20`, `spread_bottom_pct=1.0`; AND ranks |
| `docs/v1_spec_pins.md` | v1 strategy pins | Long straddle / short IF|IC; 7DTE; hold-to-expiry; Tier A/B; primary go/no-go from 2020 |
| `docs/v1_universe_protocol.md` | PIT universe | Prior snapshot `month_date < t`; top 20% dvol; signals inside universe |
| `docs/backtest_evaluation_protocol.md` | Evaluation windows / fills | Tier A full sample; Tier B `2020→latest`; cross primary; mid diagnostic |
| `docs/development_workflow.md` | Roadmap | 006 = real-data economic backtest after separate authorization (agenda is current authority for 006 shape) |
| `docs/known_bugs.md` | Out-of-scope defects | KB-001 iron-condor body credit — IC comparison excluded while open |
| `docs/decisions/001_canonical_backtest_path.md` | Engine choice | `SurfaceRunner` path is canonical |
| `docs/decisions/003_position_cap_per_side.md` | Caps | Independent `max_names_per_side` (e.g. 25+25) |
| `docs/decisions/004_tier_b_credit_financed_long.md` | Sizing boundary | Tier A conceptual unchanged; Tier B out of Sprint 006 |
| `docs/surface_engine_data_contract.md` | S1–S8 contracts | S1 AND filters; S2 count uses `count_col` only; S7 hold-to-expiry |
| `docs/surface_engine_portfolio_metrics_design.md` | Tier A / CAR / metrics | `equal_max_loss`; primary `cycle_return_on_capital_at_risk` |
| `docs/surface_straddle_observation_transform_design.md` | D2 observation contract | Body `is_body` only; mid=(bid+ask)/2 for economics |
| `configs/feature_backfill_v1.json` | Window / columns | Baseline `(42,8)`; publish `mom_*_count` and `cvg_count_*` |
| `scripts/run_surface_search.py` | Current CLI defaults / breakage | Defaults include search, `spread_bottom_pct=0.20`, `max_names=3`, earnings=5; missing `sizing_mode`; illegal `SurfaceDataPaths(..., contract_multiplier=)` |
| `src/backtest/run_config.py` | Config schema | Requires `sizing_mode`; Tier A fields; fill via `FillAssumption` |
| `src/backtest/surface_runner.py` | Orchestration gap | Empty signals → `continue` (date disappears); trade dates = feature dates in range |
| `src/backtest/pipeline.py` | Economic rules in code | S1 AND; S2 mom-count only; S5 per-side cap; Tier A sizing; settle included only |
| `src/backtest/option_surface.py` | Structure / fill / wings | `FillAssumption.cross`; iron-fly `_choose_below_nearest`; ATM body from surface meta |
| `src/backtest/surface_metrics.py` | Existing metrics | Weekly Sharpe √52 on CAR cycle returns; availability heuristic |
| `src/backtest/surface_run_config.py` | Path resolution | Explicit snapshot/derived paths required for trusted runs |
| `tests/unit/test_option_surface_ironfly.py` | Wing semantics evidence | Pins `_choose_below_nearest` (≤ target), not true closest |
| `tests/contract/test_step1_universe_contract.py` / `test_step2_signals_contract.py` / `test_step5_select_and_size_contract.py` | Contract coverage | Universe AND; signal pools; per-side sizing path |
| `docs/baseline_status.md` | Env / accepted roots | Venv + accepted snapshot/derived locations |

**Research note:** No primary-source Momentum/CVG papers live in-repo. Authoritative semantics are `feature_backfill_v1` + `005_feature_correctness_audit.md` (deliberate weekly adaptations). Transaction-cost model for this baseline is repository `FillAssumption` mid/cross, not an external TC paper.

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
| A1 meta | `…/cache/surface/option_surface_meta_weekly_2018_2026.parquet` |
| A2 quotes | `…/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |
| Liquidity panel | `…/input/liquidity/ticker_liquidity_panel.parquet` |
| Earnings artifact | **None** (path unset) |
| Mutable cache | **Forbidden** as accepted input for Sprint 006 baseline |

D0 implementation must record file digests from the D3 receipt for `features_42_8.parquet` and resolve A1/A2/liquidity digests from the snapshot manifest / on-disk hash at freeze time (read-only).

---

## 4. Proposed frozen experiment contract

### 4.1 Compact parameter table

| Area | Frozen value | Support today |
|------|--------------|---------------|
| Feature window | `(42,8)` only — columns `mom_42_8_mean`, `cvg_42_8`, `mom_42_8_count`, `cvg_count_42_8` | Yes |
| Search | None (single config × two fills) | CLI defaults search — must not use defaults |
| Available / run history | `start_date=2018-10-26`, `end_date=2026-07-10` (inclusive via `<= end`) | Yes |
| Primary reporting period | Filter metrics to trade dates in `[2020-01-01, 2026-07-10]` on the same run | Reporting = D3; run once |
| Momentum selection | `long_top_pct=0.10`, `short_bottom_pct=0.10` | Yes |
| CVG refinement | `cvg_filter_pct=0.50` (highest 50% CVG within each side) | Yes |
| Feature eligibility | **Joint:** `mom_42_8_count ≥ 28` **and** `cvg_count_42_8 ≥ 28` (`min_count_pct=0.80`, window=35) | **Partial** — code uses mom `count_col` only |
| Liquidity | `dvol_top_pct=0.20` | Yes |
| Spread filter | `spread_bottom_pct=1.0` (AND with dvol; disables extra spread cull) | Yes; matches C7 |
| Portfolio cap | `max_names_per_side=25` independent | Yes |
| Long / short structures | Long ATM straddle; short `ironfly` | Yes |
| Iron-fly wings | `wing_delta_target=0.15` via `_choose_below_nearest` (max `abs_delta ≤ 0.15`) | Yes (not true closest) |
| Holding | Weekly surface entry (~6–8 DTE observed); hold to expiry | Yes (A1 meta) |
| Sizing | `sizing_mode=conceptual`, `tier_a_mode=equal_max_loss`, `tier_a_short_budget=10000`, `tier_a_long_budget=10000` (fallback only) | Yes in pipeline; CLI broken |
| Fills | Two paired runs: `FillAssumption.mid()` diagnostic; `FillAssumption.cross()` **primary** | Yes |
| Earnings | `earnings_path=None`, `earnings_exclusion_days=0` | Yes |
| Per-leg spread gate | `max_leg_spread_pct=0.50` | Yes |
| Structure spread-cost cap | `max_spread_cost_ratio=None` | Yes |
| Diagnostics rows | `include_diagnostics=True` | Yes |
| `cost_model` | `"mid"` (legacy required field; economics use `fill`) | Yes |
| `contract_multiplier` | `100.0` | Yes (Tier A ratios scale-invariant) |
| Retuning | Forbidden after P&L exposure | Process |

### 4.2 Material rules not in the working proposal (frozen here)

| Topic | Exact rule |
|-------|------------|
| Entry dates | Feature dates in `[start,end]` from `features_42_8.parquet` (weekly schedule already aligned to surface) |
| Expiry / DTE | A1 precomputed weekly expiry + `exit_spot`; no alternate DTE search |
| ATM selection | A1 `body_strike` / A2 `is_body` call+put |
| Ranking | Momentum `rank(pct=True, method='average')` within PIT∩eligible cross-section; long high / short low |
| Cap tie-break | After side rank sort, secondary key `ticker` ascending (determinism; **needs D1/D2 sort pin** if current sort is unstable) |
| Missing structure | Keep candidate row with `structure_ok=False` / `failure_reason`; not portfolio-included |
| Long/short independence | Separate pools and caps (Decision 003) |
| Capital / return | Primary: `cycle_return_on_capital_at_risk = Σ pnl_total / Σ capital_at_risk_dollars` per date; Sharpe on that series × √52 |
| Settlement | Intrinsic payoff at A1 `exit_spot` on expiry; no exit spread |
| TC semantics | Cross: buys ask / sells bid on every leg; mid: α=0.5 both sides; entry-only friction |
| No-trade vs failed | See §8 |
| Manual sample | See §9 |
| Decision metrics | See §10 |

---

## 5. Decision register

| ID | Decision | Proposed exact value/rule | Source / rationale | Code/config support | Status | If unresolved |
|----|----------|---------------------------|--------------------|---------------------|--------|---------------|
| D-01 | Feature window | `(42,8)` only | Sprint 005 baseline + 006 agenda | Supported | **Fixed (agenda)** | Wrong research question |
| D-02 | Search | None | Agenda | CLI defaults must be overridden | **Fixed** | Contaminates baseline |
| D-03 | Full history | `2018-10-26`→`2026-07-10` | D4 ready interval | Supported | **Fixed** | Incomparable coverage |
| D-04 | Primary period | Metrics filter `2020-01-01`→`2026-07-10` | `v1_spec_pins` / eval protocol | Reporting later | **Proposed** | Ambiguous go/no-go window |
| D-05 | Mom tails | 10% / 10% | Working proposal + CLI research default | Supported | **Proposed** | Changes book composition |
| D-06 | CVG keep | Top 50% within side | Working proposal + CLI default | Supported | **Proposed** | Changes book composition |
| D-07 | Count eligibility | Joint Mom **and** CVG ≥ `0.80×35=28` | Agenda “both”; window math verified | **Mismatch** — mom-only today | **Proposed; D2 implements** | Silent quality bias |
| D-08 | Liquidity | PIT top 20% dvol | Spec + C7 | Supported | **Fixed** | Unsupported universe |
| D-09 | Spread pct | `1.0` | Avoid double 20% AND; **C7 canonical** | Supported; CLI default `0.20` differs | **Proposed (approve)** | Severely shrinks universe if left at 0.20 |
| D-10 | Cap | 25/side | Decision 003 | Supported; CLI default 3 | **Proposed** | Wrong book size |
| D-11 | Structures | Long straddle + short iron fly | Agenda / pins; KB-001 blocks IC | Supported | **Fixed** | Out of scope |
| D-12 | Wing rule | `_choose_below_nearest` @ 0.15 | Actual surface builder + tests | Supported; wording “closest” is false | **Proposed (approve wording→code)** | Reinterpretation changes wings/P&L |
| D-13 | Hold model | Hold to expiry on weekly surface | Pins + A1 | Supported | **Fixed** | Live gap remains documented |
| D-14 | Sizing | Tier A `equal_max_loss`, short budget 10000, long budget 10000 fallback | Portfolio metrics design; Sprint 006 excludes Tier B | Pipeline yes; search CLI no | **Proposed** | Absolute scale largely cancels in CAR ratios; still must freeze |
| D-15 | Fills | Mid diagnostic + cross primary | Agenda / eval protocol | Supported | **Fixed** | Wrong decision fill |
| D-16 | Earnings | Off (`days=0`, no path) | Agenda; no trusted PIT earnings | Supported | **Proposed** | Lookahead if speculative earnings used |
| D-17 | Retuning | Forbidden after P&L | Agenda | Process | **Fixed** | Invalidates experiment |
| D-18 | `max_leg_spread_pct` | `0.50` | Existing search default; material | Supported | **Proposed (approve)** | Changes fillable set |
| D-19 | Expected dates | All feature dates in run interval | Agenda DoD | Runner currently omits empties | **Proposed; D2 enforces** | Silent attrition |
| D-20 | Cap tie-break | `ticker` asc secondary | Reproducibility | Not pinned today | **Proposed; D1/D2** | Non-reproducible edge ties |
| D-21 | Entry point | Trusted single-config Surface run on snapshot/derived paths | Decision 001; D5 pattern | `run_surface_search.py` broken for v1 sizing | **D1 delivers** | Cannot reproduce |

---

## 6. P&L-exposure firewall

1. **No new Sprint 006 aggregate P&L, Sharpe, rankings, side returns, or strategy comparisons may be inspected before this contract is accepted.**
2. Read-only code, schema, coverage, and identity checks are allowed.
3. **Do not run a real-data economic backtest as part of D0 design or D0 implementation.**
4. After P&L exposure, any correctness-driven contract change creates a **new versioned experiment ID**; it must not silently overwrite the original.
5. Parameters may not be changed merely because results are unattractive.

---

## 7. Reproducibility and output identity requirements

D4 (not D0) must emit, for each fill view, enough to reproduce:

| Record | Requirement |
|--------|-------------|
| Experiment ID | e.g. `sprint006_baseline_v1` (+ `_mid` / `_cross`) |
| Code identity | Clean git HEAD SHA used for the run |
| Config identity | Frozen contract JSON SHA-256 + effective `BacktestRunConfig` dump |
| Input identities | Snapshot/build IDs; feature file digest; A1/A2/liquidity digests; D3 receipt digest |
| Outputs | Trade log, date summary, run summary, date-status table (§8) |
| Command | One documented command using explicit snapshot/derived paths (no mutable cache) |

D0 implementation only **writes the frozen contract artifact** that those later runs must cite.

---

## 8. Expected-date and failure-accounting contract

**Expected decision calendar:** sorted unique `date` values in `features_42_8.parquet` with `date ∈ [2018-10-26, 2026-07-10]`.

Every expected date must appear in a date-status table with exactly one class:

| Class | Meaning |
|-------|---------|
| `traded` | ≥1 row with `included_in_portfolio=True` |
| `valid_no_trade` | Pipeline completed; zero included names for an allowed economic reason (empty universe after filters, no names passing signal/eligibility, all candidates `no_tradeable_structure` / sizing rejects, etc.) |
| `failed` | Incomplete/aborted processing, schema/identity failure, or unresolved exception |

**Rules:** no expected date may be absent; unresolved `failed` blocks Sprint 006 acceptance; classification implementation is **D2** (current runner `continue` on empty signals violates this).

---

## 9. Manual verification sampling contract

Freeze selection **before** viewing P&L. Deterministic, non-performance-based:

| Sample | Selection rule (apply after D2 date-status exists; design-time pin) |
|--------|---------------------------------------------------------------------|
| S1 | Median expected date by sorted calendar order (same spirit as D5) |
| S2 | First `traded` date with ≥1 long and ≥1 short included (earliest) |
| S3 | First date classified `valid_no_trade` |
| S4 | First date with ≥1 `structure_ok=False` candidate (if any; else next `traded`) |
| Per included trade on S1/S2 | Independently re-check: universe membership → signal ranks/CVG → legs/strikes/expiry → fill prices → max loss → settle PnL → contribution to date CAR |

Exact ticker picks on a multi-name date: lowest `ticker` among included longs and among included shorts (deterministic). Cap at **≤6 trades** hand-checked.

---

## 10. Required decision report and metrics

D3 must support a decision (not a platform). Minimum evidence on **cross (primary)** and **mid (diagnostic)**:

| Block | Contents |
|-------|----------|
| Headline | Full-history and primary-period mean cycle CAR return, annualized Sharpe (√52), max drawdown |
| Costs | Mid vs cross delta on same dates; mean `spread_cost_ratio` |
| Attribution | Long vs short cycle returns and trade counts |
| Activity | Dates by `traded` / `valid_no_trade` / `failed`; avg names/side; turnover |
| Concentration | Top-5 ticker share of |PnL| in primary period |
| Coverage | Joint feature eligibility pass rate; structure failure reason histogram |
| Limitations | Hold-to-expiry vs live; no earnings; wing below-nearest; Tier A not integer lots |

Numeric pass/fail thresholds remain TBD per evaluation protocol — Sprint 006 success is evidence quality.

---

## 11. Minimal D0 implementation plan

**Scope:** record the approved contract so D1+ can load/cite it. **~2–4 focused hours.**

| Artifact | Action |
|----------|--------|
| `configs/sprint006_baseline_v1.json` | New frozen contract: identities, `BacktestRunConfig` fields for mid/cross twin runs, expected-date rule, sample rule, firewall statement |
| This plan file | Status → `APPROVED FOR IMPLEMENTATION` / `ACCEPTED` after review; no runtime code |
| Optional short note in progress log later | Only when user authorizes agenda update (not part of this design task) |

**Explicitly not in D0 implementation:** edits to `pipeline.py` / `surface_runner.py` / `run_surface_search.py`; eligibility logic; metrics/report code; economic execution.

**Scope alarm:** if “recording the contract” seems to require >4–6 h or runtime behavior changes, stop — that work is D1/D2.

---

## 12. Verification and acceptance criteria

### Design acceptance (this document)

- [x] Every P&L-sensitive choice listed in §5
- [x] Each proposal has source/rationale + support/mismatch
- [x] Input identities pinned (§3)
- [x] D1–D4 not pulled into D0 (§11, §14)
- [x] P&L firewall explicit (§6)
- [x] Expected dates cannot disappear silently (§8)
- [x] Reproduction / evidence expectations defined (§7, §10)
- [x] Manual sample fixed before results (§9)
- [ ] User approvals for highlighted rows in §5 / §13

### Later D0 implementation acceptance

- [ ] Approved JSON exists and matches this contract
- [ ] Digests recorded for baseline feature file + Stage A inputs
- [ ] No runtime/test changes; no backtest run; no P&L inspection

---

## 13. Risks, inconsistencies, and open decisions

### Documentation vs code mismatches (do not resolve silently)

| Mismatch | Docs / proposal | Code today | Narrowest resolution |
|----------|-----------------|------------|----------------------|
| Joint count eligibility | Agenda: both Mom+CVG ≥80% | S2 filters `mom_*_count` only; data contract I3 mom-only | **Freeze joint ≥28 in D0; implement in D2** |
| Wing wording | “Closest to \|Δ\|=0.15” | `_choose_below_nearest` (≤0.15, max abs_delta) | **Freeze code behavior**; treat “closest” as informal |
| Spread filter | Proposal + C7: `1.0` | `run_surface_search` default `0.20` (AND) | **Freeze `1.0`** |
| Caps / earnings | 25/side; no earnings | CLI defaults `3` / `5` days | **Freeze 25 and 0** |
| Trusted entrypoint | Decision 001 cites `run_surface_search.py` | Missing `sizing_mode`; illegal `contract_multiplier` kwarg | **D1**: minimal trusted runner/CLI; do not use broken defaults |
| Empty dates | DoD: classify every date | `SurfaceRunner` skips empty signals | **D2** |
| Roadmap vs agenda | Older roadmap maps go/no-go to 007 | Sprint 006 agenda = first trusted economic result | **Agenda wins for 006** |
| Eval protocol “partial spread” fill | Three fill tiers historically | Sprint 006 = mid + cross only | **Freeze mid+cross** |

### Decisions requiring user approval (recommended defaults above)

1. `spread_bottom_pct=1.0` (not 0.20).  
2. Wing rule = `_choose_below_nearest` @ 0.15 (not true closest).  
3. Joint Mom+CVG count ≥ 28 (D2 implements).  
4. `max_leg_spread_pct=0.50`.  
5. Tier A budgets `10000` / `10000` fallback.  
6. Manual sample rule in §9.  
7. Earnings fully off.

---

## 14. Explicit handoff boundaries for D1–D4

| Deliverable | Owns | Must not reopen |
|-------------|------|-----------------|
| **D0** | Experiment contract freeze (this doc + later JSON) | Runtime behavior, P&L |
| **D1** | Trusted reproducible runner/command on accepted paths; config/manifest wiring; deterministic tie-break if needed | Eligibility semantics; report pack; full economic interpretation |
| **D2** | Joint Mom+CVG eligibility; expected-date status table; no silent date loss; focused tests | Parameter retune; new features |
| **D3** | Decision-quality report/metrics from frozen outputs | Changing frozen knobs |
| **D4** | Smoke, manual sample verification, full mid+cross execution, reproducibility evidence, closeout recommendation | Silent contract replacement; Sprint 007 robustness matrix |

---

**End of D0 design.** Stop here pending review.
