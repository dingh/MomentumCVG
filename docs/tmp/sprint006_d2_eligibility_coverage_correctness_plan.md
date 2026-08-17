# Sprint 006 D2 — Eligibility and coverage correctness plan

**Status:** `PROPOSED — AWAITING ACCEPTANCE`
**Mode:** Build (this document is **design only**; D2 implementation is not authorized until this plan is accepted)
**Repo HEAD at design:** `9edab171dda22497856cd2274767f314e47ae4ab` (clean working tree on `main`)
**Confirmed ancestors:** D1 implementation `241b0d3`, review fix `c6b1735`, D1 acceptance `9edab17`
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (unchanged; SHA-256 of committed LF bytes `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715`)
**D0 plan:** [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](sprint006_d0_baseline_experiment_contract_plan.md) (`ACCEPTED — D0 COMPLETE`)
**D1 plan / record:** [`docs/tmp/sprint006_d1_trusted_baseline_runner_plan.md`](sprint006_d1_trusted_baseline_runner_plan.md) (`ACCEPTED — D1 COMPLETE`)
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md`

---

## Review summary

**Recommended design (plain language):** Keep `SurfaceRunner.run_single_config()` as the only economic engine. Teach that path three narrow correctness behaviors already frozen in D0: (1) apply the same `min_count_pct=0.80` jointly to Momentum and CVG counts with `required_count = ceil(0.80 × 35) = 28` before ranking; (2) drive the date loop from the independent A1 `entry_date` calendar, reconcile to feature dates, and classify every expected date exactly once as `traded` / `valid_no_trade` / `failed` so empty-signal and missing-coverage dates cannot disappear; (3) apply `max_leg_spread_pct=0.50` to iron-fly ATM body shorts as well as wings. Persist the resulting date-status table through the existing D1 result/output/receipt path for D3.

**Reused unchanged:** `SurfaceRunner.run_single_config` as engine; S1 universe; S3/S4/S5 selection/sizing/settle; Tier A economics; fill mid/cross; straddle body spread filter (already correct); D1 thin adapter architecture, path preflight, overwrite refusal, twin mid/cross execution; frozen JSON thresholds and periods.

**Minimum runtime changes:**
1. Optional explicit `cvg_count_col` on `BacktestRunConfig` + joint count filter in S2 using `math.ceil(min_count_pct × window_size)`.
2. A1 expected-date loop + feature reconciliation + `date_status` on `SurfaceRunResult` (replace silent empty-signal `continue`).
3. Iron-fly body `max_leg_spread_pct` filter (mirror straddle).
4. Small D1 adapter deltas only: map `feature_window.cvg_count_col`, write `date_status_*.parquet`, record completeness / failed counts in receipt, update deferred list.

**Expected footprint:** ~4 production modules touched (`pipeline.py`, `surface_runner.py`, `option_surface.py`, `sprint006_baseline.py`) + focused tests (S2, orchestration/runner date-status, iron-fly body spread, adapter persistence). No new runner, CLI, status framework, or DQ subsystem. Frozen JSON untouched.

**Date-status flow:** Runner builds `date_status` while iterating A1 dates → attaches to `SurfaceRunResult` → adapter `write_run_outputs` persists `date_status_<run_id>.parquet` beside existing artifacts → receipt lists the artifact and `unresolved_failed_count` / completeness flag. D3 reads that table; D2 does not compute return metrics.

**Focused tests:** Synthetic S2 joint/ceil/missing-column cases; runner classification invariant (`expected = traded + valid_no_trade + failed`); empty-signal observability; iron-fly body spread; adapter write/receipt. Prefer extending existing modules over new frameworks.

**Material decisions needing human approval (engineering; D0 economics stay frozen):**
1. Represent CVG count as optional `BacktestRunConfig.cvg_count_col` mapped from contract `feature_window.cvg_count_col` (recommended) — not naming-convention magic and not a generalized schema system.
2. Build and attach `date_status` inside `SurfaceRunner` / `SurfaceRunResult` (recommended) — adapter only persists it.
3. Classify known coverage/eligibility/no-structure outcomes into the taxonomy; let unexpected exceptions abort the run (recommended) rather than soft-swallowing them as `failed` rows.
4. Minimal reason vocabulary below (§4.5) — no generalized diagnostic taxonomy.
5. One coherent implementation commit (recommended) unless review finds a hard split benefit.

**Explicitly excluded:** D3 metrics/reporting; D4 smoke/manual/full-history/P&L; frozen JSON changes; second CVG threshold; `SurfaceSearch` / Sprint 007; unrelated bugs/refactors; new frameworks.

---

## 1. Context-read receipt

| Path | Why read | Fact used for D2 |
|------|----------|------------------|
| `AGENTS.md` | Session rules | Canonical path is SurfaceRunner; inspect/plan before edits; venv path |
| `docs/agenda/current_sprint.md` | Mode + authorization | `ACTIVE — D0/D1 COMPLETE; D2 AWAITING DESIGN`; Build mode; D2 design only |
| `docs/tmp/sprint006_d0_…_plan.md` | Frozen D2 outcomes | Joint `min_count_pct`; ceil→28; A1 calendar; taxonomy; all-leg spread; D2 handoff |
| `configs/sprint006_baseline_v1.json` | Exact contract | `cvg_count_col`; `joint_columns`; `expected_dates`; `max_leg_spread_pct_intent`; periods |
| `docs/tmp/sprint006_d1_…_plan.md` | Accepted D1 | Adapter + CLI + tie-break shipped; D2 gaps explicitly deferred; result model preserved |
| `src/backtest/surface_runner.py` | Date loop / result | Feature-derived dates; empty signals → `continue`; `SurfaceRunResult` has no `date_status` |
| `src/backtest/pipeline.py` S2 | Count eligibility | Mom `count_col` only; threshold = float `min_count_pct * window_size` (no ceil); missing `count_col` skips filter |
| `src/backtest/pipeline.py` S3–S5 | No-trade observability | Failed structures kept as rows; with `include_diagnostics=True`, zero-included dates remain in trade_log |
| `src/backtest/run_config.py` | Config surface | Has `count_col` / `min_count_pct`; **no** `cvg_count_col` |
| `src/backtest/sprint006_baseline.py` | Adapter / receipt | Maps only `momentum_col/cvg_col/count_col`; writes trade/date/run artifacts; lists D2 items as deferred |
| `src/backtest/option_surface.py` | Spread gate | Straddle filters body; iron fly filters **OTM wings only**; body shorts unfiltered |
| `src/backtest/surface_metrics.py` | Date summary | Summaries only dates present in trade_log — cannot resurrect skipped dates |
| `src/backtest/surface_run_config.py` | Naming helpers | `derive_cvg_and_count_cols` returns mom count only; no cvg-count helper |
| `docs/surface_engine_data_contract.md` §S2 | Documented I3 | Still documents float product guard (pre-ceil / pre-joint) |
| `tests/contract/test_step2_signals_contract.py` | S2 coverage | No joint CVG-count or ceil tests |
| `tests/contract/test_orchestration_contract.py` | Empty-date behavior | Pins today’s silent skip (`test_empty_signals_date_skipped_without_s5_call`) |
| `tests/unit/test_option_surface_{straddle,ironfly}.py` | Spread evidence | Straddle body gate tested; iron-fly wing-only gate tested |
| `tests/unit/test_sprint006_baseline_adapter.py` | Adapter pins | Asserts `cvg_count_col` is **not** a config attribute today |
| `tests/unit/test_surface_runner_data_flow.py` | Runner wiring | End-to-end synthetic runner; no date-status |
| Git HEAD / tree | Preconditions | HEAD `9edab17`; ancestors include `241b0d3` + `c6b1735`; working tree clean |

**Git state at design:** HEAD `9edab17` (`docs(sprint006): accept D1 trusted baseline runner`), branch `main`, working tree clean. No code changes and no economic execution for this design.

---

## 2. Evidence-based current-state assessment

### Already correct for D2 intent (prefer tests / leave alone)

| Behavior | Evidence |
|----------|----------|
| Canonical single-config engine | `SurfaceRunner.run_single_config` runs S1→S5 + metrics |
| Straddle all-body `max_leg_spread_pct` | `build_straddle_from_surface` filters `is_body` by `spread_pct`; unit tests cover tight/wide |
| Structure failures remain observable when diagnostics on | S3 keeps `structure_ok=False` rows; S5 keeps excluded rows if `include_diagnostics=True` (frozen True) |
| D1 twin mid/cross launcher + outside-repo outputs + receipt | Accepted `241b0d3` + `c6b1735` |
| Cap tie-break `ticker` asc | Done in D1 |
| Feature artifact publishes `cvg_count_42_8` | `feature_backfill_v1` template `cvg_count_{max}_{min}`; contract names the column |
| D0 economic thresholds | Frozen; must not change |

### Gaps that block D2 outcomes

| Gap | Current behavior | Why it blocks |
|-----|------------------|---------------|
| Mom-only count eligibility | S2 filters `count_col` only | Violates joint Mom+CVG rule |
| Float threshold, not ceil | `min_count_pct * window_size` | Violates frozen ceil rule (equal for 0.8×35 today, wrong as the pinned rule) |
| No `cvg_count_col` on config / adapter | Contract field ignored | Cannot express joint column explicitly |
| Silent empty-signal skip | `if signals.empty: continue` | Empty/ineligible dates vanish from trade_log and date_summary |
| Feature-derived trade dates | `_get_trade_dates` from features only | A1-only / `surface_valid=False`-only dates never enter the loop |
| No date-status artifact | `SurfaceRunResult` lacks it; adapter does not write it | Cannot satisfy expected-date invariant or D3 handoff |
| Iron-fly body unfiltered | Body quotes used without `max_leg_spread_pct` | Violates all-leg spread intent |
| Missing `count_col` silently disables filter | `if config.count_col in feat_slice.columns` | Would also allow silent bypass if joint column absent |

### Non-gaps for D2 (real, owned later)

* Dual conditional/calendar return metrics, Sharpe, drawdown, yearly packs → **D3**
* Real-data smoke, manual sample, full-history mid/cross P&L → **D4**
* Proving the unseen full-history run has zero `failed` dates → **D4 acceptance**, not D2
* Search CLI / `SurfaceSearch` → out of scope

---

## 3. Requirement → capability / gap mapping

| Frozen D2 outcome | Current capability | Gap / action |
|-------------------|--------------------|--------------|
| Joint `min_count_pct=0.80` on Mom + CVG | Mom only | Add joint AND filter in S2 |
| `required_count = ceil(0.80 × 35) = 28` | Float product | Switch to `math.ceil(...)` |
| Both counts ≥ 28 before ranking | Ranking after mom filter only | Apply both filters before rank |
| No second CVG threshold / no frozen-rule change | N/A | Reuse `min_count_pct` only |
| Independent A1 `entry_date` calendar incl. `surface_valid=False` | Feature dates only | Expected dates from A1 meta |
| Reconcile expected A1 dates to feature dates | None | Explicit reconciliation |
| Classify every expected date once | Silent attrition | Build `date_status` with invariant check |
| Empty-signal / ineligible / no-structure observable | Partially (no-structure only if signals non-empty) | Stop empty-signal skip; classify |
| Missing features / incomplete processing as `failed` | Missing features invisible; exceptions abort | Missing → `failed` row; unexpected exception still aborts run |
| `max_leg_spread_pct=0.50` on all four IF legs | Wings only | Filter body too |
| Date-status on shared result/output path | Not present | Extend `SurfaceRunResult` + adapter write/receipt |

Invariant D2 must enforce locally:

```text
expected dates = traded + valid_no_trade + failed
```

No missing expected dates; no duplicate membership across classes. `unresolved_failed_count > 0` ⇒ result must not be treated as complete (flag for D3/D4), but D2 does **not** require the future real-data run to be failure-free.

---

## 4. Proposed design

### 4.1 Joint Momentum / CVG eligibility

**Rule (frozen):**

```text
window_size = max_lag - min_lag + 1   # from mom_{max}_{min}_mean → 35
required_count = ceil(min_count_pct * window_size)  # ceil(0.80 × 35) = 28
keep row iff mom_42_8_count >= 28 AND cvg_count_42_8 >= 28
```

Apply **before** cross-sectional momentum ranking (current step-4 position in S2). Do not add a separate CVG threshold field. Do not change `long_top_pct`, `short_bottom_pct`, `cvg_filter_pct`, or ranking method.

**Representation (recommended):** add optional

```python
cvg_count_col: Optional[str] = None
```

to `BacktestRunConfig`.

* When `None` → preserve today’s mom-only behavior for non-006 callers/tests.
* When set (Sprint 006 adapter maps `feature_window.cvg_count_col`) → joint filter is mandatory.

**Threshold helper:** one small local function in `pipeline` (or tiny shared helper next to existing window parse) computing `required_count` via `math.ceil`. Do not invent a feature-schema registry.

### 4.2 Required-column handling

| Situation | Behavior |
|-----------|----------|
| `count_col` configured but absent from feature columns | **Hard fail the run** at feature-load / first use (do not silently skip the filter) |
| `cvg_count_col` set but absent | **Hard fail the run** the same way |
| `cvg_count_col is None` | Mom-only path; do not look for a CVG count column |
| Column present; row count below threshold / NaN count | Drop row from eligibility (existing pattern); may yield empty signals → `valid_no_trade` |
| A1 expected date with **no feature rows at all** | `failed` / `missing_features` (not a schema hard-fail) |

Optional cheap 006 preflight (adapter): when building twin configs, verify baseline feature parquet contains `momentum_col`, `cvg_col`, `count_col`, and `cvg_count_col`. Prevents a long silent wrong path before the engine starts. This is proportional because the failure mode is exactly “joint eligibility never applied.”

### 4.3 Independent expected-date construction and feature reconciliation

**Expected calendar authority (frozen):**

1. From loaded A1 meta (`OptionSurfaceDB.meta_df`): sorted unique `entry_date_key` with `entry_date ∈ [config.start_date, config.end_date]` (**include** dates that appear only on `surface_valid=False` rows).
2. Feature date set: unique `features.date` in the same closed interval.
3. Reconciliation:
   * Every expected A1 date gets exactly one status.
   * Expected date ∉ feature dates → `failed` (`missing_features`) by default.
   * Feature dates ∉ A1 → **not** expected members; record counts (and optionally a short list) in `run_summary` / receipt reconciliation evidence only.

Implement expected-date extraction as a small method on `SurfaceRunner` (or a tiny helper used by it) over already-loaded `self.surface_db.meta_df`. Do not add a calendar service.

### 4.4 Exact status boundaries

Classification runs **per expected date**, after that date’s processing attempt (or non-attempt for missing features):

| Status | Exact boundary |
|--------|----------------|
| `traded` | Pipeline completed for the date **and** ≥1 trade_log row with `included_in_portfolio=True` |
| `valid_no_trade` | Pipeline completed for the date **and** zero included names, for an allowed completed outcome (below) |
| `failed` | Date not fully/correctly processable under the frozen contract (below) |

**`valid_no_trade` (allowed completed outcomes):**

* Feature rows exist for the date, required columns present, S1–S2 complete, and S2 returns empty (universe empty after join, all NaN scores, all fail joint count, or empty long+short pools after CVG retention) → reason `empty_signals`.
* S2 non-empty, S3–S5 complete, and zero `included_in_portfolio=True` (all `no_tradeable_structure`, earnings exclusions with diagnostics retained, sizing rejects, etc.) → reason `no_included_names`.

**`failed`:**

* Expected A1 date missing from feature date set → `missing_features`.
* Required count/signal columns missing for the run → hard-fail whole run (schema), which makes the result incomplete (not a quiet per-date `valid_no_trade`).
* Unexpected exception while processing a date → **abort the run** (do not soft-map to a `failed` row). A crashed run is incomplete by definition; inventing partial status after a hard error invites false completeness.

**Not `failed`:** ordinary economic emptiness, universal structure failure on candidates, or long-only/short-zero books that still include ≥1 name (`traded`).

Replace:

```python
if signals.empty:
    continue
```

with explicit `valid_no_trade` recording (and **no** S5 call — same as today for S5, but the date remains observable).

### 4.5 Minimal reason / diagnostic information

Keep a **small fixed vocabulary**, not a taxonomy framework:

| `reason` | Used when |
|----------|-----------|
| `missing_features` | A1 expected date absent from features |
| `empty_signals` | S2 returned empty after a completed attempt |
| `no_included_names` | S5 completed; zero included |
| `none` / null | `traded` |

`date_status` columns (minimal):

* `entry_date`
* `status` ∈ {`traded`,`valid_no_trade`,`failed`}
* `reason` (as above)
* optional counts useful to D3 without becoming a report: `n_candidates`, `n_included` (0 when no rows)

Also attach to `run_summary`:

* `n_expected_dates`, `n_traded_dates`, `n_valid_no_trade_dates`, `n_failed_dates`
* `unresolved_failed_count` (= `n_failed_dates`)
* `date_status_complete` (= `unresolved_failed_count == 0`)
* reconciliation: `n_feature_dates_absent_from_a1` (and maybe truncated examples)

### 4.6 Iron-fly all-leg spread enforcement

In `build_ironfly_from_surface`, when `max_leg_spread_pct is not None`:

1. Filter **body** quotes (`is_body`) by `spread_pct <= max_leg_spread_pct` **before** selecting ATM shorts (same pattern as straddle).
2. Keep existing OTM wing filter.
3. If either body leg disappears → raise the existing style of construction error (captured by S3 as `structure_ok=False`), not a new error framework.

Do **not** change wing-selection rule, delta target, settlement, or fill math. Do **not** retarget iron-condor in this deliverable beyond leaving its existing wing-only filter alone unless a shared helper naturally covers it without scope creep; Sprint 006 baseline is iron-fly only. Prefer editing the iron-fly body path explicitly.

### 4.7 Integration with `SurfaceRunResult`, writer, and receipt

```text
A1 meta + features + config
        │
        ▼
SurfaceRunner.run_single_config
  - expected_dates ← A1
  - reconcile vs feature dates
  - per date: classify + maybe append trade rows
  - build date_status; assert partition invariant
  - existing build_date_summary(trade_log) unchanged in role
        │
        ▼
SurfaceRunResult(
  config, trade_log, date_summary, run_summary, date_status  # date_status NEW
)
        │
        ▼
sprint006_baseline.write_run_outputs
  + date_status_<run_id>.parquet
run_receipt.json
  + per-run date_status artifact digest
  + unresolved_failed_count / date_status_complete
  + deferred list updated (remove completed D2 items; keep D3/D4)
```

Adapter changes stay thin:

* Map `feature_window.cvg_count_col` into configs (extend `_FEATURE_COLUMN_FIELDS` or equivalent explicit allow-list).
* Persist `date_status`.
* Do **not** redesign CLI, overwrite policy, path refusal, or twin-run orchestration.

`date_summary` remains trade_log-derived (may omit pure `valid_no_trade`/`failed` dates with no rows). **`date_status` is the authoritative coverage table.** D3 must not infer coverage solely from `date_summary`.

---

## 5. Alternatives considered (material choices only)

### 5.1 How to resolve `cvg_count_42_8`

| Option | Verdict |
|--------|---------|
| **A. Optional `BacktestRunConfig.cvg_count_col` + adapter map from contract** (recommended) | Explicit, narrow, matches frozen JSON field, backward compatible when `None` |
| B. Derive `cvg_count_{max}_{min}` from `momentum_col` whenever present | Implicit magic; harder to see in effective config dump; still needs presence policy |
| C. Generalized feature-schema / multi-count registry | Rejected — overbuilt for one joint rule |
| D. Hard-code `cvg_count_42_8` in Sprint 006 adapter only and bypass config | Rejected — engine would remain wrong for the mapped `BacktestRunConfig`; twin path would diverge |

### 5.2 Where to build date status

| Option | Verdict |
|--------|---------|
| **A. Inside `SurfaceRunner` → `SurfaceRunResult.date_status`** (recommended) | Single engine owns the invariant; adapter stays thin; reusable for later studies |
| B. Adapter-only post-pass over trade_log | Cannot distinguish empty-signal skip from never-scheduled without runner changes anyway |
| C. New coverage service / publication framework | Rejected by scope rules |

### 5.3 Exception policy for per-date failures

| Option | Verdict |
|--------|---------|
| **A. Soft-classify known cases; abort on unexpected exceptions** (recommended) | Observable coverage without inventing false completeness after crashes |
| B. Catch-all → mark `failed` and continue | Hides severe bugs; risks “complete” accounting after partial corruption |

---

## 6. Proposed file-level changes

| File | Change |
|------|--------|
| `src/backtest/run_config.py` | Add optional `cvg_count_col: Optional[str] = None` (+ light validation if needed) |
| `src/backtest/pipeline.py` | Ceil required-count; joint filter when `cvg_count_col` set; hard-fail if configured count columns missing |
| `src/backtest/surface_runner.py` | A1 expected dates; reconciliation; no silent empty skip; build `date_status`; extend `SurfaceRunResult`; summary counters / completeness flag |
| `src/backtest/option_surface.py` | Iron-fly body `max_leg_spread_pct` filter |
| `src/backtest/sprint006_baseline.py` | Map `cvg_count_col`; write `date_status_*.parquet`; receipt fields; trim D2 items from `DEFERRED_*` |
| `tests/contract/test_step2_signals_contract.py` (and/or sibling) | Joint filter, ceil=28 for (42,8), missing-column hard-fail |
| `tests/contract/test_orchestration_contract.py` / runner unit tests | Empty-signal → `valid_no_trade`; missing features → `failed`; partition invariant; update obsolete “skipped” assertion |
| `tests/unit/test_option_surface_ironfly.py` | Body spread tight/wide cases |
| `tests/unit/test_sprint006_baseline_adapter.py` | Maps `cvg_count_col`; writes date_status; receipt completeness keys |
| `docs/surface_engine_data_contract.md` §S2 I3 | **Minimal sync only:** document ceil + optional joint CVG count (avoid doc/code drift). No other doc churn |
| **Do not edit** | `configs/sprint006_baseline_v1.json`; D3/D4 code; `surface_search.py` / search CLI; unrelated known bugs |

Approximate effort: well inside the sprint’s 12–18h review trigger if scope stays as above.

---

## 7. Focused synthetic and regression tests

1. **Ceil derivation:** `(42,8)` + `min_count_pct=0.80` → required count `28` (not a float-product accidental pass).
2. **Joint eligibility:** ticker with `mom_count>=28` but `cvg_count<28` excluded before ranking; both ≥28 retained.
3. **Mom-only backward path:** `cvg_count_col=None` unchanged vs today’s filter.
4. **Missing required column:** configured `cvg_count_col` absent → hard fail (not silent skip).
5. **A1 expected calendar:** date present only on `surface_valid=False` meta rows still appears in expected set.
6. **Missing features:** A1 date absent from features → `failed`/`missing_features`; still counted in expected.
7. **Empty signals observable:** S2 empty → `valid_no_trade`/`empty_signals`; S5 not called; date not absent.
8. **No-structure date:** signals exist, all `structure_ok=False` → `valid_no_trade`/`no_included_names` (with diagnostics on).
9. **Traded date:** ≥1 included → `traded`.
10. **Partition invariant:** for a synthetic multi-date fixture, `expected == traded ∪ valid_no_trade ∪ failed` and pairwise disjoint; violation fails the test / run guardrail.
11. **Iron-fly body spread:** body `spread_pct` above threshold fails construction even if wings are liquid; below threshold still builds.
12. **Adapter:** effective config includes `cvg_count_col`; `date_status_*.parquet` written; receipt shows artifact + `unresolved_failed_count` / `date_status_complete`; D2 items removed from deferred list.
13. **Regression subset after edits:** existing S2, orchestration, surface runner data-flow, option_surface straddle/ironfly, step5, sprint006 adapter suites.

**Forbidden in D2 validation:** full-history real-data economic run; aggregate P&L/Sharpe inspection; proving real-data `failed` count is zero.

---

## 8. Ordered implementation steps

1. Accept this plan (including the five engineering approval items in the Review summary).
2. Add optional `cvg_count_col` + S2 ceil/joint filter + missing-column hard-fail; add S2 tests.
3. Implement A1 expected-date loop + reconciliation + `date_status` on `SurfaceRunResult`; replace empty-signal skip; add orchestration/runner tests; enforce partition invariant.
4. Implement iron-fly body spread filter + unit tests.
5. Extend adapter mapping, output writer, receipt; update adapter tests and deferred list.
6. Minimal S2 data-contract I3 doc sync (optional but recommended with the code change).
7. Run focused pytest subset; record results in the implementation handoff (not this design doc).
8. Stop. Do not start D3/D4 or real-data P&L.

---

## 9. Acceptance criteria and stop conditions

### Design acceptance (this document)

- [ ] Review summary approved, including the five engineering decisions
- [ ] Frozen D0 JSON remains untouched; no P&L knobs reopened
- [ ] Plan does not authorize D3/D4, search work, or real-data economic execution
- [ ] Invariant `expected = traded + valid_no_trade + failed` is explicit
- [ ] Status boundaries and minimal reasons are explicit enough to implement without reinterpretation

### D2 implementation acceptance (after coding)

- [ ] Joint Mom+CVG eligibility uses `min_count_pct` only and `ceil` → 28 for `(42,8)`
- [ ] Configured `cvg_count_col` is mapped from the frozen contract and enforced in S2
- [ ] Missing configured count columns hard-fail (no silent bypass)
- [ ] Expected dates come from A1 (including `surface_valid=False`-only dates) within the frozen interval
- [ ] Every expected date classified exactly once; partition invariant held
- [ ] Empty-signal / no-included / missing-feature cases are observable with the agreed reasons
- [ ] `unresolved_failed_count > 0` marks the result incomplete in summary/receipt
- [ ] Iron-fly applies `max_leg_spread_pct` to body and wings
- [ ] `date_status` available on `SurfaceRunResult` and persisted by the existing adapter path
- [ ] Focused pytest subset green
- [ ] No frozen JSON edits; no D3 metrics; no D4 real-data P&L

**Stop** when the above evidence exists. Pause for rescope if work expands into reporting packs, search frameworks, generalized DQ systems, or approaches the sprint implementation review trigger without a remaining D2 blocker.

---

## 10. Explicit out of scope

* Editing or versioning `configs/sprint006_baseline_v1.json` / any frozen threshold or period
* A second CVG threshold parameter or ranking/CVG-retention changes
* D3 decision report, calendar zero-fill Sharpe/drawdown, yearly packs, attribution, costs, concentration
* D4 smoke, manual sample, full-history mid/cross execution, P&L inspection, or proving zero real-data failures
* New runner/CLI/status framework/publication/DQ platform
* Redesigning the D1 adapter beyond persistence/mapping needed for D2
* `SurfaceSearch` / `run_surface_search.py` repair; Sprint 007 study matrix
* Iron-condor comparison / KB-001; earnings filters; Tier B; new features
* Unrelated refactors or known-bug fixes that do not block the frozen D2 outcomes
* Redundant evidence documents beyond this plan + later implementation notes in the usual place

---

## 11. Guardrails (proportional only)

| Guardrail | Realistic failure prevented | Why worth the cost |
|-----------|-----------------------------|--------------------|
| Joint count AND + explicit `cvg_count_col` | Trading names with thin CVG coverage while looking “eligible” | Direct frozen-contract economic correctness; few lines in S2 |
| Ceil required-count | Threshold drift vs frozen rule on non-integer products | One `math.ceil` call; pins D0 literally |
| Hard-fail missing configured count columns | Silent disable of eligibility when a column is absent | Cheap schema check; prevents false baseline |
| A1 expected calendar + partition assert | Silent date loss / double-counting | Core DoD; local assert after building `date_status` |
| Completeness flag from `unresolved_failed_count` | Treating incomplete coverage as an accepted full result | One summary/receipt field; D3/D4 consume it |
| Iron-fly body spread filter | Illiquid ATM shorts entering the book while wings are filtered | Matches straddle behavior; small builder change + unit test |

Omit: resumable run managers, generalized diagnostic ontologies, CRLF digest frameworks, catch-all exception→`failed` swallowers, new CLIs.

---

## 12. Commit / review-boundary recommendation

**Default: one coherent implementation commit** covering S2 joint/ceil eligibility, runner date-status, iron-fly body spread, and thin adapter persistence.

Splitting is **not** recommended unless implementation review discovers that iron-fly body filtering unexpectedly couples to unrelated builder behavior. The three behaviors are one frozen D2 acceptance bundle; separate commits would leave intermediate HEADs economically incomplete relative to the contract without material risk reduction.

---

## 13. P&L firewall reminder

D2 design and implementation remain behind the Sprint 006 firewall: no new aggregate P&L, Sharpe, rankings, or side performance from accepted real data. Synthetic tests only for behavior. Real-data economic execution stays in D4.

---

**End of proposed D2 design.** Implementation requires acceptance of this plan (and the engineering approval items). No D2 code, frozen-config, sprint-status, or economic execution is authorized by drafting this document alone.
