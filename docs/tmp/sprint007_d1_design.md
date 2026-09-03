# Sprint 007 D1 — Gross economics of the frozen trade expression

**Status:** `ACCEPTED — IMPLEMENTED` (evidence `C:/MomentumCVG_env/runs/sprint007_d1_20260903T013933Z/`, commit `516235f`, verdict `D1_CONTINUE_TO_D2`)  
**Updated:** 2026-09-02  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md)  
**Prerequisite:** D0 `READY_WITH_NARROW_ENABLING_CHANGE` — evidence `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/` (commit `8a59474`)  
**Prior evidence:** [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)

---

## Summary

| Item | D1 design decision |
|---|---|
| **Question** | Does the frozen midpoint trade expression contain sufficiently broad and stable gross economic margin to justify investigating its implementation shortfall? |
| **Method** | One gross-margin notebook for narrative/charts; one small read-only helper; one unit test file |
| **Fill / window** | **Mid only**; primary window `2020-01-01` → `2026-07-10` (inclusive) |
| **Inputs** | Official Sprint 006 run `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` — `*_mid` parquets + `decision_report.json` |
| **Authority for reconciliation** | `decision_report.json` → `by_fill.mid.primary` (accepted Sprint 006 calculation) |
| **Runtime target** | Minutes on accepted artifacts; no `SurfaceRunner` rerun |
| **Authorization** | This design only. D1 implementation remains unauthorized until accepted. |

D1 evaluates **frozen `42:8` selection + frozen CVG rule + current long/short expression + current holding period + midpoint sizing**. It does **not** measure pure Momentum/CVG signal quality or executable return.

---

## Frozen decisions (D1)

- **Fill:** midpoint (`fill_label == "mid"`) only. Cross artifacts are not opened.
- **Window:** `PRIMARY_START` / `PRIMARY_END` from `surface_decision_report.py` (`2020-01-01` → `2026-07-10`). Calendar authority = `date_status_mid`.
- **Population:** `included_in_portfolio == True` on traded dates within the window (expected **9,212** trade keys per D0).
- **Primary dollar unit:** `Σ pnl_total` over included trades.
- **Primary CAR unit (sign gate):** View A **mean cycle CAR** = mean of `cycle_return_on_capital_at_risk` on **traded** dates only (conditional on traded; excludes `valid_no_trade`).
- **Side components:** `direction == "long"` and `direction == "short"` on included trade rows.
- **Year buckets:** calendar year of `trade_date`; year P&L = `Σ pnl_total` of included trades in that year (trade-level sum, not date-summary re-aggregation).
- **Output labels:** **accepted calculation** (reconciled to Sprint 006 report), **D1 gate statistic** (frozen below), **exploratory description** (charts only; no new gates).

---

## Frozen continue/stop gate (scorecard)

All four parts must pass for `D1_CONTINUE_TO_D2`. Formulas are frozen here; they must not be changed after D1 output is opened.

| Part | Statistic | Pass criterion |
|---|---|---|
| **G-Sign** | Portfolio midpoint economics | `total_pnl > 0` **and** `view_a_mean_cycle_car > 0` |
| **G-Breadth** | Tail dependence | `total_pnl_excl_top5_dates > 0` **and** `total_pnl_excl_top5_tickers > 0` |
| **G-Location** | Side participation | ∃ side ∈ {long, short} with `side_pnl > 0` **and** `side_n_trades / n_included_trades ≥ 0.10` |
| **G-Stability** | Calendar persistence | `n_years_with_positive_pnl ≥ 2` **and** `total_pnl_excl_best_year > 0` |

**Ranking rules (frozen):**

- **Top 5 dates:** rank traded `trade_date` by `date_pnl = Σ included pnl_total` on that date; exclude the five highest `date_pnl` dates; recompute `total_pnl` on remaining included trades.
- **Top 5 tickers:** rank `ticker` by `ticker_pnl = Σ included pnl_total`; exclude the five highest `ticker_pnl` tickers; recompute `total_pnl`.
- **Best year:** calendar year with highest `year_pnl`; exclude all included trades in that year; recompute `total_pnl`.

**Final verdict mapping:**

| Condition | Verdict |
|---|---|
| Reconciliation or artifact precondition fails | `D1_BLOCKED` |
| Reconciliation passes and all four gate parts pass | `D1_CONTINUE_TO_D2` |
| Reconciliation passes and any gate part fails | `D1_STOP_CURRENT_EXPRESSION` |

---

## Reconciliation (accepted Sprint 006 midpoint)

Before any gate is evaluated, D1 must reconcile recomputed mid-primary metrics to the accepted decision report.

| Metric | Recompute via | Report path | Tolerance |
|---|---|---|---|
| `total_pnl` | `Σ pnl_total` on included trades after `filter_to_window` | `by_fill.mid.primary.long_short` long+short `pnl_total` (or equivalent sum) | `abs(Δ) ≤ max($0.01, 1e-9 × abs(reference))` |
| `view_a_mean_cycle_car` | `compute_view_a(date_status, date_summary)["mean_cycle_car"]` | `by_fill.mid.primary.view_a_conditional.mean_cycle_car` | `abs(Δ) ≤ 1e-9` |
| `n_included_trades` | count included rows on traded dates | `by_fill.mid.primary.long_short` row counts sum | exact |
| `n_traded_dates` | `count_date_classes` → `n_traded_dates` | `by_fill.mid.primary.date_class_counts.n_traded_dates` | exact |

Reconciliation failure → `D1_BLOCKED`; do not interpret gate pass/fail.

Optional sanity check (non-blocking unless mismatch exceeds tolerance): `evaluate_fill_window(...)` on mid artifacts should match the same report block fields.

---

## Architecture: notebook-first + minimal helper

```
notebooks/sprint007/d1_gross_margin.ipynb          ← committed clean; narrative + charts
src/backtest/sprint007_d1_gross_margin.py          ← load, reconcile, gate scorecard, exclusion stats
tests/unit/test_sprint007_d1_gross_margin.py       ← synthetic mid frames; gate edge cases
```

**Reuse (import; do not reimplement):**

| Module | Use |
|---|---|
| `surface_decision_report.py` | `PRIMARY_START`, `PRIMARY_END`, `filter_to_window`, `compute_view_a`, `compute_long_short_attribution`, `evaluate_fill_window`, `compute_yearly_metrics` |
| `surface_metrics.py` | Cycle P&L/CAR field semantics (`pnl_total`, `capital_at_risk_dollars`, `cycle_return_on_capital_at_risk`) |
| `sprint007_artifact_validation.py` | `OFFICIAL_RUN_DIR`, parquet column lists, optional pre-flight `run_d0_validation()` call |

Do **not** add a new `src/analysis/` package, bridge math, or cross-fill logic.

**Minimal new helper surface (proposed):**

- `load_mid_primary_tables(run_dir) -> MidPrimaryBundle` — read `date_status`, `date_summary`, `trade_log` mid parquets with explicit columns; apply `filter_to_window`.
- `load_accepted_mid_primary_report(run_dir) -> dict` — parse `decision_report.json` block `by_fill.mid.primary`.
- `reconcile_mid_primary(bundle, report) -> ReconciliationResult` — table above.
- `compute_d1_gate_scorecard(bundle) -> D1Scorecard` — frozen gate parts + verdict.
- `write_d1_manifest(scorecard, reconciliation, output_path)` — JSON for evidence dir.

Target footprint: ~120–180 LOC helper; ~80–120 LOC tests.

---

## Notebook sections (proposed)

Committed notebook: `notebooks/sprint007/d1_gross_margin.ipynb` (clean; no outputs in repo).

| § | Title | Purpose | Outputs |
|---|---|---|---|
| 0 | **Preamble** | Repo-root bootstrap (`sys.path`), state D1 question and inference boundary | — |
| 1 | **Preconditions** | Optional `run_d0_validation()` summary; confirm mid-only scope | Pass/fail one-liner |
| 2 | **Reconciliation** | Call helper; show Δ vs `decision_report.json` mid-primary | Reconciliation table |
| 3 | **Portfolio gross margin** | `total_pnl`, View A mean CAR, capital at risk; label accepted calculation | KPI table |
| 4 | **Long vs short location** | Side P&L, trade counts, share of book; which side carries margin | Side table + chart |
| 5 | **Breadth** | Date and ticker P&L ranks; P&L after top-5 exclusions | Exclusion table |
| 6 | **Stability** | Calendar-year P&L bars; best-year exclusion | Year table + chart |
| 7 | **Scorecard** | G-Sign / G-Breadth / G-Location / G-Stability pass-fail + final verdict | `d1_scorecard.json` preview |
| 8 | **Limits** | Mid ≠ executable; not signal IC; D2 owns mid→cross bridge | Prose only |

**Purposeful visualizations (max 4):**

1. **Cumulative included P&L by trade date** — portfolio trajectory within primary window.
2. **Calendar-year P&L bars** — stability gate context (color pass/fail vs 0).
3. **Long vs short aggregate P&L** — location gate (horizontal bar or grouped bar).
4. **Ticker P&L distribution** (histogram or top-10 / bottom-10 table) — breadth context; not a new gate.

No cross-fill overlays, no spread-cost scatter, no filter sweeps.

---

## Required artifacts (evidence, outside repo)

Directory: `C:/MomentumCVG_env/runs/sprint007_d1_<timestamp>/`

| File | Content |
|---|---|
| `d1_scorecard.json` | Gate parts, metrics, verdict, reconciliation deltas |
| `d1_gross_margin_manifest.json` | Input paths, report SHA, `d1_code_commit_sha`, window, included-key count |
| `d1_side_attribution.csv` | long/short pnl, n_trades, share |
| `d1_breadth_exclusions.csv` | baseline pnl, excl-top5-dates, excl-top5-tickers |
| `d1_yearly_pnl.csv` | year, year_pnl, n_trades |
| `d1_gross_margin.executed.ipynb` | Fresh-kernel execution |
| `d1_gross_margin.html` | HTML export |
| `execution_receipt.json` | SHAs of executed notebook/HTML, repo SHAs, timestamps |

---

## D1 implementation steps (when authorized)

1. **Helper module** — load mid-primary bundle, reconcile, compute frozen scorecard.
2. **Unit tests** — synthetic trades covering: sign pass/fail, top-5 date/ticker exclusions, side 10% threshold, two-year stability, best-year exclusion, reconciliation tolerance.
3. **Notebook** — call helper only; render four charts; write manifest paths.
4. **Fresh-kernel execution** — `nbconvert --execute` with venv + `PYTHONPATH` (same pattern as D0 `export_d0_evidence`).
5. **Record evidence path** in sprint agenda after review.

---

## Acceptance evidence (D1 complete)

- [ ] Reconciliation to `by_fill.mid.primary` within tolerance
- [ ] All four frozen gate parts evaluated with explicit pass/fail
- [ ] Final verdict ∈ {`D1_CONTINUE_TO_D2`, `D1_STOP_CURRENT_EXPRESSION`, `D1_BLOCKED`}
- [ ] One-paragraph answer: where margin resides (long, short, both, or neither)
- [ ] `tests/unit/test_sprint007_d1_gross_margin.py` passes
- [ ] Clean committed notebook + executed evidence outside repo with hashes
- [ ] No cross-fill, execution, filter, or signal analysis opened

---

## Stop conditions

| Trigger | Action |
|---|---|
| D0 validation fails on official artifacts | `D1_BLOCKED`; stop before economics |
| Receipt / schema / included-key count ≠ 9,212 | `D1_BLOCKED` |
| Reconciliation outside tolerance | `D1_BLOCKED`; fix helper before interpreting gates |
| Proposal opens cross artifacts, bridge math, or subgroup search | Rescope to D2+ |
| Any gate statistic redefined after viewing outputs | Invalidate run; redesign required |

---

## Non-goals

- Cross-fill losses, fill-assumption sensitivity, or mid→cross bridge (D2).
- Execution quality, break-even fill, spread/liquidity filters, or package-tradability (D3).
- Alternative structures, wing rules, iron condor, or maturity/holding changes.
- Signal-window search, `42:8` retuning, or Momentum/CVG IC.
- Subgroup winner selection, best-cutoff search, or same-sample filter optimization.
- Claiming midpoint is attainable, recoverable, or expected live return.
- `SurfaceRunner` rerun or producer-cache inputs.

---

## Inference boundary

D1 answers whether the **current selected trade expression** has enough **midpoint gross margin** — broad, located, and stable — to justify spending D2–D3 on implementation shortfall. A `D1_STOP` or `D1_BLOCKED` verdict does not invalidate Momentum/CVG as signal families. A `D1_CONTINUE` verdict does not imply the expression is tradable at midpoint; it only clears the sprint gate to diagnose **why** cross economics diverge.

---

## Expected footprint (implementation)

| Path | Purpose |
|---|---|
| `src/backtest/sprint007_d1_gross_margin.py` | ~120–180 LOC |
| `tests/unit/test_sprint007_d1_gross_margin.py` | ~80–120 LOC |
| `notebooks/sprint007/d1_gross_margin.ipynb` | Gross-margin narrative (committed clean) |
