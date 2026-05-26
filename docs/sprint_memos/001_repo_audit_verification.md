# Sprint memo 001 — Repo audit and verification

**Status:** Draft / in progress  
**Sprint:** 001  
**Mode:** Audit → Verification  
**Started:** 2026-05-24  
**Closed:** _TBD_

---

## How to use this memo

This memo is intentionally a **living sprint memo** while Sprint 001 is in progress.

Use it for:

1. Handoff between Cursor sessions while context is high.
2. Tracking your review comments on `repo_audit.md` and `correctness_audit.md`.
3. Recording the final Session B test decision.
4. Closing the sprint once tests and follow-up docs are complete.

Do **not** treat this as final until the `Closed` field is filled and the closeout checklist is complete.

---

## Current state

### Session A — Audit

| Item | Status |
|------|--------|
| `docs/repo_audit.md` drafted | Done — awaiting HD review |
| `docs/correctness_audit.md` drafted | Done — awaiting HD review |
| SurfaceRunner functionality/data-flow mapping | Done — HD reviewed `docs/surface_runner_data_flow.md` |
| `docs/README.md` linked new audit docs | Done |
| Production code changes | None |
| Tests run in Session A | None |

### Session B — Verification

| Item | Status |
|------|--------|
| Test target selected | Done — synthetic `SurfaceRunner.run_single_config()` data-flow test |
| Test implemented | Not started |
| `pytest tests/ -q` run | Not started |
| Production fix needed? | Unknown |

---

## Session A summary

### Scope reviewed

Surface-first v1 backtest path:

```text
Stage A — Precompute store
scripts/precompute_option_surface.py
  → src/features/option_surface_analyzer.py::OptionSurfaceBuilder
  → surface meta + quote parquet

Stage B — Backtest scaffold
scripts/run_surface_search.py
  → BacktestRunConfig / SurfaceSearch
  → SurfaceRunner
  → OptionSurfaceDB + build_*_from_surface()
  → trade log + summaries
```

### Main conclusion

`SurfaceRunner` is still the right canonical path, but it is **not yet a decision-quality backtest engine**. The strongest layer today is the surface assembly math in `src/backtest/option_surface.py`; the weakest layer is the engine/portfolio/metrics contract around it.

### Major findings to review

| Finding | Doc section |
|---------|-------------|
| SurfaceRunner has significant undeveloped functionality; current design completeness was mapped before choosing the Session B test | `docs/surface_runner_data_flow.md` |
| Precompute store likely supports hold-to-expiry entry-date research, but needs coverage diagnostics and manifest | `docs/repo_audit.md` → Stage A / P0-P1 gaps |
| `run_surface_search.py` likely has CLI wiring issue: passes `contract_multiplier` to `SurfaceDataPaths`, which has no such field | `docs/repo_audit.md` → Known implementation concern |
| Runner currently emits `pnl_per_share`, but does not yet appear to convert max-loss budget into integer contracts / dollar PnL | `docs/repo_audit.md` → P0 gaps |
| Current cap is `max_names_per_side`, not clearly the v1 “50 max concurrent” policy | `docs/repo_audit.md` → P0 gaps / open questions |
| Metrics use return on body credit; v1 needs return on max-loss budget and Sharpe on capital/risk units | `docs/repo_audit.md`, `docs/correctness_audit.md` |
| Low-level surface assembly tests are already strong for straddle, iron fly, and iron condor | `docs/correctness_audit.md` |
| Missing tests are mostly runner/precompute/pipeline/metrics, not basic option payoff math | `docs/correctness_audit.md` |

---

## HD review checklist

Review in this order:

1. [ ] `docs/repo_audit.md`
2. [ ] `docs/correctness_audit.md`
3. [ ] This memo (`docs/sprint_memos/001_repo_audit_verification.md`)
4. [ ] Session A.1 SurfaceRunner functionality/data-flow map (`docs/surface_runner_data_flow.md`)

While reviewing, decide:

- [ ] Do the P0/P1/P2 priorities feel right?
- [ ] Is `SurfaceRunner` still the canonical path after seeing the gaps?
- [ ] Does the Session A.1 map confirm the runner selection/settlement boundary as the right Session B test?
- [ ] Should any finding be reclassified as P0/P1/P2?
- [ ] Are any missing features actually already implemented elsewhere?

---

## HD feedback log

Add review feedback here before asking the agent to revise the audit docs.

| Date | Doc / section | Feedback | Action |
|------|---------------|----------|--------|
| 2026-05-25 | `repo_audit.md`, `correctness_audit.md`, Session B plan | HD broadly agrees with the assessment, but wants stronger acknowledgement that SurfaceRunner needs significant undeveloped functionality and that the current test recommendation is not fully proven. Allocate time in current sprint to map functionality and data flow before Session B. | Added Session A.1 mapping gate; revised audit docs and memo; Session B test remains pending. |
| 2026-05-25 | Session A.1 mapping | HD clarified that A.1 mapping had not yet been done and should be comprehensive. | Drafted `docs/surface_runner_data_flow.md`; updated recommendation toward a synthetic `SurfaceRunner.run_single_config()` data-flow test. |
| 2026-05-25 | `docs/surface_runner_data_flow.md` HD review | HD agrees with implemented / partial / missing classification, but wants even implemented areas reviewed with tests in a following sprint. HD approves the full synthetic runner data-flow test and accepts that it may intentionally fail on missing desired v1 behavior. HD pins 50 max positions as total long+short and prefers completing separable implementation in `pipeline.py` for easier future unit testing. | Updated A.1 map, correctness audit, sprint agenda, v1 spec clarification, and Session B decisions. |

---

## Open decisions before Session B

### Decision 0 — SurfaceRunner functionality/data-flow map

Current decision:

> Extend Sprint 001 Session A before tests. Map the SurfaceRunner data flow and required functionality to confirm the current design is complete enough to support the intended backtest and to choose the right Session B verification target.

Artifact:

- `docs/surface_runner_data_flow.md`

Mapping should cover:

| Area | Question |
|------|----------|
| Inputs | What files/artifacts/configs must enter the runner? |
| Function ownership | Which module owns universe, signal, surface lookup, assembly, sizing, settlement, metrics, and outputs? |
| Missing functionality | Which required engine responsibilities are missing, partial, or already implemented? |
| Data contracts | Which columns and semantics must pass from Stage A to Stage B? |
| Test target | Which boundary is highest-value for the single Session B test? |

HD decision:

- [x] Do this in current Sprint 001 before Session B
- [ ] Defer to Sprint 002

Notes:

- Mapping draft reviewed by HD on 2026-05-25.
- HD agrees with the implemented / partial / missing classification.
- Even implemented areas should receive implementation/test review in a following sprint.

### Decision 1 — Sprint 001 test target

Current recommendation:

> Test `SurfaceRunner.run_single_config()` with small synthetic parquet fixtures, rather than testing only `_select_size_and_settle()` or adding another pure iron-fly payoff truth-table row.

Status:

> Approved by HD review of `docs/surface_runner_data_flow.md`.

Rationale:

- Existing `test_option_surface_ironfly.py` already covers entry credit, max loss, and expiry payoff well.
- The more valuable risk is the whole engine data flow: how input artifacts become universe, signals, structures, selected/settled trades, and summaries.
- A full-run synthetic fixture still covers selection/settlement, while also testing schema/date/runner integration.
- This test can document the current missing sizing / dollar-PnL behavior without refactoring production code.

HD decision:

- [x] Approve synthetic `run_single_config()` data-flow test
- [ ] Prefer direct `_select_size_and_settle()` boundary test
- [ ] Prefer pure assembly payoff truth-table test
- [ ] Defer final approval until HD reviews SurfaceRunner functionality/data-flow mapping

Notes:

- The A.1 mapping suggests the original private-helper test is too narrow if only one test is allowed.
- HD is comfortable with a test that intentionally fails on missing desired v1 fields, because that failure highlights required engine work before decision-quality backtesting.

### Decision 2 — Cap semantics for future build sprint

Question:

> Does “50 max concurrent positions” mean 50 total across long+short, or 50 short structures plus a separate long-side budget?

HD decision:

- [x] 50 total across both sides
- [ ] 50 short structures only; long side separate
- [ ] Decide in Sprint 002

Notes:

- Exact number can change later, but current v1 semantics are total positions across long and short.

### Decision 3 — Engine boundary

Question:

> Should `pipeline.py` steps 4–6 be completed, or should `SurfaceRunner` own the engine implementation and keep `pipeline.py` as step1/step2 helpers?

HD decision:

- [x] Complete separable implementation in `pipeline.py` for future unit testing
- [ ] Keep implementation in `SurfaceRunner`
- [ ] Decide after Session A.1 mapping / Session B evidence

Notes:

- SurfaceRunner can remain the orchestration layer, but separable universe/signal/structure/exclusion/selection/sizing/cost functions should live in or move toward `pipeline.py` so they can be unit-tested independently.

---

## Session B plan (approved after HD review of Session A.1)

### Proposed test

File:

```text
tests/unit/test_surface_runner_data_flow.py
```

Intent:

Verify that `SurfaceRunner.run_single_config()` can consume small synthetic liquidity/features/surface artifacts and produce a coherent trade log, date summary, and run summary while preserving hand-calculated settlement semantics for at least one short structure.

This target is approved after HD review of `docs/surface_runner_data_flow.md`.

Proposed scenarios:

| Scenario | Expected |
|----------|----------|
| Minimal full-run fixture | runner emits non-empty trade log, date summary, and run summary |
| PIT universe snapshot | expected names are eligible from latest `month_date <= trade_date` |
| Long/short routing | long candidate assembles straddle; short candidate assembles approved short structure |
| Included short structure settlement | `pnl_per_share` matches hand calculation |
| Missing/invalid surface row | diagnostics record exclusion/failure reason |
| Missing v1 sizing fields | test may intentionally fail or be marked expected-fail on absent `contracts`, `pnl_dollars`, and realized return-on-max-loss fields |

Constraints:

- No production refactor unless the test proves a narrow bug and HD approves the fix.
- It is acceptable for the new verification test to expose missing desired v1 behavior rather than pass immediately.
- Use venv: `C:/MomentumCVG_env/venv/Scripts/python.exe`
- Run focused test first, then full `tests/ -q`.

---

## Closeout checklist

Complete only when Sprint 001 is actually done.

- [ ] HD reviewed `repo_audit.md`
- [ ] HD reviewed `correctness_audit.md`
- [x] HD reviewed `surface_runner_data_flow.md`
- [ ] Audit docs revised from HD feedback
- [x] Session B test target approved
- [ ] One verification test implemented
- [ ] Relevant focused pytest run
- [ ] Full pytest run: `C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q`
- [ ] `docs/README.md` updated if needed
- [ ] `docs/agenda/week0_review_notes.md` moved to `docs/sprint_memos/`
- [ ] `docs/agenda/current_sprint.md` updated for next sprint
- [ ] This memo marked `Closed`

---

## Tests run

_None yet in Sprint 001._

---

## Files changed so far

Docs only:

- `docs/repo_audit.md`
- `docs/correctness_audit.md`
- `docs/surface_runner_data_flow.md`
- `docs/README.md`
- `docs/v1_spec_pins.md`
- `docs/decisions/002_position_cap_semantics.md`
- `docs/sprint_memos/001_repo_audit_verification.md`
- `docs/agenda/current_sprint.md`

Production code:

- None

Tests:

- None

---

## Next action

Session B may proceed with the approved synthetic `SurfaceRunner.run_single_config()` data-flow verification test. Do not implement broader engine features in Sprint 001 unless HD explicitly approves a narrow fix after the test exposes it.

