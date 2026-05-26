# Current sprint — 001

**Updated:** 2026-05-23  
**Mode:** Audit → Verification (no production refactors unless a test proves a bug)

---

## Goal

**Gap analysis for an effective v1 backtest on the surface path** — not a “trust the runner as-is” review.

You own this repo and have confirmed: **SurfaceRunner is the right direction**, but it is **not yet a complete backtesting engine**. Many features still need to be built before Tier B (2020 → latest) results are decision-quality.

Sprint 001 delivers:

1. **`docs/repo_audit.md`** — what exists vs what is missing on the surface pipeline (prioritized backlog).
2. **`docs/correctness_audit.md`** — what unit tests already cover; highest financial risks.
3. **Session A.1 mapping review** — `docs/surface_runner_data_flow.md`, a SurfaceRunner functionality and data-flow map before choosing the verification boundary.
4. **One verification test** — iron fly payoff / max-loss truth table on the **assembly/settlement** layer, or a higher-value runner boundary confirmed by Session A.1.

---

## Architecture context (audit must reflect this)

The surface backtest is a **two-stage** system. The audit should trace both stages and where they disconnect.

```text
STAGE A — Precompute (offline, ORATS → parquet)
  scripts/precompute_option_surface.py
    → src/features/option_surface_analyzer.py  (OptionSurfaceBuilder)
    → surface meta parquet  (per ticker, entry_date: spot, exit_spot, body_strike, expiry, surface_valid, …)
    → surface quotes parquet (per strike/side/delta grid)

STAGE B — Backtest day (SurfaceRunner)
  scripts/run_surface_search.py → SurfaceRunner
    → OptionSurfaceDB.load(meta, quotes)     # src/backtest/option_surface.py
    → step1 universe + step2 signals        # pipeline.py
    → build_*_from_surface()                  # assembles legs + entry economics from quotes
    → _select_size_and_settle()               # portfolio cap, sizing, assembly.settle(exit_spot)
    → trade log + summaries
```

**Key point:** `option_surface_analyzer.py` builds the **historical surface** used at backtest time. `option_surface.py` **reads** that surface and **assembles/settles** trades. SurfaceRunner orchestrates the loop but does not replace either module.

The audit should flag gaps in **Stage A** (whether the precomputed surface store is rich enough to support full strategy research/backtesting), **Stage B** (runner/portfolio), and **the contract between them** (metadata keys, exit_spot semantics, failed rows).

---

## Success criteria

- [ ] `docs/repo_audit.md` includes a **Missing for effective backtest** section (prioritized P0/P1/P2), not only “design strengths”
- [ ] `docs/repo_audit.md` maps gaps to files (`precompute_option_surface.py`, `option_surface_analyzer.py`, `option_surface.py`, `surface_runner.py`, `run_config.py`, `run_surface_search.py`)
- [ ] `docs/repo_audit.md` answers Decision 001 falsification checks as **met / partial / missing**
- [ ] `docs/correctness_audit.md` includes test inventory + gap vs surface pipeline
- [x] Session A.1 maps SurfaceRunner functionality/data flow in `docs/surface_runner_data_flow.md` and confirms the Session B verification target
- [ ] One new verification test: approved synthetic `SurfaceRunner.run_single_config()` data-flow test
- [ ] Existing pytest suite remains green; new verification test may intentionally fail or be marked expected-fail if it asserts missing desired v1 fields
- [ ] `docs/sprint_memos/001_repo_audit_verification.md` at sprint close
- [ ] After close: move `docs/agenda/week0_review_notes.md` → `docs/sprint_memos/`

---

## Audit focus — what to look for (starter checklist)

Use this as the backbone of `repo_audit.md`. Expand with file-level notes during Session A.

### Stage A — Precompute store (`precompute_option_surface.py` + `OptionSurfaceBuilder`)

Main audit question:

> Does the precompute system store enough historical option surface information to support full backtesting and strategy refinement without repeatedly returning to raw ORATS parquet?

Treat CLI knobs such as date range, weekly/monthly frequency, and delta buckets as **configuration levers**, not the core audit question. The core question is whether the stored surface schema, coverage diagnostics, and output artifacts are sufficient for the research/backtest loop we need.

| Topic | Question for audit |
|-------|-------------------|
| Sufficiency | Does the stored meta + quote surface contain enough history and fields to refine the strategy across signals, structures, wings, fills, and holding assumptions? |
| Coverage diagnostics | Can we quantify what is missing by ticker/date/expiry/wing/delta bucket before running a full backtest? |
| Research flexibility | Can the same surface support iron fly vs iron condor comparisons and wing-length/delta searches without rerunning ORATS extraction each time? |
| Validity contract | Are `surface_valid`, `failure_reason`, body-leg flags, and quote rows expressive enough for the runner to skip bad candidates cleanly? |
| Exit economics | Does the stored `exit_spot` / expiry information support the current hold-to-expiry baseline, and what would be missing for future early-exit logic? |
| Rebuild cost | If the store is insufficient, is the fix a CLI/config rerun, a schema change, or a new precompute artifact? |

### Stage B — Backtest engine mechanism (`SurfaceRunner` as scaffold)

Main audit question:

> Is the current SurfaceRunner architecture a good mechanism to develop into a comprehensive backtesting engine that can support the v1 strategy through research, shadow, and paper trading?

Do **not** audit SurfaceRunner as if it is already complete. It is known to be incomplete. The useful audit is: what mechanism exists, what responsibilities belong in the runner vs helper modules, what is missing, and what build sequence turns it into a decision-quality engine.

| Topic | Question for audit |
|-------|-------------------|
| Engine boundary | What should SurfaceRunner own vs `option_surface.py`, `pipeline.py`, metrics modules, and future runbooks? |
| Strategy lifecycle | Can the mechanism support universe → signal → structure selection → sizing → settlement → metrics → trade log? What is missing in each stage? |
| Research workflow | Can configs/searches support comparing iron fly vs iron condor, wing rules, fill assumptions, and universe variants? |
| Portfolio/risk layer | What abstractions are needed for 50-name cap, max-loss budget, per-name/sector caps, and integer contracts? |
| Realistic execution model | How should fills, spread cost, skips, untradeable structures, and future close-before-expiry logic plug into the runner? |
| Metrics/output contract | What trade-log/date-summary/run-summary fields are needed for Sharpe, return on max-loss, drawdown, concentration, and ops counts? |
| Shadow/paper bridge | What outputs would let the same engine produce intended orders for weekly shadow/paper trading later? |
| Build sequence | Which missing features are P0/P1/P2 to turn this scaffold into the v1 decision engine? |

### Cross-cutting

| Topic | Question for audit |
|-------|-------------------|
| Legacy path | Straddle/ironfly **history** precompute vs surface path — avoid mixing results |
| Engine V2 | Skeleton — confirm not blocking if SurfaceRunner backlog is clear |
| Integration test | None today — recommend where first smoke test should live |

---

## Context files (read order for Session A)

1. `src/features/option_surface_analyzer.py` — **surface history builder (precompute)**
2. `scripts/precompute_option_surface.py` — CLI and outputs
3. `src/backtest/option_surface.py` — `OptionSurfaceDB`, assembly, `settle`
4. `src/backtest/surface_runner.py` — daily loop, sizing, settlement
5. `src/backtest/run_config.py` — config contract (budget, caps, structures)
6. `scripts/run_surface_search.py` — how runs are launched
7. `src/backtest/pipeline.py` — step1 / step2
8. `src/strategy/builders.py` — reference truth for leg/payoff logic (compare to surface assembly)
9. `src/execution/backtest_executor.py` — legacy settlement (contrast only)
10. `docs/decisions/001_canonical_backtest_path.md` — known gaps seed list

---

## Constraints

- Do not refactor production code unless a test proves a bug
- Do not implement missing runner features this sprint (backlog only)
- Do not run full-sample Tier B backtest this sprint
- Do not change precompute pipeline this sprint (document gaps only)

---

## Agent instructions

### Session A — Audit mode (docs only)

1. Trace Stage A → Stage B using the diagram above.
2. Draft `docs/repo_audit.md` with: Current state, Strengths, **Missing for effective backtest (P0/P1/P2)**, Testing gaps, What not to refactor yet, Recommended sprint 002–004 backlog.
3. Draft `docs/correctness_audit.md` (test inventory + financial risk map).
4. **Stop for your review** — do not write tests until you approve audit + test plan.

### Session A.1 — Mapping review (docs only)

Added from HD review on 2026-05-25:

1. Map SurfaceRunner functionality and data flow from inputs through trade log / summaries.
2. Identify required responsibilities as **implemented / partial / missing**.
3. Confirm whether `_select_size_and_settle()` is the right Session B test boundary, or replace it with a higher-risk boundary.
4. Do not edit production code or write tests during this mapping review.

Reviewed artifact:

- `docs/surface_runner_data_flow.md`

### Session B — Verification mode (after Session A.1 approval)

1. Add one approved synthetic `SurfaceRunner.run_single_config()` data-flow test.
2. It is acceptable for this test to intentionally fail or be marked expected-fail on missing desired v1 fields (`contracts`, `pnl_dollars`, realized return-on-max-loss).
3. Run focused verification first, then run the existing suite as appropriate.
4. Minimal prod fix only if test exposes a narrow bug and HD approves the fix.

### Sprint close

1. `docs/sprint_memos/001_repo_audit_verification.md`
2. Update `docs/README.md` if new docs added
3. Move `week0_review_notes.md` → `docs/sprint_memos/`

---

## Expected outcomes and next steps

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **A — Clear backlog, test passes** | Gaps documented; assembly layer trusted | Sprint 002: pick **P0** items (e.g. sizing + weekly dates + 50-cap); liquidity panel review |
| **B — Test fails (real bug)** | Surface assembly/settle or builders wrong | Fix in Sprint 001 extension; re-audit affected layer |
| **C — P0 gaps dominate** | Runner cannot produce decision-quality runs yet | Sprint 002–003 = **build** on SurfaceRunner; defer fly vs condor matrix |
| **D — Precompute gaps dominate** | Surface store is not rich enough for strategy refinement/backtesting | Sprint 002 = decide whether fix is CLI rerun, schema extension, or new precompute artifact |

---

## Out of scope (Sprint 001)

- Implementing portfolio caps, contract sizing, weekly rebalance in code
- Full surface smoke backtest
- Iron fly vs iron condor comparison matrix
- `v1_weekly_runbook.md` (Sprint 002)
- Broker / shadow mode
