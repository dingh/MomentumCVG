# Session B plan — Sprint 001 (Verification)

**Sprint:** 001  
**Mode:** Verification (no production refactors unless a test proves a bug)  
**Status:** Complete (Sprint 001 closed 2026-05-28)  
**Last updated:** 2026-05-27

---

## Why Session B exists

Session A established that the surface assembly math is relatively mature, but the engine layer around it (runner/portfolio/metrics/date-alignment/contracts) is partial. Session B’s job is to add **one** verification test that exercises the **actual canonical path** end-to-end using synthetic artifacts, so we can:

- Verify the current Stage A → Stage B data contract is workable for a minimal run.
- Catch schema/date/alignment issues that private-helper tests cannot.
- Explicitly surface missing v1 requirements (contracts, dollar PnL, return-on-max-loss) without “papering over” gaps.

This Session B scope is approved by HD in `docs/surface_runner_data_flow.md` and tracked in `docs/sprint_memos/001_repo_audit_verification.md`.

---

## Target under test

**Primary test boundary:** `SurfaceRunner.run_single_config()`  

Rationale:

- It’s the canonical path used by `scripts/run_surface_search.py`.
- It forces a realistic interface: parquet artifacts + config in → trade log + summaries out.
- It still exercises selection/settlement (including `_select_size_and_settle()`), but also covers upstream assumptions (trade dates, PIT universe, feature-date joins, surface lookups).

---

## Planned test artifact

**New test file (proposed):**

```text
tests/unit/test_surface_runner_data_flow.py
```

### Fixture strategy

Use temp directory fixtures to write **tiny parquet** inputs that the runner reads exactly as it would in a real run:

- **Liquidity panel parquet**: one `month_date` snapshot with a handful of tickers and enough columns to drive `pipeline.step1_get_universe()`.
- **Features parquet**: one `trade_date` row per ticker with the momentum/CVG columns used by `pipeline.step2_score_signals()`.
- **Surface metadata parquet**: `(ticker, entry_date)` rows for:
  - one long candidate (valid surface)
  - one short candidate (valid surface)
  - one invalid/missing candidate (to validate diagnostics / skip path)
- **Surface quotes parquet**: quote rows sufficient to assemble:
  - a long straddle for the long candidate
  - a short iron fly (or condor) for the short candidate

Keep the data intentionally small and hand-auditable.

### Scenarios and assertions

**Scenario 1 — minimal full-run data flow**

- Run one `BacktestRunConfig` through `SurfaceRunner.run_single_config()`.
- Assert it produces:
  - a non-empty trade log (or an expected number of rows given the synthetic universe)
  - a date summary
  - a run summary

**Scenario 2 — PIT universe lookup is point-in-time**

- Ensure the universe comes from the latest `month_date <= trade_date`.
- Assert expected tickers are eligible given the synthetic liquidity snapshot.

**Scenario 3 — long/short routing + structure assembly**

- Assert that one ticker is routed to the long side and assembles a **long straddle**.
- Assert that one ticker is routed to the short side and assembles the configured **short structure** (iron fly or condor).

**Scenario 4 — settlement semantics (hand-calculated)**

- For the included short structure, assert `pnl_per_share` equals a hand calculation based on the synthetic `exit_spot` and the assembled payoff.

**Scenario 5 — invalid surface row produces diagnostics**

- Include at least one ticker/date with invalid or missing surface metadata/quotes.
- Assert it is excluded with a stable, inspectable reason (or at minimum is not silently included).

### Explicitly allowed outcomes (by design)

It is acceptable (and potentially desirable) for the test to be:

- **Expected-fail** or intentionally failing on missing v1 fields such as:
  - `contracts` / integer contract sizing
  - `pnl_dollars`
  - realized `return_on_max_loss` (on max-loss budget units)

If those fields do not exist yet, the test should **document that gap**, not pretend it’s implemented.

---

## Constraints (must obey in Sprint 001)

- No production code changes unless this test exposes a narrow bug and HD explicitly approves the fix.
- No broad engine work (caps, sizing, metrics overhaul) in Sprint 001.
- Do not run Tier B backtests or require ORATS data; all inputs are synthetic.
- Use the repo venv when running tests: `C:/MomentumCVG_env/venv/Scripts/python.exe`

---

## Execution plan for Session B

1. Implement the synthetic parquet fixtures and the `run_single_config()` test.
2. Run the single new test first (fast feedback).
3. Run full unit suite (`tests/ -q`) to confirm no regressions.
4. Record results in `docs/sprint_memos/001_repo_audit_verification.md` under “Tests run”.

---

## Risks to watch (what the test is expected to reveal)

- Trade-date alignment mismatches (features vs surface entry dates vs liquidity month snapshots).
- Missing required columns in artifacts (runner currently has minimal schema validation).
- Runner behavior that depends on implicit defaults (fills, scoring columns, exclusions).
- Engine gaps that prevent decision-quality backtests (contracts/dollar PnL/return-on-max-loss/cap semantics).

