# AGENTS.md

Instructions for AI agents working in **MomentumCVG** (options strategy research, backtesting, path to v1 live trading).

## Project goal

Build a **trusted v1 pipeline**: dynamic liquid universe → momentum/CVG signal → short iron fly or iron condor (per run) + long straddle → max-loss sizing → conservative backtest → shadow → paper trading.

Active spec: [docs/v1_spec_pins.md](docs/v1_spec_pins.md).

## Session start

1. Read [docs/agenda/current_sprint.md](docs/agenda/current_sprint.md) for this week's goal and **mode**.
2. Skim [docs/v1_spec_pins.md](docs/v1_spec_pins.md) before strategy or backtest changes.
3. When editing builders or `option_surface.py`, check [docs/known_bugs.md](docs/known_bugs.md).
4. Do not treat [docs/archive/](docs/archive/) as current truth.

## Workflow rules

1. **Inspect before editing.** For audits, do not change code until the user approves a plan.
2. **Plan before coding.** Propose files, tests, and risks first.
3. **Three modes** (set in sprint agenda):
   - **Audit** — docs and analysis only
   - **Verification** — tests only; no production changes unless a test proves a bug
   - **Build** — implementation scoped to the approved plan
4. Add or update tests for behavior changes in financial logic.
5. Run focused tests after changes; report command and result.
6. Summarize: files changed, tests run, remaining risks.
7. Record outcomes in [docs/sprint_memos/](docs/sprint_memos/) when a sprint completes.

## Safety rules

- Do not change strategy logic without tests.
- Do not treat mid-price-only backtests as final evidence for go/no-go.
- Verify option leg type, strike, expiry, quantity sign, premium sign, payoff, and max loss when touching builders or execution.
- Do not mix large refactors with research changes unless explicitly requested.
- Do not add broker/live execution code unless the task explicitly says so.
- Canonical backtest path is **SurfaceRunner** unless a decision memo says otherwise.

## Environment

Python venv (Windows):

```powershell
& C:/MomentumCVG_env/venv/Scripts/Activate.ps1
```

External data (not in repo):

- Raw ORATS: `C:/ORATS/data/ORATS_Data`
- Production adjusted chains (C5): `C:/MomentumCVG_env/input/adjusted_liquid`
- Legacy full-universe mirror: `C:/ORATS/data/ORATS_Adjusted` (maintenance only)
- Cache: `C:/MomentumCVG_env/cache/`

## Definition of done

A task is done when:

- Requested behavior is implemented (or doc delivered for audit tasks)
- Relevant tests pass
- Diff is reviewed
- Remaining risks are documented

## Key docs

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Active doc index |
| [docs/repo_map.md](docs/repo_map.md) | Repository layout and data flow |
| [docs/sprint_memos/004_c5_adjusted_liquid.md](docs/sprint_memos/004_c5_adjusted_liquid.md) | C5 adjusted-liquid closeout |
| [docs/sprint_memos/004_c8_4_bounded_evidence.md](docs/sprint_memos/004_c8_4_bounded_evidence.md) | C8.4 bounded snapshot evidence closeout |
| [docs/backtest_evaluation_protocol.md](docs/backtest_evaluation_protocol.md) | Go/no-go windows and fills |
| [docs/v1_universe_protocol.md](docs/v1_universe_protocol.md) | PIT liquidity universe |
| [docs/v1_ops_model.md](docs/v1_ops_model.md) | Trade volume / broker thresholds |
