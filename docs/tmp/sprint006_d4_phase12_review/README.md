# Sprint 006 D4 — Phase 1 / Phase 2 review bundle

Copied into the repository for third-party review of the Phase 1 gate and Phase 2
smoke. These are **not** the official Sprint 006 baseline and must not be cited as
economic evidence.

**Execution commit:** `5c31e4903a345f496eaca90d81981f3bc6c468e7`  
**Canonical outside-repo originals (still retained):** `C:/MomentumCVG_env/runs/`

## Start here

1. [`checkpoints/sprint006_d4_phase12_checkpoint_PASS_20260823T021859Z.md`](checkpoints/sprint006_d4_phase12_checkpoint_PASS_20260823T021859Z.md) — final Phase 1/2 checkpoint (`PASS`).
2. [`checkpoints/sprint006_d4_phase12_checkpoint_BLOCKED_20260823T020131Z.md`](checkpoints/sprint006_d4_phase12_checkpoint_BLOCKED_20260823T020131Z.md) — first attempt (`BLOCKED` on single-date smoke).
3. [`ARTIFACT_DIGESTS.tsv`](ARTIFACT_DIGESTS.tsv) — SHA-256 of every file in this bundle.

## Layout

| Path | Contents |
|------|----------|
| `smoke_failed_single_date_20260823T020000Z/` | Failed Option A smoke: contract with all four dates = `2022-09-02`, plus marker. No `run/` (preflight abort). |
| `smoke_passed_20260823T021540Z/` | Successful amended smoke: contract `2022-09-02`…`2022-09-09`, marker, and full `run/` adapter outputs (mid + cross). |

## Smoke amendment (blocker B-1)

`BacktestRunConfig` requires `start_date < end_date`. The single-date contract is
invalid. The passed smoke uses median `2022-09-02` through next A1 date
`2022-09-09`. The frozen baseline JSON was **not** changed.

## Blind / review notes

- Phase 3 (official full baseline) was **not** run.
- Smoke `decision_report.*` and `run_summary_*.json` are present for completeness;
  they cover only the two-date smoke window and are **not** official economics.
- Plumbing checks S-1…S-10 are documented in the PASS checkpoint.
