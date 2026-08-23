# Sprint 006 D4 — Phase 1 / Phase 2 checkpoint (smoke re-run)

**Document path:** `C:/MomentumCVG_env/runs/sprint006_d4_phase12_20260823T021859Z/sprint006_d4_phase12_checkpoint.md`
**Written (UTC):** 2026-08-23T02:18:59Z
**Location:** outside the Git repository
**Checkpoint verdict:** `PASS`
**Next action:** `AWAITING HUMAN REVIEW — PHASE 3 NOT AUTHORIZED`

---

## 1. Scope and blind-inspection attestation

### Scope

- Phase 1 was already completed and recorded as `PASS` in
  `C:/MomentumCVG_env/runs/sprint006_d4_phase12_20260823T020131Z/sprint006_d4_phase12_checkpoint.md`
  (SHA-256 `f1ec372d6e80f3cdaceed24130a99ac3abbd2cbbcaa662dadf8d00c8d31ca937`). That
  Phase 1 evidence is reused here; it was not re-executed.
- This document records the **Phase 2 smoke re-run** after correcting the smoke
  contract window that caused blocker B-1.
- Phase 3 (official full baseline) was **not** started.

### Config fix applied (smoke only)

The prior single-date smoke contract set all four date fields to `2022-09-02`,
which fails `BacktestRunConfig.validate()` (`start_date` must be **strictly**
before `end_date`).

**Fix (outside-repository smoke contract only):** keep
`run_start_date` / `start_date` = median `2022-09-02`, and set
`run_end_date` / `end_date` = next A1 expected date `2022-09-09`. Exactly the same
four fields still differ from the frozen contract; `primary_reporting_*` is
unchanged. The frozen file `configs/sprint006_baseline_v1.json` was **not**
modified.

This is the Option A amendment proposed under blocker B-1 (minimal two-date
window). No production code was changed.

### Blind-inspection attestation

- No `decision_report.json` / `.md` **values** were opened, printed, or summarized.
- No `run_summary_*.json` economic fields were opened, printed, or summarized.
- No aggregate return, Sharpe, drawdown, attribution, concentration, or trade-level
  P&L was interpreted as a performance result. S-7 uses per-trade identity
  arithmetic only (entry / payoff / pnl reconciliation), which is a plumbing check.
- Repository remained read-only: no repo file changes, no commit, no push.

---

## 2. Execution identity

| Item | Value |
|------|-------|
| `EXECUTION_COMMIT` | `5c31e4903a345f496eaca90d81981f3bc6c468e7` |
| Branch | `main` (clean) |
| Ancestry `10133f6` | confirmed ancestor (from Phase 1) |
| Frozen contract on-disk SHA-256 | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` (unchanged) |

---

## 3. Prior Phase 1 summary (not re-run)

| Item | Result |
|------|--------|
| Full suite | 1597 passed, 1 skipped, exit 0 |
| Focused suite | 332 passed, exit 0 |
| Dry run | exit 0; wrote nothing |
| Contract / inputs / digests | all PASS |
| Phase 1 verdict | `PASS` |

See the prior checkpoint for full tables.

---

## 4. Median / smoke window derivation

| Item | Value |
|------|-------|
| A1 expected calendar count | 403 |
| Lower median | `2022-09-02` (index 201) |
| Next A1 date after median | `2022-09-09` |
| Smoke window | `2022-09-02` … `2022-09-09` (inclusive; two A1 dates) |

---

## 5. Smoke directories and contract

| Item | Value |
|------|-------|
| Prior failed smoke (retained) | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T020000Z` |
| New smoke dir | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T021540Z` |
| Adapter output | `…/sprint006_d4_smoke_20260823T021540Z/run` |
| Smoke contract | `…/sprint006_smoke_contract.json` |
| Smoke contract SHA-256 | `2f7110bbff57c680830a949f082b1b4d46458d363f670f073990af7edcf97801` |
| Four-line diff vs frozen | only `run_start_date`, `run_end_date`, `start_date`, `end_date` |
| Diff values | start fields → `2022-09-02`; end fields → `2022-09-09` |
| Frozen file / git status | unchanged / empty |
| Marker | `NOT_THE_OFFICIAL_BASELINE.txt` written after S-2 |

### Smoke execution

```
Command: scripts/run_sprint006_baseline.py
  --contract …/sprint006_smoke_contract.json
  --output-dir …/run

SMOKE_START_UTC = 2026-08-23T02:15:51.9867212Z
SMOKE_END_UTC   = 2026-08-23T02:16:57.7307196Z
Wall clock      = 65.7 s
Exit code       = 0
```

Stdout listed both fills with 7 per-run files each, plus `decision_report.json`,
`decision_report.md`, and `run_receipt.json`. No `ERROR:` line.

Receipt identity only (no economic fields):

```
experiment_id=sprint006_baseline_v1
deliverable=sprint006_d3
repo_sha=5c31e4903a345f496eaca90d81981f3bc6c468e7
contract_id=sprint006_baseline_v1
contract_sha256=2f7110bbff57c680830a949f082b1b4d46458d363f670f073990af7edcf97801
n_runs=2
result_complete=true
```

(`deliverable=sprint006_d3` is expected D3 producer metadata.)

---

## 6. S-1 through S-10

| # | Check | Observed | Verdict |
|---|-------|----------|---------|
| S-1 | Execution | exit 0; both fills; 17 paths in stdout; no `ERROR:` | `PASS` |
| S-2 | Artifact set | exactly 17 adapter files in `$SMOKE_OUT`; no extras (count before marker) | `PASS` |
| S-3 | Schemas | `date_status` 3 cols; funnel 18; leg 21; candidate 9 — both fills | `PASS` |
| S-4 | Date status | 2 rows/fill: `2022-09-02`, `2022-09-09`; both `traded`; reasons null | `PASS` |
| S-5 | Funnel semantics | monotone stage counts; side splits sum; null/zero respected | `PASS` |
| S-6 | Leg serialization | median `structure_ok=31` each fill; included legs present with required indices/signs | `PASS` |
| S-7 | Included-trade reconciliation | 31 included trades/fill on median; four Σ identities hold within tol | `PASS` |
| S-8 | Structure failures | 13 `structure_failed` rows/fill; codes in frozen set; `reason_raw` retained; no leg rows | `PASS` |
| S-9 | Fill half-spread (conditional) | 31 overlapping constructable names on median; sample `ACN/long`; `cross−mid = ±0.5×(ask−bid)` within tol; mid columns identical | `PASS` |
| S-10 | No aggregate inspection | attested — report/summary economics not opened | `PASS` |

**Note on S-4:** with the two-date amendment, exactly **two** `date_status` rows per
fill are expected (not one). Plumbing focus remains the median date; the next A1
date is present only to satisfy `start_date < end_date`.

---

## 7. Deviations / blockers

| ID | Status |
|----|--------|
| B-1 (single-date window invalid) | **Resolved for this re-run** by amending the smoke contract to `2022-09-02`…`2022-09-09`. Failed single-date attempt retained. |
| D-1 (`runs` parent missing at Phase 1) | Unchanged benign note from prior checkpoint. |
| Plan text still describes single-date Option A | **Open documentation debt.** The in-repo D4 plan was not edited in this re-run. Recommend a docs-only plan amendment before Phase 3 so operators do not recreate the single-date failure. |

No production code was changed. No frozen-contract change. No Phase 3.

---

## 8. Checkpoint verdict

**`PASS`**

- Phase 1: `PASS` (prior checkpoint).
- Phase 2 smoke re-run: `PASS` (S-1…S-10).

Smoke artifacts remain non-official and must not be cited as Sprint 006 economic
evidence.

---

## 9. Next action

**`AWAITING HUMAN REVIEW — PHASE 3 NOT AUTHORIZED`**

Recommended before Phase 3:

1. Accept this two-date smoke amendment as the operative PG-1b Option A procedure.
2. Optionally amend `docs/tmp/sprint006_d4_execution_acceptance_plan.md` so the
   written plan matches what was executed.
3. Explicitly authorize Phase 3 (official full baseline).
