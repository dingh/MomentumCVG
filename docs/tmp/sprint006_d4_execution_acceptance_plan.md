# Sprint 006 D4 — Execution and acceptance plan

**Status:** `PROPOSED — AWAITING ACCEPTANCE`
**Mode:** Build. This document authorizes **nothing**. No D4 execution has started. Planning commit is documentation-only.
**Repo HEAD at proposal:** `10133f6c12facae26d818b7e112b94332f5e1e46` (`test(sprint006): verify D3 markdown values`), clean working tree on `main`
**Accepted D3 implementation:** `361b333` → `bb40864` + `f009684` → `6c7e44f` → `eaa8421` → `10133f6` (design `b924330`)
**Confirmed ancestors:** D0 `1cdfad7`; D1 `241b0d3` + `c6b1735`; D2 `9224068` (acceptance `62bdf38`); D3 design `b924330`
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (unchanged)
**Plans:** [D0](sprint006_d0_baseline_experiment_contract_plan.md) · [D1](sprint006_d1_trusted_baseline_runner_plan.md) · [D2](sprint006_d2_eligibility_coverage_correctness_plan.md) · [D3](sprint006_d3_decision_diagnostic_report_plan.md)
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md`
**Planning decisions:** all previously open items (PG-1a, PG-1b, PG-2, PG-3, D4-Q1–Q3) are resolved in §14. No decision remains open; the plan itself still requires acceptance.

---

## Plain-language summary

D0–D3 built a frozen experiment contract, a thin adapter that runs it reproducibly, correct date/eligibility accounting, and a deterministic decision-report generator. Every one of those deliverables was validated on **synthetic** fixtures only. Nobody has yet run the frozen `42:8` baseline on the accepted real dataset, and nobody has looked at real Sprint 006 P&L.

D4 does exactly that, once, and then decides whether the resulting evidence can be trusted.

The work is sequenced so that trust is established **before** interpretation. Phase 1 proves the machine and its inputs are in the expected state. Phase 2 runs a small real-data smoke to catch integration failures cheaply. Phase 3 executes the one official twin-fill run over the full frozen history. Phase 4 verifies that run's evidence **blind** — completeness, identities, digests, schemas, and a hand reconstruction of a small frozen trade sample from source data — with the aggregate performance blocks still unopened. Only after that gate passes does Phase 5 open the economics, characterize them descriptively, and close the sprint with one recommendation.

### The central D4 question

> Is the frozen `42:8` Momentum+CVG baseline result, produced on the accepted real dataset under conservative cross fills, **technically trustworthy** — and if so, what does it descriptively say about the economic hypothesis?

Note the ordering. "Is the result good?" is the second question, and it is only allowed to be asked after the first is answered `ACCEPTED`.

### Success criteria

D4 succeeds when all of the following exist:

1. One official twin-fill (mid + cross) run over `2018-10-26`…`2026-07-10` executed from a clean commit, with recorded command, identities, digests, and outputs, in a fresh outside-repository directory.
2. A completed blind technical verification producing an **evidence verdict** of `ACCEPTED` or `BLOCKED`, with every check's expected value, observed value, source, and verdict recorded.
3. A manual audit of the frozen D0 S1–S4 sample (≤ 6 hand-checked trades) reconstructed **from source inputs**, not from the D3 trade log.
4. An **economic characterization** (`PROMISING` / `WEAK/NEGATIVE` / `INCONCLUSIVE`) that is descriptive, cites the frozen report, and invents no new threshold.
5. One closeout recommendation: proceed to bounded Sprint 007 robustness work, investigate a named correctness/data defect, or reject/defer the economic hypothesis.
6. Zero production-code changes inside the execution/evidence commit, and zero changes to the frozen contract.

### Why a weak or negative economic result is still a successful D4

Sprint 006's deliverable is a **trustworthy answer**, not a profitable one. The sprint agenda states this explicitly (§1, §7: "Success is evidence quality, not a required Sharpe or positive-return threshold"), and D0 §6 forbids retuning after P&L exposure.

A complete, correct, reproducible run showing that the frozen baseline is unattractive after cross fills is a genuine result: it retires a hypothesis with evidence instead of guesswork, and it does so *before* any capital, broker integration, or Sprint 007 robustness budget is spent. The failure mode D4 must avoid is not a bad number — it is an **uninterpretable** number: silent date loss, unreconciled trades, mismatched inputs, or a result quietly "improved" by editing the experiment after seeing it.

---

## 1. Context-read receipt

| Path | Why read | Fact used for D4 |
|------|----------|------------------|
| `docs/agenda/current_sprint.md` | Authorization / DoD | D4 owns smoke, manual sample, full execution, closeout; sprint must record one recommendation |
| `configs/sprint006_baseline_v1.json` | Frozen contract | All identifiers, windows, fills, sizing, thresholds copied into §2/§3 |
| D0 plan §8, §8.1, §9, §10 | Frozen rules | A1 expected-calendar authority; View A/B; S1–S4 sample rules and fallbacks; required report blocks |
| D1 plan §5, §12 | Command + identity | One CLI, always both fills; outside-repo run dir; contract digest is *recorded* (on-disk), not compared to the LF value |
| D2 plan / `surface_runner.py` | Calendar + status | `_get_expected_dates_from_a1`; `date_status` partition assert; `missing_features` → `failed` |
| D3 plan §4.2–§4.6 | Report + artifacts | Output file names, JSON keys, abort semantics, leg/funnel schemas, receipt fields |
| `src/backtest/sprint006_baseline.py` | Adapter behavior | `--dry-run` writes nothing; overwrite refusal; run dir refused inside repo/mutable cache; receipt contents |
| `scripts/run_sprint006_baseline.py` | CLI surface | Only `--contract`, `--output-dir`, `--dry-run` exist. **No** date, fill, ticker, or limit flags |
| `src/backtest/surface_decision_report.py` | Report internals | `REPORT_WINDOWS`, `assert_report_preconditions`, `assert_included_trade_legs`, `report_jsonable` |
| `src/backtest/surface_metrics.py` | Aggregate exposure | `run_summary_*.json` embeds `annualized_sharpe`, `max_drawdown`, `robust_score` → must stay closed during Phase 4 |
| `src/backtest/pipeline.py` | Reconstruction spec | S1 strict-prior snapshot + AND ranks; `eligible_feature_cross_section`; S2 tails/CVG; S3 routing; S5 cap/sizing/simulate |
| `src/backtest/option_surface.py` | Leg economics | `buy_price`/`sell_price` alphas; `is_body`/`is_otm` + `spread_pct` gates; `_choose_below_nearest`; iron-fly leg order and `max_loss = wing_width − net_credit` |
| `docs/sprint_memos/sprint005_d5_surface_runner_smoke_evidence.md` | Evidence convention | Evidence-memo shape (identities / paths / command / exit / counts / tests / confirmations / residual limitation); D5 median date was `2022-09-02` |
| `docs/sprint_memos/005_closeout.md`, `004_c8_4_bounded_evidence.md` | Naming | `docs/sprint_memos/sprint00N_dN_*_evidence.md` and `docs/sprint_memos/00N_closeout.md` |
| `docs/baseline_status.md` | Test baseline | Last recorded suite: 1494 passed / 1 skipped at Sprint 005; D1 recorded 1528 passed / 1 skipped |

**Git state at proposal:** HEAD `10133f6`, branch `main`, working tree clean. No code executed against real data for this document.

---

## 2. Immutable identifiers (copied from the repository — do not retype from memory)

### 2.1 Code and configuration

| Identity | Exact value | Source |
|----------|-------------|--------|
| Planning base commit | `10133f6c12facae26d818b7e112b94332f5e1e46` | `git rev-parse HEAD` |
| Execution commit | **recorded at Phase 1**, expected to equal the planning base or the D3-acceptance docs commit | Phase 1 step 1.1 |
| Contract path | `configs/sprint006_baseline_v1.json` | contract |
| Contract committed blob id (git, LF) | `805faa5cdb94618538c60d5afdd715fec84ac608` | `git rev-parse HEAD:configs/sprint006_baseline_v1.json` |
| Contract SHA-256, committed LF bytes (11920 B) | `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715` | D0 header; verified by LF normalization |
| Contract SHA-256, on-disk CRLF bytes (12160 B) | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` | `Get-FileHash` in this CRLF working copy; this is the value the receipt records |
| `contract_id` / `contract_version` / `status` | `sprint006_baseline_v1` / `1` / `accepted` | contract |
| Experiment id | `sprint006_baseline_v1` | `EXPERIMENT_ID` |
| Run ids | `sprint006_baseline_v1_mid` (diagnostic), `sprint006_baseline_v1_cross` (**primary**) | contract `runs[]` |
| Feature config | `configs/feature_backfill_v1.json`, SHA-256 `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` | contract |

> **Line-ending note (not a defect).** D1 deliberately records the on-disk digest rather than normalizing. On this CRLF working copy the receipt will contain `4012b4a4…`, **not** `3cd57f4d…`. Both values above are correct and must both be verified in Phase 1 by their own procedure. A receipt containing `3cd57f4d…` would indicate an LF checkout, which is also acceptable — a receipt containing neither is a **stop condition**.

### 2.2 Accepted inputs (outside the repository; read-only)

| Identity | Exact value |
|----------|-------------|
| Snapshot id | `e2c1f8fd44d72176` |
| Build id | `20260724T045049097520Z_40b16886` |
| Snapshot root | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886` |
| Manifest | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/manifests/input_snapshot_e2c1f8fd44d72176.json` |
| Derived root | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/` |
| Features dir | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/` |
| Baseline feature file | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/features_42_8.parquet` |
| Feature lineage receipt | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json` |
| Lineage receipt SHA-256 | `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` |
| Lineage producer `repo_sha` | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Feature quality audit JSON | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json` |
| A1 surface meta (**expected-calendar authority**) | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_meta_weekly_2018_2026.parquet` |
| A2 surface quotes | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` |
| Liquidity panel | `C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/input/liquidity/ticker_liquidity_panel.parquet` |
| Earnings | `null` (filtering off) |
| Mutable producer cache `C:/MomentumCVG_env/cache` | **Forbidden** as input root and as run output root |

### 2.3 Frozen economic parameters that Phase 1 must confirm verbatim

| Item | Frozen value |
|------|--------------|
| Window | `max_lag=42`, `min_lag=8`, `window_size=35`, `search=false` |
| Feature columns | `mom_42_8_mean` / `cvg_42_8` / `mom_42_8_count` / `cvg_count_42_8` |
| Count eligibility | `min_count_pct=0.8`; `required_count = ceil(0.8 × 35) = 28`, applied **jointly** to both count columns |
| Selection | `long_top_pct=0.1`, `short_bottom_pct=0.1`, `cvg_filter_pct=0.5`, `max_names_per_side=25`, tie-break `ticker` ascending |
| Universe | `dvol_top_pct=0.2`, `spread_bottom_pct=1.0`, `earnings_exclusion_days=0` |
| Structures | long ATM straddle; `short_structure=ironfly`, `wing_delta_target=0.15` via `_choose_below_nearest`; `max_leg_spread_pct=0.5` on every traded leg; `max_spread_cost_ratio=null` |
| Sizing | `sizing_mode=conceptual`, `tier_a_mode=equal_max_loss`, `tier_a_short_budget=10000.0`, `tier_a_long_budget=10000.0` (**fallback only**), `contract_multiplier=100.0`, `tier_b_short_max_loss_budget=null`, `deployable_capital=null` |
| Cost assumptions | Pricing is **solely** `fill`: mid `(0.5, 0.5)`; cross `(1.0, 1.0)`. `cost_model="mid"` is inactive legacy on both runs and adds no deduction. Settlement is intrinsic at A1 `exit_spot`, no exit spread, no commissions |
| Legacy pins | `max_loss_budget_per_trade=500.0` (does not control Tier A), `include_diagnostics=true`, `wing_selection_rule="closest_delta"` (label only) |
| Run window | `2018-10-26` … `2026-07-10` inclusive |
| Primary reporting window | `2020-01-01` … `2026-07-10` inclusive |
| Output root | A **new** directory under `C:/MomentumCVG_env/runs/`, outside the repository and outside `C:/MomentumCVG_env/cache` |

---

## 3. Environment and global rules

**Interpreter (all phases):** `C:/MomentumCVG_env/venv/Scripts/python.exe`
**Working directory (all phases):** `C:/MomentumCVG`
**Shell:** PowerShell. Use `;` between commands — `&&` is not a valid separator here.

Global rules for every phase:

* **No production-code change.** D4 expects zero. See §9.
* **No edit to `configs/sprint006_baseline_v1.json`**, ever, for any reason.
* **No retry into an existing run directory.** The adapter refuses this; do not work around it.
* **Never delete or overwrite a failed run's artifacts.** Preserve, then start a new directory.
* **Aggregate P&L stays closed until Phase 5.** Files and fields that must remain unopened during Phases 1–4 are listed in §7.1.
* Large run artifacts and source data stay **outside** the repository. Only memos and tables enter Git.

---

## 4. Phase 1 — Pre-execution gate

**Purpose.** Prove, before any real-data economics runs, that the code, tests, configuration, inputs, and output location are all in the exact expected state — so that any later anomaly is attributable to the run rather than to the setup.

**May be inspected:** repository state, test output, contract JSON, dry-run stdout, file existence and digests, manifest/lineage JSON identity fields, directory listings.
**May not be inspected:** any economic result, any prior run's `decision_report.*` or `run_summary_*.json`, any feature or quote values beyond identity/coverage.

### 1.1 Clean Git state and execution commit

```powershell
cd C:\MomentumCVG
git status --porcelain      # must print nothing
git branch --show-current   # expect: main
git rev-parse HEAD          # RECORD as EXECUTION_COMMIT
git log --oneline -3
```

**Resolved (D4-Q1):** execution runs from the **final accepted plan commit** — the HEAD that carries the accepted D4 plan — not from `10133f6` itself. Documentation-only commits do not change behavior, and `10133f6` remains an ancestor. That HEAD is `EXECUTION_COMMIT` and must equal the receipt's `repo_sha` (V-11).

**Pass:** empty `git status --porcelain`; HEAD is a 40-hex commit; HEAD is the accepted-plan commit and contains the accepted D3 implementation (`10133f6` is an ancestor: `git merge-base --is-ancestor 10133f6 HEAD` exits 0).
**Stop:** any untracked or modified file; detached HEAD; `10133f6` not an ancestor; HEAD is not the accepted-plan commit.
**Retain:** `EXECUTION_COMMIT` (full 40-hex), branch, `git log --oneline -3`.

> The adapter independently re-checks a clean tree via `clean_repo_sha()` and refuses to write artifacts from a dirty tree. Phase 1 records the value; the adapter enforces it.

### 1.2 Full regression tests

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q
```

Then the focused Sprint 006 subset (all files verified to exist at `10133f6`):

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q `
  tests/unit/test_sprint006_baseline_adapter.py `
  tests/unit/test_sprint006_leg_log.py `
  tests/unit/test_surface_decision_report.py `
  tests/unit/test_surface_runner_data_flow.py `
  tests/unit/test_option_surface_straddle.py `
  tests/unit/test_option_surface_ironfly.py `
  tests/contract/test_orchestration_contract.py `
  tests/contract/test_step1_universe_contract.py `
  tests/contract/test_step2_signals_contract.py `
  tests/contract/test_step3_structures_contract.py `
  tests/contract/test_step4_exclusions_contract.py `
  tests/contract/test_step5_select_and_size_contract.py `
  tests/contract/test_settle_contract.py `
  tests/contract/test_run_metrics_contract.py `
  tests/contract/test_run_envelope_contract.py
```

**Pass:** full suite exits 0 with **zero failures and zero errors**; skips are tolerated only if they match the historical single pre-existing skip; focused subset exits 0.
**Stop:** any failure or error, any new skip, any collection error.
**Retain:** both exact commands, pass/fail/skip counts, durations, exit codes. Update `docs/baseline_status.md` only in the Phase 5 closeout commit, not now.

### 1.3 Frozen-contract dry run

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract configs/sprint006_baseline_v1.json `
  --output-dir C:/MomentumCVG_env/runs/sprint006_d4_dryrun_placeholder `
  --dry-run
```

`--output-dir` is required by the parser but the dry-run branch never calls `create_run_dir`; nothing is created or written.

**Pass:** exit 0, and stdout shows exactly:

* `contract identity: sprint006_baseline_v1 v1 (accepted)`
* `contract sha256:` equal to `4012b4a4…` (CRLF checkout) or `3cd57f4d…` (LF checkout)
* `run: sprint006_baseline_v1_mid fill=mid dates=2018-10-26..2026-07-10`
* `run: sprint006_baseline_v1_cross fill=cross dates=2018-10-26..2026-07-10`
* the four resolved accepted paths, character-for-character equal to §2.2
* `dry run: no economic execution performed`

Then confirm nothing was created:

```powershell
Test-Path C:/MomentumCVG_env/runs/sprint006_d4_dryrun_placeholder   # must be False
```

**Stop:** non-zero exit; any `ERROR:` line; a resolved path differing from §2.2; only one run printed; the placeholder directory existing afterward.
**Retain:** full stdout.

### 1.4 Frozen configuration and checksum verification

```powershell
cd C:\MomentumCVG
git rev-parse HEAD:configs/sprint006_baseline_v1.json    # expect 805faa5cdb94618538c60d5afdd715fec84ac608
git cat-file -s HEAD:configs/sprint006_baseline_v1.json  # expect 11920
git diff --quiet HEAD -- configs/sprint006_baseline_v1.json; $LASTEXITCODE   # expect 0
(Get-FileHash -Algorithm SHA256 configs/sprint006_baseline_v1.json).Hash.ToLower()
```

Committed-LF SHA-256 (must equal `3cd57f4d…`):

```powershell
$raw = [System.IO.File]::ReadAllBytes("C:/MomentumCVG/configs/sprint006_baseline_v1.json")
$text = [System.Text.Encoding]::UTF8.GetString($raw).Replace("`r`n", "`n")
$lf = [System.Text.Encoding]::UTF8.GetBytes($text)
$sha = [System.Security.Cryptography.SHA256]::Create()
(($sha.ComputeHash($lf) | ForEach-Object { $_.ToString("x2") }) -join "")
$lf.Length   # expect 11920
```

Then read the contract and tick every row of §2.3 against it. Confirm in particular: `feature_window.search=false`; `count_eligibility.derived_required_count=28` with both `joint_columns`; `periods` = `2018-10-26`/`2026-07-10` and `2020-01-01`/`2026-07-10`; exactly two `runs[]` with fill alphas `(0.5,0.5)` and `(1.0,1.0)` and exactly one `primary_decision_view: true` (cross); `accepted_inputs.earnings_path` is `null`; `accepted_inputs.mutable_cache_forbidden` is `true`.

**Pass:** every identifier and parameter matches §2.2/§2.3 exactly.
**Stop:** any mismatch, including a single digit of any digest. Do not "fix" the contract.
**Retain:** the four command outputs, the LF digest, and a completed §2.3 checklist.

### 1.5 Accepted input snapshot / build identity

```powershell
Get-Content "C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/manifests/input_snapshot_e2c1f8fd44d72176.json" | ConvertFrom-Json | Select-Object snapshot_id, build_id
Get-Content "C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json" | ConvertFrom-Json | Select-Object status, snapshot_id, build_id, repo_sha
```

**Pass:** manifest `snapshot_id=e2c1f8fd44d72176` and `build_id=20260724T045049097520Z_40b16886`; lineage `status=complete`, same snapshot/build, `repo_sha=131d0ac05e1e57749d3095923927a394fdcbc25b`.
**Stop:** any mismatch, or `status != complete`.
**Retain:** both JSON extracts verbatim.

### 1.6 Feature artifact and receipt digests

```powershell
$inputs = @(
  "C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/features_42_8.parquet",
  "C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json",
  "C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json",
  "C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_meta_weekly_2018_2026.parquet",
  "C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_quotes_weekly_2018_2026.parquet",
  "C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/input/liquidity/ticker_liquidity_panel.parquet",
  "C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/manifests/input_snapshot_e2c1f8fd44d72176.json"
)
$inputs | ForEach-Object {
  $h = (Get-FileHash -Algorithm SHA256 $_).Hash.ToLower()
  $i = Get-Item $_
  "{0}`t{1}`t{2}`t{3}" -f $h, $i.Length, $i.LastWriteTimeUtc.ToString("o"), $_
}
```

**Pass:** the lineage receipt digest equals `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5`; also verify `configs/feature_backfill_v1.json` on-disk against `764056ce…` accounting for line endings as in step 1.4. Every other digest is recorded as the **pre-run baseline** — see PG-2.
**Stop:** lineage receipt digest mismatch; any listed input missing; any input path resolving under `C:/MomentumCVG_env/cache`.
**Retain:** the full digest/size/mtime table. Phase 4 re-runs this exact command and requires byte-identical digests.

> **PG-2 — resolved: pre-run and post-run digests accepted in place of pins.** The repository pins expected digests only for the lineage receipt and the feature config. There is no pinned expected SHA-256 for `features_42_8.parquet`, the A1/A2 surface parquets, or the liquidity panel, and the adapter's receipt records accepted-input **paths without digests**. D0 §12 left this box unchecked ("record at D1/D4 run time"). D4 therefore treats the Phase 1 values as a first-observation baseline and proves *immutability across the run* rather than *equality to a pin*. Making input digests a pinned, code-enforced identity is a candidate Sprint 007 item; **do not implement it in D4.**

### 1.7 Output location

Derive the run directory from a UTC launch stamp:

```powershell
$UTCSTAMP = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$RUN_DIR  = "C:/MomentumCVG_env/runs/sprint006_baseline_v1_$UTCSTAMP"
$RUN_DIR                                             # RECORD the absolute path
Test-Path $RUN_DIR                                   # must be False
Get-ChildItem C:/MomentumCVG_env/runs -Directory      # record existing siblings
```

`$UTCSTAMP` and `$RUN_DIR` persist in the operator's PowerShell session and are reused verbatim by Phase 3. If Phase 3 runs in a new session, re-derive both and record the new values; never reuse a stamp from a previous attempt.

**Pass:** `RUN_DIR` does not exist; it is outside `C:/MomentumCVG` and outside `C:/MomentumCVG_env/cache`; its parent exists and is writable.
**Stop:** the directory already exists, or is inside either forbidden root. The adapter also refuses all three cases.
**Retain:** the chosen absolute `RUN_DIR` and the pre-existing sibling listing.

### 1.8 Phase 1 gate

Any failure in 1.1–1.7 **blocks execution**. Record the failure, stop, and escalate per §9. Do not proceed to Phase 2 with a mismatch "noted for later".

---

## 5. Phase 2 — Small real-data smoke

**Purpose.** Catch integration failures (path/schema/dtype/serialization) cheaply on one real date, and confirm that the artifacts D4 depends on — `date_status`, `funnel_summary`, `leg_log`, `candidate_view` — are populated with real data and internally consistent. This is a plumbing check, **not** a performance check.

**May be inspected:** exit codes, file presence, schemas and dtypes, row counts, `date_status` values, funnel null-vs-zero semantics, per-trade leg rows and their reconciliation arithmetic.
**May not be inspected or reported:** aggregate returns, Sharpe, drawdown, yearly tables, top-5 concentration, or any figure from `decision_report.*` / `run_summary_*.json`. Smoke numbers must never be cited as a result.

### 2.1 Deriving the smoke date (do not hardcode)

The date is the **median of the frozen expected calendar**, per D0 §9 sample S1. The calendar is defined by D0 §8.0 and implemented by `SurfaceRunner._get_expected_dates_from_a1`: sorted unique A1 `entry_date` values in the closed interval `[2018-10-26, 2026-07-10]`, **including dates that appear only on `surface_valid=False` rows**.

Read-only derivation (loads A1 metadata only; runs no economics):

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -c @"
import pandas as pd, datetime as dt
meta = pd.read_parquet(r'C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/cache/surface/option_surface_meta_weekly_2018_2026.parquet',
                       columns=['entry_date'])
d = pd.to_datetime(meta['entry_date']).dt.date
lo, hi = dt.date(2018,10,26), dt.date(2026,7,10)
dates = sorted(set(x for x in d if lo <= x <= hi))
n = len(dates)
print('n_expected_dates', n)
print('lower_median_index', (n-1)//2, dates[(n-1)//2])
print('upper_median_index', n//2, dates[n//2])
print('first', dates[0], 'last', dates[-1])
"@
```

**Resolved convention (PG-1a, decided):** `MEDIAN_DATE = dates[(n - 1) // 2]` — the **lower median** when `n` is even. For odd `n` both indices coincide and the convention is irrelevant. The expected value is **`2022-09-02`**, the date Sprint 005 D5 derived under the same median rule.

D0 §9 says "median A1 expected date (sorted)" without disambiguating an even-length calendar; the lower-median rule above closes that ambiguity for D4. It governs sample selection only and never touches economics. Record both median indices and both candidate dates in the evidence memo regardless of parity, so the choice is auditable.

**Stop condition:** if the derived `MEDIAN_DATE` is not `2022-09-02`, **stop**. Record `n`, both median indices, both candidate dates, and the first/last calendar dates, and escalate for review. Do not silently substitute another date, and do not adjust the interval or the convention to reach `2022-09-02`.

### 2.2 The smoke cannot be date-restricted with the current CLI — resolved by Option A

**Implementation limitation.** `scripts/run_sprint006_baseline.py` exposes only `--contract`, `--output-dir`, and `--dry-run`. The run window comes from the contract, and `build_run_configs` requires `shared_run_config.start_date/end_date` to equal `periods.run_start_date/run_end_date`. There is **no** flag, environment variable, or supported entry point that limits an official-path run to one date, so "run the frozen settings on `2022-09-02` only" is not achievable through the accepted command as implemented. D4 does **not** fix this; making the limitation disappear would be a production change.

**Resolved decision (PG-1b): Option A — an outside-repository, date-narrowed smoke contract.**

Procedure: copy the frozen JSON to a disposable directory outside the repository and change **only** the four date fields to `2022-09-02` — `periods.run_start_date`, `periods.run_end_date`, `shared_run_config.start_date`, `shared_run_config.end_date`. Diff the copy against the frozen file and confirm that exactly those four lines differ. Then invoke the existing CLI with `--contract <copy>`; both frozen fills still execute in one invocation. Executable commands are in §2.3.

Option B (a full frozen run into a disposable directory) was considered and **rejected**: it costs a full execution, produces a complete non-official artifact set, and creates a stronger confusion hazard than A.

**Known hazard of Option A, contained by §2.5.** `load_contract` hard-requires `contract_id == "sprint006_baseline_v1"` and `status == "accepted"`, so the smoke receipt will carry the frozen contract id and the two frozen `run_id`s while its `contract.sha256` differs from both digests recorded in §2.1 *Code and configuration*. That divergence is **expected and by design**; it is the signal that the artifacts are smoke, not baseline. The frozen file itself is never edited — only the copy.

### 2.3 Smoke execution (both fills, disposable directory)

Step 1 — derive the disposable paths. `$SMOKE_DIR` holds the smoke contract and the §2.5 marker; the adapter's own output goes one level below in `$SMOKE_OUT`, which must not exist beforehand because `create_run_dir` refuses an existing directory.

```powershell
cd C:\MomentumCVG
$SMOKE_STAMP    = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$SMOKE_DIR      = "C:/MomentumCVG_env/runs/sprint006_d4_smoke_$SMOKE_STAMP"
$SMOKE_OUT      = "$SMOKE_DIR/run"
$SMOKE_CONTRACT = "$SMOKE_DIR/sprint006_smoke_contract.json"
$FROZEN         = "C:/MomentumCVG/configs/sprint006_baseline_v1.json"
$SMOKE_DIR; $SMOKE_OUT; $SMOKE_CONTRACT          # RECORD all three
Test-Path $SMOKE_DIR                             # must be False
New-Item -ItemType Directory -Path $SMOKE_DIR | Out-Null
Test-Path $SMOKE_OUT                             # must be False — the adapter creates it
```

Step 2 — build the date-narrowed contract copy, changing only the four date fields. Each pattern anchors on the opening quote, so `primary_reporting_start_date` / `primary_reporting_end_date` and `run_start_date` / `run_end_date` cannot be matched by the bare `start_date` / `end_date` patterns.

```powershell
$MEDIAN_DATE = "2022-09-02"                      # from §2.1; must match the derivation
$frozenText  = Get-Content $FROZEN -Raw
$smokeText   = $frozenText `
  -replace '("run_start_date"\s*:\s*)"2018-10-26"', ('$1"' + $MEDIAN_DATE + '"') `
  -replace '("run_end_date"\s*:\s*)"2026-07-10"',   ('$1"' + $MEDIAN_DATE + '"') `
  -replace '("start_date"\s*:\s*)"2018-10-26"',     ('$1"' + $MEDIAN_DATE + '"') `
  -replace '("end_date"\s*:\s*)"2026-07-10"',       ('$1"' + $MEDIAN_DATE + '"')
# Write UTF-8 without a BOM: Windows PowerShell 5.1 `Set-Content -Encoding utf8` prepends a BOM,
# which json.load() would reject. Byte content otherwise matches the frozen file apart from the four dates.
[System.IO.File]::WriteAllText($SMOKE_CONTRACT, $smokeText, (New-Object System.Text.UTF8Encoding($false)))
```

Step 3 — prove the copy differs from the frozen contract in exactly four lines, and prove the frozen file is untouched.

```powershell
$delta = Compare-Object (Get-Content $FROZEN) (Get-Content $SMOKE_CONTRACT) -SyncWindow 0
$delta | Format-Table -AutoSize                  # RECORD: expect 4 '<=' and 4 '=>' lines
($delta | Measure-Object).Count                  # expect 8
(Get-FileHash -Algorithm SHA256 $SMOKE_CONTRACT).Hash.ToLower()   # RECORD; differs from the frozen digests by design
(Get-FileHash -Algorithm SHA256 $FROZEN).Hash.ToLower()           # must still equal 4012b4a4… (or 3cd57f4d… on an LF checkout)
git -C C:/MomentumCVG status --porcelain configs/sprint006_baseline_v1.json   # must print nothing
```

**Stop** if the diff is anything other than those four date lines, or if the frozen contract's digest or Git status changed.

Step 4 — run the smoke. Both frozen fills always execute in one invocation; there is no fill selector, so "mid and cross" needs no extra argument.

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract $SMOKE_CONTRACT `
  --output-dir $SMOKE_OUT
$LASTEXITCODE                                    # RECORD; expect 0
```

**Inputs:** `$SMOKE_CONTRACT`; the §2.2 accepted inputs.
**Outputs (in `$SMOKE_OUT`):** for each of `sprint006_baseline_v1_mid` and `sprint006_baseline_v1_cross` — `trade_log_<run_id>.parquet`, `date_summary_<run_id>.parquet`, `date_status_<run_id>.parquet`, `run_summary_<run_id>.json`, `candidate_view_<run_id>.parquet`, `leg_log_<run_id>.parquet`, `funnel_summary_<run_id>.parquet`; plus one `decision_report.json`, one `decision_report.md`, one `run_receipt.json`. That is 17 adapter-written files.

### 2.4 Smoke checks

| # | Check | Pass criterion |
|---|-------|----------------|
| S-1 | Execution | Exit code 0; no `ERROR:` on stderr; stdout lists all seven per-run files for both runs plus report and receipt paths |
| S-2 | Artifact set | All 17 expected **adapter-written** files present in `$SMOKE_OUT`; no extras. Count adapter outputs only, and perform the count **before** the §2.5 marker file is written. The marker `NOT_THE_OFFICIAL_BASELINE.txt` and `sprint006_smoke_contract.json` live in `$SMOKE_DIR`, one level above `$SMOKE_OUT`, so they are never part of this count |
| S-3 | Schemas | `date_status` columns exactly `["trade_date","status","reason"]`; `funnel_summary` columns exactly `FUNNEL_SUMMARY_COLUMNS` (18); `leg_log` columns exactly `LEG_LOG_COLUMNS` (21); `candidate_view` columns exactly `CANDIDATE_VIEW_COLUMNS` (9) |
| S-4 | Date status | Exactly one `date_status` row per fill, `trade_date == MEDIAN_DATE`, `status ∈ {traded, valid_no_trade, failed}`; `reason` is `null` iff `traded` |
| S-5 | Funnel semantics | For a `failed`/`missing_features` date: `n_feature_covered=0` and `n_universe`…`n_included` all **null**. For an evaluated date: `n_universe` and `n_jointly_eligible` are non-null integers, `n_post_signal ≤ n_jointly_eligible`, `n_constructable ≤ n_post_signal`, `n_included ≤ n_constructable`, and long+short splits sum to their totals. Zero ≠ null is respected |
| S-6 | Leg serialization | Every `structure_ok=True` row on the date has leg rows; every included straddle has exactly 2 (`leg_index {0,1}`), every included iron fly exactly 4 (`leg_index {0,1,2,3}`); an included short iron fly has `unit_quantity` signs `+ − − +` by `leg_index` and `portfolio_quantity = abs(trade.quantity) × unit_quantity` with the same signs; non-included constructable rows have `portfolio_quantity` and `pnl_total_leg` null |
| S-7 | Included-trade reconciliation | For every included trade on the date, and independently of the generator: `Σ entry_cash_per_unit = entry_cost_per_share`; `Σ expiry_payoff_per_unit = entry_cost_per_share + pnl_per_share`; `Σ pnl_per_unit = pnl_per_share`; `Σ pnl_total_leg = pnl_total`; tolerance `max(1e-6 abs, 1e-8 rel)` |
| S-8 | Structure failures | Any `stage=structure_failed` candidate row has `reason_code` in the four frozen classes and a retained `reason_raw`; such rows have **no** leg rows |
| S-9 | Fill differentiation (**conditional**) | Determine whether any `(ticker, direction)` is constructable in **both** the mid and the cross leg log on `MEDIAN_DATE`. **If at least one overlap exists:** for one such name, every leg must satisfy the exact fill relationship — buy legs (`unit_quantity > 0`): `cross_fill_price − mid_fill_price = +0.5 × (ask − bid)`; sell legs (`unit_quantity < 0`): `cross_fill_price − mid_fill_price = −0.5 × (ask − bid)` — within `max(1e-6 absolute, 1e-8 relative)`, while the `mid` column is identical across fills. Strict inequality (cross buys above mid, cross sells below mid) applies **only where `ask > bid`**; on a zero-spread leg the two fills are **equal**, which is correct and not a failure. **If no overlap exists:** record `N/A — no overlapping constructable name` with the mid-only and cross-only constructable key counts, and continue. Do **not** substitute another date, widen the window, or relax the frozen settings to manufacture an overlap. Only an **incorrect comparison on an existing overlap** blocks execution; absence of an overlap does not |
| S-10 | No aggregate inspection | The operator attests that `decision_report.json`, `decision_report.md`, and `run_summary_*.json` economic fields were not opened |

**Stop conditions:** non-zero exit; a `DecisionMetricsError` / `ContractError` abort; missing artifact; schema deviation; funnel null-vs-zero violation; leg count/sign violation; any reconciliation break; an incorrect fill-price comparison on an **existing** mid/cross overlap (S-9); a `failed` date whose reason is not explainable by the recorded inputs. A recorded `N/A` on S-9 is **not** a stop condition.

If the smoke aborts, that is a **correctness signal, not a nuisance**. Preserve `$SMOKE_DIR`, stop, and escalate per §9. Do not proceed to Phase 3.

### 2.5 Preventing smoke/baseline confusion

Mandatory, all of them:

1. `$SMOKE_DIR` name contains `smoke` and is a sibling of, never inside, the official `$RUN_DIR`.
2. After the S-2 file count, write the marker file:

```powershell
@(
  "Sprint 006 D4 smoke run - NOT the official baseline.",
  "Purpose: integration/plumbing check on one real date only.",
  "PG-1b resolution: Option A (outside-repository date-narrowed smoke contract).",
  "Smoke contract: $SMOKE_CONTRACT",
  "Smoke contract SHA-256: " + (Get-FileHash -Algorithm SHA256 $SMOKE_CONTRACT).Hash.ToLower(),
  "MEDIAN_DATE: $MEDIAN_DATE",
  "Adapter output directory: $SMOKE_OUT",
  "Smoke artifacts must not be cited as Sprint 006 economic evidence."
) | Set-Content -Path "$SMOKE_DIR/NOT_THE_OFFICIAL_BASELINE.txt"
```

3. Record the four-line diff of the smoke contract against the frozen file, and note that the smoke receipt's `contract.sha256` differs from both digests in §2.1 *Code and configuration* by design.
4. The Phase 4/5 evidence tables cite **only** the official `$RUN_DIR`. Smoke paths appear only in the smoke section of the evidence memo, labeled as such.
5. Smoke artifacts are never committed and never copied into `$RUN_DIR`.

---

## 6. Phase 3 — Official full baseline

**Purpose.** Produce the one official, citable Sprint 006 economic run.

**May be inspected:** the command, exit code, stdout paths and row counts, timestamps.
**May not be inspected:** any economic content of the outputs. Phase 3 ends at "the run completed and wrote these files."

### 3.1 The command (existing, unmodified)

```powershell
cd C:\MomentumCVG
# Reuse the §1.7 values if this is the same shell session; otherwise re-derive and record.
if (-not $RUN_DIR) {
  $UTCSTAMP = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
  $RUN_DIR  = "C:/MomentumCVG_env/runs/sprint006_baseline_v1_$UTCSTAMP"
}
$RUN_DIR                                              # RECORD
Test-Path $RUN_DIR                                    # must be False
$START_UTC = (Get-Date).ToUniversalTime().ToString("o"); $START_UTC   # RECORD
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract configs/sprint006_baseline_v1.json `
  --output-dir $RUN_DIR
$EXIT_CODE = $LASTEXITCODE
$END_UTC = (Get-Date).ToUniversalTime().ToString("o"); $END_UTC       # RECORD
$EXIT_CODE                                                            # RECORD
```

One invocation. Both frozen fills (`mid` diagnostic, `cross` primary) execute from the contract's `runs[]` list. No other flags exist and none may be invented.

**Inputs:** frozen contract at `EXECUTION_COMMIT`; the §2.2 accepted inputs.
**Outputs:** the 17 files listed in §2.3, in `$RUN_DIR`.

### 3.2 Record immediately after the run

| Field | Source |
|-------|--------|
| Exact command line | as typed above |
| `EXECUTION_COMMIT` | §1.1 |
| Contract on-disk SHA-256 and committed blob id | §1.4 |
| Input digest table | §1.6 |
| Run ids | `sprint006_baseline_v1_mid`, `sprint006_baseline_v1_cross` |
| `START_UTC`, `END_UTC`, wall-clock duration | shell |
| `EXIT_CODE` | shell |
| `RUN_DIR` absolute path | §1.7 |
| Full stdout / stderr | transcript |
| `run_receipt.json` `generated_utc` | receipt |

**PG-3 — resolved: shell-recorded timing accepted.** The receipt records `generated_utc` but no run start time or duration. `START_UTC` and `END_UTC` therefore come from the operator's PowerShell transcript (§3.1) and are recorded in the evidence memo. This is a recording procedure, not a defect; do not add timing code in D4.

### 3.3 Prohibitions

* **No** configuration edit before, during, or after the run.
* **No** retry into `$RUN_DIR` — the adapter refuses, and forcing it is a stop condition.
* **No** post-result code change (see §9).
* **No** second official run "for comparison."
* **No** deletion or reorganization of `$RUN_DIR` contents.

### 3.4 Failure handling

If `EXIT_CODE != 0`, or the adapter/report aborts:

1. Stop. Do not rerun immediately.
2. Preserve `$RUN_DIR` exactly as-is, including partial artifacts, plus full stdout/stderr and the traceback.
3. Record the failure in the evidence memo with the failing check and message.
4. Diagnose read-only. Understand the cause before proposing anything.
5. A rerun requires a **new** `RUN_DIR`, a documented written reason, and human approval. If the cause is a code or data defect, §9 applies and the fix is a separate reviewable commit — the rerun happens after that fix is reviewed, not before.

### 3.5 Phase 3 pass criteria

Exit code 0; all 17 artifacts present; stdout lists both runs with trade-log row counts and the report/receipt paths; `$RUN_DIR` was created by this run and contains nothing else.

---

## 7. Phase 4 — Blind technical verification

**Purpose.** Decide whether the Phase 3 evidence is trustworthy, **before** any aggregate economic number is read. The output of Phase 4 is a single **evidence verdict**: `ACCEPTED` or `BLOCKED`.

### 7.1 The blind boundary (enforced by inspection discipline)

**Closed until the evidence verdict is recorded:**

* All **values** in `decision_report.json`, and `decision_report.md` in its entirety. **Key names only** may be inspected programmatically for V-9 — see the exception below.
* In `run_summary_<run_id>.json`: `mean_cycle_return_on_capital_at_risk`, `annualized_sharpe`, `max_drawdown`, `robust_score`, `hit_rate`, `availability_rate`, `mean_/median_trade_return_on_body_credit`, `avg_long_/avg_short_cycle_return`, `avg_*_return_on_body_credit`, `avg_spread_cost_ratio`, `avg_leg_spread_to_credit_ratio`.
* `date_summary_<run_id>.parquet` **as a series** — no aggregation, no sorting by return, no min/max/mean over dates.

**Open during Phase 4:**

* `run_receipt.json` in full. It carries every completeness and identity field Phase 4 needs — `repo_sha`, `contract.*`, `accepted_inputs`, per-run `n_expected_dates`, `n_traded_dates`, `n_valid_no_trade_dates`, `n_failed_dates`, `has_unresolved_failures`, `n_feature_dates_absent_from_a1`, `feature_dates_absent_from_a1`, per-file `sha256`, `result_complete`, `deferred` — without exposing performance.
* `date_status`, `funnel_summary`, `candidate_view`, `leg_log`, `trade_log` parquets, and structural fields of `run_summary_*.json` (`run_id`, `fill_label`, `momentum_col`, `cvg_col`, `short_structure`, `n_trade_dates`, `n_candidate_rows`, `n_traded_rows`, the date-class counts).
* Per-trade economics for the sampled trades only, and the single `date_summary` row of each sampled date (needed for the frozen `date_car_contribution` check).

**Key-only exception for `decision_report.json` (V-9).** The report's schema may be verified by listing **key names** programmatically, provided no value is printed, returned, or logged. Use a key-only walk:

```powershell
$keys = New-Object System.Collections.Generic.List[string]
function Get-JsonKeys($node, $path) {
  if ($node -is [System.Management.Automation.PSCustomObject]) {
    foreach ($p in $node.PSObject.Properties) {
      $keys.Add(("{0}.{1}" -f $path, $p.Name).TrimStart('.'))
      Get-JsonKeys $p.Value ("{0}.{1}" -f $path, $p.Name)
    }
  } elseif ($node -is [System.Object[]] -and $node.Count -gt 0) {
    Get-JsonKeys $node[0] ("{0}[]" -f $path)
  }
}
Get-JsonKeys (Get-Content "$RUN_DIR/decision_report.json" -Raw | ConvertFrom-Json) ""
$keys | Sort-Object -Unique      # key names only; no values are emitted
```

Anything that renders a value — `ConvertTo-Json`, `Format-List`, `Select-Object <field>`, opening the file in an editor or pager, or reading `decision_report.md` — remains closed until the verdict is recorded.

Selecting or replacing a sample on the basis of P&L is forbidden. Sorting the trade log by P&L is forbidden.

### 7.2 Mandatory completeness and identity checks

| # | Check | Expected | Source |
|---|-------|----------|--------|
| V-1 | `result_complete` | `true` | `run_receipt.json` |
| V-2 | `has_unresolved_failures` | `false` | receipt |
| V-3 | Failed dates, both fills | `n_failed_dates == 0` | receipt per-run fields; cross-check `date_status` |
| V-4 | Expected calendar, both fills | `n_expected_dates` identical for mid and cross, and equal to the independently derived A1 count from §2.1 (`n`) | receipt + §2.1 derivation |
| V-5 | Calendar partition | `date_status` has one row per expected date, no duplicates, statuses ⊆ `{traded, valid_no_trade, failed}`, `reason` null iff `traded`, and the row set equals the derived calendar exactly | `date_status` parquets |
| V-6 | Calendar bounds | first date ≥ `2018-10-26`, last date ≤ `2026-07-10` | `date_status` |
| V-7 | Feature reconciliation | `n_feature_dates_absent_from_a1` and the listed dates recorded and explained; A1 dates missing from features would have surfaced as `failed`/`missing_features` and V-3 already forbids them | receipt |
| V-8 | Artifact presence | all 17 files present, non-zero size | directory listing |
| V-9 | Pinned schemas | `date_status` = 3 cols; `funnel_summary` = `FUNNEL_SUMMARY_COLUMNS`; `leg_log` = `LEG_LOG_COLUMNS`; `candidate_view` = `CANDIDATE_VIEW_COLUMNS`; `date_summary` = the 19 `build_date_summary` columns; `decision_report.json` top-level keys = `experiment_id, contract_id, repo_sha, result_complete, has_unresolved_failures, windows, fills, by_fill, fill_assumption_sensitivity, concentration_primary_cross_top5, limitations`, verified by the §7.1 key-only walk (**key names only — every value stays closed**) | parquet headers; §7.1 key walk |
| V-10 | Digest verification | Recomputed SHA-256 of every file in `$RUN_DIR` equals the value recorded in `run_receipt.json` | `Get-FileHash` |
| V-11 | Code identity | receipt `repo_sha == EXECUTION_COMMIT` | receipt vs §1.1 |
| V-12 | Config identity | receipt `contract.sha256` equals the §1.4 on-disk digest; `contract_id/version/status` = `sprint006_baseline_v1`/`1`/`accepted`; every `effective_config` field matches §2.3 for both runs, differing only in `run_id` and `fill` | receipt |
| V-13 | Input identity | receipt `accepted_inputs` paths equal §2.2 exactly; none under `C:/MomentumCVG_env/cache`; `earnings_path` null | receipt |
| V-14 | Input immutability | Re-running the §1.6 digest command yields digests byte-identical to the pre-run baseline | §1.6 rerun |
| V-15 | Run identity | receipt `experiment_id = sprint006_baseline_v1`; run ids exactly the two frozen ids; `deliverable == "sprint006_d3"` and `deferred` still listing the D4 item — both **expected**, see the note below | receipt |
| V-16 | No unexplained artifacts | `$RUN_DIR` contains exactly the 17 expected files; no duplicates, temp files, or `.1`/`.bak` variants; all mtimes inside `[START_UTC, END_UTC]` | listing + timestamps |
| V-17 | Funnel monotonicity | On every evaluated date and both fills: `n_included ≤ n_constructable ≤ n_post_signal ≤ n_jointly_eligible ≤ n_universe`; side splits sum to totals; nulls only on unevaluated stages | `funnel_summary` |
| V-18 | Included-trade leg completeness | Independently re-derived: every included trade has exactly 2 (straddle) or 4 (iron fly) matching legs with the required `leg_index` set, no duplicates, and no included leg rows without a matching included trade | `trade_log` + `leg_log` |
| V-19 | Portfolio-wide reconciliation | The four S-7 identities hold for **every** included trade in both fills, checked independently of `assert_included_trade_legs` | `trade_log` + `leg_log` |
| V-20 | Candidate/trade consistency | `candidate_view` row count equals `trade_log` row count per run; `stage=traded` ⟺ `included_in_portfolio=True`; `structure_failed` rows have a frozen `reason_code`; `portfolio_excluded` rows carry an S5 vocabulary code | `candidate_view` + `trade_log` |

Digest verification helper:

```powershell
$receipt = Get-Content "$RUN_DIR/run_receipt.json" | ConvertFrom-Json
Get-ChildItem $RUN_DIR -File | ForEach-Object {
  "{0}`t{1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.Name
}
```

Compare against `$receipt.runs[*].outputs.*.sha256` and `$receipt.decision_report.*.sha256`. `run_receipt.json` itself is not self-digested; that is expected.

**Receipt `deliverable` / `deferred` are D3 producer metadata, not lifecycle state.** A D4 run emits `deliverable = "sprint006_d3"` and a `deferred` list still naming "real-data smoke, manual trade sample, and full-history execution (D4)". These come from `build_receipt` and the module constant `DEFERRED_TO_LATER_DELIVERABLES` in `src/backtest/sprint006_baseline.py`: they record **which deliverable's code wrote the artifacts**, which is genuinely D3's adapter, not how far the sprint has progressed. Seeing them in a D4 receipt is therefore **correct and is not a V-15 failure**. D4's lifecycle acceptance is recorded in the D4 evidence memo and the sprint agenda instead. **Do not change production code to relabel these fields** — that would put a production change inside the evidence commit, which §9 forbids. A future deliverable-aware receipt label is a candidate Sprint 007 cleanup item, not D4 work.

**Any failure in V-1…V-20 sets the evidence verdict to `BLOCKED` and forbids Phase 5.**

### 7.3 Frozen manual sample — D0 §9 rules, unchanged

Cap: **≤ 6 hand-checked trades** total. No performance-based selection or replacement.

| Sample | Frozen rule | Frozen fallback |
|--------|-------------|-----------------|
| **S1** | Median A1 expected date (`MEDIAN_DATE` from §2.1). Date-level lineage/status is checked always. If that date is `traded`, sample the **lowest-ticker** included **long** and the **lowest-ticker** included **short** that exist on it | Sample only the sides that exist |
| **S2** | Earliest `traded` date with **both** sides present; sample the lowest-ticker included long and short | If no date has both sides: earliest long-traded date and/or earliest short-traded date, as available |
| **S3** | Earliest `valid_no_trade` date | If none exists → record `N/A` |
| **S4** | Earliest date with ≥ 1 `structure_ok=False` row | If none exists → record `N/A` |
| Shortfall | If fewer than six qualifying trade rows arise from S1/S2, audit those available and **document the shortfall** | Never substitute a P&L-selected date |

Agreed clarification (applies to selection only, changes no frozen rule):

1. Select samples from the **primary cross** run's artifacts.
2. Where a matching `(trade_date, ticker, direction)` record exists in the **mid** run, compare it alongside; a matching mid record is expected only for names constructable and included under both fills.
3. **Never** select or replace a sample based on P&L, sign, magnitude, or rank.
4. If a requested category does not exist, **record that fact** and use only the already-frozen fallback above. Invent no new fallback.

Record for each sample: the rule that produced it, the enumeration used (e.g. "earliest `traded` date with both sides, ascending scan of `date_status`"), the resulting keys, and whether a fallback fired.

### 7.4 Source-level reconstruction of included trades

The audit must be an **independent reconstruction from the accepted inputs**, not a restatement of `trade_log` / `leg_log`. Read the A1 meta, A2 quotes, liquidity panel, and `features_42_8.parquet` directly and recompute each quantity, then compare.

| Stage | Fields to recompute from source | Source | What it proves |
|-------|--------------------------------|--------|----------------|
| **Universe (S1)** | Snapshot date used = `max(month_date < trade_date)`; `has_valid_atm_pair=True`; non-null `atm_straddle_dollar_vol` and `atm_spread_pct`; `dvol_rank_pct` (ascending, `method=average`, `pct=True`) and `spread_rank_pct` (descending) over the full snapshot; membership requires `dvol_rank_pct ≥ 0.80` **and** `spread_rank_pct ≥ 0.00` | liquidity panel | PIT correctness; no same-day or future snapshot; AND logic |
| **Joint eligibility (S2 pre-rank)** | Ticker in universe ∩ trade-date feature slice; `mom_42_8_mean` and `cvg_42_8` finite; `mom_42_8_count ≥ 28` **and** `cvg_count_42_8 ≥ 28` | features | Joint count eligibility with `required_count=28` |
| **Signal eligibility and direction** | `signal_rank_pct` = `mom_42_8_mean.rank(ascending=True, method='average', pct=True)` on the eligible slice; long pool `≥ 0.90`; short pool `≤ 0.10`; within each side `cvg_rank_pct` on `cvg_42_8` and keep `≥ 0.50`; direction tag | features | Correct tail membership, CVG retention, and side assignment |
| **Option selection** | A1 row for `(ticker, trade_date)`: `surface_valid`, `entry_spot`, `exit_spot`, `body_strike`, `expiry_date`, `dte_actual`. A2 rows: body legs = `is_body` with `spread_pct ≤ 0.50`; iron-fly wings = `is_otm` per side with `spread_pct ≤ 0.50`, then highest `abs_delta ≤ 0.15` (`_choose_below_nearest`) | A1 + A2 | Correct strikes, expiry, ATM body, wing rule, and all-leg spread gate |
| **Quotes and fills** | Per leg `bid`, `ask`, `mid`; mid fill = `bid + 0.5(ask − bid)` for buys and `ask − 0.5(ask − bid)` for sells; cross fill = `ask` for buys and `bid` for sells | A2 | `fill` is the sole pricing layer; no extra deduction |
| **Structure** | Long: 2 legs, `+1` call and `+1` put at `body_strike`. Iron fly: 4 legs in order `+1` long OTM put, `−1` short ATM put, `−1` short ATM call, `+1` long OTM call; `wing_width = max(call_strike − body_strike, body_strike − put_strike)`; `net_credit = −entry_cost`; `max_loss_per_share = wing_width − net_credit` | derived | Leg type/strike/sign/order and defined risk |
| **Quantity** | Tier A `equal_max_loss`: shorts `abs_qty = (10000 / n_short) / at_risk_per_share`, sign negative; longs financed by collected short credit `Σ abs(qty) × credit_per_share`, `abs_qty = (long_budget / n_long) / premium_per_share`, sign positive; fallback to `tier_a_long_budget=10000` only when there are no usable shorts or collected credit ≤ 0 (record if the fallback fired) | derived | Sizing rule, per-side budget split, fallback disclosure |
| **Entry cash** | Per leg `+fill_price × abs(unit_quantity)` for longs, `−fill_price × abs(unit_quantity)` for shorts; summed = `entry_cost_per_share` (positive = debit, negative = credit) | derived | Entry-cash sign convention |
| **Expiry payoff** | Per leg `intrinsic(exit_spot) × unit_quantity`, summed = `exit_value`; no exit spread applied | A1 `exit_spot` | Hold-to-expiry intrinsic settlement |
| **P&L** | `pnl_per_share = exit_value − entry_cost_per_share`; `pnl_total = abs(quantity) × pnl_per_share`; `capital_at_risk_dollars = abs(quantity) × at_risk_per_share` | derived | Correct magnitude scaling, no direction double-count |
| **CAR contribution** | Sampled date's `cycle_return_on_capital_at_risk = Σ pnl_total / Σ capital_at_risk_dollars` over that date's included rows | that one `date_summary` row | The frozen D0 `date_car_contribution` check |

Also recompute the mid-versus-cross difference for each sampled trade at leg level and confirm the exact relationship, within the tolerance below: buy legs (`unit_quantity > 0`) satisfy `cross_fill_price − mid_fill_price = +0.5 × (ask − bid)`, and sell legs (`unit_quantity < 0`) satisfy `cross_fill_price − mid_fill_price = −0.5 × (ask − bid)`. Where `ask > bid` this makes bought legs cost more and sold legs receive less under cross, so a long straddle's debit rises and an iron fly's credit falls; where `ask == bid` the two fills are equal and no difference should appear.

Tolerance throughout: `max(1e-6 absolute, 1e-8 relative)`, matching the D3 leg tolerance.

### 7.5 Earliest `valid_no_trade` and earliest structure-failure checks

**S3 — earliest `valid_no_trade`.** Take the earliest such date from the cross `date_status`, read its `reason`, and confirm the reason is consistent with the funnel row and the candidate view:

* `reason=empty_signals` ⇒ `n_post_signal = 0` with `n_constructable = n_included = 0`, `n_universe`/`n_jointly_eligible` non-null, and zero `candidate_view` rows for the date. Reconstruct from source why the eligible cross-section produced no tail survivors (e.g. eligible slice empty, or tails empty after the CVG filter).
* `reason=no_included_names` ⇒ `n_post_signal > 0` and `n_included = 0`; every `candidate_view` row is `structure_failed` or `portfolio_excluded` with a frozen `reason_code`; `trade_log` has no `included_in_portfolio=True` row for the date. Confirm at least one exclusion reason against source data.
* In both cases confirm `date_summary` contains no row implying included economics for that date, consistent with `assert_report_preconditions`.
* If no `valid_no_trade` date exists, record `S3 = N/A` with the count that justifies it.

**S4 — earliest structure failure.** Take the earliest date with ≥ 1 `structure_ok=False` candidate row, pick that date's lowest-ticker failing row, and:

* Record `reason_raw` verbatim and confirm `reason_code` matches the frozen prefix mapping (`metadata_error`, `missing_quotes_or_body`, `wing_or_liquidity_selection`, `other_structure`).
* Reproduce the failure from source: e.g. `surface_valid=False` or a missing A1 row (`metadata_error`); no `is_body` row surviving `spread_pct ≤ 0.50` (`missing_quotes_or_body`); no OTM quote with `abs_delta ≤ 0.15` surviving the spread gate (`wing_or_liquidity_selection`).
* Confirm the row has **no** `leg_log` rows and is **not** `included_in_portfolio`, and that the date is classified `traded` or `valid_no_trade` (a candidate-level failure is never a `failed` date).
* If no structure failure exists anywhere, record `S4 = N/A`.

### 7.6 Audit-record template

One row per checked item, in the evidence memo:

| id | sample | stage / check | expected value | observed value | source | difference | verdict |
|----|--------|---------------|----------------|----------------|--------|-----------|---------|
| `S1-L-legs` | S1 long `2022-09-02 / <TICKER>` | leg count and strikes | 2 legs, both at `body_strike=<X>` | 2 legs at `<X>` | A1 meta + A2 quotes | 0 | PASS |
| `S1-L-entry` | same | `Σ entry_cash_per_unit` vs `entry_cost_per_share` | `<recomputed>` | `<trade_log>` | A2 + fill alphas | `<abs / rel>` | PASS |
| `V-10` | — | receipt digest, `leg_log_..._cross.parquet` | `<receipt sha256>` | `<recomputed>` | `Get-FileHash` | — | PASS |

Fill `difference` with the absolute and relative gap for numeric rows and `—` otherwise. Verdict is `PASS`, `FAIL`, or `N/A (frozen fallback)`. Every `FAIL` must name the smallest separately reviewable fix (§9) and must not be repaired inside the evidence commit.

### 7.7 Evidence verdict

**`ACCEPTED`** requires: V-1…V-20 all pass; every selected sample reconciles within tolerance; S3/S4 either verified or recorded `N/A` under a frozen fallback; every shortfall documented.

**`BLOCKED`** on any identity, completeness, digest, schema, or reconciliation failure. `BLOCKED` **forbids** Phase 5 entirely — no aggregate return, Sharpe, drawdown, yearly table, attribution, or concentration figure may be opened, quoted, or summarized from a run whose evidence is blocked.

Record the verdict, with its justification, before opening any file listed in §7.1.

---

## 8. Phase 5 — Economic review and Sprint 006 closeout

**Entry condition:** evidence verdict `ACCEPTED`. Nothing in this phase may run otherwise.

**Purpose.** Describe what the frozen report says, decide what to do next, and close the sprint. Phase 5 reads the report; it does not compute new metrics, re-run anything, or adjust the experiment.

### 8.1 Review order (cross first, mid only as sensitivity)

1. **Cross (primary economic view)** — `by_fill.cross`, both windows. Every headline claim comes from here.
2. **Mid (diagnostic)** — `by_fill.mid`, read **only** to bound fill-assumption sensitivity. Never quoted as the result.
3. **Windows** — full history `2018-10-26`…`2026-07-10` and primary `2020-01-01`…`2026-07-10`, for both View A (conditional on traded dates) and View B (calendar-aligned, `valid_no_trade` = 0).
4. **Return** — View B `compounded` and `annualized_return` (weekly, 52); View A `mean_cycle_car`. Label View A numbers *conditional on traded dates* every time.
5. **Risk** — `sharpe` and `drawdown` for both views and both windows.
6. **Yearly stability** — `by_fill.cross.primary.yearly[]`: per-year return, Sharpe, drawdown, and date-class counts. Note dispersion and whether any single year dominates.
7. **Long/short attribution** — `long_short`: row counts, `pnl_total`, `capital_at_risk_dollars`, and mean side cycle returns. Note if one side carries the result.
8. **Cross-minus-mid and spread costs** — `fill_assumption_sensitivity`: `n_dates_both_traded`, `n_dates_cross_only`, `n_dates_mid_only`, candidate overlap counts, `mean_cross_minus_mid_car_both_traded`, `mean_cross_minus_mid_pnl_both_traded`, `mean_spread_cost_ratio_*`, `mean_leg_spread_to_credit_ratio_*`. State explicitly that this is **fill-assumption** sensitivity, not pure transaction cost: fills also change sizing, inclusion, and selected structures. Disclose unmatched dates and candidates.
9. **Concentration** — `concentration_primary_cross_top5`: the five tickers, their shares, and `top5_share_sum`.
10. **Activity and coverage** — `activity` (mean included names per traded date overall and by side; turnover, presented as complete only when `result_complete`), `weekly.no_trade_frequency`, `weekly.win_rate`, `weekly.profit_factor`, `funnel_totals` (`joint_coverage_rate`, `mean_jointly_eligible`, `sum_included`), and `structure_failure_counts`.
11. **Limitations** — reproduce `limitations[]` verbatim and add anything Phase 4 surfaced: hold-to-expiry, no earnings filter, below-nearest 0.15-delta wings, Tier A fractional (not integer lots), possible long-only fallback dates, mid as a fill-assumption diagnostic, `robust_score` excluded from decisions, post-signal selection bias in candidate/funnel artifacts, no PIT earnings artifact, iron condor untested while KB-001 is open, and no pinned input-digest identity (PG-2).

**Forbidden in Phase 5:** recomputing any metric; adding a metric; window-shifting; dropping a year, week, or ticker; per-ticker or per-regime slicing; any statistical test; any change to `42:8` or any frozen parameter.

### 8.2 Two separate conclusions

Both are recorded explicitly and never merged.

**1. Evidence verdict** — `ACCEPTED` or `BLOCKED` (from Phase 4).

**2. Economic characterization** — exactly one of:

* `PROMISING`
* `WEAK/NEGATIVE`
* `INCONCLUSIVE`

The characterization is **descriptive**, supported by cited fields of the frozen report, and constrained as follows:

* Do **not** invent a Sharpe, return, or drawdown cutoff. There is no numeric go/no-go in D0 and D4 must not create one.
* `robust_score` plays no role.
* Base it on the primary cross view, with View A and View B agreement or disagreement stated, yearly dispersion, side attribution, cross-minus-mid magnitude relative to the result, concentration, and activity.
* `INCONCLUSIVE` is the correct answer when the direction is unclear, when View A and View B disagree materially, when the result rests on very few traded dates, or when concentration or one-sided attribution makes the aggregate unrepresentative.
* State the characterization as a description of *this frozen configuration on this dataset under these assumptions* — never as a general claim about momentum or CVG.

### 8.3 Closeout recommendation

Exactly one:

1. **Proceed to bounded Sprint 007 robustness work** — evidence `ACCEPTED` and economics `PROMISING` or `INCONCLUSIVE` in a way a bounded, preregistered study could resolve. List the candidate questions without designing them here.
2. **Investigate a named correctness or data defect** — name the defect, the evidence that revealed it, and the smallest separately reviewable fix. Sprint 006 does not close as accepted until it is resolved.
3. **Reject or defer the economic hypothesis** — evidence `ACCEPTED` and economics `WEAK/NEGATIVE`. Record what would have to change to revisit it. **Do not** search for a better configuration; that is precisely the retuning D0 §6 forbids.

### 8.4 Closeout mechanics

* Update `docs/agenda/current_sprint.md`: status → closed, D4 complete, both conclusions, the recommendation, and the Definition-of-Done checkboxes with their supporting evidence.
* Write `docs/sprint_memos/006_closeout.md` following the `004_closeout.md` / `005_closeout.md` shape.
* Sync `docs/baseline_status.md` with the Phase 1 suite result and the tested baseline commit.
* Add the Sprint 006 rows to the `docs/README.md` memo index.
* Delete all **five** `docs/tmp/sprint006_d*_plan.md` documents per §10.3.

---

## 9. Stop-and-escalate rule

D4 is expected to require **zero production-code changes**. Every check in Phases 1–4 is a read, a digest, a schema assertion, or an arithmetic reconstruction.

If execution exposes a genuine code or data defect:

1. **Stop D4** at that point. Do not continue to the next phase.
2. **Preserve the evidence** — the run directory as-is, full stdout/stderr, the traceback, and the failing check with expected versus observed values.
3. **Do not patch inside the execution/evidence commit.** The evidence commit records what happened; it must not also contain the remedy.
4. **Describe the blocker and the smallest separately reviewable fix**: the defect, the affected module and function, why it invalidates the result, the narrowest change that addresses it, and the test that would prove it. Propose it; do not implement it.
5. **Do not inspect or use economic results from an invalid run.** A run whose evidence is `BLOCKED` yields no economic statement of any kind — not "directionally", not "informally", not "for context".
6. After the fix is separately reviewed and accepted, a rerun uses a **new** run directory, a new execution commit, and a new evidence record. The original invalid run is retained and labeled invalid; it is never overwritten or silently replaced (D0 §6.4, `pnl_firewall.correctness_change_after_exposure`).

Distinguish clearly in the escalation note between a **defect** (code or data is wrong) and a **planning gap** (the plan asked for something the implementation cannot do). PG-1a, PG-1b, PG-2, and PG-3 are planning gaps, and each is resolved by a documented procedure in §14 rather than by code; they are not defects and must not be "fixed" by editing production code during D4. The same holds for the receipt's D3 `deliverable`/`deferred` labels (§7.2).

---

## 10. Scope

### 10.1 Explicitly out of scope

Do not plan or perform any of the following in D4:

* Searching, ranking, or sampling across the 281 feature windows
* Any change to `42:8`
* New features, structures, sizing rules, cost models, commissions, or thresholds
* Walk-forward analysis, Monte Carlo, regime analysis, rank IC, or any new statistical test
* Drop-best-period, drop-worst-period, or drop-largest-ticker experiments
* Tier B integer-lot portfolio construction or a dollar capital budget
* Shadow trading, live trading, or broker integration
* New reporting features, charts, dashboards, or metrics
* Production refactoring or unrelated known-bug fixes
* Post-result robustness experiments of any kind
* Any edit to `configs/sprint006_baseline_v1.json`
* Repairing `SurfaceSearch` / `scripts/run_surface_search.py`
* Iron-condor comparison while KB-001 is open
* Implementing single-date CLI support (PG-1b tooling), PG-2 input-digest pinning, PG-3 timing capture, or a deliverable-aware receipt label (§7.2)

### 10.2 Deliberately narrow

D4 adds no framework, no guardrail beyond the checks above, and no research. Where a check can be done by reading a file, it is done by reading a file.

### 10.3 Temporary-document cleanup (commit 3)

**Resolved: delete all five plans.** At closeout, `git rm` all five temporary Sprint 006 plans — `docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`, `…_d1_…`, `…_d2_…`, `…_d3_…`, and this D4 plan — matching the Sprint 005 precedent of deleting accepted plans at closeout rather than copying them into `docs/archive/`. They are not moved, not archived, and not partially retained. The durable record moves into `docs/sprint_memos/006_closeout.md` and the D4 evidence memo; `configs/sprint006_baseline_v1.json` remains the frozen contract of record, and Git history retains the plans themselves.

---

## 11. Artifacts and three-commit sequence

### Commit 1 — D3 acceptance and proposed D4 plan (**this commit; documentation only**)

| File | Change |
|------|--------|
| `docs/tmp/sprint006_d4_execution_acceptance_plan.md` | **New.** This plan, `PROPOSED — AWAITING ACCEPTANCE` |
| `docs/agenda/current_sprint.md` | D3 accepted through `10133f6`; D4 planning started; D4 plan proposed; no D4 execution |
| `docs/tmp/sprint006_d3_decision_diagnostic_report_plan.md` | Status header → `ACCEPTED — D3 COMPLETE` with the accepted commit range |

Commit message: `docs(sprint006): define D4 execution and acceptance plan`. No push.

### Commit 2 — Accepted-plan execution and verification evidence (**only after human acceptance of this plan**)

| File | Content |
|------|---------|
| `docs/sprint_memos/sprint006_d4_baseline_execution_evidence.md` | Phase 1 gate results; Phase 2 smoke (Option A contract diff and digest, `MEDIAN_DATE` derivation, S-1…S-10 with the S-9 conditional outcome recorded); Phase 3 command, identities, `START_UTC`/`END_UTC`, `EXIT_CODE`, `RUN_DIR`, receipt digests; Phase 4 V-1…V-20 table, sample selection record, source-level reconstruction audit table, S3/S4 records; **evidence verdict**; residual limitations. Contains **no** economic interpretation |
| `docs/agenda/current_sprint.md` | Progress-log row: D4 executed and verified; evidence verdict; economics not yet reviewed |
| `docs/tmp/sprint006_d4_execution_acceptance_plan.md` | Status → `ACCEPTED — D4 EXECUTION COMPLETE (EVIDENCE ONLY)` |

Stays **outside** the repository: `$RUN_DIR` and every artifact in it, `$SMOKE_DIR`, all snapshot/derived source data. The memo cites absolute paths and digests, never copies.

### Commit 3 — Reviewed Sprint 006 closeout and cleanup (**only after economic review**)

| File | Content |
|------|---------|
| `docs/sprint_memos/006_closeout.md` | **New.** Sprint 006 closeout: evidence verdict, economic characterization, recommendation, Definition-of-Done status, limitations, Sprint 007 handoff |
| `docs/agenda/current_sprint.md` | Sprint status → closed with the recommendation |
| `docs/baseline_status.md` | Phase 1 suite result and tested baseline commit |
| `docs/README.md` | Sprint 006 memo/index rows |
| `docs/tmp/sprint006_d0_*_plan.md`, `…_d1_…`, `…_d2_…`, `…_d3_…`, `…_d4_…` | **Deleted** — all five, per §10.3 |

---

## 12. Exact future execution commands (reference)

Nothing below runs during the planning commit. Every path is a PowerShell variable derived at execution time; no directory, contract copy, or run output is created now.

```powershell
# Phase 1 — gate
cd C:\MomentumCVG
git status --porcelain; git rev-parse HEAD
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest -q
$DRYRUN_DIR = "C:/MomentumCVG_env/runs/sprint006_d4_dryrun_" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract configs/sprint006_baseline_v1.json `
  --output-dir $DRYRUN_DIR --dry-run
git rev-parse HEAD:configs/sprint006_baseline_v1.json
(Get-FileHash -Algorithm SHA256 configs/sprint006_baseline_v1.json).Hash.ToLower()

# Phase 2 — smoke (PG-1b Option A: outside-repository date-narrowed contract; see §2.3 for the full build/diff steps)
$SMOKE_STAMP    = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$SMOKE_DIR      = "C:/MomentumCVG_env/runs/sprint006_d4_smoke_$SMOKE_STAMP"
$SMOKE_OUT      = "$SMOKE_DIR/run"
$SMOKE_CONTRACT = "$SMOKE_DIR/sprint006_smoke_contract.json"
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract $SMOKE_CONTRACT `
  --output-dir $SMOKE_OUT

# Phase 3 — official baseline
$UTCSTAMP = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$RUN_DIR  = "C:/MomentumCVG_env/runs/sprint006_baseline_v1_$UTCSTAMP"
$START_UTC = (Get-Date).ToUniversalTime().ToString("o")
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py `
  --contract configs/sprint006_baseline_v1.json `
  --output-dir $RUN_DIR
$EXIT_CODE = $LASTEXITCODE
$END_UTC = (Get-Date).ToUniversalTime().ToString("o")
$RUN_DIR; $START_UTC; $END_UTC; $EXIT_CODE      # RECORD

# Phase 4 — digest verification
Get-ChildItem $RUN_DIR -File | ForEach-Object {
  "{0}`t{1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.Name
}
```

---

## 13. Acceptance checklist for this plan

- [ ] Central question, success criteria, and the negative-result rationale approved
- [ ] Immutable identifiers in §2 confirmed correct and complete
- [ ] Phase 1 gate accepted, including the dual-digest (LF and on-disk) contract procedure
- [ ] **PG-1a** resolved: lower median `dates[(n−1)//2]`, expected `2022-09-02`
- [ ] **PG-1b** resolved: Option A, the outside-repository date-narrowed smoke contract (four date fields, contained by §2.5)
- [ ] **PG-2** resolved: pre-run and post-run input digests accepted in place of pins, with no new pinning code in D4
- [ ] **PG-3** resolved: shell-recorded `START_UTC`/`END_UTC` accepted, with no timing code added
- [ ] **D4-Q1** resolved: execution runs from the final accepted plan commit
- [ ] **D4-Q3** resolved: the smoke is not skipped
- [ ] Conditional S-9 accepted: an absent mid/cross overlap is recorded as `N/A`, never worked around by changing the date
- [ ] Blind boundary in §7.1 accepted, including keeping `run_summary_*.json` economic fields and all `decision_report` values closed until the verdict, with only the key-only walk permitted for V-9
- [ ] Accepted that a D4 receipt still reports `deliverable=sprint006_d3` with D4 deferred, and that no production code changes to relabel it
- [ ] D0 §9 S1–S4 rules, fallbacks, and the ≤ 6 cap reproduced without redesign
- [ ] Source-level reconstruction field list sufficient and correct
- [ ] Two-verdict structure accepted; no numeric go/no-go threshold introduced
- [ ] Three-commit sequence and file assignment accepted, including deleting all five D0–D4 plans at closeout
- [ ] Out-of-scope list accepted
- [ ] Confirmed: this planning commit performs no real-data execution and no performance inspection

---

## 14. Resolved planning decisions

Every open item from the first draft is now decided. None of these resolutions changes production code, tests, or the frozen contract; they are procedural commitments binding on the operator.

| id | Item | Resolution |
|----|------|-----------|
| **PG-1a** | D0 §9 does not disambiguate "median" for an even-length expected calendar | **Resolved: lower median** `dates[(n−1)//2]`, expected `2022-09-02`. Record both candidate indices and dates; stop and escalate if the derivation yields anything else, and never substitute a date by hand (§2.1) |
| **PG-1b** | The accepted CLI cannot restrict a run to one date | **Resolved: Option A** — an outside-repository date-narrowed smoke contract with only the four date fields changed, verified by a four-line diff and contained by §2.5. Option B is rejected. No CLI change is made (§2.2–§2.3) |
| **PG-2** | No pinned expected digests for `features_42_8.parquet`, A1, A2, or the liquidity panel | **Resolved: accept pre-run and post-run digests** as the input-identity record, in place of pins. Digest the inputs in Phase 1, re-verify the same values in Phase 4, and treat any change as `BLOCKED`. Pinned input digests are a candidate Sprint 007 item; nothing is implemented in D4 |
| **PG-3** | The receipt has `generated_utc` but no start time or duration | **Resolved: accept shell-recorded timing.** `START_UTC` and `END_UTC` come from the operator's PowerShell transcript (§3.1) and are recorded in the evidence memo. No timing code is added |
| **D4-Q1** | Which commit executes D4 | **Resolved: execute from the final accepted plan commit** — the HEAD that carries the accepted D4 plan, with `10133f6` as an ancestor and a clean tree. That SHA is the recorded `EXECUTION_COMMIT` and must equal the receipt's `repo_sha` |
| **D4-Q2** | Retirement mechanism for the five temporary Sprint 006 plans | **Resolved: delete all five** D0–D4 plans in commit 3, matching the Sprint 005 precedent. No move to `docs/archive/`, no partial retention (§10.3) |
| **D4-Q3** | Whether the smoke may be skipped | **Resolved: do not skip the smoke.** Phase 2 runs before Phase 3 in every case; skipping is not an operator option |
| **S-9** | Mid/cross fill differentiation may have no overlapping constructable name on the smoke date | **Resolved: conditional check.** Verify the price differences when an overlap exists; otherwise record `N/A — no overlapping constructable name` and continue. Only an incorrect comparison on an existing overlap blocks execution (§2.4) |
| **Receipt labels** | A D4 run's receipt reports `deliverable=sprint006_d3` and still defers D4 | **Resolved: expected, no code change.** These are D3 producer-metadata fields; D4 lifecycle acceptance is recorded in the evidence memo (§7.2) |

No item remains open for human decision. Acceptance of this plan is acceptance of the table above.

---

**End of proposed D4 plan.** No D4 execution, real-data run, smoke run, official run directory, or aggregate P&L inspection is authorized by drafting or committing this document. Execution requires explicit acceptance of this plan, including the resolutions recorded in §14.
