# Sprint 006 D4 — Phase 1 and Phase 2 checkpoint

**Overall verdict: `PASS`** — every Phase 1 gate check and every Phase 2 smoke check (S-1…S-10) passed.
This checkpoint awaits human review. **Phase 3 is not authorized by it.**

| Field | Value |
|-------|-------|
| Accepted plan | `docs/tmp/sprint006_d4_execution_acceptance_plan.md` (amended two-date Phase 2) |
| Acceptance commit / `EXECUTION_COMMIT` | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Branch | `main`, working tree clean at start and end |
| Execution start (UTC) | `2026-08-23T03:15:32Z` (first Phase 1 timed command) |
| Execution end (UTC) | `2026-08-23T03:22Z` (marker written); checkpoint authored `2026-08-23T04:26Z` |
| Phases executed | Phase 1 (§§1.1–1.8) and Phase 2 (§§2.1–2.5) only |
| Phase 3 | **Not started** |

No previous Phase 1 result and no provisional smoke artifact was reused. Every check below was rerun from
scratch at `e205b9a`.

---

## 1. Phase 1 — pre-execution gate

| Check | Expected | Observed | Verdict |
|-------|----------|----------|---------|
| 1.1 clean tree | `git status --porcelain` empty | empty | PASS |
| 1.1 branch | `main` | `main` | PASS |
| 1.1 `EXECUTION_COMMIT` | accepted-plan HEAD | `e205b9acc5d0400aa38169de721acb7fb8268f29` | PASS |
| 1.1 D3 ancestry | `10133f6` is an ancestor | `git merge-base --is-ancestor` exit 0 | PASS |
| 1.2 full suite | exit 0, zero failures/errors, ≤ 1 historical skip | **1597 passed, 1 skipped**, 54.16 s, exit 0 | PASS |
| 1.2 focused subset | exit 0, all 15 files collected | **332 passed**, 10.19 s, exit 0 | PASS |
| 1.3 dry run | exit 0, identity + both runs + 4 paths + dry-run line | all present verbatim; exit 0 | PASS |
| 1.3 no side effect | placeholder dir absent afterward | `Test-Path` → `False` | PASS |
| 1.4 contract blob id | `805faa5cdb94618538c60d5afdd715fec84ac608` | identical | PASS |
| 1.4 contract blob size | `11920` | `11920` | PASS |
| 1.4 contract unmodified | `git diff --quiet` exit 0 | exit 0 | PASS |
| 1.4 on-disk SHA-256 | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` (12160 B) | identical, 12160 B | PASS |
| 1.4 LF-normalized SHA-256 | `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715` (11920 B) | identical, 11920 B | PASS |
| 1.4 feature-config SHA-256 | `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` | matches the on-disk form | PASS |
| 1.4 §2.3 parameter sweep | every frozen value | all matched (table below) | PASS |
| 1.5 manifest identity | `e2c1f8fd44d72176` / `20260724T045049097520Z_40b16886` | identical | PASS |
| 1.5 lineage identity | `status=complete`, same snapshot/build, `repo_sha=131d0ac0…` | identical | PASS |
| 1.6 lineage receipt digest | `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` | identical | PASS |
| 1.6 input presence / location | all 7 present, none under `C:/MomentumCVG_env/cache` | confirmed | PASS |
| 1.7 `C:/MomentumCVG_env/runs` exists | must exist | exists, created `2026-08-23T02:00:00Z` | PASS |
| 1.7 parent writable | must be writable | four Phase 1 log files written into it this session | PASS |
| 1.7 `RUN_DIR` fresh and legal | absent, outside repo and mutable cache | `Test-Path` → `False`; both containment tests `False` | PASS |
| 1.8 gate | no failure in 1.1–1.7 | none | **PASS** |

### 1.4 frozen-parameter sweep (§2.3), all confirmed verbatim

`max_lag=42`, `min_lag=8`, `window_size=35`, `search=false`; feature columns `mom_42_8_mean` / `cvg_42_8` /
`mom_42_8_count` / `cvg_count_42_8`; `min_count_pct=0.8` with `derived_required_count=28` and joint columns
`["mom_42_8_count","cvg_count_42_8"]`; `long_top_pct=0.1`, `short_bottom_pct=0.1`, `cvg_filter_pct=0.5`,
`max_names_per_side=25`, cap tie-break ticker ascending; `dvol_top_pct=0.2`, `spread_bottom_pct=1.0`,
`earnings_exclusion_days=0`; `short_structure=ironfly`, `wing_delta_target=0.15`, `max_leg_spread_pct=0.5`,
`max_spread_cost_ratio=null`; `sizing_mode=conceptual`, `tier_a_mode=equal_max_loss`,
`tier_a_short_budget=10000.0`, `tier_a_long_budget=10000.0` (`fallback_only`), `contract_multiplier=100.0`,
`tier_b_short_max_loss_budget=null`, `deployable_capital=null`; legacy pins
`max_loss_budget_per_trade=500.0`, `include_diagnostics=true`, `wing_selection_rule="closest_delta"`;
`periods` `2018-10-26`…`2026-07-10` and `2020-01-01`…`2026-07-10`; exactly two `runs[]` with fill alphas
`(0.5,0.5)` and `(1.0,1.0)` and exactly one `primary_decision_view: true` (cross);
`accepted_inputs.earnings_path=null`; `accepted_inputs.mutable_cache_forbidden=true`.

### 1.6 input digest baseline (pre-run, PG-2)

| SHA-256 | Bytes | Input |
|---------|-------|-------|
| `f34fb2556da03e9113f4a56a23e4e7dff2296810d5c848e24ff251678991b7bc` | 5865941 | `derived/e2c1f8fd44d72176/features/features_42_8.parquet` |
| `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` | 70409 | `derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json` |
| `6737ab2073be4aab874454faf849139031bf66031e80ffc81b712ac2edff2f2c` | 808256 | `derived/e2c1f8fd44d72176/features_quality_audit_v1.json` |
| `304753a2d5ce9900bdf462442f4f11407c8ec821ec5708ef9190027b4b3b7c4a` | 16159021 | `…/cache/surface/option_surface_meta_weekly_2018_2026.parquet` (A1) |
| `e8b2b49094362fde3432b2851c47c72004a539db6c37f9a4fbda6f2e6d907ca4` | 261380171 | `…/cache/surface/option_surface_quotes_weekly_2018_2026.parquet` (A2) |
| `756d78160047554b3c158e99aa24e337be933de9b47f273f21dce35b85d07d42` | 25357630 | `…/input/liquidity/ticker_liquidity_panel.parquet` |
| `e312fd1932ca2a95b104f1c5b52bb6054270695f23c2670cdf125c10f379e1ab` | 6955 | `…/manifests/input_snapshot_e2c1f8fd44d72176.json` |

Phase 4 must reproduce these seven digests byte-for-byte.

---

## 2. Directories

All outside the repository. Nothing below was copied into Git.

| Role | Absolute path | State |
|------|---------------|-------|
| Phase 3 `RUN_DIR` (derived at §1.7) | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T031859Z` | **not created — Phase 3 not started** |
| Smoke `SMOKE_DIR` | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T031947Z` | created fresh this session |
| Smoke adapter output `SMOKE_OUT` | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T031947Z/run` | 17 adapter files |
| Smoke contract | `C:/MomentumCVG_env/runs/sprint006_d4_smoke_20260823T031947Z/sprint006_smoke_contract.json` | Option A copy |
| Phase 1 logs | `C:/MomentumCVG_env/runs/d4_phase1_{full_suite,focused_suite,dryrun}.txt`, `d4_phase1_input_digests.tsv` | retained |
| Phase 2 logs | `…/d4_phase2_date_derivation.txt`; `$SMOKE_DIR/{smoke_stdout.txt,d4_smoke_checks.py,d4_smoke_checks_output.txt}` | retained |

Pre-existing sibling directories at §1.7 (all from the earlier, unaccepted attempts):
`sprint006_d4_phase12_20260823T020131Z`, `sprint006_d4_phase12_20260823T021859Z`,
`sprint006_d4_smoke_20260823T020000Z`, `sprint006_d4_smoke_20260823T021540Z`. None was reused or cited.

---

## 3. Phase 2 — smoke date derivation (§2.1)

Derived read-only from the A1 expected calendar, not hardcoded:

```
n_expected_dates 403
lower_median_index 201 2022-09-02
upper_median_index 201 2022-09-02
median_index 201 MEDIAN_DATE 2022-09-02
next_index 202 SMOKE_END_DATE 2022-09-09
first 2018-10-26 last 2026-07-10
```

`n = 403` is odd, so the lower and upper median indices coincide and the PG-1a convention is not load-bearing
here. `MEDIAN_DATE = 2022-09-02` and `SMOKE_END_DATE = 2022-09-09` both equal their required values, and the
first/last calendar dates equal the frozen run window. **PASS.**

---

## 4. Phase 2 — smoke contract (§§2.2–2.3)

| Item | Value |
|------|-------|
| Smoke contract SHA-256 | `2f7110bbff57c680830a949f082b1b4d46458d363f670f073990af7edcf97801` |
| Frozen contract SHA-256 after the copy | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` (unchanged) |
| `git status --porcelain configs/sprint006_baseline_v1.json` | empty |
| `Compare-Object` delta entries | **8** (four changed lines) |

The divergence between the smoke digest and both frozen digests is expected by design (§2.2) and is the signal
that these artifacts are smoke, not baseline.

Exactly four date fields changed, and nothing else:

| Field | Frozen | Smoke |
|-------|--------|-------|
| `periods.run_start_date` | `2018-10-26` | `2022-09-02` |
| `periods.run_end_date` | `2026-07-10` | `2022-09-09` |
| `shared_run_config.start_date` | `2018-10-26` | `2022-09-02` |
| `shared_run_config.end_date` | `2026-07-10` | `2022-09-09` |

`primary_reporting_start_date` / `primary_reporting_end_date` were not touched. The second date is present
solely because `BacktestRunConfig` enforces `start_date < end_date`; the validator was not modified and no
single-date CLI support was added.

---

## 5. Phase 2 — smoke checks S-1…S-10

Executed `scripts/run_sprint006_baseline.py --contract $SMOKE_CONTRACT --output-dir $SMOKE_OUT`
(start `2026-08-23T03:20:02Z`, end `2026-08-23T03:21:07Z`, ~65 s). Both frozen fills ran in the one invocation.

| # | Verdict | Evidence |
|---|---------|----------|
| S-1 | PASS | Exit code 0. No `ERROR` or `Traceback` in stdout. Stdout lists 14 per-run file lines (7 × 2 runs) plus 3 report/receipt lines; both runs reported `trade_log_rows=76` |
| S-2 | PASS | 17 adapter-written files in `$SMOKE_OUT`, no missing, no extras. Counted **before** the §2.5 marker was written; the marker and smoke contract live one level up in `$SMOKE_DIR` and are excluded |
| S-3 | PASS | Exact column lists for both fills: `date_status` = 3 `["trade_date","status","reason"]`; `funnel_summary` = 18 = `FUNNEL_SUMMARY_COLUMNS`; `leg_log` = 21 = `LEG_LOG_COLUMNS`; `candidate_view` = 9 = `CANDIDATE_VIEW_COLUMNS` |
| S-4/mid | PASS | Exactly 2 rows; dates `[2022-09-02, 2022-09-09]`; statuses `['traded','traded']`; `reason` null iff `traded` |
| S-4/cross | PASS | Exactly 2 rows; dates `[2022-09-02, 2022-09-09]`; statuses `['traded','traded']`; `reason` null iff `traded` |
| S-5/mid | PASS | Both dates present. Monotone chain `n_included ≤ n_constructable ≤ n_post_signal ≤ n_jointly_eligible ≤ n_universe`; long+short splits sum to totals; null-vs-zero respected |
| S-5/cross | PASS | Same, both dates |
| S-6/mid | PASS | `2022-09-02`: 31 `structure_ok` rows, 86 leg rows. `2022-09-09`: 32 `structure_ok` rows, 90 leg rows. Straddles have `leg_index {0,1}`, iron flies `{0,1,2,3}` with `+ − − +` `unit_quantity` signs; `portfolio_quantity = abs(quantity) × unit_quantity`; non-included constructable legs have null `portfolio_quantity` and `pnl_total_leg` |
| S-6/cross | PASS | Identical structural counts and rules |
| S-7/mid | PASS | Included trades per date: `2022-09-02` → 31, `2022-09-09` → 32. All four leg-sum identities hold within `max(1e-6 abs, 1e-8 rel)` on both dates |
| S-7/cross | PASS | Same |
| S-8/mid | PASS | `structure_failed` rows: `2022-09-02` → 5, `2022-09-09` → 8. Reason codes drawn only from `{missing_quotes_or_body, wing_or_liquidity_selection}` ⊂ the four frozen classes; every row retains `reason_raw`; none has leg rows |
| S-8/cross | PASS | Same |
| S-9 | PASS | Evaluated on `MEDIAN_DATE` only. 31 overlapping constructable `(ticker, direction)` keys; sampled `ACN/long`, 2 legs. `mid` column identical across fills, `bid`/`ask` identical; both buy legs satisfy `cross − mid = +0.5 × (ask − bid)` within tolerance, and both had `ask > bid` so the strict inequality also held. No zero-spread leg arose. `SMOKE_END_DATE` was **not** substituted |
| S-10 | PASS | Attested below |

**S-10 attestation.** `decision_report.json` and `decision_report.md` were never opened. `run_summary_*.json`
was never read — the two files appear only as filenames in the S-2 directory listing. `run_receipt.json` was
read for identity keys only (`experiment_id`, `deliverable`, `repo_sha`, `contract.contract_id`,
`contract.sha256`, `len(runs)`), which is the key-only walk the plan permits.

Precisely what was and was not touched at the P&L level:

* Per-leg and per-trade P&L fields — `entry_cash_per_unit`, `expiry_payoff_per_unit`, `pnl_per_unit`,
  `pnl_total_leg`, `entry_cost_per_share`, `pnl_per_share`, `pnl_total` — **were** machine-read from the
  `leg_log` and `trade_log` parquets and summed, solely to evaluate the S-7 reconciliation identities that the
  plan requires. S-7 cannot be performed without reading them.
* **No individual P&L value was reported or economically interpreted.** The comparisons were pass/fail against
  a tolerance; no per-trade or per-leg figure was printed to the checkpoint, quoted, ranked, or read for
  economic meaning, and none appears anywhere in this document.
* **No aggregate P&L, return, Sharpe, drawdown, yearly result, or concentration metric was opened, computed,
  or reported** at any point.

The counts above are structural row counts, which §5 explicitly permits.

### Smoke receipt identity (identity fields only)

| Field | Value | Note |
|-------|-------|------|
| `experiment_id` | `sprint006_baseline_v1` | frozen |
| `repo_sha` | `e205b9acc5d0400aa38169de721acb7fb8268f29` | equals `EXECUTION_COMMIT` |
| `contract.contract_id` | `sprint006_baseline_v1` | frozen (Option A hazard, contained) |
| `contract.sha256` | `2f7110bbff57c680830a949f082b1b4d46458d363f670f073990af7edcf97801` | smoke copy, differs from frozen **by design** |
| `runs` | 2 | mid + cross |
| `deliverable` | `sprint006_d3` | expected; D3 producer metadata, see plan §7.2. Not a defect and not a reason to change code |

### §2.5 containment

`$SMOKE_DIR` name contains `smoke` and is a sibling of, never inside, the derived `RUN_DIR`. The
`NOT_THE_OFFICIAL_BASELINE.txt` marker was written after the S-2 count and records the smoke contract path and
digest, both derived dates, the two-date rationale, and the prohibition on citing smoke figures. No smoke
artifact was copied into the repository or into any baseline directory.

---

## 6. Deviations recorded (none blocking)

1. **§1.7 writability evidence.** The plan does not prescribe a probe method. A dedicated write-probe file was
   declined by this environment's command review, so writability was instead evidenced by the four Phase 1 log
   files successfully written into `C:/MomentumCVG_env/runs` during this session. The criterion — the parent
   exists and is writable — is satisfied and directly observable.
2. **Marker cosmetic.** In `NOT_THE_OFFICIAL_BASELINE.txt` the smoke digest wraps onto the line after its
   `Smoke contract SHA-256:` label, a PowerShell array-concatenation artifact. The digest value is correct.
3. **Phase 1 log placement.** The Phase 1 transcripts were written directly into `C:/MomentumCVG_env/runs/`
   rather than a dedicated subdirectory. They are outside the repository and are named `d4_phase1_*`.

No stop condition was triggered, no retry occurred, and no code, test, configuration, or input was modified to
make any check pass.

---

## 7. Status

* **Overall verdict: `PASS`.** Phase 1 (§§1.1–1.8) and Phase 2 (§§2.1–2.5) completed with no failure.
* **Phase 3 was not started.** No official run directory exists; `RUN_DIR` was derived but never created.
* **No aggregate economics were opened or interpreted** at any point. Per-leg and per-trade P&L fields were
  machine-read and summed only for the S-7 reconciliation; see the S-10 attestation in §5.
* **No raw artifacts were copied into the repository.** Parquet files, JSON reports, receipts, contracts,
  transcripts, and run outputs all remain under `C:/MomentumCVG_env/runs/`; this Markdown checkpoint is the
  only artifact entering Git.
* These results **await human review**. Phase 3 requires separate explicit authorization; this checkpoint
  grants none, and D4 is not complete.
