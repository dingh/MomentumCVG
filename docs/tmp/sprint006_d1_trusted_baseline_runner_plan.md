# Sprint 006 D1 — Trusted baseline runner plan

**Status:** `IMPLEMENTED — AWAITING REVIEW` (design accepted; implementation recorded in §12)
**Mode:** Build (D1 implementation only; D2–D4 not authorized)
**Repo HEAD at design:** `a1b7a3ccf5cf984841cd0c00062e1207e3b494a0` (clean working tree on `main`)
**Implementation starting point (parent commit):** `b380d38325eda87036e3d6962a45dc3261ae7c21` (clean working tree on `main`)
**D1 implementation commit:** `241b0d313e237e9fcb90c60a1395888c529d2b48`
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (unchanged; SHA-256 of committed LF bytes `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715`)
**D0 plan:** [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](sprint006_d0_baseline_experiment_contract_plan.md) (`ACCEPTED — D0 COMPLETE`)
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md`

---

## Review summary

**Implementation status:** The accepted design below is **implemented and awaiting review** (§12 records files, tests, and limits). The adapter (`src/backtest/sprint006_baseline.py`) plus one thin CLI (`scripts/run_sprint006_baseline.py`) map the frozen contract onto `SurfaceRunner.run_single_config()`, which remains the only economic engine. The single production change outside the adapter is the S5 cap tie-break pin. The frozen D0 JSON is unchanged, and **no real-data economic run was executed and no new P&L was inspected**.

**Recommended design:** Keep `SurfaceRunner.run_single_config()` as the only economic execution engine. Add a **thin frozen-contract adapter** that loads the accepted D0 JSON, resolves accepted snapshot/derived paths (never the mutable producer cache root), maps only recognized contract fields into twin `BacktestRunConfig`s, and always runs **both** frozen contract fills—diagnostic mid and primary cross—through the existing single-config API. Persist existing result objects plus a small run-identity receipt; refuse to overwrite an existing run output directory or target artifacts. Do not repair or redesign the search CLI.

**Main components reused:** `SurfaceRunner.run_single_config`, `SurfaceDataPaths` (with explicit path overrides), `BacktestRunConfig`, `pipeline` S1→S5, `option_surface.FillAssumption` builders/settle, `surface_metrics` date/run summaries, existing mid/cross unit/contract coverage, Sprint 005 clean-repo / light digest patterns.

**Minimum changes believed necessary:**
1. Contract→paths→twin-config mapping + path/identity preflight (new adapter code); run set comes entirely from the frozen contract `runs` list.
2. Thin CLI that always executes mid **and** cross; optional `--dry-run` only for config/path/identity validation (no economic execution). No `--fill` selector.
3. Persist existing `trade_log` / `date_summary` / `run_summary` plus a run receipt; refuse overwrite of an existing run directory or target artifacts (no staging/atomic/resume system).
4. Pin unstable per-side cap tie-break to `ticker` ascending (one focused `pipeline` sort change) — D0 D-20.
5. Focused tests: recognized-field mapping; refuse mutable producer cache **root** (not every path containing `cache`); one behavioral fill-vs-inactive-`cost_model` economics test; tie-break.

**Identity (proportional):** Clean Git HEAD; contract `contract_id` / `contract_version` / `status`; record a basic digest. No CRLF portability machinery, new hashing framework, or extensive hashing tests.

**Proposed footprint:** ~1 new small library module, ~1 thin script, ~1 small production edit (`pipeline` cap sort), ~1–2 focused test modules. No new backtest engine, metrics framework, or search platform.

**Verification approach:** Synthetic/unit/contract tests only in D1. Optional read-only identity/path preflight against accepted artifacts. **No full-history real-data economic run and no inspection of new aggregate P&L in D1** (owned by D4).

**Explicitly deferred:** D2 joint Mom+CVG count / A1 date-status / all-leg spread; D3 decision report; D4 smoke + manual sample + full mid/cross execution; search-CLI repair; Sprint 007 study matrix.

**Fixed by this revision (not open):** Official command always runs mid and cross from the frozen contract; overwrite refusal; proportional identity/digest; recognized-field mapping; mutable-cache-root (not substring) refusal; behavioral fill/`cost_model` test.

**Needs your approval before implementation:**
1. Add a **new thin CLI** (`scripts/run_sprint006_baseline.py`) backed by an importable helper — rather than patching `run_surface_search.py` or relying on an ephemeral outside-repo driver.
2. Make the **cap tie-break** production change in D1 (narrow `sort_values` pin), not defer entirely to D2.
3. Write run artifacts under an **outside-repo** output root (e.g. `C:/MomentumCVG_env/runs/…`), never into Git or the mutable producer cache root as the accepted input root.

---

## 1. Context-read receipt

| Path | Why read | Fact used for D1 |
|------|----------|------------------|
| `docs/agenda/current_sprint.md` | D1 scope + architecture | Harden `run_single_config`; thin adapter only; search not the 006 path |
| `docs/tmp/sprint006_d0_…_plan.md` | Accepted D0 | Twin mid/cross; `fill` sole pricing; D1 owns runner + fill verify + tie-break pin |
| `configs/sprint006_baseline_v1.json` | Frozen contract | Exact fields, accepted paths, reproducibility requirements |
| `src/backtest/surface_runner.py` | Canonical executor | Date loop + S1→S5 + metrics; in-memory `SurfaceRunResult` only; feature-derived dates |
| `src/backtest/surface_run_config.py` | Path bundle | Defaults to mutable `C:\MomentumCVG_env\cache` — unsafe for 006 unless overridden |
| `src/backtest/run_config.py` | Constructible config | `sizing_mode` / Tier A required; `cost_model` schema-only |
| `src/backtest/pipeline.py` | Economics path | Uses `config.fill` only; cap sort has no secondary `ticker` key |
| `src/backtest/option_surface.py` | Pricing | Mid/cross via alphas; `spread_cost` is diagnostic delta vs mid, not a second deduction layer |
| `src/backtest/surface_metrics.py` | Existing outputs | `build_date_summary` / `summarize_trade_log` already produce run summaries |
| `src/backtest/surface_search.py` | Non-path | Multi-config search wrapper over `run_single_config` — not Sprint 006 acceptance |
| `scripts/run_surface_search.py` | Broken search CLI | Missing `sizing_mode`; illegal `SurfaceDataPaths(..., contract_multiplier=…)`; wrong defaults vs D0 |
| `docs/sprint_memos/sprint005_d5_…evidence.md` | Prior consumption | Accepted paths work; ephemeral outside-repo driver; not a frozen-contract runner |
| `docs/decisions/001_…md` | Canonical path | SurfaceRunner canonical; historical CLI pointer is search — superseded for 006 by agenda |
| `tests/unit/test_option_surface_{straddle,ironfly}.py` | Fill evidence | Mid/cross entry costs and `spread_cost` already tested at builder level |
| `tests/contract/test_orchestration_contract.py` | Runner wiring | Synthetic `run_single_config` orchestration already covered |
| `tests/contract/test_step5_select_and_size_contract.py` | Caps/sizing | Cap and Tier A behavior covered; **no equal-rank tie-break test** |
| `scripts/audit_feature_quality.py` | Identity gate pattern | Startup identity + clean-repo SHA pattern reusable conceptually |
| `src/data/snapshot_orchestrator.py` | Digests / HEAD | `sha256_file` / `current_repo_sha` helpers available |

**Git state at design:** HEAD `a1b7a3c`, branch `main` tracking `origin/main`, working tree clean. No code or P&L execution performed for this design.

---

## 2. Evidence-based current-state assessment

### Already satisfies D1 intent (leave unchanged)

* **Economic engine:** `SurfaceRunner.run_single_config()` is the complete single-config S1→S5→metrics path.
* **Fill pricing mechanics:** Builders take `config.fill`; settle uses assembly `entry_cost` from that fill; exit is intrinsic at `exit_spot`.
* **No stacked `cost_model` deduction today:** `cost_model` is validated in `BacktestRunConfig` and is **not read** by `pipeline` / `option_surface` / `surface_runner`. Search script already comments this intent.
* **Twin-fill capability:** Same config object with `FillAssumption.mid()` vs `.cross()` is already the supported differentiation.
* **Tier A `equal_max_loss`:** Implemented and contract-tested; D0 field values are constructible.
* **Accepted artifact consumability:** Sprint 005 D5 proved snapshot surfaces + derived `features_42_8` load through `SurfaceDataPaths` overrides.
* **Existing result model:** `SurfaceRunResult{config, trade_log, date_summary, run_summary}` is the correct handoff shape for later D3/D4/007 reuse.

### Reusable with validation / tests only

* Mid/cross builder economics (keep as regression; D1 still needs the behavioral fill/`cost_model` test).
* Orchestration synthetic runner fixtures.
* Clean-repo SHA pattern from Sprint 005 tooling (adapt narrowly; do not expand into a hashing design).

### Gaps that actually block a supported frozen baseline run

| Gap | Why it blocks D1 | Narrow fix |
|-----|------------------|------------|
| No frozen-contract entry point | Cannot reproduce D0 twin runs via one documented command; search CLI is wrong path and currently unconstructible | Thin adapter + thin CLI |
| Mutable-cache defaults | `SurfaceDataPaths()` silently points at forbidden mutable producer cache root | Require explicit accepted paths; refuse that cache **root** (snapshot `…/cache/surface` paths remain valid) |
| No contract→`BacktestRunConfig` mapper | Manual reconstruction risks drift from frozen JSON | One mapper: recognized fields only → twin configs |
| No identity/output persistence on the single-config path | Runner returns in-memory only; D0 reproducibility needs dumps + written outputs + light identity | Adapter writes existing frames + receipt; refuse overwrite |
| Cap tie-break unpinned | Equal `signal_rank_pct` selection order is not deterministic (`sort_values` without secondary key / stable kind) | Pin secondary `ticker` ascending in S5 cap sort |
| Fill/`cost_model` trust not encoded as a 006 regression | Behavior looks correct, but D0 explicitly requires D1 verification evidence | One focused **behavioral** economics test — not static source inspection alone |

### Non-gaps for D1 (real, but owned later)

* Feature-derived trade dates + empty-signal `continue` → silent date loss (**D2**).
* Mom-only count eligibility (**D2**).
* Iron-fly body unfiltered by `max_leg_spread_pct` (**D2**).
* Dual calendar/conditional decision report (**D3**).
* Full-history real-data mid/cross economic execution (**D4**).

---

## 3. D1 requirements → capability or gap

| D1 requirement (agenda + D0 handoff) | Status | Action |
|--------------------------------------|--------|--------|
| Exercise `run_single_config` for frozen twin configs | Engine ready; no trusted launcher | Adapter builds **both** mid and cross from contract `runs` and calls API |
| Accepted snapshot/derived paths only | Possible via overrides; defaults unsafe | Preflight + explicit `SurfaceDataPaths`; refuse mutable producer cache root |
| Identity/config recording | Missing on single-config path | Run receipt (clean HEAD, contract id/version/status, basic digest) + effective config dump |
| Fill pricing correct; no stacked `cost_model` | Behavior appears correct | One behavioral economics test; no pricing redesign |
| Reproducible outputs (`trade_log`, `date_summary`, `run_summary`) | In-memory only | Persist from `SurfaceRunResult`; refuse overwrite |
| Cap tie-break `ticker` asc | Unpinned | One pipeline sort pin + test |
| Preserve single-config result model for later studies | Present | Do not invent parallel result schema |
| Search path repair | Not required for fixed-contract execution | **Out of scope** |
| `date_status_table` / joint eligibility / report pack | D0 lists outputs; implementation tagged D2/D3 | **Do not build in D1** (receipt may note deferred) |

---

## 4. Alternatives considered

| Option | Verdict |
|--------|---------|
| **A. Thin frozen-contract adapter + new thin CLI** (recommended) | Narrowest way to get one documented command without touching search or the engine loop |
| B. Fix/extend `scripts/run_surface_search.py` for baseline | Rejected: search semantics, wrong defaults, broken kwargs, invites Sprint 007 scope |
| C. Library-only / notebook / ephemeral ops_logs driver (D5 style) | Rejected as sole solution: fails “one documented command” and reproducibility in-repo |
| D. New backtest engine or generalized experiment framework | Rejected by agenda / stop rules |
| E. Change `SurfaceRunner` date loop / eligibility now | Rejected: D2 ownership; not required to launch twin configs |
| F. Defer tie-break entirely to D2 | Weaker for D1 “trusted reproducible” claim; D0 already assigns pin to D1/D2 — recommend doing the one-line pin in D1 |

---

## 5. Proposed design (technical)

### 5.1 Execution architecture

```text
configs/sprint006_baseline_v1.json
        │
        ▼
[thin adapter]  load → verify digest/status → resolve paths → build mid/cross BacktestRunConfig
        │
        ▼
SurfaceDataPaths(explicit accepted paths only; earnings_path=None)
        │
        ▼
SurfaceRunner.run_single_config(mid_cfg) then (..._cross)  # always both
        │
        ▼
SurfaceRunResult ×2  →  write trade_log / date_summary / run_summary + run_receipt.json
                        (refuse if run dir or targets already exist)
```

No change to the inner date-loop economics except the S5 cap tie-break pin.

### 5.2 Entry point form

* **Importable helper module** (for tests and CLI), e.g. `src/backtest/sprint006_baseline.py` (name may be adjusted while coding; keep Sprint-006-specific to avoid a fake general framework).
* **Thin CLI** `scripts/run_sprint006_baseline.py`: contract path, output root; optional `--dry-run` (config/path/identity only — no economic execution). **No** `--fill` option: the official command always executes both frozen contract runs (diagnostic mid and primary cross) from the contract `runs` list.
* CLI must refuse to proceed if required accepted inputs are missing or identity checks fail.
* Clean Git HEAD: **hard-fail when writing acceptance artifacts** (aligns with D0 `require_clean_git_head`). Record contract id/version/status and a basic digest — no CRLF portability or hashing framework work.

### 5.3 Mapping rules (must be literal to D0 JSON)

* Map only **recognized** `BacktestRunConfig` fields from `shared_run_config`, feature window columns, and each `runs[]` entry. Do **not** blindly unpack note/intent/summary prose fields into the dataclass.
* Twin runs differ only by `run_id` and `FillAssumption` (mid vs cross); both keep inactive `cost_model="mid"`.
* Dates from contract ISO strings → `date`.
* `tier_b_*` / condor targets remain unset/`None` as frozen.
* Do **not** implement joint CVG count in the mapper; leave current mom-only pipeline behavior until D2.

### 5.4 Outputs (D1)

Per fill role under the chosen outside-repo run directory (always both mid and cross on a non-dry-run):

* `trade_log_<run_id>.parquet` (or `.csv` if parquet tooling is unnecessary — prefer parquet for consistency with search script)
* `date_summary_<run_id>.parquet`
* `run_summary_<run_id>.json`
* One `run_receipt.json` covering both: contract id/version/status, basic digest, repo HEAD, effective config dump(s), light input/output identity, command argv.

**Overwrite safety:** Refuse if the chosen run output directory already exists or any target artifact path already exists. No staging directories, atomic publication, resumability, lifecycle states, or generalized run-management system.

**Not in D1:** `date_status_table`, decision-metric pack, primary-period filtered report.

### 5.5 Cap tie-break

In `step5_select_and_size` per-side sort, after primary `signal_rank_pct` ordering, add secondary `ticker` ascending (and use a deterministic sort kind if needed). This is a reproducibility pin, not a strategy retune.

### 5.6 Fill verification (no engine redesign)

Require **one focused behavioral test** on a synthetic fixture: `fill` controls pricing/economics, and changing inactive `cost_model` while holding `fill` fixed does **not** change economics. Static source inspection alone is not acceptance evidence. Existing mid/cross builder unit tests remain useful regression coverage but do not replace that behavioral check.

---

## 6. Proposed file-level changes

| File | Change |
|------|--------|
| `src/backtest/sprint006_baseline.py` (**new**) | Load/verify contract; build paths; build twin configs; optional run+write helpers; receipt schema |
| `scripts/run_sprint006_baseline.py` (**new**) | Thin argparse CLI over the helper |
| `src/backtest/pipeline.py` | Cap selection sort: secondary `ticker` ascending |
| `tests/unit/test_sprint006_baseline_contract_adapter.py` (**new**, name flexible) | Recognized-field mapping, identity, mutable-cache-root refusal, overwrite refusal, receipt fields, dry-run |
| `tests/contract/test_step5_select_and_size_contract.py` or sibling | Equal-rank tie-break selects lower ticker |
| `tests/unit/` or `tests/contract/` fill/`cost_model` behavioral test | Fill controls economics; inactive `cost_model` does not |
| **Do not edit** | `configs/sprint006_baseline_v1.json`, `surface_runner.py` loop (unless a blocking shared defect appears), `run_surface_search.py`, D2/D3/D4 docs/status |

Approximate effort: well inside the sprint’s 12–18h review trigger if scope stays as above.

---

## 7. Focused test and validation plan

1. **Contract identity:** load committed JSON; check `contract_id` / `contract_version` / `status`; record a basic digest (no extensive hashing suite).
2. **Twin config constructibility:** both mid and cross `BacktestRunConfig`s validate from recognized fields only; only `run_id` + fill alphas/labels differ.
3. **Path preflight:** missing/mismatched accepted inputs fail; refuse the mutable producer cache **root** as an input root; do **not** reject legitimate snapshot paths under `snapshot/.../cache/surface`.
4. **Overwrite refusal:** existing run directory or target artifact → fail.
5. **Tie-break:** two same-side equal `signal_rank_pct` names → lower ticker kept when cap=1.
6. **Fill authority (behavioral):** fill controls economics; changing inactive `cost_model` does not.
7. **Output shape:** adapter writing from a synthetic `SurfaceRunResult` produces expected files + receipt keys for both runs.
8. **Regression subset after edits:** existing orchestration + step5 + option_surface straddle/ironfly mid/cross tests.

**Forbidden in D1 validation:** full-history real-data backtest; reading/reporting new aggregate Sharpe/P&L/rankings from accepted data.

Optional allowed preflight (no economic loop): open/verify file existence for A1/A2/features/liquidity/manifest/receipt.

---

## 8. Ordered implementation steps

1. Accept this plan (and the three approval items in the Review summary).
2. Implement contract load/verify + twin config builder + path preflight (no runner execution yet).
3. Add tests for recognized-field mapping, identity, mutable-cache-root refusal, and overwrite refusal.
4. Pin cap tie-break + test.
5. Add the focused behavioral fill/`cost_model` economics test.
6. Implement run+write wrapper calling existing `run_single_config` for **both** mid and cross (synthetic-tested).
7. Add thin CLI (always both runs; `--dry-run` optional).
8. Run focused pytest subset; record results in the implementation handoff (not this design doc).
9. Stop. Do not start D2/D3/D4 work or real-data P&L.

---

## 9. Acceptance criteria and stop conditions

### Design acceptance (this document)

- [ ] Review summary approved, including entry-point, tie-break-in-D1, and outside-repo output decisions
- [ ] Fixed decisions accepted: always mid **and** cross; overwrite refusal; proportional identity; recognized-field mapping; mutable-cache-root refusal; behavioral fill test
- [ ] Frozen D0 JSON remains untouched
- [ ] Plan does not authorize D2–D4 or search redesign
- [ ] No P&L-sensitive contract values reopened

### D1 implementation acceptance (after coding)

- [ ] One documented command constructs and (when executed) runs **both** frozen mid and cross configs through `SurfaceRunner.run_single_config`
- [ ] Command uses only accepted snapshot/derived paths; mutable producer cache **root** not used as input root
- [ ] Effective configs + proportional identity recorded in a run receipt; existing run dir/artifacts not overwritten
- [ ] Existing result artifacts persisted without a parallel economics schema
- [ ] Cap tie-break pinned and tested
- [ ] Fill-only pricing verified by a focused **behavioral** test (inactive `cost_model` does not change economics)
- [ ] Focused pytest subset green
- [ ] No new real-data aggregate P&L inspected; no parameter retune
- [ ] Search CLI left unrepaired unless a newly discovered **shared** defect blocks the adapter (unlikely given current evidence)

**Stop** when the above evidence exists. Pause for rescope if work expands into date-status taxonomy, joint eligibility, reporting packs, search frameworks, or approaches the sprint implementation review trigger without a clear remaining D1 blocker.

---

## 10. Explicit out of scope

* Editing or versioning the frozen D0 contract / P&L knobs
* Joint Mom+CVG count eligibility
* A1 expected-date calendar, date-status table, silent-date fixes
* All-leg `max_leg_spread_pct` completion for iron-fly bodies
* Decision-quality report / dual metric views / `robust_score` policy UI
* Full-history or primary-window real-data economic execution and manual trade audit (D4)
* Repair/redesign of `SurfaceSearch` / `run_surface_search.py`
* Tier B sizing, iron-condor comparison, earnings filters, new features
* Sprint 007 bounded study matrix or generalized multi-config experiment platform
* Staging/atomic/resume run-management systems; CRLF hashing frameworks; `--fill` selectors
* Unrelated refactors or known-bug fixes that do not block frozen twin execution

---

## 11. Sprint 007 preservation note

D1 deliberately keeps **one config → one `SurfaceRunResult`** as the atomic unit. A later bounded study can call the same runner repeatedly and reuse the same result/receipt conventions. D1 must not pre-build ranking protocols, walk-forward search, or a second economic engine.

---

## 12. D1 implementation record (`IMPLEMENTED — AWAITING REVIEW`)

Implemented in commit `241b0d3`, whose parent (and clean starting point) was `b380d38`.

### Files

| File | Change |
|------|--------|
| `src/backtest/sprint006_baseline.py` (new) | Contract load/identity, recognized-field mapping → twin configs, accepted-path preflight, output writing, light receipt, `run_baseline` over `run_single_config` |
| `scripts/run_sprint006_baseline.py` (new) | Thin CLI: `--contract`, `--output-dir`, `--dry-run`. No `--fill`; both frozen runs always execute |
| `src/backtest/pipeline.py` | S5 per-side cap sort: secondary key `ticker` ascending (+ docstring) |
| `tests/unit/test_sprint006_baseline_adapter.py` (new) | Identity, mapping, path behavior, dry-run, overwrite refusal, synthetic end-to-end, behavioral fill/`cost_model` |
| `tests/contract/test_step5_select_and_size_contract.py` | Added equal-rank tie-break test |

Unchanged as intended: `configs/sprint006_baseline_v1.json`, `surface_runner.py`, `option_surface.py`, `surface_metrics.py`, `surface_search.py`, `run_surface_search.py`.

### Tests run

| Command | Result |
|---------|--------|
| `pytest tests/unit/test_sprint006_baseline_adapter.py -q` | **33 passed** |
| `pytest tests/contract/test_step5_select_and_size_contract.py tests/contract/test_orchestration_contract.py tests/unit/test_surface_runner_data_flow.py tests/contract/test_run_metrics_contract.py tests/contract/test_run_envelope_contract.py tests/unit/test_option_surface_{straddle,ironfly,ironcondor}.py -q` | **238 passed** |
| `pytest -q` (full suite) | **1528 passed, 1 skipped** (skip pre-existing) |
| `scripts/run_sprint006_baseline.py --dry-run` against the frozen contract | exit 0; accepted paths resolved; no execution, nothing written |

The tie-break test is non-vacuous: the previous single-key sort selected the higher ticker for one input ordering.

### Review fix (follow-up commit)

`create_run_dir` now also refuses a run output directory inside the Git repository root or inside the mutable producer cache root `C:/MomentumCVG_env/cache`; the existing overwrite refusal is unchanged. Two focused tests added (`TestOutputLocationRefusal`); adapter suite re-run at **35 passed**.

### Notes for review

* **Contract digest is recorded, not compared.** The receipt stores the SHA-256 of the contract bytes on disk (`4012b4a4…` in a CRLF working copy); the D0 header value `3cd57f4d…` is the committed-LF digest. Per the accepted design, no line-ending normalisation machinery was added.
* **Clean HEAD** is required only when writing artifacts; `--dry-run` performs no identity write-gate and no execution.
* The CLI prints paths and row counts only — never economic metrics — so D1 execution cannot double as P&L inspection.
* Still deferred (also listed in each receipt): joint Mom+CVG count eligibility, A1 expected-date/date-status table, all-leg spread on iron-fly bodies (**D2**); decision report (**D3**); smoke, manual sample, full-history execution (**D4**).

---

**End of D1 design and implementation record.** No real-data economic backtest was executed and no new aggregate P&L was inspected.
