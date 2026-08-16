# Sprint 006 D1 — Trusted baseline runner plan

**Status:** `PROPOSED — AWAITING ACCEPTANCE`
**Mode:** Build (design only until accepted; no D1 implementation authorized by this draft alone)
**Repo HEAD at design:** `a1b7a3ccf5cf984841cd0c00062e1207e3b494a0` (clean working tree on `main`)
**D0 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) (unchanged; SHA-256 of committed LF bytes `3cd57f4dc8cdf8a62af266e529459d88b4f729f369a5fb455fe84621aceef715`)
**D0 plan:** [`docs/tmp/sprint006_d0_baseline_experiment_contract_plan.md`](sprint006_d0_baseline_experiment_contract_plan.md) (`ACCEPTED — D0 COMPLETE`)
**Naming convention:** `docs/tmp/sprint00N_dN_*_plan.md`

---

## Review summary

**Recommended design:** Keep `SurfaceRunner.run_single_config()` as the only economic execution engine. Add a **thin frozen-contract adapter** that loads the accepted D0 JSON, resolves accepted snapshot/derived paths (never mutable cache defaults), builds the mid/cross twin `BacktestRunConfig`s, runs each through the existing single-config API, and writes the existing result objects plus a small run-identity receipt. Do not repair or redesign the search CLI.

**Main components reused:** `SurfaceRunner.run_single_config`, `SurfaceDataPaths` (with explicit path overrides), `BacktestRunConfig`, `pipeline` S1→S5, `option_surface.FillAssumption` builders/settle, `surface_metrics` date/run summaries, existing mid/cross unit/contract coverage, Sprint 005 identity-gate patterns (`require_clean_repo_sha` / digest helpers).

**Minimum changes believed necessary:**
1. Contract→paths→twin-config mapping + path/identity preflight (new adapter code).
2. Thin CLI entry that invokes that adapter for one documented command.
3. Persist existing `trade_log` / `date_summary` / `run_summary` plus a run receipt (config dump, digests, repo SHA).
4. Pin unstable per-side cap tie-break to `ticker` ascending (one focused `pipeline` sort change) — D0 D-20.
5. Focused tests proving mapping, path refusal of mutable defaults, fill-only pricing (no stacked `cost_model`), and tie-break.

**Proposed footprint:** ~1 new small library module, ~1 thin script, ~1 small production edit (`pipeline` cap sort), ~1–2 focused test modules. No new backtest engine, metrics framework, or search platform.

**Verification approach:** Synthetic/unit/contract tests only in D1. Optional read-only identity/path preflight against accepted artifacts. **No full-history real-data economic run and no inspection of new aggregate P&L in D1** (owned by D4).

**Explicitly deferred:** D2 joint Mom+CVG count / A1 date-status / all-leg spread; D3 decision report; D4 smoke + manual sample + full mid/cross execution; search-CLI repair; Sprint 007 study matrix.

**Needs your approval before implementation:**
1. Add a **new thin CLI** (`scripts/run_sprint006_baseline.py`) backed by an importable helper — rather than patching `run_surface_search.py` or relying on an ephemeral outside-repo driver.
2. Make the **cap tie-break** production change in D1 (narrow `sort_values` pin), not defer entirely to D2.
3. Write run artifacts under an **outside-repo** output root (e.g. `C:/MomentumCVG_env/runs/…`), never into Git or mutable producer cache as the accepted input root.

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

* Mid/cross builder economics (extend with an explicit “no second cost layer” regression if needed).
* Orchestration synthetic runner fixtures.
* Identity/digest helpers and clean-repo gate pattern from Sprint 005 tooling (adapt narrowly; do not import D4 feature-audit scope).

### Gaps that actually block a supported frozen baseline run

| Gap | Why it blocks D1 | Narrow fix |
|-----|------------------|------------|
| No frozen-contract entry point | Cannot reproduce D0 twin runs via one documented command; search CLI is wrong path and currently unconstructible | Thin adapter + thin CLI |
| Mutable-cache defaults | `SurfaceDataPaths()` silently points at forbidden mutable cache | Require explicit accepted paths from contract |
| No contract→`BacktestRunConfig` mapper | Manual reconstruction risks drift from frozen JSON | One mapper from D0 JSON → twin configs |
| No identity/output persistence on the single-config path | Runner returns in-memory only; D0 reproducibility needs digests + dumps + written outputs | Adapter writes existing frames + receipt |
| Cap tie-break unpinned | Equal `signal_rank_pct` selection order is not deterministic (`sort_values` without secondary key / stable kind) | Pin secondary `ticker` ascending in S5 cap sort |
| Fill/`cost_model` trust not encoded as a 006 regression | Behavior looks correct, but D0 explicitly requires D1 verification evidence | Focused tests / static assertions — not a new pricing system |

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
| Exercise `run_single_config` for frozen twin configs | Engine ready; no trusted launcher | Adapter builds configs and calls API |
| Accepted snapshot/derived paths only | Possible via overrides; defaults unsafe | Preflight + explicit `SurfaceDataPaths` |
| Identity/config recording | Missing on single-config path | Run receipt + effective config dump |
| Fill pricing correct; no stacked `cost_model` | Behavior appears correct | Evidence tests; no pricing redesign |
| Reproducible outputs (`trade_log`, `date_summary`, `run_summary`) | In-memory only | Persist from `SurfaceRunResult` |
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
SurfaceRunner.run_single_config(mid_cfg) / (..._cross)
        │
        ▼
SurfaceRunResult  →  write trade_log / date_summary / run_summary + run_receipt.json
```

No change to the inner date-loop economics except the S5 cap tie-break pin.

### 5.2 Entry point form

* **Importable helper module** (for tests and CLI), e.g. `src/backtest/sprint006_baseline.py` (name may be adjusted while coding; keep Sprint-006-specific to avoid a fake general framework).
* **Thin CLI** `scripts/run_sprint006_baseline.py`: contract path, output root, optional `--fill {mid,cross,both}`, optional `--dry-run` (identity/path/config only).
* CLI must refuse to proceed if required accepted inputs are missing or identity checks fail.
* Clean git HEAD: record always; **hard-fail when writing acceptance artifacts** (aligns with D0 `require_clean_git_head`).

### 5.3 Mapping rules (must be literal to D0 JSON)

* Shared fields from `shared_run_config` + feature window columns.
* Twin runs differ only by `run_id` and `FillAssumption` (mid vs cross); both keep inactive `cost_model="mid"`.
* Dates from contract ISO strings → `date`.
* `tier_b_*` / condor targets remain unset/`None` as frozen.
* Do **not** implement joint CVG count in the mapper; leave current mom-only pipeline behavior until D2.

### 5.4 Outputs (D1)

Per fill role under the chosen outside-repo run directory:

* `trade_log_<run_id>.parquet` (or `.csv` if parquet tooling is unnecessary — prefer parquet for consistency with search script)
* `date_summary_<run_id>.parquet`
* `run_summary_<run_id>.json`
* One `run_receipt.json` covering both (or per-run receipts): contract id/digest, repo HEAD, input path digests, effective config dump(s), output digests, command argv.

**Not in D1:** `date_status_table`, decision-metric pack, primary-period filtered report.

### 5.5 Cap tie-break

In `step5_select_and_size` per-side sort, after primary `signal_rank_pct` ordering, add secondary `ticker` ascending (and use a deterministic sort kind if needed). This is a reproducibility pin, not a strategy retune.

### 5.6 Fill verification (no engine redesign)

Prove with tests/evidence:

1. Surface assembly path receives only `config.fill`.
2. Changing `cost_model` while holding `fill` fixed does not change entry costs / PnL on a synthetic fixture (or equivalently: surface modules do not reference `cost_model`).
3. Mid vs cross on the same quotes produce the expected entry-cost / `spread_cost` relationship already encoded in unit tests.

---

## 6. Proposed file-level changes

| File | Change |
|------|--------|
| `src/backtest/sprint006_baseline.py` (**new**) | Load/verify contract; build paths; build twin configs; optional run+write helpers; receipt schema |
| `scripts/run_sprint006_baseline.py` (**new**) | Thin argparse CLI over the helper |
| `src/backtest/pipeline.py` | Cap selection sort: secondary `ticker` ascending |
| `tests/unit/test_sprint006_baseline_contract_adapter.py` (**new**, name flexible) | Mapping, digest, path refusal, receipt fields, dry-run |
| `tests/contract/test_step5_select_and_size_contract.py` or sibling | Equal-rank tie-break selects lower ticker |
| `tests/unit/` or `tests/contract/` fill/`cost_model` regression | No stacked cost; fill remains authoritative |
| **Do not edit** | `configs/sprint006_baseline_v1.json`, `surface_runner.py` loop (unless a blocking shared defect appears), `run_surface_search.py`, D2/D3/D4 docs/status |

Approximate effort: well inside the sprint’s 12–18h review trigger if scope stays as above.

---

## 7. Focused test and validation plan

1. **Contract digest:** loading committed JSON yields the pinned SHA-256 of LF bytes.
2. **Twin config constructibility:** both mid/cross `BacktestRunConfig`s validate; field values match D0; only `run_id` + fill alphas/labels differ.
3. **Path preflight:** missing/mismatched snapshot/build/receipt fails; constructing paths without explicit overrides is refused for the 006 entrypoint.
4. **Tie-break:** two same-side equal `signal_rank_pct` names → lower ticker kept when cap=1.
5. **Fill authority:** synthetic assembly/run shows mid vs cross friction; `cost_model` inactive.
6. **Output shape:** adapter writing from a synthetic `SurfaceRunResult` produces expected files + receipt keys.
7. **Regression subset after edits:** existing orchestration + step5 + option_surface straddle/ironfly mid/cross tests.

**Forbidden in D1 validation:** full-history real-data backtest; reading/reporting new aggregate Sharpe/P&L/rankings from accepted data.

Optional allowed preflight (no economic loop): open/verify file existence + digests for A1/A2/features/liquidity/manifest/receipt.

---

## 8. Ordered implementation steps

1. Accept this plan (and the three approval items in the Review summary).
2. Implement contract load/verify + twin config builder + path preflight (no runner execution yet).
3. Add tests for mapping/digest/path refusal.
4. Pin cap tie-break + test.
5. Add fill/`cost_model` regression coverage as needed.
6. Implement run+write wrapper calling existing `run_single_config` (synthetic-tested).
7. Add thin CLI.
8. Run focused pytest subset; record results in the implementation handoff (not this design doc).
9. Stop. Do not start D2/D3/D4 work or real-data P&L.

---

## 9. Acceptance criteria and stop conditions

### Design acceptance (this document)

- [ ] Review summary approved, including entry-point, tie-break-in-D1, and outside-repo output decisions
- [ ] Frozen D0 JSON remains untouched
- [ ] Plan does not authorize D2–D4 or search redesign
- [ ] No P&L-sensitive contract values reopened

### D1 implementation acceptance (after coding)

- [ ] One documented command can construct and (when executed) run the frozen mid and/or cross configs through `SurfaceRunner.run_single_config`
- [ ] Command uses only accepted snapshot/derived paths; mutable cache not used as input root
- [ ] Effective configs + input/code/output identities recorded in a run receipt
- [ ] Existing result artifacts persisted without a parallel economics schema
- [ ] Cap tie-break pinned and tested
- [ ] Fill-only pricing verified by focused tests (no stacked `cost_model`)
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
* Unrelated refactors or known-bug fixes that do not block frozen twin execution

---

## 11. Sprint 007 preservation note

D1 deliberately keeps **one config → one `SurfaceRunResult`** as the atomic unit. A later bounded study can call the same runner repeatedly and reuse the same result/receipt conventions. D1 must not pre-build ranking protocols, walk-forward search, or a second economic engine.

---

**End of proposed D1 design.** Implementation requires explicit acceptance of this plan (including the Review-summary approval items). No implementation or real-data P&L execution occurred while producing this document.
