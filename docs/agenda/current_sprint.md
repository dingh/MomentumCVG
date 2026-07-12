# Current sprint — 004

**Updated:** 2026-07-11 (C6 closed; option-surface producer and audit accepted)  
**Status:** Active — **C1** + **C2** + **C4** + **C5** + **C6 closed**; **C3** deferred until C4–C8  
**Mode:** Build (HD decisions locked below)

**C4 closeout memo:** [sprint_memos/004_c4_liquidity_panel.md](../sprint_memos/004_c4_liquidity_panel.md)

**C5 closeout memo:** [sprint_memos/004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md)

**C6 closeout memo:** [sprint_memos/004_c6_option_surface.md](../sprint_memos/004_c6_option_surface.md)

**C1 receipt design (canonical):** [docs/tmp/c1_manifest_design_plan.md](../tmp/c1_manifest_design_plan.md)

**C4 liquidity panel design (canonical):** [docs/tmp/c4_liquidity_panel_design_plan.md](../tmp/c4_liquidity_panel_design_plan.md)

**C5 split-adjustment design (canonical):** [docs/tmp/c5_split_adjustment_design_plan.md](../tmp/c5_split_adjustment_design_plan.md)

**C6 option-surface design (canonical):** [docs/tmp/c6_1_option_surface_design_memo.md](../tmp/c6_1_option_surface_design_memo.md)

**Execution guardrails (read with this doc):** [sprint004_execution_guardrails.md](sprint004_execution_guardrails.md)

---

## Goal

Build the **trustworthy weekly input layer** for future real-data backtesting and trade-decision generation.

**Sprint question (the only question this sprint answers):**

> Can we refresh weekly input data, build/update a point-in-time universe, and trust that split-adjusted prices/strikes/option economics are internally consistent enough for downstream backtesting?

**This sprint does NOT answer:**

- Is the strategy profitable?
- Does Sharpe survive?
- What parameters are best?
- Is the strategy ready for paper/live trading?

---

## Real-data scope (HD locked)

| In Sprint 004 | Deferred |
|---------------|----------|
| **Input-layer** validation on real cache: splits, spot, liquidity panel, **option surface precompute (A1/A2)** | **Backtesting** (SurfaceRunner S1→S8, trade logs, Sharpe) → Sprint **006** |
| PIT universe checks on real panel | Tier B baseline / go-no-go → Sprint **007** |
| Split audit on real ORATS-adjusted chains | Parameter search, paper/live |

**Rule:** “Real-data validation” in 004 means **input artifacts and precompute correctness**, not strategy backtest runs.

---

## Source of truth

- [v1_universe_protocol.md](../v1_universe_protocol.md) — PIT top-20% universe rule (update at 004 closeout for rolling panel)
- [surface_engine_data_contract.md](../surface_engine_data_contract.md) — Stage A (A1–A4) + drift register
- [surface_engine_data_flow.md](../surface_engine_data_flow.md) — Stage A vs Stage B boundary
- [surface_engine_evaluation_plan.md](../surface_engine_evaluation_plan.md) — L4/L5 deferred to Sprint 006+

**Precedence:** Sprint 003 S5/S8/ORCH economics remain locked. Sprint 004 touches **Stage A input scripts** (splits, spot, liquidity, **option surface precompute**); **no** changes to `pipeline.py` S2+ signal logic, S5 sizing, S8 metrics, or ORCH.

**Previous sprint:** [sprint_memos/003_s5_s8_build.md](../sprint_memos/003_s5_s8_build.md) _(closed 2026-06-20)_.

> **Feature branch (A4):** All momentum/CVG / straddle history / `build_features` work → **Sprint 005 only**. Not wired, audited, or inventoried in 004 except a one-line “deferred to 005” note in `plan`/`validate`.

---

## Closeout blockers vs stretch / report-only

### Closeout blockers (required to close Sprint 004)

| # | Blocker |
|---|---------|
| 1 | CLI skeleton: **`plan`**, **`validate`**, **`split-audit`**, **`surface-audit`**, **`refresh --dry-run`** |
| 2 | **`--as-of`** on all subcommands → resolves to **last trading day ≤ `--as-of`** |
| 3 | **`snapshot_id`** (deterministic) + **`build_id`** (timestamped); manifest on executed runs |
| 4 | Input **inventory / validation report** (PASS / WARN / FAIL) for **A1, A2, A3** + splits + spot — **not A4** |
| 5 | Rolling **3-month liquidity panel**: **rebuild attempted**; PASS/WARN on success, or **explicit FAIL** if source data insufficient (§ Blocks Sprint 005) |
| 6 | **PIT universe validation** on sample dates (§ PIT universe criteria) |
| 7 | **Split unit/golden tests** on ≥1 multi-split fixture |
| 8 | **Split audit** script + ≥1 substantive documented run (scope configurable) |
| 9 | **`precompute_option_surface.py` audit + tests** (§ Option surface precompute spec) — **CLOSED** (C6, 2026-07-11) |
| 10 | **[v1_weekly_runbook.md](../v1_weekly_runbook.md)** written |
| 11 | **Full pytest suite green** |
| 12 | **No** strategy / S5 / S8 / ORCH logic changed |
| 13 | **CLI plan cleanup:** remove C2 provisional/deferred copy from `render_plan()` (`deferred to C3–C8`, bracket notes, `Provisional` header, C8 WARN stubs); update `test_refresh_weekly_inputs_cli.py` in same change (**C9**) |

### Stretch / report-only (not required for closeout)

| Item | Notes |
|------|-------|
| Full **`refresh`** runs every wired script without manual steps | Surface + core chain wired; feature branch absent by design |
| Full-cache split scan **perfectly clean** | Script + ≥1 run required; WARN OK |
| Full historical **surface backfill** beyond audit sample window | Sample + coverage report sufficient |
| Tagging every artifact with `snapshot_id` | Manifest path refs enough |
| **`precompute_option_surface.py` full-universe rebuild** via CLI | Audit + sample rebuild sufficient |
| Backtest / **S1→S8 smoke** | Sprint **006** |
| Watermark-scoped audits (audit only unseen dates/tickers) | Stretch; full behavior in **005** |
| Incremental append in every precompute script | **005**; 004 may batch by window |

---

## Validation severity (PASS / WARN / FAIL)

| Status | Meaning |
|--------|---------|
| **PASS** | Safe for downstream use at this check |
| **WARN** | Usable; limitation documented; does not block closeout unless `--strict` or blocking |
| **FAIL** | Blocks downstream real-data backtest input trust |

**Exit codes:** `0` = pass · `1` = blocking FAIL · `2` = usage/config error

---

## HD decisions (locked — 2026-06-21)

| ID | Topic | Decision |
|----|-------|----------|
| HD-004-0 | Real-data scope | **Input / liquidity / precompute** on real cache = **004**; **backtesting** = **006+** |
| HD-004-1 | Rolling panel | **Rebuild preferred.** If ORATS/cache lacks months needed for rolling window, report **FAIL** with documented gap — do not silently ship monthly-only panel as rolling |
| HD-004-2 | `--as-of` | **Last trading day ≤ `--as-of`** (align ORATS / `get_trading_fridays`) |
| HD-004-3 | Split audit scope | **Configurable** (`--sample-tickers`, date window); **≥1 substantive run** for closeout |
| — | `snapshot_id` | Deterministic **16-hex** hash of receipt identity only: `schema_version`, `as_of_resolved_trading_day`, `data_source`, `artifacts`, `params` — **not** a content fingerprint |
| — | `build_id` | Timestamped execution ID (`{UTC ts}_{6-hex}`); manifest has **both**; excluded from `snapshot_id` |
| — | Feature branch | **Entirely Sprint 005** — not in 004 CLI wiring |
| — | Stale doc conflicts | Update `backtest_evaluation_protocol.md`, memos, etc. **when that later sprint starts** — not in 004 |
| — | Rolling protocol doc | **Implement rolling in 004**; update [v1_universe_protocol.md](../v1_universe_protocol.md) **at 004 closeout** |
| — | Manifest layers | `input_snapshot_manifest` (004) ≠ `run_manifest` (007) |
| HD-004-4 | Weekly operator model | **`incremental` \| `backfill` \| `repair`** — routine refresh must not require full-history recompute; split events trigger **repair** scope; schema/column changes use **backfill** window |
| HD-004-5 | Incremental engines vs orchestration | **004** delivers CLI **`--mode`** + runbook contract for modes; **receipt schema stays minimal (C1)** — `refresh_mode` / watermarks / pipeline telemetry **not** in manifest; **full incremental append** in precompute scripts → **005**; 004 scripts may still run year/window batch |

---

## Sprint boundary: 004 input + surface precompute vs 005 features

| Topic | Sprint 004 | Sprint 005 |
|-------|------------|------------|
| Splits, spot, liquidity (A3) | Build, validate, rebuild rolling | — |
| Option surface precompute (A1/A2) | **Audit + test + validate on real cache** | Maintain; earnings_date if needed |
| Straddle history | **Out of scope** | Build + audit |
| `build_features` / mom / CVG (A4) | **Out of scope** | Audit + cleanup + paths |
| Incremental append / watermark-driven audits | CLI `--mode` + runbook; stretch audit scoping (watermarks **not** in C1 receipt) | Harden all Stage A append + feature tail recompute |
| Trade-date schedule (features ∩ surface) | Surface `entry_date` / Friday logic audited in 004 | Runner calendar + feature alignment |
| CLI pipeline | Core + surface only | Adds feature branch |

---

## Option surface precompute spec (closeout blocker)

**Scope:** `scripts/precompute_option_surface.py`, `src/features/option_surface_analyzer.py`, on-disk A1/A2 parquets.  
**Existing baseline:** `tests/contract/test_precompute_input_contract.py` (producer row schema, L1).

### A. Tests to add or extend

| # | Test | Location (proposed) | Requirement |
|---|------|---------------------|-------------|
| T1 | **A1 invariant:** `surface_valid ⇔ (has_body_call ∧ has_body_put ∧ n_surface_quotes > 0)` on producer rows | extend contract or `tests/unit/test_option_surface_analyzer.py` | Green on synthetic success/failure rows |
| T2 | **A1/A2 join:** every quote row’s `(ticker, entry_date)` has a meta row; valid quotes only when `surface_valid` | contract or unit | Green on fixture parquet |
| T3 | **Quote grain:** no duplicate `(ticker, entry_date, strike, side)` | audit test | FAIL if duplicates in sample |
| T4 | **`generate_trade_dates` / `--as-of` alignment:** resolved trading day matches meta `entry_date` set for that week | unit on `get_trading_fridays` + sample meta | Same resolution rule as HD-004-2 |
| T5 | **Hold-to-expiry fields:** for `surface_valid` rows, `exit_spot`, `expiry_date`, `body_strike`, `entry_spot` non-null; `dte_actual == (expiry − entry).days` | contract/audit | FAIL on valid rows with nulls |
| T6 | **Failure vocabulary:** `failure_reason` ∈ documented snake_case set when `surface_valid == False` | contract | WARN on unknown tags |
| T7 | **Real-cache sample:** load existing `option_surface_meta_*` / `option_surface_quotes_*` from cache; assert required columns + row counts > 0 for sample window | `tests/` or audit module | ≥1 window documented in sprint memo |

### B. Audit report (`surface-audit` subcommand)

Produces PASS/WARN/FAIL report (manifest-linked), minimum sections:

| Section | Checks |
|---------|--------|
| **Artifact inventory** | Meta + quotes paths exist; date range; ticker count |
| **Schema** | A1/A2 required columns present (data contract § A1/A2) |
| **Validity rate** | `% surface_valid` overall and by year (sample) |
| **Failure breakdown** | Top `failure_reason` counts |
| **Join integrity** | Orphan quotes; meta without quotes when `surface_valid` |
| **Settlement readiness** | `% valid rows with non-null `exit_spot` |
| **Date alignment** | `entry_date` ⊆ resolved trading Fridays in sample range |
| **v1 weekly DTE** | Flag rows where `dte_actual` far from 7 (weekly) — WARN only |

**Closeout:** T1–T6 green in pytest; T7 + audit report with **≥1 substantive sample run** (e.g. 1–3 tickers × 4–12 weeks, or HD-agreed window). Full-universe rebuild **not** required.

### C. CLI / refresh wiring (surface only)

```text
── Sprint 004 pipeline (no feature branch) ──
1. fetch_splits.py
2. apply_split_adjustment.py
3. extract_spot_prices.py
4. build_liquidity_panel.py              → 3-month rolling rebuild
5. precompute_option_surface.py          → optional on refresh; required for surface-audit sample if cache stale
6. validate / split-audit / surface-audit
```

`--skip-surface` allowed on `refresh`; `surface-audit` must run against existing or freshly built cache.

---

## Weekly operator model (incremental vs backfill vs repair)

**Intent:** When new weekly ORATS data arrives, the pipeline should **adjust and extend** inputs — not recompute entire history every run — while remaining correct after splits and schema changes.

This is the **target operator model** for `refresh_weekly_inputs`. Sprint 004 proves orchestration, manifest traceability, and audits; Sprint 005 hardens incremental engines for the feature branch and surface append.

### Three modes

| Mode | Trigger | What runs | 004 closeout |
|------|---------|-----------|--------------|
| **`incremental`** (default) | Routine weekly refresh `--as-of <Fri>` | New splits → adjust new ZIPs (skip-existing) → append spot → **recompute rolling liquidity window** through `as_of` → append surface rows for new dates (when not skipped) | CLI `--mode incremental`; operator/runbook only — **not** receipt `snapshot_id` fields |
| **`backfill`** | New column/feature, schema bump, history gap, first cache build | Batch recompute declared `--start-date` … `--end-date` (or year range) across wired steps | CLI `--mode backfill` + date window; operator/runbook only |
| **`repair`** | New split in period; golden FAIL; known bad ticker | Re-adjust affected tickers (`apply_split_adjustment --tickers …`) → downstream tail for scope | CLI `--mode repair` + `--sample-tickers`; split-audit documents scope |

```text
incremental  →  touch new dates + rolling liquidity window; audit since watermark (stretch)
repair       →  split-triggered historical correction; scoped audit
backfill     →  batch window for schema/feature migrations
```

### Step semantics (not all append-only)

| Step | Incremental | Repair | Backfill |
|------|-------------|--------|----------|
| `fetch_splits` | Append/update | Re-fetch if needed | Window N/A |
| `apply_split_adjustment` | New ZIPs only (skip existing) | **`--tickers`** re-adjust history | Year/ticker window |
| `extract_spot_prices` | Append new dates | Re-extract scoped dates | Date range |
| `build_liquidity_panel` | **Bounded rolling recompute** through `as_of` (HD-004-1) | Rebuild months touching repair dates | Full or window rebuild |
| `precompute_option_surface` | New `entry_date` rows (005 hardens append) | Recompute scoped ticker×dates | `--start-year` / `--end-year` window |
| Straddle / features / signals | — | — | **Sprint 005** (tail recompute for max_lag) |

### Watermarks and audit scope

Per-artifact **watermarks** (e.g. `max_entry_date`, `max_month_date`, `max_split_date`) are **deferred from the C1 receipt schema** (Sprint 005 incremental ops). Target behavior lives in CLI/runbook and audit reports:

- **Incremental:** audits prioritize coverage with `date > watermark` (stretch in 004; full in 005).
- **Repair:** golden tests + cache scan on repair-scoped tickers.
- **Backfill:** acceptance sample over declared date window.

**`snapshot_id`** is a **logical input-receipt ID** — hash of `schema_version`, `as_of_resolved_trading_day`, `data_source`, cache-relative `artifacts`, and identity-relevant `params` only. It is **not** a bytes-on-disk fingerprint. Content-level validation is handled by validation/audit reports (C3+) and may gain artifact fingerprints later if needed. Weekly runs normally differ by `as_of_resolved_trading_day`. Implementation: `src/data/input_snapshot.py`; design: [c1_manifest_design_plan.md](../tmp/c1_manifest_design_plan.md).

---

## CLI contract (concrete)

**Entrypoint:** `scripts/refresh_weekly_inputs.py`

### Commands

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py plan --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py validate --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py split-audit --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py surface-audit --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh --as-of 2026-06-26 --dry-run
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh --mode backfill --as-of 2026-06-26 --start-date 2024-01-01 --end-date 2024-12-31
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh --mode repair --as-of 2026-06-26 --sample-tickers NVDA
```

Optional: `--strict`, `--mode incremental|backfill|repair` (default **`incremental`** on `refresh`), `--skip-surface`, `--skip-splits`, `--sample-tickers`, `--start-date`, `--end-date`.

| Subcommand | Behavior |
|------------|----------|
| **`plan`** | Step order, inputs, outputs, **`--mode`**, skip flags, expected touch surface (append vs recompute vs repair); notes **feature branch → Sprint 005** |
| **`validate`** | Inventory for splits, spot, A3, **A1/A2**; PASS/WARN/FAIL; no A4 |
| **`split-audit`** | Wraps `audit_adjusted_liquid.py` (C8); standalone script **done (C5)** |
| **`surface-audit`** | § Option surface precompute spec report |
| **`refresh --dry-run`** | Planned execution only (includes mode + date window) |
| **`refresh`** | Runs wired core + surface steps per **`--mode`**; manifest + `build_id` |

---

## Default report and manifest paths

All paths under `C:/MomentumCVG_env/cache/` unless overridden by `--cache-dir`.

| Artifact | Default path | Written by |
|----------|--------------|------------|
| Input snapshot manifest | `manifests/input_snapshot_{snapshot_id}.json` | `refresh` (executed runs) |
| Latest manifest pointer | `manifests/latest_input_snapshot.json` | `refresh` (optional convenience symlink/copy) |
| Validation report | `manifests/reports/validate_{build_id}.md` | `validate` |
| Split audit report | `manifests/reports/split_audit_{build_id}.md` | `split-audit` |
| Surface audit report | `manifests/reports/surface_audit_{build_id}.md` | `surface-audit` |
| Combined closeout bundle | `manifests/reports/sprint004_closeout_{build_id}.md` | manual or CLI `--bundle-reports` (stretch) |

**Conventions:**

- **`snapshot_id`** — stable across re-runs with same receipt identity (see § Weekly input snapshot receipt).
- **`build_id`** — unique per CLI execution (`20260621T143022Z_a1b2c3` format); excluded from `snapshot_id`.
- Caller computes `snapshot_id` via `compute_snapshot_id()` **before** `write_manifest()`; write path does not recompute silently.
- Reports reference `snapshot_id`, `build_id`, and resolved `--as-of` trading day; git commit may appear in markdown headers (C3+) but **not** in receipt schema.
- JSON manifest is the canonical **input receipt**; markdown reports are human audit output; C3+ may patch `reports.*`, `overall_status`, `blocking_failures`, `notes` on an existing manifest.

---

## Weekly input snapshot receipt (C1)

**Principle:** The manifest is a **weekly input receipt**, not pipeline telemetry or a data-platform metadata layer.

**Answers one question:**

> What input snapshot did I use, for what as-of date, with which logical artifacts and key params?

**Module:** `src/data/input_snapshot.py` · **Design:** [c1_manifest_design_plan.md](../tmp/c1_manifest_design_plan.md)

### Canonical schema (`input_snapshot_{snapshot_id}.json`)

```json
{
  "schema_version": "1",
  "snapshot_id": "a1b2c3d4e5f67890",
  "build_id": "20260621T143022Z_a1b2c3",
  "created_at_utc": "2026-06-21T14:30:22Z",
  "as_of_requested": "2026-06-26",
  "as_of_resolved_trading_day": "2026-06-26",
  "data_source": "orats_adjusted_cache",
  "cache_dir": "C:/MomentumCVG_env/cache",
  "artifacts": {
    "splits": "splits_hist.parquet",
    "spot_prices": "spot_prices_adjusted.parquet",
    "liquidity_panel": "ticker_liquidity_panel.parquet",
    "option_surface_meta": "option_surface_meta_weekly_2018_2026.parquet",
    "option_surface_quotes": "option_surface_quotes_weekly_2018_2026.parquet"
  },
  "params": {
    "rolling_months": 3,
    "universe_rule": "top_20_pct_and_filter",
    "feature_branch": "deferred_to_sprint005"
  },
  "reports": {
    "validate": null,
    "split_audit": null,
    "surface_audit": null
  },
  "overall_status": null,
  "blocking_failures": [],
  "notes": []
}
```

### Field rules

| Field | In `snapshot_id` hash? | Notes |
|-------|------------------------|-------|
| `schema_version` | **Yes** | `"1"` for C1 |
| `as_of_resolved_trading_day` | **Yes** | ISO date; HD-004-2 resolved trading day |
| `data_source` | **Yes** | Logical source id, e.g. `"orats_adjusted_cache"` |
| `artifacts` | **Yes** | Fixed keys → cache-relative path strings (forward slashes); keys sorted when hashing |
| `params` | **Yes** | Keys sorted when hashing; identity-relevant params only |
| `snapshot_id` | No | Computed output (`sha256` canonical JSON, first **16** hex chars) |
| `build_id` | No | Per execution |
| `created_at_utc` | No | ISO-8601 UTC |
| `as_of_requested` | No | User/CLI input; resolved day is in hash |
| `cache_dir` | No | Where files live; artifact values are cache-relative |
| `reports` | No | Filled by C3+; paths to markdown reports |
| `overall_status` | No | C3+ aggregate (`PASS` / `WARN` / `FAIL`) |
| `blocking_failures` | No | C3+ |
| `notes` | No | Free text |

**Artifact keys (all required on complete receipts):** `splits`, `spot_prices`, `liquidity_panel`, `option_surface_meta`, `option_surface_quotes`.

**`params` (C1 minimum):**

| Key | Example | Purpose |
|-----|---------|---------|
| `rolling_months` | `3` | Liquidity panel rolling window (HD-004-1) |
| `universe_rule` | `"top_20_pct_and_filter"` | PIT universe rule reference |
| `feature_branch` | `"deferred_to_sprint005"` | Documents 004 scope boundary |

Additional `params` keys only if they change input identity (HD approval). Do **not** put `refresh_mode`, `processed_range`, `repair_scope`, watermarks, pipeline steps, or CLI telemetry in the receipt unless explicitly approved.

### `snapshot_id` algorithm

1. Identity dict contains **only** `schema_version`, `as_of_resolved_trading_day` (ISO), `data_source`, `artifacts`, `params`.
2. Canonical JSON: `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
3. `snapshot_id = sha256(utf8).hexdigest()[:16]`.
4. Artifact path backslashes normalize to forward slashes before hash.

**Explicitly excluded from hash:** `snapshot_id`, `build_id`, `created_at_utc`, `as_of_requested`, `cache_dir`, `reports`, `overall_status`, `blocking_failures`, `notes`, row counts, validation statuses, runtime info, pipeline steps, watermarks, lineage, refresh modes, content fingerprints.

### `build_id` algorithm

```
ts = now.utc().strftime("%Y%m%dT%H%M%SZ")
suffix = sha256(f"{ts}\0{command}".encode()).hexdigest()[:6]
build_id = f"{ts}_{suffix}"
```

### Deferred from receipt schema (not C1)

| Item | Defer to |
|------|----------|
| `git_commit`, `prior_build_id`, `prior_snapshot_id` | Optional top-level fields later if HD wants; excluded from hash |
| `watermarks`, `refresh_mode`, `processed_range`, `repair_scope` | CLI + runbook operator model; Sprint **005** incremental ops |
| `pipeline_steps`, `args_hash`, per-step telemetry | C8 CLI wiring / logs; optional future manifest v2 |
| `checks[]` array | Replaced by flat `reports` dict |
| Artifact per-key `status` | Validation in `overall_status` + markdown reports |
| `content_fingerprint` | Sprint **007** `run_manifest` |

---

## Blocks Sprint 005 if…

Sprint 005 (feature pipeline) must **not** start until these 004 outcomes are clear. If any row below is true with **FAIL** and no HD waiver, **hold 005** until resolved or explicitly descoped in sprint memo.

| # | Condition | Why it blocks 005 |
|---|-----------|-------------------|
| B1 | **Split adjustment** FAIL on golden fixture (scale jump, wrong cumulative factor) | Features and surfaces inherit bad prices/strikes |
| B2 | **Rolling liquidity panel** rebuild FAIL — insufficient ORATS months for 3-month window and no acceptable fallback documented | S1 universe untrusted; feature ranks run inside bad universe |
| B3 | **PIT universe** FAIL on sample dates (future `month_date` leak, non-reproducible universe) | Feature audit meaningless without trustworthy universe |
| B4 | **Option surface A1/A2** FAIL on settlement fields (`exit_spot` null on `surface_valid` rows) or broken meta/quotes join in sample | S3/S7 depend on surface; feature/surface date work in 005 assumes A1/A2 sane |
| B5 | **`--as-of` / trading-day resolution** inconsistent between liquidity scan, surface meta, and audit harness | 005 schedule alignment builds on 004 date rules |
| B6 | **No manifest / snapshot_id** — cannot reproduce which input snapshot 004 validated | 005 cannot attach feature builds to a known input baseline |
| B7 | **pytest regression** in contract or split/surface tests | Baseline trust broken |

**Does not block 005 (WARN or documented gap OK):**

- Surface validity rate low on sample tickers (WARN + coverage note)
- Split audit WARN on thin date subset
- Full `refresh` not run on entire history
- Missing A4 / features entirely (expected — 005 scope)
- `earnings_date` absent in Stage A

**Sprint 004 may still close** with WARN on non-blocking items if blockers B1–B7 are clear. Sprint **005 start** requires no unresolved B1–B7 FAIL.

---

## Implementation commit plan

Suggested **reviewable commit sequence** (agent commits only when user asks). Each commit should keep `pytest tests/ -q` green unless noted as test-add only.

| Commit | Scope | Files (expected) | Verification |
|--------|-------|------------------|--------------|
| **C1** ✓ | Manifest types + `snapshot_id` / `build_id` hashing | `src/data/input_snapshot.py`, `tests/unit/test_input_snapshot.py` | unit tests green |
| **C2** ✓ | CLI skeleton: `plan`, `--as-of` resolution, exit codes | `src/data/trading_day.py`, `scripts/refresh_weekly_inputs.py`, unit tests | `plan --as-of …`; pytest |
| **C4** ✓ | Rolling weekly PIT panel in `build_liquidity_panel.py` | script + `tests/unit/test_build_liquidity_panel.py` | panel rebuild or FAIL report |
| **C5** ✓ | Split golden tests + adjusted-liquid backfill + `audit_adjusted_liquid` | `split_adjuster`, `audit_adjusted_liquid.py`, `paths.py`, tests | pytest + full production audit PASS |
| **C6** ✓ | Surface tests T1–T6 + `surface-audit` | extend contract/unit, audit module | pytest + `surface-audit` sample run |
| **C7** | PIT universe harness (tests + audit module) | tests + CLI harness | sample dates in PIT report section |
| **C8** | `refresh --dry-run` + bounded `refresh` subprocess wiring | CLI | dry-run manifest shape |
| **C3** | `validate` + default report paths + umbrella inventory | CLI + report writer; wires C5/C6/C7 checks | `validate --as-of …` writes markdown |
| **C9** | Runbook + `v1_universe_protocol` + data-contract drift + **CLI plan output cleanup** (blocker #13) | `docs/` + `scripts/refresh_weekly_inputs.py` + `tests/unit/test_refresh_weekly_inputs_cli.py` | review; plan has no commit-label deferrals |
| **C10** | Sprint memo + progress log (closeout) | `docs/sprint_memos/004_*.md` | — |

**Rules:**

- No S5/S8/ORCH/strategy edits in any commit.
- No feature-branch wiring (`straddle_history`, `build_features`) in 004 commits.
- **HD (2026-06-24):** defer **C3** until **C4–C8** — `validate` is post-artifact; build component logic, audits, PIT harness, and `refresh` integration first; C3 then consolidates umbrella inventory + shared report conventions.
- **C4** may land in parallel with **C5** after C1–C2.
- **C5/C6/C7** each write their own markdown audit reports; C3 reuses those paths and aggregates into `validate_{build_id}.md`.

---

## Deliverables

| # | Artifact | Path | Closeout? |
|---|----------|------|-----------|
| 1 | Weekly input CLI | `scripts/refresh_weekly_inputs.py` | **Blocker** |
| 2 | Manifest module | `src/data/input_snapshot.py` (or adjacent) | **Blocker** |
| 3 | Validation report | CLI `validate` | **Blocker** |
| 4 | Rolling liquidity panel | `scripts/build_liquidity_panel.py` | **Blocker** (rebuild attempt; FAIL if source insufficient) |
| 5 | Split tests + adjusted-liquid audit | tests + `audit_adjusted_liquid.py` (CLI `split-audit` → C8) | **Done (C5)** |
| 6 | Surface precompute audit + tests | tests + `surface-audit`; extend contract | **Done (C6)** |
| 7 | PIT universe harness | tests + harness (C7); wired into `validate` in C3 | **Blocker** |
| 8 | Runbook | [v1_weekly_runbook.md](../v1_weekly_runbook.md) | **Blocker** |
| 9 | Universe protocol update | [v1_universe_protocol.md](../v1_universe_protocol.md) | **Blocker at closeout** |
| 10 | Drift register | [surface_engine_data_contract.md](../surface_engine_data_contract.md) | **Blocker at closeout** |
| 11 | Full `refresh` on production cache | CLI | Stretch |

---

## Work breakdown (build order)

| Phase | Work | Closeout exit |
|-------|------|---------------|
| **1** | `snapshot_id` / `build_id` + CLI skeleton + `--as-of` + **`--mode`** resolution | Commands run; `plan` shows incremental vs backfill vs repair |
| **2** | Rolling 3-month panel **rebuild attempt** (C4) | Panel on disk with PASS/WARN, or **FAIL** documented (insufficient source months) |
| **3** | Split unit/golden + adjusted-liquid audit (C5) ✓ | Production backfill audited; downstream defaults wired |
| **4** ✓ | Surface tests T1–T6 + `surface-audit` (C6) | ≥1 sample run; report archived |
| **5** | PIT universe harness (C7) | § PIT universe criteria |
| **6** | Bounded `refresh` + `--dry-run` (core + surface) (C8) | Wired; no feature steps |
| **7** | `validate` umbrella inventory (A1/A2/A3, splits, spot) (C3) | PASS/WARN/FAIL report |
| **8** | Runbook + universe protocol + drift + pytest (C9) | Docs + suite green |

---

## Split golden fixtures (acceptance)

≥1 multi-split fixture ticker:

| Check | Requirement |
|-------|-------------|
| Cumulative factor | Correct before / between / after split dates |
| Spot vs strike | Same scale after adjustment |
| Premium convention | Matches settlement unit convention |
| Payoff stability | No artificial 2× / 4× / 10× jump |
| Classification | PASS / WARN / FAIL in report |

Untrusted remainder documented in sprint memo; scale-jump → FAIL; coverage gap → WARN.

---

## PIT universe sample dates (acceptance)

| Sample | Purpose |
|--------|---------|
| Normal trade date | Baseline ranks |
| Near month boundary | Rolling / `month_date` edge |
| Missing/new ticker liquidity (if in cache) | Explicit WARN/FAIL |

Invariants: snapshot date < `trade_date` (strict prior snapshot — C7.2; same-day and future snapshots prohibited); rolling uses only data ≤ resolved snapshot (< `trade_date`); same inputs → same universe; missing liquidity never silent PASS.

---

## Acceptance criteria

### A. Required for closeout

- [ ] CLI: `plan`, `validate`, `split-audit`, `surface-audit`, `refresh --dry-run`, `--as-of` → last trading day, **`--mode`** on `refresh`
- [ ] `snapshot_id` + `build_id` in manifest
- [ ] Validation report for **A1/A2/A3** + splits + spot (**no A4**)
- [ ] Rolling panel: **rebuild attempted**; PASS/WARN, or **FAIL** with documented insufficient source data (does not fake rolling)
- [ ] PIT universe samples pass
- [x] Split golden tests + adjusted-liquid audit ≥1 substantive run (`audit_adjusted_liquid.py` on production root; CLI `split-audit` wiring → C8)
- [x] Surface spec T1–T7 + surface-audit ≥1 sample run (C6; memo [004_c6_option_surface.md](../sprint_memos/004_c6_option_surface.md))
- [ ] Runbook + **v1_universe_protocol** updated at closeout
- [ ] CLI `plan` output: no C2-era **provisional/deferred** scaffolding (blocker #13); tests updated
- [ ] pytest green; no S5/S8/ORCH changes
- [ ] No backtest / S1→S8 smoke

### B. Stretch

- [ ] Full `refresh` on full cache without manual steps
- [ ] Zero FAIL on full-cache split scan
- [ ] Full surface universe rebuild via CLI
- [ ] Per-file `snapshot_id` tags

---

## Out of scope (Sprint 004)

- **All feature work:** `precompute_straddle_history`, `build_features`, mom/CVG, A4 → **005**
- Backtest / S1→S8 smoke → **006**
- Tier B / Sharpe / go-no-go → **007**
- `run_surface_search` sizing → **006**
- KB-001, strategy logic, paper/live

---

## Forward sprints (004–008)

| Sprint | Theme | Gate |
|--------|-------|------|
| **004** | Input snapshot + split + PIT + **surface precompute audit** | Closeout blockers above |
| **005** | **All feature pipeline** (straddle history, features, mom/CVG, A4 trust, paths, schedule) | Absorbs gaps found in 004 |
| **006** | Real-data **backtest** smoke + `run_surface_search` wiring | Requires **004 + 005** trustworthy |
| **007** | Tier B conservative baseline | After L4 |
| **008** | Decision sprint | After baseline |

Stale docs (e.g. `backtest_evaluation_protocol.md` “Sprint 004–005 baseline”) → update when **007** starts.

---

## Plan conflicts / decisions (resolved)

| Old statement | Resolution | When to edit old doc |
|---------------|------------|----------------------|
| 003 memo: real-data validation in 004 | Input/precompute in **004**; backtest in **006** | Memo immutable; this doc governs |
| `backtest_evaluation_protocol.md`: thresholds Sprint 004–005 | Baseline **007** | Update at **007** start |
| `v1_universe_protocol.md`: monthly panel | Rolling in **004** | Update at **004** closeout |
| Prior draft: feature branch in 004 CLI | **Removed** — all features **005** | Done in v4 |

---

## Progress log

| Date | Notes |
|------|-------|
| 2026-06-21 v5 | Commit plan, manifest schema, default paths, Blocks 005, rolling FAIL semantics |
| 2026-06-21 v6 | Weekly operator model (incremental/backfill/repair); HD-004-4/5; manifest params + watermarks |
| 2026-06-21 v7 | C1 implemented; sprint doc aligned to [c1_manifest_design_plan.md](../tmp/c1_manifest_design_plan.md) — minimal weekly input receipt |
| 2026-06-21 v8 | C2 implemented: `trading_day.py`, `refresh_weekly_inputs.py` CLI (`plan`, `refresh --dry-run`, stub subcommands), exit-code contract; closeout blocker #13 for provisional plan copy |
| 2026-06-24 v11 | C4 input path: ORATS_Data raw ZIPs (default `--data-root`); liquidity before scoped split adjust |
| 2026-06-29 v12 | **C4 closed:** smoke tests PASS; full backfill 2017→2026-02-20 on `input/liquidity`; incremental fix + progress bar; memo [004_c4_liquidity_panel.md](../sprint_memos/004_c4_liquidity_panel.md) |
| 2026-07-04 v13 | **C5 closed:** scoped splits + filtered adjust → `input/adjusted_liquid` (2299 parquets); C5.10D audit PASS; C5.11A downstream defaults; memo [004_c5_adjusted_liquid.md](../sprint_memos/004_c5_adjusted_liquid.md) |
| 2026-07-11 v14 | **C6 closed:** three-layer A1/A2 trust gate (producer + contract + readiness + C6.4 real-cache/smoke evidence); blocker #9 closed; memo [004_c6_option_surface.md](../sprint_memos/004_c6_option_surface.md) |

---

## Verification commands (closeout)

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q

# C5 adjusted-liquid audit + regression (production root; standalone until C8 CLI wiring)
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_adjusted_liquid.py `
  --raw-root C:/ORATS/data/ORATS_Data `
  --adj-root C:/MomentumCVG_env/input/adjusted_liquid `
  --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet `
  --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv
& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_fetch_splits_cli.py tests/unit/test_apply_split_adjustment_cli.py tests/unit/test_split_adjuster.py tests/unit/test_split_adjuster_filtered_zip.py tests/unit/test_ticker_universe.py tests/unit/test_audit_adjusted_liquid.py tests/unit/test_adjusted_liquid_paths.py -q

& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py plan --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py validate --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py split-audit --as-of 2026-06-26  # stub → C8; use audit_adjusted_liquid.py above
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py surface-audit --as-of 2026-06-26
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh --as-of 2026-06-26 --dry-run
```

---

## Previous sprint

Sprint 003 — [sprint_memos/003_s5_s8_build.md](../sprint_memos/003_s5_s8_build.md).
