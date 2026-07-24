# C8.4 — Bounded evidence record

| Field | Value |
|-------|-------|
| Plan | [c8_4_bounded_evidence_plan.md](c8_4_bounded_evidence_plan.md) |
| Completion report | [c8_4_bounded_backfill_evidence.md](c8_4_bounded_backfill_evidence.md) |
| Closeout memo | [../sprint_memos/004_c8_4_bounded_evidence.md](../sprint_memos/004_c8_4_bounded_evidence.md) |
| Status | **Closed** — ACCEPT WITH LIMITATIONS (2026-07-22) |
| Started | 2026-07-21 (local session) |
| Closed | 2026-07-22 |
| Build ID | `20260721T062412533463Z_d42da59c` |
| Snapshot ID | `1b1e28b262ba40be` |
| Final root | `C:/MomentumCVG_env/snapshots_c8_4_test/20260721T062412533463Z_d42da59c` |

---

## Step 0a — Pin exact executable code

| Field | Result |
|-------|--------|
| **Verdict** | **PASS** |
| Executed | `git rev-parse HEAD`; `git status --porcelain=v1`; venv Python + CLI path check |
| Expected HEAD | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` |
| Observed HEAD | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` |
| HEAD match | yes |
| Tracked dirty files | **none** (no modified/staged/deleted tracked paths) |
| Branch | `main...origin/main` |
| Python | `C:\MomentumCVG_env\venv\Scripts\python.exe` (3.13.7) |
| Repo root | `C:\MomentumCVG` |
| CLI script | `C:\MomentumCVG\scripts\refresh_weekly_inputs.py` (exists) |

### Porcelain (`git status --porcelain=v1`)

Untracked only (all under `docs/tmp/`):

```text
?? docs/tmp/c1_manifest_design_plan.md
?? docs/tmp/c4_liquidity_panel_review.ipynb
?? docs/tmp/c5_6b_smoke_report.md
?? docs/tmp/c8_3b_resumable_cold_backfill_design.md
?? docs/tmp/c8_4_bounded_backfill_evidence.md
?? docs/tmp/c8_4_bounded_evidence_plan.md
```

### Pass/fail

PASS — HEAD pinned; tracked worktree clean; executable is the expected venv Python against this repo’s `scripts/refresh_weekly_inputs.py`.

---

## Step 0b — Full automated test baseline

| Field | Result |
|-------|--------|
| **Verdict** | **PASS** |
| Executed | `& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q` |
| Working directory | `C:\MomentumCVG` |
| Platform | win32, Python 3.13.7, pytest-9.0.2 |
| Collected | 1293 items |
| Result | **1292 passed, 1 skipped** |
| Duration | 26.11s |
| Exit code | 0 |
| Scope | Full `tests/` including all `tests/contract/*` (no selective ignores) |

### Pass/fail

PASS — full checkout regression suite green; safe to proceed to API/inventory preflight.

### Next

Step 1 — Core + splits API preflight.

---

## Step 1 — Core + splits API preflight

| Field | Result |
|-------|--------|
| **Verdict** | **PASS** (retry after token set in session) |
| Prior attempt | FAIL — token absent (see superseded notes below) |
| Executed | Live `OratsCoreClient.fetch_asset_type_at_date` + classify mapping; `OratsCorporateActionsFetcher(token=env).fetch_splits_for_ticker("AAPL")` |
| Token | Present in **session env only** (length 36). **Not** written to this record or committed |
| Core company (AAPL) | **PASS** — `2024-06-28`, `assetType=3` → `company_equity` (nonempty, first attempt) |
| Core non-company (SPY) | **PASS** — `2024-06-28`, `assetType=7` → `non_company_equity` (nonempty, first attempt) |
| Splits (AAPL) | **PASS** — 2 rows; columns `ticker`, `split_date`, `divisor` |
| Valid-empty treated as success? | **No** — both Core paths returned nonempty in-domain observations |

### Superseded first attempt (token missing)

Earlier Step 1 FAIL: process/User/Machine env unset; probes not run. Operator set session env and retries succeeded.

### Pass/fail

PASS — Core company and non-company Stage-1 paths retrieved and classified usable observations; splits authenticated and parseable.

---

## Step 2 — Raw inventory + resource forecast

| Field | Result |
|-------|--------|
| **Verdict** | **PASS** |
| Executed | `scan_raw_inventory(C:/ORATS/data/ORATS_Data, 2023-12-25, 2024-06-28)`; `shutil.disk_usage` on snapshots volume |
| `raw_dependency_start` | `2023-12-25` |
| `inventory_digest` | `d0a4f9f7a7e752e347dd5875235c4133bacacb6dd2391aae6517538b44db2133` |
| Physical date count | 128 (`2023-12-26` … `2024-06-28`) |
| Resolved date count | 128 (same bounds) |
| `as_of_resolved_trading_day` | `2024-06-28` (matches requested as-of) |
| Missing weekday diagnostic count | 6 (diagnostic only) |
| Archive byte sum | 8,775,215,077 (~8.17 GiB) |
| Verified-empty archives | 0 |
| Snapshots path | `C:\MomentumCVG_env\snapshots_c8_4_test` (created) |
| Disk free | **319.46 GiB** free of 930.91 GiB (≥ 200 GiB threshold: yes) |

### Runtime forecast (order-of-magnitude; accepted for proceeding)

| Stage | Band |
|---|---|
| Liquidity / Core | Dominant; empty stage-local dict; thousands of candidates × ≤5 attempts × 0.7 s → multi-hour possible |
| Splits API | Upper ref ~2783 liquid × 0.7 s ≈ ~32 min (window-specific liquid may differ) |
| Adjust / spot / surface | CPU/IO over 128 physical days × liquid universe; material but secondary to Core if API-bound |
| Disk headroom | ~8.2 GiB raw selected; snapshot outputs TBD; 319 GiB free accepted |

### Pass/fail

PASS — inventory nonempty, as-of resolved correctly, disk OK; forecast band accepted for Step 4 readiness (not a wall-clock guarantee).

---

## Step 3 — Dry-run contract

| Field | Result |
|-------|--------|
| **Verdict** | **PASS** |
| Executed | `refresh --dry-run --mode backfill --bounded-evidence` with target roots/dates/workers 8 |
| Exit code | 0 |
| Plan `scope` | `bounded` |
| Plan `production_accepted` | `false` |
| Banner | `DRY-RUN: no inventory scan, writes, lock, or producers executed` |

### Pass/fail

PASS — dry-run reports bounded non-production contract without executing producers.

### Next

Step 4 — bounded backfill execution (requires explicit GO; durable log tee). Product decision (classification-v2) still outstanding if not yet approved.

---

## Product decision

| Field | Result |
|-------|--------|
| Decision | Accept C8.4 evidence under **implemented classification-v2** |
| Operator | Approved 2026-07-21 (session) |
| Effect | Proceed with Step 4 using current code; no classifier change |

---

## Step 4 — Bounded backfill (failed)

| Field | Result |
|-------|--------|
| **Verdict** | **FAIL** (exit 1) |
| Product decision | classification-v2 **approved** before launch |
| Started UTC | `20260721T062411Z` |
| Ended | ~`2026-07-21T07:10:38Z` (~46.5 min wall) |
| HEAD at launch | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` |
| Command | `refresh --mode backfill --bounded-evidence --snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test --raw-root C:/ORATS/data/ORATS_Data --start-date 2024-04-01 --as-of 2024-06-28 --workers 8` |
| Durable log | `C:/MomentumCVG_env/snapshots_c8_4_test/c8_4_refresh_20260721T062411Z.log` |
| Exit code sidecar | `.../c8_4_refresh_20260721T062411Z.log.exitcode` → **1** |
| `build_id` | `20260721T062412533463Z_d42da59c` |
| Staging retained | `.../20260721T062412533463Z_d42da59c.building` (resumable) |
| Published final | **no** |
| Stage markers | **none** (failed before first marker) |

### Failure

Liquidity / Core classification (Stage 1):

- Security-type dictionary absent → classifying **5929** candidate tickers.
- Producer error: `ORATS Core HTTP 404 for ticker ADTH tradeDate 2024-06-28: {"message":"Not Found."}`
- Mapped to CLI exit **1** (`StageExecutionError` / liquidity producer failed).

Fail-closed behavior matches current Core client contract (HTTP errors are not valid-empty).

### Resume note

`.building` retained with frozen `run_config.json` / `raw_inventory.json`. Normal resume path is available after a product/ops decision on how to handle Core 404 (do **not** change code in this evidence step unless approved as a separate prerequisite).

### Next

Operator decision: investigate/fix Core 404 handling vs skip/resume strategy; then either resume or start a fresh bounded run.

---

## Prerequisite fix (post–Step 4 failure)

| Field | Result |
|-------|--------|
| Change | Core HTTP **404** → empty observation (date fallback); unresolved tickers **skipped** (not company equity, **not** written to security-type dictionary; retry when still missing) |
| Files | `src/data/orats_core_client.py`, `src/data/security_types.py`, `scripts/build_liquidity_panel.py`, unit tests |
| Tests | `test_orats_core_client`, `test_security_types`, `test_liquidity_security_integration` — **84 passed** |
| Intent | Unblock minority Core-404 tickers (e.g. ADTH on some dates) without aborting Stage 1 |

Hard failures (auth, 5xx exhausted, parse/validation) still abort the batch.

### Checkpoint follow-up (power-loss recovery)

| Field | Result |
|-------|--------|
| Change | Checkpoint security-type dictionary every **100** newly classified tickers; cold-backfill default path is durable **`C:/MomentumCVG_env/reference/orats_security_types.parquet`** (append-only for new tickers; existing rows never re-fetched) |
| Files | `src/data/security_types.py`, `src/data/snapshot_stage_adapters.py`, tests |
| Tests | adapter/security/liquidity paths green |
| Effect | Interrupted Core backfill loses at most ~100 fetches; later backfills reuse the shared reference file |

---

## Step 4 retry — resume after Core-404 fix (interrupted)

| Field | Result |
|-------|--------|
| **Verdict** | **INTERRUPTED** (power loss / process death) |
| Mode | `refresh --resume 20260721T062412533463Z_d42da59c` |
| Durable log | `C:/MomentumCVG_env/snapshots_c8_4_test/c8_4_resume_20260721T183123Z.log` (empty; no exitcode sidecar) |
| Durable Core dict | Absent at interrupt — no checkpoint yet |
| Note | `.building` retained; OS lock released on process death (sibling `.lock` file may remain as empty sibling) |

---

## Step 4 retry #2 — resume with progress + durable Core dict

| Field | Result |
|-------|--------|
| **Verdict** | **FAIL** (exit **2** = usage/config; stages completed, final validation rejected) |
| Started UTC | `20260721T192119Z` |
| Ended UTC | `20260721T221113Z` (~2h 50m wall) |
| Mode | `refresh --resume 20260721T062412533463Z_d42da59c --snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test --workers 8` |
| Code pin note | HEAD still `a70b63f…` plus local tracked diffs: Core-404 graceful handling, checkpoint-every-100, durable `C:/MomentumCVG_env/reference/orats_security_types.parquet`, `run_progress.json` |
| Token probe | PASS (AAPL type 3, SPY type 7; ADTH empty-OK / type 0) — token value **not** recorded |
| Durable log | `C:/MomentumCVG_env/snapshots_c8_4_test/c8_4_resume_20260721T192119Z.log` |
| Exit code sidecar | `.../c8_4_resume_20260721T192119Z.log.exitcode` → **2** |
| Progress end state | `surface` / `complete` / `pct=100` (all four stages) |
| Stage markers | **all four present** under `.building/markers/` |
| Durable Core dict | `C:/MomentumCVG_env/reference/orats_security_types.parquet` — **5861** rows |
| Published final | **no** |
| Final report | **not written** (failed before `reports/final/final_validation.json`) |

### Failure

Final cross-stage validation / candidate-manifest construction:

```text
no feature-ready interval: no contiguous Surface-supported entry dates
satisfy strict-prior PIT membership and input coverage
```

Producers + stage gates reached surface completion; publication did not run.

### Next

Diagnose why no Surface-supported entry date clears feature-ready (universe ⊆ surface meta, adjusted+spot coverage including expiries, strict-prior PIT). Decide ops/product fix vs window/universe change; then resume or fresh run.

---

## Closeout (2026-07-22)

| Field | Result |
|-------|--------|
| **Verdict** | **CLOSED** — ACCEPT WITH LIMITATIONS |
| Feature-ready fix | `8e9d4b83e6cb1dff8a20f54e1d13c5e3a9074d60` |
| Finalize resume log | `C:/MomentumCVG_env/snapshots_c8_4_test/c8_4_finalize_resume_20260723T053115Z.log` → exit **0** |
| Snapshot ID | `1b1e28b262ba40be` |
| Final root | `C:/MomentumCVG_env/snapshots_c8_4_test/20260721T062412533463Z_d42da59c` |
| Feature-ready | `2024-04-12` → `2024-06-21` (11) |
| Completion report | [c8_4_bounded_backfill_evidence.md](c8_4_bounded_backfill_evidence.md) |
| Closeout memo | [../sprint_memos/004_c8_4_bounded_evidence.md](../sprint_memos/004_c8_4_bounded_evidence.md) |
