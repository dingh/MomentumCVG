# C8.4 — Bounded evidence execution and evidence plan (final)

| Field | Value |
|-------|-------|
| Status | **Closed** — executed; see completion report |
| Completion report | [c8_4_bounded_backfill_evidence.md](c8_4_bounded_backfill_evidence.md) |
| Closeout memo | [../sprint_memos/004_c8_4_bounded_evidence.md](../sprint_memos/004_c8_4_bounded_evidence.md) |
| Expected / verified HEAD (plan pin) | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` |
| Finalize / publish HEAD | `8e9d4b83e6cb1dff8a20f54e1d13c5e3a9074d60` |
| Worktree at last inspection | See completion report provenance |
| Note | Supersedes earlier drafts; obsolete early NO-GO content replaced by the completion report |

---

## Outcome

One bounded real-data backfill through the existing production control plane:

```text
liquidity → adjusted → spot → surface → final validation → atomic publication
```

Published snapshot truthfully marked:

```text
scope="bounded"
production_accepted=false
```

Continue to use the existing C8.3B/C8.4 CLI, orchestration, producers, gates, markers, finalizer, and publication lifecycle. No new runner, framework, orchestration path, or acceptance contract.

---

## Claims C8.4 does and does not establish

**Does establish (if all steps pass):**

- Four-stage cold backfill can execute end to end on real ORATS under the existing CLI/orchestrator.
- Each stage’s existing acceptance contract is satisfied for this run.
- Universe, date, key-count, and digest identities reconcile across stages.
- A nonempty feature-ready interval is derived.
- The published manifest for **this** `build_id`/`snapshot_id` accurately describes artifacts and validates on readback.
- End-state after publish is consistent with the publication path (final root present; `.building` and `work/` absent).
- Frozen raw inventory still matches a post-run rescan.
- After the publisher process exits, the sibling OS lock can be acquired (no residual holder).
- Manifest helpers can resolve and open key artifacts for **this** published root.

**Does not establish:**

- Full-history scalability, full-universe economic correctness, incremental/repair modes.
- Feature correctness, strategy performance, Sprint 005/production consumer acceptance.
- That `production_accepted=false` snapshots will be accepted by a future consumer gate.
- Mid-flight atomicity of `os.replace` on this machine (covered by unit tests + end-state consistency, not a live race observation).
- That Apr–Jun is the unique minimal date window.
- That Core policy matches the older “any type 4–9 in history excludes” design.

---

## Schedule note

For `2024-04-01`…`2024-06-28`:

| Set | Approximate count | Notes |
|---|---|---|
| Raw warm-up pad | automatic | `raw_dependency_start = 2023-12-25` (`start − 14` weeks) |
| Liquidity panel snapshots | week-ends in `[start, as_of]` ≈ 12–13 | `resolve_weekly_snapshot_dates` |
| Surface entry candidates | ≈ 13 | weekly entries in `[start, as_of]` |
| **Supported** Surface entries | ≈ 12 (through **2024-06-21**) | **2024-06-28 excluded** — no successor inside inventory |
| Feature-ready upper bound | ≤ supported | Needs prior panel snap; upper bound ≈ 11 if all later weeks pass |

Exact dates are last trading day of each week from file presence (holiday walk-back). Apr–Jun is a **margin** operating window, not proven minimal. Structural minimum for ≥1 feature-ready date is ≥3 consecutive weekly schedule dates plus the automatic 14-week pad.

---

## Remaining product decision (only one)

**Accept C8.4 evidence under implemented classification-v2** (latest observed Core `assetType`, bounded valid-empty fallback, 0–3 company / 4–9 non-company), knowing this differs from the older C8.3B design text (“any type 4–9 anywhere in bounded history excludes”).

Do not change code in C8.4. Do not mix this with dates, paths, or workers.

**Operating choices (not product):**

- `--start-date 2024-04-01`
- `--as-of 2024-06-28`
- `--snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test`
- `--workers 8`
- Evidence record: `docs/tmp/c8_4_bounded_evidence_record.md`
- Durable CLI log under the snapshots root (see Step 4)

---

## Authoritative evidence record

Create at execution time:

`docs/tmp/c8_4_bounded_evidence_record.md`

Required sections:

1. Identity: HEAD SHA, porcelain status, command line, durable log path, exit code, `build_id`, `snapshot_id`, `final_root`, bound manifest path
2. Preflight: API probes, inventory digest/counts, disk free, runtime forecast
3. Stage table: marker paths, gate report paths, status
4. Final: `final_validation.json` + manifest `scope` / `production_accepted` / digests / `feature_ready_*`
5. Source immutability: post-run `rescan_and_verify_raw_inventory` result
6. Publication end-state + lock-ownership probe outcome
7. Bound artifact resolve/read smoke
8. Deviations / resume history

Published snapshot artifacts remain machine-canonical; the record is the human audit index.

---

## 1. Preconditions and test baseline

### Step 0a — Pin exact executable code

| | |
|---|---|
| **Execute** | `git rev-parse HEAD`; `git status --porcelain=v1`; confirm Python = `C:/MomentumCVG_env/venv/Scripts/python.exe`; record that `scripts/refresh_weekly_inputs.py` loads from this repo root |
| **Expected** | HEAD = expected SHA; **no modified/staged/deleted tracked files**; untracked files only under `docs/tmp/` (document each) or none |
| **Evidence** | Paste porcelain output + HEAD into the evidence record |
| **Pass/fail** | **FAIL / NO-GO** if any tracked path is dirty or HEAD mismatches — do not attribute the run to HEAD while executing other code |
| **Machine / human** | &lt;1 min / ~5 min |

### Step 0b — Full automated test baseline

| | |
|---|---|
| **Execute** | `& C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/ -q` |
| **Expected** | Full suite green (Stage-A units **and** all `tests/contract/*`). Selective ignores of step2–step5 contracts are **not** used — they are part of this checkout’s regression surface and were not justified by C8.4 scope |
| **Evidence** | pytest summary (pass count, exit code) in evidence record |
| **Pass/fail** | Exit 0; any failure is NO-GO for the long run |
| **Machine / human** | ~5–20 min / ~5 min |

---

## 2. API, inventory, resource, and environment preflight

### Step 1 — Core + splits API preflight (no false confidence)

| | |
|---|---|
| **Execute** | With `ORATS_API_TOKEN` set, using existing clients only: (1) Core `fetch_asset_type_at_date` for a known company equity (e.g. `AAPL`) on trading dates in-window until a **nonempty** row with `assetType ∈ {0,1,2,3}` parses; (2) Core for a known non-company (e.g. `SPY`) until a **nonempty** row with `assetType ∈ {4..9}` parses; (3) map both through the same accept/classify rules Stage 1 uses; (4) splits `fetch_splits_for_ticker` for one liquid equity — HTTP/parse success (empty split history OK if structurally valid) |
| **Expected** | Both Core paths return **usable nonempty** observations (not valid-empty alone); splits authenticated and parseable |
| **Evidence** | Ticker, date, returned `assetType`, derived classification; splits OK/FAIL — **no token logged** |
| **Pass/fail** | **FAIL / stop** if auth fails, transport exhausted, parse fails, or either Core path cannot obtain a nonempty in-domain observation after a small fixed set of in-window dates. Valid-empty on one date may retry another date; exhausting retries without a nonempty row = fail. Do **not** treat valid-empty as probe success |
| **Machine / human** | ~2–10 min / ~15 min |

### Step 2 — Raw inventory + resource forecast

| | |
|---|---|
| **Execute** | `scan_raw_inventory(raw, 2023-12-25, 2024-06-28)`; sum archive sizes; check free space on snapshots volume; forecast Core time ≈ (candidate tickers × ≤5 × 0.7s) with “thousands” as prior order-of-magnitude; splits ≈ liquid-superset × 0.7s (prior full-history liquid ~2783 as upper reference only) |
| **Expected** | Nonempty inventory; `as_of_resolved_trading_day == 2024-06-28`; free disk ≥ **200 GB** unless forecast revises; operator accepts runtime band |
| **Evidence** | `inventory_digest`, physical/resolved counts, byte sum, free disk, forecast table |
| **Pass/fail** | Scan OK + bounds covered + disk OK + operator accepts forecast |
| **Machine / human** | ~5–30 min / ~20 min |

**Forecast framing (not a promise):**

| Stage | Dominant cost | Order-of-magnitude basis |
|---|---|---|
| Liquidity / Core | API-bound | empty stage-local dictionary → classify all candidates; 0.7 s/request; up to 5 attempts/ticker |
| Adjusted / splits | API then CPU | ~1 request/liquid ticker + ZIP adjust over `physical` days × workers |
| Spot / surface | CPU/IO | days × tickers; Surface keys ≈ liquid × supported entries |
| Disk | snapshots volume | adjusted dailies + surface pair + reports; size unknown until inventory byte sum + headroom check |

### Step 3 — Dry-run contract

| | |
|---|---|
| **Execute** | `refresh --dry-run --mode backfill --bounded-evidence` with target flags (tee to preflight log) |
| **Expected** | `scope: bounded`, `production_accepted: false`; no scan/lock/producers |
| **Evidence** | Captured stdout |
| **Pass/fail** | Exit 0 + bounded flags present |
| **Machine / human** | &lt;1 min / ~2 min |

---

## 3. Bounded backfill execution

### Step 4 — Execute (with durable process capture)

| | |
|---|---|
| **Execute** | Run the exact command in § Exact proposed command with **stdout and stderr tee’d** to a durable log under the snapshots root; record the process exit code into the evidence record |
| **Expected** | Exit **0**; last lines include `build_id`, `final_root`, `snapshot_id` |
| **Evidence** | Durable log file + recorded exit code + the three identity lines |
| **Pass/fail** | Exit 0 **and** `final_root` exists as `snapshots_root / build_id` (no leftover `.building`) |
| **Machine / human** | Per Step 2 forecast (multi-hour likely; Core may dominate) / ~15–30 min monitoring |

Do **not** artificially interrupt. Natural failure → § Failure and normal-resume handling.

---

## 4. Durable evidence collection (identity binding)

### Step 5 — Bind run identity before any further inspection

| | |
|---|---|
| **Execute** | From the durable log, extract `build_id`, `final_root`, `snapshot_id`. Verify: `final_root == resolve(snapshots_root / build_id)`; `run_config.json` at `final_root` has matching `build_id`; manifest path **must** be exactly `final_root / manifests / input_snapshot_{snapshot_id}.json` (use `default_manifest_path(final_root, snapshot_id)` — **never** “first/glob any manifest”); `read_manifest` → `manifest.build_id == build_id` and `manifest.snapshot_id == snapshot_id` and `compute_snapshot_id(manifest) == snapshot_id` |
| **Expected** | All identities agree |
| **Evidence** | Absolute paths + the three IDs in the evidence record |
| **Pass/fail** | Any mismatch = FAIL (stop; do not inspect unrelated manifests) |
| **Machine / human** | ~2 min / ~10 min |

---

## 5. Stage-by-stage expected outcomes and acceptance evidence

### Step 6 — Stage gates (under bound `final_root` only)

| Stage | Expected | Evidence under `final_root` | Pass/fail |
|---|---|---|---|
| liquidity | Strict C7 PASS; marker accepted | `markers/liquidity.done.json`; `reports/liquidity/pit_universe_audit.md` | `stage_accepted=true`; audit PASS |
| adjusted | C5 PASS w/ expected physical dates | adjusted marker + `reports/adjusted/*` | PASS |
| spot | Gate SP PASS or **only** accepted ambiguous WARN | spot marker + `reports/spot/gate_spot_reconciliation.json` | No FAIL; WARN only if accepted policy |
| surface | Exact A1 + C6; only accepted join WARN allowed | surface marker + `reports/surface/surface_contract_checks.json` | No FAIL; WARN policy held |
| final | Cross-check + feature-ready ≥ 1 | `reports/final/final_validation.json` | Digests OK; `feature_ready_entry_count >= 1` |

| | |
|---|---|
| **Machine / human** | ~5–10 min / ~30–40 min |

Confirm terminal week is **not** required to be Surface-supported; feature-ready ⊆ supported.

---

## 6. Source-immutability verification

### Step 7 — Rescan frozen inventory

| | |
|---|---|
| **Execute** | Load `final_root / run_config.json`; call `rescan_and_verify_raw_inventory(config)` |
| **Expected** | Rescan digest equals frozen `inventory_digest` |
| **Evidence** | Both digests in evidence record |
| **Pass/fail** | Equal digests; else FAIL |
| **Machine / human** | ~5–30 min / ~5 min |

---

## 7. Publication and lock checks

### Step 8 — Publication end-state (real run)

| | |
|---|---|
| **Execute** | Inspect filesystem for **this** `build_id` only |
| **Expected** | `final_root` exists; `build_id.building` absent; `final_root/work` absent; markers/manifests/reports present |
| **Evidence** | Directory listing notes in record |
| **Pass/fail** | End-state matches; else FAIL |
| **Machine / human** | ~2 min / ~10 min |

**Claim limit:** proves **observed end-state consistency** with successful publication. Does **not** by itself prove mid-flight atomic rename. Separately, existing unit tests (`test_publish_removes_work_renames_and_keeps_sibling_lock`, rename-failure leaves `.building`, bounded publish with `production_accepted=false`) prove the publication function’s contract on fixtures.

### Step 9 — Sibling lock ownership probe (not “read-only”)

| | |
|---|---|
| **Execute** | After publisher process has fully exited: `SiblingBuildLock(snapshots_root, build_id).acquire()` then `release()` |
| **Expected** | Acquire succeeds (no other holder). The `.lock` **file may still exist** — that is normal and proves nothing about ownership |
| **Evidence** | Acquire/release outcome in record |
| **Pass/fail** | Acquire succeeds; fail if lock contention remains |
| **Machine / human** | &lt;1 min / ~5 min |

**Claim limit:** this is an **active lock-ownership probe**, not a read-only check. File absence/presence is **not** evidence of lock state.

---

## 8. Post-publication manifest and artifact-resolution checks

### Step 10 — Bound manifest + artifact smoke

| | |
|---|---|
| **Execute** | Using the **bound** manifest path from Step 5: confirm `params.scope=="bounded"`, `production_accepted is False`, `data_source=="orats_raw_rebuild"`, `Path(cache_dir).resolve()==final_root`; `resolve_under_root` each artifact/report; open panel + surface meta (row counts &gt; 0) |
| **Expected** | Flags correct; files open |
| **Evidence** | Paths, counts, flags in record |
| **Pass/fail** | All true |
| **Machine / human** | ~5–15 min / ~10 min |

**Claim limit:** proves schema-v1 helpers can resolve/read **this** published tree. Does **not** prove a wired production consumer or future acceptance of non-production snapshots.

---

## 9. Failure and normal-resume handling

On producer/gate failure (exit **1**): `.building` remains; completed markers stay; failed stage has **no** marker; upstream accepted outputs untouched.

Resume (same snapshots root; frozen config authoritative; no identity flags):

```powershell
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh `
  --resume <BUILD_ID> `
  --snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test `
  --workers 8
```

Tee resume stdout/stderr to a second durable log; append to the evidence record. Frozen `scope` remains `bounded`.

On raw drift / corruption (exit **2**): stop; do not force-continue. On interrupt (130): same resume path. On rename failure: `.building` left; resume retries finalize/publish after re-validation.

Do **not** introduce an artificial interruption merely to demonstrate resume.

---

## 10. Final evidence matrix

| Claim | Support | Expected | Persisted proof |
|---|---|---|---|
| Exact code pin | Step 0a | HEAD + clean tracked tree | Evidence record porcelain |
| Checkout safe | Step 0b full pytest | Exit 0 | pytest summary |
| Core usable for Stage 1 paths | Step 1 nonempty company + non-company | In-domain assetTypes | Probe rows in record |
| Splits API usable | Step 1 | Auth+parse OK | Probe note |
| Inventory sized / forecast accepted | Step 2 | Bounds + disk OK | Digest/counts/forecast |
| E2E bounded run | Step 4 | Exit 0; published | Durable CLI log + `final_root` |
| Identity unambiguous | Step 5 | build/snapshot/path agree | Bound manifest path |
| Stage contracts | Step 6 | PASS / accepted WARN | Markers + reports under `final_root` |
| Feature-ready nonempty | Step 6 final | count ≥ 1 | `final_validation.json` + manifest params |
| Manifest truthful bounded | Steps 5–10 | `bounded` / `production_accepted=false` | Bound manifest JSON |
| Source unchanged | Step 7 | Digests equal | Record digests |
| Publish end-state | Step 8 | Final; no `.building`/`work/` | FS notes + unit tests for atomic path |
| Lock released by publisher | Step 9 acquire | Succeeds | Record (not file existence) |
| Artifact resolve/read smoke | Step 10 | Opens OK | Counts in record |

**Evidence source buckets:**

| Bucket | Contents |
|---|---|
| Existing automated tests | Step 0b |
| Preflight | Steps 1–3 |
| Bounded-run evidence | Steps 4–6 |
| Post-publication checks | Steps 7–10 |
| Not established | See “Claims C8.4 does not establish” |

---

## Exact proposed command

```powershell
$log = "C:/MomentumCVG_env/snapshots_c8_4_test/c8_4_refresh_{0:yyyyMMddTHHmmssZ}.log" -f (Get-Date).ToUniversalTime()
New-Item -ItemType Directory -Force -Path "C:/MomentumCVG_env/snapshots_c8_4_test" | Out-Null
& C:/MomentumCVG_env/venv/Scripts/python.exe scripts/refresh_weekly_inputs.py refresh `
  --mode backfill `
  --bounded-evidence `
  --snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test `
  --raw-root C:/ORATS/data/ORATS_Data `
  --start-date 2024-04-01 `
  --as-of 2024-06-28 `
  --workers 8 `
  2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
# Record $log and $code into docs/tmp/c8_4_bounded_evidence_record.md
```

---

## Runtime and human-review estimates

| Phase | Machine | Human |
|---|---|---|
| Steps 0–3 | ~40–75 min | ~1–1.25 h |
| Step 4 | **TBD from Step 2** (multi-hour likely; Core may dominate) | ~15–30 min |
| Steps 5–10 + record | ~45–90 min | ~2 h |
| **Total** | preflight + forecasted run + ~1–1.5 h post | ~3.5–4.5 h review |

Do not treat a fixed multi-hour wall time as certified until Step 2 numbers exist.

---

## Completion checklist

- [x] Product decision: classification-v2 evidence accepted
- [x] HEAD pinned; tracked worktree clean; porcelain recorded (Step 0a PASS — see `c8_4_bounded_evidence_record.md`)
- [x] Full `pytest tests/ -q` green (Step 0b PASS — 1292 passed, 1 skipped)
- [x] Core nonempty company **and** non-company probes PASS; splits PASS (Step 1 PASS — AAPL assetType 3; SPY assetType 7; AAPL splits OK)
- [x] Inventory sized; disk OK; runtime band accepted (Step 2 PASS — 128 dates, ~8.2 GiB raw, 319 GiB free)
- [x] Dry-run bounded contract (Step 3 PASS — scope=bounded, production_accepted=false)
- [x] Durable CLI logs + exit codes retained (original exit 1; producer resume exit 2; finalize resume exit 0 — see completion report)
- [x] `build_id` / `snapshot_id` / `final_root` / bound manifest path agree (`1b1e28b262ba40be`)
- [x] Four stage markers/gates accepted; feature-ready ≥ 1 (`2024-04-12`→`2024-06-21`, 11)
- [x] Manifest `scope=bounded`, `production_accepted=false` (overall **WARN** — accepted spot/surface warnings)
- [x] Post-run raw rescan on finalize resume succeeded (exit 0 with `rescan_raw=True`)
- [x] Publish end-state OK (`.building` gone; final present; no `work/`)
- [x] Manifest readback binds build + snapshot + `cache_dir`
- [x] Evidence record + completion report + closeout memo complete

**Closed 2026-07-22** — ACCEPT WITH LIMITATIONS.