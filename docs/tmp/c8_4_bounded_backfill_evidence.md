# C8.4 bounded backfill — evidence-based completion report

| Field | Value |
|-------|-------|
| Report written | 2026-07-22 (post-publish inspection) |
| Status | **Closed** — C8.4 accepted with limitations |
| Closeout memo | [../sprint_memos/004_c8_4_bounded_evidence.md](../sprint_memos/004_c8_4_bounded_evidence.md) |
| Build ID | `20260721T062412533463Z_d42da59c` |
| Snapshot ID | `1b1e28b262ba40be` |
| Snapshots root | `C:/MomentumCVG_env/snapshots_c8_4_test` |
| Inspection | Read-only reconciliation of logs + published snapshot (no ORATS, no resume, no code edits) |

---

## 1. Bottom line

### Verdict: **ACCEPT WITH LIMITATIONS**

Evidence supports concluding that **C8.4 completed successfully as a published bounded, non-production snapshot** for this build: four stages accepted, finalize selected a nonempty feature-ready interval under the corrected rule, manifest readback binds to this build/snapshot ID, and the final root exists with `.building` gone.

**Correctness blockers:** none identified that invalidate accepting this **bounded / non-production** snapshot.

**Non-blocking limitations / diagnostics:**

1. Manifest / final report `overall_status` is **`WARN`**, not `PASS`, because spot and surface carried **accepted contract WARN** warnings into finalization (by design).
2. **Provenance mismatch:** markers stamp `repo_sha_at_freeze=a70b63f…`, but producer resume #2 ran with local Core-404/checkpoint/progress changes later committed as `e5e9a8b…`. Finalize/publish ran under `8e9d4b8…`. Do not claim producers executed as clean `a70b63f`.
3. Sibling `.lock` file still exists **beside** the snapshot (0 bytes); **not** inside the final root. OS lock-hold state after exit is **NOT EVIDENCED** beyond successful publish + exit 0.
4. Mid-flight rename atomicity is **not** proven by end-state alone (unit tests cover mechanism; filesystem shows successful publish outcome only).

---

## 2. Run identity and provenance

| Item | Evidence |
|------|----------|
| Build ID | `20260721T062412533463Z_d42da59c` |
| Snapshot ID | `1b1e28b262ba40be` (`manifests/input_snapshot_1b1e28b262ba40be.json`; CLI summary) |
| Original `.building` | `…/20260721T062412533463Z_d42da59c.building` — **absent after publish** |
| Final snapshot | `C:/MomentumCVG_env/snapshots_c8_4_test/20260721T062412533463Z_d42da59c` — **present** |
| Output window | `2024-04-01` → as-of `2024-06-28` |
| Raw dependency start | `2023-12-25` |
| Physical/resolved dates | **128** / **128**; `2023-12-26` … `2024-06-28` |
| Inventory digest | `d0a4f9f7a7e752e347dd5875235c4133bacacb6dd2391aae6517538b44db2133` |
| `run_config_id` | `94f5e4674695d9f501e90f3b1965cdcc24b8c06d9375ab06d900dfc611a0372f` |
| Frozen scope | `bounded` |
| Manifest scope / production | `bounded` / `production_accepted=false` |
| Manifest overall | **`WARN`** |
| `repo_sha_at_freeze` | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` |
| Raw root | `C:/ORATS/data/ORATS_Data` |
| Manifest `cache_dir` | equals final root (readback OK) |
| `data_source` | `orats_raw_rebuild` |

### Commands, logs, exits

| Attempt | Command (evidence) | Log / exit | Exit | Role |
|---------|--------------------|------------|------|------|
| Original backfill | `refresh --mode backfill --bounded-evidence … --start-date 2024-04-01 --as-of 2024-06-28 --workers 8` (`docs/tmp/c8_4_bounded_evidence_record.md`) | `c8_4_refresh_20260721T062411Z.log` + `.exitcode` | **1** | Failed Core 404 (ADTH) before markers |
| Interrupted resume | resume meta in `c8_4_refresh_latest_meta.txt` | `c8_4_resume_20260721T183123Z.log` (0 B); no `.exitcode` | **NOT EVIDENCED** | Empty / interrupted |
| Producer resume #2 | `refresh --resume 20260721T062412533463Z_d42da59c --snapshots-root C:/MomentumCVG_env/snapshots_c8_4_test --workers 8` (`.meta`) | `c8_4_resume_20260721T192119Z.log` + `.exitcode` | **2** | Built all 4 stages; finalize failed pre–feature-ready fix |
| Finalize resume | same resume CLI; meta head=`8e9d4b8…` (`c8_4_finalize_resume_20260723T053115Z.log.meta`) | `c8_4_finalize_resume_20260723T053115Z.log` + `.exitcode` | **0** | Stages skipped; finalize+publish |

### Code state by phase

| Phase | Code evidence |
|-------|----------------|
| Freeze | `a70b63f…` in `run_config.json` |
| Marker `producer_repo_sha` (all four) | `a70b63f…` |
| Producer resume #2 actual tree | Local uncommitted Core-404 / durable security-types / progress (later `e5e9a8b770b7a2c18d9ddb7c08d6cdc7d754e0fa`). **Not** clean `a70b63f`. |
| Feature-ready fix | `8e9d4b83e6cb1dff8a20f54e1d13c5e3a9074d60` |
| Finalize/publish resume | Ran at HEAD `8e9d4b8…` (log `.meta`). **Producers did not re-execute under this commit.** |

---

## 3. Stage results

Markers live under the **final** root (`…/markers/*.done.json`). Timestamps are from producer resume #2 (`2026-07-21T21:54Z`–`22:11Z`). Finalize resume **did not rewrite** marker `completed_at_utc`.

| Stage | Original | Producer resume #2 | Finalize resume | Marker / gate | Measured outputs | Accepted warnings | Paths |
|-------|----------|--------------------|-----------------|---------------|------------------|-------------------|-------|
| **liquidity** | Failed (exit 1) | **Executed** | **Skipped** | accepted; **PASS** | liquid **668**; classified **5861**; daily **489834**; weekly **102331**; panel **56233**; files_read **115**; equity `c64d4a68…`; class `86870e71…` | none | `markers/liquidity.done.json`; `input/liquidity/*`; `reports/liquidity/pit_universe_audit.md` (overall **PASS**, strict **True**) |
| **adjusted** | No | **Executed** | **Skipped** | **PASS** | **128** dates `2023-12-26`…`2024-06-28`; produced **128**; bytes **5,711,255,792**; universe digest matches liquidity; C5 `audit_verdict=PASS`; adj inv `7910463b…` | none | `markers/adjusted.done.json`; `input/adjusted_liquid/`; `reports/adjusted/adjusted_liquid_audit.md` |
| **spot** | No | **Executed** | **Skipped** | **WARN** | source **84946** → output **84818**; amb. exclusions **128** (XSP); Gate SP **WARN** | 128 ambiguous; 128 dropped inconsistent spot keys | `markers/spot.done.json`; `cache/spot/spot_prices_adjusted.parquet` (84818 rows / 667 tickers / 128 dates); `reports/spot/gate_spot_reconciliation.json` |
| **surface** | No | **Executed** | **Skipped** | **WARN** | supported **12** entries; A1 expected=actual **8016**, digest `762d351e…`; valid **5531** / invalid **2485**; A2 **84198**; Gate SF overall **WARN** (`a1_a2_join_integrity` WARN; others PASS) | 61 invalid-meta-with-quotes | `markers/surface.done.json`; `cache/surface/option_surface_meta_weekly_2024_2024.parquet`; `…_quotes_…`; `reports/surface/surface_contract_checks.json` |

### Surface invalid reasons (meta)

| Reason | Count |
|--------|------:|
| `target_weekly_expiry_not_listed` | 2383 |
| `target_weekly_body_not_quotable` | 68 |
| `no_spot_price` | 34 |
| null `expiry_date` rows | 2417 |

No unexpected stage FAIL markers. Producer anomalies limited to accepted WARN contracts and Core skip warnings during classification (resume #2 log).

---

## 4. Cross-stage reconciliation

| Check | Result |
|-------|--------|
| Liquidity ↔ adjusted universe digest | **Match** `c64d4a68…` (668) |
| Manifest params carry same equity / class / adj / spot / surface digests | **Yes** (readback) |
| Adjusted date count vs frozen physical | **128/128** |
| Surface expected vs actual A1 | **Equal** 8016 / same digest |
| Frozen inventory digest vs inventory doc | **Match** `d0a4f9f7…` |
| Post-run raw immutability | Finalize resume uses `open_resume_run(..., rescan_raw=True)` and exited **0** → inventory rescan **passed** at publish time. Dedicated standalone rescan report file: **NOT EVIDENCED** |
| Feature-ready (persisted) | **2024-04-12 → 2024-06-21**, **11** entries (`final_validation.json` + manifest params) |
| First supported entry `2024-04-05` | Not in ready range (no strict-prior PIT panel) — expected |

### Corrected `_select_feature_ready` (`8e9d4b8`)

Invalid A1 rows remain required for complete schedule coverage (`present == universe`), but **null/unavailable expiries on `surface_valid=false` rows do not invalidate the calendar date**. Only `surface_valid=true` expiries must be covered in adjusted∩spot. Producer resume #2 finalize failed under the old rule (exit 2); finalize resume under `8e9d4b8` selected the nonempty interval above.

---

## 5. Resume and publication

| Question | Evidence answer |
|----------|-----------------|
| Finalize resume skipped all four producers? | **Yes** — marker timestamps unchanged; finalize log contains only publish summary (318 B); no producer tqdm/Core traffic |
| External API / real-data producers on finalize resume? | **No producer rerun evidenced.** Local raw **rescan** ran in-process (`rescan_raw=True`); not Core/options API |
| Final validation success? | **Yes** — exit 0; `reports/final/final_validation.json` present with `feature_ready_entry_count=11` |
| Manifest readback? | **PASS** — `read_manifest` loads; `build_id`/`snapshot_id`/`cache_dir` bind to this final root |
| `.building` gone? | **Yes** |
| Final snapshot exists? | **Yes** |
| Lock file **inside** snapshot? | **None** (`rglob('*.lock')` empty). Sibling `…/d42da59c.lock` remains **outside** |
| `scope="bounded"`? | **Yes** (manifest) |
| `production_accepted=false`? | **Yes** |
| `overall_status="PASS"`? | **No — `WARN`** (accepted spot/surface warnings) |
| Nonempty feature-ready? | **Yes** — `2024-04-12`…`2024-06-21` |

**Atomic publication:** Successful end-state (final present, `.building` absent, `work/` absent) is **outcome evidence**. Rename atomicity itself remains covered by unit tests, not by a live race observation here.

---

## 6. Manifest and artifact summary

Manifest:  
`C:/MomentumCVG_env/snapshots_c8_4_test/20260721T062412533463Z_d42da59c/manifests/input_snapshot_1b1e28b262ba40be.json` (3098 B)

| Manifest key | Rel path | Exists | Size / count | Validation |
|--------------|----------|--------|--------------|------------|
| `liquid_tickers` | `input/liquidity/liquid_tickers.csv` | yes | 7248 B; **668** rows | present |
| `liquidity_daily` | `…/ticker_liquidity_daily_observations.parquet` | yes | 3,119,607 B | present |
| `liquidity_weekly` | `…/ticker_liquidity_weekly_observations.parquet` | yes | 946,722 B | present |
| `liquidity_panel` | `…/ticker_liquidity_panel.parquet` | yes | 801,200 B | present |
| `splits` | `input/adjusted_liquid/splits_hist_liquid.parquet` | yes | 5,291 B | present |
| `adjusted_chains_root` | `input/adjusted_liquid` | yes | **5,711,255,792** B; **128** day parquets | present |
| `spot_prices` | `cache/spot/spot_prices_adjusted.parquet` | yes | 651,167 B; **84818** rows | present |
| `option_surface_meta` | `cache/surface/option_surface_meta_weekly_2024_2024.parquet` | yes | 247,944 B; **8016** rows | present |
| `option_surface_quotes` | `cache/surface/option_surface_quotes_weekly_2024_2024.parquet` | yes | 6,366,390 B; **84198** rows | present |

Also present (not all are separate manifest artifact keys): stage markers; C7/C5/SP/SF reports; `reports/final/final_validation.json`; frozen `run_config.json` / `raw_inventory.json`.

**Absent (good):** `work/`, `candidate/` trees, `.building`, locks inside final.

**Note:** frozen `run_config.json` still has `snapshot_id: null`; published identity is the **manifest** `snapshot_id=1b1e28b262ba40be`.

---

## 7. Warnings, limitations, and evidence gaps

| Item | Class |
|------|-------|
| Spot WARN (XSP ambiguous ×128) | **Accepted contract warning** |
| Surface WARN (partial weeklies; 61 invalid-with-quotes) | **Accepted contract warning** |
| Manifest `overall_status=WARN` | **Accepted contract warning** (aggregates above) — not a bounded-accept blocker |
| Producer SHA stamp vs dirty `e5e9a8b` tree | **C8.4 evidence limitation** (provenance) |
| Finalize under `8e9d4b8` after producers completed earlier | **C8.4 evidence limitation** (split code lineage) — expected for this fix path |
| Weekly-expiry sparsity (`target_weekly_expiry_not_listed`) | **Diagnostic only** |
| Interrupted empty resume log (`…183123Z`) | **Evidence gap** (superseded by later resumes) |
| Live proof of rename atomicity | **NOT EVIDENCED** (end-state ≠ race proof) |
| Sibling lock file residual outside final | **Diagnostic only** |
| Standalone post-run immutability memo beyond successful `rescan_raw` | **Evidence gap** (thin) |

---

## 8. What C8.4 establishes

| Claim | Established? |
|-------|----------------|
| Bounded real-data execution through all four stages | **Yes** |
| Existing stage acceptance contracts satisfied (PASS/WARN as allowed) | **Yes** |
| Internally consistent, readable published snapshot + manifest readback | **Yes** |
| Usable nonempty feature-ready interval | **Yes** (`2024-04-12`…`2024-06-21`, 11) |
| Truthful bounded / non-production labeling | **Yes** (`scope=bounded`, `production_accepted=false`) |

**Does not prove:** full-history performance/scalability; full-universe economic correctness; weekly incremental mode; CVG/Momentum feature correctness; strategy/backtest correctness; production acceptance.

---

## 9. Final acceptance checklist

| # | Item | Result |
|---|------|--------|
| 1 | Four accepted stage markers on published root | **PASS** |
| 2 | Stage reports bind (C7 PASS, C5 PASS, SP WARN, SF WARN) | **PASS** |
| 3 | Universe / date / A1 identity reconcile | **PASS** |
| 4 | Final validation written; nonempty feature-ready | **PASS** |
| 5 | Manifest written; readback binds build+snapshot+cache_dir | **PASS** |
| 6 | Final root exists; `.building` removed; no `work/` | **PASS** |
| 7 | `scope=bounded`, `production_accepted=false` | **PASS** |
| 8 | `overall_status=PASS` | **FAIL** (actual **`WARN`**) |
| 9 | Finalize resume exit 0; producers skipped | **PASS** |
| 10 | Raw inventory digest stable; resume rescan succeeded | **PASS** (implicit) |

### Recommendation

**Accept C8.4 for build `20260721T062412533463Z_d42da59c` / snapshot `1b1e28b262ba40be` with limitations:** treat as a successful **bounded, non-production** published evidence snapshot whose overall status is **WARN** due to accepted spot/surface warnings, and whose producer code lineage is **`e5e9a8b`-era dirty tree stamped `a70b63f`**, finalized under **`8e9d4b8`**.

No further publish evidence is required for this acceptance class unless a clean single-SHA producer+finalize replay is demanded as a separate provenance upgrade.
