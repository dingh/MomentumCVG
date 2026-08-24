# Sprint 006 D4 — Baseline execution evidence (Phases 3–4)

**Evidence verdict: `ACCEPTED`**

| Field | Value |
|-------|-------|
| Phase 1/2 checkpoint | [`docs/tmp/sprint006_d4_phase12_checkpoint.md`](../tmp/sprint006_d4_phase12_checkpoint.md) — `PASS`, accepted through docs commit `aa697a7` |
| Official execution commit | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Docs HEAD when memo written | `aa697a7385b4a90325c281a083e67fd1d0d1e6b4` (branch `main`) |
| Official `RUN_DIR` | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Verification directory | `C:/MomentumCVG_env/runs/sprint006_d4_verification_20260823T204430Z` |
| Outside-repo verdict record | `…/EVIDENCE_VERDICT.md` |

Phase 5 was **not** started. No aggregate economics were opened or interpreted.

---

## 1. Phase 3 — official full baseline

Executed at **detached HEAD** `e205b9a` so `run_receipt.json.repo_sha` matches the tested `EXECUTION_COMMIT`. Later commits `fd60905` / `aa697a7` are documentation-only and were not used for the run.

| Item | Value |
|------|-------|
| Command | `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint006_baseline.py --contract configs/sprint006_baseline_v1.json --output-dir C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Contract | `configs/sprint006_baseline_v1.json` (frozen; on-disk SHA-256 `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c`) |
| `START_UTC` (reconstructed) | `2026-08-23T20:44:30Z` (process start; capturing shell was interrupted earlier) |
| `END_UTC` | `2026-08-23T23:41:49+00:00` (`run_receipt.generated_utc`) |
| Duration | ~3.0 hours wall clock |
| Shell `EXIT_CODE` | Not captured (parent capture interrupted). Completion evidenced by 17 artifacts + receipt `result_complete=true` |
| Fills | Both frozen fills in one invocation: mid then cross |

### Artifact result

Exactly **17** adapter files in `$RUN_DIR` (mid×7 + cross×7 + `decision_report.json` + `decision_report.md` + `run_receipt.json`). No extras. Mid artifacts mtime ~15:12 local; cross ~16:40; report/receipt ~16:41–16:42.

Receipt identity (structural only):

| Field | Value |
|-------|-------|
| `experiment_id` | `sprint006_baseline_v1` |
| `repo_sha` | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| `contract.sha256` | `4012b4a4…` (matches on-disk frozen digest) |
| `result_complete` | `true` |
| `has_unresolved_failures` | `false` |
| `deliverable` | `sprint006_d3` (expected D3 producer metadata; not a defect) |
| Per fill | `n_expected_dates=403`, `n_traded_dates=403`, `n_valid_no_trade_dates=0`, `n_failed_dates=0`, `n_feature_dates_absent_from_a1=0` |

### Contract and input identities

Unchanged from the accepted Phase 1 gate: snapshot `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886`; seven Phase 1 input digests re-verified byte-identical at V-14.

---

## 2. Phase 4 — V-1…V-20

Independent checks. V-18/V-19 re-derived from Parquet fields; production `assert_included_trade_legs()` was **not** called. `decision_report.md` never opened; `decision_report.json` key-only walk for V-9; `run_summary_*.json` structural fields only.

| # | Expected | Observed | Source | Verdict |
|---|----------|----------|--------|---------|
| V-1 | `result_complete=true` | `true` | receipt | PASS |
| V-2 | `has_unresolved_failures=false` | `false` | receipt | PASS |
| V-3 | `n_failed_dates=0` both fills | 0; no failed `date_status` rows | receipt + date_status | PASS |
| V-4 | `n_expected_dates` = A1 count 403 | `[403,403]` | receipt + A1 | PASS |
| V-5/mid | one row per A1 date; statuses; reason null iff traded | n=403 exact calendar | date_status | PASS |
| V-5/cross | same | n=403 exact calendar | date_status | PASS |
| V-6/mid | bounds in `[2018-10-26,2026-07-10]` | exact | date_status | PASS |
| V-6/cross | same | exact | date_status | PASS |
| V-7 | feature-absent count recorded; no unresolved failures | absent=0 both | receipt | PASS |
| V-8 | 17 non-zero files | 17; none missing/extra | listing | PASS |
| V-9 | pinned schemas + report top keys | exact; 607 key paths (values unused) | headers + key walk | PASS |
| V-10 | recomputed SHA-256 = receipt | 16/16 non-receipt files match | hash | PASS |
| V-11 | `repo_sha == EXECUTION_COMMIT` | equal | receipt | PASS |
| V-12 | contract sha/id/version/status | `4012b4a4…` / v1 / accepted | receipt | PASS |
| V-13 | accepted_inputs paths = §2.2; earnings null | all recorded paths match; earnings null | receipt | PASS |
| V-14 | input digests = Phase 1 baseline | all 7 match | §1.6 rerun | PASS |
| V-15 | experiment + run ids; deliverable=d3 expected | match | receipt | PASS |
| V-16 | exactly 17; mtimes in window | ok | listing + mtime | PASS |
| V-17/mid | funnel monotone + splits | 0 issues | funnel_summary | PASS |
| V-17/cross | same | 0 issues | funnel_summary | PASS |
| V-18/mid | included legs 2/4; no orphans | n_included=10486; 0 bad | trade+leg independent | PASS |
| V-18/cross | same | n_included=10486; 0 bad | independent | PASS |
| V-19/mid | four S-7 identities all included | 0 bad cells | independent | PASS |
| V-19/cross | same | 0 bad cells | independent | PASS |
| V-20/mid | candidate/trade consistency | ok | cand+trade | PASS |
| V-20/cross | same | ok | cand+trade | PASS |

**V summary:** 26 PASS / 0 FAIL.

---

## 3. Frozen S1–S4 selection (primary cross)

| Sample | Rule / enumeration | Result |
|--------|--------------------|--------|
| **S1** | Median A1 date `2022-09-02` (`traded`); lowest-ticker included long + short | **ACN** long; **AMC** short |
| **S2** | Earliest traded date with both sides (ascending scan) | **2018-10-26**; **ABBV** long; **MRVL** short |
| **S3** | Earliest `valid_no_trade` | **N/A** — `n_valid_no_trade_dates=0` (frozen fallback) |
| **S4** | Earliest date with `structure_ok=False`; lowest-ticker failing row | **2018-10-26 / AMBA / short**; `wing_or_liquidity_selection`; raw: `No quotes with abs_delta <= 0.15 available for selection` |

Included trades audited: **4** (≤6). Shortfall documented: S3 unavailable; S4 is a structure failure, not an included trade. No performance-based substitution.

Matching mid records compared at leg level for all four included samples (half-spread equations).

---

## 4. Source-level reconstruction (audit evidence — not economic analysis)

Per-leg / per-trade P&L fields were machine-read solely for reconstruction and reconciliation. Sample numeric values below are **audit evidence**, not strategy performance interpretation.

Stages verified from accepted inputs (liquidity panel, `features_42_8`, A1 meta, A2 quotes) for S1-L/S1-S/S2-L/S2-S:

- Universe membership and `dvol_rank_pct ≥ 0.80` via `step1_get_universe`
- Signal side membership and rank equality via `step2_score_signals`
- A1 `entry_spot` / `exit_spot` / `body_strike` / `expiry_date`
- Leg count, `leg_index`, short iron-fly `unit_quantity` signs `+ − − +`
- A2 bid/ask and cross fills; mid↔cross half-spread deltas
- Entry / pnl / pnl_total identities; intrinsic payoff from `exit_spot`
- Short `max_loss = wing_width − net_credit`
- Sampled-date CAR vs that one `date_summary` row only

S4: A2 shows zero OTM quotes with `abs_delta ≤ 0.15` after the spread gate (explains reason); no leg rows; not included; date status `traded`.

**S1–S4 audit summary:** 110 PASS / 0 FAIL / 1 N/A. Full machine record: `phase4_s1_s4_audit.json` (outside repo).

Illustrative audit rows (not an economic report):

| id | expected | observed | verdict |
|----|----------|----------|---------|
| S1-L-fill legs | A2 ask for buys | match | PASS |
| S1-S-maxloss | 10.9 | 10.9 | PASS |
| S2-S-pnl_total | −5000.0 | −5000.0 | PASS |
| S4-wings | 0 eligible OTM | 0 | PASS |

---

## 5. Deviations and limitations

1. **Phase 3 stdout/stderr / shell EXIT_CODE** not retained — capturing shell interrupted while the Python process continued. Completion evidenced by artifacts and receipt. Timing reconstructed from process start + `generated_utc`.
2. **V-13** checks the adapter’s recorded `accepted_inputs` keys (manifest, features, A1/A2, liquidity, lineage). Receipt does not re-emit `snapshot_root` / `derived_root` as separate fields; paths present match §2.2.
3. **Wall clock ~3 hours** for one twin-fill full history — operational observation only; not an evidence failure.
4. Blind boundary held: no `decision_report.md`; no report values; no aggregate `run_summary` economics; no `date_summary` series aggregation.

---

## 6. Evidence verdict

**`ACCEPTED`** — V-1…V-20 all pass; every frozen sample reconciles within tolerance; S3 recorded N/A under the frozen rule; shortfall documented.

**Phase 5 remains forbidden until separately authorized.** Aggregate returns, Sharpe, drawdowns, yearly tables, concentration, and report values stay closed.

No raw run artifacts, verification scripts, or transcripts were copied into the repository.
