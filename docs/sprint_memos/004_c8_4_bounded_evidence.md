# Sprint 004 — C8.4 bounded snapshot evidence closeout

**Status:** Closed — **C8.4 accepted with limitations**
**Closed:** 2026-07-22
**Superseded for production by:** [004_closeout.md](004_closeout.md) (full C8.5 snapshot)
**Note:** Temporary C8.4 evidence drafts were removed at Sprint 004 closeout; facts below remain authoritative for the bounded evidence window.

---

## 1. Lineage

| Label | Commit | Meaning |
|-------|--------|---------|
| Freeze / marker stamp | `a70b63f335a55183d9caf8b937dd4fe482c2c6d1` | `repo_sha_at_freeze` and all four stage marker `producer_repo_sha` values |
| Producer-era recovery | `e5e9a8b770b7a2c18d9ddb7c08d6cdc7d754e0fa` | Core HTTP 404 → empty; durable security-types checkpoints; `run_progress.json` (ran as dirty tree during producer resume #2, committed after) |
| Feature-ready fix + finalize HEAD | `8e9d4b83e6cb1dff8a20f54e1d13c5e3a9074d60` | Invalid A1 null expiries excluded from expiry-coverage; finalize/publish resume |

**Accepted published snapshot:**

| Field | Value |
|-------|-------|
| Build ID | `20260721T062412533463Z_d42da59c` |
| Snapshot ID | `1b1e28b262ba40be` |
| Final root | `C:/MomentumCVG_env/snapshots_c8_4_test/20260721T062412533463Z_d42da59c` |
| Manifest | `…/manifests/input_snapshot_1b1e28b262ba40be.json` |

---

## 2. C8.4 question and answer

**Question:** Can a **bounded** real-data cold backfill run through the existing control plane

```text
liquidity → adjusted → spot → surface → final validation → atomic publication
```

and publish a truthful `scope="bounded"`, `production_accepted=false` snapshot with a nonempty feature-ready interval?

**Answer:** **Yes**, with documented limitations (see evidence report verdict **ACCEPT WITH LIMITATIONS**).

**Establishes:**

- Four-stage bounded real-data execution with accepted stage markers/gates
- Cross-stage universe / date / A1 identity reconciliation
- Published readable snapshot + manifest readback
- Nonempty feature-ready range: **2024-04-12 → 2024-06-21** (11 weekly entries)
- Truthful bounded / non-production labeling

**Does not establish:**

- Full-history performance or scalability (→ C8.5)
- Full-universe economic correctness
- Weekly incremental / repair modes
- CVG / Momentum feature correctness
- Strategy or backtest correctness
- Production acceptance (`production_accepted` remains false)

---

## 3. Operating window and outcomes

| Item | Value |
|------|-------|
| Window | `--start-date 2024-04-01` / `--as-of 2024-06-28` |
| Raw pad | `raw_dependency_start=2023-12-25`; 128 physical/resolved days |
| Scope | `bounded` |
| Manifest overall | **`WARN`** (accepted spot + surface WARN warnings) |
| `production_accepted` | `false` |
| Liquid universe | 668 tickers |
| Surface A1 | 8016 expected=actual; 5531 valid / 2485 invalid |
| Dominant invalid reason | `target_weekly_expiry_not_listed` (2383) |

Product decision locked for this evidence: **classification-v2** (date-specific Core `assetType`).

---

## 4. Execution path (summary)

1. Original backfill exit **1** (Core 404 hard-fail on ADTH).
2. Producer resume #2 exit **2** (four stages OK; finalize empty feature-ready under pre-fix rule).
3. Feature-ready correctness fix landed as `8e9d4b8`.
4. Finalize-only resume exit **0**: all four stages **skipped**; publish succeeded.

---

## 5. Limitations (accepted)

- Split code provenance (producer dirty/`e5e9a8b` era stamped `a70b63f`; finalize under `8e9d4b8`).
- `overall_status=WARN` from accepted Gate SP / Gate SF warnings (XSP ambiguous spot; partial-surface informational WARN).
- End-state publish proves outcome, not mid-flight rename atomicity (covered by unit tests separately).

---

## 6. Next

| Item | Status |
|------|--------|
| **C8.5** full production snapshot evidence / Sprint 004 closeout | **Closed** — see [004_closeout.md](004_closeout.md) |
| **C3** umbrella `validate` | Deferred / optional later polish |
| **C9** runbook + plan cleanup | Deferred / optional later polish |

Canonical human evidence for the bounded C8.4 window is retained in this memo. Production acceptance is recorded in [004_closeout.md](004_closeout.md).
