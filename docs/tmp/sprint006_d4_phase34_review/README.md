# Sprint 006 D4 Phase 3/4 — review bundle

**Purpose.** Third-party review of the **technical evidence** for the official baseline run and Phase 4 blind verification.

**Prior evidence verdict (V-1…V-20 + initial S1–S4): `ACCEPTED`**

**Source-reconstruction reverification:** `PASS` — **184 PASS / 0 FAIL / 1 N/A** (audit-local verifier; commit `326e13d`)

**Phase 4 acceptance:** accepted through `326e13de46ab13ddc5f584e0bfc3fb431052fbd2` (documentation commit after review)

**Current lifecycle status:** `ACTIVE — D0/D1/D2/D3 COMPLETE; D4 PHASE 4 ACCEPTED; PHASE 5 AUTHORIZED — NOT STARTED`

**Phase 5:** Authorized under the existing accepted D4 plan; **not started** by the acceptance record. Aggregate economics remain closed until Phase 5 is executed.

This folder is a **copy for review**. Canonical artifacts remain outside the repository under:

- Official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- Verification: `C:/MomentumCVG_env/runs/sprint006_d4_verification_20260823T204430Z`

Repo memo:
[`docs/sprint_memos/sprint006_d4_baseline_execution_evidence.md`](../../sprint_memos/sprint006_d4_baseline_execution_evidence.md)

## What was done

1. **Phase 3** — Official twin-fill baseline at execution commit `e205b9a`, frozen contract, into the `RUN_DIR` above (~3h wall clock).
2. **Phase 4** — Blind V-1…V-20 + frozen S1–S4; initial verdict `ACCEPTED`.
3. **Source-reconstruction reverification** — Corrected audit-local §7.4 audit; result `PASS` (184 PASS / 0 FAIL / 1 N/A).
4. **Phase 4 acceptance** — Accepted after review of `326e13d`; Phase 5 authorized but not started.

## What is in this bundle

| Path | Contents |
|------|----------|
| `ARTIFACT_DIGESTS.tsv` | SHA-256 + size for all 17 official run files (hashes only) |
| `verification/EVIDENCE_VERDICT.md` | Outside-repo prior Phase 4 verdict record |
| `verification/phase4_v_checks.json` | V-1…V-20 expected/observed/verdict |
| `verification/phase4_v_checks_stdout.txt` | V-check transcript |
| `verification/phase4_s1_s4_audit.json` | Initial S1–S4 selection + partial reconstruction |
| `verification/phase4_s1_s4_stdout.txt` | Initial S1–S4 transcript |
| `verification/phase4_source_reconstruction_audit.py` | Corrected audit-local §7.4 audit script |
| `verification/phase4_source_reconstruction_audit.json` | Complete machine-readable audit rows + coverage checklist |
| `verification/phase4_source_reconstruction_audit.md` | Concise human summary (generated at audit time) |
| `verification/phase4_source_reconstruction_stdout.txt` | Expanded audit transcript |
| `verification/run_receipt.json` | Run identity / completeness (not an economic report) |
| `verification/paths.txt` | Stamp / RUN_DIR / VERIFY_DIR paths |

Generated audit artifacts (`phase4_source_reconstruction_audit.json`, `.md`, transcript) retain their time-of-generation lifecycle wording where applicable; final Phase 4 acceptance is recorded in the repo memo §8 and sprint agenda.

## Explicitly excluded (still closed / too large)

- `decision_report.json` / `decision_report.md` (aggregate economics — Phase 5)
- `run_summary_*.json` economic fields
- All Parquet trade/leg/funnel/date logs (remain only under `$RUN_DIR`)
- Source datasets (liquidity panel, features, A1/A2)

Sample-level numeric fields appear inside the audit JSON files **as audit evidence only**, not as strategy performance analysis.

## Review focus (historical)

This bundle supported third-party review before Phase 4 acceptance. Review questions included V-1…V-20 coverage, frozen S1–S4 selection, and complete §7.4 source reconstruction. Phase 4 is now accepted; Phase 5 may proceed under the accepted plan.
