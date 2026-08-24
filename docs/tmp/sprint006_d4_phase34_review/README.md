# Sprint 006 D4 Phase 3/4 — review bundle

**Purpose.** Third-party review of the **technical evidence** for the official baseline run and Phase 4 blind verification.

**Evidence verdict (already recorded): `ACCEPTED`**

This folder is a **copy for review**. Canonical artifacts remain outside the repository under:

- Official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- Verification: `C:/MomentumCVG_env/runs/sprint006_d4_verification_20260823T204430Z`

Repo memo (already on `main` / commit `a621987`):  
[`docs/sprint_memos/sprint006_d4_baseline_execution_evidence.md`](../../sprint_memos/sprint006_d4_baseline_execution_evidence.md)

## What was done

1. **Phase 3** — Official twin-fill baseline at execution commit `e205b9a`, frozen contract, into the `RUN_DIR` above (~3h wall clock).
2. **Phase 4** — Blind V-1…V-20 + frozen S1–S4 source reconstruction; verdict `ACCEPTED`.
3. **Docs** — Evidence memo + agenda/plan status updated in `a621987`.

## What is in this bundle

| Path | Contents |
|------|----------|
| `ARTIFACT_DIGESTS.tsv` | SHA-256 + size for all 17 official run files (hashes only) |
| `verification/EVIDENCE_VERDICT.md` | Outside-repo verdict record |
| `verification/phase4_v_checks.json` | V-1…V-20 expected/observed/verdict |
| `verification/phase4_v_checks_stdout.txt` | V-check transcript |
| `verification/phase4_s1_s4_audit.json` | S1–S4 selection + reconstruction audit rows |
| `verification/phase4_s1_s4_stdout.txt` | S1–S4 transcript |
| `verification/run_receipt.json` | Run identity / completeness (not an economic report) |
| `verification/paths.txt` | Stamp / RUN_DIR / VERIFY_DIR paths |

## Explicitly excluded (still closed / too large)

- `decision_report.json` / `decision_report.md` (aggregate economics — Phase 5)
- `run_summary_*.json` economic fields
- All Parquet trade/leg/funnel/date logs (remain only under `$RUN_DIR`)
- Verification Python scripts

Sample-level numeric fields appear inside `phase4_s1_s4_audit.json` **as audit evidence only**, not as strategy performance analysis.

## Review focus

- Does V-1…V-20 coverage match the D4 plan §§7.1–7.2?
- Is S1–S4 selection frozen-rule correct (including S3=N/A)?
- Is source reconstruction adequate for trust?
- Any gap before authorizing Phase 5?

**Phase 5 is not authorized** by packaging this review bundle.
