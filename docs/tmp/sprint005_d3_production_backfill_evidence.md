# Sprint 005 D3 — production feature backfill evidence

**Production verdict:** `PASS / PUBLISHED`

## Identity

| Field | Value |
|-------|-------|
| Code SHA (receipt `repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| Config | `configs/feature_backfill_v1.json` |
| Config SHA-256 | `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` |

## Command

```text
C:\MomentumCVG_env\venv\Scripts\python.exe scripts\backfill_features.py --observations C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.parquet --d2-lineage C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.lineage.json --config C:\MomentumCVG\configs\feature_backfill_v1.json --output-root C:\MomentumCVG_env\derived\e2c1f8fd44d72176 --expected-snapshot-id e2c1f8fd44d72176 --expected-build-id 20260724T045049097520Z_40b16886
```

| Field | Value |
|-------|-------|
| Python | `C:\MomentumCVG_env\venv\Scripts\python.exe` / 3.13.7 |
| Start (UTC) | `2026-08-08T23:47:52Z` |
| End (UTC) | `2026-08-09T00:31:19Z` |
| Duration | `00:43:26` (2606.241 s) |
| Log | `C:\MomentumCVG_env\ops_logs\d3_feature_backfill_20260808T164752.log` |
| Process stdout/stderr | empty (success path is silent) |
| Wrapper note | `Start-Process` returned a null `ExitCode` to the PowerShell wrapper (logged as empty; wrapper therefore exited 1). Independent acceptance below confirms a complete successful publication. |

## Pre-run verification

| Suite | Result |
|-------|--------|
| `tests/unit/test_backfill_features.py` | 65 passed |
| `tests/unit/test_feature_backfill_v1_contract.py` | 12 passed |
| Full `pytest -q` | 1478 passed, 1 skipped |
| `git diff --check` | clean |
| Working tree before run | clean at `131d0ac` |

## Accepted D2 input (unchanged)

| Field | Value |
|-------|-------|
| Observations | `C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.parquet` |
| D2 lineage | `C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.lineage.json` |
| `output.file_sha256` | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` |
| `output.row_count` / `key_count` | `1063995` / `1063995` |
| `output.output_key_digest` | `faa7e943e71b8aeaf4ea354713ab5558f44a03c9c211c6a68f53236acaa2cced` |

## Published outputs

| Path | Result |
|------|--------|
| `...\features.building\` | absent after success |
| `...\features\` | present; exactly **281** `features_{max}_{min}.parquet` files |
| `...\features_backfill_v1.lineage.json` | present; `status = "complete"` |
| Receipt SHA-256 | `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` |
| Receipt `created_at_utc` | `2026-08-09T00:31:19Z` |
| Receipt windows | full ordered 281-window list `(6,2)` … `(60,24)` |
| Receipt `files[]` | 281 ordered per-file records with SHA-256 digests |

## Independent acceptance (read-only)

Performed after publication with a temporary verifier outside the repository (removed afterward).

| Check | Result |
|-------|--------|
| Staging absent; final `features/` + receipt present | PASS |
| Exact 281 expected filenames; no extras | PASS |
| Receipt identities (repo SHA, snapshot/build, D2 digests/counts, config SHA, 281 windows/files, `status=complete`) | PASS |
| Independent SHA-256 of all 281 published files vs receipt | **281 / 281** PASS |
| Parquet metadata (readable; exact six columns; row count `1063995`) for all 281 | **281 / 281** PASS |
| Sentinels `features_6_2`, `features_42_8`, `features_60_24`: unique non-null keys, exact canonical D2 key equality, deterministic sort, six-column schema, NA features preserved | PASS |

## Confirmations

* Production D2 observations and lineage were not modified.
* No production code was changed during this run.
* D4 was not started.
* This document is the only repository change for the evidence commit.
* Nothing was pushed as part of this evidence step.
