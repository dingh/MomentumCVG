# Sprint 005 D4 — production feature quality audit evidence

**Production verdict:** `PASS / ACCEPT`

## Frozen provenance

| Field | Value |
|-------|-------|
| Snapshot ID | `e2c1f8fd44d72176` |
| Build ID | `20260724T045049097520Z_40b16886` |
| D3 evidence SHA | `816e28f7b63cb9668de94f9cee037d76758fff71` |
| D3 producer SHA (`d3_producer_repo_sha`) | `131d0ac05e1e57749d3095923927a394fdcbc25b` |
| D4 audit SHA (`d4_audit_repo_sha`) | `22a8375d2d6c3b2dbd661697d9524548ea6def9a` |
| Config | `configs/feature_backfill_v1.json` |
| Config SHA-256 | `764056ce7153751d93c1764b1b4cae13a521bf5c3baee729db30bb69543132dd` |
| D3 receipt SHA-256 | `c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5` |
| D2 observations SHA-256 | `f0c1461ea4643154d6b26393159d2b9fc78ce2f9cd5dbdde1a0d1e3d700845c9` |
| Audit JSON path | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json` |
| Audit JSON SHA-256 | `6737ab2073be4aab874454faf849139031bf66031e80ffc81b712ac2edff2f2c` |
| Audit JSON size | `808256` bytes |

D3 publication was not re-validated (no 281-file re-hash).

## Execution

```text
C:\MomentumCVG_env\venv\Scripts\python.exe scripts\audit_feature_quality.py --features-dir C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features --d3-receipt C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features_backfill_v1.lineage.json --observations C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.parquet --d2-lineage C:\MomentumCVG_env\derived\e2c1f8fd44d72176\straddle_observations_weekly.lineage.json --config C:\MomentumCVG\configs\feature_backfill_v1.json --output-json C:\MomentumCVG_env\derived\e2c1f8fd44d72176\features_quality_audit_v1.json --expected-snapshot-id e2c1f8fd44d72176 --expected-build-id 20260724T045049097520Z_40b16886 --expected-d3-repo-sha 131d0ac05e1e57749d3095923927a394fdcbc25b
```

| Field | Value |
|-------|-------|
| Start (UTC) | `2026-08-09T05:50:37Z` |
| End (UTC) | `2026-08-09T07:06:22Z` |
| Duration | `01:15:44` (`4544.84` s) |
| Exit code | `0` |
| Stdout/stderr | empty (success path) |
| `.tmp` sibling | absent |

## Coverage (all 281 windows)

| Metric | Min rate | Max rate |
|--------|----------|----------|
| Momentum non-null | `0.4148891677122543` | `0.6998153186810089` |
| CVG non-null | `0.4567653043482347` | `0.7250193844895888` |
| Both non-null | `0.41449348916113327` | `0.6993989633409932` |

Window order: `(6,2)` … `(60,24)`; `window_count=281`. Full per-window table is in the JSON.

### Sentinels

| Window | mom rate | cvg rate | both rate |
|--------|----------|----------|-----------|
| `(6,2)` | `0.5483230654279391` | `0.6084558668038853` | `0.5479067100879234` |
| `(42,8)` | `0.6598743415147628` | `0.6903246725783486` | `0.659463625298991` |
| `(60,24)` | `0.6387670994694524` | `0.6674533244987053` | `0.6383714209183314` |

## Baseline `(42,8)` ready interval

Frozen rule: 43rd → last common ordered D2 entry date (`common_date_count=445`).

| Field | Value |
|-------|-------|
| `ready_start` | `2018-10-26` |
| `ready_end` | `2026-07-10` |
| Rows inside interval | `963573` |
| Mom non-null | `659320` / rate `0.6842449923358168` |
| CVG non-null | `687528` / rate `0.7135193700944298` |
| Both non-null | `658917` / rate `0.6838267572877198` |

Interval bounds were not moved based on coverage.

### Baseline full-panel missingness (`n_rows=1063995`)

| Dimension | Counts |
|-----------|--------|
| Structural | `no_slots=19128`, `truncated_window=81294`, `full_window=963573` |
| Momentum economic (slots available) | `zero_finite=342764`, `partial_finite=549478`, `all_available_finite=152625` |
| CVG economic (slots available) | `zero_finite=310365`, `partial_finite=561830`, `all_available_finite=172672` |
| Mom count summary (finite) | min `0`, median `5`, max `35`; full-window share `0.13299498587869305` |
| CVG count summary (finite) | min `0`, median `7`, max `35`; full-window share `0.15027420241636474` |

Structural warm-up (`no_slots` / truncated geometry) is distinct from missing economics. Dominant economic labels are `partial_finite` and `zero_finite`, consistent with D2 null `return_pct` / `vol_gap` on the full A1 key grid—not an unexplained usability defect for acceptance.

## PIT evidence (compact `j+2` proof)

Exhaustive for the frozen grid (`min_lag ≥ 2`).

| Signal | Eligible | Checked | Violations | Min safety gap (days) |
|--------|----------|---------|------------|------------------------|
| Momentum | `292470` | `292470` | `0` | `6` |
| CVG | `312033` | `312033` | `0` | `6` |

## Residual limitations

D4 does not select windows, evaluate returns or Sharpe, test predictive performance, run the D5 `SurfaceRunner` smoke, or start a Sprint 006 backtest.
