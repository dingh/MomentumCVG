# C6.4 — Surface Audit Summary

**Generated:** 2026-07-11 (UTC)  
**Audit commit:** `723a17961eb8b6e8b6dba33c9d8620f3a6b9959a`

## Pass verdicts

| Pass | Artifacts | Verdict |
|------|-----------|---------|
| Pass 1 — existing real cache | `option_surface_meta_weekly_2018_2026.parquet` + quotes pair | **PASS** |
| Pass 2 — fresh C6 smoke | `c6_4_surface_smoke/option_surface_meta_weekly_2024_2024.parquet` + quotes pair | **PASS** |

## Scope (both passes)

- **Frequency:** weekly
- **Tickers:** AAPL, MSFT, NVDA, SPY, QQQ (all present)
- **Date window:** 2024-01-01 .. 2024-03-31
- **Resolved entry dates:** 13 (2024-01-05 .. 2024-03-28)
- **Meta rows (scoped):** 65 (5 tickers × 13 weeks)

## Key findings

### Duplicate triage

- **A1 grain:** 0 duplicate keys (both passes)
- **A2 grain:** 0 duplicate keys (both passes)
- No identical or conflicting duplicates in the bounded scope.

### Weekly expiry

- **Pass 1 (legacy):** 65/65 eligible rows match `target_weekly_expiry_from_schedule`; 0 silent mismatches.
- **Pass 2 (fresh):** 65/65 exact target matches; 0 silent fallback violations.

### Readiness (scoped sample)

- `surface_valid`: 65/65 (100%)
- `straddle_ready`: 65/65 (100%)
- `ironfly_candidate_ready`: 65/65 (100%)
- `ironcondor_candidate_ready`: 65/65 (100%)

### Legacy notes (Pass 1)

- Historical producer commit and upstream lineage: **unknown** (documented, not invented).
- No blocking legacy findings in this bounded window.

## Tests recorded

| Suite | Result |
|-------|--------|
| C6.1 weekly expiry / CLI / diagnostic | 41 passed |
| C6.2 contract + audit (+ C6.4 helpers) | 36 passed |
| C6.3 readiness | 39 passed |
| C6.4 audit helpers | 4 passed |

## Producer fix required before C6.6?

**No** — bounded scope shows clean grains, strict weekly expiry pairing, and full readiness on both legacy cache (this window) and fresh producer output. No duplicate repair needed in C6.4.

## C6.4 closeout readiness

Evidence reports archived:

- [c6_4_real_cache_surface_audit.md](c6_4_real_cache_surface_audit.md)
- [c6_4_smoke_surface_audit.md](c6_4_smoke_surface_audit.md)

**Conservative conclusion:** C6.4 supplies the required real-artifact A1/A2 audit evidence for defensive review. This does **not** certify strategy backtest readiness or full-universe surface trust.
