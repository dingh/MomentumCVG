# C6.4 — Surface Audit Summary

**Generated:** 2026-07-11 (UTC)  
**C6.4 audit implementation:** `c75417de79ae19ed8bcdd3fa9d0afce6045275f8`

## Pass verdicts

| Pass | Artifacts | Verdict | Rationale |
|------|-----------|---------|-----------|
| Pass 1 — existing real cache | `option_surface_meta_weekly_2018_2026.parquet` + quotes pair | **WARN** | Bounded data checks pass; historical producer/upstream lineage unknown |
| Pass 2 — fresh C6 smoke | `c6_4_surface_smoke/option_surface_meta_weekly_2024_2024.parquet` + quotes pair | **PASS** | Contract, strict expiry, duplicate, and readiness checks pass |

## Scope (both passes)

- **Frequency:** weekly
- **Tickers:** AAPL, MSFT, NVDA, SPY, QQQ (all present)
- **Date window:** 2024-01-01 .. 2024-03-31
- **Resolved entry dates:** 13 (2024-01-05 .. 2024-03-28)
- **Meta rows (scoped, raw):** 65 (5 tickers × 13 weeks)

## Key findings

### Duplicate triage

- **A1 grain:** 0 duplicate keys (both passes)
- **A2 grain:** 0 duplicate keys (both passes)
- No identical or conflicting duplicates in the bounded scope.

### Weekly expiry (explicit classification)

- **Pass 1:** 65 exact target matches; 0 silent mismatches; 0 missing expiry without failure
- **Pass 2:** 65 exact target matches; 0 silent fallback violations; 0 missing expiry without failure
- Ordinary producer failures (e.g. `no_spot_price`) are counted separately and are **not** labeled silent expiry fallback.

### Readiness (normalized metrics view)

- `surface_valid`: 65/65 (100%)
- `straddle_ready`: 65/65 (100%)
- `ironfly_candidate_ready`: 65/65 (100%)
- `ironcondor_candidate_ready`: 65/65 (100%)

### Lineage

| Field | Pass 1 | Pass 2 |
|-------|--------|--------|
| Audit implementation | `c75417de79ae19ed8bcdd3fa9d0afce6045275f8` | same |
| Repository HEAD when producer ran | n/a (legacy cache) | `0a386f2517deff8be116f4729abf7e2cfc09531d` |
| Strict weekly-expiry producer | n/a (unknown legacy) | `af9d9a08772b6e8c82c32acc39cbc84b32bb4326` |
| Historical producer commit | **unknown** → WARN | n/a |

Pass 1 WARN is **audit-policy / lineage confidence**, not a bounded integrity failure.

## Tests recorded

| Suite | Result |
|-------|--------|
| C6.1 weekly expiry / CLI / diagnostic | 41 passed in 0.41s |
| C6.2 contract | 25 passed in 0.15s |
| C6.3 readiness | 39 passed in 0.08s |
| C6.4 helpers + audit CLI integration | 33 passed in 0.86s |

## Producer fix or artifact regeneration required?

**No.** Existing bounded smoke artifacts were reused unchanged. This commit corrects test command/result alignment in the C6.4 evidence reports only.

## C6.4 closeout readiness

Evidence reports archived:

- [c6_4_real_cache_surface_audit.md](c6_4_real_cache_surface_audit.md)
- [c6_4_smoke_surface_audit.md](c6_4_smoke_surface_audit.md)

**Conservative conclusion:** C6.4 supplies real-artifact A1/A2 audit evidence suitable for defensive review. Artifact correctness in the bounded window is good; Pass 1 lineage confidence is intentionally limited. This does **not** certify strategy backtest readiness.
