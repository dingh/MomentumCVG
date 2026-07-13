# Sprint 004 — C7 point-in-time universe closeout

**Status:** Closed — **C7 accepted**  
**Closed:** 2026-07-12  
**Reality map:** [c7_0_pit_universe_reality_map.md](../tmp/c7_0_pit_universe_reality_map.md)  
**Design memo:** [c7_1_pit_universe_design_memo.md](../tmp/c7_1_pit_universe_design_memo.md)  
**Accepted production evidence:** [c7_pit_universe_audit.md](../tmp/c7_pit_universe_audit.md) (C7.4R)

---

## 1. Lineage

| Label | Commit | Meaning |
|-------|--------|---------|
| **Accepted design/documentation head** | `9dc7ed2c2d1d8c166b35d693ef3b44bd044bae93` | C7.0/C7.1 follow-up documentation corrections |
| **Strict prior-snapshot implementation** | `9cf5ba13996fbecc0950ebdc190656b58e99` | C7.2 — S1 `<=` → `<`; audit module foundation; contract + sprint PIT wording |
| **Accepted final audit implementation** | `e0cf020430465c9f5a033017d5736dc819e681ff` | C7.2A–C7.3B hardening + C7.5A complete-membership certification repair |
| **Accepted corrected production-evidence head** | `e92ceb23e8ebd0eaf36f10757e01b1abd0fa4d1a` | C7.4R evidence-only commit; supersedes initial C7.4 evidence |
| **Superseded initial production-evidence head** | `47060d56a74885a5aca37413aebc541ccbbd000d` | Initial C7.4 production audit — superseded by C7.4R after C7.5A repair |

**Accepted implementation head:** `e0cf020430465c9f5a033017d5736dc819e681ff`  
**Accepted production-evidence head:** `e92ceb23e8ebd0eaf36f10757e01b1abd0fa4d1a`

`47060d56a74885a5aca37413aebc541ccbbd000d` is **superseded evidence**.  
`e92ceb23e8ebd0eaf36f10757e01b1abd0fa4d1a` is the **accepted C7.4R evidence**.

The evidence-only commit did **not** change accepted implementation code.

### Implementation history (intermediate commits)

| Label | Commit | Meaning |
|-------|--------|---------|
| C7.2A audit hardening | `28b63234c0c6c4af5b41b81e0e962e95525e90de` | Additional artifact validation and audit robustness |
| C7.3 standalone audit CLI | `b9e08c9c98296fbecc0950ebdc971631ffdf7fbf` | `scripts/audit_pit_universe.py` + markdown rendering |
| C7.3A fail-closed CLI hardening | `c7e38d76b3cd017793b845619d841358de2650bc` | CLI path/read/write and controlled FAIL conversion |
| C7.3B provenance validation hardening | `77b61eca6442c869a909138deba17941e6c1da2d` | Weekly/panel provenance checks; exact invalid counts |
| C7.5A complete-membership repair | `e0cf020430465c9f5a033017d5736dc819e681ff` | Sample superset certification carries complete selected membership |

---

## 2. C7 objective and answer

**C7 question:**

> Can the weekly trading universe be proven point-in-time from the accepted rolling liquidity panel without future leakage, stale-snapshot ambiguity, nondeterminism, or silent missing-liquidity acceptance?

**Answer:** **Yes**, for the accepted canonical artifact envelope and production artifacts identified in the C7.4R evidence.

**What C7 proves:**

- A3 liquidity-panel integrity
- strict PIT snapshot selection
- S1/reference parity
- bounded independent rolling recomputation
- future invariance
- deterministic membership
- explicit missing-liquidity classification
- sample and full-history liquid-superset coverage

**What C7 does not prove:**

- feature correctness
- momentum/CVG alpha
- strategy profitability
- Sharpe
- trade construction
- sizing
- execution
- portfolio behavior
- paper/live readiness

---

## 3. Locked final semantics

### Snapshot resolution (production rule)

```text
resolved_snapshot_date = max(month_date where month_date < trade_date)
```

- **Global cross-section** — one snapshot date for the entire market, not per-ticker carry-forward.
- **Strict prior snapshot** — the latest completed weekly snapshot strictly before the trade date.
- **Same-day snapshot prohibited** — `resolved_snapshot_date == trade_date` is a blocking FAIL.
- **Future snapshot prohibited** — `resolved_snapshot_date > trade_date` is a blocking FAIL.
- **Before-first-snapshot** — when no `month_date < trade_date` exists, S1 returns an explicit empty universe with correct output columns.

### Ranking and eligibility (unchanged from pre-C7 except snapshot timing)

**Eligible** (all must hold):

```text
has_valid_atm_pair == True
dvol non-null
spread non-null
```

**Ranks** (on eligible cross-section before threshold filter):

```text
dvol_rank_pct   = atm_straddle_dollar_vol.rank(ascending=True,  method="average", pct=True)
spread_rank_pct = atm_spread_pct.rank(ascending=False, method="average", pct=True)
```

**Selection:**

```text
dvol_rank_pct   >= 1 - dvol_top_pct
AND
spread_rank_pct >= 1 - spread_bottom_pct
```

### Supported production artifact envelope

```text
requested dvol_top_pct        <= stamped dvol_top_pct
requested spread_bottom_pct   <= 1.0

accepted canonical baseline:
dvol_top_pct      = 0.20
spread_bottom_pct = 1.0
```

Broader `dvol_top_pct` requests are **unsupported** by the accepted current superset unless broader artifacts are built and audited.

---

## 4. Implementation summary

| Component | Responsibility |
|-----------|----------------|
| `src/backtest/pipeline.py` | Production S1 — `step1_get_universe` with strict prior-snapshot rule |
| `src/data/pit_universe_audit.py` | Pure audit module: artifact validation, independent reference universe, rolling provenance recompute, envelope enforcement, superset coverage, report assembly |
| `scripts/audit_pit_universe.py` | Standalone CLI — discovers/maps samples, runs full audit, writes markdown report, exit codes 0/1/2 |
| `tests/contract/test_step1_universe_contract.py` | S1 contract tests — strict `<` snapshot semantics on synthetic fixtures |
| `tests/unit/test_pit_universe_audit.py` | Synthetic audit-module tests (reference, rolling, envelope, coverage, determinism) |
| `tests/unit/test_audit_pit_universe_cli.py` | CLI exit-code, path, and report-file tests |
| `docs/surface_engine_data_contract.md` | S1 invariant I1 updated to `max(month_date < trade_date)` in same C7.2 commit as production change |

**Key distinctions:**

- Production S1 is called **as the production path**; the audit independently computes a **reference universe** and compares membership and ranks.
- Rolling provenance is **independently recomputed from weekly observations** on bounded substantive samples — not a tautological weekly-row filter.
- Diagnostic examples (exclusions, mismatches, invalid provenance values) remain **capped** by `--max-examples` (default 20).
- Sample superset certification uses **complete uncapped membership** — `selected_tickers` carries the full selected set; `selected_count` must match coverage certification count.
- Full-history superset coverage checks the **canonical complete history** (477 snapshots) — universe/superset coverage, not per-row rolling recomputation of all 2.4M panel rows.
- The bounded 20-ticker rolling-provenance sample per substantive case is **not** full-panel rolling recomputation.

---

## 5. Material design-to-implementation refinements

Compared with the original C7.1 design, the accepted final implementation includes these material refinements (verified in final code):

| Design area | Refinement |
|-------------|------------|
| Date parsing | Timezone-aware dates **fail closed** (rejected with explicit FAIL) rather than risking silent calendar-day shifts |
| Membership hashing | `_stable_num` uses full-precision float hex representation — stable across platforms |
| Error handling | Artifact validation and runtime errors are converted into **controlled audit FAILs** with structured `ArtifactCheckResult` records rather than uncaught exceptions in evidence runs |
| Provenance invalid counts | CLI tracks **exact** `invalid_count` for provenance integer/date/boolean checks; displayed examples remain capped |
| Sample superset certification | `PitResolutionResult.selected_tickers` carries **complete selected membership** separately from capped diagnostic classifications; `check_sample_superset_coverage_consistency` verifies count alignment |
| Weekly artifact validation | `validate_weekly_artifact` is called once and reused for rolling recompute and future-invariance — invalid weekly input never silently consumed |

These are accepted defensive hardening, not unresolved design drift.

---

## 6. Production evidence

**Canonical evidence report:** [c7_pit_universe_audit.md](../tmp/c7_pit_universe_audit.md) (C7.4R, accepted 2026-07-13 UTC execution on implementation head `e0cf020`).

### Input artifacts and hashes

| Artifact | Path | SHA-256 |
|----------|------|---------|
| Panel | `C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet` | `67e30956cd78bea97e9f90bfbd699f5e12e302e30db0e91912abd564dcf778de` |
| Weekly observations | `C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet` | `40f507fb165add28c5fdfb1dc9ef7a2e0176874f3bc240732bc538b990673f42` |
| Liquid ticker superset | `C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv` | `e3094e6f1c8138ef5934f2b3158a37b3cc92ea01250c0dcfb8703ec03eb4b68a` |

### Inventory

| Metric | Count |
|--------|------:|
| Panel rows | 2,434,339 |
| Weekly rows | 2,438,191 |
| Panel snapshots | 477 |
| Weekly dates | 488 |
| Liquid tickers | 2,783 |

### Three production samples

| Label | Target snapshot | Trade date | Selected | Result |
|-------|-----------------|------------|---------:|--------|
| missing_or_new_liquidity | 2017-01-06 | 2017-01-13 | 737 | PASS |
| boundary_or_gap | 2017-04-21 | 2017-04-28 | 728 | PASS |
| normal | 2021-09-17 | 2021-09-24 | 735 | PASS |

For all three samples:

- `target_snapshot == resolved_snapshot`
- `resolved_snapshot < trade_date`
- production/reference parity = **true**
- sample superset coverage count equals complete selected count
- missing from liquid superset = **none**

### Rolling evidence (bounded)

- 3 substantive rolling-provenance samples (one per mapped case)
- 20 deliberately bounded checked tickers per sample
- stored-panel recomputation match = **true**
- future invariance = **true**
- field mismatch count = **0**

Rolling recomputation was **bounded to substantive samples** — not all tickers and all snapshots.

### Full-history evidence

| Metric | Value |
|--------|-------|
| Snapshots checked | 477 |
| Unique selected tickers | 2,783 |
| Missing ticker count | 0 |
| Canonical params | (0.20, 1.0) |

The complete-history check was **universe/superset coverage** across all snapshots — not independent recomputation of all 2.4M panel rows from weekly observations.

### Final audit disposition

| Field | Value |
|-------|-------|
| Exit code | 0 |
| Overall status | PASS |
| Blocking failures | none |
| Warnings | none |
| Input hashes unchanged before/after | yes |
| Runtime | 545.6 seconds |

---

## 7. Test evidence

**Accepted focused test result (recorded at implementation head `e0cf020`):**

```text
127 passed, 1 skipped in 6.26s
```

**Focused gate:**

```text
tests/unit/test_pit_universe_audit.py
tests/unit/test_audit_pit_universe_cli.py
tests/contract/test_step1_universe_contract.py
```

This is **local recorded test evidence**.  
**No GitHub Actions run is claimed for C7.**  
The full repository pytest suite remains Sprint 004 blocker #11 and is **not** claimed as completed by C7.

Tests were **not** rerun for this C7.6 documentation-only task.

---

## 8. C7 acceptance-criteria mapping

| C7.1 acceptance criterion | Final evidence | Status |
|-----------------------------|----------------|--------|
| C7.0/C7.1 design accepted | Reality map + design memo + follow-up commit `9dc7ed2` | ✓ |
| Strict prior-snapshot rule implemented (`<`) | C7.2 `9cf5ba1`; `pipeline.py`; contract tests | ✓ |
| Contract and sprint PIT wording updated with C7.2 | `surface_engine_data_contract.md` § S1 I1; sprint agenda PIT section | ✓ |
| Independent rolling recomputation passed | C7.4R — 3 bounded samples, 0 field mismatches | ✓ |
| Future invariance passed | C7.4R — all 3 rolling samples `future_invariance_pass=True` | ✓ |
| Supported envelope enforced | C7.4R — stamped 0.20/1.0; requested 0.20/1.0; `supported=True` | ✓ |
| Full-history canonical superset coverage passed | C7.4R — 477 snapshots, 2783 unique tickers, 0 missing | ✓ |
| Automatic sample mapping passed | C7.4R — 3 distinct cases; `target == resolved` for all | ✓ |
| Missing liquidity explicit | Exclusion counts per sample; never silent PASS | ✓ |
| Focused unit/contract/CLI tests passed | 127 passed, 1 skipped in 6.26s | ✓ |
| Substantive production report archived | [c7_pit_universe_audit.md](../tmp/c7_pit_universe_audit.md) | ✓ |
| No S2–S8/A4/backtest/Sharpe/portfolio changes | Scope preserved across C7 commits | ✓ |
| Sprint blocker #6 closed | PIT universe validation — CLOSED (C7, 2026-07-12) | ✓ |
| Sprint 004 remains active; C3 deferred until after C8 | This closeout does not close Sprint 004 | ✓ |

---

## 9. Scope preservation

C7 did **not**:

- modify S2–S8
- modify ranking thresholds beyond the approved strict-snapshot change
- modify A4 features
- wire the audit into `refresh_weekly_inputs.py`
- run a strategy backtest
- evaluate Sharpe
- modify production liquidity artifacts
- modify accepted C4/C5/C6 evidence

The **only approved production behavior change** was:

```text
S1 snapshot resolution: <= changed to <
```

---

## 10. Deferred work

| Item | Owner |
|------|-------|
| Refresh/validate integration of PIT audit | C3 after C8 |
| Refresh orchestration (`refresh --dry-run`, bounded `refresh`) | C8 |
| `v1_universe_protocol` global-cross-section alignment | C9 |
| Runbook and CLI plan cleanup | C9 |
| Full Sprint 004 pytest gate | Sprint 004 closeout (blocker #11) |
| A4 features and straddle history | Sprint 005 |
| Real-data S1→S8 backtest | Sprint 006 |
| Tier B / Sharpe go-no-go | Sprint 007 |

Accepted C7 decisions are **not** reopened in deferred work.

---

## 11. Remaining Sprint 004 work

**C7 is closed.**

Sprint 004 **remains active**.

**Next implementation task:** C8.

After C8: C3 validation umbrella.

Then C9 documentation/runbook/CLI cleanup and C10 Sprint closeout.

C8, C3, C9, and C10 are **not** marked complete by this memo.

---

## 12. References

| Document | Topic |
|----------|-------|
| [c7_0_pit_universe_reality_map.md](../tmp/c7_0_pit_universe_reality_map.md) | C7.0 reconnaissance |
| [c7_1_pit_universe_design_memo.md](../tmp/c7_1_pit_universe_design_memo.md) | C7.1 design (canonical) |
| [c7_pit_universe_audit.md](../tmp/c7_pit_universe_audit.md) | C7.4R accepted production evidence |
| [surface_engine_data_contract.md](../surface_engine_data_contract.md) | A3 + S1 data contract |
| [current_sprint.md](../agenda/current_sprint.md) | Sprint 004 active agenda |
| [004_c4_liquidity_panel.md](004_c4_liquidity_panel.md) | Upstream C4 liquidity panel closeout |
| [004_c5_adjusted_liquid.md](004_c5_adjusted_liquid.md) | Upstream C5 adjusted-liquid closeout |
| [004_c6_option_surface.md](004_c6_option_surface.md) | Upstream C6 option-surface closeout |

The C7 `docs/tmp` files are **retained evidence** and should not be rewritten merely for closeout consistency.
