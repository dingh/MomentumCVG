# Sprint 004 — C6 option-surface layer closeout

**Status:** Closed — **C6 accepted**  
**Closed:** 2026-07-11  
**Design memo:** [docs/tmp/c6_1_option_surface_design_memo.md](../tmp/c6_1_option_surface_design_memo.md)

### Lineage

| Label | Commit | Meaning |
|-------|--------|---------|
| **Accepted C6.4 evidence head** | `70527cade605a0f1ee2f90f600f6ae9884018eda` | Repository head at which the accepted C6.4 evidence reports were finalized |
| **Fresh-smoke producer-run repository HEAD** | `0a386f2517deff8be116f4729abf7e2cfc09531d` | Full repository HEAD used to generate the C6.4 Pass 2 fresh bounded smoke artifacts |
| **Strict weekly-expiry implementation** | `af9d9a08772b6e8c82c32acc39cbc84b32bb4326` | Accepted C6.1C commit that implements strict calendar-paired weekly expiry (a component of the producer, not the full smoke-run HEAD) |
| **C6.4 audit implementation** | `c75417de79ae19ed8bcdd3fa9d0afce6045275f8` | Accepted C6.4 audit-helper and CLI implementation used for both passes |

---

## 1. Deliverable

C6 established a **three-layer trust gate** for A1/A2 option-surface artifacts:

| Layer | Scope |
|-------|--------|
| **Layer 1 — producer semantics** | How `OptionSurfaceBuilder` and `precompute_option_surface.py` build rows (weekly entry schedule, strict weekly expiry, failure vocabulary, CLI safety) |
| **Layer 2 — A1/A2 artifact integrity** | Schema, `surface_valid` invariant, join/grain, settlement fields, date alignment on on-disk parquets |
| **Layer 3 — downstream assembly-readiness** | Whether valid surfaces can supply straddle / iron-fly / iron-condor legs at S3 assembly time |

**Canonical artifacts:**

| ID | On-disk pattern |
|----|-----------------|
| **A1** | `option_surface_meta_*.parquet` |
| **A2** | `option_surface_quotes_*.parquet` |

**Core semantic decision (unchanged):**

```text
surface_valid
⇔ has_body_call
  AND has_body_put
  AND n_surface_quotes > 0
```

`surface_valid` remains a **general surface-validity flag**. Iron-fly and iron-condor readiness are **derived audit metrics** (`ironfly_candidate_ready`, `ironcondor_candidate_ready`), not redefinitions of `surface_valid`.

**Sprint question answered:**

> Can we trust A1/A2 option-surface artifacts well enough that downstream `SurfaceRunner` / S3 assembly is not operating on silently broken inputs?

**C6 proves:** producer semantics · artifact integrity · assembly-readiness auditability · bounded real-artifact evidence.

**C6 does not prove:** strategy profitability · Sharpe · parameter quality · execution quality · full-universe opportunity capacity · paper/live readiness.

---

## 2. C6 decision and implementation history

| Phase | Result | Commit / evidence | Summary |
|-------|--------|-------------------|---------|
| **C6.0 — reality map** | Accepted | [c6_0_option_surface_reality_map.md](../tmp/c6_0_option_surface_reality_map.md) | Mapped producer, cache, consumers, and tests; separated schema validity from assembly-readiness; identified risks (path defaults, overwrite risk, chain-scanned weekly expiry vs calendar target, duplicate grain ambiguity, stale lineage) without overstating defects later disproven by C6.1B/C6.4 |
| **C6.1 — design** | Accepted | [c6_1_option_surface_design_memo.md](../tmp/c6_1_option_surface_design_memo.md) | Locked three-layer design, PASS/WARN/FAIL policy, subtask sequence C6.1A→C6.6, and pytest vs audit-gate responsibilities |
| **C6.1A — producer safety** | PASS | `9017a6b` · [c6_1a_producer_safety_report.md](../tmp/c6_1a_producer_safety_report.md) | Bounded ticker/date flags, `--dry-run`, `--output-root` isolation, overwrite protection, central path defaults, shared weekly entry-date schedule (`weekly_trade_dates_in_range`), Jan 1–Dec 31 year bounds. **C6.1A changed weekly entry scheduling and producer safety, but did not change expiry selection.** |
| **C6.1B — weekly-expiry diagnostic** | PASS (policy) | `7049037` · [c6_1b_weekly_expiry_diagnostic.md](../tmp/c6_1b_weekly_expiry_diagnostic.md) | See policy conclusion below |
| **C6.1C — strict weekly expiry** | PASS | `af9d9a0` · [c6_1c_calendar_paired_weekly_expiry_report.md](../tmp/c6_1c_calendar_paired_weekly_expiry_report.md) | Calendar-paired weekly expiry; monthly path unchanged |
| **C6.2 — A1/A2 contract** | PASS | `8d776e6` · [c6_2_surface_artifact_contract_report.md](../tmp/c6_2_surface_artifact_contract_report.md) | Contract + read-only audit foundation |
| **C6.3 — assembly-readiness** | PASS | `e2ffc2b` · [c6_3_surface_assembly_readiness_report.md](../tmp/c6_3_surface_assembly_readiness_report.md) | Straddle / wing / iron-fly / iron-condor candidate metrics |
| **C6.4 — real-artifact evidence** | Pass 1 WARN · Pass 2 PASS | `c75417d` (audit) · `70527ca` (evidence) · [c6_4_surface_audit_summary.md](../tmp/c6_4_surface_audit_summary.md) | Bounded 5×13 real-cache and fresh-smoke audits |

### C6.0 — reality map (risks)

Initial reconnaissance found: unscoped precompute could overwrite canonical cache; weekly **entry** schedule was script-local; weekly **expiry** used permissive chain scan (`_find_best_expiry`) rather than calendar-paired target; `surface_valid` vs S3 wing requirements were conflated in some docs; duplicate quote grain needed triage before FAIL; historical cache lineage was undocumented. C6.0 did **not** claim the historical cache was corrupt — it defined what evidence was required to accept it.

### C6.1 — design

Accepted three-layer gate, PASS/WARN/FAIL table, and the rule that pytest (Layer 1 code) plus read-only audit CLI (Layers 2–3 on disk) constitute closeout evidence — not schema tests alone.

### C6.1A — producer safety and weekly entry schedule

Producer safety: `--dry-run`, `--output-root`, `--overwrite` guard, `--tickers` / date window scoping, path defaults in `src/data/paths.py`, shared `weekly_trade_dates_in_range` in `src/data/trading_day.py`. Expiry selection **unchanged** in C6.1A.

### C6.1B — weekly-expiry diagnostic

**Known-weekly sample (Sample A — AAPL, MSFT, NVDA, SPY, QQQ):**

| Metric | Rate |
|--------|------|
| Observations diagnosed | 60/60 weekly-tradable |
| Exact target listed | 100% |
| Body-pair quotable | 100% |

**Broader C4 liquid-universe sample (Sample B):**

| Metric | Rate |
|--------|------|
| Weekly-tradable among diagnosed observations | **57.2%** |

**Interpretation:** The broad rate measures **opportunity coverage/capacity**, not correctness. Missing exact target weekly expiry is expected no-trade behavior for non-weekly-option names.

**Policy decision (HD accepted):**

- Use the **exact next schedule week** as expiry.
- **Do not** fall back to a nearby or nearest-DTE expiry.
- Missing exact expiry means **no weekly trade**.

### C6.1C — strict weekly expiry

```text
entry_date = resolved trading date for week i
expiry_date = resolved trading date for week i+1
```

**Failure outcomes:**

| Tag | Meaning |
|-----|---------|
| `no_target_weekly_expiry` | No successor week in schedule |
| `no_expiries_on_entry_chain` | No listed expiries on entry date |
| `target_weekly_expiry_not_listed` | Exact target not on chain |
| `target_weekly_body_not_quotable` | Target listed but body legs not quotable |

Monthly expiry behavior **remained unchanged**.

### C6.1D status

No standalone C6.1D phase was completed for closeout.

- A limited weekly soft-failure improvement (`target_weekly_body_not_quotable`) landed in **C6.1C**.
- Broader soft-failure cleanup was **not required** for C6 closeout.
- Producer deduplication was **not implemented** because C6.4 found **no duplicates** in the bounded real-cache or fresh-smoke samples.

### C6.2 — A1/A2 contract

Checks enforced: required schema · `surface_valid` invariant · failure vocabulary · settlement completeness · A1 metadata grain `(ticker, entry_date)` · A1/A2 join integrity · A2 quote grain · weekly date alignment.

Module: `src/features/option_surface_contract.py` · CLI: `scripts/audit_option_surface_artifacts.py`.

### C6.3 — assembly-readiness

Metrics: `body_pair_ready` · `straddle_ready` · `otm_call_wing_available` · `otm_put_wing_available` · `ironfly_candidate_ready` · `ironcondor_candidate_ready`.

**Iron-fly readiness** requires body-pair readiness plus **independently** available OTM call and OTM put wings. Symmetric wing-pair availability (`otm_wing_pair_available`) is **informational only**.

Module: `src/features/option_surface_readiness.py`.

### C6.4 — real-artifact evidence

| Pass | Scope | Verdict |
|------|-------|---------|
| **Pass 1 — existing historical cache** | Read-only audit of `option_surface_meta_weekly_2018_2026.parquet` + quotes pair; bounded filter 5 tickers × Q1 2024 | **WARN** — bounded integrity checks pass; historical producer/upstream lineage unknown |
| **Pass 2 — fresh bounded C6 smoke** | Producer run to `c6_4_surface_smoke/` then audit with known lineage | **PASS** |

Reports: [c6_4_real_cache_surface_audit.md](../tmp/c6_4_real_cache_surface_audit.md) · [c6_4_smoke_surface_audit.md](../tmp/c6_4_smoke_surface_audit.md) · [c6_4_surface_audit_summary.md](../tmp/c6_4_surface_audit_summary.md).

C6.4 helpers: `src/features/option_surface_c64_audit.py` (duplicate triage, weekly-expiry classification, legacy-lineage WARN policy).

---

## 3. Verification summary

| Phase | Result | Commit / evidence | Key conclusion |
|-------|--------|-------------------|----------------|
| C6.1A | PASS | `9017a6b` · [c6_1a](../tmp/c6_1a_producer_safety_report.md) | Safe bounded producer; shared entry schedule; expiry unchanged |
| C6.1B | PASS (policy) | `7049037` · [c6_1b](../tmp/c6_1b_weekly_expiry_diagnostic.md) | Known-weekly 60/60; strict exact-next-week policy approved |
| C6.1C | PASS | `af9d9a0` · [c6_1c](../tmp/c6_1c_calendar_paired_weekly_expiry_report.md) | Strict calendar-paired weekly expiry in producer |
| C6.2 | PASS | `8d776e6` · [c6_2](../tmp/c6_2_surface_artifact_contract_report.md) | A1/A2 contract checks + audit CLI foundation |
| C6.3 | PASS | `e2ffc2b` · [c6_3](../tmp/c6_3_surface_assembly_readiness_report.md) | Assembly-readiness metrics aligned with S3 builders |
| C6.4 Pass 1 | **WARN** | `70527ca` · [real cache](../tmp/c6_4_real_cache_surface_audit.md) | Bounded sample integrity OK; legacy lineage unknown |
| C6.4 Pass 2 | **PASS** | `70527ca` · [smoke](../tmp/c6_4_smoke_surface_audit.md) | Fresh smoke with known producer lineage passes all blocking checks |

---

## 4. Test evidence

Accepted C6 test evidence (recorded at evidence head `70527ca`; no code or tests changed after these runs):

| Suite | Result |
|-------|--------|
| C6.1 weekly expiry / CLI / diagnostic | **41 passed** in 0.41s |
| C6.2 contract | **25 passed** in 0.15s |
| C6.3 readiness | **39 passed** in 0.08s |
| C6.4 helpers + audit CLI integration | **33 passed** in 0.86s |

**Commands:**

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_weekly_expiry.py tests/unit/test_precompute_option_surface_cli.py tests/unit/test_diagnose_weekly_expiry_policy.py -q
```

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_contract.py -q
```

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_readiness.py -q
```

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest tests/unit/test_option_surface_c64_audit.py tests/unit/test_audit_option_surface_artifacts.py -q
```

No GitHub CI evidence is claimed for C6. The full repository pytest suite was **not** run as part of C6.6; Sprint 004’s eventual full-suite gate (blocker #11) remains separate.

---

## 5. Real-artifact evidence

**Bounded scope (C6.4 Pass 1 and Pass 2):**

| Field | Value |
|-------|-------|
| Tickers | AAPL, MSFT, NVDA, SPY, QQQ |
| Requested window | 2024-01-01 through 2024-03-31 |
| Resolved entry dates | 13 |
| Resolved entry range | 2024-01-05 through 2024-03-28 |
| A1 metadata rows | 65 |
| A2 quote rows | 2,114 |

**Key results (both passes, normalized metrics view):**

| Check | Result |
|-------|--------|
| `surface_valid` | 65/65 |
| `straddle_ready` | 65/65 |
| `ironfly_candidate_ready` | 65/65 |
| `ironcondor_candidate_ready` | 65/65 |
| Strict weekly-expiry exact matches | 65/65 |
| Silent expiry mismatches | 0 |
| A1 duplicate keys | 0 |
| A2 duplicate keys | 0 |
| Orphan quote rows | 0 |
| Valid metadata without quotes | 0 |
| Settlement-field failures | 0 |

These results apply to the **bounded evidence window** only — not the entire ticker universe or full historical cache.

---

## 6. Duplicate-triage conclusion

No A1 or A2 duplicate keys were found in either bounded C6.4 pass. **No producer deduplication or artifact repair was required.**

**Accepted future audit policy** (audit layer only; does not mutate artifacts):

| Case | Verdict |
|------|---------|
| Legacy identical duplicates | **WARN** — raw evidence retained |
| Legacy conflicting duplicates | **FAIL** |
| Fresh-output duplicates | **FAIL** |

---

## 7. Accepted warnings and limitations

| Item | Status | Notes |
|------|--------|-------|
| **Unknown legacy lineage** | Accepted **WARN** | Existing historical cache only; Pass 1 bounded checks pass |
| **Bounded sample** | Accepted | C6.4 evidence is 5 tickers × 13 weeks — substantive but not full-universe or full-history certification |
| **Broad weekly opportunity coverage** | Accepted | C6.1B broad sample: 57.2% weekly-tradable among successfully diagnosed observations — affects strategy opportunity count/capacity, not correctness |
| **No full historical rebuild** | Accepted | Not required for Sprint 004 C6 closeout |
| **No profitability or fill evidence** | Deferred | Sprint 006+ |
| **No surface-audit orchestration wiring** | Deferred to **C8** | Standalone audit accepted for C6; wiring into `refresh_weekly_inputs surface-audit` is the C8 owner (historically numbered C6.5) |
| **No incremental append/watermark hardening** | Deferred | Sprint 005 |
| **No PIT universe validation** | Deferred | C7 |

### On the historical "C6.5" numbering

There is **no missing C6.5 closeout requirement**. The work originally numbered C6.5 is wiring the accepted standalone surface audit into:

```text
refresh_weekly_inputs surface-audit
```

That orchestration work is intentionally deferred to **C8** and does **not** block C6 closeout. C6 delivered and accepted the standalone audit (`scripts/audit_option_surface_artifacts.py`); orchestration is an integration concern owned by C8. No required C6 subtask remains unfinished.

### Why the Pass 1 lineage WARN is acceptable

1. The historical bounded sample passed schema, grain, join, settlement, date-alignment, expiry, and assembly-readiness checks.
2. The fresh C6 smoke has known current-producer lineage and passed all blocking checks.
3. The warning limits confidence in the **provenance** of the old historical cache; it does not indicate a detected integrity failure in the bounded sample.
4. The warning is **not** treated as certification of the complete historical cache or full universe.

---

## 8. C6 acceptance boundary

HD accepts the remaining **unknown-historical-lineage WARN** for Sprint 004 C6 closeout.

All blocking C6 **FAIL** conditions are resolved in the accepted bounded evidence.

At the C6 gate, the current strict weekly A1/A2 producer and audit contract are **accepted for downstream integration**.

This is **not** blanket certification of the complete legacy historical cache, the full ticker universe, or strategy performance.

**Sprint 006 real-data backtesting remains gated by:**

1. Completion of the remaining Sprint 004 work (C7–C9, C3 umbrella);
2. Sprint 005 A4/feature-layer trust;
3. Use of artifacts with **explicit current lineage** (prefer fresh or documented producer runs);
4. The Sprint 006 evaluation protocol.

C6 alone does **not** make the strategy ready for Sprint 006 execution.

---

## 9. Closeout checklist mapping

| Sprint 004 blocker #9 requirement | C6 evidence |
|-----------------------------------|-------------|
| T1–T6 pytest | C6.1C / C6.2 test evidence (41 + 25 tests) |
| T7 real-cache sample | C6.4 Pass 1 and Pass 2 |
| Surface audit report | C6.2 / C6.3 audit CLI + C6.4 reports |
| Substantive documented run | 5 tickers × 13 weeks |
| No strategy / S5 / S8 / ORCH changes | Confirmed across C6 |

**Blocker #9:** **CLOSED** (2026-07-11)

Sprint 004 itself remains **active** (C7–C9, C3, blockers #1, #4, #6, #10, #11, #13, etc.).

---

## 10. Operator commands

### Safe bounded producer dry-run

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/precompute_option_surface.py `
  --frequency weekly `
  --start-year 2024 --end-year 2024 `
  --start-date 2024-01-01 --end-date 2024-03-31 `
  --tickers AAPL MSFT NVDA SPY QQQ `
  --data-root C:/MomentumCVG_env/input/adjusted_liquid `
  --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet `
  --output-root C:/MomentumCVG_env/cache/c6_smoke_dryrun `
  --dry-run
```

### Isolated bounded producer run

Use a **noncanonical output root**. Do **not** overwrite canonical historical artifacts casually.

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/precompute_option_surface.py `
  --data-root C:/MomentumCVG_env/input/adjusted_liquid `
  --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet `
  --output-root C:/MomentumCVG_env/cache/c6_4_surface_smoke `
  --frequency weekly `
  --start-year 2024 --end-year 2024 `
  --start-date 2024-01-01 --end-date 2024-03-31 `
  --tickers AAPL MSFT NVDA SPY QQQ `
  --overwrite `
  --workers 8
```

(Accepted C6.4 Pass 2 smoke command; artifacts already exist at that path.)

### Standalone surface audit

Until C8 wiring lands, use `scripts/audit_option_surface_artifacts.py` directly.

```powershell
C:/MomentumCVG_env/venv/Scripts/python.exe scripts/audit_option_surface_artifacts.py `
  --meta-path C:/MomentumCVG_env/cache/option_surface_meta_weekly_2018_2026.parquet `
  --quotes-path C:/MomentumCVG_env/cache/option_surface_quotes_weekly_2018_2026.parquet `
  --frequency weekly `
  --data-root C:/MomentumCVG_env/input/adjusted_liquid `
  --spot-db-path C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet `
  --start-date 2024-01-01 --end-date 2024-03-31 `
  --sample-tickers AAPL MSFT NVDA SPY QQQ `
  --report-format c6.4 `
  --legacy-cache `
  --include-assembly-readiness `
  --output-report docs/tmp/surface_audit_pass1.md
```

Full Pass 1 / Pass 2 command lines: [c6_4_real_cache_surface_audit.md](../tmp/c6_4_real_cache_surface_audit.md) · [c6_4_smoke_surface_audit.md](../tmp/c6_4_smoke_surface_audit.md).

---

## 11. Deferred work

| Item | Sprint / task |
|------|----------------|
| `refresh_weekly_inputs surface-audit` wiring | **C8** (historically numbered C6.5; not a C6 closeout blocker) |
| PIT universe validation | C7 |
| Weekly refresh orchestration | C8 |
| Runbook and CLI cleanup | C9 |
| A4 features, straddle history, incremental append/watermarks | Sprint 005 |
| SurfaceRunner real-data evaluation and trade logs | Sprint 006 |
| Tier B / Sharpe go-no-go | Sprint 007 |

---

## 12. Remaining Sprint 004 work

**C6 is closed.**

Sprint 004 remains **active** for **C7–C9** and the later **C3** validation umbrella.

---

## 13. References and active documentation map

| Report / doc | Topic |
|--------------|-------|
| [c6_0_option_surface_reality_map.md](../tmp/c6_0_option_surface_reality_map.md) | C6.0 reconnaissance |
| [c6_1_option_surface_design_memo.md](../tmp/c6_1_option_surface_design_memo.md) | C6.1 design (canonical) |
| [c6_1a_producer_safety_report.md](../tmp/c6_1a_producer_safety_report.md) | C6.1A producer safety |
| [c6_1b_weekly_expiry_diagnostic.md](../tmp/c6_1b_weekly_expiry_diagnostic.md) | C6.1B expiry policy diagnostic |
| [c6_1c_calendar_paired_weekly_expiry_report.md](../tmp/c6_1c_calendar_paired_weekly_expiry_report.md) | C6.1C strict weekly expiry |
| [c6_2_surface_artifact_contract_report.md](../tmp/c6_2_surface_artifact_contract_report.md) | C6.2 contract |
| [c6_3_surface_assembly_readiness_report.md](../tmp/c6_3_surface_assembly_readiness_report.md) | C6.3 readiness |
| [c6_4_surface_audit_summary.md](../tmp/c6_4_surface_audit_summary.md) | C6.4 summary |
| [surface_engine_data_contract.md](../surface_engine_data_contract.md) | A1/A2 data contract |
| [current_sprint.md](../agenda/current_sprint.md) | Sprint 004 active agenda |
| [004_c5_adjusted_liquid.md](004_c5_adjusted_liquid.md) | Upstream C5 adjusted-liquid closeout |

Historical `docs/tmp/c6_*` evidence files are **retained evidence** — not rewritten for closeout consistency.

---

## Active documentation map (post-C6)

| Doc | C6-relevant content |
|-----|---------------------|
| [AGENTS.md](../../AGENTS.md) | Canonical backtest path; cache paths |
| [repo_map.md](../repo_map.md) | Stage A data flow |
| [current_sprint.md](../agenda/current_sprint.md) | C6 ✓; C7–C9 remaining |
| [known_bugs.md](../known_bugs.md) | KB-001 not in C6 scope |
