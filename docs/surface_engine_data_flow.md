# Surface engine — data flow and component diagram

**Status:** Draft — Sprint 002 (canonical after sign-off)  
**Last updated:** 2026-05-28  
**Companion:** [surface_engine_data_contract.md](surface_engine_data_contract.md), [surface_engine_evaluation_plan.md](surface_engine_evaluation_plan.md)

---

## Purpose

Step-by-step **source of truth** for how data moves through the surface backtest. Each box is a component you can evaluate independently: what is built, what criteria it must meet, and which paths (success / skip / fail) it supports.

**Legend — implementation status**

| Tag | Meaning |
|-----|---------|
| 🟢 `built` | Implemented and covered by behavior or contract tests |
| 🟡 `partial` | Runs but missing contract fields or invariants |
| 📋 `spec-only` | Contract + test defined; code not implemented |
| 🔴 `drift` | Code path differs from target architecture |
| 📦 `given` | Stage A input; assumed this sprint |

---

## End-to-end diagram

```mermaid
flowchart TB
  subgraph stageA ["Stage A — Given inputs 📦"]
    META["A1 Surface meta<br/>ticker × entry_date"]
    QUOTES["A2 Surface quotes<br/>ticker × entry_date × strike × side"]
    LIQ["A3 Liquidity panel<br/>ticker × month_date"]
    FEAT["A4 Features<br/>ticker × date"]
  end

  subgraph envelope ["R0 Run envelope 🟡"]
    CFG["BacktestRunConfig"]
    PATHS["SurfaceDataPaths"]
    DATES["Trade date list"]
  end

  subgraph pipeline ["Stage B — Pipeline steps"]
    S1["S1 Universe<br/>step1_get_universe 🟢"]
    S2["S2 Signals<br/>step2_score_signals 🟢"]
    S3["S3 Structures<br/>step3_get_eligible_structures 🟢/🔴"]
    S4["S4 Exclusions<br/>step4_apply_exclusions 📋"]
    S5["S5 Select + size<br/>step5_select_and_size 📋"]
    S6["S6 Cost + return<br/>step6_apply_cost 📋"]
    S7["S7 Settle<br/>assembly.settle 🟢"]
  end

  subgraph outputs ["Run outputs 🟡"]
    LOG["Trade log"]
    DSUM["Date summary"]
    RSUM["Run summary"]
    MAN["Run manifest 📋"]
  end

  subgraph metrics ["S8 Metrics 🟡"]
    MET["surface_metrics"]
  end

  META --> S3
  QUOTES --> S3
  LIQ --> S1
  FEAT --> S2
  CFG --> envelope
  PATHS --> envelope
  envelope --> DATES
  DATES --> S1
  S1 --> S2
  S2 --> S3
  S3 --> S4
  S4 --> S5
  S5 --> S7
  S7 --> S6
  S6 --> LOG
  LOG --> DSUM
  LOG --> RSUM
  LOG --> MET
  MET --> RSUM
  CFG --> MAN
```

---

## Box reference (fill criteria during Sprint 002)

Use this table with the diagram. Link each row to contract § and evaluation plan §.

| Box | Target owner | Status | Criteria to pass (structural) | Paths out |
|-----|--------------|--------|------------------------------|-----------|
| A1 Meta | precompute | 📦 given | Valid row for traded `(ticker, entry_date)`; `exit_spot`, `body_strike`, `surface_valid` | valid → S3; invalid → skip reason |
| A2 Quotes | precompute | 📦 given | Enough legs to build configured structure | ok → S3; missing → `no_tradeable_structure` |
| A3 Liquidity | build_liquidity_panel | 📦 given | PIT snapshot exists for `trade_date` | ok → S1; missing → empty universe |
| A4 Features | build_features | 📦 given | Row for `trade_date` per ticker | ok → S2; missing → drop |
| R0 Envelope | config + manifest | 🟡 | `start_date < end_date`; paths resolve; dates explicit | → S1 loop |
| S1 Universe | `step1` | 🟢 | PIT; AND filters documented | tickers → S2 |
| S2 Signals | `step2` | 🟢 | Disjoint long/short pools; CVG filter | candidates → S3 |
| S3 Structures | `step3` | 🟢/🔴 | One row per candidate; `build_ok` | built → S4; fail → exclusion |
| S4 Exclusions | `step4` | 📋 | `had_earnings_nearby` flag | → S5 |
| S5 Select+size | `step5` | 📋 | 50 total cap; budget units; `contracts` | included → S7; excluded → diagnostics |
| S7 Settle | `settle` | 🟢 | `pnl` consistent with `exit_spot` | → S6 |
| S6 Cost+return | `step6` | 📋 | `return_on_max_loss` in budget units | → LOG |
| S8 Metrics | `surface_metrics` | 🟡 | Sharpe on agreed return series | → RSUM |
| ORCH | `SurfaceRunner` | 🔴 | Calls S1–S8 only; no duplicate logic | full run |

---

## Decision paths (which backtest questions need which boxes)

| Decision you want to make | Minimum boxes that must be 🟢 + contract-clean |
|---------------------------|-----------------------------------------------|
| Is universe PIT-correct? | A3, S1 |
| Is signal ranking correct? | A4, S1, S2 |
| Is iron fly math correct? | A1, A2, S3, S7 |
| Is portfolio cap correct? | S2, S3, S5 |
| Is go/no-go Sharpe meaningful? | All S1–S8 + manifest |

---

## Architecture note

**Target:** `SurfaceRunner` orchestrates; **all** behavior in `pipeline.py` steps (decoupled). Current 🔴 drift: runner inlines S3–S5. Sprint 002 documents target; refactor sprint TBD after contract sign-off.

---

## Relation to Sprint 001 doc

[surface_runner_data_flow.md](surface_runner_data_flow.md) captured the Sprint 001 audit map. This document becomes the **canonical** flow after HD review.

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-28 | Sprint 002 scaffold + mermaid diagram |
