# Surface engine — portfolio and metrics design (S5 / S8)

**Status:** Draft — Sprint 002 Session C  
**Last updated:** 2026-06-07 (cap: decision 003)  
**Audience:** HD + agents  
**Companion:** [surface_engine_data_contract.md](surface_engine_data_contract.md), [surface_engine_data_flow.md](surface_engine_data_flow.md), [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md)

---

## Purpose

Sprint 002 Sessions A–B pinned contracts through **structure build, exclusions, and per-share settlement** (S1–S4, S7). The remaining pipeline tail — **S5, S8, and full orchestration** — is not ready for contract tests because:

1. **S5 today is not a portfolio layer** — it is selection + settle wiring in `SurfaceRunner._select_size_and_settle`.
2. **Return denominator** is not fully wired (body credit vs max-loss budget on trade log).
3. **Sizing is a constraint-satisfaction decision**, not an optimizer — config supplies limits (capital, max loss per name, cap count); S5 applies rules to preserve signal alpha at the chosen fidelity tier.

**S6 collapsed into S5 (HD decision):** Entry fill/cost is fixed at structure assembly (S3 + `FillAssumption`). A separate post-trade cost step is redundant for v1; `return_on_allocated_budget` and `fill_label` are computed in S5 Phase 3.

This document defines **what each step must achieve** before Sprint 003 writes contracts and implementation together.

**Out of scope here:** broker execution, live early exit, sector caps, deployable $ pinning (deferred per Sprint 002 HD decision).

---

## Design stance (HD — Sprint 002 Session C)

| Topic | Stance |
|-------|--------|
| Sizing | **Constraint-driven decision** under config limits — not an optimizer; **both** Tier A (conceptual) and Tier B (integer lots + capital) in Sprint 003, selected via `sizing_mode` |
| S5 | **Select, size, and simulate trades** — turn S4 candidates into sized trade log rows (config dials, not an optimizer) |
| S6 | **Collapsed into S5** — fill at S3; no separate `step6_apply_cost` for v1 |
| S8 | Metrics must support **decision-quality** per [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md) (Sharpe on agreed return series) |
| Contracts | **Deferred** for S5/S8/ORCH until this doc is reviewed and Sprint 003 build starts |

---

## Current vs target (honest snapshot)

| Step | Code today | Gap |
|------|------------|-----|
| **S5** | Runner: eligibility filter + per-**side** cap → settle → `pnl_per_share` only | **No sizing policy** (`quantity` / tier weights missing); no `return_on_allocated_budget`; settle bundled here; cap semantics **aligned** with [decision 003](decisions/003_position_cap_per_side.md) |
| **S5 returns** | Fill at S3; no separate cost step | No `return_on_allocated_budget` on trade log yet |
| **S8** | `surface_metrics` on **body credit** (`pnl / body_credit_per_share`) | Evaluation protocol primary metric is **return on max-loss budget** |
| **ORCH** | S1–S4 via `pipeline.py`; S5 inline in runner | Not a thin S1–S8 loop |

---

## What a “quality backtest” must support (v1)

Before real-money shadow/paper, a run must let you answer:

1. **Selection** — Who was a candidate, who traded, who excluded, and **why** (traceable S1→S5).
2. **Sizing** — Given constraints (total capital, max loss per position, position count, lot size), how large each simulated trade is — so returns still reflect **signal alpha**, not artifact of bad sizing.
3. **Simulated trade** — Each included row is a complete trade record (structure + size + settled PnL), not just a flagged candidate.
4. **Book composition** — How many positions per rebalance; respects **`max_names_per_side` per direction** (e.g. 25 long + 25 short ≈ 50-book).
5. **Economics** — Per-trade PnL comparable across structures via `return_on_allocated_budget` (see § Return normalization).
6. **Conservative fills** — Go/no-go uses harsh fill ([backtest_evaluation_protocol.md](backtest_evaluation_protocol.md)); mid is diagnostic only.
7. **Run summary** — Sharpe, drawdown, availability on the **agreed return series** — not body-credit proxy alone.

Structural success checklist in [surface_engine_data_contract.md](surface_engine_data_contract.md) § Run-level success remains the north star; items 4–6 of that checklist depend on this design.

---

## S5 — Select, size, and simulate trades (not “portfolio layer” v1)

### What S5 is for

S4 leaves you with **candidates** — structure rows that passed build and earnings flags. S5 is where those candidates become **simulated trades** for the backtest.

S5 has three jobs on each rebalance date:

```text
S4 candidates (structure_ok; earnings flag already set)
  → (1) SELECT   — honor upstream eligibility; rank **per side**; apply max_names_per_side;
                   mark included_in_portfolio + exclusion_reason on every row
  → (2) SIZE     — apply sizing policy under config constraints (not config quantities):
                   Tier A: equal risk fraction / conceptual weight per included name
                   Tier B: integer lots from max-loss budget + capital binding
  → (3) SIMULATE — S7 settle at chosen size; per-share + total PnL + return_on_allocated_budget
                   → trade log row (fill_label from S3)
```

**Selection** answers: *among eligible long (or short) candidates, who gets a slot on that side?* Long and short pools are capped **independently** — no mixed ranking. (S4 already applied earnings; S5 does not re-run exclusions.)  
**Sizing** answers: *under capital, count, and max-loss limits, how large is each included trade?*  
**Simulate** answers: *what was the hold-to-expiry outcome at that size, in the agreed return units?*

Upstream steps (S1–S4) determine **who is eligible**. S5 determines **who is traded, how they are sized under constraints, and what simulated result follows**. That is the minimum trade-construction surface for v1 — without multi-date book state, optimizers, or sector logic.

**Important:** Config does not “assign quantity” directly. Config defines **constraints**; S5 applies a **sizing policy** that respects them. The policy is not a portfolio optimizer — it is a small set of rules to avoid sizing artifacts that would drown out signal alpha.

### What S5 is not (v1)

- Multi-date portfolio / margin / replacement policy across rebalance dates
- Signal-weighted or risk-parity **optimizer**
- Sector / cluster caps

Those are a **portfolio layer** in a stronger sense; defer until v1 backtest is trustworthy.

### Inputs (target)

- `structures` after S3+S4 (signal columns already on row; `structure_ok`, `had_earnings_nearby`, `_assembly`, `max_loss_per_share`, …)
- `config` (`BacktestRunConfig`)

No separate `signals` join if S3 preserves signal columns (current design).

### Phase 1 — Select (cap + rank; not a second exclusion pass)

S4 has already set `had_earnings_nearby` and S3 has set `structure_ok`. S5 **reads** those flags — it does not re-apply earnings or structure filters.

| Step | Rule |
|------|------|
| Eligibility | Drop or mark excluded rows where `structure_ok == False` or `had_earnings_nearby == True`; set `exclusion_reason` from upstream (`no_tradeable_structure`, `earnings_exclusion`, …) |
| Rank | Split by `direction`; within each side rank by `signal_rank_pct` (long: descending; short: ascending) |
| Cap | [Decision 003](decisions/003_position_cap_per_side.md): top **`max_names_per_side`** per direction (independent pools) |
| Include | Top `max_names_per_side` per side get `included_in_portfolio == True`; overflow gets `exclusion_reason = cap_exceeded` (name TBD) |
| Output flags | `included_in_portfolio`, `exclusion_reason` on **every** candidate row (included and excluded) |

**Runner today:** per-side cap via `max_names_per_side` matches v1 target ([decision 003](decisions/003_position_cap_per_side.md)); still partial — earnings/eligibility inline rather than clean S4→S5 handoff; no sizing/returns.

### Phase 2 — Size (constraint-driven; required for simulated trade)

Sizing is where candidates become **trades with economic weight**. The goal is to measure **signal alpha** without letting sizing mechanics dominate the backtest.

#### Constraints (config supplies limits, not quantities)

| Constraint | Typical config / source | Role |
|------------|-------------------------|------|
| Position count | `max_names_per_side` (per direction; e.g. 25 → up to 50 total book) | Diversification per side |
| Per-position max loss | `max_loss_budget_per_trade` (or per-name cap) | Risk per signal |
| Total deployable capital | Global budget (abstract units until $ pinned) | Book-level binding constraint |
| Lot granularity | 1 contract = 100 shares (equity options) | Realistic trade simulation |
| Structure geometry | `max_loss_per_share`, premium (credit/debit) from S3 | Converts budget → contracts |

When constraints conflict (e.g. 50 names × budget exceeds total capital), S5 must apply a **documented rule** — scale down, drop lowest rank, or fail loud — not silently mis-size.

#### Two fidelity tiers (HD — not an optimizer)

**Tier A — Conceptual / signal-preserving base case**

Purpose: assess whether the **signal** has alpha before lot rounding and capital binding distort results.

| Assumption | Intent |
|------------|--------|
| Equal economic weight per included position | Each trade uses the same **fraction** of the book (e.g. `1 / n_included`) |
| Equal premium / max-loss **notional** across names (conceptual) | Short and long structures compared on a **normalized** basis — e.g. return on equal allocated risk or equal premium slot |
| Fractional size allowed | `weight` or `risk_fraction` per row; no integer contract requirement |
| Return metric | Per-trade return on allocated risk (or on conceptual premium slot); aggregate without dollar PnL |

This tier answers: *“If I spread capital evenly across selected signals, does the signal side still win?”* — alpha at a **conceptual** level.

**Tier B — Deployable simulation (next level)**

Purpose: approximate what is **placeable** under real constraints while still not doing portfolio optimization.

| Rule | Intent |
|------|--------|
| Integer contracts | `quantity = floor(budget_for_name / max_loss_per_share)`; reject or flag if `< 1` |
| Contract multiplier | `pnl_dollars = quantity × pnl_per_share × 100` |
| Per-name budget | Start from equal `max_loss_budget_per_trade`; adjust if total capital binds |
| Capital check | `sum(quantity × max_loss_per_share × 100) ≤ deployable_capital` (or abstract equivalent) |
| Residual policy | If over budget: scale all quantities proportionally, or trim by rank — **pin in Q8b** |

This tier answers: *“After lots and capital bind, is alpha still there?”* — closer to shadow/paper, still not full optimization.

Both tiers live in **S5** (same select phase; sizing policy branch). S8 aggregates S5 return fields.

#### What this is not

- End-to-end portfolio optimization (weights from signal strength, correlation, sector)
- Dynamic rebalancing of size across dates from prior PnL
- Margin model / broker-specific rounding

Those can come later if Tier B backtests justify them.

**Runner today:** **no sizing phase** — `quantity` absent; per-share settle only; no tier A weights or tier B capital check.

**Sprint 003 (HD — 2026-06-07):** implement **both** tiers in S5. Config `sizing_mode` switches between Tier A (`conceptual`) and Tier B (`integer_lots`). Same select + simulate path; sizing policy branch only. Default mode at run time TBD (Q8b); both code paths required for Sprint 003 done.

### Phase 3 — Simulate trade (settle + returns)

Entry economics (mid vs cross) are fixed in **S3** via `config.fill` — no separate cost pass.

For each sized, included row:

| Step | Rule |
|------|------|
| Settle | `position = assembly.settle(exit_spot)` (S7 math) |
| Per share | `pnl_per_share = position.pnl` (already reflects fill chosen at assembly) |
| Total | `pnl_total = quantity × pnl_per_share × contract_multiplier` (Tier B; multiplier typically 100) |
| Return | `return_on_allocated_budget = pnl_total / max_loss_budget_per_trade` (+ structure-native diagnostics — see § Return normalization) |
| Audit | `fill_label` = `config.fill.label` on row / run summary |
| Trade log | Row is a **simulated trade**: identity + structure + size + PnL + return |

**Former S6 scope** (`step6_apply_cost`, `cost_model` spread penalty) is **not** a v1 pipeline step — would double-count if cross fill is already used. `cost_model` in `BacktestRunConfig` is legacy; use `fill` only.

### Outputs (target trade log row)

**All candidates (when `include_diagnostics=True`)**

| Field | Purpose |
|-------|---------|
| `included_in_portfolio` | Selected for simulated trade |
| `exclusion_reason` | Set when not traded |
| `trade_date`, `ticker`, `direction` | Identity |

**Included rows only — simulated trade**

| Field | Purpose |
|-------|---------|
| `sizing_mode` | Echo config: `conceptual` (Tier A) or `integer_lots` (Tier B) |
| `risk_fraction` | Tier A: equal weight per included name (e.g. `1 / n_included`) |
| `quantity` | Tier B: sized integer contracts; NaN or 0 in Tier A |
| `max_loss_budget_per_trade` | Config echo (Tier B); conceptual budget slot in Tier A |
| `max_loss_per_share` | From structure |
| `pnl_per_share` | S7 settle |
| `pnl_total` (or `pnl_dollars`) | Tier B: `quantity × pnl_per_share × multiplier`; Tier A: optional or derived from fraction |
| `return_on_allocated_budget` | Primary per-trade metric for S8 (cross-book; see § Return normalization) |
| `return_on_risk_geometry` / `return_on_premium` / `return_on_credit` | Structure-native diagnostics (optional columns) |
| `fill_label` | Echo `config.fill` (reproducibility) |

### Settle boundary (S5 vs S7)

- **S7** — pure payoff math at `exit_spot` (per share).
- **S5** — owns **when** settle runs (included rows only) and **combines** settle result with **quantity** to form the simulated trade.

### Config levers (constraints + policy switch, not quantity assignment)

| Config field | Role |
|--------------|------|
| `sizing_mode` (name TBD) | `conceptual` (Tier A) vs `integer_lots` (Tier B) |
| `max_names_per_side` | Per-direction position cap (long and short ranked separately) |
| `max_loss_budget_per_trade` | Per-name max-loss budget (Tier B starting point) |
| `deployable_capital` (abstract; optional v1) | Total book constraint for Tier B |
| `include_diagnostics` | Keep excluded candidates in log |
| `earnings_exclusion_days` | Already applied in S4 |
| Short straddle proxy | Max-loss proxy for unbounded short straddle (Q5) |
| Capital overrun policy | Scale vs trim when Tier B exceeds budget (Q8c) |

---

## Return normalization (S5 Phase 3 + S8)

Mixed books — e.g. **short iron fly + long straddle** (v1 pin) — do not share one geometric “max loss” definition. Normalization must separate **sizing at risk**, **primary cross-book return** (for Sharpe / go/no-go), and **structure-native diagnostics**.

### Two questions (do not conflate)

| Question | Needs | Answered by |
|----------|-------|-------------|
| **How large is each trade?** | `at_risk_per_share` per instrument (true or proxy) | S5 Phase 2 sizing |
| **How do we score the trade?** | Return denominator for aggregation | S5 Phase 3 + S8 |

Premium normalization (`pnl / premium paid or received`) measures performance vs the **premium slot**. Max-loss geometry measures performance vs **defined downside**. Allocated-budget return measures performance vs the **dollar slot you assigned** that name — the right primary metric for a mixed book.

### True economic risk by structure (v1)

| Structure | v1 role | Bounded downside (hold to expiry) | `max_loss_per_share` in S3 |
|-----------|---------|----------------------------------|------------------------------|
| **Iron fly / iron condor** | Short side | `wing_width − net_credit` per share | Set (geometric) |
| **Long straddle** | Long side | Premium paid (debit) | `entry_cost` / premium |
| **Short straddle** | Not v1 short pin | Theoretically unbounded | `None` — proxy only if code path used |

Short straddle tail risk can approach “stock goes against you” scale; v1 does **not** treat unlimited loss as a honest denominator without an explicit proxy ([Q5](#open-questions-for-hd-resolve-before-sprint-003-build)).

### Layer 1 — Sizing: `at_risk_per_share` (Phase 2)

Each instrument type supplies **at risk per share** to convert `max_loss_budget_per_trade` → contracts (Tier B) or conceptual weight (Tier A):

| Instrument | `at_risk_per_share` for sizing | Notes |
|------------|-------------------------------|-------|
| Iron fly / condor | `max_loss_per_share` (geometric) | From S3 assembly |
| Long straddle | Premium paid per share | Same as true max loss |
| Short straddle (if used) | `net_credit × short_straddle_risk_multiplier` | **Proxy** — not true max loss |

Equal **max-loss budget per trade** (v1 spec) means: each included name receives the same config budget slot; quantity (or Tier A weight) derives from that instrument’s `at_risk_per_share`.

### Layer 2 — Primary return (cross-book, S8 / go/no-go)

**Recommended primary per-trade metric:**

```text
return_on_allocated_budget = pnl_total / max_loss_budget_per_trade
```

- Same denominator for short iron fly and long straddle on a given run — “I allocated $X risk budget to this signal; what did it return?”
- Aligns with [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md) **“return on max-loss budget”** (budget = allocated slot, not necessarily geometric max loss for every structure).
- S8 Sharpe, drawdown, and side splits use this series by default ([Q10](#open-questions-for-hd-resolve-before-sprint-003-build)).

**Trade log column (target):** `return_on_allocated_budget`. Implementation may alias `return_on_max_loss` until rename — semantics are **allocated budget**, not geometric max loss for every row.

Tier A: same formula on conceptual dollar PnL (or equal `risk_fraction × book` PnL) with budget slot from config.

### Layer 3 — Structure-native diagnostics (per row, not mixed blindly)

Persist optional columns for audit and structure-specific research:

| Instrument | Diagnostic return | Denominator |
|------------|-------------------|-------------|
| Iron fly / condor | `return_on_risk_geometry` | `pnl_per_share / max_loss_per_share` |
| Long straddle | `return_on_premium` | `pnl_per_share / premium_paid` |
| Short straddle | `return_on_credit` | `pnl_per_share / net_credit` (ROC on credit; not worst-case loss) |

Do **not** average these across structures for go/no-go — denominators have different economic meaning.

### Layer 4 — Body credit (interim / optional diagnostic)

Today `surface_metrics.py` uses `pnl_per_share / body_credit_per_share` — ATM body premium as a cross-direction bridge (see `option_surface.py` diagnostics). Useful for **relative** config search; **not** sufficient for v1 go/no-go.

| Use | Keep? |
|-----|-------|
| Interim Sharpe while S5 return columns are built | Yes (until S8 migrated) |
| Secondary diagnostic after S5 | HD choice — [Q7](#open-questions-for-hd-resolve-before-sprint-003-build) |

### Mixed-book example (v1)

One rebalance, `max_loss_budget_per_trade = 500`:

| Row | Structure | Size | Settle PnL | Primary return |
|-----|-----------|------|------------|----------------|
| AAPL short | Iron fly | `qty` from geometric max loss | `pnl_total` | `pnl_total / 500` |
| NVDA long | Long straddle | `qty` from premium at risk | `pnl_total` | `pnl_total / 500` |

Same **500** budget slot; different sizing math; **comparable** primary return for S8 aggregation.

### What not to do

- Force all structures onto **premium only** — wrong for iron fly (wings define risk).
- Force all structures onto **geometric max loss** — undefined for short straddle without proxy.
- Apply **cross fill at S3** and a **post-trade spread penalty** (former S6) — double conservatism.
- Use unlimited-loss semantics for short straddle in go/no-go without an ADR.

### Return fields — target trade log (included rows)

| Field | Layer | Purpose |
|-------|-------|---------|
| `return_on_allocated_budget` | Primary | S8 / go/no-go |
| `return_on_risk_geometry` | Diagnostic | Iron fly / condor |
| `return_on_premium` | Diagnostic | Long straddle |
| `return_on_credit` | Diagnostic | Short straddle / credit structures |
| `at_risk_per_share` | Sizing audit | Echo value used for quantity |
| `body_credit_per_share` | Legacy diagnostic | Optional per [Q7](#open-questions-for-hd-resolve-before-sprint-003-build) |

---

## S6 — Collapsed into S5 (no separate v1 step)

| Was | Now |
|-----|-----|
| `step6_apply_cost` — spread penalty after settle | **Removed from target pipeline** — fill applied at S3 assembly |
| `return_on_allocated_budget` on trade log | **S5 Phase 3** after size + settle |
| `cost_model` vs `fill` dual knobs | **`fill` only** for v1; `cost_model` legacy / unused |

`pipeline.step6_apply_cost` remains a `pass` stub with a deprecation note until code cleanup in Sprint 003. No contract test for S6.

---

## S8 — Run metrics

### What S8 is for

Collapse trade log → **date summary** + **run summary** so config search and go/no-go use one return definition.

### Current (keep as interim)

- `date_return_on_body_credit`, `mean_trade_return_on_body_credit`, Sharpe on that series
- Useful for **relative** config search until max-loss series exists
- **Not** sufficient for v1 go/no-go ([backtest_evaluation_protocol.md](backtest_evaluation_protocol.md))

### Target (Sprint 003+)

| Metric | Definition |
|--------|------------|
| Per-trade return | `return_on_allocated_budget` from S5 trade log |
| Per-date return | Mean (or budget-weighted sum) of traded `return_on_allocated_budget` per `trade_date` — pin in Q10 |
| Sharpe | Annualized on per-date return series (weekly ≈ √52) |
| `availability_rate` | Traded / candidates (keep) |
| `hit_rate` | Fraction `pnl_per_share > 0` (keep) |
| `max_drawdown` | On per-date return series (same formula, new input) |
| Side splits | Long/short counts and means on **same** return denominator |

Deprecate body-credit Sharpe for go/no-go once max-loss series is validated; may keep body-credit columns for diagnostics.

### ORCH (orchestration)

**Target:** `SurfaceRunner.run_single_config` calls pipeline steps only:

```text
S1 → S2 → S3 → S4 → S5 (select, size, settle, return) → trade log → S8
```

S7 `settle()` is invoked **inside** S5, not a separate orchestration step after S5.

**Today:** S5+settle inline in runner; S8 after loop.

Sprint 003: extract runner tail to `step5`; add orchestration contract test.

---

## Open questions for HD (resolve before Sprint 003 build)

| # | Question | Options / notes |
|---|----------|-----------------|
| Q1 | ~~Total cap selection~~ | **Resolved (2026-06-07):** per-side cap via `max_names_per_side`; no global long+short pool — [decision 003](decisions/003_position_cap_per_side.md) |
| Q2 | ~~Config naming for cap~~ | **Resolved:** keep `max_names_per_side`; document example `25` per side for ~50-book |
| Q3 | ~~S6 vs fill~~ | **Resolved:** cross (or chosen) fill at S3 only; no S6 |
| Q4 | **Contract multiplier** | 100 for equity options — single constant in config or runner settings? |
| Q5 | **Short straddle risk proxy** | v1 short = iron fly/condor only; if `short_straddle` path remains: pin `short_straddle_risk_multiplier` (e.g. 2× credit) for **sizing**; use `return_on_credit` for diagnostic only — never unlimited loss in go/no-go |
| Q6 | **Trade log grain** | One row per (date, ticker, direction) — confirm no multi-structure per ticker |
| Q7 | **Body-credit metrics** | Drop from run summary vs keep as secondary diagnostic alongside `return_on_allocated_budget` |
| Q8a | ~~Sizing tier scope for Sprint 003~~ | **Resolved (2026-06-07):** implement **both** Tier A and Tier B in S5; `sizing_mode` switch |
| Q8b | **Default `sizing_mode` at run time** | `conceptual` vs `integer_lots` when caller does not specify |
| Q8c | **Tier B capital overrun** | Scale all quantities proportionally vs trim by rank |
| Q9 | **Tier A normalization** | Equal **budget slot** (`risk_fraction` × book) vs equal **premium slot** vs equal **weight** (`1/n`) — must align with `return_on_allocated_budget` |
| Q10 | **Primary return confirmation** | Adopt `return_on_allocated_budget` as S8 / go/no-go series (recommended); per-date = mean vs budget-weighted sum |
| Q11 | **Structure-native columns** | Which diagnostics to persist on trade log: all three vs iron-fly + long-straddle only (v1 book) |

---

## Sprint 003 build scope (preview — after HD review)

| Order | Work | Done when |
|-------|------|-----------|
| 1 | HD answers Q8b, Q9–Q11 (minimum); Q10 confirm primary return | Design doc → Accepted |
| 2 | Implement S5 in `pipeline.py` — both tiers, `return_on_allocated_budget` + diagnostics; runner delegates | Per-side cap per [decision 003](decisions/003_position_cap_per_side.md); contract tests cover both sizing modes |
| 3 | S8 on max-loss series | Sharpe uses new denominator; protocol aligned |
| 4 | ORCH contract test | Runner has no business logic duplication |
| 5 | Contract tests for S5/S8/ORCH | Written with implementation, not before |

---

## Relation to other docs

| Doc | Update when |
|-----|-------------|
| [surface_engine_data_contract.md](surface_engine_data_contract.md) | Sprint 003 — fill § S5/S8 from this doc |
| [v1_spec_pins.md](v1_spec_pins.md) | When cap config name and global budget pinned |
| [decisions/003_position_cap_per_side.md](decisions/003_position_cap_per_side.md) | Accepted 2026-06-07 |

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-31 | Sprint 002 Session C — initial design; contracts for S5/S6/S8/ORCH deferred |
| 2026-05-31 | S5 clarified: select + **size** + simulate trade (not disposition-only) |
| 2026-05-31 | Sizing reframed: constraint-driven tiers (conceptual vs integer lots + capital), not “quantity from config” |
| 2026-06-07 | **S6 collapsed into S5** (HD approved) — fill at S3; `return_on_max_loss` in S5 Phase 3; docs aligned |
| 2026-06-07 | S5 overview + Phase 1 aligned: select = cap/rank on S4 output; size = constraint tiers, not config quantity |
| 2026-06-07 | **Q8a resolved:** Sprint 003 implements both Tier A and Tier B in S5 (`sizing_mode` switch) |
| 2026-06-07 | **Q1/Q2 resolved:** per-side cap via `max_names_per_side` ([decision 003](decisions/003_position_cap_per_side.md)) |
| 2026-06-07 | § Return normalization — allocated budget primary; structure-native diagnostics; Q9–Q11 |
