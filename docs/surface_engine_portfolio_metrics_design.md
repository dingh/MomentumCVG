# Surface engine — portfolio and metrics design (S5 / S8)

**Status:** Accepted — design complete, all open questions resolved; HD sign-off 2026-06-10; ready for Sprint 003 build  
**Last updated:** 2026-06-10  
**Audience:** HD + agents  
**Companion:** [surface_engine_data_contract.md](surface_engine_data_contract.md), [surface_engine_data_flow.md](surface_engine_data_flow.md), [backtest_evaluation_protocol.md](backtest_evaluation_protocol.md)

---

## Purpose

Sprint 002 Sessions A–B pinned contracts through **structure build, exclusions, and per-share settlement** (S1–S4, S7). The remaining pipeline tail — **S5, S8, and full orchestration** — is not ready for contract tests because:

1. **S5 today is not a portfolio layer** — it is selection + settle wiring in `SurfaceRunner._select_size_and_settle`.
2. **Return columns** are not fully wired — M1–M3, `pnl_total`, and `capital_at_risk_dollars` are designed but not on the trade log yet; S8 cycle returns not implemented.
3. **Sizing is a constraint-satisfaction decision**, not an optimizer — config supplies limits (capital, max loss per name, cap count); S5 applies rules to preserve signal alpha at the chosen fidelity tier.

**S6 collapsed into S5 (HD decision):** Entry fill/cost is fixed at structure assembly (S3 + `FillAssumption`). A separate post-trade cost step is redundant for v1; trade returns (M1–M3, `pnl_total`, `capital_at_risk_dollars`) and `fill_label` are computed in S5 Phase 3.

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
| **S5** | Runner: eligibility filter + per-**side** cap → settle → `pnl_per_share` only | **No sizing policy** (`quantity` / tier weights missing); no `capital_at_risk_dollars`; settle bundled here; cap semantics **aligned** with [decision 003](decisions/003_position_cap_per_side.md) |
| **S5 returns** | Fill at S3; no separate cost step | No M1–M3, `pnl_total`, or `capital_at_risk_dollars` on trade log yet |
| **S8** | `surface_metrics` on **body credit** (`pnl / body_credit_per_share`) | Target: **`cycle_return_on_capital_at_risk`** per rebalance + side splits |
| **ORCH** | S1–S4 via `pipeline.py`; S5 inline in runner | Not a thin S1–S8 loop |

---

## What a “quality backtest” must support (v1)

Before real-money shadow/paper, a run must let you answer:

1. **Selection** — Who was a candidate, who traded, who excluded, and **why** (traceable S1→S5).
2. **Sizing** — Given constraints (total capital, max loss per position, position count, lot size), how large each simulated trade is — so returns still reflect **signal alpha**, not artifact of bad sizing.
3. **Simulated trade** — Each included row is a complete trade record (structure + size + settled PnL), not just a flagged candidate.
4. **Book composition** — How many positions per rebalance; respects **`max_names_per_side` per direction** (e.g. 25 long + 25 short ≈ 50-book).
5. **Economics** — Record **M1–M3** per trade; aggregate **cycle return** in S8 = `Σ pnl_total / Σ capital_at_risk_dollars` per rebalance (plus long/short side splits — see § Return normalization).
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
                   Tier A: per-side total budget ÷ name count → fractional units (premium or max-loss); no 100× lot
                   Tier B: integer lots from max-loss budget + capital binding
  → (3) SIMULATE — S7 settle; M1–M3 + pnl_total + capital_at_risk_dollars + quantity (fractional in A, integer in B)
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
| Eligibility | Drop or mark excluded rows where `structure_ok == False` or `had_earnings_nearby == True`; set `exclusion_reason` from upstream (`no_tradeable_structure`, `earnings_exclusion`) |
| Rank | Split by `direction`; within each side rank by `signal_rank_pct` (long: descending; short: ascending) |
| Cap | [Decision 003](decisions/003_position_cap_per_side.md): top **`max_names_per_side`** per direction (independent pools) |
| Include | Top `max_names_per_side` per side get `included_in_portfolio == True`; overflow gets `exclusion_reason = 'max_names_cap'` (existing code value) |
| Sizing reject | A row that passes selection but fails sizing (e.g. non-positive geometric max loss) gets `exclusion_reason = 'invalid_max_loss'` and `included_in_portfolio == False` |
| Output flags | `included_in_portfolio`, `exclusion_reason` on **every** candidate row (included and excluded) |

**Exclusion-reason vocabulary (match existing code):** `no_tradeable_structure`, `earnings_exclusion`, `max_names_cap`, `invalid_max_loss`. Sprint 003 reuses these strings — do not invent new labels without updating `surface_runner.py` and tests.

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
| Structure geometry | `at_risk_per_share` from S3 — **defined-risk only** for geometric max loss | Converts budget → contracts (see below) |

**Structure geometry note:** Only **defined-risk** structures (iron fly, iron condor) supply a true geometric `max_loss_per_share` for sizing. **Long straddle** uses **premium paid** as `at_risk_per_share` (debit at risk, not wing geometry). **Naked / short straddle is out of scope for v1** — the short side is defined-risk only (iron fly/condor).

When constraints conflict (e.g. up to `2 × max_names_per_side` included names × `max_loss_budget_per_trade` exceeds `deployable_capital`), **capital is the hard constraint**: drop names by rank (lowest `signal_rank_pct` first) until the book fits, or skip the date if even the minimum book is infeasible. Never silently exceed `deployable_capital` (see Tier B residual policy).

#### Two fidelity tiers (HD — not an optimizer)

**Tier A — Conceptual / signal-preserving base case**

Purpose: assess whether the **signal** has alpha before lot rounding and capital binding distort results.

**Defining difference from Tier B:** Tier A sizes in **fractional units and ignores the contract multiplier (the 100-share lot)**. No integer-contract rounding, no `< 1 contract` rejection.

**Control = a per-side *total* dollar budget, split equally across that side's included names** (not a per-name slot). Per name: `per_name_budget = side_total / n_side`, then `quantity = per_name_budget / per_share_denominator`.

| Assumption | Intent |
|------------|--------|
| Fractional size, no 100× lot | `quantity` may be fractional (e.g. 24.7 units); `contract_multiplier` **not** applied |
| Equal split of a **per-side total** | Configure a **total** dollar budget per side; divide equally by the number of included names on that side → per-name budget |
| Return metric | M1–M3 per share are size-agnostic; Tier A also emits `pnl_total` / `capital_at_risk_dollars` (fractional `quantity`) so cycle metrics match Tier B |

**Two modes (set by `tier_a_mode`):**

- **(a) `equal_premium`** — configure **total short-side premium to collect** `tier_a_short_budget` and **total long-side spend** `tier_a_long_budget` (two separate totals):
  - Short: each of `n_short` names targets `tier_a_short_budget / n_short` of credit → `quantity = (tier_a_short_budget / n_short) / credit_per_share`.
  - Long: each of `n_long` names spends `tier_a_long_budget / n_long` → `quantity = (tier_a_long_budget / n_long) / premium_paid_per_share`.
- **(b) `equal_max_loss`** — configure **total short-side max loss** `tier_a_short_budget`; the long side is **financed by the premium actually collected from the short side** (`tier_a_long_budget` is ignored):
  - Short: each of `n_short` names targets `tier_a_short_budget / n_short` of max loss → `quantity = (tier_a_short_budget / n_short) / max_loss_per_share`.
  - Long: `collected_short_premium = Σ(short quantity × credit_per_share)`; each of `n_long` longs spends `collected_short_premium / n_long` → `quantity = (collected_short_premium / n_long) / premium_paid_per_share`. Keeps the book conceptually **self-financing**.
  - **Edge rule:** cycles with no shorts, or non-positive `collected_short_premium` — fall back to `tier_a_long_budget` (a fixed long total) or skip the long side.

**Worked example — `equal_premium` (mode a):** `tier_a_short_budget = $10,000`, `tier_a_long_budget = $10,000`; this cycle has **10** short iron flies and **5** long straddles.

| Side | Per-name budget | Example per-share denom | Fractional `quantity` |
|------|-----------------|--------------------------|------------------------|
| Short (×10) | `10,000 / 10 = $1,000` credit | net credit `$2.00`/sh | `1,000 / 2.00 = 500` units |
| Long (×5) | `10,000 / 5 = $2,000` spend | straddle premium `$8.00`/sh | `2,000 / 8.00 = 250` units |

Then per name `capital_at_risk_dollars = abs(quantity) × at_risk_per_share` (short: `× max_loss_per_share`; long: `× premium_paid_per_share`) and `pnl_total = abs(quantity) × pnl_per_share`. **`quantity` sign** encodes direction (long `+`, short `−`); **`pnl_per_share`** is settle P&L per unit (positive = profit). Dollar fields always scale by **magnitude**, not sign. **`equal_max_loss` (mode b)** is identical on the short side but uses `max_loss_per_share` as the per-name denominator, and the long side spends the **realized** `collected_short_premium` (not a configured `$10,000`) split across the 5 longs.

This tier answers: *“If I spread premium/risk evenly across selected signals, does the signal side still win?”* — alpha at a **conceptual** level, free of lot rounding.

**Tier B — Deployable simulation (next level)**

Purpose: approximate what is **placeable** under real constraints while still not doing portfolio optimization.

| Rule | Intent |
|------|--------|
| Integer contracts | `quantity = floor(max_loss_budget_per_trade / (at_risk_per_share × contract_multiplier))` where `at_risk_per_share` is structure-specific (see § Return normalization — not always `max_loss_per_share`); reject or flag if `< 1`. **`at_risk_per_share` is per share; `contract_multiplier = 100` (pinned, equity options) converts to per-contract dollars — omitting it oversizes by 100×.** |
| Contract multiplier | Tier B: `quantity` = share-equivalent units (`contracts × 100`); simulate uses `abs(quantity) × pnl_per_share` (no second multiplier). Legacy design line `× contract_multiplier` at simulate time is **superseded** by ADR 004. |
| Per-name budget | Equal `max_loss_budget_per_trade` per name (no proportional rescaling — when the book is too large, drop names by rank, see residual policy) |
| Capital check | If `deployable_capital` is set: `sum(quantity × at_risk_per_share × contract_multiplier) ≤ deployable_capital` — same as `Σ capital_at_risk_dollars`. **If `deployable_capital` is `None` (v1 default): no book-level binding** — only the per-name `max_loss_budget_per_trade` applies and no capital-driven drops occur. |
| Residual policy | **Capital is a hard constraint (only when `deployable_capital` is set).** With correct sizing, no overrun is expected. If even the **minimum** position (1 contract) for a name cannot fit within remaining capital: **drop names by rank** (lowest `signal_rank_pct` first) until the book fits, or **skip the date** if the minimum book is infeasible. Never silently exceed `deployable_capital` and never proportionally rescale quantities. |

This tier answers: *“After lots and capital bind, is alpha still there?”* — closer to shadow/paper, still not full optimization.

Both tiers live in **S5** (same select phase; sizing policy branch). S8 aggregates S5 return fields.

#### What this is not

- End-to-end portfolio optimization (weights from signal strength, correlation, sector)
- Dynamic rebalancing of size across dates from prior PnL
- Margin model / broker-specific rounding

Those can come later if Tier B backtests justify them.

**Runner today:** **no sizing phase** — `quantity` absent; per-share settle only; no tier A weights or tier B capital check.

**Sprint 003 (HD — 2026-06-07):** implement **both** tiers in S5. Runner config `sizing_mode` switches between Tier A (`conceptual`) and Tier B (`integer_lots`). **`sizing_mode` is required — there is no default; a backtest fails fast if it is not specified.** Same select + simulate path; sizing policy branch only. Both code paths required for Sprint 003 done.

### Phase 3 — Simulate trade (settle + returns)

Entry economics (mid vs cross) are fixed in **S3** via `config.fill` — no separate cost pass.

For each sized, included row:

| Step | Rule |
|------|------|
| Settle | `position = assembly.settle(exit_spot)` (S7 math) |
| Per share | `pnl_per_share = position.pnl` (already reflects fill chosen at assembly) |
| Total | `pnl_total = abs(quantity) × pnl_per_share` (both tiers; no extra `contract_multiplier` at simulate — embedded in Tier B `quantity`) |
| Return | **M1–M3** per share + **`pnl_total`** + **`capital_at_risk_dollars`** (structure-specific; see § Return normalization) |
| Audit | `fill_label` = `config.fill.label` on row / run summary |
| Trade log | Row is a **simulated trade**: identity + structure + size + PnL + return |

**Former S6 scope** (`step6_apply_cost`, `cost_model` spread penalty) is **not** a v1 pipeline step — would double-count if cross fill is already used. `cost_model` in `BacktestRunConfig` is legacy; use `fill` only.

### Outputs (target trade log row)

**Grain:** one row per `(trade_date, ticker, direction)`. **At most one structure per ticker per side per cycle** — no multi-structure-per-ticker fan-out. `direction` ∈ {long, short}; a ticker may in principle appear once long and once short, but not twice on the same side.

**All candidates (when `include_diagnostics=True`)**

| Field | Purpose |
|-------|---------|
| `included_in_portfolio` | Selected for simulated trade |
| `exclusion_reason` | Set when not traded |
| `trade_date`, `ticker`, `direction` | Identity (grain key) |

**Included rows only — simulated trade**

| Field | Purpose |
|-------|---------|
| `sizing_mode` | Echo config: `conceptual` (Tier A) or `integer_lots` (Tier B) |
| `quantity` | **Tier B:** integer contracts; **Tier A:** fractional units (e.g. `1.23`). Always populated (no NaN) for included rows |
| `risk_fraction` | **Descriptive only** (per-side weight, e.g. realized `capital_at_risk_dollars` share within the side); **not** the sizing mechanism — Tier A sizes by the per-side total budget ÷ name count |
| `max_loss_budget_per_trade` | Config echo — **Tier B only** (per-name max-loss budget); Tier A uses `tier_a_short_budget` / `tier_a_long_budget` instead |
| `max_loss_per_share` | From structure |
| `pnl_per_share` | S7 settle |
| `pnl_total` (or `pnl_dollars`) | `abs(quantity) × pnl_per_share` (both tiers; Tier B `quantity` already share-equivalent) |
| `capital_at_risk_dollars` | Structure-specific total capital at risk (see § Portfolio return) — **S8 cycle denominator building block** |
| `return_on_premium` (M1) | Per § Return normalization |
| `return_on_max_loss` (M2) | Realized `pnl_per_share / max_loss_per_share`; `NaN` for straddles — **not** S3 entry `theoretical_return_on_max_loss` |
| `return_on_atm_straddle` (M3) | Per § Return normalization; **= M1** on pure straddles; **≠ M1** on iron fly / condor |
| `return_on_capital_at_risk` | Optional per-trade audit: `pnl_total / capital_at_risk_dollars` |
| `fill_label` | Echo `config.fill` (reproducibility) |
| `theoretical_return_on_max_loss` | Entry economics from S3 (audit); not M2 |

### Settle boundary (S5 vs S7)

- **S7** — pure payoff math at `exit_spot` (per share).
- **S5** — owns **when** settle runs (included rows only) and **combines** settle result with **quantity** to form the simulated trade.

### Config levers (constraints + policy switch, not quantity assignment)

| Config field | Status | Role |
|--------------|--------|------|
| `sizing_mode` | **NEW** (add to `BacktestRunConfig`) | **Required** — no default; `conceptual` (Tier A) vs `integer_lots` (Tier B); validation fails if unset |
| `tier_a_mode` | **NEW** | Tier A only — `equal_premium` (mode a) vs `equal_max_loss` (mode b); see § Tier A |
| `tier_a_short_budget` | **NEW** | Tier A only — **total** short-side budget (premium to collect in mode a; max loss in mode b), split equally across short names |
| `tier_a_long_budget` | **NEW** | Tier A only — **total** long-side spend (mode a); **ignored in mode b** (long financed by collected short premium; used only as the edge-case fallback) |
| `contract_multiplier` | **NEW** | **Pinned = 100** (equity options); Tier B per-contract conversion |
| `deployable_capital` | **NEW** (optional; `None` in v1) | Total book **hard** constraint for Tier B; `None` ⇒ only per-name budget binds |
| `max_names_per_side` | exists | Per-direction position cap (long and short ranked separately) |
| `max_loss_budget_per_trade` | exists | **Tier B** per-name max-loss budget (Tier A uses `tier_a_*_budget` totals) |
| `include_diagnostics` | exists | Keep excluded candidates in log |
| `earnings_exclusion_days` | exists | Already applied in S4 |

**Config tension to resolve in Sprint 003:** `short_structure` currently accepts `'straddle'` (`VALID_SHORT_STRUCTURES`), but Q5 pins v1 short side to **defined-risk only** (`ironfly` / `ironcondor`). Either reject `short_structure == 'straddle'` in `__post_init__` for v1, or leave it explicitly **unsupported / untested**. Implementer must not silently size a naked short straddle.

---

## Return normalization (S5 Phase 3 + S8)

Mixed books — e.g. **short iron fly + long straddle** (v1 pin) — need **three recorded return metrics** on every traded row, plus a **portfolio-level** metric for S8 aggregation. Not every metric applies to every structure — use **`NaN`** when a denominator is undefined.

### HD pin (2026-06-07): three normalized returns

All three are computed in **S5 Phase 3** after settle and stored on the trade log. Denominators come from S3 assembly fields (per share).

| # | Column (target) | Formula | Denominator meaning | Example |
|---|-----------------|---------|---------------------|---------|
| **M1** | `return_on_premium` | `pnl_per_share / structure_premium_per_share` | Premium **received** (short / credit) or **paid** (long / debit) for this structure | Short straddle: `pnl / credit received` |
| **M2** | `return_on_max_loss` | `pnl_per_share / max_loss_per_share` | Defined-risk capital at stake (`wing_width − net_credit` for iron fly/condor) | Iron fly: `pnl / max_loss` |
| **M3** | `return_on_atm_straddle` | `pnl_per_share / atm_straddle_premium_per_share` | **ATM body straddle premium only** (call + put at body) — **not** net structure credit | Iron fly: compares winged PnL to “what a naked ATM straddle would have collected” |

**Denominator sources (S3):**

| Field on structure row | Used for |
|--------------------------|----------|
| `structure_premium_per_share` | M1 — net credit received or debit paid for the assembled structure |
| `max_loss_per_share` | M2 — geometric max loss (iron fly / condor) |
| `atm_straddle_premium_per_share` | M3 — today `body_credit_per_share` in `option_surface.py` (ATM body call + put premium **only**, excluding wing legs) |

**Where these are produced (v1 pin):** **derive the three denominators (`structure_premium_per_share`, `max_loss_per_share`, `atm_straddle_premium_per_share`) and `at_risk_per_share` inside S5 Phase 3 from existing S3 fields** — do **not** add new columns to the S3 schema (S1–S4 contracts are already frozen). `atm_straddle_premium_per_share` maps to the existing `body_credit_per_share`; `max_loss_per_share` already exists; `structure_premium_per_share` = net credit/debit from the `_assembly`. If a future sprint prefers materializing them at S3, update the S3 contract first.

**M1 vs M3 on iron fly / condor:** M1 divides by **net structure premium** (credit received after wings). M3 divides by **ATM straddle premium** (body only). They are **not** the same denominator — both are required to separate “return on credit kept” from “return vs naked straddle economics.”

M2 uses **max loss** (`width − credit`), not raw wing width alone. Raw `pnl / wing_width` may be kept as an optional diagnostic but is not one of the three pinned metrics.

**Denominator sign convention:** all M1–M3 denominators are stored as **positive magnitudes** (premium paid/received, max loss, ATM straddle premium are all `> 0`), so the ratio's sign tracks `pnl_per_share`. Any metric is **`NaN`** when its denominator is `≤ 0` or undefined (e.g. M2 when `max_loss_per_share ≤ 0`, M1 when premium `≤ 0`).

### Worked example — iron fly vs naked short straddle (HD)

Per-share economics at entry:

| Item | $/share |
|------|---------|
| ATM body straddle premium (call + put) | **$3** ← M3 denominator |
| Wing cost | $1 |
| **Net credit received** (structure premium) | **$2** ← M1 denominator |

Hold-to-expiry **PnL = +$1**:

| Metric | Calculation | Value |
|--------|-------------|-------|
| **M1** `return_on_premium` | `1 / 2` | **50%** — return on net credit kept |
| **M3** `return_on_atm_straddle` | `1 / 3` | **33%** — return vs ATM straddle premium |

Compare to a **pure short straddle** at the same body with PnL **+$2** and premium **$3** _(illustrative only — short straddle is not a v1 traded structure)_:

| Metric | Calculation | Value |
|--------|-------------|-------|
| **M1** (= **M3** on pure straddle) | `2 / 3` | **67%** |

**Interpretation:** wings and structure cost reduced alpha from **67%** (naked straddle, M3) to **33%** (iron fly, M3) on an ATM-premium basis — the structural drag is visible only when M3 uses the **$3** ATM denominator, not the **$2** net credit.

### M3 on straddles — same as M1

On a **pure long or short straddle**, the structure premium **is** the ATM body straddle premium (no wings). Therefore:

```text
return_on_atm_straddle (M3)  ==  return_on_premium (M1)
```

Still **persist both columns** on every row so fly/condor and straddle rows share the same schema and M3 is always the cross-structure comparison line (e.g. mean M3 across short book compares winged vs naked on equal footing).

### Entry vs realized returns (do not mix)

| When | Column (target) | Formula | Code today (S3) |
|------|-----------------|---------|-----------------|
| **Entry** (S3 assembly) | `theoretical_return_on_max_loss` | `net_credit / max_loss_per_share` at open | On structure row from `option_surface.py` |
| **Realized** (S5 settle) | `return_on_max_loss` (M2) | `pnl_per_share / max_loss_per_share` | Not wired |

Same name family, different numerator: entry uses **credit**, realized uses **settled PnL**. Never use entry ROC as backtest performance.

### `structure_premium_per_share` mapping (M1 denominator)

| Structure | `structure_premium_per_share` (M1) | `atm_straddle_premium_per_share` (M3) |
|-----------|-----------------------------------|--------------------------------------|
| Iron fly / condor | **Net credit** after wings (e.g. **$2**) | **ATM body** call+put only (e.g. **$3**) — `body_credit_per_share` |
| Long straddle | Premium paid = ATM straddle debit | **Same value** as M1 |
| Short straddle _(out of scope for v1)_ | Credit received = ATM straddle credit | **Same value** as M1 |

Sprint 003: **derive these in S5 from existing S3 fields** (see § "Where these are produced") — do not add new S3 columns.

### Availability matrix (`NaN` when not applicable)

| Structure | M1 `return_on_premium` | M2 `return_on_max_loss` | M3 `return_on_atm_straddle` |
|-----------|------------------------|-------------------------|------------------------------|
| **Iron fly / condor** (short) | ✓ (net credit) | ✓ | ✓ (ATM body prem; **≠ M1**) |
| **Long straddle** | ✓ (premium paid) | `NaN` | ✓ (**= M1**) |
| **Short straddle** _(out of scope for v1)_ | ✓ (credit received) | `NaN` | ✓ (**= M1**) |

**M2** is `NaN` for long straddle — no wing-defined geometric max loss separate from premium. **Short straddle is out of scope for v1** (short side is defined-risk only); the row is kept for schema reference, not traded.

**M3 on straddles:** not `NaN` — equals M1 because ATM straddle premium is the whole structure premium. M3 remains useful so short-book aggregates can mix iron fly and straddle rows on one comparison basis.

### Sizing vs return (do not conflate)

| Phase | Question | Answer |
|-------|----------|--------|
| **S5 Phase 2** | How large is the trade? | per-share **sizing denominator** (premium or at-risk, per mode) → `quantity` (fractional in Tier A, integer in Tier B) |
| **S5 Phase 3** | How did the trade score? | M1 / M2 / M3 per share + `pnl_total` / `capital_at_risk_dollars` (both tiers) |

**Sizing `at_risk_per_share`:**

| Instrument | For quantity / budget |
|------------|----------------------|
| Iron fly / condor | `max_loss_per_share` |
| Long straddle | Premium paid (= structure premium) |
| Short straddle | **Out of scope for v1** — short side is defined-risk only; no naked short straddle |

**Note — sizing denominator vs at-risk denominator:** the table above (`at_risk_per_share`) is the sizing denominator for **Tier B** and **Tier A `equal_max_loss`**. In **Tier A `equal_premium`**, sizing instead divides the per-name budget by the **structure premium per share** (credit for shorts, debit for longs). Regardless of mode, **`capital_at_risk_dollars` always uses `at_risk_per_share`** (max loss for defined-risk shorts; premium paid for long straddles) — so the sizing denominator and the at-risk denominator can differ for a short iron fly in `equal_premium` mode.

### Portfolio return (S8 / go-no-go) — separate from M1–M3

**M1–M3** are **per-trade, per-share** structure economics. **Portfolio return** is **per rebalance cycle** (e.g. one weekly trade date): total PnL across all included names divided by **total capital at risk** in that cycle — using each structure’s natural at-risk dollars.

#### Per-trade: `capital_at_risk_dollars` (cycle building block)

| Structure | `capital_at_risk_dollars` | Matches |
|-----------|---------------------------|---------|
| Iron fly / condor (short) | `abs(quantity) × max_loss_per_share` | Total **max-loss** capital |
| Long straddle | `abs(quantity) × premium_paid_per_share` | Total **premium paid** |
| Short straddle | **Out of scope for v1** | Short side is defined-risk only |

Also store **`pnl_total`** per included row. Optional per-trade audit: `return_on_capital_at_risk = pnl_total / capital_at_risk_dollars`.

#### Per cycle (S8 — primary go/no-go series)

For one **`trade_date`** (rebalance cycle), over **included** rows only:

```text
cycle_return_on_capital_at_risk
    = Σ pnl_total  /  Σ capital_at_risk_dollars
```

**Side splits** (same cycle, same formula within direction):

```text
short_cycle_return = Σ pnl_total (short) / Σ capital_at_risk_dollars (short)
long_cycle_return  = Σ pnl_total (long)  / Σ capital_at_risk_dollars (long)
```

When only one name trades on a side, this reduces to that name’s `pnl_total / capital_at_risk_dollars` — e.g. short iron fly: `pnl_ticker1 / max_loss_dollars_ticker1`; long straddle: `pnl_ticker2 / premium_dollars_ticker2`.

**Empty-side / zero-denominator rule:** if `Σ capital_at_risk_dollars == 0` for a cycle (or a side has no included trades), the corresponding cycle return is **`NaN`** (excluded from the Sharpe series), not `0`.

**Sharpe / drawdown / go/no-go** ([backtest_evaluation_protocol.md](backtest_evaluation_protocol.md)) use the **per-date** `cycle_return_on_capital_at_risk` series (and side splits for diagnostics). **Do not** average M1–M3 across structures for portfolio Sharpe.

#### Worked example — one rebalance cycle (HD)

Two included names, Tier B, multiplier = 100:

| Ticker | Structure | `pnl_total` | `capital_at_risk_dollars` | Per-name `pnl / at_risk` |
|--------|-----------|-------------|---------------------------|--------------------------|
| Ticker1 | Short iron fly | **+$500** | **$2,000** (max loss) | **25%** |
| Ticker2 | Long straddle | **+$300** | **$1,500** (premium paid) | **20%** |

**Cycle (book):**

```text
cycle_return = (500 + 300) / (2000 + 1500) = 800 / 3500 ≈ 22.9%
```

**Sides:**

```text
short_cycle_return = 500 / 2000 = 25%
long_cycle_return  = 300 / 1500 = 20%
```

#### Relation to `max_loss_budget_per_trade`

Config **`max_loss_budget_per_trade`** is a **sizing constraint** (target per-name budget when deriving `quantity`). **`capital_at_risk_dollars`** is the **realized** dollars at risk after sizing (`quantity × at_risk_per_share × multiplier`). Cycle return uses **realized** capital at risk in the denominator, not `n × max_loss_budget_per_trade`, unless sizing makes them equal by construction.

### Mixed-book example (per-share M1–M3)

| Row | M1 | M2 | M3 |
|-----|----|----|-----|
| Ticker1 short iron fly | `1/2` (pnl / **$2** credit) | `pnl / max_loss` | `1/3` (pnl / **$3** ATM) |
| Ticker2 long straddle | `pnl / premium_paid` | `NaN` | **same as M1** |

### What not to do

- Use **M1** where you mean net structure premium; use **M3** where you mean ATM straddle comparison — on iron fly they differ ($2 vs $3 in the worked example).
- Average M1, M2, or M3 **across structure types** for go/no-go — denominators differ (M3 is the preferred line for short vol “vs naked straddle” research).
- Report a finite **M2 for any straddle** (long or short) — M2 is `NaN` by definition (no wing-defined max loss); short straddle is out of v1 regardless.

### Return fields — target trade log (included rows)

| Field | Purpose |
|-------|---------|
| `return_on_premium` | **M1** — always try; `NaN` only if premium ≤ 0 |
| `return_on_max_loss` | **M2** — iron fly / condor; else `NaN` |
| `return_on_atm_straddle` | **M3** — all structures; **= M1** on straddles; ATM body premium only on fly/condor |
| `capital_at_risk_dollars` | S8 cycle denominator building block |
| `return_on_capital_at_risk` | Optional per-trade: `pnl_total / capital_at_risk_dollars` |
| `structure_premium_per_share` | M1 denominator (audit) |
| `max_loss_per_share` | M2 denominator (from S3) |
| `atm_straddle_premium_per_share` | M3 denominator (`body_credit_per_share` today) |
| `at_risk_per_share` | Sizing audit |

---

## Implementation annotations (pin before Sprint 003 build)

Items below were deliberate TBDs during design; **all are now resolved** (2026-06-07). They are retained as implementation notes for the Sprint 003 build.

### Tier A — fractional sizing, no contract multiplier

> **Annot.** Tier A's defining feature: **fractional `quantity`, no contract multiplier (100-share lot), no integer rounding** — split a per-side total budget equally by name count, then solve for fractional units. Tier B applies the multiplier and rounds to integer contracts.
>
> **Minimum v1:** always compute **M1–M3 from `pnl_per_share`** (size-agnostic per-share economics).
>
> **Resolved (2026-06-07):** Tier A **does** emit `pnl_total` and `capital_at_risk_dollars` from its fractional `quantity`:
> - `pnl_total = abs(quantity) × pnl_per_share` (multiplier omitted; sign = direction only)
> - `capital_at_risk_dollars = abs(quantity) × at_risk_per_share`
>
> So Tier A produces the **same** `cycle_return_on_capital_at_risk` and side splits as Tier B. **The `contract_multiplier` cancels in the cycle ratio** (`Σ pnl_total / Σ capital_at_risk_dollars`), so Tier A and Tier B cycle returns differ **only** by Tier B's integer-lot rounding. Tier A sizing is set by `tier_a_mode` (`equal_premium` / `equal_max_loss`) over **per-side total budgets** (`tier_a_short_budget`, `tier_a_long_budget`) split equally across each side's included names — see § Tier A.

### `risk_fraction` — per side or whole book?

> **Annot.** Earlier drafts sized Tier A by `risk_fraction ≈ 1 / n_included`. With per-side cap ([decision 003](decisions/003_position_cap_per_side.md)), **`n` was ambiguous**:
>
> - **Per side:** `1 / n_long_included` within long pool (and separately for short) — keeps long/short books balanced.
> - **Whole book:** `1 / (n_long + n_short)` — single book weight.
>
> **Resolved (2026-06-07):** weighting is **per side** (consistent with per-side cap and mode (b), which finances the long book from short-side credit each cycle). Note: Tier A sizes by a **per-side total budget ÷ name count**, which *is* an equal per-side split — so `risk_fraction` is descriptive only, not a separate sizing knob. Document the per-side convention in the S5 contract.

### `max_names_per_side` default

> **Annot.** [Decision 003](decisions/003_position_cap_per_side.md) uses **25 per side** as an example (~50-book when both sides fill). This is **not** pinned as the config default in `BacktestRunConfig` — search scripts must set it explicitly. Setting `50` per side allows **100** total names.

### S8 scope vs M1–M3

> **Annot.** **M1–M3** are **per-trade** columns on the trade log for structure-level analysis and export.
>
> **S8** (date/run summary) computes **`cycle_return_on_capital_at_risk`** per `trade_date` plus **`short_cycle_return`** / **`long_cycle_return`**. Sharpe and drawdown use the cycle series (see § Portfolio return). M1–M3 remain trade-log only unless research adds structure-level date means later.

### Entry economics on trade log

> **Annot.** Carry forward S3 `theoretical_return_on_max_loss` on the trade log row for audit (entry credit vs max-loss geometry). Do not conflate with realized M2.

---

## S6 — Collapsed into S5 (no separate v1 step)

| Was | Now |
|-----|-----|
| `step6_apply_cost` — spread penalty after settle | **Removed from target pipeline** — fill applied at S3 assembly |
| `pnl_total`, `capital_at_risk_dollars`, M1–M3 on trade log | **S5 Phase 3** after size + settle |
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
| Per-cycle return (book) | `cycle_return_on_capital_at_risk` = `Σ pnl_total / Σ capital_at_risk_dollars` per `trade_date` |
| Per-cycle return (sides) | `short_cycle_return`, `long_cycle_return` — same formula within direction |
| Sharpe | Annualized on **`cycle_return_on_capital_at_risk`** series (weekly ≈ √52) |
| `availability_rate` | Traded / candidates (keep) |
| `hit_rate` | Fraction `pnl_per_share > 0` (keep) |
| `max_drawdown` | On per-date **cycle** return series |
| Side diagnostics | Long/short PnL totals, capital at risk totals, and side cycle returns |

Migrate S8 from interim `body_credit` Sharpe to **`cycle_return_on_capital_at_risk`**. M1–M3 remain **trade-log only** (see § Implementation annotations).

### ORCH (orchestration)

**Target:** `SurfaceRunner.run_single_config` calls pipeline steps only:

```text
S1 → S2 → S3 → S4 → S5 (select, size, settle, return) → trade log → S8
```

S7 `settle()` is invoked **inside** S5, not a separate orchestration step after S5.

**Today:** S5+settle inline in runner; S8 after loop.

Sprint 003: extract runner tail to `step5`; add orchestration contract test.

---

## Open questions for HD — all resolved (2026-06-07)

**No open items remain.** All design questions are resolved and specified in the relevant sections above and the change log. Summary of the final decisions:

| # | Decision |
|---|----------|
| Q4 | **`contract_multiplier` pinned = 100** (equity options); a config field, not a per-run unknown. |
| Q5 | **No naked short straddle in v1** — the short side is **defined-risk only** (iron fly / condor). No `short_straddle_risk_multiplier` proxy needed; short-straddle rows kept only for schema reference. |
| Q6 | **Trade log grain = one row per `(trade_date, ticker, direction)`** — no multi-structure per ticker per side. |
| Q8b | **`sizing_mode` is a required runner-config field** — no default; a backtest fails fast if it is not specified. |
| Q8c | **Capital is a hard constraint.** Correct sizing never overruns. If the minimum position (1 contract) cannot fit, **drop names by rank** (lowest `signal_rank_pct` first) or **skip the date** — never silently exceed `deployable_capital`. |
| Q9 | **Tier A emits `pnl_total` and `capital_at_risk_dollars`** from fractional `quantity` (no multiplier), so it produces the same `cycle_return_on_capital_at_risk` as Tier B (multiplier cancels in the ratio; only integer-lot rounding differs). Sizing control is a **per-side total budget** (`tier_a_short_budget` / `tier_a_long_budget`) split equally across that side's names, via `tier_a_mode`: **(a) `equal_premium`** or **(b) `equal_max_loss` with long side financed by collected short premium**. |

**Design status: ready for HD sign-off / Sprint 003 build.**

---

## Sprint 003 build scope (preview — after HD review)

| Order | Work | Done when |
|-------|------|-----------|
| 1 | ✅ HD resolved **Q4, Q5, Q6, Q8b, Q8c, Q9** (2026-06-07) | Design doc → Accepted |
| 2 | Implement S5 in `pipeline.py` — both tiers; M1–M3 + `pnl_total` + `capital_at_risk_dollars`; runner delegates | Per-side cap per [decision 003](decisions/003_position_cap_per_side.md); contract tests cover both sizing modes |
| 3 | S8 cycle returns | `cycle_return_on_capital_at_risk` + side splits; Sharpe on cycle series |
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
| 2026-06-07 | **S6 collapsed into S5** (HD approved) — fill at S3; returns in S5 Phase 3; docs aligned |
| 2026-06-07 | S5 overview + Phase 1 aligned: select = cap/rank on S4 output; size = constraint tiers, not config quantity |
| 2026-06-07 | **Q8a resolved:** Sprint 003 implements both Tier A and Tier B in S5 (`sizing_mode` switch) |
| 2026-06-07 | **Q1/Q2 resolved:** per-side cap via `max_names_per_side` ([decision 003](decisions/003_position_cap_per_side.md)) |
| 2026-06-07 | § Return normalization — allocated budget primary; structure-native diagnostics; Q9–Q11 |
| 2026-06-07 | HD pin: three return metrics M1–M3 (premium / max loss / ATM straddle); NaN matrix; Q7/Q11 resolved |
| 2026-06-07 | Consistency pass: Tier B sizing by `at_risk_per_share`; entry vs realized returns; § Implementation annotations |
| 2026-06-07 | Sizing constraints: geometric max loss applies to defined-risk structures only; not naked short straddle |
| 2026-06-07 | M3 clarified: ATM body premium denominator; worked example; M3 = M1 on straddles; persist both columns |
| 2026-06-07 | **Q10 resolved:** cycle portfolio return = Σ pnl / Σ capital_at_risk; short/long side cycle returns |
| 2026-06-07 | Fix Tier B sizing: `quantity = floor(budget / (at_risk_per_share × contract_multiplier))` — prevents 100× oversizing; capital check uses `contract_multiplier` |
| 2026-06-07 | Tier A clarified: fractional sizing, no 100× multiplier; modes (a) equal premium slot, (b) equal max-loss slot with premium-financed long side |
| 2026-06-07 | **All open questions resolved (Q4/Q5/Q6/Q8b/Q8c/Q9):** `contract_multiplier=100` pinned; no naked short straddle in v1 (defined-risk short only); trade-log grain = (date, ticker, direction); `sizing_mode` required (no default); capital is a hard constraint (drop by rank / skip date, never overrun); Tier A emits `pnl_total`/`capital_at_risk_dollars` (multiplier cancels in cycle ratio). Design ready for HD sign-off. |
| 2026-06-07 | **Implementation-readiness pass (vs code):** exclusion strings aligned to code (`max_names_cap`, `invalid_max_loss`); flagged NEW config fields vs existing; defined `deployable_capital=None` behavior; Tier A `quantity` fractional (not NaN); `risk_fraction` demoted to descriptive; denominator sign convention + zero-denominator NaN rules; noted `short_structure='straddle'` config tension. |
| 2026-06-07 | **Tier A sizing reframed (HD):** control is a **per-side total budget** split equally by name count (not a per-name slot). `tier_a_mode` ∈ {`equal_premium`, `equal_max_loss`}; `tier_a_short_budget` / `tier_a_long_budget` (long financed by collected short premium in `equal_max_loss`). Replaces the `$T = max_loss_budget_per_trade` reuse; added worked example. |
| 2026-06-07 | Final sweep: clarified sizing denominator (premium in `equal_premium`) vs at-risk denominator (always `at_risk_per_share` for `capital_at_risk_dollars`); Tier A `pnl_total` has no multiplier; denominators derived in S5 (not S3); tagged short-straddle reference rows out-of-scope. |
| 2026-06-16 | **Sprint 003 Phase 4 pin:** `quantity` sign = long/short only; dollar fields use `abs(quantity)`. `pnl_total = abs(quantity) × pnl_per_share`; `capital_at_risk_dollars = abs(quantity) × at_risk_per_share`. Aligns S7 settle (pnl positive = profit) with S8 cycle sums. Supersedes earlier `quantity × pnl_per_share` wording and Tier B simulate-time `× contract_multiplier`. |
