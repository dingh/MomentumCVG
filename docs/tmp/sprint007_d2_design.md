# Sprint 007 D2 — Implementation-shortfall mechanism

**Status:** `PROPOSED — AWAITING ACCEPTANCE`  
**Updated:** 2026-09-03  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md) — sprint acceptance status **unchanged** by this document  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md) §6.4–§6.6, §9  
**D0:** `READY_WITH_NARROW_ENABLING_CHANGE` — `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/` (commit `8a59474`)  
**D1 design / evidence:** [`sprint007_d1_design.md`](sprint007_d1_design.md); [`sprint007_d1_evidence_review.md`](sprint007_d1_evidence_review.md) (rule verdict `D1_CONTINUE_TO_D2`; D1 not marked accepted here)  
**Authorization:** **D2 design only.** No D2 implementation, no new economics, no D3.

---

## Summary

| Item | D2 design decision |
|---|---|
| **Question** | Of the accepted midpoint-to-cross dollar-P&L shortfall, how much comes from direct entry-price concession, how much from fill-dependent sizing and financing, and where is the dominant mechanism concentrated? |
| **Method** | Notebook-first. **D2A** (mandatory bridge) then a **human checkpoint**. **D2B** is one frozen follow-up, executed only after the checkpoint. |
| **Fill / window** | Paired **mid and cross**; primary window `2020-01-01` → `2026-07-10`; included keys (D0: 9,212, mid=cross) |
| **Primary unit** | Dollar `pnl_total`. CAR is companion only — never the residual. |
| **Fixed-quantity convention** | **Laspeyres / Q_mid:** reprice official cross per-unit P&L at midpoint `abs(quantity)`. |
| **Runtime** | Minutes on official artifacts. No `SurfaceRunner` rerun. |

D2 diagnoses **why** the frozen expression’s midpoint book (~+$159k, D1) becomes the accepted full-cross result. It does not estimate attainable fills, choose filters, or price alternative structures.

---

## Competing explanations D2 must distinguish

From the sprint plan, mapped onto D2 outputs (not D4 labels):

| Explanation | D2A/D2B signature |
|---|---|
| **Direct quote concession** | `Δ_price` dominates `G`; `Δ_size` not material; residual in tolerance |
| **Fill-dependent sizing / financing** | `Δ_size` material; Tier-A chain (D2B) shows short fill → short Q and/or short credit → long Q |
| **Side / structure concentration** | Dominant component lives on one side or iron-fly body vs wing (D2A side + optional D2B body/wing) |
| **Selective package tradability** | After side/structure, `Δ_price` still concentrates in an ex-ante **package half-spread / \|mid cashflow\|** slice **and** mid margin remains outside it |
| **Trade-set / opportunity difference** | `Δ_set ≠ 0` on included keys (D0 expects **0**) |
| **Calculation / evidence gap** | Residual out of tolerance, join failure, or D0/D1 prerequisite fail → `D2_BLOCKED` |

D2 does **not** decide D4. It hands D3 a mechanism class (below).

---

## Inspection notes (do not silently expand D2A)

Independent reading of `pipeline._apply_tier_a_sizing`, `_apply_simulate`, `surface_runner.serialize_constructable_legs`, and `FillAssumption`:

1. **Accounting identity (use this; do not re-settle).**  
   `pnl_total = abs(quantity) × pnl_per_share`, `pnl_per_share = Σ pnl_per_unit`, `pnl_per_unit = expiry_payoff_per_unit − entry_cash_per_unit`. D0 already showed settlement (`exit_spot`, `expiry_payoff_per_unit`) and `unit_quantity` match mid vs cross. Per-unit P&L therefore differs **only** through entry cash / `fill_price`. Use `fill_price` / `entry_cash_per_unit`, **not** stored ORATS `mid`.

2. **Order sensitivity (one companion, not a fourth residual).**  
   Official D2A is two-step at **Q_mid then p_cross**. Algebraically  
   `G = Q_mid·Δp + p_cross·ΔQ = Q_mid·Δp + p_mid·ΔQ + Δp·ΔQ`.  
   The interaction `I = Σ Δp·ΔQ` is **inside `Δ_size`**. The Laspeyres–Paasche price difference is the same object: `|Δ_price − Δ_price_Paasche| = |I|` where `Δ_price_Paasche = P_cross − P(Q_cross, p_mid)`. Disclose **one** statistic `S_order = |I| / |G|` (`G ≠ 0`). Treat it as **attribution sensitivity**, not a second waterfall. It does **not** by itself make the mechanism mixed or select the sizing-chain branch. Use it for class only when the **dual order would change** whether `Δ_price` or `Δ_size` is material, or which of the two is dominant (see D3 precedence).

3. **Financing is date-coupled.**  
   Frozen contract is Tier A `equal_max_loss`, `sizing_mode=conceptual`: shorts sized to `max_loss_per_share` (fill-dependent); **long budget = collected short credit** that day (fallback `$10,000` only if no usable shorts). Trade-level `Δ_size` on longs therefore includes **other names’** short fills. Side-split `Δ_size` is not an independent long-vs-short “execution tax.” That is why D2B **sizing chain** exists and why body/wing on longs without it can mislead.

4. **Trade-set is expected zero but still a named term.** D0 included-key parity is exact. `Δ_set` is still computed so a future mismatch cannot hide in residual.

5. **Not in D2:** fill ladders, break-even alpha, yearly **return**/CAR tables (D1 reporting gap stays in D1), alternative-structure P&L, signal IC, `SurfaceRunner`.

---

## Frozen D2A bridge

### Population, keys, joins

- Window: `PRIMARY_START`/`PRIMARY_END`; calendar = `date_status_*` via `filter_to_window`.
- Rows: `included_in_portfolio == True` on traded dates.
- Trade join: `(trade_date, ticker, direction)`. Inner join mid ∩ cross; D0 expects 9,212 keys, zero symmetric difference.
- Leg join: trade key + `(expiry_date, option_type, strike, leg_index)`.
- `Q = abs(quantity)` from `trade_log`. `p = pnl_per_share` from `trade_log` after leg-to-trade check.

### Sign convention (frozen)

All terms in **dollars of P&L**, same sign as `pnl_total` (profit > 0).

```
G      = P_cross − P_mid                          # accepted gap (expected < 0)
Δ_price = P(Q_mid, p_cross) − P(Q_mid, p_mid)     # direct concession at frozen mid size
Δ_size  = P(Q_cross, p_cross) − P(Q_mid, p_cross) # quantity/financing at frozen cross unit P&L
Δ_set   = P_cross_unmatched − P_mid_unmatched     # 0 if key sets match
R       = G − (Δ_price + Δ_size + Δ_set)
```

`P(Q, p) = Σ_i Q_i · p_i` over the **intersection** key set. Unmatched keys, if any, go only into `Δ_set`, not into `Δ_price`/`Δ_size`.

**Read `Δ_price < 0` as:** same contracts, same mid size, worse entry (cross).  
**Read `Δ_size < 0` as:** at cross unit economics, the cross **size vector** earned less than the mid size vector.

Do not flip to “shortfall = −G” except as a display alias labeled **exploratory**.

### Calculation order (frozen)

1. D0 `run_d0_validation`; if any gate fails → `D2_BLOCKED` (no economics).
2. Recompute D1 `total_pnl` mid; must match D1 / `by_fill.mid.primary` within D1 dollar tolerance. If D1 helper verdict is not `D1_CONTINUE_TO_D2` → `D2_BLOCKED` (do not interpret a stop-expression gap as a mechanism).
3. Leg-to-trade: for each included fill, `Σ pnl_total_leg = pnl_total` and `Σ pnl_per_unit = pnl_per_share` within `max($0.01, 1e-9·|ref|)` per trade; else `D2_BLOCKED`.
4. Official `P_mid`, `P_cross` from included `trade_log_*` (also match `by_fill.*.primary.long_short` sums).
5. `P(Q_mid, p_cross)` from joined trades; cross-check `Σ |Q_mid| · pnl_per_unit_cross` on matched legs.
6. Form `Δ_price`, `Δ_size`, `Δ_set`, `R`.
7. Repeat the **same** formulas by `direction` (long/short) and by calendar year of `trade_date` (dollar P&L only — **not** yearly CAR).
8. One order-sensitivity statistic `S_order` and the dual-order materiality/dominance check (inspection note 2). **Stop.**

### Tolerances

| Check | Pass |
|---|---|
| `P_mid` vs D1 / report mid | `abs(Δ) ≤ max($0.01, 1e-9 × abs(ref))` |
| `P_cross` vs report cross long+short `pnl_total` | same |
| Per-trade and aggregate leg-to-trade | same dollar rule |
| `Δ_set` | exactly 0 keys unmatched; dollar `Δ_set` within $0.01 of 0 |
| Residual `R` | `abs(R) ≤ max($0.01, 1e-9 × (|P_mid| + |P_cross|))` |

Any fail → `D2_BLOCKED`; do not run D2B or interpret shares.

### Required fields

| Artifact | Fields |
|---|---|
| `trade_log_{mid,cross}` | `trade_date`, `ticker`, `direction`, `included_in_portfolio`, `pnl_total`, `pnl_per_share`, `quantity`, `capital_at_risk_dollars`, `entry_cost_per_share`, `net_credit_per_share`, `max_loss_per_share`, `instrument_type`, `fill_label` |
| `leg_log_{mid,cross}` | D0 `LEG_LOG_COLUMNS` (especially `bid`, `ask`, `fill_price`, `entry_cash_per_unit`, `pnl_per_unit`, `pnl_total_leg`, `unit_quantity`, `portfolio_quantity`, `expiry_payoff_per_unit`, `leg_index`) |
| `date_status_*`, `date_summary_*` | window filter + integrity vs `Σ pnl_total` |
| `decision_report.json` | `by_fill.mid.primary` and `by_fill.cross.primary` long_short / date counts |

### Reuse (import; do not reimplement settle/sizing)

| Module | Use |
|---|---|
| `sprint007_artifact_validation.py` | `run_d0_validation`, keys, fill-price formula, pairing |
| `sprint007_d1_gross_margin.py` | window load pattern, included rows, D1 continue check |
| `surface_decision_report.py` | `filter_to_window`, `PRIMARY_*`, `compute_long_short_attribution` |
| `pipeline._apply_tier_a_sizing` / `_structure_premium_per_share` / `_at_risk_per_share` | **D2B sizing chain only** — diagnostic reconstruction from logged fields, not a second engine |

---

## D2A notebook (when implementation is authorized)

```
notebooks/sprint007/d2_shortfall_bridge.ipynb     ← committed clean
src/backtest/sprint007_d2_shortfall_bridge.py     ← D2A only until checkpoint
tests/unit/test_sprint007_d2_shortfall_bridge.py
```

| § | Content | Label |
|---|---|---|
| 0 | Question, sign convention, non-goals | — |
| 1 | D0 + D1 continue | accepted / blocker |
| 2 | Joins, leg-to-trade, `Δ_set` | accepted calculation |
| 3 | Aggregate waterfall `P_mid → Δ_price → Δ_size → Δ_set → R → P_cross` | accepted calculation |
| 4 | Long vs short component table | accepted calculation |
| 5 | Yearly **dollar** component table (not returns) | accepted calculation |
| 6 | One order-sensitivity statistic + dual-order materiality/dominance | exploratory description |
| 7 | Checkpoint: D2B branch + **provisional** D3 class; **no D2B tables**; **no final class** | D2 gate statistic |

**Visualizations (D2A, max 3):** (1) aggregate P&L waterfall; (2) grouped bars `Δ_price`/`Δ_size` by side; (3) yearly stacked `Δ_price`+`Δ_size`. No fill-alpha curves, no spread-cutoff charts.

**Evidence dir (outside repo):** `C:/MomentumCVG_env/runs/sprint007_d2a_<timestamp>/` with `d2a_bridge.json`, side/year CSVs, executed notebook + HTML, `execution_receipt.json`.

---

## Human review checkpoint

After D2A artifacts exist:

- Review residual, `Δ_set`, side and year tables, and `S_order` / dual-order check.
- Confirm the **single** D2B branch and the **provisional** D3 class from the frozen rules below (or `D2_BLOCKED`).
- **Do not** open D2B outputs, assign a final D3 class, start D3 design, or add filters until this checkpoint is accepted.

If D2A is `D2_BLOCKED`, stop. No D2B, no D3. The final class is `D2_BLOCKED`.

---

## D2B — exactly one follow-up (rules frozen now; run later)

Let `G = P_cross − P_mid`. Shares use `|Δ|/|G|` when `G ≠ 0`. **Material** = share ≥ **0.25**. **Dominant** = larger of `|Δ_price|`, `|Δ_size|` (ignore `Δ_set` if 0). Dual-order materiality uses `Δ_price_Paasche = P_cross − P(Q_cross, p_mid)` and `Δ_size_dual = P(Q_cross, p_mid) − P_mid`. `order_sensitive` is true **only if** that dual order would change whether `Δ_price` or `Δ_size` is material, or which is dominant. `S_order > 0.10` alone is not `order_sensitive` and does not select a D2B branch.

**Select one branch, first match wins (official Laspeyres terms):**

| Priority | Condition | D2B diagnostic | Why it can change the decision |
|---|---|---|---|
| 0 | Residual/`Δ_set`/prereq fail | **none** | Mechanism unknown |
| 1 | `Δ_size` material | **Sizing chain** | Long Q is financed by short credit; D3 cannot treat fill as a per-share tax only |
| 2 | Else `Δ_price` dominant **and** one side ≥ 70% of `Δ_price` **and** that side is short iron-fly | **Body/wing** | Refines D3 from side-conditioned to role-specific only if body or wings concentrate; a diffuse split does not erase the side finding |
| 3 | Else `Δ_price` dominant | **Ex-ante package tradability** | After D2A side split; may support selective friction vs diffuse quoted cost |

Long straddles: body/wing N/A; if longs dominate `Δ_price` with `Δ_size` not material, branch 3.

D2B must not produce a second residual target or a filter winner.

### Body/wing (branch 2)

Iron-fly legs: `unit_quantity` `+ − − +` → wing `{0,3}`, body `{1,2}`. For each short included trade, role `r ∈ {body, wing}`:

```
Δ_price_r = Σ_{ℓ ∈ r} −Q_mid × (entry_cash_per_unit_cross,ℓ − entry_cash_per_unit_mid,ℓ)
```

The minus sign matches the D2A identity: `p_cross − p_mid = −(entry_cash_cross − entry_cash_mid)` at frozen payoff. Require

```
Δ_price_body + Δ_price_wing = Δ_price_short
```

within `max($0.01, 1e-9 × |Δ_price_short|)` after summing over the short intersection keys. Fail → `D2_BLOCKED` (no final non-blocked class). This is **not** the P&L of a wingless book.

### Package tradability (branch 3)

Do **not** use `leg_spread_to_credit_ratio` (undefined / biased when package cashflow is a debit). Compute from **shared** mid/cross quotes (`bid`, `ask`, `unit_quantity` identical per D0):

```
package_half_spread          = 0.5 × Σ_ℓ |unit_quantity_ℓ| × (ask_ℓ − bid_ℓ)
midpoint_package_cashflow    = Σ_ℓ unit_quantity_ℓ × (bid_ℓ + 0.5 × (ask_ℓ − bid_ℓ))
package_width_to_cashflow    = package_half_spread / abs(midpoint_package_cashflow)
```

Skip the trade if `abs(midpoint_package_cashflow) = 0`. Form **terciles separately within** `(direction, instrument_type)` so long straddles and short iron flies are not pooled. Report `Δ_price` and mid `pnl_total` by tercile. **No best-cutoff search.** Selective friction needs concentration in the expensive tercile **and** remaining mid margin outside it.

---

## D3 handoff (deterministic; final class only after D2B)

D2A outputs a **D2B branch** and a **provisional** class. The **final** class is assigned only after D2B (or immediately as `D2_BLOCKED` if D2A fails). Choose **exactly one** class by **first true** in this precedence:

`D2_BLOCKED` → `D3_MIXED_MECHANISM` → `D3_SIZING_AWARE` → `D3_STRUCTURE_CONDITIONED` → `D3_EXECUTION_FOCUSED`

**Predicates (official Laspeyres unless noted):**

| Predicate | True when |
|---|---|
| `blocked` | D2A fail, or D2B fail (including body+wing identity) |
| `mixed` | (`Δ_price` material **and** `Δ_size` material) **or** `order_sensitive` |
| `sizing` | `Δ_size` material |
| `structure` | One side ≥ 70% of `Δ_price` (provisional and final). Higher-priority `mixed` / `sizing` still win. Branch 2 **refines** the diagnosis only: body or wings ≥ 70% of `Δ_price_short` → role-specific D3; otherwise D3 stays **side-conditioned**. A diffuse body/wing split must **not** set `structure = false` or drop to `D3_EXECUTION_FOCUSED`. Branch 3 does not by itself set `structure`. |

`S_order` magnitude is not a MIXED trigger.

| Class | First-true role | What D3 is allowed to design |
|---|---|---|
| `D2_BLOCKED` | `blocked` | **No D3.** Fix evidence/calculation. |
| `D3_MIXED_MECHANISM` | else `mixed` | Joint requirement; a single fill number is not identified. |
| `D3_SIZING_AWARE` | else `sizing` | Requirement **cannot** be a single per-share concession; Q and long budget move with fill. No live shadow. |
| `D3_STRUCTURE_CONDITIONED` | else `structure` | Break-even / headroom **by side**. If branch 2 found body or wings ≥ 70% of `Δ_price_short`, also by that **role**. Diffuse body/wing → side-conditioned only. Not a wingless counterfactual. |
| `D3_EXECUTION_FOCUSED` | else | One book-level **requirement**. Attainability unknown. |

Sprint attainability forbid-list still applies (§6.7). Selective-friction evidence from branch 3 informs D4 later; it does not add a sixth class.

---

## Acceptance evidence

**D2A (this design’s implementation scope until checkpoint):**

- [ ] D0 passed; D1 continue; official `P_mid`/`P_cross` reconciled
- [ ] Identity `G = Δ_price + Δ_size + Δ_set + R` within residual tolerance
- [ ] `Δ_set` = 0 (or explicit unmatched keys → blocked)
- [ ] Side and yearly **dollar** bridges
- [ ] One `S_order` companion; dual-order check recorded; not a second official gap
- [ ] One D2B recommendation + one **provisional** D3 class (not final)
- [ ] Focused tests: synthetic Q/p frames (price-only, size-only, interaction identity `|I| = |Δ_price − Δ_price_Paasche|`, unmatched keys, residual fail); no official-run economics in unit tests

**D2B (separate acceptance after checkpoint):** one diagnostic only; no threshold winner; body+wing identity if branch 2; **exactly one final** class from the precedence list.

---

## Stop conditions

| Trigger | Action |
|---|---|
| D0 fail, D1 not continue, join/leg mismatch, residual/`Δ_set` fail | `D2_BLOCKED` |
| Proposal adds fill ladders, yearly return tables, filters, alt structures, or `SurfaceRunner` | Rescope |
| D2B run before checkpoint, or more than one D2B branch | Invalidate |
| Language of recoverability / ORATS attainability | Forbidden |

---

## Non-goals

- D3 break-even algebra, fill ladders, shadow orders  
- Spread/liquidity cutoff search  
- Alternative payoff P&L  
- Yearly return/CAR table (D1 item)  
- Pure Momentum/CVG IC  
- Mutating Sprint 006 artifacts or contract  

---

## Inference boundary (required conclusion shape)

D2A must end with four sentences, filled from numbers:

1. **Supports:** which of {direct concession, sizing/financing, side concentration} the **dollar** identity supports, with shares of `G`.  
2. **Weakens:** which competing explanation the same identity weakens (e.g. trade-set if `Δ_set=0`; “pure execution tax” if `Δ_size` material).  
3. **Unknown:** package fill probability; whether required quality is attainable; counterfactual structures; any residual if blocked.  
4. **Provisional D3 class** and selected D2B branch. The **final** class is written only after D2B, as exactly one of `D2_BLOCKED` | `D3_MIXED_MECHANISM` | `D3_SIZING_AWARE` | `D3_STRUCTURE_CONDITIONED` | `D3_EXECUTION_FOCUSED`.

Midpoint remaining positive (D1) is **not** executable return. Body/wing dollars are **not** a wingless book. Side `Δ_size` is **not** a one-sided strategy test.

---

## Expected footprint (when authorized)

| Path | Purpose |
|---|---|
| `src/backtest/sprint007_d2_shortfall_bridge.py` | ~150–220 LOC D2A |
| `tests/unit/test_sprint007_d2_shortfall_bridge.py` | synthetic identity/order/tolerance |
| `notebooks/sprint007/d2_shortfall_bridge.ipynb` | D2A narrative + checkpoint |

D2B helper/tests only after checkpoint acceptance.
