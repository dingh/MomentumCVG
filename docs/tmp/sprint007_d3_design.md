# Sprint 007 D3 — Required execution envelope

**Status:** `PROPOSED — AWAITING REVIEW`  
**Updated:** 2026-09-05  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md) §6.7, §10  
**D0:** `READY_WITH_NARROW_ENABLING_CHANGE` — `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/` (commit `8a59474`)  
**D1:** **accepted** — `D1_CONTINUE_TO_D2` — [`sprint007_d1_design.md`](sprint007_d1_design.md); [`sprint007_d1_evidence_review.md`](sprint007_d1_evidence_review.md)  
**D2:** **accepted** — final class `D3_EXECUTION_FOCUSED` — [`sprint007_d2_design.md`](sprint007_d2_design.md); D2A [`sprint007_d2a_evidence_review.md`](sprint007_d2a_evidence_review.md); D2B [`sprint007_d2b_evidence_review.md`](sprint007_d2b_evidence_review.md)  
**Authorization:** This design only. Do not implement, execute, or interpret D3 economics until the design is accepted. D4 is not authorized.

---

## Summary

| Item | D3 design decision |
|---|---|
| **Question** | What fraction of the quoted half-spread can the frozen Sprint 006 selected option book afford to pay while remaining profitable and retaining meaningful economic margin? |
| **Class** | `D3_EXECUTION_FOCUSED` — one **book-level** requirement. Side splits are descriptive only. |
| **Method** | Notebook-first. One read-only helper. One unit-test file. No `SurfaceRunner`. |
| **Fill coordinate** | \(h \in [0,1]\): \(h=0\) midpoint, \(h=1\) full cross. \(h\) is the fraction of the quoted half-spread paid (bought legs) or given up (sold legs). |
| **Population** | Frozen 9,212 included keys; frozen contracts, structures, quotes, and expiry payoffs. |
| **Two paths** | **F** = midpoint quantities held fixed. **R** = Tier-A quantities recomputed by trade date at each \(h\). |
| **Primary unit** | Dollar `pnl_total`. View A mean cycle CAR is the companion profitability check. |
| **Runtime** | Minutes on official artifacts. |

D3 states a **requirement** and **unmodeled-friction headroom**. It does not state that the required package execution is attainable from historical end-of-day quotes.

---

## Question D3 must answer

> What fraction of the quoted half-spread can the frozen Sprint 6 selected option book afford to pay while remaining profitable and retaining meaningful economic margin?

Accepted facts this design consumes (do not reopen):

| Source | Fact |
|---|---|
| D1 | Mid-primary \(P_{\mathrm{mid}} = +159{,}283.23\); View A mean cycle CAR \(= +2.396\%\); 9,212 trades; 341 traded dates; `D1_CONTINUE_TO_D2` |
| D2A | \(P_{\mathrm{cross}} = -163{,}272.84\); \(P(Q_{\mathrm{mid}}, p_{\mathrm{cross}}) = -184{,}084.69\); \(\Delta_{\mathrm{price}}\) dominant; \(\Delta_{\mathrm{size}}\) not material |
| D2B | Final class `D3_EXECUTION_FOCUSED`. Selective-friction evidence is reserved for D4; it does not add a D3 class or a filter. |

---

## Frozen decisions

### Population and inputs

- Official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`.
- Window: `PRIMARY_START` / `PRIMARY_END` (`2020-01-01` → `2026-07-10`).
- Keys: `included_in_portfolio == True` on traded dates; join `(trade_date, ticker, direction)`. D0: 9,212 keys, mid = cross.
- **Frozen:** selected keys, `unit_quantity`, strikes, expiries, `bid` / `ask`, and `expiry_payoff_per_unit`. Do not add, drop, or replace trades.
- If Path R would flip `included_in_portfolio` or change `n_traded_dates` from 341 → `D3_BLOCKED`.

### Fill coordinate \(h\)

Use the existing `FillAssumption` interpolation, not stored ORATS `mid`.

```
buy_alpha(h)  = 0.5 + 0.5 h
sell_alpha(h) = 0.5 + 0.5 h
```

Equivalent per-leg formula (matches D0 `expected_mid_fill_price` / `expected_cross_fill_price` at the endpoints):

```
mid_fill   = bid + 0.5 (ask − bid)
cross_fill = ask if unit_quantity > 0 else bid
fill(h)    = mid_fill + h (cross_fill − mid_fill)
```

\(h=0\) is midpoint. \(h=1\) is full cross. \(h\) is the fraction of the quoted half-spread paid or given up. It is a **mechanical interpolation**, not expected real execution.

Package entry cash at \(h\):

```
entry_cash_ℓ(h) = +fill_ℓ(h) · |unit_quantity_ℓ|   if unit_quantity_ℓ > 0
                = −fill_ℓ(h) · |unit_quantity_ℓ|   if unit_quantity_ℓ < 0
entry_cost(h)   = Σ_ℓ entry_cash_ℓ(h)
net_credit(h)   = −entry_cost(h)
```

This is `_build_strategy_entry_cost` under `FillAssumption(buy_alpha=0.5+0.5h, sell_alpha=0.5+0.5h)`. Expiry payoffs stay frozen, so per-share P&L is linear in \(h\):

```
p_i(h) = (1 − h) p_i,mid + h p_i,cross
```

### Two evaluation paths

| Path | Quantity | What it isolates |
|---|---|---|
| **F — fixed \(Q_{\mathrm{mid}}\)** | Official midpoint `abs(quantity)` | Direct price concession (D2 \(\Delta_{\mathrm{price}}\) / Laspeyres) |
| **R — resized** | Recompute Tier A **separately by `trade_date`** at that \(h\) | Official engine path, including fill-dependent short size and long financing |

**Path F P&L (closed form):**

```
P_F(h) = Σ_i Q_mid,i · p_i(h) = P_mid + h · Δ_price
```

At \(h=1\), \(P_F = P(Q_{\mathrm{mid}}, p_{\mathrm{cross}})\) (accepted D2 hybrid).

**Path R P&L:**

```
P_R(h) = Σ_i |Q_i(h)| · p_i(h)
```

\(Q(h)\) is **not** interpolated between official mid and cross quantities. Rebuild per-share sizing inputs at \(h\), then call the existing `_apply_tier_a_sizing` once per trade date.

Frozen Sprint 006 sizing constants (from `configs/sprint006_baseline_v1.json`; do not edit that file):

| Field | Value |
|---|---|
| `sizing_mode` | `conceptual` |
| `tier_a_mode` | `equal_max_loss` |
| `tier_a_short_budget` | `10000.0` |
| `tier_a_long_budget` | `10000.0` (fallback only) |

Rebuild at each \(h\), using official mid fields only as geometry:

```
# longs (straddle): max loss = premium paid
entry_cost_long(h)     = Σ entry_cash_ℓ(h)
max_loss_long(h)       = |entry_cost_long(h)|

# shorts (iron fly): max loss = wing_width − net_credit
wing_width             = max_loss_mid + net_credit_mid     # strike geometry; verify vs cross
net_credit_short(h)    = −entry_cost_short(h)
max_loss_short(h)      = max(wing_width − net_credit_short(h), 0)
```

Then size that date: shorts first from `max_loss_per_share`; long budget = collected short credit that day; fallback `$10,000` only if no usable shorts (`pipeline._apply_tier_a_sizing`).

Do not reimplement settle or sizing. Import `_apply_tier_a_sizing`, `_structure_premium_per_share`, and `_at_risk_per_share`.

### Capital and CAR

```
capital_i(h) = |Q_i| · at_risk_per_share_i(h)
```

`at_risk` is existing `_at_risk_per_share`: long premium; short `max_loss_per_share`. Path F still updates capital because premium and max-loss move with \(h\) even when \(Q\) is frozen. Path F CAR is therefore **not** linear.

Rebuild `date_summary` with `build_date_summary` from the \(h\)-path trade rows. View A mean cycle CAR = `compute_view_a(official_date_status, date_summary(h))["mean_cycle_car"]` — same D1 definition (mean of `cycle_return_on_capital_at_risk` on traded dates; no `valid_no_trade` zero-fill). Use official `date_status` so the calendar is frozen.

### Evaluation grid (bounded; not a fill ladder)

Path F dollar P&L uses the closed form. CAR (both paths) and Path R P&L / quantities / capital are nonlinear.

Freeze one visualization-and-scan grid:

```
H = {0.00, 0.05, 0.10, …, 1.00}     # 21 points
```

This is bounded sensitivity, not a strategy search. Intermediate \(h\) values are not candidate execution policies.

### First-adverse crossing

For a scalar series \(m(h)\) and target \(T\), with \(m(0) > T\):

```
h*(m, T) = min { h ∈ [0, 1] : m(h) ≤ T }
```

Scan \(H\) from \(h=0\). On the **first** interval \([h_i, h_{i+1}]\) where \(m(h_{i+1}) \le T\), linearly interpolate inside that interval only. If \(m\) later recovers above \(T\), ignore the recovery.

If \(m(0) \le T\) → `D3_BLOCKED` (conflicts with accepted D1 margin).  
If \(m(h) > T\) for all \(h \in H\) → that target does **not** constrain (`no_crossing`).  
Do not take the most favorable root. Do not fit a global polynomial.

Closed-form check (Path F P&L only, \(\Delta_{\mathrm{price}} < 0\)):

```
h_F(P ≤ α P_mid) = (1 − α) P_mid / (−Δ_price)     for α ∈ {1.00, 0.50, 0.25}
```

Interpolated Path F P&L crossings must match this within \(10^{-6}\) in \(h\).

### Thresholds (frozen before output)

Let \(M = P_{\mathrm{mid}}\) (accepted D1 dollar margin). **Economic margin** is dollar P&L. CAR is not used for the 50% / 25% buffers.

| ID | Condition | Meaning |
|---|---|---|
| `h_margin_50` | first \(P(h) \le 0.50\,M\) | half of midpoint dollar margin remains |
| `h_margin_25` | first \(P(h) \le 0.25\,M\) | one-quarter remains (declared meaningful-margin floor) |
| `h_pnl_0` | first \(P(h) \le 0\) | dollar break-even |
| `h_car_0` | first View A mean cycle CAR \(\le 0\) | companion profitability break-even |

Compute all four on Path F and Path R (eight book-level numbers). Side-level analogues are **not** requirement candidates.

The 50% and 25% levels are the unmodeled-friction buffers required by the working plan §6.7 / §10. They are frozen here; they must not be changed after output.

### Portfolio requirement

**Relevant** = book-level first-adverse crossings above. Side crossings are descriptive.

```
h_req = min { finite relevant h* }
```

That is the most restrictive (smallest) affordable half-spread fraction among the eight book-level crossings. Disclose which (path, metric) binds.

If Path F and Path R disagree, that is the disclosed sizing-feedback effect (D2 found \(\Delta_{\mathrm{size}}\) not material; D3 still evaluates both because official Sprint 006 economics are Path R). The stated requirement is still the single most restrictive book-level number.

### Headroom

On the **binding path**:

```
headroom_to_pnl_0 = h_pnl_0 − h_req     if h_pnl_0 exists, else no_crossing
headroom_to_car_0 = h_car_0 − h_req     if h_car_0 exists, else no_crossing
headroom_to_cross = 1 − h_req
```

Headroom \(\le 0\) means the requirement already uses the break-even (or worse): **no** remaining room for commissions, missed fills, timing, or adverse selection under the frozen buffers. That is a disclosed fact, not a viability slogan.

Commissions, fill probability, and other unmodeled frictions are **not** subtracted. They are why the 50% / 25% buffers exist.

---

## Endpoint reconciliation (required)

| Check | Must match | Tolerance |
|---|---|---|
| Path F \(h=0\) \(P\) and View A CAR | official mid-primary (D1) | D1: \(P\) `max($0.01, 1e-9·|ref|)`; CAR `1e-9` |
| Path F \(h=1\) \(P\) | D2 hybrid \(P(Q_{\mathrm{mid}}, p_{\mathrm{cross}})\) | D2 dollar rule |
| Path R \(h=0\) \(P\), CAR, and \(Q\) | official mid \(P\), CAR, and `quantity` | \(P\)/CAR as above; per-trade \|ΔQ\| \(\le \max(10^{-6}, 10^{-9}·|Q_{\mathrm{ref}}|)\) |
| Path R \(h=1\) \(P\), CAR, and \(Q\) | official cross-primary \(P\), CAR, and `quantity` | same |
| Path F closed-form vs grid \(P_F(h)\) | \(P_{\mathrm{mid}} + h\,\Delta_{\mathrm{price}}\) | \(P\) dollar rule at every \(h \in H\) |
| `wing_width` mid vs cross reconstruction | equal | \(P\) dollar rule |
| Key set | 9,212 at every \(h\); `n_traded_dates` = 341 | exact |

Any fail → `D3_BLOCKED`. Do not interpret crossings.

Path F \(h=1\) CAR is **not** official cross CAR (quantities differ). Do not reconcile it to `by_fill.cross`.

---

## Architecture: notebook-first + minimal helper

```
notebooks/sprint007/d3_execution_envelope.ipynb     ← committed clean
src/backtest/sprint007_d3_execution_envelope.py     ← load, interpolate, size-by-date, crossings
tests/unit/test_sprint007_d3_execution_envelope.py  ← synthetic only
```

**Reuse (import; do not reimplement settle / sizing / View A):**

| Module | Use |
|---|---|
| `sprint007_artifact_validation.py` | `run_d0_validation`, official paths, mid/cross fill formulas |
| `sprint007_d1_gross_margin.py` | D1 continue check; mid-primary load pattern |
| `sprint007_d2_shortfall_bridge.py` | D2 class / hybrid \(P(Q_{\mathrm{mid}}, p_{\mathrm{cross}})\) / \(\Delta_{\mathrm{price}}\) |
| `option_surface.FillAssumption` | interpolation identity (or the equivalent `fill(h)` formula above) |
| `pipeline._apply_tier_a_sizing`, `_structure_premium_per_share`, `_at_risk_per_share` | Path R only |
| `surface_metrics.build_date_summary` | cycle P&L / capital / CAR at each \(h\) |
| `surface_decision_report` | `filter_to_window`, `compute_view_a`, `PRIMARY_*` |

Do not add a second economic engine, a fill-ladder search, or a new `src/analysis/` package.

**Minimal helper surface:**

- `fill_price_at_h(bid, ask, unit_quantity, h) -> float`
- `package_entry_cost_at_h(leg_rows, h) -> float`
- `per_share_economics_at_h(...)` — `entry_cost`, `net_credit`, `max_loss`, `p(h)`
- `size_book_at_h(trades, h, config) -> DataFrame` — Path R; one `_apply_tier_a_sizing` call per date
- `evaluate_paths(bundle, H) -> PathCurves` — \(P\), CAR, \(\sum|Q|\), \(\sum\) capital, side \(P\) on F and R
- `first_adverse_crossing(h_grid, values, target) -> float | None`
- `reconcile_d3_endpoints(...) -> ReconciliationResult`
- `run_d3_analysis() -> D3Result` — prereqs, curves, eight crossings, `h_req`, headroom, verdict

Target footprint: ~180–260 LOC helper; ~100–160 LOC tests.

---

## Notebook sections (when implementation is authorized)

Committed notebook: `notebooks/sprint007/d3_execution_envelope.ipynb` (clean; no outputs in repo).

| § | Title | Label |
|---|---|---|
| 0 | Question, \(h\) definition, two paths, attainability boundary | — |
| 1 | D0 + D1 continue + D2 class `D3_EXECUTION_FOCUSED` | accepted / blocker |
| 2 | Endpoint reconciliation (three identities) | accepted calculation |
| 3 | Path F and Path R curves on \(H\) | accepted calculation |
| 4 | First-adverse crossing table (four metrics × two paths) | D3 gate statistic |
| 5 | Portfolio `h_req`, binding constraint, headroom | D3 gate statistic |
| 6 | Long vs short \(P(h)\) at \(\{0, h_{\mathrm{req}}, 1\}\) | exploratory description |
| 7 | Visualizations | exploratory description |
| 8 | Limits: requirement ≠ attainability; no D4 label | — |

**Visualizations (exactly 4):**

1. **Portfolio P&L vs \(h\)** — Path F and Path R; horizontal lines at \(M\), \(0.50M\), \(0.25M\), \(0\).
2. **View A mean cycle CAR vs \(h\)** — both paths; horizontal line at \(0\).
3. **\(\sum |Q|\) vs \(h\)** — both paths.
4. **\(\sum\) capital-at-risk vs \(h\)** — both paths.

Mark `h_req` on each chart. No spread-cutoff scatter, no filter sweep, no per-side requirement chart. Side dollars appear only as the §6 table.

---

## Required artifacts (evidence, outside repo)

Directory: `C:/MomentumCVG_env/runs/sprint007_d3_<timestamp>/`

| File | Content |
|---|---|
| `d3_envelope.json` | prereqs, endpoint Δ, eight crossings, `h_req`, binding (path, metric), headroom, verdict |
| `d3_curves.csv` | one row per \(h \in H\): \(P\), CAR, \(\sum|Q|\), capital, side \(P\) for F and R |
| `d3_crossings.csv` | metric, path, target, \(h^*\), `no_crossing` flag |
| `d3_side_snapshot.csv` | long/short \(P\) at \(h \in \{0, h_{\mathrm{req}}, 1\}\) |
| `d3_execution_envelope.executed.ipynb` | fresh-kernel execution |
| `d3_execution_envelope.html` | HTML export |
| `execution_receipt.json` | SHAs, repo HEAD, timestamps |

---

## Acceptance evidence (when D3 is later executed)

- [ ] D0 passed; D1 continue; D2 class is `D3_EXECUTION_FOCUSED`
- [ ] Three endpoint identities pass (Path F \(h=0\) mid; Path F \(h=1\) D2 hybrid; Path R \(h=1\) official cross), plus Path R \(h=0\) mid
- [ ] 9,212 keys and 341 traded dates at every evaluated \(h\)
- [ ] Four book-level first-adverse crossings computed on both paths; non-monotonic series use the first crossing from \(h=0\)
- [ ] `h_req` is the minimum finite relevant crossing; binding (path, metric) is named
- [ ] Headroom under the frozen 50% / 25% buffers is reported, or shown to be \(\le 0\)
- [ ] Requirement and attainability are separate sentences; forbid-list language is absent
- [ ] `tests/unit/test_sprint007_d3_execution_envelope.py` passes
- [ ] Clean committed notebook + executed evidence outside repo
- [ ] No filter, alternative structure, commission model, fill probability, or `SurfaceRunner`

---

## Tests (synthetic; no official-run economics)

| Case | Assert |
|---|---|
| `fill_price_at_h` | \(h=0\) mid; \(h=1\) buy=ask / sell=bid; \(h=0.5\) halfway |
| Package entry | `entry_cost(h)` linear; \(p(h)\) linear; expiry frozen |
| Path F identity | \(P_F(h) = P_{\mathrm{mid}} + h\,\Delta_{\mathrm{price}}\) |
| Closed-form vs interpolate | Path F P&L crossings match \((1-\alpha)P_{\mathrm{mid}}/(-\Delta_{\mathrm{price}})\) |
| First adverse, monotonic | decreasing series hits 0.50, 0.25, 0 in order |
| First adverse, non-monotonic | dips below 0 then recovers → first crossing, not the later root |
| No crossing | series stays above target → `None` |
| Path R sizing | one synthetic date; \(Q(h)\) matches `_apply_tier_a_sizing` on the same rows |
| Endpoint Q | synthetic mid/cross books; Path R \(h=0\)/\(h=1\) recover those quantities |
| Exclusion guard | sizing that would drop a name → blocked, not a smaller book |
| Prerequisite | D2 class ≠ `D3_EXECUTION_FOCUSED` → `D3_BLOCKED` |

---

## Stop conditions

| Trigger | Action |
|---|---|
| D0 fail, D1 not continue, D2 class not `D3_EXECUTION_FOCUSED` | `D3_BLOCKED` |
| Endpoint, wing-width, key-count, or inclusion fail | `D3_BLOCKED` |
| Path R would drop a frozen key or a traded date | `D3_BLOCKED` |
| Proposal adds filters, alt structures, commissions, fill odds, or `SurfaceRunner` | Rescope |
| Crossing rule changed after viewing output | Invalidate |
| Language of recoverability / ORATS attainability | Forbidden |

---

## Non-goals

- Package-order fill probability, shadow orders, or live / paper execution
- Commissions, borrow, or other unmodeled cost **models** (buffers only)
- Spread/liquidity cutoff search or a filtered book
- Alternative structures, wingless books, iron condor, or side-only strategies
- Signal-window search or `42:8` retuning
- Assigning a D4 label
- Mutating Sprint 006 artifacts or the frozen contract
- `SurfaceRunner` / `scripts/run_surface_search.py`

---

## Inference boundary (required conclusion shape)

D3 must end with four sentences, filled from numbers:

1. **Requirement:** the frozen book can afford to pay \(h_{\mathrm{req}}\) of the quoted half-spread before the most restrictive relevant threshold is hit, and which (path, metric) binds.
2. **Headroom:** remaining distance to dollar and CAR break-even under the frozen 50% / 25% buffers, or that none remains.
3. **Unknown:** whether any package order would fill at that \(h\); commissions, missed fills, timing, and adverse selection; counterfactual structures. Historical quotes do not validate complex-order execution.
4. **Not claimed:** that midpoint is attainable, that \(h_{\mathrm{req}}\) is a live limit price, or that a filter / side / structure change would preserve the midpoint book.

If \(0 < h_{\mathrm{req}} < 1\), the **shape** required by working-plan §6.7 is “requirement strictly between mid and full cross.” D3 records that shape. D4, not D3, chooses among `EXECUTION_CALIBRATION_REQUIRED`, `SELECTIVE_FRICTION_HYPOTHESIS`, and the other sprint outcomes.

Path F remaining positive is not executable return. Path R is not a new official Sprint 006 result. Side dollars are not a long-only or short-only test.

---

## Expected footprint (when implementation is authorized)

| Path | Purpose |
|---|---|
| `src/backtest/sprint007_d3_execution_envelope.py` | ~180–260 LOC |
| `tests/unit/test_sprint007_d3_execution_envelope.py` | ~100–160 LOC |
| `notebooks/sprint007/d3_execution_envelope.ipynb` | Envelope narrative (committed clean) |
