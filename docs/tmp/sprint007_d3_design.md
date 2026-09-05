# Sprint 007 D3 — Required execution envelope

**Status:** `ACCEPTED`  
**Updated:** 2026-09-05  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)  
**Working plan:** [`docs/agenda/sprint7_shortfall_plan.md`](../agenda/sprint7_shortfall_plan.md) §6.7, §10  
**D0:** `READY_WITH_NARROW_ENABLING_CHANGE` — `C:/MomentumCVG_env/runs/sprint007_d0_20260830T001015Z/` (commit `8a59474`)  
**D1:** **accepted** — `D1_CONTINUE_TO_D2` — [`sprint007_d1_design.md`](sprint007_d1_design.md); [`sprint007_d1_evidence_review.md`](sprint007_d1_evidence_review.md)  
**D2:** **accepted** — final class `D3_EXECUTION_FOCUSED` — [`sprint007_d2_design.md`](sprint007_d2_design.md); D2A [`sprint007_d2a_evidence_review.md`](sprint007_d2a_evidence_review.md); D2B [`sprint007_d2b_evidence_review.md`](sprint007_d2b_evidence_review.md)  
**Authorization:** D3 **implementation** is authorized. Do not execute against official Sprint 006 artifacts, generate D3 evidence, or start D4 until that execution is separately authorized.

---

## Summary

| Item | D3 design decision |
|---|---|
| **Question** | What fraction of the quoted half-spread can the frozen Sprint 006 selected option book afford to pay while remaining profitable and retaining meaningful economic margin? |
| **Class** | `D3_EXECUTION_FOCUSED` — one **book-level Path R envelope**. Side splits are descriptive only. |
| **Answer shape** | A **range**, not one chosen fill: below \(h_{R,50}\) retains 50% of midpoint P&L; below \(h_{R,25}\) retains 25%; below \(h_{R,P0}\) remains dollar-profitable. CAR break-even is companion only. |
| **Primary path** | **R** (Tier-A quantities recomputed by trade date at each \(h\)). |
| **Diagnostic path** | **F** (midpoint quantities held fixed). Shows the effect of resizing. Must not bind the Path R envelope. |
| **Method** | Notebook-first. One read-only helper. One unit-test file. No `SurfaceRunner`. |
| **Fill coordinate** | \(h \in [0,1]\): \(h=0\) midpoint, \(h=1\) full cross. \(h\) is the fraction of the quoted half-spread paid (bought legs) or given up (sold legs). |
| **Population** | Frozen 9,212 included keys; frozen contracts, structures, quotes, and expiry payoffs. |
| **Primary unit** | Dollar `pnl_total`. View A mean cycle CAR is the companion profitability check. |
| **Runtime** | Minutes on official artifacts. |

D3 states a Path R **envelope** (50% / 25% / dollar break-even, plus companion CAR) and **unmodeled-friction headroom** between those Path R marks. Distances from break-even to full cross are reported separately and are not headroom. It does not collapse those marks into a single executable fill, and it does not state that the required package execution is attainable from historical end-of-day quotes.

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

| Path | Quantity | Role |
|---|---|---|
| **R — resized** | Recompute Tier A **separately by `trade_date`** at that \(h\) | **Primary.** Official engine path, including fill-dependent short size and long financing. The D3 envelope is Path R only. |
| **F — fixed \(Q_{\mathrm{mid}}\)** | Official midpoint `abs(quantity)` | **Diagnostic.** Isolates direct price concession (D2 \(\Delta_{\mathrm{price}}\) / Laspeyres) so resizing can be compared. Must not bind Path R. |

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

### Visualization grid (not the Path R root)

Path F dollar P&L uses the closed form. CAR (both paths) and Path R P&L / quantities / capital are nonlinear.

Freeze one **visualization** grid for the four charts and `d3_curves.csv`:

```
H_vis = {0.00, 0.05, 0.10, …, 1.00}     # 21 points
```

`H_vis` is bounded sensitivity for plots, not a strategy search, and **not** the authoritative Path R root. Do not take a linear interpolate across a 5% `H_vis` interval as \(h_{R,\cdot}\). Intermediate \(h\) values are not candidate execution policies.

### First-adverse crossing (authoritative root)

For a scalar \(m(h)\) and target \(T\), with \(m(0) > T\):

```
h*(m, T) = min { h ∈ [0, 1] : m(h) ≤ T }
```

If \(m\) later recovers above \(T\), ignore the recovery. Do not take the most favorable root. Do not fit a global polynomial.

If \(m(0) \le T\) → `D3_BLOCKED` (conflicts with accepted D1 margin).  
If \(m(h) > T\) for all evaluated detection points and the refined search never brackets → that target does **not** constrain (`no_crossing`).

**Path F P&L** uses the closed form (no grid root). With \(\Delta_{\mathrm{price}} < 0\) and \(M = P_{\mathrm{mid}}\):

```
h_F(P ≤ α M) = (1 − α) M / (−Δ_price)     for α ∈ {0.50, 0.25, 0.00}
```

That is \(h_{F,50}\), \(h_{F,25}\), and \(h_{F,P0}\). Closed-form Path F P&L crossings must match this within \(10^{-6}\) in \(h\).

**Path R P&L, Path R CAR, and Path F CAR** are nonlinear. Authoritative \(h^*\) is **bracket then refine**, not 5% interpolation.

**1. Detection (cannot miss a first crossing that exists on the detection set)**

```
H_det = {0.00, 0.01, 0.02, …, 1.00}     # 101 points, step 0.01
```

Evaluate \(m\) on `H_det` from \(h=0\). For every adjacent pair with **both** endpoints \(> T\), also evaluate the midpoint (the missed-dip check). Scan `H_det` ∪ those midpoints in increasing \(h\).

The **first bracket** is the leftmost pair \((h_L, h_R)\) in that ordered set such that \(m(h_L) > T\) and \(m(h_R) \le T\).

Also record a **monotonicity diagnostic** on `H_det`: `monotonic_nonincreasing` is true iff \(m(h_{i+1}) \le m(h_i) + \tau_m\) for every adjacent pair, where \(\tau_m\) is the metric tolerance below. Non-monotonicity does **not** change the first-adverse rule; it is disclosed so a later recovery cannot be mistaken for the requirement.

The 0.01 detection step plus the midpoint check is the frozen guard against a crossing that `H_vis` (step 0.05) would skip. A dip that stays entirely between a detection point and its midpoint (width 0.005) and still hits \(T\) is accepted residual risk; it is not repaired by denser ad-hoc grids after output.

**2. Refinement (declared tolerance)**

Bisection on the first bracket \([h_L, h_R]\):

```
while h_R − h_L > H_TOL:
    h_mid = 0.5 (h_L + h_R)
    if m(h_mid) ≤ T:
        h_R = h_mid
    else:
        h_L = h_mid
return h_R
```

\(h_R\) is the first evaluated point known to satisfy \(m \le T\), within `H_TOL`.

| Tolerance | Value |
|---|---|
| `H_TOL` | \(10^{-4}\) in \(h\) |
| \(\tau_P\) (P&L) | `max($0.01, 1e-9 · |M|)` |
| \(\tau_{\mathrm{CAR}}\) | \(10^{-9}\) |

Bisection stops on `H_TOL` only. \(\tau_P\) / \(\tau_{\mathrm{CAR}}\) are endpoint-reconciliation and monotonicity-slack tolerances, not a second root rule.

**3. Path F must not enter the Path R envelope**

Path F thresholds are computed with the same first-adverse definition (closed form for P&L; bracket-and-refine for CAR). They are diagnostics only. They must not replace, minimize with, or bind \(h_{R,50}\), \(h_{R,25}\), \(h_{R,P0}\), or \(h_{R,\mathrm{CAR}0}\).

### Thresholds (frozen before output)

Let \(M = P_{\mathrm{mid}}\) (accepted D1 dollar margin). **Economic margin** is dollar P&L. CAR is not used for the 50% / 25% buffers.

**Primary (Path R) — the D3 envelope:**

| ID | Condition | Meaning |
|---|---|---|
| \(h_{R,50}\) | first \(P_R(h) \le 0.50\,M\) | half of midpoint dollar margin remains |
| \(h_{R,25}\) | first \(P_R(h) \le 0.25\,M\) | one-quarter remains (declared meaningful-margin floor) |
| \(h_{R,P0}\) | first \(P_R(h) \le 0\) | dollar break-even |
| \(h_{R,\mathrm{CAR}0}\) | first Path R View A mean cycle CAR \(\le 0\) | companion profitability break-even |

**Diagnostic (Path F) — resize comparison only:**

| ID | Condition |
|---|---|
| \(h_{F,50}\), \(h_{F,25}\), \(h_{F,P0}\) | same dollar targets on \(P_F\) (closed form) |
| \(h_{F,\mathrm{CAR}0}\) | first Path F View A mean cycle CAR \(\le 0\) (bracket-and-refine) |

Side-level analogues are **not** envelope candidates.

The 50% and 25% levels are the unmodeled-friction buffers required by the working plan §6.7 / §10. They are frozen here; they must not be changed after output.

There is **no** single `h_req` and **no** `min` across the eight crossings. A 50%-margin mark and a fixed-quantity Path F mark must not be selected as “the” executable fill.

### Path R answer (a range, not one fill)

State the primary result as:

> On the resized book, \(h < h_{R,50}\) retains at least 50% of midpoint dollar P&L; \(h < h_{R,25}\) retains at least 25%; \(h < h_{R,P0}\) remains dollar-profitable. Path R CAR first reaches zero at \(h_{R,\mathrm{CAR}0}\) (companion; not a dollar-margin step).

Report Path F as a comparison table and as \(h_{R,\cdot} - h_{F,\cdot}\) (positive means resizing **relaxes** that threshold relative to frozen \(Q_{\mathrm{mid}}\)). Path F remaining positive is not executable return.

### Headroom and distances (Path R only)

**Headroom** is only the gap between consecutive Path R dollar-margin marks:

```
headroom_50_to_25 = h_{R,25} − h_{R,50}      if both exist, else no_crossing
headroom_25_to_P0 = h_{R,P0} − h_{R,25}      if both exist, else no_crossing
```

A **positive** value means additional execution-cost capacity remains between those marks. **Zero or negative** means no additional room. Zero or negative headroom, or a later mark occurring **at or before** the prior mark, means **no** remaining room for commissions, missed fills, timing, or adverse selection under that frozen buffer. That is a disclosed fact, not a viability slogan.

A missing later mark (`no_crossing`) means that buffer or break-even is never hit on \([0,1]\); headroom to it is not a finite \(h\) gap.

**Distances to full cross** measure how far \(h=1\) lies beyond break-even. They are **not** remaining headroom for commissions or other unmodeled costs:

```
distance_P0_to_cross   = 1 − h_{R,P0}      if h_{R,P0} exists, else no_crossing
distance_CAR0_to_cross = 1 − h_{R,CAR0}    if h_{R,CAR0} exists, else no_crossing
```

Do not name these `headroom_*`. A large distance to cross can coexist with zero headroom between 50% and 25%, or between 25% and dollar break-even.

Commissions, fill probability, and other unmodeled frictions are **not** subtracted. They are why the 50% / 25% buffers exist.

---

## Endpoint reconciliation (required)

| Check | Must match | Tolerance |
|---|---|---|
| Path F \(h=0\) \(P\) and View A CAR | official mid-primary (D1) | D1: \(P\) `max($0.01, 1e-9·|ref|)`; CAR `1e-9` |
| Path F \(h=1\) \(P\) | D2 hybrid \(P(Q_{\mathrm{mid}}, p_{\mathrm{cross}})\) | D2 dollar rule |
| Path R \(h=0\) \(P\), CAR, and \(Q\) | official mid \(P\), CAR, and `quantity` | \(P\)/CAR as above; per-trade \|ΔQ\| \(\le \max(10^{-6}, 10^{-9}·|Q_{\mathrm{ref}}|)\) |
| Path R \(h=1\) \(P\), CAR, and \(Q\) | official cross-primary \(P\), CAR, and `quantity` | same |
| Path F closed-form vs grid \(P_F(h)\) | \(P_{\mathrm{mid}} + h\,\Delta_{\mathrm{price}}\) | \(P\) dollar rule at every \(h \in H_{\mathrm{vis}}\) |
| Path F P&L roots vs closed form | \(\alpha \in \{0.50, 0.25, 0.00\}\) | \(10^{-6}\) in \(h\) |
| `wing_width` mid vs cross reconstruction | equal | \(P\) dollar rule |
| Key set | 9,212 at every evaluated \(h\); `n_traded_dates` = 341 | exact |

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
- `evaluate_paths(bundle, H) -> PathCurves` — \(P\), CAR, \(\sum|Q|\), \(\sum\) capital, side \(P\) on F and R (`H_vis` for curves; `H_det` inside the root finder)
- `path_f_pnl_crossing(alpha, P_mid, delta_price) -> float` — closed form for \(\alpha \in \{0.50, 0.25, 0.00\}\)
- `first_adverse_crossing(eval_fn, target) -> float | None` — `H_det` + midpoint miss-check + bisection to `H_TOL`; not `H_vis` interpolation
- `reconcile_d3_endpoints(...) -> ReconciliationResult`
- `run_d3_analysis() -> D3Result` — prereqs, curves, Path R envelope \(\{h_{R,50}, h_{R,25}, h_{R,P0}, h_{R,\mathrm{CAR}0}\}\), Path F diagnostics, Path R headroom (`headroom_50_to_25`, `headroom_25_to_P0`), distances to cross (`distance_P0_to_cross`, `distance_CAR0_to_cross`), monotonicity flags, verdict. **No `h_req`.**

Target footprint: ~200–280 LOC helper; ~120–180 LOC tests.

---

## Notebook sections (when implementation is authorized)

Committed notebook: `notebooks/sprint007/d3_execution_envelope.ipynb` (clean; no outputs in repo).

| § | Title | Label |
|---|---|---|
| 0 | Question, \(h\) definition, Path R primary / Path F diagnostic, attainability boundary | — |
| 1 | D0 + D1 continue + D2 class `D3_EXECUTION_FOCUSED` | accepted / blocker |
| 2 | Endpoint reconciliation (three identities) | accepted calculation |
| 3 | Path R and Path F curves on \(H_{\mathrm{vis}}\) | accepted calculation |
| 4 | Path R envelope \(\{h_{R,50}, h_{R,25}, h_{R,P0}\}\) and companion \(h_{R,\mathrm{CAR}0}\) | D3 gate statistic |
| 5 | Path F diagnostics and \(h_R - h_F\) resize gaps (F does not bind) | exploratory description |
| 6 | Path R headroom (`50→25`, `25→P0`) and distances to cross (`P0→1`, `CAR0→1`) | D3 gate statistic |
| 7 | Long vs short \(P(h)\) at \(\{0, h_{R,50}, h_{R,25}, h_{R,P0}, 1\}\) | exploratory description |
| 8 | Visualizations | exploratory description |
| 9 | Limits: envelope ≠ one fill; requirement ≠ attainability; no D4 label | — |

**Visualizations (exactly 4):**

1. **Portfolio P&L vs \(h\)** — Path R solid, Path F dashed; horizontal lines at \(M\), \(0.50M\), \(0.25M\), \(0\).
2. **View A mean cycle CAR vs \(h\)** — Path R solid, Path F dashed; horizontal line at \(0\).
3. **\(\sum |Q|\) vs \(h\)** — both paths.
4. **\(\sum\) capital-at-risk vs \(h\)** — both paths.

Mark \(h_{R,50}\), \(h_{R,25}\), and \(h_{R,P0}\) on the P&L chart; mark \(h_{R,\mathrm{CAR}0}\) on the CAR chart. Do not draw a single `h_req`. No spread-cutoff scatter, no filter sweep, no per-side requirement chart. Side dollars appear only as the §7 table.

---

## Required artifacts (evidence, outside repo)

Directory: `C:/MomentumCVG_env/runs/sprint007_d3_<timestamp>/`

| File | Content |
|---|---|
| `d3_envelope.json` | prereqs, endpoint Δ, Path R \(\{h_{R,50}, h_{R,25}, h_{R,P0}, h_{R,\mathrm{CAR}0}\}\), Path F diagnostics, \(h_R-h_F\) gaps, `headroom_50_to_25`, `headroom_25_to_P0`, `distance_P0_to_cross`, `distance_CAR0_to_cross`, monotonicity flags, root-finder tolerances, verdict. **No `h_req`.** |
| `d3_curves.csv` | one row per \(h \in H_{\mathrm{vis}}\): \(P\), CAR, \(\sum|Q|\), capital, side \(P\) for R and F |
| `d3_crossings.csv` | path (`R` primary / `F` diagnostic), metric, target, \(h^*\), `no_crossing`, method (`closed_form` \| `bracket_bisection`) |
| `d3_side_snapshot.csv` | long/short \(P\) at \(h \in \{0, h_{R,50}, h_{R,25}, h_{R,P0}, 1\}\) |
| `d3_execution_envelope.executed.ipynb` | fresh-kernel execution |
| `d3_execution_envelope.html` | HTML export |
| `execution_receipt.json` | SHAs, repo HEAD, timestamps |

---

## Acceptance evidence (when D3 is later executed)

- [ ] D0 passed; D1 continue; D2 class is `D3_EXECUTION_FOCUSED`
- [ ] Three endpoint identities pass (Path F \(h=0\) mid; Path F \(h=1\) D2 hybrid; Path R \(h=1\) official cross), plus Path R \(h=0\) mid
- [ ] 9,212 keys and 341 traded dates at every evaluated \(h\)
- [ ] Path R envelope reports \(h_{R,50}\), \(h_{R,25}\), \(h_{R,P0}\), and companion \(h_{R,\mathrm{CAR}0}\) as a range, not one `h_req`
- [ ] Path F thresholds are present only as diagnostics; they do not bind Path R
- [ ] Nonlinear roots use `H_det` + midpoint miss-check + bisection to `H_TOL`; `H_vis` interpolation is not the root
- [ ] Non-monotonic series use the first crossing from \(h=0\); monotonicity is disclosed
- [ ] Path F P&L roots match the closed form for \(\alpha \in \{0.50, 0.25, 0.00\}\)
- [ ] Path R `headroom_50_to_25` and `headroom_25_to_P0` are reported (positive = remaining execution-cost capacity; zero/negative = none), or shown `no_crossing`
- [ ] `distance_P0_to_cross` and `distance_CAR0_to_cross` are reported as distance beyond break-even, not as headroom
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
| Path F closed form | roots match \((1-\alpha)P_{\mathrm{mid}}/(-\Delta_{\mathrm{price}})\) for \(\alpha \in \{0.50, 0.25, 0.00\}\) only |
| First adverse, monotonic | decreasing series hits 0.50, 0.25, 0 in that order |
| First adverse, non-monotonic | dips below 0 then recovers → first crossing, not the later root |
| Missed-dip guard | series that crosses \(T\) only near the midpoint of a 0.01 interval is still bracketed |
| Bisection tolerance | returned \(h^*\) is within `H_TOL` of the first \(m \le T\) point on the refined bracket |
| `H_vis` is not the root | a crossing between 0.05 grid nodes is **not** reported as the 5% interpolant |
| No crossing | series stays above target → `None` |
| Path R sizing | one synthetic date; \(Q(h)\) matches `_apply_tier_a_sizing` on the same rows |
| Endpoint Q | synthetic mid/cross books; Path R \(h=0\)/\(h=1\) recover those quantities |
| Exclusion guard | sizing that would drop a name → blocked, not a smaller book |
| Envelope schema | result has Path R range fields and **no** `h_req`; Path F fields are labeled diagnostic |
| Path F does not bind | a synthetic case with \(h_{F,50} < h_{R,50}\) still reports the Path R envelope from Path R only |
| Prerequisite | D2 class ≠ `D3_EXECUTION_FOCUSED` → `D3_BLOCKED` |

---

## Stop conditions

| Trigger | Action |
|---|---|
| D0 fail, D1 not continue, D2 class not `D3_EXECUTION_FOCUSED` | `D3_BLOCKED` |
| Endpoint, wing-width, key-count, or inclusion fail | `D3_BLOCKED` |
| Path R would drop a frozen key or a traded date | `D3_BLOCKED` |
| Proposal adds filters, alt structures, commissions, fill odds, or `SurfaceRunner` | Rescope |
| Proposal restores `h_req = min(…)` or lets Path F bind Path R | Reject |
| Crossing or root-finder rule changed after viewing output | Invalidate |
| Language of recoverability / ORATS attainability | Forbidden |

---

## Non-goals

- Package-order fill probability, shadow orders, or live / paper execution
- Commissions, borrow, or other unmodeled cost **models** (buffers only)
- Spread/liquidity cutoff search or a filtered book
- Alternative structures, wingless books, iron condor, or side-only strategies
- Signal-window search or `42:8` retuning
- Assigning a D4 label
- Selecting one \(h\) as *the* executable fill
- Mutating Sprint 006 artifacts or the frozen contract
- `SurfaceRunner` / `scripts/run_surface_search.py`

---

## Inference boundary (required conclusion shape)

D3 must end with four sentences, filled from **Path R** numbers:

1. **Envelope:** on the resized book, \(h < h_{R,50}\) retains at least 50% of midpoint dollar P&L; \(h < h_{R,25}\) retains at least 25%; \(h < h_{R,P0}\) remains dollar-profitable. Path R CAR first reaches zero at \(h_{R,\mathrm{CAR}0}\) (companion).
2. **Headroom:** Path R `headroom_50_to_25` and `headroom_25_to_P0` (positive = additional execution-cost capacity; zero/negative = none), or `no_crossing` where a later mark is never hit. Separately, `distance_P0_to_cross` and `distance_CAR0_to_cross` state how far full cross lies beyond break-even; those distances are not headroom.
3. **Unknown:** whether any package order would fill inside that envelope; commissions, missed fills, timing, and adverse selection; counterfactual structures. Historical quotes do not validate complex-order execution.
4. **Not claimed:** that midpoint is attainable; that any one of \(h_{R,50}\), \(h_{R,25}\), or \(h_{R,P0}\) is a live limit price; that Path F is the executable book; or that a filter / side / structure change would preserve the midpoint book.

If \(0 < h_{R,P0} < 1\) (or the 50% / 25% marks lie strictly inside \((0,1)\)), the **shape** required by working-plan §6.7 is “requirement strictly between mid and full cross.” D3 records that shape as a range. D4, not D3, chooses among `EXECUTION_CALIBRATION_REQUIRED`, `SELECTIVE_FRICTION_HYPOTHESIS`, and the other sprint outcomes.

Path F remaining positive is not executable return. Path R is not a new official Sprint 006 result. Side dollars are not a long-only or short-only test.

---

## Expected footprint (when implementation is authorized)

| Path | Purpose |
|---|---|
| `src/backtest/sprint007_d3_execution_envelope.py` | ~200–280 LOC |
| `tests/unit/test_sprint007_d3_execution_envelope.py` | ~120–180 LOC |
| `notebooks/sprint007/d3_execution_envelope.ipynb` | Envelope narrative (committed clean) |
