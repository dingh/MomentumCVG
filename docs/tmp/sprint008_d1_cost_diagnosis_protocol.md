# Sprint 008 D1 — cost diagnosis and fixed U-exclusion follow-up

**Status:** `PROTOCOL FROZEN — FOLLOW-UP EXECUTED; EVIDENCE AWAITING REVIEW`  
**Drafted / frozen:** 2026-09-11 (before examining new results)  
**Evidence:** [`sprint008_d1_cost_diagnosis_evidence.md`](sprint008_d1_cost_diagnosis_evidence.md) — `C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260911T162501Z/`  
**Authorization:** User request; **limited amendment** authorizing exactly two fixed U-group exclusions (M1, M2). Does **not** reopen broader D2 threshold search.  
**Preserved:** Historical D1 labels and gate **`STOP_NO_THRESHOLDS`**. Prior within-date L−U HAC inference remains unchanged when reproduced.  
**Post-hoc disclosure:** Designed **after** seeing D1 and within-date follow-up results. Exploratory evidence, **not** independent confirmation.  
**Parents:** [`sprint008_d1_design.md`](sprint008_d1_design.md), [`sprint008_d1_within_date_followup_protocol.md`](sprint008_d1_within_date_followup_protocol.md)  
**Agenda:** [`docs/agenda/current_sprint.md`](../agenda/current_sprint.md)

---

## Research questions

1. Why do lower-cost (L) groups show favorable but uncertain net returns vs high-cost (U)?
2. Is within-date spread-drag dispersion large enough to matter economically?
3. Does a **fixed** exclusion of the existing U group improve whole-book P&L on original budget \(B\) while retaining valuable winners?

A negative or inconclusive answer is a valid completion.

---

## Bounded amendment (threshold gate)

| Allowed | Not allowed |
|---|---|
| Reproduce existing within-date L / middle / U membership | Search additional cutoffs or fractions |
| Diagnose cost dispersion and exact \(r=g-a\) decomposition | Combine M1+M2 filters |
| Compare baseline vs **exclude U** for M1 and for M2 separately | Promote whichever looks best |
| Four frozen HAC contrasts (below) | New significance gates on exploratory diagnostics |
| Development 2020–2023 only | Open evaluation-period outcomes; start next experiment automatically |

Historical D1 `STOP_NO_THRESHOLDS` and labels are **not** rewritten by this study.

---

## Population and economics (frozen)

| Pin | Value |
|---|---|
| Sample | Development `2020-01-01`–`2023-12-31`; eval closed |
| Universe | Frozen `42:8` long ATM straddles; weekly hold-to-expiry |
| Budget | \(B=\$10{,}000\) per entry date; original \(N\) unchanged |
| Stake | Equal \(B/N\); fractional research quantities; rejected capital stays cash (0 return) |
| Execution | Full cross \(h=1\); fees \(=0\) |
| Cost / qty / returns | \(C_i=M_i+H_i\); \(q_i=(B/N)/C_i\); \(r_i=(X_i-C_i)/C_i\); \(p_i=(B/N)\,r_i\) |
| Measurements | **M1** \(H/M\); **M2** \(H/S_0\) only |
| Crossed quotes | Policy `sprint008_d0_crossed_quote_v1` |

### Input enforcement

Reuse D0 `_required_input_ok` (geometry, strike/expiry agreement, finite quotes/mids, MH-vs-ask, payoff reconcile) on the analysis path. Midpoint/payoff checks alone are insufficient. Missing required outcomes for **any** executed baseline trade (including middle) → hard failure.

---

## Grouping (reuse prior entry-only rule)

For each measurement and date: ≥5 scored eligible names; score variation; sort `(score, ticker)`; \(k=\lfloor n/5\rfloor\); \(L\)=first \(k\); \(U\)=last \(k\); **middle** = remaining scored. Disclose exclusions, ties, actual fractions.

Reconcile membership and prior mean \(d_t=L-U\) net returns with `sprint008_d1_within_date_20260908T195615Z` before interpreting new results.

---

## Decomposition (exact identity)

\[
g_i=(X_i-M_i)/C_i,\quad a_i=H_i/C_i,\quad r_i=g_i-a_i
\]

Quantities remain full-cross \(q_i\); do not resize for midpoint.

Date-level (eligible split dates):

\[
d_{\mathrm{net}}=\overline{r}_L-\overline{r}_U,\quad
d_{\mathrm{gross}}=\overline{g}_L-\overline{g}_U,\quad
\mathrm{spread\_saving}=\overline{a}_U-\overline{a}_L
\]

Verify \(d_{\mathrm{net}}=d_{\mathrm{gross}}+\mathrm{spread\_saving}\).

Also report cost dispersion (\(H/C\), \(M/S_0\)), win rates (strict \(p_i>0\)), gross→net non-winner frequency, means of winners/losers, medians; label date-weighted vs pooled.

---

## Exactly two exclusion comparisons

For M1 and M2 separately: unfiltered baseline vs exclude that measurement’s **U** group. Keep other trades’ quantities. Dates without a valid split: no additional exclusion. Preserve full development entry-date calendar.

\[
R^{\mathrm{base}}_t=\sum_i p_i/B,\quad
R^{\mathrm{filt}}_t=\sum_{i\notin U}p_i/B,\quad
\mathrm{uplift}_t=R^{\mathrm{filt}}_t-R^{\mathrm{base}}_t=-\sum_{i\in U}p_i/B
\]

Report P&L on \(B\), uplift, coverage, cash/invested fractions, losses avoided, winning profits sacrificed, winning-profit retention, top-5/10 winner retention. Verify: total P&L improvement = losses avoided − winning profits sacrificed. Cumulative dollar P&L and peak-to-trough on fixed-\(B\) accounting (not compounded equity).

---

## Inference (four contrasts; family size 4)

One observation per entry date. Frozen contrasts:

1. M1 mean whole-book uplift  
2. M2 mean whole-book uplift  
3. M1 mean within-date net-win-rate difference \(L-U\)  
4. M2 mean within-date net-win-rate difference \(L-U\)

HAC: intercept-only, `maxlags=3`, Bartlett, small-sample correction, \(t_{T-1}\) (same helper as prior follow-up). Report ordinary 95% CIs; Bonferroni-adjusted \(p=\min(1,4\cdot p_{\mathrm{raw}})\) and **98.75%** intervals. Reproduce prior L−U mean-\(d_t\) inference unchanged (exploratory reference only).

No outlier removal to obtain significance. Half-period descriptive splits allowed.

---

## Proposed footprint

| Path | Role |
|---|---|
| `docs/tmp/sprint008_d1_cost_diagnosis_protocol.md` | This protocol |
| `src/backtest/sprint008_d1_cost_diagnosis.py` | Analysis |
| `scripts/run_sprint008_d1_cost_diagnosis.py` | Official runner |
| `notebooks/sprint008/d1_cost_diagnosis.ipynb` | Entrypoint |
| `tests/unit/test_sprint008_d1_cost_diagnosis.py` | Focused tests |
| `docs/tmp/sprint008_d1_cost_diagnosis_evidence.md` | Evidence review |

### Focused tests

- Exact \(d_{\mathrm{net}}=d_{\mathrm{gross}}+\mathrm{spread\_saving}\)
- Original \(N\) / cash accounting under U exclusion
- Winner-retention denominators (incl. NA when zero)
- Entry-only group selection (outcomes do not change membership); middle group
- Invalid input / missing middle outcome fails on real analysis path
- Bonferroni family size 4 / 98.75% CI level
- Evaluation rows excluded

---

## Interpretation deliverable

Answer the six decision questions in the user prompt; recommend **at most one** next experiment from the prescribed menu. Do not open eval data or start that experiment in this run.

**Fees remain unmodeled; quote-based results do not establish achievable fills or dependable income.**
