# Sprint 008 — Long-straddle measurement filter plan

**Status:** `ACCEPTED`  
**Accepted:** 2026-09-06  
**Updated:** 2026-09-12
**Agenda:** [`docs/agenda/current_sprint.md`](current_sprint.md) — Sprint 008 **Build/Audit**; **D0 accepted**; **D1 closed**; **D2 design draft awaiting review** (`STOP_NO_THRESHOLDS` preserved)
**Prior closeouts:** [`docs/sprint_memos/007_closeout.md`](../sprint_memos/007_closeout.md), [`docs/sprint_memos/006_closeout.md`](../sprint_memos/006_closeout.md)  
**Frozen Sprint 006 contract:** [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json) — immutable; not edited by this sprint  
**D2B H/M precedent:** [`docs/tmp/sprint007_d2b_evidence_review.md`](../tmp/sprint007_d2b_evidence_review.md); `src/backtest/sprint007_d2b_package_tradability.py`  
**D0 design:** [`docs/tmp/sprint008_d0_design.md`](../tmp/sprint008_d0_design.md) — `ACCEPTED`  
**D0 evidence:** [`docs/tmp/sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md) — **accepted**; `C:/MomentumCVG_env/runs/sprint008_d0_20260907T204449Z/` (SHA `af24f50`; policy `sprint008_d0_crossed_quote_v1`)  
**D1 design:** [`docs/tmp/sprint008_d1_design.md`](../tmp/sprint008_d1_design.md) — `ACCEPTED`  
**D1 evidence:** [`docs/tmp/sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md) — **reviewed / D1 closed** 2026-09-12; `C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/` (SHA `72629a0`, clean tree)
**D1 follow-ups (reviewed):** within-date [`sprint008_d1_within_date_followup_evidence.md`](../tmp/sprint008_d1_within_date_followup_evidence.md); corrected cost diagnosis [`sprint008_d1_cost_diagnosis_evidence.md`](../tmp/sprint008_d1_cost_diagnosis_evidence.md) (`870d4b7` accepted)
**D2 design (draft):** [`docs/tmp/sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) — `DRAFT — AWAITING REVIEW` (revised in place after `269fde0`). Not accepted. Does not yet replace §10 D2.
**Canonical path:** `docs/agenda/sprint8_long_filter_plan.md` — do not duplicate under `docs/tmp/`.  
**Purpose:** Accepted sprint-level research protocol for a long-side-only measurement and conditional-threshold study. This plan freezes questions, gates, inference boundaries, and deliverable sequence. It deliberately defers deliverable-specific formulas, notebooks, schemas, and code footprints until each deliverable is designed and accepted.

---

## 1. Human-readable summary

| Item | Sprint 008 decision |
|---|---|
| **Central question** | Among current `42:8` long-straddle candidates, do entry-time measurements reliably distinguish net profitability per trade, and, if so, can a simple threshold improve the long book while retaining valuable winners? |
| **Motivation** | Separately motivated research experiment. Sprint 007 diagnosed book-level execution requirements and disclosed expensive-package concentration as a **secondary** finding only. This sprint does **not** reopen or revise Sprint 006/007 conclusions. |
| **Side** | Long ATM straddles only |
| **Frozen selection** | `42:8` signal, CVG-within-side filter, liquidity/structure rules, `max_names_per_side=25`, weekly timing, hold-to-expiry |
| **New baseline** | Independent equal-dollar long research book: fixed budget \(B\), stake \(B/N\) per eligible name sized on scenario all-in entry cost; unused cash stays cash |
| **Method** | Define measurements → validate measurement–profitability relationship (required gate) → only then test simple thresholds |
| **Not the goal** | Force profitability; rescue Sprint 006; retune signal windows; redesign short structures; claim fill attainability |
| **Outcomes allowed** | Supported / unsupported / inconclusive measurement; effective / ineffective threshold — all valid completions |
| **Approval boundary** | Plan **accepted**. D0 **accepted**. D1 **closed** (`STOP_NO_THRESHOLDS` preserved). D2 amendment **drafted, awaiting review**, not accepted. D3 pending. |

---

## 2. Why this document exists

`docs/agenda/current_sprint.md` is the stable Sprint 008 contract (intent, mode, DoD, authorization).

This working plan is the accepted detailed scope for the long-filter experiment: candidate population, equal-dollar baseline, measurement catalog, validation gate, chronological firewall, conditional threshold study, and completion criteria. Deliverable-specific designs are written only when that deliverable begins.

---

## 3. Relationship to Sprint 006 and 007

### Preserved accepted results

| Item | Status in Sprint 008 |
|---|---|
| Sprint 006 frozen `42:8` economics (mid positive; cross weak/negative) | **Unchanged** |
| Sprint 006 official run and contract | **Read-only inputs** |
| Sprint 007 diagnosis `EXECUTION_CALIBRATION_REQUIRED` | **Unchanged** |
| Sprint 007 D3 Path R envelope and unknown attainability | **Unchanged** |
| D2B expensive-package concentration | **Motivates hypotheses only**; does not validate a filter |

Sprint 008 must not:

- reinterpret Sprint 006 cross economics as revised by a long filter;
- treat Sprint 007 secondary findings as proof that a cutoff works;
- claim that historical quote scenarios are attainable package fills;
- or replace the authorized future execution-observation project with this research.

This sprint answers a different question on a **new research baseline**.

### Frozen experiment identity reused as selection authority

- Contract: [`configs/sprint006_baseline_v1.json`](../../configs/sprint006_baseline_v1.json)
- Feature window: `(max_lag, min_lag) = (42, 8)`
- Long expression: ATM straddle (`long_straddle`)
- Frequency / holding: weekly surface entry; hold to expiry; intrinsic settlement at exit spot
- Universe / liquidity: PIT dvol top 20%; `spread_bottom_pct = 1.0`; `max_leg_spread_pct = 0.5` on traded long legs
- Selection: top 10% momentum within eligible cross-section; within-side keep highest 50% CVG; sides independent; `max_names_per_side = 25`; earnings exclusion off
- Official artifacts: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- Official execution commit: `e205b9acc5d0400aa38169de721acb7fb8268f29`
- Primary reporting window: `2020-01-01` → `2026-07-10`
- Full history: `2018-10-26` → `2026-07-10`

---

## 4. Scope

### 4.1 In scope

1. **Measurement definition** for long straddles at entry time.
2. **Measurement–profitability validation** against an independent equal-dollar long baseline.
3. **Conditional threshold selection and evaluation**, only if the measurement gate passes.
4. Narrow analysis code / notebooks required for trusted, reproducible evidence.
5. Explicit go/no-go language and limitations for later work (including Sprint 009 short-side topics).

### 4.2 Explicitly out of scope

- Short-side measurements, iron-fly wings, protection, and margin (Sprint **009**).
- Signal-window searches or `42:8` retuning.
- New trade structures (naked call, condor, wingless fly, etc.).
- Execution infrastructure, broker code, live or paper order placement.
- Sizing-optimization search beyond the pinned equal-dollar research convention.
- Claiming ORATS / historical quotes imply attainable package fills.
- Editing `configs/sprint006_baseline_v1.json` or mutating official Sprint 006/007 evidence directories.
- Using future realized outcomes as entry features.
- Using `H / abs(S0 − K)` (unstable ATM intrinsic denominator; wrong economic object).

### 4.3 Candidate population (precise)

For each weekly entry date in the study calendar, the Sprint 008 **eligible constructable long set** is the long-side set that the frozen pipeline would have considered **before any experimental measurement filter**, defined as:

1. PIT liquid universe membership and joint Mom+CVG count eligibility (`min_count_pct = 0.80` → required count 28 of 35 for `(42,8)`).
2. Long pool: top `long_top_pct = 0.10` by momentum rank within the eligible cross-section.
3. Within-long CVG keep: highest `cvg_filter_pct = 0.50` by CVG rank within the long pool.
4. Constructable ATM long straddle: `structure_ok = true` under frozen surface construction and `max_leg_spread_pct`.
5. Earnings exclusion: none under the frozen contract (`earnings_exclusion_days = 0`).
6. Name cap: after rank sort (signal rank, ticker ascending tie-break), keep at most `max_names_per_side = 25` longs.

That capped constructable long set is the **pre-filter candidate population** \(N\) for equal-dollar allocation on that date.

**Research baseline inclusion** uses those candidates with the new equal-dollar sizing (below). It does **not** reuse historical Tier-A quantities financed by short premium, and it does not re-run short-side construction for budget determination.

Preferred input path (confirm in D0): official Sprint 006 mid/cross `candidate_view_*`, `trade_log_*`, and `leg_log_*` long rows, joined to shared bid/ask legs as in D2B. If a required field for a proposed measurement is missing, D0 must name the narrowest enabling post-pass — not a full `SurfaceRunner` redesign.

---

## 5. Independent equal-dollar long baseline

### 5.1 Motivation

Historical Sprint 006 long quantities are fill- and short-premium-dependent. That coupling confounds measurement–profitability inference. Sprint 008 therefore creates a **new research baseline**:

- independent fixed long-side budget \(B\);
- equal dollars per eligible long candidate on each entry date;
- no redistribution of rejected capital after a measurement filter.

### 5.2 Allocation rule

For each entry date and each execution scenario \(h\):

1. Identify the eligible constructable long set of size \(N\) **before** any experimental measurement filter.
2. If \(N = 0\): invest nothing; entire budget remains cash; date contributes zero trading P&L and full unused cash.
3. If \(N \ge 1\): each candidate is assigned stake \(B / N\).
4. Convert stake to quantity from the scenario’s **all-in** entry cost in consistent per-share research units (including fees):

\[
q_i(h) = \frac{B / N}{M_i + h\,H_i + \mathrm{fees}_i}
\]

   so each candidate consumes **exactly** \(B / N\) of the fixed budget under that \(h\).
5. Within a given \(h\) scenario, freeze these quantities across the unfiltered baseline and every threshold comparison.
6. Rejected allocations remain cash. Do **not** redistribute them to survivors.

Quantities **may differ across** \(h\) scenarios because the all-in entry cost depends on \(h\). Threshold contrasts must be within the same \(h\); identical quantities across \(h\) are **not** required.

Example: \(B = \$10{,}000\), \(N = 20\) → \$500 stake each. Under a given \(h\), retaining 12 leaves \$6,000 invested at that scenario’s all-in costs and \$4,000 cash against the original \$10,000 budget. The same date under a different \(h\) may use different quantities, but each retained name still consumes \$500 under that scenario.

### 5.3 Performance accounting

- Report portfolio performance against the **original budget** \(B\), including unused cash in the denominator where returns are portfolio-level.
- Primary trade-level outcome for measurement validation: **net return per dollar invested**, with invested dollars = the equal stake \(B / N\) (= \(q_i(h) \times (M_i + h H_i + \mathrm{fees}_i)\)).
- For filtered books under the same \(h\): invested capital = \(k \times (B / N)\) for \(k\) retained names; cash = \(B - k \times (B / N)\); portfolio return uses original \(B\) as the denominator.
- Portfolio-level threshold metrics: net return on original budget; drawdown on the budget path; coverage; cash retained; losses avoided; winning profits sacrificed.

### 5.4 Pins to freeze in D0 (accepted defaults in §12)

| Pin | Why it matters |
|---|---|
| Budget \(B\) | Level of reported dollars; not a search dimension |
| Entry-cost convention for quantities | All-in scenario cost \(M + hH + \mathrm{fees}\); must not silently mix mid-only sizing with cross P&L |
| Fees | Explicit modeled per-share research fees in the all-in denominator (may be zero) |
| Fractional quantities | Allowed for research simplicity |
| Zero-candidate dates | Cash-only; not dropped from calendar views unless declared |
| Missing outcomes | Exclude from trade-level association with documented reason; do not impute winners |
| Return denominators | Trade-level stake \(B/N\); portfolio original budget \(B\) including cash |

---

## 6. Measurement definitions and execution assumptions

### 6.1 Core quantities (consistent units)

All quote-based quantities use the **same call and put entry quotes** for the completed long straddle package. Do not back out \(H\) from portfolio P&L differences or historically resized quantities.

| Symbol | Definition |
|---|---|
| \(M\) | Entry **midpoint** debit of the complete long straddle (per share) |
| \(H\) | Package midpoint-to-full-cross concession from the same call/put quotes (per share). Matches D2B spirit: half-spread package width / cashflow magnitude when expressed as \(H/M\) |
| \(S_0\) | Entry spot |
| \(K\) | Common ATM body strike |
| \(X\) | \(\lvert S_T - K \rvert\) — expiration intrinsic payoff per share (straddle payoff) |
| \(h\) | Execution fraction in \([0,1]\): \(h=0\) midpoint; \(h=1\) full cross |
| Entry friction at \(h\) | \(hH\) **plus** explicitly modeled fees; all-in per-share research entry cost is \(M + hH + \mathrm{fees}\) (used for sizing and net P&L) |

**Benchmark measurement (required):** \(H/M\) — package width relative to midpoint debit. Already computed in Sprint 007 D2B as `package_width_to_cashflow` for included packages.

### 6.2 Additional measurements (at most two)

Propose at most two increments beyond \(H/M\), each with a distinct purpose:

| ID | Candidate | Purpose | Expected direction |
|---|---|---|---|
| **M1** | \(H/M\) | Benchmark cost intensity vs debit | Higher → worse net returns (mechanically and economically) |
| **M2** | \(H / S_0\) | Execution cost as additional underlying move | Higher → worse net returns |
| **M3** | Simple past-only payoff-hurdle comparison | Compare total payoff hurdle (entry debit + friction) to a transparent expected-payoff or past-move scale | Higher hurdle relative to available scale → worse net returns |

**Forbidden:** \(H / \lvert S_0 - K \rvert\); any feature that uses same-trade \(S_T\), \(X\), or realized P&L at entry time.

For every measurement advanced past D0, the deliverable design must document:

- formula and units;
- interpretation;
- expected relationship direction;
- required fields and artifact source;
- information-availability timing (must be known at entry);
- missing-data treatment.

### 6.3 Expected-payoff estimator (only if M3 proceeds)

If M3 needs an expected-payoff scale, use **one** transparent, bounded, past-only method — not a forecasting-model search. Accepted default (§12):

- At entry \(t\), use completed historical long-straddle observations with expiry strictly before \(t\).
- Estimate a simple central scale for payoff (e.g. mean or trimmed mean of \(X\) in a rolling calendar window, or a fixed lookback of completed trades), optionally normalized by \(S_0\) or \(M\) in a predeclared way.
- No parameter search against Sprint 008 profitability.
- Document assumptions, coverage holes, and cold-start rules for early dates.

### 6.4 Fixed execution-scenario set

Freeze scenarios **before** analysis. Do not pick the \(h\) that looks best.

| Scenario | Role |
|---|---|
| \(h = 1\) (full cross) | **Primary conservative comparison** |
| \(h = 0\) (midpoint) | Diagnostic gross reference only |
| Limited intermediates (\(h \in \{0.25, 0.50\}\); optional D3-envelope marks only if later justified) | Sensitivity only |

Hypothetical fills are **not** claimed attainable. Language from Sprint 007 attainability forbid-list remains in force.

---

## 7. Validate measurements before searching thresholds

### 7.1 Required gate

Threshold testing is **conditional**. It proceeds only when measurement evidence is classified **supported**.

Outcome labels:

- `supported` — relationship is economically meaningful, stable enough, and directionally consistent with a simple cutoff rationale;
- `unsupported` — no usable discriminatory relationship under preregistered criteria;
- `inconclusive` — mixed, underpowered, or unstable evidence.

A statistically significant Spearman correlation alone is **insufficient**.

### 7.2 Association design (freeze details in D1)

Examine, for each candidate measurement, trade-level **net return per dollar invested** on the equal-dollar baseline:

1. **Primary endpoint (D1):** predefined contrast \(\Delta=\overline{r}_{\mathrm{Q1}}-\overline{r}_{\mathrm{Q5}}\) with Bonferroni-adjusted consecutive-date block-bootstrap intervals over the three-measurement family (formulas in the D1 design).
2. **Five predefined score groups**, formed without optimizing against profitability (equal-count quintiles; tie/missing/freeze rules in D1).
3. Per group: counts, mean net returns, uncertainty, gross midpoint P&L vs execution drag, and contribution from large winners.
4. Consistency across development chronological halves (group \(\Delta\)) and **within entry dates** (robustness check in D1).

Pooled Spearman is a **supporting diagnostic** in D1; it is not the multiplicity-controlled gate.

### 7.3 Dependence and multiplicity

Do **not** treat every trade as an independent observation. Uncertainty must account for:

- common entry-date shocks (full cross-section on a date moves together);
- serial dependence involving recurring tickers across nearby weeks;
- multiple candidate measurements tested.

**Default dependence method:** consecutive-date **block** resampling. Each sampled block preserves (a) every trade in that date’s complete cross-section and (b) consecutive dated observations within the block, so shared date shocks and short-horizon serial dependence are retained together. Do **not** default to resampling individual dates in isolation while merely disclosing repeated tickers.

D1 must specify and justify—**before** examining association results—block length, resampling procedure, assumptions, and multiplicity treatment. Do not choose these settings from measurement–profitability performance.

### 7.4 Cost-based measurements: mechanical induction check

For cost-based measurements (\(H/M\), \(H/S_0\)):

- Separately report **gross midpoint P&L** \((X-M)/C\) and **execution drag** \((hH+\mathrm{fees})/C\) with \(C=M+hH+\mathrm{fees}\) (D1 formulas).
- Acknowledge that subtracting costs induces some mechanical association with net profitability.
- Significance on net alone does **not** establish that a better score group retains enough **gross midpoint** edge while cutting drag.

### 7.5 Preregistered go/no-go (architecture; formulas in D1)

Before opening new association results, D1 freezes criteria covering:

| Dimension | Intent |
|---|---|
| Economic magnitude | Group mean gaps large enough to matter vs noise |
| Uncertainty | Interval / consecutive-date block-resampling evidence not driven by one short block |
| Stability | Same qualitative pattern across development subperiods |
| Coverage | Enough trades/dates in extreme groups; not a tiny tail |
| Cutoff support | Evidence that a **simple** threshold could remove a consistently poor region without requiring a complex model |
| Gross-edge check | For cost measures: Q1 retains non-trivial gross midpoint P&L vs sample (explicit D1 formulas) |

Weak overall correlation may coexist with a consistently poor extreme group; the predefined group analysis must evaluate that possibility explicitly.

---

## 8. Development and evaluation separation

### 8.1 Chronological split (accepted default in §12)

| Period | Dates | Uses |
|---|---|---|
| **Development** | Earlier primary-window segment | Measurement selection among ≤3 candidates; threshold selection if gate passes |
| **Evaluation** | Later primary-window segment | Frozen-rule evaluation only |
| **Full-history companion** | `2018-10-26` → `2026-07-10` | Descriptive robustness; not for free selection |

**Accepted default:** development `2020-01-01` → `2023-12-31`; evaluation `2024-01-01` → `2026-07-10` (primary window). Confirm at D0 if any pin needs a versioned revision.

### 8.2 Honesty about prior inspection

Much of `2020–2026` was examined in Sprints 006–007 (aggregates, mid/cross, side splits, D2B terciles). Therefore:

- Later-period tests are **retrospective validation** relative to prior sprint inspection, not genuinely untouched holdout data.
- Do not label previously inspected history as pristine out-of-sample proof.
- Stronger confirmation requires **future** or otherwise unused observations (post-snapshot live weeks, or a later sealed window not used in measurement/threshold selection).

### 8.3 Information timing for estimators

Any historical estimator (including M3) may use only information available at the relevant entry time, with **completed** holding periods only.

### 8.4 Firewall

- Measurement selection and threshold selection must not consume final evaluation labels for decision-making.
- After freezing a rule on development, evaluation is a single locked pass (plus preregistered nearby-threshold sensitivity that does not re-select the winner).

---

## 9. Conditional threshold study

### 9.1 Entry condition

Run only if D1 classifies at least one measurement as `supported`.

### 9.2 Method

1. Test a **small, justified** set of simple cutoffs on the supported measurement(s) (e.g. exclude above a development quintile boundary or a small grid of economically motivated levels — exact set frozen in D2 design **before** output).
2. Select the rule on **development** data using preregistered selection criteria (not “best Sharpe shopping”).
3. Freeze the rule; evaluate on the later period.
4. Compare under matching conditions:
   - unfiltered equal-dollar baseline;
   - \(H/M\)-only filter (if another measurement won, still report \(H/M\) as benchmark comparator);
   - nearby thresholds;
   - the fixed execution-scenario set.
5. Within each \(h\), keep the scenario’s frozen equal-stake quantities and unused-cash treatment unchanged across baseline and threshold comparisons.

### 9.3 Required reports

- Net return on original budget and drawdown.
- Trade/date coverage and capital retained as cash.
- Losses avoided.
- Winning-trade profits sacrificed.
- Share of baseline winning-trade profits retained, including largest contributors.
- Stability across periods and nearby cutoffs.
- Reconciliation: within-\(h\) fixed-stake P&L improvement ≈ losses avoided − winning profits sacrificed.

### 9.4 Interpretation discipline

Distinguish:

- **improves the baseline** (relative to unfiltered equal-dollar long book under the same \(h\));
- **is profitable after modeled costs** (absolute sign under that \(h\)).

Do not select a cutoff solely because it produces the best historical result. Do not claim attainability of intermediate \(h\).

If the gate fails, D2 is a short **stop record**: measurement unsupported/inconclusive; no threshold search performed.

---

## 10. Deliverables

Detailed methods freeze in one-page designs immediately before each deliverable.

### D0 — Protocol freeze and input readiness

**Question:** Are candidate population, equal-dollar baseline pins, measurement catalog (≤3), scenario set, and chronological split frozen, and do accepted artifacts support them without unjustified engine work?

**Required answer:** `READY` / `READY_WITH_NARROW_ENABLING_CHANGE` / `BLOCKED_BY_SPECIFIC_INPUT_GAP`.

**Behavior:** Confirm official artifact identity; map required fields for \(M,H,S_0,K,X\) and legs; confirm long candidate reconstruction; pin §5/§6/§8 defaults that remain open; prefer notebooks + reuse of Sprint 006/007 artifacts; propose new engine work only for a concrete input gap.

### D1 — Measurement validation and gate decision

**Question:** Which entry-time measurements, if any, reliably distinguish equal-dollar net profitability in a way that supports a simple cutoff?

**Required answer:** Per measurement: `supported` / `unsupported` / `inconclusive`, plus one sprint-level gate decision for whether D2 threshold work is authorized. Dependence protocol (overlapping consecutive-date blocks, resampling, seed, Bonferroni multiplicity on \(\Delta\)) must be frozen in the D1 design before association output. Evaluation-period outcomes stay closed during D1.

### D2 — Conditional threshold study (if justified)

**Question:** Does a simple development-selected threshold improve the equal-dollar long book under matching conditions without destroying valuable winners?

**Required answer:** Effective / ineffective / inconclusive under primary \(h=1\), with evaluation-period evidence and sensitivity disclosure. Skip with stop record if D1 gate fails.

**Draft amendment (not accepted):** [`docs/tmp/sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) proposes replacing threshold selection with one frozen retrospective validation of the existing M1 and M2 exclude-U rules. Status `DRAFT — AWAITING REVIEW`. Until accepted, this section remains the plan text and `STOP_NO_THRESHOLDS` stands.

### D3 — Closeout

**Question:** What is the defensible answer to the central question, and what does it imply for subsequent work?

**Required answer:** Closeout memo with conclusions, limitations, relationship to Sprint 006/007, and implications for Sprint 009 / execution-observation work. Profitability must not be forced.

---

## 11. Cross-deliverable rules

1. **Preserve accepted history.** No silent revision of Sprint 006/007 verdicts.
2. **Long only.** No short-side filter research in this sprint.
3. **Equal-dollar independence.** No reuse of short-financed historical quantities for the research baseline; size from scenario all-in cost so each eligible name consumes exactly \(B/N\).
4. **No redistribution.** Filtered capital stays cash; within each \(h\), freeze scenario quantities across baseline vs threshold comparisons.
5. **Validate before thresholds.** Hard gate.
6. **Freeze before new granular output.** One-deliverable authorization.
7. **Scenario set fixed.** No \(h\)-shopping.
8. **Artifact-first.** Prefer post-pass analysis; minimal new code.
9. **Attainability forbid-list** from Sprint 007 remains in force for fill claims.
10. **Success ≠ profitability.** Unsupported or ineffective results complete the sprint.

---

## 12. Accepted protocol defaults

| # | Decision | Accepted default | Reason |
|---|---|---|---|
| 1 | Long research budget \(B\) | \$10,000 per entry date | Matches Sprint 006 side budget scale; keeps dollars comparable without implying identical economics |
| 2 | Quantity entry-cost convention | \(q_i(h)=(B/N)/(M_i + h H_i + \mathrm{fees}_i)\); freeze quantities within each \(h\) across threshold comparisons | Each name consumes exactly \(B/N\); within-\(h\) contrasts stay clean; quantities may differ across \(h\) |
| 3 | Modeled fees | \(\mathrm{fees}_i = 0\) in v1 of this experiment; still appear formally in the all-in denominator; report as limitation | Avoid confounding measurement study with an uncalibrated fee schedule; can be added later as sensitivity |
| 4 | Fractional quantities | Allowed | Research simplicity; not a live trading claim |
| 5 | Extra measurements beyond \(H/M\) | Advance **M2** \(H/S_0\); advance **M3** only if D0 confirms a clean past-only hurdle scale without model search | Caps degrees of freedom; M2 is simple and non-redundant; M3 is optional |
| 6 | M3 estimator (if used) | Rolling mean of completed historical \(X/S_0\) (or \(X\)) with fixed lookback; cold-start = missing measurement | Transparent and past-only |
| 7 | Execution scenarios | Primary \(h=1\); diagnostic \(h=0\); sensitivity \(h \in \{0.25, 0.50\}\) | Matches “full cross primary / mid diagnostic / limited intermediates” |
| 8 | Chronological split | Dev `2020–2023`; eval `2024–2026-07-10` | Simple calendar split; label eval as retrospective validation |
| 9 | Score groups | 5 equal-count quintiles on development-eligible trades with deterministic tie-break | Predefined; no profitability optimization |
| 10 | Dependence handling | Consecutive-date block resampling (preserve full date cross-sections and consecutive observations); D1 freezes block length, procedure, assumptions, and multiplicity **before** association output | Captures shared date shocks and serial dependence from recurring tickers; avoids performance-tuned inference choices |
| 11 | Threshold grid (only if gated) | Small set anchored to development quintile edges / 1–2 predeclared cost levels; pick by preregistered utility (loss avoided vs winner retention), not max return | Prevents cutoff shopping |
| 12 | Engine work | None unless D0 finds a concrete missing field | Preserves Sprint 007 artifact-first discipline |

These defaults are **accepted** with the sprint plan. Changing them after D1/D2 output is opened is not allowed without a versioned protocol revision.

---

## 13. Definition of done

Sprint 008 is complete when:

- [x] The research protocol (candidate population, equal-dollar baseline, scenarios, chronological firewall) is frozen and input-ready (D0).
- [x] Measurements are validated with an explicit `supported` / `unsupported` / `inconclusive` gate decision (D1 closed 2026-09-12; follow-ups reviewed; gate unchanged).
- [ ] D2 is accepted and executed, or a stop is recorded without threshold search. Design draft is awaiting review ([`sprint008_d2_design.md`](../tmp/sprint008_d2_design.md)); not accepted. Does not yet replace the original D2 definition.
- [ ] Closeout answers the central question defensibly, including limitations and implications (D3).
- [ ] Sprint 006/007 accepted results remain unreinterpreted.
- [ ] No signal-window, structure, short-side, or execution-policy winner is selected from this sprint.
- [ ] Hypothetical fills are not claimed attainable.
- [ ] Focused tests pass for any new financial calculation code.
- [ ] Remaining risks and required future/unused confirmation are documented.

An unsupported measurement, inconclusive relationship, or ineffective threshold is a **successful** completion if the evidence is trustworthy.

---

## 14. Authorization sequence

1. ~~Review and accept this sprint-level plan.~~ **Done** — plan accepted 2026-09-06.
2. ~~Update `docs/agenda/current_sprint.md` to Sprint 008 **Build/Audit**.~~ **Done.**
3. ~~D0 design~~ **Accepted**; ~~implementation/execution~~ **complete**; ~~evidence~~ **accepted** ([`sprint008_d0_evidence_review.md`](../tmp/sprint008_d0_evidence_review.md)).
4. ~~D1 design~~ **Accepted**; ~~implementation/execution~~ **complete**; ~~evidence~~ **reviewed** 2026-09-12 ([`sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md)). Within-date and corrected cost-diagnosis follow-ups reviewed. They extended D1 by a bounded amendment (two fixed exclusions) and did **not** complete planned D2 threshold selection or later-period evaluation. Gate remains `STOP_NO_THRESHOLDS`.
5. For each deliverable thereafter: inspect → one-page design → wait for acceptance → implement/execute → evidence review → next design.
6. Do not implement D2 or open evaluation outcomes until the draft amendment is accepted. Current gate: stop. Draft: [`sprint008_d2_design.md`](../tmp/sprint008_d2_design.md) (`DRAFT — AWAITING REVIEW`).

Pause and rescope if proposed work:

- changes frozen `42:8` selection rules;
- requires a full-history economic engine rewrite without a named input gap;
- expands into short-side / wing / margin research;
- searches many measurements or cutoffs after seeing results;
- or claims execution attainability from quotes alone.

---

## 15. Next action

**D1 is closed and reviewed.** Computed gate unchanged: `STOP_NO_THRESHOLDS` (M1/M2 `inconclusive`; M3 `unsupported`).

Original evidence: [`docs/tmp/sprint008_d1_evidence_review.md`](../tmp/sprint008_d1_evidence_review.md); `C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/` (executing SHA `72629a0d29f56771d1ff4a4ee3fe9cb227d593e4`, clean working tree).

Follow-ups reviewed and do not complete D2: within-date comparison; corrected cost diagnosis (commit `870d4b7` accepted).

**D2 design in progress / awaiting review.** Draft: [`docs/tmp/sprint008_d2_design.md`](../tmp/sprint008_d2_design.md). Bounded amendment only; D1 remains closed; `STOP_NO_THRESHOLDS` unchanged. Do not implement, open evaluation outcomes, search thresholds, or change sizing, signals, or the short side until the draft is accepted.

**D3** remains the subsequent sprint closeout.
