# Sprint 008 D1 cost diagnosis — Evidence Review

**Date:** 2026-09-11  
**Protocol:** [`sprint008_d1_cost_diagnosis_protocol.md`](sprint008_d1_cost_diagnosis_protocol.md) (frozen before results)  
**Evidence dir:** `C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260911T162501Z/`  
**HEAD SHA at execution:** `db0e859`  
**Working tree:** **dirty** (this follow-up not yet committed); source provenance in `execution_receipt.json` (`tracked_diff_sha256_16=0d26c7778336d755`, file hashes recorded)  
**Command:** `C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint008_d1_cost_diagnosis.py`  
**Runtime:** ~50.4s (panel ~45.7s)  
**Environment:** Python 3.13.7; numpy 2.4.1; pandas 3.0.0; scipy 1.17.1; matplotlib 3.10.8  
**Preserved historical D1 gate:** **`STOP_NO_THRESHOLDS`** (unchanged)  
**Post-hoc disclosure:** Designed after D1 and within-date follow-up results; exploratory, not independent confirmation.

Bounded amendment: exactly two fixed U-exclusions (M1, M2). Broader D2 threshold search remains outside authorization.

---

## Reconciliation to prior within-date follow-up

| Measurement | Eligible dates | Mean \(d_{\mathrm{net}}\) | Prior mean \(d_t\) | Match |
|---|---:|---:|---:|---|
| M1 | 209 | +0.052389 | +0.052389 | exact |
| M2 | 209 | +0.075514 | +0.075514 | exact |

Excluded split dates: 0. Group sizes: L=643, middle=2299, U=643 (pooled scored trades). Actual group fraction ≈ \(k/n\) with median \(k=3\).

---

## Findings (development 2020–2023, \(h=1\))

### 1) Baseline still has positive gross midpoint economics

Unfiltered executed book (\(n=3585\), 209 dates):

| Metric | Value |
|---|---:|
| Mean gross \(g=(X-M)/C\) | **+0.0349** |
| Mean spread drag \(a=H/C\) | +0.0360 |
| Mean net \(r\) | −0.0011 |
| Total dollar P&L on \(B\) path | +\$6,628 |
| Gross / net win rates | 41.8% / 40.3% |

Gross midpoint P&L on full-cross capital is positive on average; spread drag approximately cancels it in the pooled mean net.

### 2) Spread variation is large enough to matter mechanically

Date-weighted mean \(\overline{a}_U-\overline{a}_L\):

| | M1 | M2 |
|---|---:|---:|
| Mean spread saving (\(U-L\) on \(H/C\)) | **+5.92 pp** | **+4.69 pp** |
| Pooled mean \(H/C\) L / mid / U | 1.35% / 3.23% / 7.17% | 1.65% / 3.41% / 6.23% |

Among selected ATM straddles, within-date high-score names carry several percentage points more spread drag on invested capital than low-score names. This sample establishes dispersion among ATM trades; it does **not** prove ATM selection caused that dispersion.

### 3) For M1, net L−U is mostly mechanical; gross does not help

Exact identity (equal-weight eligible dates): \(d_{\mathrm{net}}=d_{\mathrm{gross}}+\mathrm{spread\_saving}\)

| | M1 | M2 |
|---|---:|---:|
| Mean \(d_{\mathrm{net}}\) | +0.0524 | +0.0755 |
| Mean \(d_{\mathrm{gross}}\) | **−0.0068** | +0.0286 |
| Mean spread_saving | +0.0592 | +0.0469 |

**M1:** Almost all of the favorable net L−U gap is spread saving. Gross midpoint slightly **favors U**.  
**M2:** Both gross and spread saving contribute positively to net L−U.

Gross→net non-winner frequency (pooled scored): ~1.5% of trades.

### 4) Fixed U-exclusion: small positive point uplift, not significant; winners matter

Excluding the within-date U group (~18% of capital → cash; quantities of retained trades unchanged):

| | M1 exclude-U | M2 exclude-U |
|---|---:|---:|
| Mean weekly uplift on \(B\) | +0.68 pp | +0.88 pp |
| Total P&L baseline → filtered | \$6,628 → \$20,823 | \$6,628 → \$25,088 |
| Total P&L improvement | +\$14,195 | +\$18,460 |
| Losses avoided | \$121,604 | \$124,639 |
| Winning profits sacrificed | \$107,409 | \$106,179 |
| Winning-profit retention | 83.8% | 84.0% |
| Top-5 winner profit retention | 80.0% | **51.0%** |
| Top-10 winner profit retention | 88.0% | 62.9% |
| HAC adj. \(p\) (family 4) on mean uplift | 1.00 (n.s.) | 0.91 (n.s.) |

Identity check: P&L improvement = losses avoided − winning profits sacrificed (passed).

Point economics look better than baseline, but **multiplicity-adjusted HAC intervals include zero**. M2 in particular sacrifices a large share of top-winner profits. Higher mean return on retained trades is not enough without reliable total-budget improvement.

Peak-to-trough on cumulative fixed-budget date P&L remains large (~−\$63.6k) for baseline and filtered (not a compounded equity curve).

### 5) Consistency

- M1 date-level \(d_{\mathrm{net}}\) similar across halves (~+5.7 / +4.8 pp).  
- M2 stronger in 2022–2023 (~+3.4 / +11.8 pp).  
- Win-rate L−U differences are small and not significant (M1 ~+1.1 pp; M2 ~+4.3 pp; adj. \(p\) 1.00 / 0.41).  
- Results are not a few-trade artifact for M1 top winners (top5 retained 80%), but M2 top-winner sacrifice is material.

### 6) Unresolved

- Fees unmodeled; quote full-cross ≠ achievable fills or dependable income.  
- Post-hoc study; not independent confirmation.  
- No evaluation-period validation.  
- Broader cutoff search still unauthorized under historical D1 stop.

---

## Inference (family size 4)

| Contrast | Mean | Adj. \(p\) | Adj. sig? |
|---|---:|---:|---|
| M1 mean uplift | +0.0068 | 1.00 | no |
| M2 mean uplift | +0.0088 | 0.91 | no |
| M1 net win-rate \(L-U\) | +0.0108 | 1.00 | no |
| M2 net win-rate \(L-U\) | +0.0431 | 0.41 | no |

Prior within-date mean-\(d_t\) inference reproduced (family-size-2 style) and remains non-significant — unchanged.

---

## Decision answers

1. **Positive gross midpoint on baseline?** Yes (~+3.5% mean \(g\)).  
2. **Meaningful spread dispersion?** Yes (~5–6 pp \(H/C\) U−L).  
3. **Savings offset by gross?** M1: net gap is almost pure cost; gross slightly hurts L. M2: gross also helps L.  
4. **Does fixed exclusion improve total profit?** Point yes (~\$14–18k), but **not statistically supported**; M2 pays with large top-winner sacrifice.  
5. **Broad vs concentrated?** M1 more stable; M2 half-skewed and top-winner sensitive.  
6. **Unresolved?** Fees, fill realism, forward validation, independent confirmation.

---

## Recommended next action (one)

**Close this cost-filter direction as inconclusive for decision use**, unless separately authorized independent evidence is obtained (e.g., pre-registered forward/chronological validation of a *frozen* rule — not implied by this study).

Rationale: favorable point U-exclusion economics exist, but (i) M1’s within-date net edge is largely mechanical spread accounting rather than better gross payoff, (ii) none of the four frozen HAC contrasts is multiplicity-adjusted significant, and (iii) M2 damages top-winner contribution. This does **not** reopen D2 search.

Historical D1 **`STOP_NO_THRESHOLDS`** remains the binding gate.

---

## Artifacts

- `trade_level.parquet/.csv`, `date_decomposition.*`, `date_portfolio.*`
- `decomp_bars.png`, `cum_pnl_M1.png`, `cum_pnl_M2.png`, `drag_boxplot_M*.png`
- `cost_diagnosis_report.json/.md`, `execution_receipt.json`

Notebook: `notebooks/sprint008/d1_cost_diagnosis.ipynb`  
Tests: `tests/unit/test_sprint008_d1_cost_diagnosis.py` (9) + existing D0/D1/FU regressions

**Fixes applied before official evidence:** preserved original DataFrame indices in within-date group selection so U labels map to the parent panel (required for portfolio exclusion). D0 `_required_input_ok` enforced on the analysis path (3585/3585 passed).

**Disclaimer:** Fees remain unmodeled; quote-based results do not establish achievable fills or dependable income.
