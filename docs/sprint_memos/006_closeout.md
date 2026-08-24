# Sprint 006 — closeout

**Status:** `CLOSED — EVIDENCE ACCEPTED; FROZEN 42:8 ECONOMICS WEAK/NEGATIVE; HYPOTHESIS REJECTED/DEFERRED`
**Closed:** 2026-08-24
**Closeout documentation commit:** on top of `1ac82e1` (Phase 5 economic review)
**Execution / tested baseline SHA:** `e205b9acc5d0400aa38169de721acb7fb8268f29`

---

## 1. Verdict

Sprint 006 delivered the **first trusted real-data economic backtest** of the frozen `42:8` Momentum+CVG baseline on accepted Sprint 004/005 artifacts, with blind technical verification before aggregate economics were opened.

| Conclusion | Result |
|------------|--------|
| **Evidence verdict** | `ACCEPTED` |
| **Economic characterization** | `WEAK/NEGATIVE` |
| **Recommendation** | **Reject/defer this frozen `42:8` economic hypothesis** |

This closeout accepts Phase 5 substantive conclusions from the official `decision_report.json` only. No profitability cutoff was invented. The economic characterization applies **only** to this frozen configuration, accepted dataset, and execution assumptions. It is **not** a general rejection of Momentum or CVG as signal families.

---

## 2. Sprint objective and frozen experiment identity

**Central question:** Does the frozen `42:8` Momentum+CVG signal produce believable economic results on the accepted real dataset after conservative cross fills?

| Field | Value |
|-------|-------|
| Experiment / contract id | `sprint006_baseline_v1` |
| Contract file | `configs/sprint006_baseline_v1.json` |
| Contract SHA-256 (on-disk CRLF, receipt) | `4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c` |
| Feature window | `max_lag=42`, `min_lag=8`, `window_size=35`, `search=false` |
| Structures | long ATM straddle; short iron fly (`wing_delta_target=0.15`) |
| Sizing | Tier A `equal_max_loss`; short budget `$10,000` / side |
| Primary fill | cross `(1.0, 1.0)` — **primary economic view** |
| Diagnostic fill | mid `(0.5, 0.5)` — fill-assumption sensitivity only |
| Full-history window | `2018-10-26` … `2026-07-10` |
| Primary reporting window | `2020-01-01` … `2026-07-10` |
| Snapshot | `e2c1f8fd44d72176` / build `20260724T045049097520Z_40b16886` |
| Baseline features | `C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/features_42_8.parquet` |

D0–D3 implementation record: contract `1cdfad7`; D1 `241b0d3` + `c6b1735`; D2 `9224068`; D3 through `10133f6`.

---

## 3. Official run and execution identity

| Field | Value |
|-------|-------|
| Execution commit | `e205b9acc5d0400aa38169de721acb7fb8268f29` |
| Official `RUN_DIR` | `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z` |
| Command | `scripts/run_sprint006_baseline.py --contract configs/sprint006_baseline_v1.json --output-dir <RUN_DIR>` |
| Receipt `result_complete` | `true` |
| Receipt `has_unresolved_failures` | `false` |
| Per fill | `n_expected_dates=403`, `n_traded_dates=403`, `n_valid_no_trade_dates=0`, `n_failed_dates=0` |
| Authoritative report | `…/decision_report.json` |
| Phase 1/2 checkpoint | [`docs/tmp/sprint006_d4_phase12_checkpoint.md`](../tmp/sprint006_d4_phase12_checkpoint.md) |
| D4 evidence memo | [`sprint006_d4_baseline_execution_evidence.md`](sprint006_d4_baseline_execution_evidence.md) |

Smoke economics were never cited. No baseline rerun occurred at closeout.

---

## 4. Phase 4 evidence verdict — `ACCEPTED`

From [`sprint006_d4_baseline_execution_evidence.md`](sprint006_d4_baseline_execution_evidence.md) and review bundle [`docs/tmp/sprint006_d4_phase34_review/`](../tmp/sprint006_d4_phase34_review/):

| Item | Result |
|------|--------|
| V-1…V-20 | all **PASS** |
| Independent §7.4 source audit | **184 PASS / 0 FAIL / 1 N/A** (accepted through `326e13d`) |
| S3 | **N/A** only (`n_valid_no_trade_dates=0`) |
| Frozen samples | S1-L `2022-09-02/ACN/long`, S1-S `2022-09-02/AMC/short`, S2-L `2018-10-26/ABBV/long`, S2-S `2018-10-26/MRVL/short`, S4 `2018-10-26/AMBA/short` — preserved without substitution |
| Verifier | audit-local calculations only (no production calculation helpers) |
| Phase 3 shell limitation | `EXIT_CODE`/stdout/stderr not retained — accepted as non-blocking |

---

## 5. Phase 5 economic characterization and recommendation

**Characterization:** `WEAK/NEGATIVE` (primary cross fill)

**Recommendation:** **Reject/defer this frozen `42:8` economic hypothesis.**

Primary cross results are uniformly negative across views and reporting windows. View B compounded return is essentially total loss on both windows. View A mean cycle CAR is negative. Sharpe is negative. Both long and short sides lose; short iron-fly losses dominate magnitude. Win rate and profit factor are below breakeven. Every primary-window year is negative on View B compounded return.

Mid-fill diagnostics are materially positive while cross is deeply negative. This is **fill-assumption sensitivity**, not a pure transaction-cost or pure economic comparison: in this run the included date/ticker/direction key sets matched between fills (`n_dates_mid_only=0`, `n_dates_cross_only=0`, `n_candidates_mid_only=0`, `n_candidates_cross_only=0`), but fills still change quantities and per-trade economics. Mean cross-minus-mid CAR ≈ −5.1% per traded date under cross conservatism.

**Revisit condition:** this hypothesis should remain deferred unless a **separately motivated and preregistered experiment** directly addresses implementable execution economics. Do not search for a better window, structure, filter, or parameter in response to these results.

---

## 6. Primary cross-fill headline results

Source: `decision_report.json` → `by_fill.cross`. View A = **conditional on traded dates**.

| Window | View | mean cycle CAR | compounded | annualized return | Sharpe | drawdown | n traded / complete |
|--------|------|----------------|------------|-------------------|--------|----------|---------------------|
| primary | A conditional | −0.0270836 | n/a | n/a | −1.30292 | −1.00000 | 341 |
| primary | B calendar | n/a | −0.999998 | −0.867194 | −1.30292 | −1.00000 | complete |
| full_history | A conditional | −0.0355691 | n/a | n/a | −1.58427 | −1.00000 | 403 |
| full_history | B calendar | n/a | −1.000000 | −0.929433 | −1.58427 | −1.00000 | complete |

---

## 7. Yearly stability (cross, primary window)

| Year | n traded | View B compounded | View B annualized | View A Sharpe *(conditional)* | View B drawdown |
|------|----------|-------------------|-------------------|-------------------------------|-----------------|
| 2020 | 53 | −0.314408 | −0.309508 | 0.500566 | −0.843350 |
| 2021 | 52 | −0.975471 | −0.975471 | −2.85447 | −0.972685 |
| 2022 | 52 | −0.547406 | −0.547406 | −0.495020 | −0.703878 |
| 2023 | 52 | −0.664648 | −0.664648 | −0.855788 | −0.527277 |
| 2024 | 52 | −0.800947 | −0.800947 | −1.06172 | −0.892461 |
| 2025 | 52 | −0.942086 | −0.942086 | −2.75104 | −0.940825 |
| 2026 | 28 | −0.939517 | −0.994538 | −5.04087 | −0.937364 |

Only 2020 shows a positive View A mean cycle CAR (+0.0156632); its calendar compounded return remains negative.

---

## 8. Long/short attribution (cross, primary window)

| Side | n traded rows | mean cycle return *(conditional)* | pnl_total | capital_at_risk_dollars |
|------|---------------|-------------------------------------|-----------|---------------------------|
| long | 5890 | −0.00963737 | −16,992.99 | 2,701,558 |
| short | 3322 | −0.0430235 | −146,279.85 | 3,400,000 |

---

## 9. Concentration (cross, primary window)

| Rank | Ticker | abs_pnl | share |
|------|--------|---------|-------|
| 1 | GME | 61,003.43 | 0.0187395 |
| 2 | DASH | 40,407.38 | 0.0124126 |
| 3 | ZM | 34,290.69 | 0.0105337 |
| 4 | AVGO | 34,069.86 | 0.0104658 |
| 5 | VZ | 33,048.02 | 0.0101519 |

**top5_share_sum:** 0.0623036 (6.23% of total |PnL|).

---

## 10. Activity, coverage, and structure failures (cross, primary window)

| Metric | Value |
|--------|-------|
| avg included names / traded date | 27.01466275659824 |
| turnover `mean_included_names` | 27.01466275659824 |
| turnover complete | true |
| win_rate | 0.369501 |
| profit_factor | 0.631695 |
| no_trade_frequency | 0.0 |
| joint_coverage_rate | 1.0 |
| mean jointly eligible | 332.806 |
| sum included | 9212 |
| structure failures (metadata / missing body / wing-liq / other) | 239 / 306 / 2096 / 0 |

Every expected date traded (`n_valid_no_trade=0`, `n_failed=0`).

---

## 11. Fill-assumption sensitivity (mid vs cross)

Cross-minus-mid is **fill-assumption sensitivity**, not a pure transaction-cost or pure economic comparison. In this run, included date/ticker/direction key sets matched between fills, but fills can still affect quantities and trade economics.

| Window | n dates both traded | mid-only / cross-only dates | n candidates both included | mid-only / cross-only candidates | mean cross−mid CAR | mean cross−mid PnL | mean spread_cost_ratio cross | mean spread_cost_ratio mid | mean leg_spread_to_credit cross | mean leg_spread_to_credit mid |
|--------|---------------------|----------------------------|----------------------------|----------------------------------|--------------------|--------------------|------------------------------|----------------------------|---------------------------------|-------------------------------|
| primary | 341 | 0 / 0 | 9212 | 0 / 0 | −0.0510452 | −945.912 | 0.0530103 | 0.0 | 0.153203 | 0.138944 |
| full_history | 403 | 0 / 0 | 10486 | 0 / 0 | −0.0512981 | −946.927 | 0.0529556 | 0.0 | 0.152504 | 0.138324 |

Mid diagnostic (not the economic result): primary View B compounded +48.8551 vs cross −0.999998 — sign reversal under fill assumption alone.

---

## 12. Definition of done

| Outcome | Status | Evidence |
|---------|--------|----------|
| One frozen baseline runs reproducibly from accepted inputs | ✓ | Official run + receipt at `e205b9a`; [`sprint006_d4_baseline_execution_evidence.md`](sprint006_d4_baseline_execution_evidence.md) §1 |
| Reproducible through one documented command | ✓ | Adapter CLI + frozen contract; evidence memo §1 |
| Input, configuration, code, output identities recorded | ✓ | Receipt + Phase 1 gate; V-10…V-14 |
| Joint eligibility and date handling verified | ✓ | D2 implementation; V-3…V-7, V-17 |
| Every expected date classified; no silent loss | ✓ | 403/403 traded; V-5; `n_failed=0` |
| Small trade sample independently checked | ✓ | §7.4 audit 184/0/1 through `326e13d` |
| Full-history mid + cross runs complete | ✓ | 17 artifacts; both fills in one invocation |
| Final report supports credibility judgment | ✓ | `decision_report.json` |
| Documented without post-P&L retuning | ✓ | Frozen contract unchanged |
| Clear recommendation recorded | ✓ | **Reject/defer** this frozen hypothesis |

---

## 13. Accepted limitations

### Report (`limitations[]`)

- Hold-to-expiry; positions are not managed intra-week.
- No earnings filter.
- Iron-fly wings use below-nearest 0.15-delta selection.
- Tier A sizing is not integer lots.
- Long-only fallback dates are possible.
- Mid is a fill-assumption diagnostic, not a pure transaction-cost attribution.
- `robust_score` is not a decision metric and is not used for go/no-go.
- Post-signal candidate/funnel artifacts cannot support full-universe Momentum IC or CVG increment tests.

### Phase 4 / execution (additional)

- Phase 3 shell `EXIT_CODE`, stdout, and stderr not retained (non-blocking).
- Input digests recorded but not pinned in code (PG-2).
- Receipt reports `deliverable=sprint006_d3` (expected producer metadata).
- S3 audit N/A (`n_valid_no_trade=0`).
- Iron-condor alternative untested while KB-001 remains open.

---

## 14. Phase 1 test gate (recorded at D4 execution)

From accepted Phase 1 evidence at `e205b9a` ([`sprint006_d4_phase12_checkpoint.md`](../tmp/sprint006_d4_phase12_checkpoint.md)):

| Item | Value |
|------|-------|
| Full suite | **1597 passed**, 1 skipped |
| Focused Sprint 006 subset | **332 passed** |
| Tested baseline / execution commit | `e205b9acc5d0400aa38169de721acb7fb8268f29` |

Closeout does **not** claim a new suite run.

---

## 15. Post-Sprint-006 handoff

Sprint 006 is **closed**. Sprint 007 is **not** authorized by this closeout.

The accepted single-config runner, result schema, and official run artifacts remain available for reference. Any future work on this economic question requires a **new preregistered experiment contract** — not retuning of `configs/sprint006_baseline_v1.json` or parameter search motivated by these P&L results.

Preserve outside-repository artifacts:

- Official run: `C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z`
- Verification: `C:/MomentumCVG_env/runs/sprint006_d4_verification_20260823T204430Z`
- Phase 4 review bundle: `docs/tmp/sprint006_d4_phase34_review/`
