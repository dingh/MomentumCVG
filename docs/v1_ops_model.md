# V1 operations model

**Status:** Active (estimates — update after shadow mode)  
**Last updated:** 2026-05-23 (Week 0 review revision)

---

## Purpose

Estimate **weekly operational load** from v1 pins so broker choice (manual vs IBKR) is data-driven, not guessed.

---

## V1 pinned assumptions

| Parameter | Value |
|-----------|-------|
| Rebalance | Weekly |
| Max concurrent positions | 50 |
| Short structure | Iron fly **or** iron condor (one per backtest/live config; 4 legs each) |
| Long structure | Long straddle (2 legs per name; naked long call deferred) |
| DTE | 7 |
| Backtest exit | Hold to expiry |
| Live exit | Early close (TBD before paper) — **increases ops vs backtest** |
| Delta hedge | Off |

---

## Short-structure backtest comparison

Iron fly vs iron condor (and wing width / delta) are evaluated as **separate backtest runs**, not mixed in one portfolio on day one. Each run uses one `short_structure` in `BacktestRunConfig`; compare Sharpe and return on max-loss budget under conservative fills before picking a live default.

Implementation: `SurfaceRunner` + `run_surface_search.py` grid over `short_structure` and wing parameters.

---

## Order volume estimates

### Weekly rebalance — new short structures (4 legs each)

Assumes one short structure type per week (fly or condor, not both in the same book).

| Scenario | New short structures / week | Leg orders (×4) | Notes |
|----------|----------------------------|-----------------|-------|
| Low churn | 5–10 | 20–40 | Most names roll off at expiry; universe stable |
| Medium churn | 10–20 | 40–80 | Typical if ~30–50% book turns weekly |
| High churn | 20–35 | 80–140 | Universe or ranking unstable |
| Full refresh (worst) | up to 50 | up to 200 | Unlikely every week; stress case |

### Long side (long straddle, 2 legs each)

Add roughly **2× leg count** for high-momentum names opened the same week (fewer names than short side depending on `long_top_pct`).

### Monitoring (open book)

| Open positions | Legs to monitor | Frequency |
|----------------|-----------------|-----------|
| 50 | 200 | Daily mark / risk check recommended |
| 30 | 120 | Lighter |

### Live early close (not in v1 backtest)

When live exit rule is added (e.g. close at 1 DTE or 50% max profit), add **closing orders** — potentially **up to 50 × 4 = 200 legs/week** in worst case if all positions close on schedule. Plan IBKR or batch workflows before scaling.

---

## Broker decision thresholds (provisional)

| Avg structure-opens per week (4-week shadow avg) | Recommendation |
|----------------------------------------------------|----------------|
| ≤ 15 | Manual execution feasible with strict checklist |
| 16–25 | IBKR with saved order templates; semi-manual |
| > 25 | IBKR paper/live; plan order generation script |

**Accounts:** IBKR primary for paper and multi-leg; Robinhood not on critical path.

---

## Capital and risk framing (~$1M)

- Plan in **max-loss budget**, not premium notional.
- v1 live deployment: start **small fraction** of deployable max-loss budget (exact % pinned in Sprint 002 / go/no-go memo).
- 50 concurrent short structures (fly or condor) with equal max-loss sizing can deploy meaningful capital if per-trade max-loss is set appropriately — **budget caps are mandatory** before live.

---

## Weekly human time (rough)

| Activity | Hours/week (manual-friendly) | Hours/week (high churn) |
|----------|------------------------------|-------------------------|
| Review signals / intended trades | 1–2 | 1–2 |
| Enter / adjust orders | 1–3 | 3–6 |
| Reconcile fills vs shadow | 1 | 2–3 |
| Risk / drawdown check | 0.5 | 0.5 |
| **Total** | **3.5–6.5** | **6.5–11.5** |

Target: keep **≤ 8 hours/week ops** at v1 scale; if shadow exceeds thresholds, automate order generation before live capital.

---

## Shadow mode outputs (Month 2)

For each rebalance date, log:

- Universe size and top liquidity cutoff
- Ranked signals and selected names (≤ 50)
- Intended structures: short (fly or condor) and long (straddle) — strikes, expiry, credit, max loss, contracts
- Compare to backtest intent for same date

Update this doc with **measured** 4-week averages before paper trading.

---

## Change log

| Date | Change |
|------|--------|
| 2026-05-23 | Initial ops model; 50 max concurrent |
| 2026-05-23 | Week 0 review: fly vs condor as separate runs; long straddle leg counts |
