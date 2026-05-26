# Week 0 review notes

**Reviewer:** HD  
**Started:** 05-21-2026  
**Completed:** 2026-05-23

Use this file while reviewing Week 0 deliverables. When done, paste the **Questions for agent** section into Cursor to kick off Sprint 001 or doc fixes.

**Review checklist:** see chat / `docs/sprint_memos/000_week0_kickoff.md` for the full todo list.

---

## Structure decision (from review)

**Short side:** Iron fly vs iron condor (and wing width / delta) are **backtest options** — compare via separate runs, not necessarily both in one portfolio on day one. Some implementation work expected; SurfaceRunner already supports both as `short_structure` variants.

**Long side:** Naked long call remains a future research item (code today uses long straddle on high-momentum names). Defined-risk framing applies to **short** structures only.

---

## Progress

| Section | Read? | OK? | Notes |
|---------|-------|-----|-------|
| `docs/README.md` + archive | [x] | [x] | Update lightly at sprint close when new docs are added; no automation needed. |
| `docs/v1_spec_pins.md` | [x] | [ ] | Short structure line too narrow; see Needs change. Rest looks good. |
| `docs/v1_universe_protocol.md` | [x] | [x] | Fine for now; improve after liquidity panel review + weekly runbook. |
| `docs/v1_ops_model.md` | [x] | [ ] | Update structure section for fly vs condor comparison runs. |
| `docs/backtest_evaluation_protocol.md` | [x] | [ ] | Add Sharpe; clarify ranking vs return on max-loss. |
| `docs/repo_map.md` | [x] | [x] | Looks good. |
| `docs/decisions/001_canonical_backtest_path.md` | [x] | [x] | Agree with SurfaceRunner; note it is not fully complete. |
| `AGENTS.md` + `.cursor/rules/` | [x] | [x] | v1 sufficient; modes not multiple agents. |
| `docs/development_workflow.md` | [x] | [x] | Looks fine; add README update at sprint close. |
| `docs/baseline_status.md` + pytest | [x] | [x] | Looks fine. |
| Archive spot-check (optional) | [ ] | [ ] | Skipped for now. |

---

## Approved as-is

- **Archive split** — `docs/archive/` vs active docs; do not maintain archive in place.
- **Core v1 pins** — Momentum + CVG signal; 7 DTE; weekly rebalance (min); 50 max concurrent; Tier B go/no-go **2020 → latest**; conservative fills for go/no-go.
- **`docs/repo_map.md`** — layout, three backtest paths, data flow.
- **Decision 001** — SurfaceRunner as canonical v1 path (provisional until audit).
- **`docs/development_workflow.md`** — inspect → plan → test loop; ~20 hr/week rhythm.
- **`docs/baseline_status.md`** — 326 tests green via `C:/MomentumCVG_env/venv/`.
- **`AGENTS.md` + `.cursor/rules/`** — sufficient for now; Audit / Verification / Build modes; no extra agent products required yet.
- **`docs/v1_universe_protocol.md`** — PIT top-20% liquidity rule; Option A for first wiring; details after liquidity panel review.
- **Capital framing** — ~$1M as max-loss budget, not notional; tiny live fraction after paper.

---

## Needs change

| Doc | Change requested |
|-----|------------------|
| `docs/v1_spec_pins.md` | Replace "symmetric iron fly only" with: **short side** = iron fly **or** iron condor (chosen per backtest run; wing width/delta is a search dimension). **Long side** = document current long straddle; naked long call as deferred research. Defined risk applies to short structures only. |
| `docs/backtest_evaluation_protocol.md` | Add **Sharpe ratio** to Tier A/B metrics; state Sharpe as primary risk-adjusted metric alongside return on max-loss budget. |
| `docs/v1_ops_model.md` | Describe ops for **iron fly vs iron condor** as separate config runs (same weekly rebalance); defer naked long call ops until long-side structure is chosen. |
| `docs/v1_universe_protocol.md` | Add pointer: weekly trade sequence to live in future `docs/v1_weekly_runbook.md` after liquidity panel review (Sprint 002). |
| `docs/decisions/001_canonical_backtest_path.md` | Add **Known gaps**: SurfaceRunner not complete; short-structure comparison via config; caps/fills/weekly wiring TBD. |
| `docs/development_workflow.md` | Sprint close: update `docs/README.md` index when new docs or memos are added. |
| `docs/README.md` | After doc pass: refresh "Last updated" and any new links. |

---

## Questions for agent

1. **Doc pass:** Update the files in "Needs change" per this review (no code yet).
2. **Audit (Sprint 001):** What is missing in SurfaceRunner for weekly rebalance + iron fly vs iron condor comparison runs + 50-name cap?
3. **Liquidity panel (Sprint 002):** After review, draft `v1_weekly_runbook.md` so weekly trade flow is explicit.

---

## Questions for myself (research later)

1. Naked long call vs long straddle on high-momentum names — when to test after short-side structure is chosen?
2. Draft Sharpe and max drawdown thresholds for Tier B (2020 → latest, conservative fills).
3. Pull anything critical from `docs/archive/production_ready_checklist_options_strategy.md` into active docs before paper trading?

---

## Personal go/no-go thresholds (draft)

| Metric | Draft threshold |
|--------|-----------------|
| Sharpe ratio (Tier B, conservative fills) | _TBD_ |
| Return on max-loss budget (Tier B, conservative fills) | _TBD_ |
| Max drawdown (Tier A or B) | _TBD_ |
| Top-5 name PnL concentration limit | _TBD_ |

---

## Review outcome

- [ ] **Ready for Sprint 001** — no doc changes needed
- [x] **Revise Week 0 docs first** — see "Needs change" above (**applied 2026-05-23**)
- [ ] **Blocked** — explain:

**Next action:** Start **Sprint 001** (repo audit + one verification test). Say *"start Sprint 001"* in Cursor.
