# Sprint 006 D4 — Phase 4 source-reconstruction audit

**Audit verdict:** `PASS`
**Lifecycle status:** `PHASE 4 REVERIFICATION COMPLETE — AWAITING REVIEW`
**Phase 5 authorized:** `false`

- PASS: 184
- FAIL: 0
- N/A: 1 (S3 only permitted: `True`)
- Samples replaced: `false`
- Baseline rerun: `false`
- Aggregate economics opened: `false`

## Frozen samples

| Sample | Key |
|--------|-----|
| S1-L | 2022-09-02 / ACN / long |
| S1-S | 2022-09-02 / AMC / short |
| S2-L | 2018-10-26 / ABBV / long |
| S2-S | 2018-10-26 / MRVL / short |
| S3 | N/A (n_valid_no_trade=0) |
| S4 | 2018-10-26 / AMBA / short |

## §7.4 coverage checklist

| Stage | Covered | Verdict | n_checks |
|-------|---------|---------|----------|
| `identity` | True | PASS | 4 |
| `sample_selection` | True | PASS | 4 |
| `universe_snapshot_date` | True | PASS | 4 |
| `universe_atm_pair` | True | PASS | 4 |
| `universe_dvol_spread_fields` | True | PASS | 4 |
| `universe_dvol_rank` | True | PASS | 4 |
| `universe_spread_rank` | True | PASS | 4 |
| `universe_and_membership` | True | PASS | 4 |
| `joint_universe_feature_membership` | True | PASS | 4 |
| `joint_finite_values` | True | PASS | 4 |
| `joint_mom_count` | True | PASS | 4 |
| `joint_cvg_count` | True | PASS | 4 |
| `joint_eligible_slice` | True | PASS | 4 |
| `signal_rank_recompute` | True | PASS | 4 |
| `cvg_rank_recompute` | True | PASS | 4 |
| `direction_and_cvg_retention` | True | PASS | 4 |
| `option_a1_surface_valid` | True | PASS | 4 |
| `option_entry_spot` | True | PASS | 4 |
| `option_exit_spot` | True | PASS | 4 |
| `option_body_strike` | True | PASS | 4 |
| `option_expiry_date` | True | PASS | 4 |
| `option_dte_actual` | True | PASS | 4 |
| `option_body_selection` | True | PASS | 4 |
| `option_spread_gates` | True | PASS | 4 |
| `option_wing_delta_rule` | True | PASS | 2 |
| `option_leg_count` | True | PASS | 4 |
| `option_leg_identity` | True | PASS | 12 |
| `option_mid_cross_half_spread` | True | PASS | 12 |
| `tier_a_n_short` | True | PASS | 4 |
| `tier_a_n_long` | True | PASS | 4 |
| `tier_a_at_risk_per_share` | True | PASS | 4 |
| `tier_a_short_budget_split` | True | PASS | 2 |
| `tier_a_long_budget` | True | PASS | 2 |
| `tier_a_long_budget_split` | True | PASS | 2 |
| `tier_a_collected_credit` | True | PASS | 4 |
| `tier_a_fallback` | True | PASS | 4 |
| `tier_a_quantity` | True | PASS | 4 |
| `pnl_entry_cost` | True | PASS | 4 |
| `pnl_exit_value` | True | PASS | 4 |
| `pnl_per_share` | True | PASS | 4 |
| `capital_at_risk_dollars` | True | PASS | 4 |
| `pnl_total` | True | PASS | 4 |
| `date_car_contribution` | True | PASS | 2 |
| `S3_valid_no_trade` | True | N/A | 1 |
| `S4_structure_failure` | True | PASS | 6 |

## Identities

- Execution commit: `e205b9acc5d0400aa38169de721acb7fb8268f29`
- RUN_DIR: `C:\MomentumCVG_env\runs\sprint006_baseline_v1_20260823T204430Z`
- VERIFY_DIR: `C:\MomentumCVG_env\runs\sprint006_d4_verification_20260823T204430Z`
- Artifact digests re-verified against receipt
- Phase 1 accepted-input digests re-verified

## Phase 3 shell limitation (non-blocking)

Phase 3 capturing shell EXIT_CODE and stdout/stderr were not retained (documented non-blocking operational limitation). Not recovered; baseline not rerun.

## Failed rows

- none
