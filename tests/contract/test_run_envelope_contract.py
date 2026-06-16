"""
Contract: R0 — run envelope (BacktestRunConfig validation).

R0 turns user intent into a validated, immutable run configuration. These tests
pin the invariants the rest of the pipeline relies on: date ordering and
fractional thresholds in [0, 1].

See docs/surface_engine_data_contract.md § R0.
"""
from __future__ import annotations

from datetime import date

import pytest

from tests.contract.conftest import make_contract_config


def test_valid_config_constructs():
    cfg = make_contract_config()
    assert cfg.start_date < cfg.end_date
    assert cfg.run_id


def test_start_must_be_before_end():
    with pytest.raises(ValueError):
        make_contract_config(start_date=date(2024, 1, 6), end_date=date(2024, 1, 6))


@pytest.mark.parametrize(
    "field",
    ["long_top_pct", "short_bottom_pct", "cvg_filter_pct", "dvol_top_pct", "spread_bottom_pct"],
)
def test_fractional_thresholds_reject_out_of_range(field):
    with pytest.raises(ValueError):
        make_contract_config(**{field: 1.5})


def test_invalid_short_structure_rejected():
    with pytest.raises(ValueError):
        make_contract_config(short_structure="butterfly")


def test_long_and_short_pct_sum_must_not_exceed_one():
    with pytest.raises(ValueError):
        make_contract_config(long_top_pct=0.6, short_bottom_pct=0.6)


# ---------------------------------------------------------------------------
# Sprint 003 Deliverable 1 — sizing-policy config fields + validation
# See docs/surface_engine_portfolio_metrics_design.md § S5 / § Config levers.
# ---------------------------------------------------------------------------


def test_sizing_mode_is_required_no_default():
    """HD Q8b: sizing_mode has no usable default — an unset run fails fast."""
    with pytest.raises(ValueError):
        make_contract_config(sizing_mode=None)


def test_invalid_sizing_mode_rejected():
    with pytest.raises(ValueError):
        make_contract_config(sizing_mode="optimizer")


def test_short_straddle_rejected_v1_defined_risk_only():
    """HD Q5: v1 short side is defined-risk only — no naked short straddle."""
    with pytest.raises(ValueError):
        make_contract_config(short_structure="straddle")


@pytest.mark.parametrize(
    "tier_a_mode",
    ["equal_premium", "equal_max_loss"],
)
def test_conceptual_tier_a_combos_accepted(tier_a_mode):
    """Each valid sizing_mode='conceptual' / tier_a_mode combo constructs."""
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode=tier_a_mode,
        tier_a_short_budget=10_000.0,
        tier_a_long_budget=10_000.0,
    )
    assert cfg.sizing_mode == "conceptual"
    assert cfg.tier_a_mode == tier_a_mode


def test_integer_lots_requires_tier_b_short_max_loss_budget():
    """Tier B requires a total short-side max-loss budget."""
    with pytest.raises(ValueError):
        make_contract_config(
            sizing_mode="integer_lots",
            tier_b_short_max_loss_budget=None,
        )


def test_integer_lots_accepted_with_tier_b_short_budget():
    """Tier B does not require Tier A budget fields."""
    cfg = make_contract_config(
        sizing_mode="integer_lots",
        tier_a_mode=None,
        tier_a_short_budget=None,
        tier_a_long_budget=None,
        tier_b_short_max_loss_budget=50_000.0,
    )
    assert cfg.sizing_mode == "integer_lots"
    assert cfg.tier_b_short_max_loss_budget == 50_000.0


def test_tier_b_short_max_loss_budget_must_be_positive():
    with pytest.raises(ValueError):
        make_contract_config(
            sizing_mode="integer_lots",
            tier_b_short_max_loss_budget=0.0,
        )


def test_conceptual_requires_tier_a_mode():
    with pytest.raises(ValueError):
        make_contract_config(sizing_mode="conceptual", tier_a_mode=None)


def test_conceptual_requires_positive_short_budget():
    with pytest.raises(ValueError):
        make_contract_config(
            sizing_mode="conceptual",
            tier_a_mode="equal_premium",
            tier_a_short_budget=0.0,
        )


def test_equal_premium_requires_long_budget():
    with pytest.raises(ValueError):
        make_contract_config(
            sizing_mode="conceptual",
            tier_a_mode="equal_premium",
            tier_a_short_budget=10_000.0,
            tier_a_long_budget=None,
        )


def test_equal_max_loss_long_budget_optional():
    """In equal_max_loss the long side is financed by collected short premium."""
    cfg = make_contract_config(
        sizing_mode="conceptual",
        tier_a_mode="equal_max_loss",
        tier_a_short_budget=10_000.0,
        tier_a_long_budget=None,
    )
    assert cfg.tier_a_long_budget is None


def test_contract_multiplier_defaults_to_100():
    cfg = make_contract_config()
    assert cfg.contract_multiplier == 100.0


def test_contract_multiplier_must_be_positive():
    with pytest.raises(ValueError):
        make_contract_config(contract_multiplier=0.0)


def test_deployable_capital_defaults_to_none():
    cfg = make_contract_config()
    assert cfg.deployable_capital is None


def test_deployable_capital_must_be_positive_when_set():
    with pytest.raises(ValueError):
        make_contract_config(deployable_capital=-1.0)


def test_deployable_capital_accepts_positive_value():
    cfg = make_contract_config(sizing_mode="integer_lots", deployable_capital=1_000_000.0)
    assert cfg.deployable_capital == 1_000_000.0
