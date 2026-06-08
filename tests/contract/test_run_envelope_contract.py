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
