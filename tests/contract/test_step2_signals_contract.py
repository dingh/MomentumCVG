"""
Contract: S2 — step2_score_signals (cross-sectional momentum + CVG filter).

Invariants pinned here:
  I1. Only tickers present in the universe AND dated == trade_date are scored.
  I2. Rows with NaN momentum or CVG are dropped (no NaN in output scores).
  I3. Long pool = top long_top_pct by momentum rank; short = bottom short_bottom_pct.
  I4. Long and short pools are disjoint (no ticker on both sides).
  I5. Output schema is exactly the six signal columns.

See docs/surface_engine_data_contract.md § S2.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.backtest.pipeline import (
    eligible_feature_cross_section,
    required_count_threshold,
    step2_score_signals,
    validate_feature_count_columns,
)
from tests.contract.conftest import SIGNALS_OUT_COLS, make_contract_config


def test_output_schema(features_four_tickers, universe_four_tickers):
    cfg = make_contract_config()
    out = step2_score_signals(date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg)
    assert list(out.columns) == SIGNALS_OUT_COLS


def test_long_short_assignment_and_disjoint(features_four_tickers, universe_four_tickers):
    # long_top_pct=0.25 → rank >= 0.75 → {A(1.0), B(0.75)}.
    # short_bottom_pct=0.5 → rank <= 0.5 → {C(0.5), D(0.25)}.
    cfg = make_contract_config(long_top_pct=0.25, short_bottom_pct=0.5)
    out = step2_score_signals(date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg)
    longs = set(out.loc[out["direction"] == "long", "ticker"])
    shorts = set(out.loc[out["direction"] == "short", "ticker"])
    assert longs == {"A", "B"}
    assert shorts == {"C", "D"}
    assert longs.isdisjoint(shorts)


def test_no_nan_scores(features_four_tickers, universe_four_tickers):
    cfg = make_contract_config()
    out = step2_score_signals(date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg)
    assert not out["signal_score"].isna().any()
    assert not out["cvg_score"].isna().any()


def test_respects_universe_membership(features_four_tickers):
    # Universe restricted to A, B → C, D must never appear even though they
    # exist in features.
    import pandas as pd

    universe = pd.DataFrame(
        {"ticker": ["A", "B"], "dvol_rank_pct": [1.0, 0.5], "spread_rank_pct": [1.0, 0.5]}
    )
    # Disjoint fractions over the 2-ticker universe (A=rank 1.0, B=rank 0.5).
    cfg = make_contract_config(long_top_pct=0.25, short_bottom_pct=0.5)
    out = step2_score_signals(date(2024, 1, 5), features_four_tickers, universe, cfg)
    assert set(out["ticker"]) <= {"A", "B"}


def test_nan_momentum_row_dropped(features_four_tickers, universe_four_tickers):
    feats = features_four_tickers.copy()
    feats.loc[feats["ticker"] == "A", "mom_42_8_mean"] = np.nan
    cfg = make_contract_config()
    out = step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)
    assert "A" not in set(out["ticker"])


def test_wrong_trade_date_yields_empty(features_four_tickers, universe_four_tickers):
    cfg = make_contract_config()
    out = step2_score_signals(date(2024, 1, 12), features_four_tickers, universe_four_tickers, cfg)
    assert out.empty
    assert list(out.columns) == SIGNALS_OUT_COLS


# =============================================================================
# Sprint 006 D2 — joint count eligibility / ceil rule
# =============================================================================

def test_required_count_ceil_for_42_8():
    assert required_count_threshold("mom_42_8_mean", 0.80) == 28


def test_joint_cvg_count_failure_excludes_before_ranking(
    features_four_tickers, universe_four_tickers
):
    feats = features_four_tickers.copy()
    feats.loc[feats["ticker"] == "A", "cvg_count_42_8"] = 10
    cfg = make_contract_config(
        min_count_pct=0.80,
        cvg_count_col="cvg_count_42_8",
        long_top_pct=0.25,
        short_bottom_pct=0.5,
    )
    out = step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)
    assert "A" not in set(out["ticker"])
    assert set(out["ticker"]) == {"B", "D"}


def test_joint_mom_count_failure_excludes_before_ranking(
    features_four_tickers, universe_four_tickers
):
    feats = features_four_tickers.copy()
    feats.loc[feats["ticker"] == "D", "mom_42_8_count"] = 10
    cfg = make_contract_config(
        min_count_pct=0.80,
        cvg_count_col="cvg_count_42_8",
        long_top_pct=0.25,
        short_bottom_pct=0.5,
    )
    out = step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)
    assert "D" not in set(out["ticker"])


def test_joint_both_counts_pass(features_four_tickers, universe_four_tickers):
    cfg = make_contract_config(
        min_count_pct=0.80,
        cvg_count_col="cvg_count_42_8",
        long_top_pct=0.25,
        short_bottom_pct=0.5,
    )
    out = step2_score_signals(date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg)
    assert set(out["ticker"]) == {"A", "B", "C", "D"}


def test_missing_cvg_count_col_hard_fails(features_four_tickers, universe_four_tickers):
    feats = features_four_tickers.drop(columns=["cvg_count_42_8"])
    cfg = make_contract_config(cvg_count_col="cvg_count_42_8", min_count_pct=0.80)
    with pytest.raises(ValueError, match="cvg_count_col"):
        step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)


def test_mom_only_path_ignores_absent_cvg_count(
    features_four_tickers, universe_four_tickers
):
    feats = features_four_tickers.drop(columns=["cvg_count_42_8"])
    cfg = make_contract_config(cvg_count_col=None, min_count_pct=0.80)
    out = step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)
    assert set(out["ticker"]) == {"A", "B", "C", "D"}


def test_validate_feature_columns_missing_hard_fails(features_four_tickers):
    feats = features_four_tickers.drop(columns=["cvg_count_42_8"])
    cfg = make_contract_config(cvg_count_col="cvg_count_42_8")
    with pytest.raises(ValueError, match="missing required configured columns"):
        validate_feature_count_columns(feats, cfg)


def test_eligible_helper_is_s2_pre_rank_set(features_four_tickers, universe_four_tickers):
    cfg = make_contract_config(
        min_count_pct=0.80,
        cvg_count_col="cvg_count_42_8",
        long_top_pct=0.25,
        short_bottom_pct=0.5,
    )
    eligible = eligible_feature_cross_section(
        date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg
    )
    out = step2_score_signals(
        date(2024, 1, 5), features_four_tickers, universe_four_tickers, cfg
    )
    assert set(out["ticker"]).issubset(set(eligible["ticker"]))
    assert set(eligible["ticker"]) == {"A", "B", "C", "D"}
    assert len(out) <= len(eligible)

    feats = features_four_tickers.copy()
    feats.loc[feats["ticker"] == "A", "cvg_count_42_8"] = 10
    eligible_excl = eligible_feature_cross_section(
        date(2024, 1, 5), feats, universe_four_tickers, cfg
    )
    out_excl = step2_score_signals(date(2024, 1, 5), feats, universe_four_tickers, cfg)
    assert "A" not in set(eligible_excl["ticker"])
    assert set(out_excl["ticker"]) == set(out_excl["ticker"]) & set(eligible_excl["ticker"])
    assert "A" not in set(out_excl["ticker"])
