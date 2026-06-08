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

from src.backtest.pipeline import step2_score_signals
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