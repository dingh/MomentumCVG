"""Oracle equivalence and calculator compatibility for the D2 transform.

Two claims are proven here that the semantic unit tests deliberately leave
alone:

1. The vectorized transform is the *same* arithmetic as the production
   ``build_straddle_from_surface`` plus ``settle`` path, which stays the single
   source of truth for the economics even though it is far too slow to run over
   a full history.
2. The emitted table drops straight into ``FeatureDataContext`` and runs through
   the real ``MomentumCalculator`` and ``CVGCalculator`` with no adapter,
   rename, or unit conversion anywhere.

Everything runs on synthetic surfaces, so this module never touches the
accepted snapshot.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.backtest.option_surface import (
    FillAssumption,
    OptionSurfaceDB,
    build_straddle_from_surface,
)
from src.features.base import FeatureDataContext
from src.features.cvg_calculator import CVGCalculator
from src.features.momentum_calculator import MomentumCalculator
from src.features.straddle_observations import (
    MAX_LEG_SPREAD_PCT,
    content_digest,
    transform_surface_frames,
)

ORACLE_TOLERANCE = 1e-9


def _surface_frames(cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build A1/A2 frames carrying every column the oracle's DB requires."""
    meta_rows: list[dict] = []
    quote_rows: list[dict] = []

    for case in cases:
        ticker = case["ticker"]
        entry_date = pd.Timestamp(case["entry_date"])
        expiry_date = entry_date + pd.Timedelta(days=7)
        strike = case["body_strike"]
        meta_rows.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "expiry_date": expiry_date,
                "dte_actual": 7.0,
                "entry_spot": strike,
                "exit_spot": case["exit_spot"],
                "body_strike": strike,
                "spot_move_pct": (case["exit_spot"] - strike) / strike * 100,
                "realized_volatility": case["realized_volatility"],
                "surface_valid": True,
                "failure_reason": None,
            }
        )
        for side in ("call", "put"):
            bid, ask = case[f"{side}_quote"]
            mid = (bid + ask) / 2
            quote_rows.append(
                {
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "side": side,
                    "is_body": True,
                    "is_otm": False,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pct": (ask - bid) / mid,
                    "iv": case[f"{side}_iv"],
                    "delta": 0.5 if side == "call" else -0.5,
                    "abs_delta": 0.5,
                    "gamma": 0.05,
                    "vega": 0.10,
                    "theta": -0.03,
                    "volume": 1000,
                    "open_interest": 5000,
                }
            )
    return pd.DataFrame(meta_rows), pd.DataFrame(quote_rows)


def _oracle_cases() -> list[dict]:
    """Hand-varied keys: different strikes, spreads, IVs, and payoff sides."""
    return [
        dict(ticker="AAA", entry_date="2024-01-05", body_strike=100.0, exit_spot=102.0,
             call_quote=(2.00, 2.40), put_quote=(1.80, 2.20), call_iv=0.20, put_iv=0.20,
             realized_volatility=0.18),
        dict(ticker="AAA", entry_date="2024-01-12", body_strike=100.0, exit_spot=100.0,
             call_quote=(2.00, 2.40), put_quote=(1.80, 2.20), call_iv=0.20, put_iv=0.20,
             realized_volatility=0.25),
        dict(ticker="BBB", entry_date="2024-01-05", body_strike=57.5, exit_spot=41.37,
             call_quote=(0.07, 0.13), put_quote=(1.11, 1.29), call_iv=0.41, put_iv=0.39,
             realized_volatility=0.63),
        dict(ticker="BBB", entry_date="2024-01-12", body_strike=250.0, exit_spot=311.09,
             call_quote=(9.85, 10.15), put_quote=(8.40, 9.60), call_iv=0.55, put_iv=0.55,
             realized_volatility=0.0),
        # A sub-penny bid: D2 adds no minimum-bid rule, matching the surface path.
        dict(ticker="CCC", entry_date="2024-01-05", body_strike=12.5, exit_spot=12.49,
             call_quote=(0.0025, 0.0035), put_quote=(0.31, 0.35), call_iv=0.88, put_iv=0.88,
             realized_volatility=1.02),
        dict(ticker="CCC", entry_date="2024-01-12", body_strike=1000.0, exit_spot=1234.56,
             call_quote=(31.10, 34.90), put_quote=(28.75, 29.05), call_iv=0.33, put_iv=0.33,
             realized_volatility=0.47),
    ]


def test_matches_build_straddle_from_surface():
    """T4: the vectorized economics reproduce the production builder exactly."""
    cases = _oracle_cases()
    meta_df, quotes_df = _surface_frames(cases)
    observations = transform_surface_frames(meta_df, quotes_df).set_index(
        ["ticker", "entry_date"]
    )

    surface_db = OptionSurfaceDB(meta_df, quotes_df)
    for case in cases:
        entry_date = pd.Timestamp(case["entry_date"])
        oracle = build_straddle_from_surface(
            surface_db,
            case["ticker"],
            entry_date.date(),
            direction="long",
            fill=FillAssumption.mid(),
            max_leg_spread_pct=MAX_LEG_SPREAD_PCT,
        )
        position = oracle.settle(Decimal(str(case["exit_spot"])))
        row = observations.loc[(case["ticker"], entry_date)]

        assert row["observation_status"] == "ok"
        assert row["entry_cost"] == pytest.approx(
            float(oracle.entry_cost), abs=ORACLE_TOLERANCE
        )
        assert row["exit_value"] == pytest.approx(
            float(position.exit_value), abs=ORACLE_TOLERANCE
        )
        assert row["pnl"] == pytest.approx(float(position.pnl), abs=ORACLE_TOLERANCE)
        assert row["return_pct"] == pytest.approx(
            position.pnl_pct * 100, abs=ORACLE_TOLERANCE
        )


def test_oracle_agrees_on_spread_rejection():
    """T7/T4: a leg past ``0.99`` is ineligible here and unbuildable in the oracle."""
    case = dict(
        ticker="AAA", entry_date="2024-01-05", body_strike=100.0, exit_spot=104.0,
        call_quote=(0.10, 5.00), put_quote=(1.80, 2.20), call_iv=0.20, put_iv=0.20,
        realized_volatility=0.30,
    )
    meta_df, quotes_df = _surface_frames([case])
    row = transform_surface_frames(meta_df, quotes_df).iloc[0]

    assert row["call_spread_pct"] > MAX_LEG_SPREAD_PCT
    assert row["observation_status"] == "body_spread_ineligible"
    with pytest.raises(ValueError, match="Missing tradeable body"):
        build_straddle_from_surface(
            OptionSurfaceDB(meta_df, quotes_df),
            case["ticker"],
            pd.Timestamp(case["entry_date"]).date(),
            direction="long",
            fill=FillAssumption.mid(),
            max_leg_spread_pct=MAX_LEG_SPREAD_PCT,
        )


# =============================================================================
# Calculator compatibility
# =============================================================================


def _dense_history_frames(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    weeks: int = 20,
    unavailable: tuple[tuple[str, int], ...] = (("BBB", 4), ("BBB", 5), ("CCC", 11)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A multi-ticker weekly grid with a few deliberately unavailable weeks."""
    entry_dates = pd.date_range("2024-01-05", periods=weeks, freq="7D")
    rng = np.random.default_rng(20260801)
    cases: list[dict] = []
    gaps = set(unavailable)

    for ticker in tickers:
        for index, entry_date in enumerate(entry_dates):
            strike = 100.0
            cases.append(
                dict(
                    ticker=ticker,
                    entry_date=entry_date,
                    body_strike=strike,
                    exit_spot=strike + float(rng.normal(0.0, 6.0)),
                    call_quote=(2.00, 2.40),
                    put_quote=(1.80, 2.20),
                    call_iv=0.20 + index * 0.001,
                    put_iv=0.20 + index * 0.001,
                    realized_volatility=float(rng.uniform(0.05, 0.45)),
                )
            )
    meta_df, quotes_df = _surface_frames(cases)

    # Turn the chosen weeks into ordinary unavailable observations: the row
    # stays, only the economics go away.
    for ticker, index in gaps:
        entry_date = entry_dates[index]
        key = (meta_df["ticker"] == ticker) & (meta_df["entry_date"] == entry_date)
        meta_df.loc[key, "surface_valid"] = False
        meta_df.loc[key, "failure_reason"] = "no_spot_price"
        quote_key = (quotes_df["ticker"] == ticker) & (quotes_df["entry_date"] == entry_date)
        quotes_df = quotes_df[~quote_key]

    return meta_df, quotes_df.reset_index(drop=True)


@pytest.fixture
def observations() -> pd.DataFrame:
    return transform_surface_frames(*_dense_history_frames())


def test_momentum_calculator_consumes_output(observations):
    """T17: the real MomentumCalculator runs on the artifact with no adapter."""
    context = FeatureDataContext(straddle_history=observations)
    calculator = MomentumCalculator(windows=[(8, 2)], min_periods=3)
    target_date = observations["entry_date"].max()
    tickers = sorted(observations["ticker"].unique())

    single = calculator.calculate(context, target_date, tickers).set_index("ticker")
    bulk = calculator.calculate_bulk(
        context, start_date=target_date, end_date=target_date
    ).set_index("ticker")

    for ticker in tickers:
        for feature in calculator.feature_names:
            assert single.loc[ticker, feature] == pytest.approx(
                bulk.loc[ticker, feature], rel=1e-12
            )

    # Null-return rows occupy a window slot without being counted, so row
    # positions still advance across them.
    history = observations[observations["ticker"] == "BBB"].reset_index(drop=True)
    target_position = history.index[history["entry_date"] == target_date][0]
    window = history.iloc[target_position - 8 : target_position - 2 + 1]
    assert window["return_pct"].isna().sum() == 0  # gaps sit outside this window
    assert single.loc["BBB", "mom_8_2_count"] == window["return_pct"].notna().sum()

    early_date = observations["entry_date"].unique()[8]
    early = calculator.calculate(context, early_date, ["BBB"]).set_index("ticker")
    early_history = history[history["entry_date"] <= early_date]
    early_position = len(early_history) - 1
    early_window = early_history.iloc[early_position - 8 : early_position - 2 + 1]
    assert early_window["return_pct"].isna().sum() == 2
    assert early.loc["BBB", "mom_8_2_count"] == early_window["return_pct"].notna().sum()


def test_cvg_calculator_consumes_output(observations):
    """T18: CVG uses the emitted vol_gap directly and derives the same values."""
    calculator = CVGCalculator(windows=[(8, 2)], min_periods=3)
    target_date = observations["entry_date"].max()
    tickers = sorted(observations["ticker"].unique())

    emitted = calculator.calculate(
        FeatureDataContext(straddle_history=observations), target_date, tickers
    )
    # Dropping vol_gap forces _resolve_vol_gap_col to derive it from the two
    # components; the emitted column must be exactly that difference.
    derived_source = observations.drop(columns=["vol_gap"])
    derived = calculator.calculate(
        FeatureDataContext(straddle_history=derived_source), target_date, tickers
    )

    pd.testing.assert_frame_equal(emitted, derived)
    assert emitted[calculator.feature_names].notna().to_numpy().any()

    both_present = observations["entry_iv"].notna() & observations["realized_volatility"].notna()
    np.testing.assert_allclose(
        observations.loc[both_present, "vol_gap"].to_numpy(),
        (observations["realized_volatility"] - observations["entry_iv"])[both_present].to_numpy(),
        atol=1e-15,
    )


def test_entry_date_is_datetime64_and_index_unique(observations, tmp_path):
    """T19: the dtype contract survives a Parquet round-trip."""
    path = tmp_path / "straddle_observations_weekly.parquet"
    observations.to_parquet(path, index=False, compression="snappy")
    reloaded = pd.read_parquet(path)

    # The unit matters, not just the kind: pandas infers second resolution from
    # A1's date32 columns, and a reader that gets a different unit back than the
    # writer emitted would compute a different content digest for the same data.
    for column in ("entry_date", "expiry_date"):
        assert observations[column].dtype == np.dtype("datetime64[ns]")
        assert reloaded[column].dtype == np.dtype("datetime64[ns]")
    assert isinstance(reloaded.index, pd.RangeIndex)
    assert reloaded.index.is_unique
    assert all(isinstance(value, str) and value == value.upper() for value in reloaded["ticker"])
    for column in ("return_pct", "entry_iv", "realized_volatility", "vol_gap"):
        assert reloaded[column].dtype == np.float64
    assert content_digest(reloaded) == content_digest(observations)

    # MomentumCalculator.calculate compares entry_date to the target with no
    # dtype coercion, so an object-of-date column would raise here.
    calculator = MomentumCalculator(windows=[(8, 2)], min_periods=3)
    features = calculator.calculate(
        FeatureDataContext(straddle_history=reloaded),
        datetime(2024, 5, 3),
        ["AAA"],
    )
    assert len(features) == 1
