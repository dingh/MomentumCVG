"""Unit tests for the surface -> straddle observation transform (Sprint 005 D2).

Every expected number is derived by hand from one tiny synthetic surface, using
the same parameters as ``test_option_surface_straddle.py``:

    body_strike = 100.00
    call: bid 2.00 / ask 2.40      put: bid 1.80 / ask 2.20
    exit_spot = 102.00             realized_volatility = 0.18, leg iv = 0.20

    entry_cost = 2.20 + 2.00 = 4.20
    exit_value = max(102 - 100, 0) + max(100 - 102, 0) = 2.00
    pnl        = 2.00 - 4.20 = -2.20
    return_pct = -2.20 / 4.20 * 100 = -52.380952...
    entry_iv   = (0.20 + 0.20) / 2 = 0.20
    vol_gap    = 0.18 - 0.20 = -0.02

These tests never read the accepted snapshot; the whole unit layer is
self-contained.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.input_snapshot import InputSnapshotManifest, write_manifest
from src.features import straddle_observations as so
from src.features.straddle_observations import (
    LINEAGE_FILENAME,
    OBSERVATION_COLUMNS,
    OBSERVATIONS_FILENAME,
    TRANSFORM_CONFIG_VERSION,
    StraddleObservationStructuralError,
    a1_key_digest,
    build_observations,
    content_digest,
    join_body_legs,
    load_surface_frames,
    observation_coverage,
    publish_observations,
    resolve_surface_inputs,
    transform_surface_frames,
    validate_output_contract,
)

TICKER = "TEST"
ENTRY_DATE = pd.Timestamp("2024-01-05")
EXPIRY_DATE = pd.Timestamp("2024-01-12")
BODY_STRIKE = 100.0
ENTRY_SPOT = 100.0
EXIT_SPOT = 102.0
CALL_BID, CALL_ASK = 2.00, 2.40
PUT_BID, PUT_ASK = 1.80, 2.20
LEG_IV = 0.20
REALIZED_VOL = 0.18

EXPECTED_ENTRY_COST = 4.20
EXPECTED_EXIT_VALUE = 2.00
EXPECTED_PNL = -2.20
EXPECTED_RETURN_PCT = -2.20 / 4.20 * 100
EXPECTED_ENTRY_IV = 0.20
EXPECTED_VOL_GAP = REALIZED_VOL - EXPECTED_ENTRY_IV

TOLERANCE = 1e-12
FAKE_REPO_SHA = "a1b2c3d4" + "0" * 32
CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_straddle_observations.py"


# =============================================================================
# Synthetic surface builders
# =============================================================================


def meta_row(
    ticker: str = TICKER,
    entry_date: pd.Timestamp = ENTRY_DATE,
    *,
    surface_valid: bool = True,
    failure_reason: str | None = None,
    expiry_date: pd.Timestamp = EXPIRY_DATE,
    exit_spot: float = EXIT_SPOT,
    body_strike: float = BODY_STRIKE,
    realized_volatility: float | None = REALIZED_VOL,
) -> dict:
    """One synthetic A1 row."""
    return {
        "ticker": ticker,
        "entry_date": entry_date,
        "expiry_date": expiry_date,
        "dte_actual": 7.0,
        "entry_spot": ENTRY_SPOT,
        "exit_spot": exit_spot,
        "body_strike": body_strike,
        "spot_move_pct": (exit_spot - ENTRY_SPOT) / ENTRY_SPOT * 100,
        "realized_volatility": realized_volatility,
        "surface_valid": surface_valid,
        "failure_reason": failure_reason,
    }


def body_quote(
    side: str,
    *,
    ticker: str = TICKER,
    entry_date: pd.Timestamp = ENTRY_DATE,
    expiry_date: pd.Timestamp = EXPIRY_DATE,
    strike: float = BODY_STRIKE,
    bid: float | None = None,
    ask: float | None = None,
    iv: float | None = LEG_IV,
    spread_pct: float | None = None,
) -> dict:
    """One synthetic A2 body-leg row."""
    if bid is None:
        bid = CALL_BID if side == "call" else PUT_BID
    if ask is None:
        ask = CALL_ASK if side == "call" else PUT_ASK
    if spread_pct is None:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid else np.nan
    return {
        "ticker": ticker,
        "entry_date": entry_date,
        "expiry_date": expiry_date,
        "strike": strike,
        "side": side,
        "is_body": True,
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "iv": iv,
    }


def body_pair(**kwargs) -> list[dict]:
    """The call and put legs of one synthetic body straddle."""
    return [body_quote("call", **kwargs), body_quote("put", **kwargs)]


def frames(meta_rows: list[dict], quote_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build A1/A2 frames shaped the way ``load_surface_frames`` returns them."""
    template = body_quote("call")
    quotes_df = pd.DataFrame(quote_rows or [template], columns=list(template))
    if not quote_rows:
        # Slice rather than build empty so the join key dtypes stay correct.
        quotes_df = quotes_df.iloc[0:0]
    return pd.DataFrame(meta_rows), quotes_df


def default_surface() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One valid, fully priceable key."""
    return frames([meta_row()], body_pair())


def only_row(observations: pd.DataFrame) -> pd.Series:
    assert len(observations) == 1
    return observations.iloc[0]


# =============================================================================
# T1-T3 — key grid and literal economics
# =============================================================================


def test_output_keys_match_a1_exactly():
    """T1: one row per A1 key, no duplicates, sorted, nothing added or dropped."""
    weeks = pd.date_range("2024-01-05", periods=4, freq="7D")
    meta_rows = [
        meta_row(ticker, week, surface_valid=(index % 2 == 0),
                 failure_reason=None if index % 2 == 0 else "no_spot_price")
        for ticker in ("BBB", "AAA")
        for index, week in enumerate(weeks)
    ]
    quote_rows = [
        quote
        for ticker in ("BBB", "AAA")
        for index, week in enumerate(weeks)
        if index % 2 == 0
        for quote in body_pair(ticker=ticker, entry_date=week)
    ]
    meta_df, quotes_df = frames(meta_rows, quote_rows)

    observations = transform_surface_frames(meta_df, quotes_df)

    assert len(observations) == len(meta_df) == 8
    output_keys = set(zip(observations["ticker"], observations["entry_date"]))
    assert output_keys == set(zip(meta_df["ticker"], meta_df["entry_date"]))
    assert not observations.duplicated(["ticker", "entry_date"]).any()
    assert observations["ticker"].tolist() == ["AAA"] * 4 + ["BBB"] * 4
    sorted_copy = observations.sort_values(["ticker", "entry_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(observations, sorted_copy)


def test_literal_long_straddle_values():
    """T2: hand-calculated midpoint long-straddle economics and volatilities."""
    observations = transform_surface_frames(*default_surface())
    row = only_row(observations)

    assert row["observation_status"] == "ok"
    assert pd.isna(row["missing_reason"])
    assert row["entry_cost"] == pytest.approx(EXPECTED_ENTRY_COST, abs=TOLERANCE)
    assert row["exit_value"] == pytest.approx(EXPECTED_EXIT_VALUE, abs=TOLERANCE)
    assert row["pnl"] == pytest.approx(EXPECTED_PNL, abs=TOLERANCE)
    assert row["return_pct"] == pytest.approx(EXPECTED_RETURN_PCT, abs=TOLERANCE)
    assert row["entry_iv"] == pytest.approx(EXPECTED_ENTRY_IV, abs=TOLERANCE)
    assert row["realized_volatility"] == pytest.approx(REALIZED_VOL, abs=TOLERANCE)
    assert row["vol_gap"] == pytest.approx(EXPECTED_VOL_GAP, abs=TOLERANCE)


@pytest.mark.parametrize(
    ("exit_spot", "expected_return_pct"),
    [
        (BODY_STRIKE, -100.0),  # both legs expire worthless: the hard floor
        (120.0, (20.0 - EXPECTED_ENTRY_COST) / EXPECTED_ENTRY_COST * 100),
        (80.0, (20.0 - EXPECTED_ENTRY_COST) / EXPECTED_ENTRY_COST * 100),
    ],
)
def test_return_pct_floor_and_units(exit_spot, expected_return_pct):
    """T3: returns are percentage points, floored at -100 for a long straddle."""
    meta_df, quotes_df = frames([meta_row(exit_spot=exit_spot)], body_pair())
    row = only_row(transform_surface_frames(meta_df, quotes_df))
    assert row["return_pct"] == pytest.approx(expected_return_pct, abs=TOLERANCE)
    assert row["return_pct"] >= -100.0


# =============================================================================
# T5-T6 — unavailable weeks stay as rows
# =============================================================================


def test_surface_invalid_rows_are_rows_not_gaps():
    """T5: an invalid A1 row keeps its key, its A1 values, and its own reason."""
    meta_df, quotes_df = frames(
        [meta_row(surface_valid=False, failure_reason="target_weekly_expiry_not_listed")],
        [],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "surface_invalid"
    assert row["missing_reason"] == "target_weekly_expiry_not_listed"
    assert not row["surface_valid"]
    # A1 passthroughs survive; body-leg and derived straddle fields do not.
    assert row["expiry_date"] == EXPIRY_DATE
    assert row["exit_spot"] == pytest.approx(EXIT_SPOT)
    assert row["body_strike"] == pytest.approx(BODY_STRIKE)
    assert row["realized_volatility"] == pytest.approx(REALIZED_VOL)
    for column in ("call_bid", "call_ask", "put_bid", "put_ask", "call_iv", "put_iv"):
        assert pd.isna(row[column])
    for column in ("entry_cost", "exit_value", "pnl", "return_pct", "entry_iv", "vol_gap"):
        assert pd.isna(row[column])


def test_invalid_row_partial_quotes_are_ignored():
    """Partial A2 rows on an invalid A1 key are never harvested."""
    meta_df, quotes_df = frames(
        [meta_row(surface_valid=False, failure_reason="target_weekly_body_not_quotable")],
        body_pair(),
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "surface_invalid"
    assert pd.isna(row["call_bid"])
    assert pd.isna(row["entry_cost"])
    assert pd.isna(row["entry_iv"])


def test_surface_invalid_without_reason_gets_placeholder():
    """A null A1 failure_reason still yields a non-null missing_reason."""
    meta_df, quotes_df = frames([meta_row(surface_valid=False, failure_reason=None)], [])
    row = only_row(transform_surface_frames(meta_df, quotes_df))
    assert row["missing_reason"] == "surface_invalid_reason_missing"


def test_scheduled_week_never_dropped_or_backfilled():
    """T6: an unavailable middle week keeps its row; neighbours are untouched."""
    weeks = pd.date_range("2024-01-05", periods=3, freq="7D")
    meta_rows = [
        meta_row(entry_date=weeks[0], exit_spot=104.0),
        meta_row(entry_date=weeks[1], surface_valid=False, failure_reason="no_spot_price"),
        meta_row(entry_date=weeks[2], exit_spot=96.0),
    ]
    quote_rows = body_pair(entry_date=weeks[0]) + body_pair(entry_date=weeks[2])
    observations = transform_surface_frames(*frames(meta_rows, quote_rows))

    assert observations["entry_date"].tolist() == list(weeks)
    assert observations["observation_status"].tolist() == ["ok", "surface_invalid", "ok"]
    # Nothing was copied forward into the gap, and the neighbours differ.
    assert pd.isna(observations.loc[1, "return_pct"])
    assert observations.loc[0, "exit_value"] == pytest.approx(4.0)
    assert observations.loc[2, "exit_value"] == pytest.approx(4.0)
    assert observations.loc[0, "exit_spot"] == pytest.approx(104.0)
    assert observations.loc[2, "exit_spot"] == pytest.approx(96.0)


# =============================================================================
# T7-T10 — row-level eligibility, precedence, and preserved information
# =============================================================================


@pytest.mark.parametrize(
    ("call_spread", "put_spread", "expected_status", "expected_reason"),
    [
        (0.99, 0.99, "ok", None),
        (0.990001, 0.50, "body_spread_ineligible", "body_spread_above_threshold"),
        (0.50, 0.990001, "body_spread_ineligible", "body_spread_above_threshold"),
        (-0.10, 0.50, "ok", None),  # crossed quote: negative spread passes (R-12)
    ],
)
def test_spread_threshold_boundary(call_spread, put_spread, expected_status, expected_reason):
    """T7: the per-leg rule is ``spread_pct <= 0.99``, inclusive at the boundary."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [
            body_quote("call", spread_pct=call_spread),
            body_quote("put", spread_pct=put_spread),
        ],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == expected_status
    if expected_reason is None:
        assert pd.isna(row["missing_reason"])
    else:
        assert row["missing_reason"] == expected_reason


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("field", ["bid", "ask"])
def test_unusable_quote_is_row_level_not_fatal(field, bad_value):
    """T8: a non-positive or non-finite body quote is a row, not an exception."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [body_quote("call", **{field: bad_value}, spread_pct=0.20), body_quote("put")],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "body_quote_unusable"
    assert row["missing_reason"] == "body_quote_not_positive_finite"
    for column in ("entry_cost", "exit_value", "pnl", "return_pct", "entry_iv", "vol_gap"):
        assert pd.isna(row[column])
    # Raw leg values and A1 realized volatility remain auditable.
    assert row["call_iv"] == pytest.approx(LEG_IV)
    assert row["realized_volatility"] == pytest.approx(REALIZED_VOL)


def test_unusable_quote_outranks_unavailable_spread():
    """T8: quote usability is decided before spread eligibility."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [body_quote("call", bid=0.0, spread_pct=np.nan), body_quote("put")],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))
    assert row["observation_status"] == "body_quote_unusable"


@pytest.mark.parametrize("bad_spread", [np.nan, np.inf, -np.inf])
def test_unavailable_spread_is_ineligible_not_fatal(bad_spread):
    """T8: a spread that cannot confirm eligibility yields its own reason."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [body_quote("call", spread_pct=bad_spread), body_quote("put")],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "body_spread_ineligible"
    assert row["missing_reason"] == "body_spread_unavailable"


@pytest.mark.parametrize("bad_iv", [None, 0.0, -0.1, np.nan, np.inf])
def test_unusable_leg_iv_preserves_return(bad_iv):
    """T8b: an unusable leg IV nulls the vol fields but never the return."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [body_quote("call", iv=bad_iv), body_quote("put")],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "ok"
    assert row["return_pct"] == pytest.approx(EXPECTED_RETURN_PCT, abs=TOLERANCE)
    assert row["entry_cost"] == pytest.approx(EXPECTED_ENTRY_COST, abs=TOLERANCE)
    assert row["put_iv"] == pytest.approx(LEG_IV)
    assert pd.isna(row["entry_iv"])
    assert pd.isna(row["vol_gap"])


@pytest.mark.parametrize("bad_rv", [None, -0.1, np.nan, np.inf])
def test_unusable_rv_preserves_return(bad_rv):
    """T9: an unusable realized volatility nulls only RV and the gap."""
    meta_df, quotes_df = frames([meta_row(realized_volatility=bad_rv)], body_pair())
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "ok"
    assert row["return_pct"] == pytest.approx(EXPECTED_RETURN_PCT, abs=TOLERANCE)
    assert row["entry_iv"] == pytest.approx(EXPECTED_ENTRY_IV, abs=TOLERANCE)
    assert pd.isna(row["realized_volatility"])
    assert pd.isna(row["vol_gap"])


def test_zero_realized_volatility_is_preserved():
    """RV of exactly 0.0 is a real observation, not a missing value."""
    meta_df, quotes_df = frames([meta_row(realized_volatility=0.0)], body_pair())
    row = only_row(transform_surface_frames(meta_df, quotes_df))
    assert row["realized_volatility"] == 0.0
    assert row["vol_gap"] == pytest.approx(-EXPECTED_ENTRY_IV, abs=TOLERANCE)


def test_spread_ineligible_preserves_volatility():
    """T10: frozen decision D-1 — trade economics null, volatilities retained."""
    meta_df, quotes_df = frames(
        [meta_row()],
        [body_quote("call", spread_pct=1.5), body_quote("put", spread_pct=0.20)],
    )
    row = only_row(transform_surface_frames(meta_df, quotes_df))

    assert row["observation_status"] == "body_spread_ineligible"
    assert row["missing_reason"] == "body_spread_above_threshold"
    for column in ("entry_cost", "exit_value", "pnl", "return_pct"):
        assert pd.isna(row[column])
    assert row["entry_iv"] == pytest.approx(EXPECTED_ENTRY_IV, abs=TOLERANCE)
    assert row["realized_volatility"] == pytest.approx(REALIZED_VOL, abs=TOLERANCE)
    assert row["vol_gap"] == pytest.approx(EXPECTED_VOL_GAP, abs=TOLERANCE)


# =============================================================================
# T11-T16 — structural failures
# =============================================================================


@pytest.mark.parametrize(
    ("quote_rows", "expected_fragment"),
    [
        ([body_quote("put")], "no body call"),
        ([body_quote("call")], "no body put"),
        (body_pair() + [body_quote("call")], "duplicate body call"),
        (body_pair() + [body_quote("put")], "duplicate body put"),
        (
            [body_quote("call", strike=105.0), body_quote("put")],
            "body call strike disagrees",
        ),
        (
            [body_quote("call"), body_quote("put", expiry_date=pd.Timestamp("2024-01-19"))],
            "body put expiry disagrees",
        ),
    ],
)
def test_body_leg_contradictions_are_structural(quote_rows, expected_fragment):
    """T11-T14: A1/A2 disagreement about the body legs fails the run."""
    meta_df, quotes_df = frames([meta_row()], quote_rows)
    with pytest.raises(StraddleObservationStructuralError, match=expected_fragment):
        transform_surface_frames(meta_df, quotes_df)


def test_structural_error_names_the_key():
    """The aborting error identifies which key failed."""
    meta_df, quotes_df = frames([meta_row()], [body_quote("put")])
    with pytest.raises(StraddleObservationStructuralError, match=r"TEST@2024-01-05"):
        transform_surface_frames(meta_df, quotes_df)


def test_duplicate_a1_key_violates_key_postcondition():
    """T15: a duplicated A1 key is caught by the output key post-condition."""
    meta_df, quotes_df = frames([meta_row(), meta_row()], body_pair())
    with pytest.raises(StraddleObservationStructuralError, match="duplicate"):
        transform_surface_frames(meta_df, quotes_df)


def test_structural_errors_reported_together():
    """T16: several distinct violations produce one aggregated error."""
    other = pd.Timestamp("2024-01-12")
    third = pd.Timestamp("2024-01-19")
    meta_rows = [meta_row(), meta_row(entry_date=other), meta_row(entry_date=third)]
    quote_rows = (
        [body_quote("call")]  # missing put
        + body_pair(entry_date=other)
        + [body_quote("call", entry_date=other)]  # duplicate call
        + [body_quote("call", entry_date=third, strike=101.0), body_quote("put", entry_date=third)]
    )
    meta_df, quotes_df = frames(meta_rows, quote_rows)

    with pytest.raises(StraddleObservationStructuralError) as excinfo:
        transform_surface_frames(meta_df, quotes_df)

    message = str(excinfo.value)
    assert "no body put" in message
    assert "duplicate body call" in message
    assert "body call strike disagrees" in message
    assert "1 key(s)" in message


def test_output_status_and_dependent_null_contract():
    """T16b: a hand-corrupted frame is rejected before publication."""
    meta_df, quotes_df = default_surface()
    observations = transform_surface_frames(meta_df, quotes_df)

    corrupted = observations.copy()
    corrupted.loc[0, "return_pct"] = np.nan
    with pytest.raises(StraddleObservationStructuralError, match="return_pct must be non-null"):
        validate_output_contract(corrupted, meta_df)

    corrupted = observations.copy()
    corrupted.loc[0, "missing_reason"] = "body_spread_above_threshold"
    with pytest.raises(StraddleObservationStructuralError, match="missing_reason must be null"):
        validate_output_contract(corrupted, meta_df)

    corrupted = observations.copy()
    corrupted.loc[0, "vol_gap"] = np.nan
    with pytest.raises(StraddleObservationStructuralError, match="vol_gap must be populated"):
        validate_output_contract(corrupted, meta_df)

    corrupted = observations.copy()
    corrupted.loc[0, "entry_cost"] = -1.0
    with pytest.raises(StraddleObservationStructuralError, match="entry_cost must be positive"):
        validate_output_contract(corrupted, meta_df)


# =============================================================================
# T23-T24 — scope and frozen rules
# =============================================================================


def test_output_contains_no_feature_columns():
    """T23: D2 emits no momentum, CVG, ranking, or eligibility column."""
    observations = transform_surface_frames(*default_surface())
    assert list(observations.columns) == list(OBSERVATION_COLUMNS)
    forbidden_prefixes = ("mom_", "cvg_", "dvg_", "cgap_", "pct_pos", "pct_neg", "volgap_")
    for column in observations.columns:
        assert not column.startswith(forbidden_prefixes)
        assert "rank" not in column
        assert "eligib" not in column
    # No windowing parameter leaks into the transform config.
    config_text = json.dumps(so.TRANSFORM_CONFIG)
    for token in ("window", "min_periods", "lag"):
        assert token not in config_text
    # iv_rv_spread has the opposite sign to vol_gap and is deliberately absent.
    assert "iv_rv_spread" not in observations.columns


def test_frozen_rules_are_not_overridable():
    """T24: the builder takes no economic-policy argument and the CLI has no flag."""
    import inspect
    import re

    assert set(inspect.signature(build_observations).parameters) == {"joined"}

    cli_flags = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', CLI_PATH.read_text(encoding="utf-8")))
    assert cli_flags == {"--snapshot-root", "--output-root", "--dry-run"}

    assert so.SPREAD_INELIGIBLE_VOLATILITY_RULE == "preserve"
    assert so.MAX_LEG_SPREAD_PCT == 0.99
    assert so.VOL_GAP_RULE == "realized_volatility_minus_entry_iv"


# =============================================================================
# Snapshot-backed fixture for I/O, lineage, and publication
# =============================================================================

SNAPSHOT_ID = "abcdef0123456789"
BUILD_ID = "20240105T000000000000Z_deadbeef"


def _write_surface_parquet(path, frame: pd.DataFrame) -> None:
    """Write a synthetic A1/A2 file with ``date32`` date columns, as A1/A2 are."""
    out = frame.copy()
    for column in ("entry_date", "expiry_date"):
        out[column] = pd.to_datetime(out[column]).dt.date
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def _synthetic_snapshot_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small multi-ticker grid covering every status the transform can emit."""
    weeks = pd.date_range("2024-01-05", periods=4, freq="7D")
    meta_rows: list[dict] = []
    quote_rows: list[dict] = []

    for index, week in enumerate(weeks):
        expiry = week + pd.Timedelta(days=7)
        # AAA: ok, invalid, ok, spread-ineligible
        if index == 1:
            meta_rows.append(
                meta_row("AAA", week, expiry_date=expiry, surface_valid=False,
                         failure_reason="target_weekly_expiry_not_listed")
            )
        else:
            meta_rows.append(meta_row("AAA", week, expiry_date=expiry, exit_spot=101.0 + index))
            spread = 1.5 if index == 3 else None
            quote_rows.extend(
                body_pair(ticker="AAA", entry_date=week, expiry_date=expiry, spread_pct=spread)
            )
        # BBB: ok, ok, ok with unusable RV, invalid
        if index == 3:
            meta_rows.append(
                meta_row("BBB", week, expiry_date=expiry, surface_valid=False,
                         failure_reason="no_spot_price")
            )
        else:
            meta_rows.append(
                meta_row(
                    "BBB",
                    week,
                    expiry_date=expiry,
                    exit_spot=99.0 + index,
                    realized_volatility=None if index == 2 else REALIZED_VOL,
                )
            )
            quote_rows.extend(body_pair(ticker="BBB", entry_date=week, expiry_date=expiry))

    return pd.DataFrame(meta_rows), pd.DataFrame(quote_rows)


@pytest.fixture
def snapshot_root(tmp_path):
    """An accepted synthetic snapshot: manifest plus A1/A2 parquet files."""
    root = tmp_path / "snapshots" / BUILD_ID
    meta_rel = "cache/surface/option_surface_meta.parquet"
    quotes_rel = "cache/surface/option_surface_quotes.parquet"

    meta_df, quotes_df = _synthetic_snapshot_frames()
    _write_surface_parquet(root / meta_rel, meta_df)
    _write_surface_parquet(root / quotes_rel, quotes_df)

    manifest = InputSnapshotManifest(
        schema_version="1",
        snapshot_id=SNAPSHOT_ID,
        build_id=BUILD_ID,
        created_at_utc=datetime(2024, 1, 5, tzinfo=timezone.utc),
        as_of_requested=date(2024, 1, 5),
        as_of_resolved_trading_day=date(2024, 1, 5),
        data_source="synthetic",
        cache_dir=str(root),
        artifacts={
            "option_surface_meta": meta_rel,
            "option_surface_quotes": quotes_rel,
        },
        params={"surface_actual_a1_key_digest": a1_key_digest(meta_df)},
        reports={},
        overall_status="WARN",
        blocking_failures=[],
        notes=[],
        production_accepted=True,
    )
    write_manifest(root / "manifests" / f"input_snapshot_{SNAPSHOT_ID}.json", manifest)
    return root


@pytest.fixture(autouse=True)
def committed_repo(monkeypatch):
    """Publication demands a clean, committed tree; mock that git state.

    Only the git plumbing is mocked, so every publishing test still runs the
    real ``resolve_publication_repo_sha`` guard.
    """
    monkeypatch.setattr(so, "_git_output", _fake_git(FAKE_REPO_SHA, ""))
    return FAKE_REPO_SHA


def _fake_git(head: str, pending: str):
    """Return a ``_git_output`` stand-in reporting one HEAD and tree state."""

    def run(*args: str) -> str:
        return {("rev-parse", "HEAD"): head, ("status", "--porcelain"): pending}[args]

    return run


def _run_pipeline(root, destination):
    """Resolve, load, transform, and publish — the CLI's production path."""
    inputs = resolve_surface_inputs(root)
    meta_df, quotes_df = load_surface_frames(inputs)
    observations = transform_surface_frames(meta_df, quotes_df)
    return observations, publish_observations(
        observations,
        inputs=inputs,
        destination_dir=destination,
        meta_row_count=len(meta_df),
        quote_row_count=len(quotes_df),
    )


def _patch_manifest(root, **changes):
    path = next((root / "manifests").glob("input_snapshot_*.json"))
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.update(changes)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# =============================================================================
# T20-T22, T25-T26 — determinism, lineage, guards, publication
# =============================================================================


def test_transform_is_deterministic(snapshot_root):
    """T20: two runs over the same input agree on content, order, and columns."""
    inputs = resolve_surface_inputs(snapshot_root)
    first = transform_surface_frames(*load_surface_frames(inputs))
    second = transform_surface_frames(*load_surface_frames(inputs))

    assert content_digest(first) == content_digest(second)
    assert list(first.columns) == list(second.columns) == list(OBSERVATION_COLUMNS)
    pd.testing.assert_frame_equal(first, second)


def test_lineage_receipt_contents(snapshot_root, tmp_path):
    """T21: the receipt records identity, sources, key proof, config, coverage."""
    destination = tmp_path / "derived" / SNAPSHOT_ID
    observations, result = _run_pipeline(snapshot_root, destination)
    receipt = result.receipt

    assert result.written is True
    assert receipt["snapshot_id"] == SNAPSHOT_ID
    assert receipt["build_id"] == BUILD_ID
    assert receipt["production_accepted"] is True
    assert receipt["a1_key_digest_matches_manifest"] is True
    assert receipt["a1_key_digest"] == receipt["manifest_a1_key_digest"]
    for artifact in ("option_surface_meta", "option_surface_quotes"):
        assert len(receipt["sources"][artifact]["sha256"]) == 64
    assert receipt["transform_config"] == so.TRANSFORM_CONFIG
    assert receipt["output"]["content_digest"] == content_digest(observations)
    assert receipt["output"]["row_count"] == len(observations)
    assert receipt["output"]["column_order"] == list(OBSERVATION_COLUMNS)
    assert receipt["coverage"] == observation_coverage(observations)

    published = pd.read_parquet(result.observations_path)
    assert receipt["coverage"]["status_counts"] == {
        status: int(count)
        for status, count in published["observation_status"].value_counts().sort_index().items()
    }


def test_frozen_rules_are_recorded_in_the_receipt(snapshot_root, tmp_path):
    """T24: the receipt carries the module's frozen economic constants."""
    _, result = _run_pipeline(snapshot_root, tmp_path / "derived" / SNAPSHOT_ID)
    config = result.receipt["transform_config"]

    assert config["direction"] == "long"
    assert config["fill"] == "mid"
    assert config["max_leg_spread_pct"] == 0.99
    assert config["entry_iv_rule"] == "mean_body_call_put_iv"
    assert config["vol_gap_rule"] == "realized_volatility_minus_entry_iv"
    assert config["spread_ineligible_volatility_rule"] == "preserve"
    assert config["return_pct_units"] == "percentage_points"
    assert config["volatility_units"] == "annualized_decimal"
    assert config["transform_config_version"] == TRANSFORM_CONFIG_VERSION


def test_coverage_counts_match_the_emitted_frame(snapshot_root):
    """Coverage is an honest summary of the frame it describes."""
    inputs = resolve_surface_inputs(snapshot_root)
    observations = transform_surface_frames(*load_surface_frames(inputs))
    coverage = observation_coverage(observations)

    assert coverage["row_count"] == 8
    assert coverage["key_count"] == 8
    assert coverage["status_counts"] == {
        "body_spread_ineligible": 1,
        "ok": 5,
        "surface_invalid": 2,
    }
    assert coverage["non_null_counts"]["return_pct"] == 5
    # The spread-ineligible row keeps its IV under D-1; the null-RV row does not
    # get a gap.
    assert coverage["non_null_counts"]["entry_iv"] == 6
    assert coverage["non_null_counts"]["realized_volatility"] == 7
    assert coverage["non_null_counts"]["vol_gap"] == 5


@pytest.mark.parametrize(
    ("changes", "expected_fragment"),
    [
        ({"production_accepted": False}, "not production_accepted"),
        ({"blocking_failures": ["surface"]}, "blocking failures"),
        ({"overall_status": "FAIL"}, "overall_status"),
    ],
)
def test_rejects_non_accepted_snapshot(snapshot_root, tmp_path, changes, expected_fragment):
    """T22: acceptance guards abort before anything is written."""
    _patch_manifest(snapshot_root, **changes)
    destination = tmp_path / "derived" / SNAPSHOT_ID

    with pytest.raises(StraddleObservationStructuralError, match=expected_fragment):
        _run_pipeline(snapshot_root, destination)
    assert not destination.exists()


def test_rejects_missing_manifest(snapshot_root, tmp_path):
    """T22: a snapshot root with no manifest cannot be resolved."""
    for path in (snapshot_root / "manifests").glob("*.json"):
        path.unlink()
    with pytest.raises(StraddleObservationStructuralError, match="exactly one"):
        _run_pipeline(snapshot_root, tmp_path / "derived")


def test_rejects_a1_key_digest_mismatch(snapshot_root, tmp_path):
    """T22: an A1 carrying a different key grid is refused."""
    _patch_manifest(snapshot_root, params={"surface_actual_a1_key_digest": "0" * 64})
    destination = tmp_path / "derived" / SNAPSHOT_ID

    with pytest.raises(StraddleObservationStructuralError, match="A1 key-set digest"):
        _run_pipeline(snapshot_root, destination)
    assert not destination.exists()


def test_identical_rerun_is_a_no_op(snapshot_root, tmp_path):
    """T25: republishing the same content leaves the artifact untouched."""
    destination = tmp_path / "derived" / SNAPSHOT_ID
    _, first = _run_pipeline(snapshot_root, destination)
    original_bytes = first.observations_path.read_bytes()

    _, second = _run_pipeline(snapshot_root, destination)

    assert second.written is False
    assert second.observations_path.read_bytes() == original_bytes


def test_refuses_to_overwrite_divergent_artifact(snapshot_root, tmp_path):
    """T25: a different config version or content digest always raises."""
    destination = tmp_path / "derived" / SNAPSHOT_ID
    observations, result = _run_pipeline(snapshot_root, destination)
    inputs = resolve_surface_inputs(snapshot_root)

    def republish(frame):
        return publish_observations(
            frame,
            inputs=inputs,
            destination_dir=destination,
            meta_row_count=8,
            quote_row_count=12,
        )

    divergent = observations.copy()
    divergent.loc[0, "return_pct"] = divergent.loc[0, "return_pct"] + 1.0
    with pytest.raises(StraddleObservationStructuralError, match="content_digest"):
        republish(divergent)

    receipt = json.loads(result.lineage_path.read_text(encoding="utf-8"))
    receipt["transform_config"]["transform_config_version"] = "other-version"
    result.lineage_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(StraddleObservationStructuralError, match="transform_config_version"):
        republish(observations)


@pytest.mark.parametrize("corruption", ["missing", "mutated"])
def test_refuses_rerun_when_published_artifact_is_corrupt(snapshot_root, tmp_path, corruption):
    """T25: a receipt without a matching artifact is corruption, not a rerun."""
    destination = tmp_path / "derived" / SNAPSHOT_ID
    observations, result = _run_pipeline(snapshot_root, destination)

    if corruption == "missing":
        result.observations_path.unlink()
        expected = "is missing"
    else:
        result.observations_path.write_bytes(b"not a parquet file")
        expected = "does not match the SHA-256"

    with pytest.raises(StraddleObservationStructuralError, match=expected):
        publish_observations(
            observations,
            inputs=resolve_surface_inputs(snapshot_root),
            destination_dir=destination,
            meta_row_count=8,
            quote_row_count=12,
        )


def test_publication_is_atomic_and_receipt_last(snapshot_root, tmp_path, monkeypatch):
    """T26: a failure before the receipt leaves no receipt at the canonical path."""
    destination = tmp_path / "derived" / SNAPSHOT_ID
    write_receipt = so._write_json_atomic

    def explode(payload, target):
        raise OSError("simulated crash before the receipt is published")

    monkeypatch.setattr(so, "_write_json_atomic", explode)
    with pytest.raises(OSError, match="simulated crash"):
        _run_pipeline(snapshot_root, destination)

    assert not (destination / LINEAGE_FILENAME).exists()
    # The Parquet is published whole or not at all, and no temp file survives a
    # clean failure path.
    published = destination / OBSERVATIONS_FILENAME
    assert published.exists()
    assert pd.read_parquet(published).shape[0] == 8
    assert not list(destination.glob("*.tmp-*"))

    # With the receipt missing, the next run republishes rather than no-opping.
    # Only the receipt writer is restored; undoing every patch would also drop
    # the mocked git state that publication requires.
    monkeypatch.setattr(so, "_write_json_atomic", write_receipt)
    _, result = _run_pipeline(snapshot_root, destination)
    assert result.written is True
    assert result.receipt["transform_config"]["transform_config_version"] == (
        TRANSFORM_CONFIG_VERSION
    )


# =============================================================================
# Publication safety — repository identity and snapshot immutability
# =============================================================================


def _unavailable_git(*args: str) -> str:
    raise OSError("git executable not found")


def test_receipt_records_the_committed_revision(snapshot_root, tmp_path):
    """A published receipt names the one commit its artifact came from."""
    _, result = _run_pipeline(snapshot_root, tmp_path / "derived" / SNAPSHOT_ID)
    assert result.receipt["repo_sha"] == FAKE_REPO_SHA


@pytest.mark.parametrize(
    ("git_state", "expected_fragment"),
    [
        (_fake_git(FAKE_REPO_SHA, " M src/features/straddle_observations.py"), "uncommitted"),
        (_fake_git(FAKE_REPO_SHA, "?? scripts/unstaged_change.py"), "uncommitted"),
        (_fake_git("abc123", ""), "not a 40-character"),
        (_fake_git(FAKE_REPO_SHA.upper(), ""), "not a 40-character"),
        (_fake_git("", ""), "not a 40-character"),
        (_unavailable_git, "cannot determine"),
    ],
)
def test_publication_requires_a_committed_repository(
    snapshot_root, tmp_path, monkeypatch, git_state, expected_fragment
):
    """An artifact that cannot be attributed to one commit is never written."""
    monkeypatch.setattr(so, "_git_output", git_state)
    destination = tmp_path / "derived" / SNAPSHOT_ID

    with pytest.raises(StraddleObservationStructuralError, match=expected_fragment):
        _run_pipeline(snapshot_root, destination)

    # The guard runs before the destination is created, so nothing exists to
    # half-publish.
    assert not destination.exists()


def test_dry_run_is_allowed_from_a_dirty_tree(snapshot_root, tmp_path, monkeypatch):
    """--dry-run writes nothing, so it does not need a committed revision."""
    spec = importlib.util.spec_from_file_location("build_straddle_observations", CLI_PATH)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    monkeypatch.setattr(
        so, "_git_output", _fake_git(FAKE_REPO_SHA, " M src/features/straddle_observations.py")
    )
    output_root = tmp_path / "derived"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_straddle_observations.py",
            "--snapshot-root",
            str(snapshot_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert cli.main() == 0
    assert not output_root.exists()


@pytest.mark.parametrize("relative_destination", [None, "derived", "cache/surface/derived"])
def test_rejects_destination_inside_the_snapshot(snapshot_root, relative_destination):
    """Sprint 004's snapshot is immutable: nothing may be written into it."""
    destination = (
        snapshot_root if relative_destination is None else snapshot_root / relative_destination
    )

    with pytest.raises(StraddleObservationStructuralError, match="inside the accepted snapshot"):
        _run_pipeline(snapshot_root, destination)

    assert not (destination / OBSERVATIONS_FILENAME).exists()
    assert not (destination / LINEAGE_FILENAME).exists()


def test_accepts_a_destination_outside_the_snapshot(snapshot_root, tmp_path):
    """A sibling derived root is the normal case and stays allowed."""
    destination = snapshot_root.parent / "derived" / SNAPSHOT_ID
    _, result = _run_pipeline(snapshot_root, destination)

    assert result.written is True
    assert result.observations_path.is_file()
    assert tmp_path in result.observations_path.parents


def test_join_ignores_non_body_quotes():
    """OTM wings can never contribute to a body straddle."""
    wing = body_quote("call", strike=110.0)
    wing["is_body"] = False
    meta_df, quotes_df = frames([meta_row()], body_pair() + [wing])

    joined = join_body_legs(meta_df, quotes_df)
    assert int(joined.loc[0, "call_leg_count"]) == 1
    assert joined.loc[0, "call_strike"] == pytest.approx(BODY_STRIKE)
