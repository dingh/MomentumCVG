"""Unit tests for the hardened spot-price extractor (Sprint 004 C8.2).

No real ORATS data. All fixtures use ``tmp_path`` with tiny synthetic parquet
files laid out as ``<root>/<YYYY>/ORATS_SMV_Strikes_YYYYMMDD.parquet``.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.spot_price_db import SpotPriceDB

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "extract_spot_prices.py"

DAY_1 = date(2024, 1, 2)
DAY_2 = date(2024, 1, 3)

OUTPUT_COLUMNS = ["date", "ticker", "adj_spot_price", "spot_price"]


@pytest.fixture
def cli_module():
    spec = importlib.util.spec_from_file_location("extract_spot_prices", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "adjusted_liquid"
    root.mkdir()
    return root


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "spot_prices.parquet"


def _day_path(data_root: Path, day: date) -> Path:
    return (
        data_root
        / str(day.year)
        / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.parquet"
    )


def _write_day(data_root: Path, day: date, frame: pd.DataFrame) -> Path:
    path = _day_path(data_root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _good_frame() -> pd.DataFrame:
    """Two tickers, two strike rows each, consistent repeated spot values."""
    return pd.DataFrame(
        [
            {"ticker": "MSFT", "stkPx": 85.0, "adj_stkPx": 85.0, "strike": 85.0},
            {"ticker": "MSFT", "stkPx": 85.0, "adj_stkPx": 85.0, "strike": 90.0},
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5, "strike": 165.0},
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5, "strike": 170.0},
        ]
    )


def _argv(data_root: Path, output: Path, extra: list[str] | None = None) -> list[str]:
    argv = [
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--year",
        "2024",
    ]
    if extra:
        argv.extend(extra)
    return argv


def _seed_two_good_days(data_root: Path) -> None:
    _write_day(data_root, DAY_1, _good_frame())
    _write_day(data_root, DAY_2, _good_frame())


# ---------------------------------------------------------------------------
# Successful behavior
# ---------------------------------------------------------------------------


def test_happy_path_two_dates_returns_zero(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)

    code = cli_module.main(_argv(data_root, output_path))

    assert code == 0
    assert output_path.exists()


def test_output_schema_sorting_and_grain(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)

    assert cli_module.main(_argv(data_root, output_path)) == 0

    out = pd.read_parquet(output_path)
    assert list(out.columns) == OUTPUT_COLUMNS

    # Sorted by date, then ticker; one row per source ticker/date.
    expected_keys = [
        (DAY_1, "AAPL"),
        (DAY_1, "MSFT"),
        (DAY_2, "AAPL"),
        (DAY_2, "MSFT"),
    ]
    actual_keys = [
        (pd.Timestamp(row["date"]).date(), row["ticker"])
        for _, row in out.iterrows()
    ]
    assert actual_keys == expected_keys

    aapl = out[(out["ticker"] == "AAPL")].iloc[0]
    assert aapl["adj_spot_price"] == pytest.approx(42.5)
    assert aapl["spot_price"] == pytest.approx(170.0)


def test_consistent_repeated_values_collapse_to_one_row(
    cli_module, data_root, output_path
):
    _write_day(data_root, DAY_1, _good_frame())
    (data_root / str(DAY_2.year)).mkdir(exist_ok=True)
    _write_day(data_root, DAY_2, _good_frame())

    assert cli_module.main(_argv(data_root, output_path)) == 0

    out = pd.read_parquet(output_path)
    assert len(out) == 4  # 2 dates x 2 tickers despite 4 source rows per day
    assert not out.duplicated(subset=["date", "ticker"]).any()


def test_spot_price_db_can_load_output(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)

    assert cli_module.main(_argv(data_root, output_path)) == 0

    spot_db = SpotPriceDB.load(str(output_path))
    assert spot_db.get_spot("AAPL", DAY_1) == pytest.approx(42.5)
    assert spot_db.get_spot("MSFT", DAY_2) == pytest.approx(85.0)


def test_ticker_normalization_merges_whitespace_and_case(
    cli_module, data_root, output_path
):
    frame = pd.DataFrame(
        [
            {"ticker": " aapl ", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
        ]
    )
    _write_day(data_root, DAY_1, frame)

    assert cli_module.main(_argv(data_root, output_path)) == 0

    out = pd.read_parquet(output_path)
    assert list(out["ticker"]) == ["AAPL"]


# ---------------------------------------------------------------------------
# Inventory failures
# ---------------------------------------------------------------------------


def test_missing_data_root_is_usage_error(cli_module, tmp_path, output_path):
    missing_root = tmp_path / "does_not_exist"
    code = cli_module.main(_argv(missing_root, output_path))
    assert code == 2
    assert not output_path.exists()


def test_discover_dates_missing_data_root_raises(cli_module, tmp_path):
    with pytest.raises(cli_module.SpotExtractionError, match="data root"):
        cli_module.discover_adjusted_dates(tmp_path / "nope", 2024, 2024)


def test_missing_requested_year_directory_fails(cli_module, data_root, output_path):
    _write_day(data_root, DAY_1, _good_frame())  # only 2024 exists

    code = cli_module.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            "--start-year",
            "2023",
            "--end-year",
            "2024",
        ]
    )
    assert code == 1
    assert not output_path.exists()


def test_no_adjusted_dates_fails(cli_module, data_root, output_path):
    (data_root / "2024").mkdir()  # year dir exists but is empty

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_malformed_adjusted_filename_fails(cli_module, data_root, output_path):
    _write_day(data_root, DAY_1, _good_frame())
    bad = data_root / "2024" / "ORATS_SMV_Strikes_2024010.parquet"
    bad.write_bytes(b"parquet")

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_invalid_date_in_filename_fails(cli_module, data_root, output_path):
    _write_day(data_root, DAY_1, _good_frame())
    bad = data_root / "2024" / "ORATS_SMV_Strikes_20241301.parquet"
    bad.write_bytes(b"parquet")

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_duplicate_discovered_date_fails(cli_module, data_root, output_path):
    _write_day(data_root, DAY_1, _good_frame())
    # Same 2024 date placed in the 2023 year directory as well.
    duplicate = (
        data_root / "2023" / f"ORATS_SMV_Strikes_{DAY_1.strftime('%Y%m%d')}.parquet"
    )
    duplicate.parent.mkdir()
    _good_frame().to_parquet(duplicate, index=False)

    code = cli_module.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            "--start-year",
            "2023",
            "--end-year",
            "2024",
        ]
    )
    assert code == 1
    assert not output_path.exists()


def test_file_date_outside_containing_year_directory_fails(
    cli_module, data_root, output_path
):
    _write_day(data_root, DAY_1, _good_frame())
    stray = data_root / "2024" / "ORATS_SMV_Strikes_20230103.parquet"
    _good_frame().to_parquet(stray, index=False)

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_discover_dates_sorted_ascending(cli_module, data_root):
    _write_day(data_root, DAY_2, _good_frame())
    _write_day(data_root, DAY_1, _good_frame())

    dates = cli_module.discover_adjusted_dates(data_root, 2024, 2024)
    assert dates == [DAY_1, DAY_2]


def test_empty_weekend_file_is_excluded_from_inventory(
    cli_module, data_root, output_path
):
    _seed_two_good_days(data_root)
    sunday = date(2024, 1, 7)
    empty = pd.DataFrame(columns=["ticker", "stkPx", "adj_stkPx"])
    _write_day(data_root, sunday, empty)

    dates = cli_module.discover_adjusted_dates(data_root, 2024, 2024)
    assert dates == [DAY_1, DAY_2]  # Sunday excluded

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 0
    out = pd.read_parquet(output_path)
    out_dates = set(pd.to_datetime(out["date"]).dt.date)
    assert sunday not in out_dates


def test_empty_weekend_file_in_wrong_year_directory_fails(
    cli_module, data_root, output_path
):
    """Year membership is checked before weekend exclusion."""
    _seed_two_good_days(data_root)
    sunday_2024 = date(2024, 1, 7)
    empty = pd.DataFrame(columns=["ticker", "stkPx", "adj_stkPx"])
    # Place a 2024 Sunday file under the 2023 year directory.
    wrong_year = data_root / "2023" / f"ORATS_SMV_Strikes_{sunday_2024.strftime('%Y%m%d')}.parquet"
    wrong_year.parent.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(wrong_year, index=False)

    with pytest.raises(cli_module.SpotExtractionError, match="does not belong"):
        cli_module.discover_adjusted_dates(data_root, 2023, 2024)

    code = cli_module.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            "--start-year",
            "2023",
            "--end-year",
            "2024",
        ]
    )
    assert code == 1
    assert not output_path.exists()


def test_nonempty_weekend_file_fails(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)
    saturday = date(2024, 1, 6)
    _write_day(data_root, saturday, _good_frame())  # data on a non-trading day

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_empty_weekday_file_still_fails(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)
    thursday = date(2024, 1, 4)
    empty = pd.DataFrame(columns=["ticker", "stkPx", "adj_stkPx"])
    _write_day(data_root, thursday, empty)

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Date-processing failures (one bad date among good dates -> whole run fails)
# ---------------------------------------------------------------------------


def _run_with_bad_second_day(
    cli_module, data_root: Path, output_path: Path, bad_frame: pd.DataFrame
) -> int:
    _write_day(data_root, DAY_1, _good_frame())
    _write_day(data_root, DAY_2, bad_frame)
    return cli_module.main(_argv(data_root, output_path))


def test_provider_read_failure_fails_complete_run(
    cli_module, data_root, output_path
):
    _write_day(data_root, DAY_1, _good_frame())
    corrupt = _day_path(data_root, DAY_2)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not a parquet file")

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


def test_empty_daily_frame_fails(cli_module, data_root, output_path):
    empty = pd.DataFrame(columns=["ticker", "stkPx", "adj_stkPx"])
    code = _run_with_bad_second_day(cli_module, data_root, output_path, empty)
    assert code == 1
    assert not output_path.exists()


def test_missing_required_source_column_fails(cli_module, data_root, output_path):
    bad = _good_frame().drop(columns=["adj_stkPx"])
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


def test_null_ticker_fails(cli_module, data_root, output_path):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": None, "stkPx": 85.0, "adj_stkPx": 85.0},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


def test_blank_ticker_fails(cli_module, data_root, output_path):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "   ", "stkPx": 85.0, "adj_stkPx": 85.0},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize(
    "bad_value",
    ["abc", float("nan"), float("inf"), float("-inf"), 0.0, -5.0],
    ids=["non_numeric", "nan", "inf", "neg_inf", "zero", "negative"],
)
def test_invalid_stk_px_value_fails(cli_module, data_root, output_path, bad_value):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": bad_value, "adj_stkPx": 42.5},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), 0.0, -5.0],
    ids=["nan", "inf", "zero", "negative"],
)
def test_invalid_adj_stk_px_value_fails(
    cli_module, data_root, output_path, bad_value
):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": bad_value},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


def test_extract_for_date_raises_instead_of_returning_empty(
    cli_module, data_root
):
    """A failed date must raise, never come back as an empty record list."""
    provider = cli_module.ORATSDataProvider(
        data_root=str(data_root), min_volume=0, min_open_interest=0,
        min_bid=0.0, max_spread_pct=1.0, cache_size=1,
    )
    with pytest.raises(cli_module.SpotExtractionError, match=DAY_1.isoformat()):
        cli_module.extract_spot_prices_for_date(provider, DAY_1)


# ---------------------------------------------------------------------------
# Repeated-value consistency
# ---------------------------------------------------------------------------


def test_inconsistent_ticker_is_dropped_not_fatal(
    cli_module, data_root, output_path, caplog
):
    """A ticker with inconsistent repeated stkPx is dropped for that date;
    the remaining tickers still publish."""
    day2 = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "DJX", "stkPx": 489.0, "adj_stkPx": 489.0},
            {"ticker": "DJX", "stkPx": 501.0, "adj_stkPx": 501.0},
        ]
    )
    _write_day(data_root, DAY_1, _good_frame())
    _write_day(data_root, DAY_2, day2)

    with caplog.at_level(logging.WARNING):
        code = cli_module.main(_argv(data_root, output_path))

    assert code == 0
    out = pd.read_parquet(output_path)
    out["date"] = pd.to_datetime(out["date"]).dt.date

    day2_tickers = set(out[out["date"] == DAY_2]["ticker"])
    assert day2_tickers == {"AAPL"}  # DJX dropped on the inconsistent date
    day1_tickers = set(out[out["date"] == DAY_1]["ticker"])
    assert day1_tickers == {"AAPL", "MSFT"}  # other dates unaffected
    assert any(
        "dropping" in record.getMessage() and "DJX" in record.getMessage()
        for record in caplog.records
    )


def test_inconsistent_adj_stk_px_also_drops_ticker(
    cli_module, data_root, output_path
):
    day2 = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "MSFT", "stkPx": 85.0, "adj_stkPx": 85.0},
            {"ticker": "MSFT", "stkPx": 85.0, "adj_stkPx": 85.1},
        ]
    )
    _write_day(data_root, DAY_1, _good_frame())
    _write_day(data_root, DAY_2, day2)

    code = cli_module.main(_argv(data_root, output_path))

    assert code == 0
    out = pd.read_parquet(output_path)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    assert set(out[out["date"] == DAY_2]["ticker"]) == {"AAPL"}


def test_all_tickers_inconsistent_fails_date(cli_module, data_root, output_path):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.5, "adj_stkPx": 42.5},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


def test_extract_for_date_reports_dropped_tickers(cli_module, data_root):
    frame = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "SPX", "stkPx": 5000.0, "adj_stkPx": 5000.0},
            {"ticker": "SPX", "stkPx": 5050.0, "adj_stkPx": 5050.0},
        ]
    )
    _write_day(data_root, DAY_1, frame)
    provider = cli_module.ORATSDataProvider(
        data_root=str(data_root), min_volume=0, min_open_interest=0,
        min_bid=0.0, max_spread_pct=1.0, cache_size=1,
    )

    result = cli_module.extract_spot_prices_for_date(provider, DAY_1)

    assert result.dropped_tickers == frozenset({"SPX"})
    assert result.expected_tickers == frozenset({"AAPL"})
    assert [r["ticker"] for r in result.records] == ["AAPL"]


def test_differences_within_tolerance_pass(cli_module, data_root, output_path):
    # abs difference 1e-9 is inside abs_tol=1e-8.
    frame = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.0 + 1e-9, "adj_stkPx": 42.5 + 1e-9},
        ]
    )
    _write_day(data_root, DAY_1, frame)

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 0

    out = pd.read_parquet(output_path)
    assert len(out) == 1
    # First-row selection after consistency is established.
    assert math.isclose(out["spot_price"].iloc[0], 170.0, rel_tol=1e-9, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# Completeness and uniqueness (global validation helper)
# ---------------------------------------------------------------------------


def _valid_combined_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": DAY_1, "ticker": "AAPL", "adj_spot_price": 42.5, "spot_price": 170.0},
            {"date": DAY_1, "ticker": "MSFT", "adj_spot_price": 85.0, "spot_price": 85.0},
            {"date": DAY_2, "ticker": "AAPL", "adj_spot_price": 43.0, "spot_price": 172.0},
            {"date": DAY_2, "ticker": "MSFT", "adj_spot_price": 86.0, "spot_price": 86.0},
        ]
    )


def _expected_map() -> dict[date, set[str]]:
    return {DAY_1: {"AAPL", "MSFT"}, DAY_2: {"AAPL", "MSFT"}}


def test_validate_complete_frame_passes_and_sorts(cli_module):
    frame = _valid_combined_frame().iloc[::-1]  # reverse ordering
    result = cli_module.validate_complete_spot_frame(
        frame, [DAY_1, DAY_2], _expected_map()
    )
    assert list(result.columns) == OUTPUT_COLUMNS
    assert list(result.index) == [0, 1, 2, 3]
    assert list(zip(result["date"], result["ticker"])) == [
        (DAY_1, "AAPL"),
        (DAY_1, "MSFT"),
        (DAY_2, "AAPL"),
        (DAY_2, "MSFT"),
    ]


def test_validate_missing_expected_date_fails(cli_module):
    frame = _valid_combined_frame()
    frame = frame[frame["date"] != DAY_2]
    expected = _expected_map()
    with pytest.raises(cli_module.SpotExtractionError, match="missing expected dates"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], expected)


def test_validate_unexpected_date_fails(cli_module):
    frame = _valid_combined_frame()
    with pytest.raises(cli_module.SpotExtractionError, match="unexpected dates"):
        cli_module.validate_complete_spot_frame(
            frame, [DAY_1], {DAY_1: {"AAPL", "MSFT"}}
        )


def test_validate_missing_expected_ticker_fails(cli_module):
    frame = _valid_combined_frame()
    frame = frame[~((frame["date"] == DAY_2) & (frame["ticker"] == "MSFT"))]
    with pytest.raises(cli_module.SpotExtractionError, match="missing expected tickers"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], _expected_map())


def test_validate_unexpected_ticker_fails(cli_module):
    frame = pd.concat(
        [
            _valid_combined_frame(),
            pd.DataFrame(
                [
                    {
                        "date": DAY_2,
                        "ticker": "NVDA",
                        "adj_spot_price": 500.0,
                        "spot_price": 500.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(cli_module.SpotExtractionError, match="unexpected tickers"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], _expected_map())


def test_validate_duplicate_date_ticker_fails(cli_module):
    frame = pd.concat(
        [_valid_combined_frame(), _valid_combined_frame().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(cli_module.SpotExtractionError, match="duplicate"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], _expected_map())


def test_validate_empty_frame_fails(cli_module):
    frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    with pytest.raises(cli_module.SpotExtractionError, match="empty"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1], _expected_map())


def test_validate_nonpositive_value_fails(cli_module):
    frame = _valid_combined_frame()
    frame.loc[0, "adj_spot_price"] = 0.0
    with pytest.raises(cli_module.SpotExtractionError, match="non-positive"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], _expected_map())


def test_validate_null_value_fails(cli_module):
    frame = _valid_combined_frame()
    frame.loc[0, "spot_price"] = None
    with pytest.raises(cli_module.SpotExtractionError, match="null"):
        cli_module.validate_complete_spot_frame(frame, [DAY_1, DAY_2], _expected_map())


# ---------------------------------------------------------------------------
# Publication behavior
# ---------------------------------------------------------------------------


def test_success_atomically_replaces_existing_output(
    cli_module, data_root, output_path
):
    _seed_two_good_days(data_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"old artifact bytes")

    assert cli_module.main(_argv(data_root, output_path)) == 0

    out = pd.read_parquet(output_path)  # replaced with a valid parquet
    assert len(out) == 4


def test_failed_run_leaves_existing_output_unchanged(
    cli_module, data_root, output_path
):
    existing_bytes = b"existing valid artifact"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(existing_bytes)

    bad = pd.DataFrame([{"ticker": "AAPL", "stkPx": -1.0, "adj_stkPx": 42.5}])
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)

    assert code == 1
    assert output_path.read_bytes() == existing_bytes
    assert list(output_path.parent.glob("*.tmp-*")) == []


def test_failed_run_with_no_existing_output_creates_no_artifact(
    cli_module, data_root, output_path
):
    bad = pd.DataFrame([{"ticker": "AAPL", "stkPx": -1.0, "adj_stkPx": 42.5}])
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)

    assert code == 1
    assert not output_path.exists()
    if output_path.parent.exists():
        assert list(output_path.parent.iterdir()) == []


def test_temp_file_removed_after_readback_failure(
    cli_module, tmp_path, monkeypatch
):
    frame = _valid_combined_frame()
    validated = cli_module.validate_complete_spot_frame(
        frame, [DAY_1, DAY_2], _expected_map()
    )
    output = tmp_path / "out" / "spot.parquet"

    def _boom(path):
        raise OSError("simulated read-back failure")

    monkeypatch.setattr(cli_module, "read_parquet_for_validation", _boom)

    with pytest.raises(cli_module.SpotExtractionError, match="read-back|failed to write"):
        cli_module.write_parquet_atomically(validated, output)

    assert not output.exists()
    assert list(output.parent.glob("*.parquet")) == []
    assert list(output.parent.glob("*.tmp-*")) == []


def test_temp_file_removed_after_readback_mismatch(
    cli_module, tmp_path, monkeypatch
):
    frame = _valid_combined_frame()
    validated = cli_module.validate_complete_spot_frame(
        frame, [DAY_1, DAY_2], _expected_map()
    )
    output = tmp_path / "out" / "spot.parquet"

    truncated = validated.iloc[:2].copy()
    monkeypatch.setattr(
        cli_module, "read_parquet_for_validation", lambda path: truncated
    )

    with pytest.raises(cli_module.SpotExtractionError, match="row count"):
        cli_module.write_parquet_atomically(validated, output)

    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_write_parquet_atomically_success_roundtrip(cli_module, tmp_path):
    frame = _valid_combined_frame()
    validated = cli_module.validate_complete_spot_frame(
        frame, [DAY_1, DAY_2], _expected_map()
    )
    output = tmp_path / "out" / "spot.parquet"

    cli_module.write_parquet_atomically(validated, output)

    assert output.exists()
    assert list(output.parent.iterdir()) == [output]  # no leftover temp file
    out = pd.read_parquet(output)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 4


def test_one_failed_date_prevents_publication_of_successful_dates(
    cli_module, data_root, output_path
):
    _write_day(data_root, DAY_1, _good_frame())
    _write_day(data_root, DAY_2, _good_frame())
    _write_day(
        data_root,
        date(2024, 1, 4),
        pd.DataFrame([{"ticker": "AAPL", "stkPx": float("nan"), "adj_stkPx": 42.5}]),
    )

    code = cli_module.main(_argv(data_root, output_path))
    assert code == 1
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# CLI / exit behavior
# ---------------------------------------------------------------------------


def test_year_combined_with_start_year_is_usage_error(
    cli_module, data_root, output_path
):
    _seed_two_good_days(data_root)
    code = cli_module.main(
        _argv(data_root, output_path, extra=["--start-year", "2024"])
    )
    assert code == 2
    assert not output_path.exists()


def test_year_combined_with_end_year_is_usage_error(
    cli_module, data_root, output_path
):
    _seed_two_good_days(data_root)
    code = cli_module.main(
        _argv(data_root, output_path, extra=["--end-year", "2024"])
    )
    assert code == 2


def test_inverted_year_range_is_usage_error(cli_module, data_root, output_path):
    _seed_two_good_days(data_root)
    code = cli_module.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            "--start-year",
            "2025",
            "--end-year",
            "2024",
        ]
    )
    assert code == 2


def test_non_parquet_output_is_usage_error(cli_module, data_root, tmp_path):
    _seed_two_good_days(data_root)
    code = cli_module.main(
        [
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / "spot.csv"),
            "--year",
            "2024",
        ]
    )
    assert code == 2


def test_resolve_year_range_defaults(cli_module):
    args = cli_module.parse_args([])
    assert cli_module.resolve_year_range(args) == (2018, 2026)


def test_resolve_year_range_single_year(cli_module):
    args = cli_module.parse_args(["--year", "2024"])
    assert cli_module.resolve_year_range(args) == (2024, 2024)


def test_resolve_year_range_explicit_bounds(cli_module):
    args = cli_module.parse_args(["--start-year", "2019", "--end-year", "2021"])
    assert cli_module.resolve_year_range(args) == (2019, 2021)


# ---------------------------------------------------------------------------
# C8.2R Fix 1 — output path inside the input data root is rejected
# ---------------------------------------------------------------------------


def _snapshot_tree_bytes(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_output_equal_to_adjusted_daily_parquet_is_rejected(
    cli_module, data_root
):
    _seed_two_good_days(data_root)
    before = _snapshot_tree_bytes(data_root)

    target = _day_path(data_root, DAY_1)
    code = cli_module.main(_argv(data_root, target))

    assert code == 2
    assert _snapshot_tree_bytes(data_root) == before  # source bytes untouched
    assert list(data_root.rglob("*.tmp-*")) == []


def test_output_elsewhere_inside_data_root_is_rejected(cli_module, data_root):
    _seed_two_good_days(data_root)
    before = _snapshot_tree_bytes(data_root)

    inside = data_root / "cache" / "spot_prices_adjusted.parquet"
    code = cli_module.main(_argv(data_root, inside))

    assert code == 2
    assert not inside.exists()
    assert not inside.parent.exists()  # no directory created inside the root
    assert _snapshot_tree_bytes(data_root) == before


def test_output_inside_unrequested_year_directory_is_rejected(
    cli_module, data_root
):
    _seed_two_good_days(data_root)
    (data_root / "2023").mkdir()

    inside = data_root / "2023" / "spot.parquet"
    code = cli_module.main(_argv(data_root, inside))

    assert code == 2
    assert not inside.exists()
    assert list(data_root.rglob("*.tmp-*")) == []


def test_output_equal_to_data_root_is_rejected(cli_module, data_root):
    _seed_two_good_days(data_root)

    code = cli_module.main(_argv(data_root, data_root))
    assert code == 2

    # The guard itself also rejects the root, independent of the parquet
    # suffix check that fires first in main.
    with pytest.raises(cli_module.UsageError, match="outside data root"):
        cli_module.ensure_output_outside_data_root(data_root, data_root)


def test_guard_rejects_nested_output_directly(cli_module, data_root):
    nested = data_root / "2024" / "deep" / "spot.parquet"
    with pytest.raises(cli_module.UsageError, match="outside data root"):
        cli_module.ensure_output_outside_data_root(data_root, nested)


def test_sibling_cache_output_outside_data_root_succeeds(cli_module, tmp_path):
    # Valid snapshot layout: output is a sibling of the adjusted root.
    snapshot = tmp_path / "snapshot"
    adjusted_root = snapshot / "input" / "adjusted_liquid"
    adjusted_root.mkdir(parents=True)
    _write_day(adjusted_root, DAY_1, _good_frame())
    _write_day(adjusted_root, DAY_2, _good_frame())

    output = snapshot / "cache" / "spot_prices_adjusted.parquet"
    code = cli_module.main(_argv(adjusted_root, output))

    assert code == 0
    assert output.exists()
    out = pd.read_parquet(output)
    assert len(out) == 4


def test_similarly_prefixed_separate_directory_is_allowed(cli_module, tmp_path):
    # C:/data/root2 must not be treated as a child of C:/data/root.
    adjusted_root = tmp_path / "data" / "root"
    adjusted_root.mkdir(parents=True)
    _write_day(adjusted_root, DAY_1, _good_frame())

    output = tmp_path / "data" / "root2" / "spot.parquet"
    cli_module.ensure_output_outside_data_root(adjusted_root, output)  # no raise

    code = cli_module.main(_argv(adjusted_root, output))
    assert code == 0
    assert output.exists()


# ---------------------------------------------------------------------------
# C8.2R Fix 2 — output-directory creation failure is controlled (exit 1)
# ---------------------------------------------------------------------------


def test_mkdir_failure_returns_one_and_preserves_output(
    cli_module, data_root, output_path, monkeypatch
):
    _seed_two_good_days(data_root)  # seed before patching mkdir

    existing_bytes = b"existing sentinel artifact"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(existing_bytes)

    real_mkdir = Path.mkdir

    def _failing_mkdir(self, *args, **kwargs):
        if self == output_path.parent:
            raise OSError("simulated mkdir failure")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _failing_mkdir)

    code = cli_module.main(_argv(data_root, output_path))  # must not raise

    assert code == 1
    assert output_path.read_bytes() == existing_bytes
    assert list(output_path.parent.glob("*.tmp-*")) == []


def test_write_parquet_atomically_mkdir_failure_raises_spot_error(
    cli_module, tmp_path, monkeypatch
):
    validated = cli_module.validate_complete_spot_frame(
        _valid_combined_frame(), [DAY_1, DAY_2], _expected_map()
    )
    output = tmp_path / "blocked" / "spot.parquet"

    real_mkdir = Path.mkdir

    def _failing_mkdir(self, *args, **kwargs):
        if self == output.parent:
            raise OSError("simulated mkdir failure")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _failing_mkdir)

    with pytest.raises(cli_module.SpotExtractionError, match="failed to write"):
        cli_module.write_parquet_atomically(validated, output)

    # temp_path was never created; cleanup must tolerate that.
    assert not output.parent.exists()


# ---------------------------------------------------------------------------
# C8.2R Fix 3 — os.replace failure is controlled and non-destructive
# ---------------------------------------------------------------------------


def test_replace_failure_through_main_preserves_existing_output(
    cli_module, data_root, output_path, monkeypatch
):
    _seed_two_good_days(data_root)

    existing_bytes = b"published sentinel that must survive"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(existing_bytes)

    def _failing_replace(src, dst, *args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cli_module.os, "replace", _failing_replace)

    code = cli_module.main(_argv(data_root, output_path))

    assert code == 1
    assert output_path.read_bytes() == existing_bytes
    assert list(output_path.parent.iterdir()) == [output_path]  # no extras


def test_replace_failure_direct_cleans_temp_and_raises(
    cli_module, tmp_path, monkeypatch
):
    validated = cli_module.validate_complete_spot_frame(
        _valid_combined_frame(), [DAY_1, DAY_2], _expected_map()
    )
    output = tmp_path / "out" / "spot.parquet"

    def _failing_replace(src, dst, *args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cli_module.os, "replace", _failing_replace)

    with pytest.raises(cli_module.SpotExtractionError, match="failed to write"):
        cli_module.write_parquet_atomically(validated, output)

    assert not output.exists()
    assert list(output.parent.iterdir()) == []  # temp parquet deleted


# ---------------------------------------------------------------------------
# C8.2R Fix 4 — post-publication size stat is best effort
# ---------------------------------------------------------------------------


def test_stat_failure_after_publication_still_returns_zero(
    cli_module, data_root, output_path, monkeypatch, caplog
):
    _seed_two_good_days(data_root)

    real_stat = Path.stat

    def _failing_stat(self, *args, **kwargs):
        if self == output_path:
            raise OSError("simulated stat failure")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _failing_stat)

    with caplog.at_level(logging.WARNING):
        code = cli_module.main(_argv(data_root, output_path))

    assert code == 0
    monkeypatch.undo()
    assert output_path.exists()
    out = pd.read_parquet(output_path)
    assert len(out) == 4
    assert any(
        "size could not be read" in record.getMessage()
        for record in caplog.records
    )


def test_log_output_size_best_effort_missing_file_warns(
    cli_module, tmp_path, caplog
):
    missing = tmp_path / "never_written.parquet"
    with caplog.at_level(logging.WARNING):
        cli_module.log_output_size_best_effort(missing)  # must not raise
    assert any(
        "size could not be read" in record.getMessage()
        for record in caplog.records
    )
