"""Unit tests for the hardened spot-price extractor (Sprint 004 C8.2).

No real ORATS data. All fixtures use ``tmp_path`` with tiny synthetic parquet
files laid out as ``<root>/<YYYY>/ORATS_SMV_Strikes_YYYYMMDD.parquet``.
"""

from __future__ import annotations

import importlib.util
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


def test_inconsistent_repeated_stk_px_fails(cli_module, data_root, output_path):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.5, "adj_stkPx": 42.5},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


def test_inconsistent_repeated_adj_stk_px_fails(cli_module, data_root, output_path):
    bad = pd.DataFrame(
        [
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.5},
            {"ticker": "AAPL", "stkPx": 170.0, "adj_stkPx": 42.6},
        ]
    )
    code = _run_with_bad_second_day(cli_module, data_root, output_path, bad)
    assert code == 1
    assert not output_path.exists()


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
