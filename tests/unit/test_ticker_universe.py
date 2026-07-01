"""Unit tests for C5.3 ticker universe loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.ticker_universe import load_ticker_universe


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, index=False)


class TestLoadTickerUniverseCsv:
    def test_loads_csv_with_ticker_column(self, tmp_path: Path):
        path = tmp_path / "universe.csv"
        _write_csv(path, pd.DataFrame({"ticker": ["aapl", "msft"]}))
        assert load_ticker_universe(path) == ["AAPL", "MSFT"]

    def test_loads_csv_with_Ticker_column(self, tmp_path: Path):
        path = tmp_path / "liquid_tickers.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["nvda", "tsla"],
                    "snapshots_qualified": [10, 8],
                }
            ),
        )
        assert load_ticker_universe(path) == ["NVDA", "TSLA"]


class TestLoadTickerUniverseParquet:
    def test_loads_parquet_with_ticker_column(self, tmp_path: Path):
        path = tmp_path / "universe.parquet"
        _write_parquet(path, pd.DataFrame({"Ticker": ["goog", "meta"]}))
        assert load_ticker_universe(path) == ["GOOG", "META"]


class TestLoadTickerUniverseCleaning:
    def test_dedupes_tickers(self, tmp_path: Path):
        path = tmp_path / "dupes.csv"
        _write_csv(path, pd.DataFrame({"Ticker": ["AAPL", "aapl", "MSFT", "msft"]}))
        assert load_ticker_universe(path) == ["AAPL", "MSFT"]

    def test_strips_whitespace(self, tmp_path: Path):
        path = tmp_path / "spaces.csv"
        _write_csv(path, pd.DataFrame({"ticker": ["  AAPL  ", "\tMSFT\n"]}))
        assert load_ticker_universe(path) == ["AAPL", "MSFT"]

    def test_drops_null_and_blank_tickers(self, tmp_path: Path):
        path = tmp_path / "nulls.csv"
        _write_csv(
            path,
            pd.DataFrame({"Ticker": ["AAPL", None, "", "   ", "MSFT"]}),
        )
        assert load_ticker_universe(path) == ["AAPL", "MSFT"]

    def test_uppercases_lowercase_tickers(self, tmp_path: Path):
        path = tmp_path / "case.csv"
        _write_csv(path, pd.DataFrame({"ticker": ["aapl", "MsFt"]}))
        assert load_ticker_universe(path) == ["AAPL", "MSFT"]

    def test_returns_sorted_ticker_list(self, tmp_path: Path):
        path = tmp_path / "order.csv"
        _write_csv(path, pd.DataFrame({"Ticker": ["ZZZ", "AAA", "MMM"]}))
        assert load_ticker_universe(path) == ["AAA", "MMM", "ZZZ"]


class TestLoadTickerUniverseQualificationFilter:
    def test_filters_tickers_below_min_snapshots_qualified(self, tmp_path: Path):
        path = tmp_path / "liquid.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAA", "BBB", "CCC"],
                    "snapshots_qualified": [15, 11, 12],
                    "months_qualified": [15, 11, 12],
                }
            ),
        )
        assert load_ticker_universe(path, min_snapshots_qualified=12) == [
            "AAA",
            "CCC",
        ]

    def test_default_does_not_filter_when_snapshots_qualified_present(
        self, tmp_path: Path
    ):
        """Default ``None`` loads the full C4-style universe regardless of counts."""
        path = tmp_path / "liquid.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAA", "BBB"],
                    "snapshots_qualified": [5, 20],
                }
            ),
        )
        assert load_ticker_universe(path) == ["AAA", "BBB"]

    def test_uses_months_qualified_only_when_snapshots_column_absent(
        self, tmp_path: Path
    ):
        path = tmp_path / "legacy.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAA", "BBB"],
                    "months_qualified": [12, 8],
                }
            ),
        )
        assert load_ticker_universe(path, min_snapshots_qualified=12) == ["AAA"]

    def test_prefers_snapshots_qualified_over_months_qualified_when_both_present(
        self, tmp_path: Path
    ):
        """``months_qualified`` is ignored when ``snapshots_qualified`` exists."""
        path = tmp_path / "both_qual.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAA", "BBB"],
                    "snapshots_qualified": [5, 20],
                    "months_qualified": [20, 5],
                }
            ),
        )
        assert load_ticker_universe(path, min_snapshots_qualified=12) == ["BBB"]

    def test_raises_when_min_snapshots_qualified_set_without_qualification_column(
        self, tmp_path: Path
    ):
        path = tmp_path / "no_count.csv"
        _write_csv(path, pd.DataFrame({"Ticker": ["AAPL"]}))
        with pytest.raises(ValueError, match="No qualification column found"):
            load_ticker_universe(path, min_snapshots_qualified=12)

    def test_raises_when_qualification_filter_removes_all_tickers(
        self, tmp_path: Path
    ):
        path = tmp_path / "all_below.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAA", "BBB"],
                    "snapshots_qualified": [5, 8],
                }
            ),
        )
        with pytest.raises(ValueError, match="No valid tickers remain"):
            load_ticker_universe(path, min_snapshots_qualified=12)


class TestLoadTickerUniverseColumnPreference:
    def test_prefers_Ticker_when_both_columns_exist(self, tmp_path: Path):
        """When both columns exist, ``Ticker`` wins (C4 ``liquid_tickers`` convention)."""
        path = tmp_path / "both.csv"
        _write_csv(
            path,
            pd.DataFrame(
                {
                    "Ticker": ["AAPL"],
                    "ticker": ["MSFT"],
                }
            ),
        )
        assert load_ticker_universe(path) == ["AAPL"]


class TestLoadTickerUniverseErrors:
    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path):
        missing = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_ticker_universe(missing)

    def test_raises_value_error_for_unsupported_extension(self, tmp_path: Path):
        path = tmp_path / "universe.json"
        path.write_text('{"ticker": "AAPL"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported ticker universe file extension"):
            load_ticker_universe(path)

    def test_raises_value_error_when_no_ticker_column(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        _write_csv(path, pd.DataFrame({"symbol": ["AAPL"]}))
        with pytest.raises(ValueError, match="No ticker column found"):
            load_ticker_universe(path)

    def test_raises_value_error_when_no_valid_tickers_remain(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        _write_csv(path, pd.DataFrame({"Ticker": [None, "", "   "]}))
        with pytest.raises(ValueError, match="No valid tickers remain"):
            load_ticker_universe(path)
