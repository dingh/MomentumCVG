"""C5.4A — golden contract tests for SplitAdjuster core math."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.split_adjuster import (
    PRICE_COLS,
    SplitAdjuster,
    SplitAdjustmentError,
)


def _write_splits(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "splits_hist.parquet"
    if rows:
        df = pd.DataFrame(rows)
        df["split_date"] = pd.to_datetime(df["split_date"])
    else:
        df = pd.DataFrame(columns=["ticker", "split_date", "divisor"])
    df.to_parquet(path, index=False)
    return path


def _make_adjuster(
    tmp_path: Path,
    split_rows: list[dict],
    min_split_date: str = "2014-01-01",
) -> SplitAdjuster:
    splits_path = _write_splits(tmp_path, split_rows)
    return SplitAdjuster(
        raw_root=tmp_path / "raw",
        adj_root=tmp_path / "adj",
        splits_path=splits_path,
        min_split_date=min_split_date,
    )


def _sample_options_df(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {
                "ticker": "AAA",
                "stkPx": 100.0,
                "strike": 100.0,
                "cBidPx": 5.0,
                "cAskPx": 5.5,
                "pBidPx": 4.0,
                "pAskPx": 4.5,
            }
        ]
    return pd.DataFrame(rows)


def _assert_prices_equal_raw(out: pd.DataFrame, row_idx: int = 0) -> None:
    row = out.iloc[row_idx]
    assert row["split_factor"] == pytest.approx(1.0)
    for col in PRICE_COLS:
        if col in out.columns:
            assert row[f"adj_{col}"] == pytest.approx(row[col])


class TestNoSplit:
    def test_no_split_rows_factor_one_and_prices_unchanged(self, tmp_path: Path):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df([{"ticker": "AAA", "stkPx": 50.0, "strike": 50.0}])

        out = adjuster.adjust_dataframe(df, trade_date)

        assert out["split_factor"].iloc[0] == pytest.approx(1.0)
        assert out["adj_stkPx"].iloc[0] == pytest.approx(50.0)
        assert out["adj_strike"].iloc[0] == pytest.approx(50.0)
        assert out["spot_px"].iloc[0] == pytest.approx(out["adj_stkPx"].iloc[0])


class TestOneFutureSplit:
    def test_future_split_divides_prices(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                }
            ],
        )
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df()

        out = adjuster.adjust_dataframe(df, trade_date)

        assert out["split_factor"].iloc[0] == pytest.approx(2.0)
        assert out["adj_stkPx"].iloc[0] == pytest.approx(50.0)
        assert out["adj_strike"].iloc[0] == pytest.approx(50.0)
        assert out["adj_cBidPx"].iloc[0] == pytest.approx(2.5)
        assert out["adj_cAskPx"].iloc[0] == pytest.approx(2.75)
        assert out["adj_pBidPx"].iloc[0] == pytest.approx(2.0)
        assert out["adj_pAskPx"].iloc[0] == pytest.approx(2.25)
        assert out["spot_px"].iloc[0] == pytest.approx(out["adj_stkPx"].iloc[0])


class TestOnSplitDate:
    def test_split_on_trade_date_not_applied(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                }
            ],
        )
        trade_date = pd.Timestamp("2020-02-01")
        df = _sample_options_df()

        out = adjuster.adjust_dataframe(df, trade_date)

        _assert_prices_equal_raw(out)


class TestAfterSplitDate:
    def test_trade_after_split_has_factor_one(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                }
            ],
        )
        trade_date = pd.Timestamp("2020-03-01")
        df = _sample_options_df()

        out = adjuster.adjust_dataframe(df, trade_date)

        _assert_prices_equal_raw(out)


class TestMultipleFutureSplits:
    def test_future_splits_multiply(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                },
                {
                    "ticker": "AAA",
                    "split_date": "2020-04-01",
                    "divisor": 3.0,
                },
            ],
        )
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df()

        out = adjuster.adjust_dataframe(df, trade_date)

        assert out["split_factor"].iloc[0] == pytest.approx(6.0)
        assert out["adj_stkPx"].iloc[0] == pytest.approx(100.0 / 6.0)
        assert out["adj_strike"].iloc[0] == pytest.approx(100.0 / 6.0)
        assert out["adj_pBidPx"].iloc[0] == pytest.approx(4.0 / 6.0)


class TestMultipleTickers:
    def test_each_ticker_gets_independent_factor(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-06-01",
                    "divisor": 2.0,
                },
                {
                    "ticker": "BBB",
                    "split_date": "2020-08-01",
                    "divisor": 4.0,
                },
            ],
        )
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df(
            [
                {"ticker": "AAA", "stkPx": 80.0, "strike": 80.0},
                {"ticker": "BBB", "stkPx": 40.0, "strike": 40.0},
                {"ticker": "CCC", "stkPx": 20.0, "strike": 20.0},
            ]
        )

        out = adjuster.adjust_dataframe(df, trade_date)

        aaa = out[out["ticker"] == "AAA"].iloc[0]
        bbb = out[out["ticker"] == "BBB"].iloc[0]
        ccc = out[out["ticker"] == "CCC"].iloc[0]

        assert aaa["split_factor"] == pytest.approx(2.0)
        assert aaa["adj_stkPx"] == pytest.approx(40.0)

        assert bbb["split_factor"] == pytest.approx(4.0)
        assert bbb["adj_stkPx"] == pytest.approx(10.0)

        assert ccc["split_factor"] == pytest.approx(1.0)
        assert ccc["adj_stkPx"] == pytest.approx(20.0)


class TestMissingOptionalPriceColumn:
    def test_omitted_price_column_does_not_crash(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                }
            ],
        )
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df()
        df = df.drop(columns=["pAskPx"])

        out = adjuster.adjust_dataframe(df, trade_date)

        assert "adj_pAskPx" not in out.columns
        assert out["adj_stkPx"].iloc[0] == pytest.approx(50.0)
        assert out["adj_pBidPx"].iloc[0] == pytest.approx(2.0)


class TestSpotPxAlias:
    def test_spot_px_equals_adj_stkpx(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2020-02-01",
                    "divisor": 2.0,
                }
            ],
        )
        trade_date = pd.Timestamp("2020-01-01")
        df = _sample_options_df([{"ticker": "AAA", "stkPx": 120.0, "strike": 100.0}])

        out = adjuster.adjust_dataframe(df, trade_date)

        assert "adj_stkPx" in out.columns
        assert out["spot_px"].iloc[0] == pytest.approx(out["adj_stkPx"].iloc[0])


class TestMinSplitDate:
    def test_splits_before_min_split_date_are_ignored(self, tmp_path: Path):
        adjuster = _make_adjuster(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "split_date": "2013-06-01",
                    "divisor": 10.0,
                }
            ],
            min_split_date="2014-01-01",
        )
        trade_date = pd.Timestamp("2012-01-01")
        df = _sample_options_df([{"ticker": "AAA", "stkPx": 100.0, "strike": 100.0}])

        out = adjuster.adjust_dataframe(df, trade_date)

        assert out["split_factor"].iloc[0] == pytest.approx(1.0)
        assert out["adj_stkPx"].iloc[0] == pytest.approx(100.0)


# ── C8.3B exact-ZIP-list entry point ──────────────────────────────────────────


def _write_raw_zip(
    tmp_path: Path,
    date_str: str,
    csv_text: str = "ticker,stkPx\nAAA,100.0\n",
) -> Path:
    year_dir = tmp_path / "raw" / date_str[:4]
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"ORATS_SMV_Strikes_{date_str}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("strikes.csv", csv_text)
    return path


class TestProcessZipPaths:
    def test_processes_exactly_supplied_paths_and_returns_outputs(
        self, tmp_path: Path
    ):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        zips = [
            _write_raw_zip(tmp_path, "20240405"),
            _write_raw_zip(tmp_path, "20240412"),
        ]
        # An extra ZIP in the raw tree must NOT be picked up.
        _write_raw_zip(tmp_path, "20240419")

        produced = adjuster.process_zip_paths(zips)

        assert produced == sorted(
            tmp_path / "adj" / "2024" / f"ORATS_SMV_Strikes_{d}.parquet"
            for d in ("20240405", "20240412")
        )
        assert all(p.is_file() for p in produced)
        assert not (
            tmp_path / "adj" / "2024" / "ORATS_SMV_Strikes_20240419.parquet"
        ).exists()

    def test_one_failing_zip_raises_and_reports_the_file(self, tmp_path: Path):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        good = _write_raw_zip(tmp_path, "20240405")
        bad = _write_raw_zip(tmp_path, "20240412", csv_text="foo,bar\n1.0,2.0\n")

        with pytest.raises(SplitAdjustmentError, match="20240412"):
            adjuster.process_zip_paths([good, bad])

    def test_missing_zip_path_raises(self, tmp_path: Path):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        good = _write_raw_zip(tmp_path, "20240405")
        missing = tmp_path / "raw" / "2024" / "ORATS_SMV_Strikes_20240412.zip"
        missing.unlink(missing_ok=True)

        with pytest.raises(SplitAdjustmentError, match="do not exist"):
            adjuster.process_zip_paths([good, missing])

    def test_duplicate_zip_paths_raise(self, tmp_path: Path):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        good = _write_raw_zip(tmp_path, "20240405")

        with pytest.raises(SplitAdjustmentError, match="duplicate"):
            adjuster.process_zip_paths([good, good])

    def test_empty_list_raises(self, tmp_path: Path):
        adjuster = _make_adjuster(tmp_path, split_rows=[])

        with pytest.raises(SplitAdjustmentError, match="at least one"):
            adjuster.process_zip_paths([])

    def test_unexpected_existing_output_raises_without_overwrite(
        self, tmp_path: Path
    ):
        adjuster = _make_adjuster(tmp_path, split_rows=[])
        zip_path = _write_raw_zip(tmp_path, "20240405")
        out_dir = tmp_path / "adj" / "2024"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ORATS_SMV_Strikes_20240405.parquet").write_bytes(b"stale")

        with pytest.raises(SplitAdjustmentError, match="unexpected skip"):
            adjuster.process_zip_paths([zip_path])
