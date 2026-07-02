"""C5.5 — synthetic filtered ZIP-to-parquet tests for SplitAdjuster."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.split_adjuster import PRICE_COLS, SplitAdjuster

_DATE_STR = "20200102"
_ZIP_NAME = f"ORATS_SMV_Strikes_{_DATE_STR}.zip"
_CSV_NAME = f"ORATS_SMV_Strikes_{_DATE_STR}.csv"


def _write_splits(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "splits_hist.parquet"
    df = pd.DataFrame(rows)
    df["split_date"] = pd.to_datetime(df["split_date"])
    df.to_parquet(path, index=False)
    return path


def _sample_csv_rows() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "stkPx": 100.0,
            "strike": 100.0,
            "cBidPx": 10.0,
            "cAskPx": 11.0,
            "pBidPx": 9.0,
            "pAskPx": 10.0,
        },
        {
            "ticker": "BBB",
            "stkPx": 50.0,
            "strike": 55.0,
            "cBidPx": 3.0,
            "cAskPx": 4.0,
            "pBidPx": 5.0,
            "pAskPx": 6.0,
        },
        {
            "ticker": "CCC",
            "stkPx": 200.0,
            "strike": 200.0,
            "cBidPx": 20.0,
            "cAskPx": 22.0,
            "pBidPx": 18.0,
            "pAskPx": 19.0,
        },
    ]


def _write_zip(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    if rows is None:
        rows = _sample_csv_rows()
    zip_dir = tmp_path / "raw" / "2020"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / _ZIP_NAME

    csv_buf = io.StringIO()
    pd.DataFrame(rows).to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(_CSV_NAME, csv_bytes)

    return zip_path


def _make_adjuster(
    tmp_path: Path,
    *,
    ticker_universe: set[str] | list[str] | None = None,
    overwrite: bool = True,
) -> SplitAdjuster:
    splits_path = _write_splits(
        tmp_path,
        [{"ticker": "AAA", "split_date": "2020-02-01", "divisor": 2.0}],
    )
    return SplitAdjuster(
        raw_root=tmp_path / "raw",
        adj_root=tmp_path / "adj",
        splits_path=splits_path,
        overwrite=overwrite,
        ticker_universe=ticker_universe,
    )


def _expected_out_path(tmp_path: Path) -> Path:
    return tmp_path / "adj" / "2020" / f"ORATS_SMV_Strikes_{_DATE_STR}.parquet"


def test_process_zip_filters_to_ticker_universe(tmp_path: Path):
    zip_path = _write_zip(tmp_path)
    adjuster = _make_adjuster(tmp_path, ticker_universe={"AAA", "BBB"})

    out_path = adjuster.process_zip(zip_path)

    assert out_path == _expected_out_path(tmp_path)
    assert out_path.is_file()

    out = pd.read_parquet(out_path)
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert "CCC" not in set(out["ticker"])
    assert len(out) == 2


def test_filtered_output_has_correct_split_adjustments(tmp_path: Path):
    zip_path = _write_zip(tmp_path)
    adjuster = _make_adjuster(tmp_path, ticker_universe={"AAA", "BBB"})
    adjuster.process_zip(zip_path)

    out = pd.read_parquet(_expected_out_path(tmp_path))
    aaa = out[out["ticker"] == "AAA"].iloc[0]
    bbb = out[out["ticker"] == "BBB"].iloc[0]

    assert aaa["split_factor"] == pytest.approx(2.0)
    assert aaa["adj_stkPx"] == pytest.approx(50.0)
    assert aaa["adj_strike"] == pytest.approx(50.0)
    assert aaa["adj_cBidPx"] == pytest.approx(5.0)
    assert aaa["adj_cAskPx"] == pytest.approx(5.5)
    assert aaa["adj_pBidPx"] == pytest.approx(4.5)
    assert aaa["adj_pAskPx"] == pytest.approx(5.0)
    assert aaa["spot_px"] == pytest.approx(aaa["adj_stkPx"])

    assert bbb["split_factor"] == pytest.approx(1.0)
    for col in PRICE_COLS:
        assert bbb[f"adj_{col}"] == pytest.approx(bbb[col])
    assert bbb["spot_px"] == pytest.approx(bbb["adj_stkPx"])


def test_process_zip_without_ticker_universe_preserves_existing_all_rows_behavior(
    tmp_path: Path,
):
    zip_path = _write_zip(tmp_path)
    adjuster = _make_adjuster(tmp_path, ticker_universe=None)

    adjuster.process_zip(zip_path)

    out = pd.read_parquet(_expected_out_path(tmp_path))
    assert set(out["ticker"]) == {"AAA", "BBB", "CCC"}
    assert len(out) == 3


def test_process_zip_overwrite_false_skips_existing_output(tmp_path: Path):
    zip_path = _write_zip(tmp_path)
    adjuster = _make_adjuster(tmp_path, ticker_universe={"AAA", "BBB"}, overwrite=True)
    first = adjuster.process_zip(zip_path)
    assert first is not None

    out_path = _expected_out_path(tmp_path)
    first_mtime = out_path.stat().st_mtime
    first_contents = pd.read_parquet(out_path).copy()

    time.sleep(0.05)

    skip_adjuster = _make_adjuster(
        tmp_path, ticker_universe={"AAA", "BBB"}, overwrite=False
    )
    second = skip_adjuster.process_zip(zip_path)

    assert second is None
    assert out_path.stat().st_mtime == first_mtime
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), first_contents)


def test_process_zip_overwrite_true_rewrites_existing_output(tmp_path: Path):
    zip_path = _write_zip(tmp_path)
    adjuster = _make_adjuster(tmp_path, ticker_universe={"AAA", "BBB"}, overwrite=True)
    adjuster.process_zip(zip_path)

    out_path = _expected_out_path(tmp_path)
    first = pd.read_parquet(out_path)
    assert len(first) == 2

    filtered_rows = [row for row in _sample_csv_rows() if row["ticker"] != "BBB"]
    _write_zip(tmp_path, rows=filtered_rows)

    rewrite_adjuster = _make_adjuster(
        tmp_path, ticker_universe={"AAA", "BBB"}, overwrite=True
    )
    result = rewrite_adjuster.process_zip(zip_path)

    assert result == out_path
    second = pd.read_parquet(out_path)
    assert set(second["ticker"]) == {"AAA"}
    assert len(second) == 1


def _norm_ticker(value: object) -> str:
    return str(value).strip().upper()


def _whitespace_csv_rows() -> list[dict]:
    rows = _sample_csv_rows()
    rows[0] = {**rows[0], "ticker": " aaa "}
    rows[1] = {**rows[1], "ticker": "bbb"}
    return rows


def test_process_zip_matches_whitespace_and_lowercase_raw_tickers(tmp_path: Path):
    zip_path = _write_zip(tmp_path, rows=_whitespace_csv_rows())
    adjuster = _make_adjuster(tmp_path, ticker_universe={"AAA", "BBB"})
    adjuster.process_zip(zip_path)

    out = pd.read_parquet(_expected_out_path(tmp_path))
    assert len(out) == 2
    assert set(out["ticker"].map(_norm_ticker)) == {"AAA", "BBB"}
    assert "CCC" not in set(out["ticker"].map(_norm_ticker))

    aaa = out[out["ticker"].map(_norm_ticker) == "AAA"].iloc[0]
    assert aaa["split_factor"] == pytest.approx(2.0)
    assert aaa["adj_stkPx"] == pytest.approx(50.0)
    assert aaa["adj_strike"] == pytest.approx(50.0)
    assert aaa["spot_px"] == pytest.approx(aaa["adj_stkPx"])


def test_empty_ticker_universe_raises_value_error(tmp_path: Path):
    splits_path = _write_splits(
        tmp_path,
        [{"ticker": "AAA", "split_date": "2020-02-01", "divisor": 2.0}],
    )
    kwargs = {
        "raw_root": tmp_path / "raw",
        "adj_root": tmp_path / "adj",
        "splits_path": splits_path,
    }

    with pytest.raises(ValueError, match="no valid tickers remain after cleaning"):
        SplitAdjuster(**kwargs, ticker_universe=[])

    with pytest.raises(ValueError, match="no valid tickers remain after cleaning"):
        SplitAdjuster(**kwargs, ticker_universe=["", "   ", None])


def test_none_ticker_universe_still_preserves_all_rows(tmp_path: Path):
    zip_path = _write_zip(tmp_path, rows=_whitespace_csv_rows())
    adjuster = _make_adjuster(tmp_path, ticker_universe=None)
    adjuster.process_zip(zip_path)

    out = pd.read_parquet(_expected_out_path(tmp_path))
    assert len(out) == 3
    assert set(out["ticker"].map(_norm_ticker)) == {"AAA", "BBB", "CCC"}
