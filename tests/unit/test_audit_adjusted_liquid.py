"""Unit tests for the adjusted liquid audit CLI (Sprint 004 C5.8A).

No real ORATS data. All fixtures use ``tmp_path`` synthetic ZIP/parquet files.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "audit_adjusted_liquid.py"

_DATE_STR = "20200102"
_ZIP_NAME = f"ORATS_SMV_Strikes_{_DATE_STR}.zip"
_PARQUET_NAME = f"ORATS_SMV_Strikes_{_DATE_STR}.parquet"
_CSV_NAME = f"ORATS_SMV_Strikes_{_DATE_STR}.csv"
_UNIVERSE = ["AAA", "BBB"]


@pytest.fixture
def cli_module():
    spec = importlib.util.spec_from_file_location("audit_adjusted_liquid", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_universe_csv(path: Path, tickers: list[str] | None = None) -> Path:
    tickers = tickers if tickers is not None else _UNIVERSE
    pd.DataFrame({"Ticker": tickers}).to_csv(path, index=False)
    return path


def _write_splits(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["split_date"] = pd.to_datetime(df["split_date"])
    df.to_parquet(path, index=False)
    return path


def _raw_csv_rows() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "expirDate": "2020-01-17",
            "stkPx": 100.0,
            "strike": 100.0,
            "cBidPx": 10.0,
            "cAskPx": 11.0,
            "pBidPx": 9.0,
            "pAskPx": 10.0,
        },
        {
            "ticker": "BBB",
            "expirDate": "2020-01-17",
            "stkPx": 50.0,
            "strike": 55.0,
            "cBidPx": 3.0,
            "cAskPx": 4.0,
            "pBidPx": 5.0,
            "pAskPx": 6.0,
        },
    ]


def _write_raw_zip(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    rows = rows if rows is not None else _raw_csv_rows()
    zip_dir = tmp_path / "raw" / "2020"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / _ZIP_NAME
    csv_buf = io.StringIO()
    pd.DataFrame(rows).to_csv(csv_buf, index=False)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(_CSV_NAME, csv_buf.getvalue().encode("utf-8"))
    return zip_path


def _valid_adjusted_rows(*, split_factor: float = 2.0, spot_override: float | None = None) -> list[dict]:
    sf = split_factor
    rows = []
    for raw in _raw_csv_rows():
        adj_stk = raw["stkPx"] / sf
        spot = spot_override if spot_override is not None else adj_stk
        rows.append(
            {
                "ticker": raw["ticker"],
                "expirDate": raw["expirDate"],
                "trade_date": pd.Timestamp("2020-01-02"),
                "split_factor": sf,
                "stkPx": raw["stkPx"],
                "strike": raw["strike"],
                "adj_stkPx": adj_stk,
                "adj_strike": raw["strike"] / sf,
                "spot_px": spot,
                "cBidPx": raw["cBidPx"],
                "cAskPx": raw["cAskPx"],
                "pBidPx": raw["pBidPx"],
                "pAskPx": raw["pAskPx"],
                "adj_cBidPx": raw["cBidPx"] / sf,
                "adj_cAskPx": raw["cAskPx"] / sf,
                "adj_pBidPx": raw["pBidPx"] / sf,
                "adj_pAskPx": raw["pAskPx"] / sf,
            }
        )
    return rows


def _write_adjusted_parquet(
    tmp_path: Path,
    rows: list[dict] | None = None,
) -> Path:
    rows = rows if rows is not None else _valid_adjusted_rows()
    adj_dir = tmp_path / "adj" / "2020"
    adj_dir.mkdir(parents=True, exist_ok=True)
    out = adj_dir / _PARQUET_NAME
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _passing_fixture(tmp_path: Path) -> dict[str, Path]:
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv")
    splits = _write_splits(
        tmp_path / "splits.parquet",
        [{"ticker": "AAA", "split_date": "2020-02-01", "divisor": 2.0}],
    )
    _write_raw_zip(tmp_path)
    _write_adjusted_parquet(tmp_path)
    return {
        "raw_root": tmp_path / "raw",
        "adj_root": tmp_path / "adj",
        "splits": splits,
        "universe": universe,
    }


# ── 1. split file audit pass ──────────────────────────────────────────────────

def test_split_file_audit_passes_valid_scoped_file(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    universe = set(_UNIVERSE)
    result = cli_module.audit_split_file(fx["splits"], universe)
    assert result.status == "PASS"
    assert result.metrics["split_row_count"] == 1
    assert result.metrics["outside_universe_split_ticker_count"] == 0


# ── 2. split file outside universe ────────────────────────────────────────────

def test_split_file_audit_fails_outside_universe_ticker(cli_module, tmp_path):
    splits = _write_splits(
        tmp_path / "splits.parquet",
        [{"ticker": "ZZZ", "split_date": "2020-02-01", "divisor": 2.0}],
    )
    result = cli_module.audit_split_file(splits, set(_UNIVERSE))
    assert result.status == "FAIL"
    assert result.metrics["outside_universe_split_ticker_count"] == 1


# ── 3. split file bad divisor ─────────────────────────────────────────────────

def test_split_file_audit_fails_nonpositive_or_null_divisor(cli_module, tmp_path):
    splits = _write_splits(
        tmp_path / "splits.parquet",
        [
            {"ticker": "AAA", "split_date": "2020-02-01", "divisor": 0.0},
            {"ticker": "BBB", "split_date": "2020-03-01", "divisor": None},
        ],
    )
    result = cli_module.audit_split_file(splits, set(_UNIVERSE))
    assert result.status == "FAIL"
    assert result.metrics["null_divisor_count"] >= 1
    assert result.metrics["nonpositive_divisor_count"] >= 1


# ── 4. inventory missing adjusted ─────────────────────────────────────────────

def test_inventory_audit_detects_missing_adjusted_file(cli_module, tmp_path):
    _write_raw_zip(tmp_path)
    result = cli_module.audit_inventory(
        tmp_path / "raw",
        tmp_path / "adj",
        [2020],
    )
    assert result.status == "FAIL"
    yr = result.metrics["years"][0]
    assert yr["missing_adjusted_count"] == 1
    assert yr["raw_zip_count"] == 1
    assert yr["adjusted_parquet_count"] == 0


# ── 5. universe containment fail ──────────────────────────────────────────────

def test_universe_containment_fails_outside_adjusted_ticker(cli_module, tmp_path):
    rows = _valid_adjusted_rows()
    rows.append(
        {
            **rows[0],
            "ticker": "ZZZ",
        }
    )
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_universe_containment(
        tmp_path / "adj", [2020], set(_UNIVERSE)
    )
    assert result.status == "FAIL"
    assert result.metrics["outside_universe_ticker_count"] == 1
    assert "ZZZ" in result.metrics["outside_universe_examples"]


# ── 6. bad split_factor ───────────────────────────────────────────────────────

def test_adjusted_structure_fails_bad_split_factor(cli_module, tmp_path):
    rows = _valid_adjusted_rows(split_factor=2.0)
    rows[0]["split_factor"] = float("nan")
    rows[1]["split_factor"] = 0.0
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_adjusted_structure(tmp_path / "adj", [2020])
    assert result.status == "FAIL"
    assert result.metrics["bad_split_factor_count"] >= 2


# ── 7. spot_px mismatch ───────────────────────────────────────────────────────

def test_adjusted_structure_fails_spot_px_mismatch(cli_module, tmp_path):
    rows = _valid_adjusted_rows(split_factor=2.0, spot_override=999.0)
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_adjusted_structure(tmp_path / "adj", [2020])
    assert result.status == "FAIL"
    assert result.metrics["spot_px_mismatch_count"] >= 1


# ── 8. trade_date mismatch ────────────────────────────────────────────────────

def test_adjusted_structure_fails_trade_date_mismatch(cli_module, tmp_path):
    rows = _valid_adjusted_rows()
    rows[0]["trade_date"] = pd.Timestamp("2020-01-03")
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_adjusted_structure(tmp_path / "adj", [2020])
    assert result.status == "FAIL"
    assert result.metrics["trade_date_mismatch_count"] >= 1


def test_adjusted_structure_fails_missing_optional_adjusted_column(cli_module, tmp_path):
    adj_dir = tmp_path / "adj" / "2020"
    adj_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_valid_adjusted_rows()).drop(columns=["adj_cBidPx"])
    df.to_parquet(adj_dir / _PARQUET_NAME, index=False)
    result = cli_module.audit_adjusted_structure(tmp_path / "adj", [2020])
    assert result.status == "FAIL"
    assert result.metrics["missing_optional_adjusted_columns"] >= 1
    assert any("adj_cBidPx" in f for f in result.failures)


def test_adjusted_structure_fails_nonfinite_optional_adjusted_price(cli_module, tmp_path):
    rows = _valid_adjusted_rows()
    rows[0]["adj_cAskPx"] = float("nan")
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_adjusted_structure(tmp_path / "adj", [2020])
    assert result.status == "FAIL"
    assert result.metrics["bad_optional_adjusted_price_count"] >= 1
    assert any("adj_cAskPx" in f for f in result.failures)


# ── 9. raw math pass ──────────────────────────────────────────────────────────

def test_raw_math_spot_check_passes_valid_fixture(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    result = cli_module.audit_raw_math_sample(
        fx["raw_root"],
        fx["adj_root"],
        [2020],
        set(_UNIVERSE),
        sample_files=10,
        sample_rows=20000,
        seed=57,
    )
    assert result.status == "PASS"
    assert result.metrics["math_mismatch_count"] == 0
    assert result.metrics["matched_rows"] >= 1


# ── 10. raw math fail ─────────────────────────────────────────────────────────

def test_raw_math_spot_check_fails_bad_adjusted_math(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    rows = _valid_adjusted_rows()
    rows[0]["adj_stkPx"] = 999.0
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_raw_math_sample(
        fx["raw_root"],
        fx["adj_root"],
        [2020],
        set(_UNIVERSE),
        sample_files=10,
        sample_rows=20000,
        seed=57,
    )
    assert result.status == "FAIL"
    assert result.metrics["math_mismatch_count"] >= 1


def test_raw_math_spot_check_fails_bad_optional_bid_ask_math(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    rows = _valid_adjusted_rows()
    rows[0]["adj_cBidPx"] = 999.0
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_raw_math_sample(
        fx["raw_root"],
        fx["adj_root"],
        [2020],
        set(_UNIVERSE),
        sample_files=10,
        sample_rows=20000,
        seed=57,
    )
    assert result.status == "FAIL"
    assert result.metrics["math_mismatch_count"] >= 1
    assert any("adj_cBidPx" in f for f in result.failures)


def test_raw_math_spot_check_normalizes_adjusted_ticker_before_merge(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    rows = _valid_adjusted_rows()
    rows[0]["ticker"] = " aaa "
    _write_adjusted_parquet(tmp_path, rows=rows)
    result = cli_module.audit_raw_math_sample(
        fx["raw_root"],
        fx["adj_root"],
        [2020],
        set(_UNIVERSE),
        sample_files=10,
        sample_rows=20000,
        seed=57,
    )
    assert result.status == "PASS"
    assert result.metrics["matched_rows"] >= 1
    assert result.metrics["unmatched_rows"] == 0


# ── 11. main writes report ────────────────────────────────────────────────────

def test_main_writes_markdown_report(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    report = tmp_path / "report.md"
    cli_module.main(
        [
            "--raw-root", str(fx["raw_root"]),
            "--adj-root", str(fx["adj_root"]),
            "--splits", str(fx["splits"]),
            "--ticker-universe", str(fx["universe"]),
            "--years", "2020",
            "--sample-files", "1",
            "--sample-rows", "100",
            "--seed", "57",
            "--report-path", str(report),
        ]
    )
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Overall verdict" in text
    assert "Split file audit" in text
    assert "Adjusted output inventory audit" in text
    assert "Universe containment audit" in text
    assert "Adjusted-column structural audit" in text
    assert "Raw-vs-adjusted math spot-check" in text


# ── 12. no legacy mirror arg ──────────────────────────────────────────────────

def test_cli_has_no_legacy_adj_root_argument(cli_module):
    with pytest.raises(SystemExit):
        cli_module.parse_args(
            [
                "--raw-root", "r",
                "--adj-root", "a",
                "--splits", "s",
                "--ticker-universe", "u",
                "--years", "2020",
                "--report-path", "p",
                "--legacy-adj-root", "legacy",
            ]
        )
    parser = cli_module.parse_args(
        [
            "--raw-root", "r",
            "--adj-root", "a",
            "--splits", "s",
            "--ticker-universe", "u",
            "--years", "2020",
            "--report-path", "p",
        ]
    )
    assert not hasattr(parser, "legacy_adj_root")
