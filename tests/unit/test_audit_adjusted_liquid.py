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
            "cOpra": "AAA200117C00100000",
            "pOpra": "AAA200117P00100000",
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
            "cOpra": "BBB200117C00055000",
            "pOpra": "BBB200117P00055000",
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
                "cOpra": raw["cOpra"],
                "pOpra": raw["pOpra"],
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


def _spx_duplicate_raw_rows() -> list[dict]:
    """SPX monthly vs SPXW weekly: same 3-key, different OPRA and stkPx."""
    base = {
        "ticker": "SPX",
        "expirDate": "2/17/2023",
        "strike": 5025.0,
        "cBidPx": 0.05,
        "cAskPx": 0.15,
        "pBidPx": 1154.8,
        "pAskPx": 1167.6,
    }
    return [
        {
            **base,
            "stkPx": 3831.79,
            "cOpra": "SPX230217C05025000",
            "pOpra": "SPX230217P05025000",
        },
        {
            **base,
            "stkPx": 3831.75,
            "cOpra": "SPXW230217C05025000",
            "pOpra": "SPXW230217P05025000",
        },
    ]


def _adjusted_from_raw_rows(raw_rows: list[dict], *, split_factor: float = 1.0) -> list[dict]:
    sf = split_factor
    out = []
    for raw in raw_rows:
        adj_stk = raw["stkPx"] / sf
        out.append(
            {
                "ticker": raw["ticker"],
                "expirDate": raw["expirDate"],
                "cOpra": raw["cOpra"],
                "pOpra": raw["pOpra"],
                "trade_date": pd.Timestamp("2020-01-02"),
                "split_factor": sf,
                "stkPx": raw["stkPx"],
                "strike": raw["strike"],
                "adj_stkPx": adj_stk,
                "adj_strike": raw["strike"] / sf,
                "spot_px": adj_stk,
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
    return out


def test_raw_math_opra_aware_merge_passes_spx_spxw_duplicate_keys(cli_module, tmp_path):
    """Regression for C5.10C: 3-key merge false-fails; OPRA-aware merge passes."""
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["SPX"])
    splits = _write_splits(tmp_path / "splits.parquet", [])
    raw_rows = _spx_duplicate_raw_rows()
    _write_raw_zip(tmp_path, rows=raw_rows)
    _write_adjusted_parquet(tmp_path, rows=_adjusted_from_raw_rows(raw_rows))

    result = cli_module.audit_raw_math_sample(
        tmp_path / "raw",
        tmp_path / "adj",
        [2020],
        {"SPX"},
        sample_files=1,
        sample_rows=10,
        seed=57,
    )
    assert result.status == "PASS"
    assert result.metrics["math_mismatch_count"] == 0
    assert result.metrics["matched_rows"] <= result.metrics["sampled_rows"]
    assert result.metrics["matched_rows"] == result.metrics["sampled_rows"]
    assert result.metrics["join_key_columns_used"] == [
        "ticker",
        "expirDate",
        "strike",
        "cOpra",
        "pOpra",
    ]


def _write_zip_named(tmp_path: Path, date_str: str, rows: list[dict] | None = None) -> Path:
    rows = rows if rows is not None else _raw_csv_rows()
    zip_dir = tmp_path / "raw" / date_str[:4]
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"ORATS_SMV_Strikes_{date_str}.zip"
    csv_buf = io.StringIO()
    pd.DataFrame(rows).to_csv(csv_buf, index=False)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"ORATS_SMV_Strikes_{date_str}.csv", csv_buf.getvalue().encode("utf-8"))
    return zip_path


def _write_parquet_named(tmp_path: Path, date_str: str, rows: list[dict] | None = None) -> Path:
    rows = rows if rows is not None else _valid_adjusted_rows()
    adj_dir = tmp_path / "adj" / date_str[:4]
    adj_dir.mkdir(parents=True, exist_ok=True)
    out = adj_dir / f"ORATS_SMV_Strikes_{date_str}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _write_expected_dates(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── C8.3A: --expected-dates frozen inventory boundary ──────────────────────────


def test_expected_dates_legacy_behavior_unchanged_without_flag(cli_module, tmp_path):
    _passing_fixture(tmp_path)
    result = cli_module.audit_inventory(tmp_path / "raw", tmp_path / "adj", [2020])
    assert result.status == "PASS"
    yr = result.metrics["years"][0]
    assert yr["missing_adjusted_count"] == 0
    assert "expected_date_count" not in yr


def test_expected_dates_later_raw_outside_set_is_ignored(cli_module, tmp_path):
    _write_zip_named(tmp_path, "20200102")
    _write_zip_named(tmp_path, "20200109")  # later raw date, not expected
    _write_parquet_named(tmp_path, "20200102")

    result = cli_module.audit_inventory(
        tmp_path / "raw", tmp_path / "adj", [2020], {"20200102"}
    )
    assert result.status == "PASS"
    yr = result.metrics["years"][0]
    assert yr["expected_date_count"] == 1
    assert yr["missing_adjusted_count"] == 0
    assert yr["extra_adjusted_count"] == 0
    assert yr["missing_raw_count"] == 0


def test_expected_date_missing_raw_zip_fails(cli_module, tmp_path):
    _write_zip_named(tmp_path, "20200109")  # raw dir exists but not expected date
    _write_parquet_named(tmp_path, "20200102")

    result = cli_module.audit_inventory(
        tmp_path / "raw", tmp_path / "adj", [2020], {"20200102"}
    )
    assert result.status == "FAIL"
    assert any("missing raw" in f.lower() for f in result.failures)


def test_expected_date_missing_adjusted_parquet_fails(cli_module, tmp_path):
    _write_zip_named(tmp_path, "20200102")
    _write_parquet_named(tmp_path, "20200109")  # adj dir exists but not expected date

    result = cli_module.audit_inventory(
        tmp_path / "raw", tmp_path / "adj", [2020], {"20200102"}
    )
    assert result.status == "FAIL"
    assert any("missing" in f.lower() and "adjusted parquet" in f.lower() for f in result.failures)


def test_adjusted_date_outside_expected_set_fails(cli_module, tmp_path):
    _write_zip_named(tmp_path, "20200102")
    _write_zip_named(tmp_path, "20200103")
    _write_parquet_named(tmp_path, "20200102")
    _write_parquet_named(tmp_path, "20200103")  # outside expected set

    result = cli_module.audit_inventory(
        tmp_path / "raw", tmp_path / "adj", [2020], {"20200102"}
    )
    assert result.status == "FAIL"
    assert any("outside the expected set" in f for f in result.failures)


def test_duplicate_expected_date_fails(cli_module, tmp_path):
    path = _write_expected_dates(tmp_path / "exp.txt", ["2020-01-02", "2020-01-02"])
    with pytest.raises(cli_module.ExpectedDatesError, match="duplicate"):
        cli_module.parse_expected_dates(path)


def test_malformed_expected_date_fails(cli_module, tmp_path):
    path = _write_expected_dates(tmp_path / "exp.txt", ["2020-13-40"])
    with pytest.raises(cli_module.ExpectedDatesError, match="malformed"):
        cli_module.parse_expected_dates(path)


def test_expected_dates_parses_comments_and_blank_lines(cli_module, tmp_path):
    path = _write_expected_dates(
        tmp_path / "exp.txt",
        ["# header", "", "2020-01-03", "  2020-01-02  ", "# trailing"],
    )
    dates = cli_module.parse_expected_dates(path)
    assert [d.isoformat() for d in dates] == ["2020-01-02", "2020-01-03"]


def test_empty_expected_dates_file_fails(cli_module, tmp_path):
    path = _write_expected_dates(tmp_path / "exp.txt", ["# only comments", ""])
    with pytest.raises(cli_module.ExpectedDatesError, match="no dates"):
        cli_module.parse_expected_dates(path)


def test_math_sampling_limited_to_expected_dates(cli_module, tmp_path):
    # Two adjusted files; only the expected one has a matching raw ZIP.
    _write_zip_named(tmp_path, "20200102")
    _write_parquet_named(tmp_path, "20200102")
    _write_parquet_named(tmp_path, "20200103")  # not expected, no raw

    result = cli_module.audit_raw_math_sample(
        tmp_path / "raw",
        tmp_path / "adj",
        [2020],
        set(_UNIVERSE),
        sample_files=10,
        sample_rows=100,
        seed=57,
        expected_date_strs={"20200102"},
    )
    assert result.metrics["sampled_files"] == 1
    assert result.status == "PASS"
    assert result.metrics["matched_rows"] >= 1


def test_expected_dates_incompatible_years_fails_via_main(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)
    exp = _write_expected_dates(tmp_path / "exp.txt", ["2021-01-04"])  # year 2021
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--raw-root", str(fx["raw_root"]),
                "--adj-root", str(fx["adj_root"]),
                "--splits", str(fx["splits"]),
                "--ticker-universe", str(fx["universe"]),
                "--years", "2020",
                "--report-path", str(tmp_path / "report.md"),
                "--expected-dates", str(exp),
            ]
        )
    assert excinfo.value.code == 1


def test_expected_dates_happy_path_via_main(cli_module, tmp_path):
    fx = _passing_fixture(tmp_path)  # raw+adj for 20200102
    exp = _write_expected_dates(tmp_path / "exp.txt", ["2020-01-02"])
    report = tmp_path / "report.md"
    # PASS path returns None (no SystemExit).
    cli_module.main(
        [
            "--raw-root", str(fx["raw_root"]),
            "--adj-root", str(fx["adj_root"]),
            "--splits", str(fx["splits"]),
            "--ticker-universe", str(fx["universe"]),
            "--years", "2020",
            "--sample-files", "1",
            "--sample-rows", "100",
            "--report-path", str(report),
            "--expected-dates", str(exp),
        ]
    )
    assert report.is_file()


def test_raw_math_fails_on_fallback_duplicate_keys_without_opra(cli_module, tmp_path):
    """Fallback 3-key join must fail clearly when OPRA columns are absent."""
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["SPX"])
    splits = _write_splits(tmp_path / "splits.parquet", [])
    raw_rows = _spx_duplicate_raw_rows()
    adj_rows = _adjusted_from_raw_rows(raw_rows)
    _write_raw_zip(tmp_path, rows=raw_rows)
    _write_adjusted_parquet(tmp_path, rows=adj_rows)

    # Strip OPRA columns to force fallback join keys.
    zip_path = tmp_path / "raw" / "2020" / _ZIP_NAME
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw_df = pd.read_csv(zf.open(csv_name), dtype={"ticker": str})
    raw_df = raw_df.drop(columns=["cOpra", "pOpra"])
    csv_buf = io.StringIO()
    raw_df.to_csv(csv_buf, index=False)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(_CSV_NAME, csv_buf.getvalue().encode("utf-8"))

    adj_path = tmp_path / "adj" / "2020" / _PARQUET_NAME
    pd.DataFrame(adj_rows).drop(columns=["cOpra", "pOpra"]).to_parquet(adj_path, index=False)

    result = cli_module.audit_raw_math_sample(
        tmp_path / "raw",
        tmp_path / "adj",
        [2020],
        {"SPX"},
        sample_files=1,
        sample_rows=10,
        seed=57,
    )
    assert result.status == "FAIL"
    assert result.metrics["raw_duplicate_join_key_rows"] > 0
    assert result.metrics["math_mismatch_count"] == 0
    assert any("non-unique join keys" in f for f in result.failures)
