"""Unit tests for audit_option_surface_artifacts CLI (Sprint 004 C6.2)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.features.option_surface_analyzer import _metadata_failure_row, _metadata_success_row

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "audit_option_surface_artifacts.py"


@pytest.fixture
def cli_module():
    spec = importlib.util.spec_from_file_location("audit_option_surface_artifacts", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_success_pair(tmp_path: Path) -> tuple[Path, Path]:
    meta_row = _metadata_success_row(
        ticker="AAPL",
        entry_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        dte_target=7,
        frequency="weekly",
        entry_spot=100.0,
        exit_spot=101.0,
        body_strike=100.0,
        spot_move_pct=0.01,
        realized_volatility=0.2,
        has_body_call=True,
        has_body_put=True,
        n_surface_quotes=1,
        processing_time=0.1,
    )
    quote_row = {
        "ticker": "AAPL",
        "entry_date": date(2024, 1, 5),
        "expiry_date": date(2024, 1, 12),
        "entry_spot": 100.0,
        "body_strike": 100.0,
        "side": "call",
        "is_body": True,
        "is_otm": False,
        "strike": 100.0,
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "spread_pct": 0.18,
        "iv": 0.2,
        "delta": 0.5,
        "abs_delta": 0.5,
        "gamma": 0.01,
        "vega": 0.1,
        "theta": -0.01,
        "volume": 100,
        "open_interest": 1000,
    }
    meta_path = tmp_path / "meta.parquet"
    quotes_path = tmp_path / "quotes.parquet"
    pd.DataFrame([meta_row]).to_parquet(meta_path, index=False)
    pd.DataFrame([quote_row]).to_parquet(quotes_path, index=False)
    return meta_path, quotes_path


def _seed_weekly_schedule(data_root: Path, day: date) -> None:
    path = data_root / str(day.year) / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet")


def test_cli_passes_clean_fixture(tmp_path: Path, cli_module) -> None:
    meta_path, quotes_path = _write_success_pair(tmp_path)
    data_root = tmp_path / "adjusted"
    _seed_weekly_schedule(data_root, date(2024, 1, 5))
    _seed_weekly_schedule(data_root, date(2024, 1, 12))
    report_path = tmp_path / "report.md"

    code = cli_module.main(
        [
            "--meta-path",
            str(meta_path),
            "--quotes-path",
            str(quotes_path),
            "--frequency",
            "weekly",
            "--data-root",
            str(data_root),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--output-report",
            str(report_path),
        ]
    )
    assert code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "# C6.2 — Surface Artifact Contract and Audit Foundation" in text
    assert "## Verdict" in text
    assert "**PASS**" in text


def test_cli_fails_on_orphan_quote(tmp_path: Path, cli_module) -> None:
    meta_path, quotes_path = _write_success_pair(tmp_path)
    quotes = pd.read_parquet(quotes_path)
    orphan = quotes.iloc[0].copy()
    orphan["ticker"] = "MSFT"
    quotes = pd.concat([quotes, pd.DataFrame([orphan])], ignore_index=True)
    quotes.to_parquet(quotes_path, index=False)

    data_root = tmp_path / "adjusted"
    _seed_weekly_schedule(data_root, date(2024, 1, 5))
    report_path = tmp_path / "report.md"

    code = cli_module.main(
        [
            "--meta-path",
            str(meta_path),
            "--quotes-path",
            str(quotes_path),
            "--frequency",
            "weekly",
            "--data-root",
            str(data_root),
            "--output-report",
            str(report_path),
        ]
    )
    assert code == 1
    assert "**FAIL**" in report_path.read_text(encoding="utf-8")


def test_cli_missing_meta_returns_usage_error(tmp_path: Path, cli_module) -> None:
    report_path = tmp_path / "report.md"
    code = cli_module.main(
        [
            "--meta-path",
            str(tmp_path / "missing_meta.parquet"),
            "--quotes-path",
            str(tmp_path / "missing_quotes.parquet"),
            "--output-report",
            str(report_path),
        ]
    )
    assert code == 2


def test_cli_warn_on_unknown_failure_reason(tmp_path: Path, cli_module) -> None:
    meta_row = _metadata_failure_row(
        ticker="AAPL",
        entry_date=date(2024, 1, 5),
        dte_target=7,
        frequency="weekly",
        failure_reason="totally_unknown",
        processing_time=0.1,
    )
    meta_path = tmp_path / "meta.parquet"
    quotes_path = tmp_path / "quotes.parquet"
    pd.DataFrame([meta_row]).to_parquet(meta_path, index=False)
    pd.DataFrame(
        columns=[
            "ticker",
            "entry_date",
            "expiry_date",
            "entry_spot",
            "body_strike",
            "side",
            "is_body",
            "is_otm",
            "strike",
            "bid",
            "ask",
            "mid",
            "spread_pct",
            "iv",
            "delta",
            "abs_delta",
            "gamma",
            "vega",
            "theta",
            "volume",
            "open_interest",
        ]
    ).to_parquet(quotes_path, index=False)

    data_root = tmp_path / "adjusted"
    _seed_weekly_schedule(data_root, date(2024, 1, 5))
    report_path = tmp_path / "report.md"

    code = cli_module.main(
        [
            "--meta-path",
            str(meta_path),
            "--quotes-path",
            str(quotes_path),
            "--frequency",
            "weekly",
            "--data-root",
            str(data_root),
            "--output-report",
            str(report_path),
        ]
    )
    assert code == 0
    assert "**WARN**" in report_path.read_text(encoding="utf-8")


def test_cli_fail_on_warn_flag(tmp_path: Path, cli_module) -> None:
    meta_row = _metadata_failure_row(
        ticker="AAPL",
        entry_date=date(2024, 1, 5),
        dte_target=7,
        frequency="weekly",
        failure_reason="totally_unknown",
        processing_time=0.1,
    )
    meta_path = tmp_path / "meta.parquet"
    quotes_path = tmp_path / "quotes.parquet"
    pd.DataFrame([meta_row]).to_parquet(meta_path, index=False)
    pd.DataFrame(
        columns=[
            "ticker",
            "entry_date",
            "expiry_date",
            "entry_spot",
            "body_strike",
            "side",
            "is_body",
            "is_otm",
            "strike",
            "bid",
            "ask",
            "mid",
            "spread_pct",
            "iv",
            "delta",
            "abs_delta",
            "gamma",
            "vega",
            "theta",
            "volume",
            "open_interest",
        ]
    ).to_parquet(quotes_path, index=False)

    data_root = tmp_path / "adjusted"
    _seed_weekly_schedule(data_root, date(2024, 1, 5))
    report_path = tmp_path / "report.md"

    code = cli_module.main(
        [
            "--meta-path",
            str(meta_path),
            "--quotes-path",
            str(quotes_path),
            "--frequency",
            "weekly",
            "--data-root",
            str(data_root),
            "--output-report",
            str(report_path),
            "--fail-on-warn",
        ]
    )
    assert code == 1
