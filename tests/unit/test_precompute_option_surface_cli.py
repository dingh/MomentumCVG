"""Unit tests for precompute_option_surface CLI safety (Sprint 004 C6.1A)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.trading_day import orats_daily_parquet_path

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "precompute_option_surface.py"


@pytest.fixture
def cli_module():
    spec = importlib.util.spec_from_file_location("precompute_option_surface", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "adjusted_liquid"


@pytest.fixture
def seed_orats_day(data_root: Path):
    def _seed(day: date) -> None:
        path = orats_daily_parquet_path(data_root, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")

    return _seed


def _write_tickers_file(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Ticker": tickers}).to_csv(path, index=False)


def _argv(
    *,
    data_root: Path,
    output_root: Path,
    spot_db: Path,
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--spot-db-path",
        str(spot_db),
        "--frequency",
        "weekly",
        "--start-year",
        "2024",
        "--end-year",
        "2024",
    ]
    if extra:
        argv.extend(extra)
    return argv


def test_dry_run_prints_summary_and_writes_nothing(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
    capsys,
) -> None:
    seed_orats_day(date(2024, 1, 5))
    seed_orats_day(date(2024, 1, 12))

    output_root = tmp_path / "out"
    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")
    tickers_file = tmp_path / "tickers.csv"
    _write_tickers_file(tickers_file, ["AAPL", "MSFT"])

    code = cli_module.main(
        _argv(
            data_root=data_root,
            output_root=output_root,
            spot_db=spot_db,
            extra=[
                "--tickers-file",
                str(tickers_file),
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-14",
                "--dry-run",
            ],
        )
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Option surface precompute — dry run" in captured.out
    assert "requested_start_date: 2024-01-01" in captured.out
    assert "requested_end_date: 2024-01-14" in captured.out
    assert "resolved_entry_date_count:" in captured.out
    assert "ticker_count: 2" in captured.out
    assert str(data_root) in captured.out
    assert str(output_root) in captured.out
    assert str(spot_db) in captured.out
    assert "option_surface_meta_weekly_2024_2024.parquet" in captured.out
    assert "meta_exists: False" in captured.out
    assert list(output_root.glob("*.parquet")) == []


def test_overwrite_guard_blocks_without_flag(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
) -> None:
    seed_orats_day(date(2024, 1, 5))

    output_root = tmp_path / "out"
    output_root.mkdir()
    meta = output_root / "option_surface_meta_weekly_2024_2024.parquet"
    meta.write_bytes(b"existing")

    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")
    tickers_file = tmp_path / "tickers.csv"
    _write_tickers_file(tickers_file, ["AAPL"])

    with patch.object(cli_module, "Parallel") as parallel_mock:
        code = cli_module.main(
            _argv(
                data_root=data_root,
                output_root=output_root,
                spot_db=spot_db,
                extra=["--tickers-file", str(tickers_file)],
            )
        )

    assert code == 2
    parallel_mock.assert_not_called()


def test_overwrite_allows_existing_outputs(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
) -> None:
    seed_orats_day(date(2024, 1, 5))

    output_root = tmp_path / "out"
    output_root.mkdir()
    meta = output_root / "option_surface_meta_weekly_2024_2024.parquet"
    meta.write_bytes(b"existing")

    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")
    tickers_file = tmp_path / "tickers.csv"
    _write_tickers_file(tickers_file, ["AAPL"])

    fake_meta = [{"ticker": "AAPL", "surface_valid": False, "failure_reason": "no_spot_price"}]
    fake_quotes: list[dict] = []

    with patch.object(cli_module, "Parallel") as parallel_mock:
        parallel_instance = MagicMock()
        parallel_instance.return_value = [(fake_meta, fake_quotes)]
        parallel_mock.return_value = parallel_instance
        code = cli_module.main(
            _argv(
                data_root=data_root,
                output_root=output_root,
                spot_db=spot_db,
                extra=["--tickers-file", str(tickers_file), "--overwrite", "--workers", "1"],
            )
        )

    assert code == 0
    assert meta.exists()
    assert (output_root / "option_surface_quotes_weekly_2024_2024.parquet").exists()


def test_output_root_redirects_output_paths(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
    capsys,
) -> None:
    seed_orats_day(date(2024, 1, 5))

    output_root = tmp_path / "c6_smoke"
    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")

    cli_module.main(
        _argv(
            data_root=data_root,
            output_root=output_root,
            spot_db=spot_db,
            extra=["--tickers", "AAPL", "--dry-run"],
        )
    )

    captured = capsys.readouterr()
    assert f"output_root: {output_root}" in captured.out
    assert str(output_root / "option_surface_meta_weekly_2024_2024.parquet") in captured.out


def test_inline_tickers_scope(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
    capsys,
) -> None:
    seed_orats_day(date(2024, 1, 5))

    output_root = tmp_path / "out"
    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")

    cli_module.main(
        _argv(
            data_root=data_root,
            output_root=output_root,
            spot_db=spot_db,
            extra=["--tickers", "AAPL", "MSFT", "NVDA", "--dry-run"],
        )
    )

    captured = capsys.readouterr()
    assert "ticker_source: inline --tickers" in captured.out
    assert "ticker_count: 3" in captured.out
    assert "tickers: AAPL, MSFT, NVDA" in captured.out


def test_bounded_date_scope_reduces_schedule(
    cli_module,
    data_root: Path,
    seed_orats_day,
    tmp_path: Path,
    capsys,
) -> None:
    seed_orats_day(date(2024, 1, 5))
    seed_orats_day(date(2024, 1, 12))
    seed_orats_day(date(2024, 1, 19))

    output_root = tmp_path / "out"
    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")

    cli_module.main(
        _argv(
            data_root=data_root,
            output_root=output_root,
            spot_db=spot_db,
            extra=[
                "--tickers",
                "AAPL",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-10",
                "--dry-run",
            ],
        )
    )

    captured = capsys.readouterr()
    assert "resolved_entry_date_count: 1" in captured.out
    assert "resolved_schedule_max: 2024-01-05" in captured.out


def test_tickers_and_tickers_file_mutually_exclusive(cli_module, data_root: Path, tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    spot_db = tmp_path / "spot.parquet"
    spot_db.write_bytes(b"spot")
    tickers_file = tmp_path / "tickers.csv"
    _write_tickers_file(tickers_file, ["AAPL"])

    with pytest.raises(SystemExit) as exc:
        cli_module.main(
            _argv(
                data_root=data_root,
                output_root=output_root,
                spot_db=spot_db,
                extra=[
                    "--tickers",
                    "AAPL",
                    "--tickers-file",
                    str(tickers_file),
                    "--dry-run",
                ],
            )
        )
    assert exc.value.code == 2


def test_load_tickers_inline(cli_module) -> None:
    args = cli_module.parse_args(["--tickers", "AAPL", "MSFT", "NVDA"])
    tickers, source = cli_module.load_tickers(args)
    assert tickers == ["AAPL", "MSFT", "NVDA"]
    assert source == "inline --tickers"


def test_load_tickers_from_file(cli_module, tmp_path: Path) -> None:
    tickers_file = tmp_path / "tickers.csv"
    _write_tickers_file(tickers_file, ["QQQ", "SPY"])
    args = cli_module.parse_args(["--tickers-file", str(tickers_file)])
    tickers, source = cli_module.load_tickers(args)
    assert tickers == ["QQQ", "SPY"]
    assert source == str(tickers_file)


def test_resolve_date_bounds_default_full_year(cli_module) -> None:
    args = cli_module.parse_args(["--start-year", "2024", "--end-year", "2024"])
    start, end = cli_module.resolve_date_bounds(args)
    assert start == date(2024, 1, 1)
    assert end == date(2024, 12, 31)
