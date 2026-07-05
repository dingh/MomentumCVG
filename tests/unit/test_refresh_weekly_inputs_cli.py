"""Unit tests for refresh_weekly_inputs CLI skeleton (Sprint 004 C2).

Strategy
--------
* Load ``scripts/refresh_weekly_inputs.py`` via importlib (not a package).
* Seed minimal ORATS parquet paths under ``tmp_path`` so ``--as-of`` resolves.
* Call ``main(argv)`` directly and assert exit codes + stdout/stderr content.
* No subprocess, no real cache, no manifest writes — matches C2 scope.

Plan output assertions (Sprint 004 closeout)
--------------------------------------------
``test_plan_output_includes_required_sections`` locks *current* plan text. Some
strings are **C2–C7 scaffolding** (e.g. ``deferred to C3–C8``, ``Provisional``)
and must be **removed from the CLI** at closeout (blocker #13 in
``docs/agenda/current_sprint.md``). When ``render_plan()`` is cleaned up in C9,
update this test in the **same commit**: drop TEMP assertions; keep stable
contract checks (as-of fields, step names, artifact keys).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from src.data.input_snapshot import (
    ARTIFACT_LIQUIDITY_PANEL,
    ARTIFACT_OPTION_SURFACE_META,
    ARTIFACT_OPTION_SURFACE_QUOTES,
    ARTIFACT_SPLITS,
    ARTIFACT_SPOT_PRICES,
)
from src.data.trading_day import orats_daily_parquet_path

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "refresh_weekly_inputs.py"


@pytest.fixture
def cli_module():
    """Import CLI script as a module so tests can call ``main()`` without a shell."""
    spec = importlib.util.spec_from_file_location("refresh_weekly_inputs", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def orats_root(tmp_path: Path) -> Path:
    """Isolated fake ORATS adjusted root per test."""
    return tmp_path / "ORATS_Adjusted"


@pytest.fixture
def mock_orats_day(orats_root: Path):
    """Create an empty parquet file for ``day`` so resolver ``Path.is_file`` succeeds."""

    def _seed(day: date) -> None:
        path = orats_daily_parquet_path(orats_root, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")

    return _seed


def _argv(
    command: str,
    *,
    as_of: str = "2026-06-26",
    orats_root: Path,
    cache_dir: Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build argv list with required global flags and optional extras."""
    cache = cache_dir or (orats_root.parent / "cache")
    base = [
        command,
        "--as-of",
        as_of,
        "--orats-adj-root",
        str(orats_root),
        "--cache-dir",
        str(cache),
    ]
    if extra:
        base.extend(extra)
    return base


def _run_main(cli_module, argv: list[str], *, capsys) -> tuple[int, str, str]:
    """Invoke ``main(argv)`` and return (exit_code, stdout, stderr)."""
    exit_code = cli_module.main(argv)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


class TestPlanCommand:
    """``plan`` is the primary C2 success path — provisional output only."""

    def test_plan_returns_zero(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, _out, _err = _run_main(
            cli_module,
            _argv("plan", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 0

    def test_plan_output_includes_required_sections(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        # Permanent contract (keep at closeout): resolved as-of, step names, artifact keys.
        # TEMP scaffolding (remove assertions when render_plan() is cleaned up — C9):
        #   "Provisional ...", "deferred to C3–C8", Sprint 005 feature-branch deferral.
        mock_orats_day(date(2026, 6, 26))
        code, out, _err = _run_main(
            cli_module,
            _argv("plan", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 0

        # --- stable operator contract ---
        assert "as_of_requested:" in out
        assert "as_of_resolved_trading_day:" in out
        assert "resolve_candidate_universe_scope" in out
        assert "build_liquidity_panel" in out
        for key in (
            ARTIFACT_SPLITS,
            ARTIFACT_SPOT_PRICES,
            ARTIFACT_LIQUIDITY_PANEL,
            ARTIFACT_OPTION_SURFACE_META,
            ARTIFACT_OPTION_SURFACE_QUOTES,
        ):
            assert f"{key}:" in out

        # --- C2–C7 temporary plan copy (delete this block at Sprint 004 closeout) ---
        assert "Provisional high-level Stage A plan:" in out
        assert "deferred to C3–C8" in out
        assert "Feature branch (straddle history, build_features, A4): deferred to Sprint 005" in out


class TestRefreshDryRun:
    """``refresh --dry-run`` must mirror ``plan`` and never execute subprocesses."""

    def test_refresh_dry_run_returns_zero_and_includes_banner(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, out, _err = _run_main(
            cli_module,
            _argv("refresh", orats_root=orats_root, extra=["--dry-run"]),
            capsys=capsys,
        )
        assert code == 0
        assert "=== Weekly input refresh plan (no execution) ===" in out
        assert "DRY-RUN: no subprocesses executed" in out

    def test_refresh_dry_run_does_not_import_subprocess(self, cli_module):
        # C2 contract: no subprocess module — real wiring is C8.
        assert "subprocess" not in cli_module.__dict__


class TestStubCommands:
    """Not-implemented commands return exit 2 (HD-C2-3) with commit hint in stderr."""

    def test_validate_returns_two_and_mentions_c3(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, _out, err = _run_main(
            cli_module,
            _argv("validate", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2
        assert "C3" in err

    def test_split_audit_returns_two_and_points_to_standalone_audit(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, _out, err = _run_main(
            cli_module,
            _argv("split-audit", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2
        assert "C8" in err
        assert "audit_adjusted_liquid" in err

    def test_surface_audit_returns_two_and_mentions_c6(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, _out, err = _run_main(
            cli_module,
            _argv("surface-audit", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2
        assert "C6" in err

    def test_refresh_without_dry_run_returns_two_and_mentions_c8(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, _out, err = _run_main(
            cli_module,
            _argv("refresh", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2
        assert "C8" in err


class TestAsOfErrors:
    """Bad or unresolvable ``--as-of`` must surface as exit 2, not success."""

    def test_invalid_as_of_returns_two(self, cli_module, orats_root, capsys):
        code, _out, _err = _run_main(
            cli_module,
            _argv("plan", as_of="not-a-date", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2

    def test_missing_as_of_returns_two(self, cli_module, capsys):
        # argparse required-flag error → SystemExit(2) caught in main().
        code, _out, _err = _run_main(
            cli_module,
            ["plan", "--orats-adj-root", "C:/tmp/orats", "--cache-dir", "C:/tmp/cache"],
            capsys=capsys,
        )
        assert code == 2

    def test_unresolvable_orats_day_returns_two(
        self, cli_module, orats_root, capsys
    ):
        # Empty orats_root: no seeded parquet → resolver ValueError → stderr + exit 2.
        code, _out, err = _run_main(
            cli_module,
            _argv("plan", orats_root=orats_root),
            capsys=capsys,
        )
        assert code == 2
        assert "No ORATS adjusted daily parquet found" in err


class TestModeWarnings:
    """C2 displays mode gaps as WARN in plan stdout but still exits 0."""

    def test_backfill_without_dates_returns_zero_and_warns(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, out, _err = _run_main(
            cli_module,
            _argv(
                "plan",
                orats_root=orats_root,
                extra=["--mode", "backfill"],
            ),
            capsys=capsys,
        )
        assert code == 0
        assert "WARN: backfill date-window enforcement deferred until refresh execution (C8)" in out

    def test_repair_without_sample_tickers_returns_zero_and_warns(
        self, cli_module, orats_root, mock_orats_day, capsys
    ):
        mock_orats_day(date(2026, 6, 26))
        code, out, _err = _run_main(
            cli_module,
            _argv(
                "plan",
                orats_root=orats_root,
                extra=["--mode", "repair"],
            ),
            capsys=capsys,
        )
        assert code == 0
        assert "WARN: repair ticker-scope enforcement deferred until refresh execution (C8)" in out
