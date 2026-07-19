"""Unit tests for the refresh_weekly_inputs CLI (Sprint 004 C8.3B contract).

Strategy
--------
* Load ``scripts/refresh_weekly_inputs.py`` via importlib (not a package).
* Call ``main(argv)`` directly and assert exit codes + stdout/stderr content.
* No subprocess, no real data, no network — C8.3B blocks real execution, so
  every test also asserts the blocked paths perform no filesystem mutation.

Exit-code contract: 0 plan/dry-run · 1 future runtime/gate failure ·
2 usage/config/corruption/unsupported · 130 interrupt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "refresh_weekly_inputs.py"

BUILD_ID = "20260717T220000123456Z_abcdef01"


@pytest.fixture
def cli_module():
    """Import the CLI script as a module so tests can call ``main()`` directly."""
    spec = importlib.util.spec_from_file_location("refresh_weekly_inputs", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshots_root(tmp_path: Path) -> Path:
    root = tmp_path / "snapshots"
    root.mkdir()
    return root


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    return root


def _run_main(cli_module, argv: list[str], *, capsys) -> tuple[int, str, str]:
    exit_code = cli_module.main(argv)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def _new_run_argv(
    snapshots_root: Path, raw_root: Path, *, extra: list[str] | None = None
) -> list[str]:
    argv = [
        "refresh",
        "--mode",
        "backfill",
        "--snapshots-root",
        str(snapshots_root),
        "--raw-root",
        str(raw_root),
        "--start-date",
        "2024-01-05",
        "--as-of",
        "2024-01-07",
        "--workers",
        "4",
    ]
    if extra:
        argv.extend(extra)
    return argv


def _assert_no_snapshot_state(snapshots_root: Path, raw_root: Path) -> None:
    """The blocked/dry paths must create no .building, config, inventory, or lock."""
    assert list(snapshots_root.iterdir()) == []
    assert list(raw_root.iterdir()) == []


# ── plan ───────────────────────────────────────────────────────────────────────


class TestPlan:
    def test_plan_returns_zero_and_describes_four_stage_cold_plan(
        self, cli_module, capsys
    ):
        code, out, _err = _run_main(cli_module, ["plan"], capsys=capsys)
        assert code == 0
        # Stage order: liquidity → adjusted → spot → surface → atomic publish.
        for token in (
            "1. liquidity",
            "2. adjusted",
            "3. spot",
            "4. surface",
            "atomic publish",
        ):
            assert token in out
        assert out.index("1. liquidity") < out.index("2. adjusted")
        assert out.index("2. adjusted") < out.index("3. spot")
        assert out.index("3. spot") < out.index("4. surface")
        assert out.index("4. surface") < out.index("atomic publish")

    def test_plan_labels_incremental_and_repair_unsupported(self, cli_module, capsys):
        code, out, _err = _run_main(cli_module, ["plan"], capsys=capsys)
        assert code == 0
        assert "'incremental' and 'repair' are execute-unsupported" in out
        assert "ORATS_API_TOKEN" in out

    def test_plan_echoes_supplied_run_arguments(
        self, cli_module, snapshots_root, raw_root, capsys
    ):
        code, out, _err = _run_main(
            cli_module,
            [
                "plan",
                "--mode",
                "backfill",
                "--snapshots-root",
                str(snapshots_root),
                "--raw-root",
                str(raw_root),
                "--start-date",
                "2024-01-05",
                "--as-of",
                "2024-01-07",
                "--workers",
                "4",
            ],
            capsys=capsys,
        )
        assert code == 0
        assert "2024-01-05" in out
        assert "2024-01-07" in out
        assert str(snapshots_root) in out
        _assert_no_snapshot_state(snapshots_root, raw_root)


# ── refresh --dry-run ──────────────────────────────────────────────────────────


class TestRefreshDryRun:
    def test_dry_run_returns_zero_with_banner_and_no_mutation(
        self, cli_module, snapshots_root, raw_root, capsys
    ):
        code, out, _err = _run_main(
            cli_module,
            _new_run_argv(snapshots_root, raw_root, extra=["--dry-run"]),
            capsys=capsys,
        )
        assert code == 0
        assert cli_module.DRY_RUN_BANNER in out
        assert "atomic publish" in out
        _assert_no_snapshot_state(snapshots_root, raw_root)

    def test_bare_dry_run_prints_plan_without_required_flags(self, cli_module, capsys):
        code, out, _err = _run_main(
            cli_module, ["refresh", "--dry-run"], capsys=capsys
        )
        assert code == 0
        assert "1. liquidity" in out

    def test_dry_run_does_not_import_producers_or_orchestrator(self, cli_module):
        # The CLI must not be able to scan, lock, or execute in this commit.
        assert "subprocess" not in cli_module.__dict__
        assert "snapshot_orchestrator" not in cli_module.__dict__
        assert not hasattr(cli_module, "prepare_new_backfill_run")


# ── refresh new-run contract ───────────────────────────────────────────────────


class TestRefreshNewRunContract:
    def test_valid_backfill_is_blocked_with_exit_two_and_no_state(
        self, cli_module, snapshots_root, raw_root, capsys
    ):
        code, _out, err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 2
        assert "deferred to the next C8.3B commit" in err
        _assert_no_snapshot_state(snapshots_root, raw_root)

    def test_backfill_requires_all_identity_flags(
        self, cli_module, snapshots_root, capsys
    ):
        code, _out, err = _run_main(
            cli_module,
            [
                "refresh",
                "--mode",
                "backfill",
                "--snapshots-root",
                str(snapshots_root),
            ],
            capsys=capsys,
        )
        assert code == 2
        for flag in ("--raw-root", "--start-date", "--as-of"):
            assert flag in err

    def test_refresh_without_mode_or_resume_is_usage_error(self, cli_module, capsys):
        code, _out, err = _run_main(cli_module, ["refresh"], capsys=capsys)
        assert code == 2
        assert "--mode backfill" in err

    @pytest.mark.parametrize("mode", ["incremental", "repair"])
    def test_incremental_and_repair_remain_unsupported(
        self, cli_module, snapshots_root, raw_root, mode, capsys
    ):
        argv = _new_run_argv(snapshots_root, raw_root)
        argv[argv.index("backfill")] = mode
        code, _out, err = _run_main(cli_module, argv, capsys=capsys)
        assert code == 2
        assert "deferred" in err
        assert "backfill" in err
        _assert_no_snapshot_state(snapshots_root, raw_root)

    @pytest.mark.parametrize("workers", ["0", "-3"])
    def test_workers_must_be_positive(
        self, cli_module, snapshots_root, raw_root, workers, capsys
    ):
        argv = _new_run_argv(snapshots_root, raw_root)
        argv[argv.index("--workers") + 1] = workers
        code, _out, err = _run_main(cli_module, argv, capsys=capsys)
        assert code == 2
        assert "--workers must be a positive integer" in err

    def test_invalid_iso_date_is_usage_error(
        self, cli_module, snapshots_root, raw_root, capsys
    ):
        argv = _new_run_argv(snapshots_root, raw_root)
        argv[argv.index("--as-of") + 1] = "not-a-date"
        code, _out, _err = _run_main(cli_module, argv, capsys=capsys)
        assert code == 2

    @pytest.mark.parametrize(
        "flag",
        [
            "--copy-source",
            "--evidence-id",
            "--scope",
            "--skip-stage",
            "--skip-surface",
            "--skip-splits",
            "--sample-tickers",
            "--security-master",
            "--token",
            "--orats-token",
        ],
    )
    def test_forbidden_flags_are_not_exposed(
        self, cli_module, snapshots_root, raw_root, flag, capsys
    ):
        argv = _new_run_argv(snapshots_root, raw_root, extra=[flag, "x"])
        code, _out, _err = _run_main(cli_module, argv, capsys=capsys)
        assert code == 2  # argparse rejects unknown flags
        _assert_no_snapshot_state(snapshots_root, raw_root)


# ── refresh resume contract ────────────────────────────────────────────────────


class TestRefreshResumeContract:
    def test_resume_shape_is_validated_then_blocked_without_state(
        self, cli_module, snapshots_root, raw_root, capsys
    ):
        code, _out, err = _run_main(
            cli_module,
            [
                "refresh",
                "--resume",
                BUILD_ID,
                "--snapshots-root",
                str(snapshots_root),
                "--workers",
                "8",
            ],
            capsys=capsys,
        )
        assert code == 2
        assert "deferred to the next C8.3B commit" in err
        _assert_no_snapshot_state(snapshots_root, raw_root)

    @pytest.mark.parametrize(
        "identity_args",
        [
            ["--mode", "backfill"],
            ["--raw-root", "X:/raw"],
            ["--start-date", "2024-01-05"],
            ["--as-of", "2024-01-07"],
        ],
    )
    def test_resume_rejects_identity_defining_flags(
        self, cli_module, snapshots_root, identity_args, capsys
    ):
        code, _out, err = _run_main(
            cli_module,
            [
                "refresh",
                "--resume",
                BUILD_ID,
                "--snapshots-root",
                str(snapshots_root),
                *identity_args,
            ],
            capsys=capsys,
        )
        assert code == 2
        assert "identity-defining flags" in err
        assert identity_args[0] in err

    def test_resume_allows_different_workers(self, cli_module, snapshots_root, capsys):
        # --workers is not identity-defining; the failure is the blocked
        # execution, not a flag rejection.
        code, _out, err = _run_main(
            cli_module,
            [
                "refresh",
                "--resume",
                BUILD_ID,
                "--snapshots-root",
                str(snapshots_root),
                "--workers",
                "16",
            ],
            capsys=capsys,
        )
        assert code == 2
        assert "identity-defining" not in err
        assert "deferred" in err

    def test_resume_requires_snapshots_root(self, cli_module, capsys):
        code, _out, err = _run_main(
            cli_module, ["refresh", "--resume", BUILD_ID], capsys=capsys
        )
        assert code == 2
        assert "--snapshots-root" in err


# ── blocked subcommands ────────────────────────────────────────────────────────


class TestBlockedSubcommands:
    def test_validate_is_blocked_and_points_to_standalone_audits(
        self, cli_module, capsys
    ):
        code, _out, err = _run_main(cli_module, ["validate"], capsys=capsys)
        assert code == 2
        assert "audit_adjusted_liquid" in err

    def test_split_audit_is_blocked_and_points_to_standalone_audit(
        self, cli_module, capsys
    ):
        code, _out, err = _run_main(cli_module, ["split-audit"], capsys=capsys)
        assert code == 2
        assert "audit_adjusted_liquid" in err

    def test_surface_audit_is_blocked_and_points_to_standalone_audit(
        self, cli_module, capsys
    ):
        code, _out, err = _run_main(cli_module, ["surface-audit"], capsys=capsys)
        assert code == 2
        assert "audit_option_surface_artifacts" in err


# ── exit-code mapping ──────────────────────────────────────────────────────────


class TestExitCodes:
    def test_missing_subcommand_is_usage_error(self, cli_module, capsys):
        code, _out, _err = _run_main(cli_module, [], capsys=capsys)
        assert code == 2

    def test_keyboard_interrupt_maps_to_130(self, cli_module, monkeypatch, capsys):
        def _interrupt(_args):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_module, "cmd_plan", _interrupt)
        code, _out, err = _run_main(cli_module, ["plan"], capsys=capsys)
        assert code == 130
        assert "interrupted" in err

    def test_exit_code_constants(self, cli_module):
        assert cli_module.EXIT_OK == 0
        assert cli_module.EXIT_RUNTIME == 1
        assert cli_module.EXIT_USAGE == 2
        assert cli_module.EXIT_INTERRUPT == 130
