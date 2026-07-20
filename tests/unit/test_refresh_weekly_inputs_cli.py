"""Unit tests for the refresh_weekly_inputs CLI (Sprint 004 C8.3B contract).

Strategy
--------
* Load ``scripts/refresh_weekly_inputs.py`` via importlib (not a package).
* Call ``main(argv)`` directly and assert exit codes + stdout/stderr content.
* Execution paths use mocks / tiny temporary directories — never real producers
  or ORATS.

Exit-code contract: 0 published/plan/dry-run · 1 runtime/gate/lock/publish ·
2 usage/config/corruption/unsupported · 130 interrupt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.snapshot_foundation import SnapshotLockError, derive_snapshot_roots
from src.data.snapshot_orchestrator import RunConfigError
from src.data.snapshot_stage_adapters import StageExecutionError

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
    """Dry-run / usage paths must create no .building, config, inventory, or lock."""
    assert list(snapshots_root.iterdir()) == []
    assert list(raw_root.iterdir()) == []


def _fake_run(snapshots_root: Path, build_id: str = BUILD_ID):
    roots = derive_snapshot_roots(snapshots_root, build_id)
    return SimpleNamespace(
        build_id=build_id,
        snapshots_root=snapshots_root,
        roots=roots,
        run_config={"run_config_id": "x"},
    )


def _fake_manifest(snapshot_id: str = "abcd1234abcd1234"):
    return SimpleNamespace(snapshot_id=snapshot_id)


class _RecordingLock:
    """Test double for SiblingBuildLock with explicit call recording."""

    instances: list["_RecordingLock"] = []

    def __init__(self, snapshots_root, build_id):
        self.snapshots_root = Path(snapshots_root)
        self.build_id = build_id
        self.path = self.snapshots_root / f"{build_id}.lock"
        self.held = False
        self.calls: list[str] = []
        _RecordingLock.instances.append(self)

    def acquire(self):
        self.calls.append("acquire")
        self.held = True
        return self

    def release(self):
        self.calls.append("release")
        self.held = False


# ── plan ───────────────────────────────────────────────────────────────────────


class TestPlan:
    def test_plan_returns_zero_and_describes_four_stage_cold_plan(
        self, cli_module, capsys
    ):
        code, out, _err = _run_main(cli_module, ["plan"], capsys=capsys)
        assert code == 0
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

    def test_dry_run_does_not_execute_orchestrator(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        def boom(*_a, **_k):
            raise AssertionError("dry-run must not call orchestrator")

        monkeypatch.setattr(cli_module, "prepare_new_backfill_run", boom)
        monkeypatch.setattr(cli_module, "open_resume_run", boom)
        monkeypatch.setattr(cli_module, "execute_backfill_stages", boom)
        monkeypatch.setattr(cli_module, "finalize_candidate_snapshot", boom)
        monkeypatch.setattr(cli_module, "publish_candidate_snapshot", boom)
        monkeypatch.setattr(cli_module, "generate_snapshot_build_id", boom)
        monkeypatch.setattr(cli_module, "SiblingBuildLock", boom)
        code, out, _err = _run_main(
            cli_module,
            _new_run_argv(snapshots_root, raw_root, extra=["--dry-run"]),
            capsys=capsys,
        )
        assert code == 0
        assert cli_module.DRY_RUN_BANNER in out
        _assert_no_snapshot_state(snapshots_root, raw_root)


# ── refresh new-run / resume execution (mocked) ────────────────────────────────


class TestRefreshExecution:
    def _install_success_mocks(self, cli_module, monkeypatch, snapshots_root, calls):
        _RecordingLock.instances = []
        monkeypatch.setattr(cli_module, "SiblingBuildLock", _RecordingLock)
        monkeypatch.setattr(
            cli_module,
            "generate_snapshot_build_id",
            lambda: calls.append("generate") or BUILD_ID,
        )

        def prepare(**kwargs):
            assert _RecordingLock.instances[0].held
            assert _RecordingLock.instances[0].calls == ["acquire"]
            calls.append("prepare")
            assert kwargs["build_id"] == BUILD_ID
            return _fake_run(snapshots_root)

        def execute(run, lock, **kwargs):
            calls.append("execute")
            assert lock.held
            return {}

        def finalize(run, lock):
            calls.append("finalize")
            assert lock.held
            return _fake_manifest(), Path("manifest.json")

        def publish(run, lock):
            calls.append("publish")
            assert lock.held
            final = Path(snapshots_root) / BUILD_ID
            final.mkdir(exist_ok=True)
            return final

        monkeypatch.setattr(cli_module, "prepare_new_backfill_run", prepare)
        monkeypatch.setattr(cli_module, "execute_backfill_stages", execute)
        monkeypatch.setattr(cli_module, "finalize_candidate_snapshot", finalize)
        monkeypatch.setattr(cli_module, "publish_candidate_snapshot", publish)

    def test_new_run_call_order_lock_before_prepare(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        calls: list[str] = []
        self._install_success_mocks(cli_module, monkeypatch, snapshots_root, calls)
        code, out, _err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 0
        assert calls == ["generate", "prepare", "execute", "finalize", "publish"]
        lock = _RecordingLock.instances[0]
        assert lock.calls == ["acquire", "release"]
        assert "build_id:" in out and BUILD_ID in out
        assert "snapshot_id:" in out

    def test_resume_acquires_lock_before_open(
        self, cli_module, snapshots_root, monkeypatch, capsys
    ):
        calls: list[str] = []
        _RecordingLock.instances = []
        monkeypatch.setattr(cli_module, "SiblingBuildLock", _RecordingLock)

        def open_resume(snapshots_root_arg, build_id, **kwargs):
            calls.append("open")
            assert _RecordingLock.instances[0].held
            assert kwargs.get("rescan_raw") is True
            return _fake_run(Path(snapshots_root_arg), build_id)

        def execute(run, lock, **kwargs):
            calls.append("execute")
            return {}

        def finalize(run, lock):
            calls.append("finalize")
            return _fake_manifest(), Path("m.json")

        def publish(run, lock):
            calls.append("publish")
            return Path(snapshots_root) / BUILD_ID

        monkeypatch.setattr(cli_module, "open_resume_run", open_resume)
        monkeypatch.setattr(cli_module, "execute_backfill_stages", execute)
        monkeypatch.setattr(cli_module, "finalize_candidate_snapshot", finalize)
        monkeypatch.setattr(cli_module, "publish_candidate_snapshot", publish)

        code, out, _err = _run_main(
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
        assert code == 0
        assert calls == ["open", "execute", "finalize", "publish"]
        assert _RecordingLock.instances[0].calls == ["acquire", "release"]
        assert BUILD_ID in out

    def test_workers_reach_adjusted_and_surface(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        calls: list[str] = []
        self._install_success_mocks(cli_module, monkeypatch, snapshots_root, calls)
        seen = {}

        def execute(run, lock, **kwargs):
            seen.update(kwargs)
            calls.append("execute")
            return {}

        monkeypatch.setattr(cli_module, "execute_backfill_stages", execute)
        code, _out, _err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 0
        assert seen["max_workers"] == 4
        assert seen["surface_workers"] == 4

    def test_rename_failure_releases_lock_and_leaves_building(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        calls: list[str] = []
        self._install_success_mocks(cli_module, monkeypatch, snapshots_root, calls)
        building = snapshots_root / f"{BUILD_ID}.building"
        building.mkdir()

        def publish(run, lock):
            calls.append("publish")
            assert lock.held
            assert building.is_dir()
            raise OSError("rename failed")

        monkeypatch.setattr(cli_module, "publish_candidate_snapshot", publish)
        code, _out, err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 1
        assert "rename failed" in err
        assert building.is_dir()
        assert not _RecordingLock.instances[0].held
        assert _RecordingLock.instances[0].calls == ["acquire", "release"]

    def test_stage_failure_maps_to_exit_one(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        calls: list[str] = []
        self._install_success_mocks(cli_module, monkeypatch, snapshots_root, calls)

        def execute(run, lock, **kwargs):
            raise StageExecutionError("gate failed")

        monkeypatch.setattr(cli_module, "execute_backfill_stages", execute)
        code, _out, err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 1
        assert "gate failed" in err
        assert not _RecordingLock.instances[0].held

    def test_corruption_maps_to_exit_two(
        self, cli_module, snapshots_root, monkeypatch, capsys
    ):
        _RecordingLock.instances = []
        monkeypatch.setattr(cli_module, "SiblingBuildLock", _RecordingLock)

        def open_resume(*_a, **_k):
            raise RunConfigError("marker corruption")

        monkeypatch.setattr(cli_module, "open_resume_run", open_resume)
        code, _out, err = _run_main(
            cli_module,
            [
                "refresh",
                "--resume",
                BUILD_ID,
                "--snapshots-root",
                str(snapshots_root),
            ],
            capsys=capsys,
        )
        assert code == 2
        assert "marker corruption" in err
        assert not _RecordingLock.instances[0].held

    def test_lock_contention_maps_to_exit_one(
        self, cli_module, snapshots_root, raw_root, monkeypatch, capsys
    ):
        class BusyLock:
            def __init__(self, *_a, **_k):
                self.held = False

            def acquire(self):
                raise SnapshotLockError("another process holds the snapshot build lock")

            def release(self):
                self.held = False

        monkeypatch.setattr(cli_module, "SiblingBuildLock", BusyLock)
        monkeypatch.setattr(cli_module, "generate_snapshot_build_id", lambda: BUILD_ID)
        code, _out, err = _run_main(
            cli_module, _new_run_argv(snapshots_root, raw_root), capsys=capsys
        )
        assert code == 1
        assert "holds the snapshot build lock" in err


# ── refresh new-run contract ───────────────────────────────────────────────────


class TestRefreshNewRunContract:
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
        assert code == 2
        _assert_no_snapshot_state(snapshots_root, raw_root)


# ── refresh resume contract ────────────────────────────────────────────────────


class TestRefreshResumeContract:
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

    def test_resume_allows_different_workers(
        self, cli_module, snapshots_root, monkeypatch, capsys
    ):
        _RecordingLock.instances = []
        monkeypatch.setattr(cli_module, "SiblingBuildLock", _RecordingLock)
        monkeypatch.setattr(
            cli_module,
            "open_resume_run",
            lambda *a, **k: _fake_run(snapshots_root),
        )
        monkeypatch.setattr(cli_module, "execute_backfill_stages", lambda *a, **k: {})
        monkeypatch.setattr(
            cli_module,
            "finalize_candidate_snapshot",
            lambda *a, **k: (_fake_manifest(), Path("m.json")),
        )
        monkeypatch.setattr(
            cli_module,
            "publish_candidate_snapshot",
            lambda *a, **k: snapshots_root / BUILD_ID,
        )
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
        assert code == 0
        assert "identity-defining" not in err

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
