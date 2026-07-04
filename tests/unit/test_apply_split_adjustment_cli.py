"""Unit tests for the apply_split_adjustment CLI wiring (Sprint 004 C5.6A).

Scope
-----
C5.6A only wires an optional ``--ticker-universe`` flag into
``scripts/apply_split_adjustment.py`` and adds a filtered-mode safety guard.
These tests exercise *CLI parsing / wiring* only:

* No real ORATS ZIPs are read.
* No real split-adjustment backfill is run.
* ``SplitAdjuster`` is monkeypatched to record how it was constructed and run.
* ``load_ticker_universe`` is monkeypatched to spy on the path and return a
  deterministic ticker list.

The script is loaded via ``importlib`` (it is a script, not a package module),
matching ``tests/unit/test_refresh_weekly_inputs_cli.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "apply_split_adjustment.py"


@pytest.fixture
def cli_module():
    """Import the CLI script as a module so tests can call ``main(argv)``."""
    spec = importlib.util.spec_from_file_location("apply_split_adjustment", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Existing raw-root directory so ``main`` passes its existence check."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def splits_path(tmp_path: Path) -> Path:
    """Existing splits file so ``main`` passes its existence check.

    Contents are irrelevant because ``SplitAdjuster`` is monkeypatched and never
    actually reads the file.
    """
    p = tmp_path / "splits_hist.parquet"
    p.write_bytes(b"parquet")
    return p


@pytest.fixture
def universe_path(tmp_path: Path) -> Path:
    """A CSV path used as the ``--ticker-universe`` argument.

    ``load_ticker_universe`` is monkeypatched in the tests that supply this, so
    the file only needs to exist to be a realistic argument value.
    """
    p = tmp_path / "liquid_tickers.csv"
    p.write_text("Ticker\nAAA\nBBB\n", encoding="utf-8")
    return p


@pytest.fixture
def adj_root(tmp_path: Path) -> Path:
    """A safe, dedicated adjusted-output root for filtered mode."""
    return tmp_path / "adjusted_liquid"


@pytest.fixture
def patched_adjuster(cli_module, monkeypatch):
    """Replace ``SplitAdjuster`` with a recorder capturing init + run/readjust."""
    records: list["_FakeAdjuster"] = []

    class _FakeAdjuster:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.run_calls: list[dict] = []
            self.readjust_calls: list[dict] = []
            records.append(self)

        def run(self, **kwargs):
            self.run_calls.append(kwargs)

        def readjust_tickers(self, **kwargs):
            self.readjust_calls.append(kwargs)

    monkeypatch.setattr(cli_module, "SplitAdjuster", _FakeAdjuster)
    return records


@pytest.fixture
def patched_loader(cli_module, monkeypatch):
    """Replace ``load_ticker_universe`` with a spy returning ['AAA', 'BBB']."""
    calls: list = []

    def _fake_loader(path, **kwargs):
        calls.append(path)
        return ["AAA", "BBB"]

    _fake_loader.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setattr(cli_module, "load_ticker_universe", _fake_loader)
    return _fake_loader


def test_apply_split_adjustment_without_ticker_universe_preserves_default_behavior(
    cli_module, patched_adjuster, raw_root, splits_path
):
    cli_module.main(
        [
            "--raw-root",
            str(raw_root),
            "--splits",
            str(splits_path),
        ]
    )

    assert len(patched_adjuster) == 1
    adjuster = patched_adjuster[0]
    kwargs = adjuster.init_kwargs

    # No universe → filtered mode is off.
    assert kwargs["ticker_universe"] is None

    # Existing args are still forwarded exactly as before.
    assert kwargs["raw_root"] == raw_root
    assert kwargs["adj_root"] == cli_module.DEFAULT_ADJ_ROOT
    assert kwargs["splits_path"] == splits_path
    assert kwargs["overwrite"] is False
    assert kwargs["min_split_date"] == "2014-01-01"

    # Default run() forwarding (no years / workers restriction).
    assert adjuster.run_calls == [{"years": None, "max_workers": None}]
    assert adjuster.readjust_calls == []


def test_apply_split_adjustment_with_ticker_universe_loads_and_passes_tickers(
    cli_module, patched_adjuster, patched_loader, raw_root, splits_path, universe_path, adj_root
):
    cli_module.main(
        [
            "--raw-root",
            str(raw_root),
            "--splits",
            str(splits_path),
            "--ticker-universe",
            str(universe_path),
            "--adj-root",
            str(adj_root),
        ]
    )

    # Loader was called exactly once with the provided universe path.
    assert patched_loader.calls == [universe_path]

    assert len(patched_adjuster) == 1
    kwargs = patched_adjuster[0].init_kwargs

    # The loaded ticker list is forwarded into SplitAdjuster.
    assert kwargs["ticker_universe"] == ["AAA", "BBB"]
    # Filtered output goes to the explicitly supplied safe root, not the mirror.
    assert kwargs["adj_root"] == adj_root
    assert kwargs["adj_root"] != cli_module.DEFAULT_ADJ_ROOT


def test_filtered_mode_requires_explicit_adj_root(
    cli_module, patched_adjuster, patched_loader, universe_path
):
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--ticker-universe",
                str(universe_path),
            ]
        )

    message = str(excinfo.value)
    assert "--adj-root" in message
    assert "filtered mode" in message.lower()

    # Fail fast: no adjuster constructed, universe never loaded.
    assert patched_adjuster == []
    assert patched_loader.calls == []


def test_apply_split_adjustment_without_ticker_universe_does_not_call_loader(
    cli_module, patched_adjuster, patched_loader, raw_root, splits_path
):
    cli_module.main(
        [
            "--raw-root",
            str(raw_root),
            "--splits",
            str(splits_path),
        ]
    )

    assert patched_loader.calls == []
    assert patched_adjuster[0].init_kwargs["ticker_universe"] is None


def test_years_and_workers_are_forwarded_to_run(
    cli_module, patched_adjuster, raw_root, splits_path
):
    cli_module.main(
        [
            "--raw-root",
            str(raw_root),
            "--splits",
            str(splits_path),
            "--years",
            "2022",
            "2023",
            "--workers",
            "4",
        ]
    )

    assert len(patched_adjuster) == 1
    assert patched_adjuster[0].run_calls == [{"years": [2022, 2023], "max_workers": 4}]
