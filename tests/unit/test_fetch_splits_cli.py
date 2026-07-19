"""Unit tests for the fetch_splits CLI wiring (Sprint 004 C5.7).

Scope
-----
C5.7 wires a scoped ``--ticker-universe`` fetch path, an explicit ``--out``
alias, ``ORATS_API_TOKEN`` env fallback, and split-file validation/reporting
into ``scripts/fetch_splits.py``. These tests exercise *CLI parsing / wiring /
validation* only:

* No real ORATS API calls are made.
* ``OratsCorporateActionsFetcher`` is monkeypatched to record how it was
  constructed and to return a deterministic split DataFrame.
* ``load_ticker_universe`` is either the real loader (fed a tiny temp CSV) or a
  monkeypatched spy, depending on the test.
* Temp CSV/parquet universe inputs and temp output paths only — the real broad
  default ``C:/MomentumCVG_env/cache/splits_hist.parquet`` is never touched.

The script is loaded via ``importlib`` (it is a script, not a package module),
matching ``tests/unit/test_apply_split_adjustment_cli.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "scripts" / "fetch_splits.py"


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_universe_csv(path: Path, tickers: list[str]) -> Path:
    pd.DataFrame({"Ticker": tickers}).to_csv(path, index=False)
    return path


def _write_legacy_parquet(path: Path, tickers: list[str]) -> Path:
    pd.DataFrame({"ticker": tickers}).to_parquet(path, index=False)
    return path


def _split_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "split_date", "divisor"])


# ── fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli_module():
    """Import the CLI script as a module so tests can call ``main(argv)``."""
    spec = importlib.util.spec_from_file_location("fetch_splits", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """Ensure ORATS_API_TOKEN never leaks in from the real environment."""
    monkeypatch.delenv("ORATS_API_TOKEN", raising=False)


class _FetcherSpy:
    """Holds constructed fake fetchers and the DataFrame they should return."""

    def __init__(self) -> None:
        self.instances: list = []
        self._result: pd.DataFrame | None = None

    def set_result(self, df: pd.DataFrame) -> None:
        self._result = df


@pytest.fixture
def fetcher_spy(cli_module, monkeypatch):
    spy = _FetcherSpy()

    class _FakeFetcher:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fetch_calls: list[dict] = []
            spy.instances.append(self)

        def fetch_all_splits(self, tickers, checkpoint_path=None):
            self.fetch_calls.append(
                {"tickers": list(tickers), "checkpoint_path": checkpoint_path}
            )
            if spy._result is None:
                return _split_df([])
            return spy._result.copy()

    monkeypatch.setattr(cli_module, "OratsCorporateActionsFetcher", _FakeFetcher)
    return spy


@pytest.fixture
def loader_spy(cli_module, monkeypatch):
    """Replace ``load_ticker_universe`` with a spy returning ['AAA', 'BBB']."""
    calls: list = []

    def _fake_loader(path, **kwargs):
        calls.append(path)
        return ["AAA", "BBB"]

    _fake_loader.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setattr(cli_module, "load_ticker_universe", _fake_loader)
    return _fake_loader


# ── 1. scoped universe loads + passes tickers ───────────────────────────────────

def test_fetch_splits_with_ticker_universe_loads_universe_and_passes_tickers(
    cli_module, fetcher_spy, tmp_path
):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "BBB", "split_date": date(2021, 6, 1), "divisor": 1.5},
            ]
        )
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
            "--token", "FAKE",
        ]
    )

    # Fetcher received exactly the normalized universe tickers.
    assert len(fetcher_spy.instances) == 1
    assert fetcher_spy.instances[0].fetch_calls[0]["tickers"] == ["AAA", "BBB"]

    # Output was written to the explicit provided path.
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert set(written["ticker"]) == {"AAA", "BBB"}


# ── 2. no universe → legacy behavior preserved, loader not called ────────────────

def test_fetch_splits_without_ticker_universe_preserves_existing_behavior(
    cli_module, fetcher_spy, loader_spy, tmp_path
):
    legacy = _write_legacy_parquet(tmp_path / "all_tickers.parquet", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist.parquet"
    fetcher_spy.set_result(
        _split_df(
            [{"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0}]
        )
    )

    cli_module.main(
        [
            "--tickers", str(legacy),
            "--out", str(out_path),
            "--token", "FAKE",
        ]
    )

    # Scoped loader must not run in legacy mode.
    assert loader_spy.calls == []

    # Legacy parquet ticker list is forwarded to the fetcher unchanged.
    assert len(fetcher_spy.instances) == 1
    assert fetcher_spy.instances[0].fetch_calls[0]["tickers"] == ["AAA", "BBB"]
    assert out_path.exists()


# ── 3. scoped mode requires explicit output path ────────────────────────────────

def test_scoped_fetch_requires_explicit_output_path(
    cli_module, fetcher_spy, loader_spy, tmp_path
):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--ticker-universe", str(universe),
                "--token", "FAKE",
            ]
        )

    message = str(excinfo.value).lower()
    assert "explicit output path" in message
    assert "scoped" in message

    # Fail fast: nothing fetched, universe never loaded.
    assert fetcher_spy.instances == []
    assert loader_spy.calls == []


# ── 4. legacy --tickers conflicts with --ticker-universe ─────────────────────────

def test_ticker_universe_conflicts_with_legacy_tickers_arg(
    cli_module, fetcher_spy, loader_spy, tmp_path
):
    legacy = _write_legacy_parquet(tmp_path / "all_tickers.parquet", ["AAA"])
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--tickers", str(legacy),
                "--ticker-universe", str(universe),
                "--token", "FAKE",
            ]
        )

    message = str(excinfo.value)
    assert "--tickers" in message
    assert "--ticker-universe" in message

    assert fetcher_spy.instances == []
    assert loader_spy.calls == []


# ── 5. scoped output rejects out-of-universe tickers ─────────────────────────────

def test_written_scoped_split_file_has_expected_schema_and_no_outside_tickers(
    cli_module, fetcher_spy, tmp_path
):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "BBB", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "CCC", "split_date": date(2020, 1, 1), "divisor": 2.0},
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--ticker-universe", str(universe),
                "--out", str(out_path),
                "--token", "FAKE",
            ]
        )

    message = str(excinfo.value)
    assert "CCC" in message
    assert "outside" in message.lower()

    # Fail fast: no unsafe scoped file is written.
    assert not out_path.exists()


# ── 6. invalid divisor is not silently written ───────────────────────────────────

def test_invalid_divisor_fails_or_is_rejected(cli_module, fetcher_spy, tmp_path):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 0.0},
                {"ticker": "BBB", "split_date": date(2020, 1, 1), "divisor": None},
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--ticker-universe", str(universe),
                "--out", str(out_path),
                "--token", "FAKE",
            ]
        )

    assert "divisor" in str(excinfo.value).lower()
    assert not out_path.exists()


# ── 7. conflicting duplicate split rows fail fast ────────────────────────────────

def test_duplicate_conflicting_split_rows_fail(cli_module, fetcher_spy, tmp_path):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 3.0},
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--ticker-universe", str(universe),
                "--out", str(out_path),
                "--token", "FAKE",
            ]
        )

    assert "conflict" in str(excinfo.value).lower()
    assert not out_path.exists()


# ── 8. identical duplicate split rows are deduped ────────────────────────────────

def test_duplicate_identical_split_rows_are_deduped(cli_module, fetcher_spy, tmp_path):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0},
            ]
        )
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
            "--token", "FAKE",
        ]
    )

    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert len(written) == 1
    assert written.iloc[0]["ticker"] == "AAA"


# ── normalization: fetched tickers are stripped/upper-cased ──────────────────────

def test_fetched_tickers_are_normalized_to_uppercase(cli_module, fetcher_spy, tmp_path):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df(
            [
                {"ticker": " aaa ", "split_date": date(2020, 1, 1), "divisor": 2.0},
                {"ticker": "bbb", "split_date": date(2021, 1, 1), "divisor": 1.5},
            ]
        )
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
            "--token", "FAKE",
        ]
    )

    written = pd.read_parquet(out_path)
    assert set(written["ticker"]) == {"AAA", "BBB"}


# ── token handling: --token wins, else env, else fail fast ───────────────────────

def test_token_from_cli_arg_is_used(cli_module, fetcher_spy, tmp_path):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df([{"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0}])
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
            "--token", "CLI_TOKEN",
        ]
    )

    assert fetcher_spy.instances[0].init_kwargs["token"] == "CLI_TOKEN"


def test_token_from_env_when_no_cli_arg(cli_module, fetcher_spy, monkeypatch, tmp_path):
    monkeypatch.setenv("ORATS_API_TOKEN", "ENV_TOKEN")
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df([{"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0}])
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
        ]
    )

    assert fetcher_spy.instances[0].init_kwargs["token"] == "ENV_TOKEN"


def test_cli_token_wins_over_env(cli_module, fetcher_spy, monkeypatch, tmp_path):
    monkeypatch.setenv("ORATS_API_TOKEN", "ENV_TOKEN")
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA", "BBB"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    fetcher_spy.set_result(
        _split_df([{"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0}])
    )

    cli_module.main(
        [
            "--ticker-universe", str(universe),
            "--out", str(out_path),
            "--token", "CLI_TOKEN",
        ]
    )

    assert fetcher_spy.instances[0].init_kwargs["token"] == "CLI_TOKEN"


def test_missing_token_fails_fast(cli_module, fetcher_spy):
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main([])

    message = str(excinfo.value)
    assert "--token" in message
    assert "ORATS_API_TOKEN" in message
    assert fetcher_spy.instances == []


def test_failed_final_write_does_not_replace_existing_output(
    cli_module, fetcher_spy, monkeypatch, tmp_path
):
    universe = _write_universe_csv(tmp_path / "liquid_tickers.csv", ["AAA"])
    out_path = tmp_path / "splits_hist_liquid.parquet"
    original = _split_df(
        [{"ticker": "OLD", "split_date": date(2019, 1, 1), "divisor": 2.0}]
    )
    original.to_parquet(out_path, index=False)
    original_bytes = out_path.read_bytes()
    fetcher_spy.set_result(
        _split_df(
            [{"ticker": "AAA", "split_date": date(2020, 1, 1), "divisor": 2.0}]
        )
    )

    def fail_write(_self, _path, **_kwargs):
        raise OSError("mocked parquet write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="mocked parquet write failure"):
        cli_module.main(
            [
                "--ticker-universe", str(universe),
                "--out", str(out_path),
                "--token", "FAKE",
            ]
        )

    assert out_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(f".{out_path.name}.*.tmp")) == []
