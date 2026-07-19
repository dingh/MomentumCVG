"""Focused tests for the C8.3B cold-backfill stage adapters.

All producer and audit boundaries are stubbed via the adapter module's
script-module accessors; no ORATS call, no production path, no real C4-C7
producer suites are duplicated here. Frozen runs are real ``PreparedRun``
objects built from tiny temporary raw ZIP fixtures.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.data.snapshot_stage_adapters as adapters
from src.data.snapshot_foundation import digest_json, ticker_date_keys_digest
from src.data.snapshot_orchestrator import prepare_new_backfill_run
from src.data.snapshot_stage_adapters import (
    SECURITY_TYPES_FILENAME,
    StageExecutionError,
    _promote_candidate,
    run_adjusted_stage,
    run_liquidity_stage,
    run_spot_stage,
    run_surface_stage,
)
from src.features.option_surface_contract import (
    ContractCheckResult,
    check_expected_meta_keys,
)

BUILD_ID = "20240412T000000000000Z_deadbeef"
DAY_1 = date(2024, 4, 5)   # Friday
DAY_2 = date(2024, 4, 12)  # Friday
DAY_3 = date(2024, 4, 19)  # Friday successor beyond as-of
GOOD_CSV = "ticker,stkPx\nAAA,10.0\nBBB,20.0\n"


# ── fixtures ───────────────────────────────────────────────────────────────────


def _write_raw_zip(raw_root: Path, day: date, csv_text: str = GOOD_CSV) -> Path:
    year_dir = raw_root / f"{day.year:04d}"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("strikes.csv", csv_text)
    return path


def _prepare_run(tmp_path: Path, csv_by_day: dict[date, str] | None = None):
    raw_root = tmp_path / "raw"
    for day, csv_text in (csv_by_day or {DAY_1: GOOD_CSV, DAY_2: GOOD_CSV}).items():
        _write_raw_zip(raw_root, day, csv_text)
    return prepare_new_backfill_run(
        snapshots_root=tmp_path / "snaps",
        raw_root=raw_root,
        requested_output_start=DAY_1,
        as_of_requested=DAY_2,
        repo_sha_at_freeze="test-sha",
        build_id=BUILD_ID,
    )


@pytest.fixture
def run(tmp_path):
    return _prepare_run(tmp_path)


def _stage_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


def _write_liquid_csv(building: Path, tickers: list[str]) -> Path:
    path = building / "input" / "liquidity" / "liquid_tickers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Ticker": tickers,
            "snapshots_qualified": [3] * len(tickers),
            "months_qualified": [3] * len(tickers),
        }
    ).to_csv(path, index=False)
    return path


def _write_adjusted_parquets(
    building: Path, tickers_by_day: dict[date, list[str]]
) -> Path:
    adjusted_root = building / "input" / "adjusted_liquid"
    for day, tickers in tickers_by_day.items():
        year_dir = adjusted_root / str(day.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ticker": tickers,
                "stkPx": [10.0] * len(tickers),
                "adj_stkPx": [10.0] * len(tickers),
            }
        ).to_parquet(
            year_dir / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.parquet",
            index=False,
        )
    return adjusted_root


# ── liquidity stage ────────────────────────────────────────────────────────────


def _fake_liquidity_module(calls: dict, *, fail_with: BaseException | None = None):
    classification = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "classification": ["company_equity", "company_equity"],
            "observed_asset_types": ["[0]", "[0]"],
        }
    )
    result = SimpleNamespace(
        daily=pd.DataFrame({"ticker": ["AAA", "BBB"]}),
        weekly=pd.DataFrame({"ticker": ["AAA", "BBB"]}),
        panel=pd.DataFrame({"ticker": ["AAA", "BBB"]}),
        files_read=2,
        warnings=[],
        classification=classification,
    )

    def run_backfill(data_root, start, end, load_day_fn, all_trading_dates, **kwargs):
        if fail_with is not None:
            raise fail_with
        calls["run_backfill"] = {
            "data_root": Path(data_root),
            "start": start,
            "end": end,
            "load_day_fn": load_day_fn,
            "all_trading_dates": list(all_trading_dates),
            **kwargs,
        }
        return result

    def load_raw_day_from_zip(data_root, trade_date):
        calls.setdefault("loaded_days", []).append(trade_date)
        return pd.DataFrame({"ticker": ["AAA"]})

    def make_core_classifier(path, fetch_observation_fn=None):
        calls["classifier_path"] = Path(path)
        return lambda candidates: classification

    def write_artifacts(cache_dir, build_result, *, liquid_tickers):
        cache_dir = Path(cache_dir)
        build_result.daily.to_parquet(
            cache_dir / "ticker_liquidity_daily_observations.parquet", index=False
        )
        build_result.weekly.to_parquet(
            cache_dir / "ticker_liquidity_weekly_observations.parquet", index=False
        )
        build_result.panel.to_parquet(
            cache_dir / "ticker_liquidity_panel.parquet", index=False
        )
        liquid_tickers.to_csv(cache_dir / "liquid_tickers.csv", index=False)
        build_result.classification.to_parquet(
            cache_dir / "security_classification.parquet", index=False
        )

    return SimpleNamespace(
        PANEL_FILENAME="ticker_liquidity_panel.parquet",
        WEEKLY_FILENAME="ticker_liquidity_weekly_observations.parquet",
        LIQUID_TICKERS_FILENAME="liquid_tickers.csv",
        run_backfill=run_backfill,
        load_raw_day_from_zip=load_raw_day_from_zip,
        make_core_classifier=make_core_classifier,
        stamp_panel_universe_params=lambda panel, **kwargs: panel,
        build_liquid_tickers=lambda panel, dvol, spread: pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "snapshots_qualified": [3, 3],
                "months_qualified": [3, 3],
            }
        ),
        write_artifacts=write_artifacts,
    )


def _fake_pit_audit(calls: dict, exit_code: int = 0):
    def main(argv):
        calls["pit_audit_argv"] = list(argv)
        return exit_code

    return SimpleNamespace(main=main)


def test_liquidity_stage_passes_frozen_inputs_and_promotes(run, monkeypatch, tmp_path):
    calls: dict = {}
    monkeypatch.setattr(
        adapters, "_liquidity_module", lambda: _fake_liquidity_module(calls)
    )
    monkeypatch.setattr(adapters, "_pit_audit_module", lambda: _fake_pit_audit(calls))

    evidence = run_liquidity_stage(
        run,
        security_types_path=tmp_path / "security_types.parquet",
        fetch_observation_fn=lambda ticker, day: pd.DataFrame(),
    )

    backfill = calls["run_backfill"]
    assert backfill["start"] == DAY_1
    assert backfill["end"] == DAY_2
    assert backfill["all_trading_dates"] == [DAY_1, DAY_2]
    assert backfill["lookback_weeks"] == 12
    assert calls["classifier_path"] == tmp_path / "security_types.parquet"

    # The loader is constrained to the frozen inventory.
    loader = backfill["load_day_fn"]
    loader(DAY_1)
    assert calls["loaded_days"] == [DAY_1]
    with pytest.raises(StageExecutionError, match="outside the frozen inventory"):
        loader(date(2024, 4, 8))

    # The strict C7 audit ran against the candidate artifacts.
    argv = calls["pit_audit_argv"]
    assert "--strict" in argv
    assert any("work" in a and "candidate" in a for a in argv)

    building = Path(run.roots.building)
    stable = building / "input" / "liquidity"
    assert _stage_files(stable) == [
        "liquid_tickers.csv",
        "security_classification.parquet",
        "ticker_liquidity_daily_observations.parquet",
        "ticker_liquidity_panel.parquet",
        "ticker_liquidity_weekly_observations.parquet",
    ]
    assert not (building / "work" / "liquidity" / "candidate").exists()
    assert (building / "reports" / "liquidity") == Path(evidence["report_path"]).parent

    assert evidence["stage"] == "liquidity"
    assert evidence["status"] == "PASS"
    assert evidence["liquid_ticker_count"] == 2
    assert evidence["equity_universe_digest"] == digest_json(["AAA", "BBB"])
    assert evidence["classification_digest"] is not None
    # No completion marker is written by any adapter.
    assert _stage_files(building / "markers") == []


def test_liquidity_default_security_types_path_is_building_local(run, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        adapters, "_liquidity_module", lambda: _fake_liquidity_module(calls)
    )
    monkeypatch.setattr(adapters, "_pit_audit_module", lambda: _fake_pit_audit(calls))

    run_liquidity_stage(
        run,
        fetch_observation_fn=lambda ticker, day: pd.DataFrame(),
    )

    building = Path(run.roots.building)
    expected = building / "work" / "liquidity" / SECURITY_TYPES_FILENAME
    assert calls["classifier_path"] == expected
    assert str(calls["classifier_path"]).startswith(str(building))


def test_liquidity_gate_failure_promotes_nothing(run, monkeypatch, tmp_path):
    calls: dict = {}
    monkeypatch.setattr(
        adapters, "_liquidity_module", lambda: _fake_liquidity_module(calls)
    )
    monkeypatch.setattr(
        adapters, "_pit_audit_module", lambda: _fake_pit_audit(calls, exit_code=1)
    )

    with pytest.raises(StageExecutionError, match="strict C7"):
        run_liquidity_stage(
            run,
            security_types_path=tmp_path / "security_types.parquet",
            fetch_observation_fn=lambda ticker, day: pd.DataFrame(),
        )

    assert _stage_files(Path(run.roots.building) / "input" / "liquidity") == []


def test_liquidity_keyboard_interrupt_propagates(run, monkeypatch, tmp_path):
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_liquidity_module",
        lambda: _fake_liquidity_module(calls, fail_with=KeyboardInterrupt()),
    )
    monkeypatch.setattr(adapters, "_pit_audit_module", lambda: _fake_pit_audit(calls))

    with pytest.raises(KeyboardInterrupt):
        run_liquidity_stage(
            run,
            security_types_path=tmp_path / "security_types.parquet",
            fetch_observation_fn=lambda ticker, day: pd.DataFrame(),
        )

    assert _stage_files(Path(run.roots.building) / "input" / "liquidity") == []


# ── adjusted stage ─────────────────────────────────────────────────────────────


def _fake_fetch_splits(calls: dict):
    def main(argv):
        calls["fetch_argv"] = list(argv)
        out = Path(argv[argv.index("--out") + 1])
        pd.DataFrame(
            {"ticker": ["AAA"], "split_date": [date(2024, 2, 1)], "divisor": [2.0]}
        ).to_parquet(out, index=False)
        # Simulate the resume checkpoint the real producer leaves behind.
        out.with_name(f"{out.stem}.checkpoint{out.suffix}").write_bytes(b"ckpt")

    return SimpleNamespace(main=main)


def _fake_adjusted_audit(calls: dict, verdict: str = "PASS"):
    def main(argv):
        calls["audit_argv"] = list(argv)
        report = Path(argv[argv.index("--report-path") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"# audit\n\n## Overall verdict: **{verdict}**\n", encoding="utf-8"
        )
        if verdict == "FAIL":
            raise SystemExit(1)

    return SimpleNamespace(main=main)


class _SpyAdjuster:
    """Records exact-list inputs and produces one parquet per supplied ZIP."""

    last: "_SpyAdjuster | None" = None

    def __init__(self, *, raw_root, adj_root, splits_path, overwrite, ticker_universe):
        self.init = {
            "raw_root": Path(raw_root),
            "adj_root": Path(adj_root),
            "splits_path": Path(splits_path),
            "overwrite": overwrite,
            "ticker_universe": list(ticker_universe),
        }
        self.zip_paths: list[Path] | None = None
        type(self).last = self

    def process_zip_paths(self, zip_paths, *, max_workers=None):
        self.zip_paths = [Path(p) for p in zip_paths]
        self.max_workers = max_workers
        produced = []
        for zip_path in self.zip_paths:
            date_str = zip_path.stem.split("_")[-1]
            out = (
                self.init["adj_root"]
                / date_str[:4]
                / f"ORATS_SMV_Strikes_{date_str}.parquet"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ticker": ["AAA"]}).to_parquet(out, index=False)
            produced.append(out)
        return produced


def test_adjusted_stage_supplies_exact_frozen_zip_list(run, monkeypatch):
    building = Path(run.roots.building)
    _write_liquid_csv(building, ["AAA", "BBB"])
    monkeypatch.setenv("ORATS_API_TOKEN", "FAKE_TOKEN")

    calls: dict = {}
    monkeypatch.setattr(adapters, "_fetch_splits_module", lambda: _fake_fetch_splits(calls))
    monkeypatch.setattr(
        adapters, "_adjusted_audit_module", lambda: _fake_adjusted_audit(calls)
    )
    monkeypatch.setattr(adapters, "SplitAdjuster", _SpyAdjuster)

    evidence = run_adjusted_stage(run, max_workers=2)

    raw_root = Path(run.inventory.raw_root)
    expected_zips = [
        raw_root / "2024" / f"ORATS_SMV_Strikes_{d.strftime('%Y%m%d')}.zip"
        for d in (DAY_1, DAY_2)
    ]
    spy = _SpyAdjuster.last
    assert spy is not None
    assert spy.zip_paths == expected_zips
    assert spy.max_workers == 2
    assert spy.init["overwrite"] is False
    assert spy.init["ticker_universe"] == ["AAA", "BBB"]

    # No token flag was passed to the split fetch (env only).
    assert "--token" not in calls["fetch_argv"]

    # Expected-dates file carries exactly the frozen physical dates.
    expected_dates_path = Path(
        calls["audit_argv"][calls["audit_argv"].index("--expected-dates") + 1]
    )
    assert expected_dates_path.read_text(encoding="utf-8") == (
        f"{DAY_1.isoformat()}\n{DAY_2.isoformat()}\n"
    )

    stable = building / "input" / "adjusted_liquid"
    files = _stage_files(stable)
    assert "splits_hist_liquid.parquet" in files
    assert "ORATS_SMV_Strikes_20240405.parquet" in files
    assert "ORATS_SMV_Strikes_20240412.parquet" in files
    assert not any("checkpoint" in name for name in files)
    assert not (building / "work" / "adjusted" / "candidate").exists()

    assert evidence["stage"] == "adjusted"
    assert evidence["date_min"] == DAY_1.isoformat()
    assert evidence["date_max"] == DAY_2.isoformat()
    assert evidence["expected_zip_count"] == 2
    assert evidence["produced_file_count"] == 2
    assert evidence["universe_digest"] == digest_json(["AAA", "BBB"])
    assert evidence["audit_verdict"] == "PASS"


def test_adjusted_stage_fails_on_real_processing_error(tmp_path, monkeypatch):
    # The second frozen ZIP has no ticker column, so the real exact-list
    # adjustment fails; nothing may be promoted.
    run = _prepare_run(
        tmp_path, {DAY_1: GOOD_CSV, DAY_2: "foo,bar\n1.0,2.0\n"}
    )
    building = Path(run.roots.building)
    _write_liquid_csv(building, ["AAA", "BBB"])
    monkeypatch.setenv("ORATS_API_TOKEN", "FAKE_TOKEN")

    calls: dict = {}
    monkeypatch.setattr(adapters, "_fetch_splits_module", lambda: _fake_fetch_splits(calls))
    monkeypatch.setattr(
        adapters, "_adjusted_audit_module", lambda: _fake_adjusted_audit(calls)
    )

    with pytest.raises(StageExecutionError, match="exact-list split adjustment failed"):
        run_adjusted_stage(run)

    assert _stage_files(building / "input" / "adjusted_liquid") == []
    assert "audit_argv" not in calls


def test_adjusted_stage_rejects_non_pass_audit_verdict(run, monkeypatch):
    building = Path(run.roots.building)
    _write_liquid_csv(building, ["AAA", "BBB"])
    monkeypatch.setenv("ORATS_API_TOKEN", "FAKE_TOKEN")

    calls: dict = {}
    monkeypatch.setattr(adapters, "_fetch_splits_module", lambda: _fake_fetch_splits(calls))
    monkeypatch.setattr(
        adapters,
        "_adjusted_audit_module",
        lambda: _fake_adjusted_audit(calls, verdict="PASS WITH WARNINGS"),
    )
    monkeypatch.setattr(adapters, "SplitAdjuster", _SpyAdjuster)

    with pytest.raises(StageExecutionError, match="strict PASS is required"):
        run_adjusted_stage(run)

    assert _stage_files(building / "input" / "adjusted_liquid") == []


def test_adjusted_ordinary_fetch_exception_becomes_stage_error(run, monkeypatch):
    building = Path(run.roots.building)
    _write_liquid_csv(building, ["AAA", "BBB"])
    monkeypatch.setenv("ORATS_API_TOKEN", "FAKE_TOKEN")

    def boom(argv):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        adapters, "_fetch_splits_module", lambda: SimpleNamespace(main=boom)
    )

    with pytest.raises(StageExecutionError, match="scoped split fetch failed"):
        run_adjusted_stage(run)

    assert _stage_files(building / "input" / "adjusted_liquid") == []


def test_adjusted_keyboard_interrupt_from_fetch_propagates(run, monkeypatch):
    building = Path(run.roots.building)
    _write_liquid_csv(building, ["AAA", "BBB"])
    monkeypatch.setenv("ORATS_API_TOKEN", "FAKE_TOKEN")

    def interrupt(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        adapters, "_fetch_splits_module", lambda: SimpleNamespace(main=interrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        run_adjusted_stage(run)

    assert _stage_files(building / "input" / "adjusted_liquid") == []


# ── promotion ──────────────────────────────────────────────────────────────────


def test_promote_candidate_uses_one_directory_rename(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    stable = tmp_path / "stable"
    candidate.mkdir()
    (candidate / "a.parquet").write_text("a", encoding="utf-8")
    (candidate / "b.parquet").write_text("b", encoding="utf-8")
    stable.mkdir()
    (stable / "old.parquet").write_text("old", encoding="utf-8")

    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(adapters.os, "replace", spy)

    promoted = _promote_candidate(candidate, stable)

    assert calls == [(candidate, stable)]
    assert promoted == ["a.parquet", "b.parquet"]
    assert not candidate.exists()
    assert _stage_files(stable) == ["a.parquet", "b.parquet"]


def test_promote_candidate_failed_rename_leaves_candidate_intact(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    stable = tmp_path / "stable"
    candidate.mkdir()
    (candidate / "a.parquet").write_text("a", encoding="utf-8")
    (candidate / "b.parquet").write_text("b", encoding="utf-8")
    stable.mkdir()
    (stable / "old.parquet").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        adapters.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(StageExecutionError, match="failed to promote"):
        _promote_candidate(candidate, stable)

    assert candidate.is_dir()
    assert _stage_files(candidate) == ["a.parquet", "b.parquet"]
    # Prior uncertified stable was removed; no partial promotion remains.
    assert not stable.exists()


# ── spot stage ─────────────────────────────────────────────────────────────────


def _spot_rows(keys: list[tuple[date, str]]) -> list[dict]:
    return [
        {"date": d, "ticker": t, "adj_spot_price": 10.0, "spot_price": 10.0}
        for d, t in keys
    ]


def _spot_summary(
    source_keys: list[tuple[date, str]],
    output_keys: list[tuple[date, str]],
    resolved_dates: list[date],
    *,
    ambiguous: list[tuple[date, str]] = (),
    producer_status: str = "PASS",
    warnings: list[str] = (),
) -> dict:
    return {
        "resolved_date_count": len(resolved_dates),
        "resolved_date_min": resolved_dates[0].isoformat(),
        "resolved_date_max": resolved_dates[-1].isoformat(),
        "weekend_excluded_dates": [],
        "source_ticker_date_key_count": len(source_keys),
        "source_ticker_date_key_digest": ticker_date_keys_digest(source_keys),
        "output_ticker_date_key_count": len(output_keys),
        "output_ticker_date_key_digest": ticker_date_keys_digest(output_keys),
        "ambiguous_exclusion_count": len(ambiguous),
        "ambiguous_exclusions": sorted(
            [d.isoformat(), t] for d, t in ambiguous
        ),
        "output_row_count": len(output_keys),
        "producer_status": producer_status,
        "warnings": list(warnings),
    }


def _fake_spot_module(rows: list[dict], summary: dict, calls: dict):
    def main(argv):
        calls["spot_argv"] = list(argv)
        out = Path(argv[argv.index("--output") + 1])
        pd.DataFrame(
            rows, columns=["date", "ticker", "adj_spot_price", "spot_price"]
        ).to_parquet(out, index=False)
        summary_path = Path(argv[argv.index("--summary-path") + 1])
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return 0

    return SimpleNamespace(main=main)


def test_spot_stage_pass_promotes_and_reports_digests(run, monkeypatch):
    building = Path(run.roots.building)
    _write_adjusted_parquets(building, {DAY_1: ["AAA", "BBB"], DAY_2: ["AAA", "BBB"]})

    keys = [(d, t) for d in (DAY_1, DAY_2) for t in ("AAA", "BBB")]
    summary = _spot_summary(keys, keys, [DAY_1, DAY_2])
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_spot_module",
        lambda: _fake_spot_module(_spot_rows(keys), summary, calls),
    )

    evidence = run_spot_stage(run)

    argv = calls["spot_argv"]
    assert argv[argv.index("--data-root") + 1] == str(
        building / "input" / "adjusted_liquid"
    )
    assert argv[argv.index("--start-year") + 1] == "2024"

    stable = building / "cache" / "spot"
    assert _stage_files(stable) == ["spot_prices_adjusted.parquet", "spot_summary.json"]
    assert not (building / "work" / "spot" / "candidate").exists()

    assert evidence["stage"] == "spot"
    assert evidence["status"] == "PASS"
    assert evidence["source_key_count"] == 4
    assert evidence["output_key_count"] == 4
    assert evidence["source_key_digest"] == ticker_date_keys_digest(keys)
    assert evidence["ambiguous_exclusion_count"] == 0
    assert Path(evidence["report_path"]).is_file()


def test_spot_stage_accepts_only_reconciled_ambiguous_warn(run, monkeypatch):
    building = Path(run.roots.building)
    _write_adjusted_parquets(
        building, {DAY_1: ["AAA", "BBB", "CCC"], DAY_2: ["AAA", "BBB"]}
    )

    source = [(d, t) for d in (DAY_1, DAY_2) for t in ("AAA", "BBB")]
    source.append((DAY_1, "CCC"))
    output = [(d, t) for d in (DAY_1, DAY_2) for t in ("AAA", "BBB")]
    summary = _spot_summary(
        source,
        output,
        [DAY_1, DAY_2],
        ambiguous=[(DAY_1, "CCC")],
        producer_status="WARN",
        warnings=[
            "dropped 1 ticker-date keys with inconsistent repeated spot values"
        ],
    )
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_spot_module",
        lambda: _fake_spot_module(_spot_rows(output), summary, calls),
    )

    evidence = run_spot_stage(run)

    assert evidence["status"] == "WARN"
    assert evidence["ambiguous_exclusion_count"] == 1
    assert evidence["accepted_warnings"]
    assert _stage_files(building / "cache" / "spot") == [
        "spot_prices_adjusted.parquet",
        "spot_summary.json",
    ]


def test_spot_stage_rejects_unrecognized_warn(run, monkeypatch):
    building = Path(run.roots.building)
    _write_adjusted_parquets(building, {DAY_1: ["AAA", "BBB"], DAY_2: ["AAA", "BBB"]})

    keys = [(d, t) for d in (DAY_1, DAY_2) for t in ("AAA", "BBB")]
    summary = _spot_summary(
        keys,
        keys,
        [DAY_1, DAY_2],
        producer_status="WARN",
        warnings=["raw drift suspected in source frames"],
    )
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_spot_module",
        lambda: _fake_spot_module(_spot_rows(keys), summary, calls),
    )

    with pytest.raises(StageExecutionError, match="not the accepted"):
        run_spot_stage(run)

    assert _stage_files(building / "cache" / "spot") == []


def test_spot_stage_gate_fail_promotes_nothing(run, monkeypatch):
    building = Path(run.roots.building)
    _write_adjusted_parquets(building, {DAY_1: ["AAA", "BBB"], DAY_2: ["AAA", "BBB"]})

    keys = [(d, t) for d in (DAY_1, DAY_2) for t in ("AAA", "BBB")]
    summary = _spot_summary(keys, keys, [DAY_1, DAY_2])
    summary["output_ticker_date_key_count"] = 3  # inconsistent on purpose
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_spot_module",
        lambda: _fake_spot_module(_spot_rows(keys), summary, calls),
    )

    with pytest.raises(StageExecutionError, match="Gate SP FAILED"):
        run_spot_stage(run)

    assert _stage_files(building / "cache" / "spot") == []


def test_spot_stage_keyboard_interrupt_propagates(run, monkeypatch):
    building = Path(run.roots.building)
    _write_adjusted_parquets(building, {DAY_1: ["AAA"], DAY_2: ["AAA"]})

    def interrupt(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        adapters, "_spot_module", lambda: SimpleNamespace(main=interrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        run_spot_stage(run)

    assert _stage_files(building / "cache" / "spot") == []


# ── surface stage ──────────────────────────────────────────────────────────────


def _surface_frames(keys: list[tuple[str, date]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows = []
    quote_rows = []
    for ticker, entry in keys:
        expiry = entry + timedelta(days=7)
        meta_rows.append(
            {
                "ticker": ticker,
                "entry_date": entry,
                "expiry_date": expiry,
                "dte_actual": 7,
                "entry_spot": 10.0,
                "exit_spot": 11.0,
                "body_strike": 10.0,
                "surface_valid": True,
                "failure_reason": None,
                "has_body_call": True,
                "has_body_put": True,
                "n_surface_quotes": 2,
            }
        )
        for side, delta in (("call", 0.5), ("put", -0.5)):
            quote_rows.append(
                {
                    "ticker": ticker,
                    "entry_date": entry,
                    "expiry_date": expiry,
                    "entry_spot": 10.0,
                    "body_strike": 10.0,
                    "side": side,
                    "is_body": True,
                    "is_otm": False,
                    "strike": 10.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                    "spread_pct": 0.18,
                    "iv": 0.3,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "gamma": 0.01,
                    "vega": 0.02,
                    "theta": -0.01,
                    "volume": 10,
                    "open_interest": 100,
                }
            )
    return pd.DataFrame(meta_rows), pd.DataFrame(quote_rows)


def _fake_surface_module(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    trade_dates: list[date],
    calls: dict,
    *,
    schedule: list[date] | None = None,
):
    schedule = list(schedule) if schedule is not None else list(trade_dates)

    def main(argv):
        calls["surface_argv"] = list(argv)
        output_root = Path(argv[argv.index("--output-root") + 1])
        start_year = argv[argv.index("--start-year") + 1]
        end_year = argv[argv.index("--end-year") + 1]
        meta_df.to_parquet(
            output_root
            / f"option_surface_meta_weekly_{start_year}_{end_year}.parquet",
            index=False,
        )
        quotes_df.to_parquet(
            output_root
            / f"option_surface_quotes_weekly_{start_year}_{end_year}.parquet",
            index=False,
        )
        return 0

    def generate_trade_dates(start, end, frequency, data_root):
        calls["schedule_request"] = (start, end, frequency, Path(data_root))
        return list(trade_dates), list(schedule)

    return SimpleNamespace(main=main, generate_trade_dates=generate_trade_dates)


def _surface_inputs(
    building: Path,
    *,
    adjusted_days: list[date] | None = None,
    spot_days: list[date] | None = None,
) -> None:
    _write_liquid_csv(building, ["AAA", "BBB"])
    days = adjusted_days or [DAY_1, DAY_2]
    _write_adjusted_parquets(
        building, {day: ["AAA", "BBB"] for day in days}
    )
    spot_days = spot_days or [DAY_1, DAY_2]
    spot_dir = building / "cache" / "spot"
    spot_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": day,
            "ticker": ticker,
            "adj_spot_price": 10.0,
            "spot_price": 10.0,
        }
        for day in spot_days
        for ticker in ("AAA", "BBB")
    ]
    pd.DataFrame(rows).to_parquet(
        spot_dir / "spot_prices_adjusted.parquet", index=False
    )


def test_surface_stage_exact_coverage_promotes_pair(run, monkeypatch):
    building = Path(run.roots.building)
    # DAY_3 is the schedule successor that makes DAY_2 supported.
    _surface_inputs(
        building,
        adjusted_days=[DAY_1, DAY_2, DAY_3],
        spot_days=[DAY_1, DAY_2, DAY_3],
    )

    keys = [(t, d) for t in ("AAA", "BBB") for d in (DAY_1, DAY_2)]
    meta_df, quotes_df = _surface_frames(keys)
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_surface_module",
        lambda: _fake_surface_module(
            meta_df,
            quotes_df,
            [DAY_1, DAY_2],
            calls,
            schedule=[DAY_1, DAY_2, DAY_3],
        ),
    )

    evidence = run_surface_stage(run, workers=2)

    argv = calls["surface_argv"]
    assert argv[argv.index("--frequency") + 1] == "weekly"
    assert argv[argv.index("--tickers-file") + 1].endswith("liquid_tickers.csv")
    assert argv[argv.index("--workers") + 1] == "2"
    assert argv[argv.index("--start-date") + 1] == DAY_1.isoformat()
    assert argv[argv.index("--end-date") + 1] == DAY_2.isoformat()

    stable = building / "cache" / "surface"
    assert _stage_files(stable) == [
        "option_surface_meta_weekly_2024_2024.parquet",
        "option_surface_quotes_weekly_2024_2024.parquet",
    ]
    assert not (building / "work" / "surface" / "candidate").exists()

    assert evidence["stage"] == "surface"
    assert evidence["status"] == "PASS"
    assert evidence["supported_entry_dates"] == [
        DAY_1.isoformat(),
        DAY_2.isoformat(),
    ]
    assert evidence["expected_a1_key_count"] == 4
    assert evidence["actual_a1_key_count"] == 4
    assert evidence["expected_a1_key_digest"] == evidence["actual_a1_key_digest"]
    assert evidence["a2_row_count"] == 8
    assert evidence["surface_valid_true_count"] == 4
    assert evidence["surface_valid_false_count"] == 0
    assert evidence["accepted_warnings"] == []


def test_surface_terminal_weekly_entry_excluded_from_expected_denominator(
    run, monkeypatch
):
    building = Path(run.roots.building)
    _surface_inputs(building)  # adjusted+spot cover DAY_1/DAY_2 only

    # Schedule ends at DAY_2 with no successor → DAY_2 is unsupported.
    keys = [(t, DAY_1) for t in ("AAA", "BBB")]
    meta_df, quotes_df = _surface_frames(keys)
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_surface_module",
        lambda: _fake_surface_module(
            meta_df,
            quotes_df,
            [DAY_1, DAY_2],
            calls,
            schedule=[DAY_1, DAY_2],
        ),
    )

    evidence = run_surface_stage(run)

    argv = calls["surface_argv"]
    assert argv[argv.index("--start-date") + 1] == DAY_1.isoformat()
    assert argv[argv.index("--end-date") + 1] == DAY_1.isoformat()
    assert evidence["supported_entry_dates"] == [DAY_1.isoformat()]
    assert evidence["expected_a1_key_count"] == 2
    assert DAY_2.isoformat() not in evidence["supported_entry_dates"]


def test_surface_stage_missing_a1_key_fails_and_promotes_nothing(run, monkeypatch):
    building = Path(run.roots.building)
    _surface_inputs(
        building,
        adjusted_days=[DAY_1, DAY_2, DAY_3],
        spot_days=[DAY_1, DAY_2, DAY_3],
    )

    # Producer output is missing (BBB, DAY_2).
    keys = [("AAA", DAY_1), ("AAA", DAY_2), ("BBB", DAY_1)]
    meta_df, quotes_df = _surface_frames(keys)
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_surface_module",
        lambda: _fake_surface_module(
            meta_df,
            quotes_df,
            [DAY_1, DAY_2],
            calls,
            schedule=[DAY_1, DAY_2, DAY_3],
        ),
    )

    with pytest.raises(StageExecutionError, match="surface acceptance FAILED"):
        run_surface_stage(run)

    assert _stage_files(building / "cache" / "surface") == []
    report = json.loads(
        (building / "reports" / "surface" / "surface_contract_checks.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["overall_verdict"] == "FAIL"
    expected_check = next(
        c for c in report["checks"] if c["name"] == "expected_meta_keys"
    )
    assert expected_check["status"] == "FAIL"


def test_surface_accepts_informational_a1_a2_join_warning(run, monkeypatch):
    building = Path(run.roots.building)
    _surface_inputs(building)

    keys = [(t, DAY_1) for t in ("AAA", "BBB")]
    meta_df, quotes_df = _surface_frames(keys)
    # Mark one row invalid but keep its quote rows → informational a1_a2_join WARN.
    meta_df = meta_df.copy()
    meta_df.loc[meta_df["ticker"] == "BBB", "surface_valid"] = False
    meta_df.loc[meta_df["ticker"] == "BBB", "failure_reason"] = "no_spot_price"
    meta_df.loc[meta_df["ticker"] == "BBB", "has_body_call"] = False
    meta_df.loc[meta_df["ticker"] == "BBB", "has_body_put"] = False
    meta_df.loc[meta_df["ticker"] == "BBB", "n_surface_quotes"] = 0

    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_surface_module",
        lambda: _fake_surface_module(
            meta_df,
            quotes_df,
            [DAY_1, DAY_2],
            calls,
            schedule=[DAY_1, DAY_2],
        ),
    )

    evidence = run_surface_stage(run)

    assert evidence["status"] == "WARN"
    assert evidence["accepted_warnings"]
    assert any("surface_valid=False" in w for w in evidence["accepted_warnings"])
    assert _stage_files(building / "cache" / "surface")


def test_surface_structural_c6_warning_blocks_promotion(run, monkeypatch):
    building = Path(run.roots.building)
    _surface_inputs(building)

    keys = [(t, DAY_1) for t in ("AAA", "BBB")]
    meta_df, quotes_df = _surface_frames(keys)
    calls: dict = {}
    monkeypatch.setattr(
        adapters,
        "_surface_module",
        lambda: _fake_surface_module(
            meta_df,
            quotes_df,
            [DAY_1, DAY_2],
            calls,
            schedule=[DAY_1, DAY_2],
        ),
    )

    real_run_contract_checks = adapters.run_contract_checks

    def inject_structural_warn(meta, quotes, **kwargs):
        results = real_run_contract_checks(meta, quotes, **kwargs)
        results.append(
            ContractCheckResult(
                name="failure_vocabulary",
                status="WARN",
                warnings=["unknown failure_reason tag(s): ['made_up_tag']"],
            )
        )
        return results

    monkeypatch.setattr(adapters, "run_contract_checks", inject_structural_warn)

    with pytest.raises(StageExecutionError, match="failure_vocabulary"):
        run_surface_stage(run)

    assert _stage_files(building / "cache" / "surface") == []


# ── expected-A1-key contract check ─────────────────────────────────────────────


def _meta_frame(keys: list[tuple[str, date]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": [t for t, _ in keys], "entry_date": [d for _, d in keys]}
    )


class TestCheckExpectedMetaKeys:
    EXPECTED = [("AAA", DAY_1), ("AAA", DAY_2), ("BBB", DAY_1), ("BBB", DAY_2)]

    def test_exact_coverage_passes(self):
        result = check_expected_meta_keys(_meta_frame(self.EXPECTED), self.EXPECTED)
        assert result.status == "PASS"
        assert result.metrics["missing_key_count"] == 0
        assert result.metrics["unexpected_key_count"] == 0
        assert result.metrics["duplicate_key_count"] == 0

    def test_missing_key_fails(self):
        result = check_expected_meta_keys(
            _meta_frame(self.EXPECTED[:-1]), self.EXPECTED
        )
        assert result.status == "FAIL"
        assert result.metrics["missing_key_count"] == 1
        assert any("missing expected" in f for f in result.failures)

    def test_unexpected_key_fails(self):
        result = check_expected_meta_keys(
            _meta_frame(self.EXPECTED + [("CCC", DAY_1)]), self.EXPECTED
        )
        assert result.status == "FAIL"
        assert result.metrics["unexpected_key_count"] == 1
        assert any("unexpected" in f for f in result.failures)

    def test_duplicate_key_fails(self):
        result = check_expected_meta_keys(
            _meta_frame(self.EXPECTED + [("AAA", DAY_1)]), self.EXPECTED
        )
        assert result.status == "FAIL"
        assert result.metrics["duplicate_key_count"] == 1
        assert any("duplicate" in f for f in result.failures)

    def test_empty_expected_set_fails(self):
        result = check_expected_meta_keys(_meta_frame(self.EXPECTED), [])
        assert result.status == "FAIL"
