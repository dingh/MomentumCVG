"""Thin cold-backfill stage adapters (Sprint 004 C8.3B).

Four ordinary functions — :func:`run_liquidity_stage`,
:func:`run_adjusted_stage`, :func:`run_spot_stage`, and
:func:`run_surface_stage` — that each:

1. resolve frozen inputs and stage-owned paths from a prepared/resumed run;
2. run the existing accepted producer (C4 / C5 split path / C8.2 spot /
   C6 weekly surface) without copying its formulas;
3. run the existing acceptance gate/audit (strict C7, C5 with expected
   dates, Gate SP, expected-A1-keys + C6 contract checks);
4. promote successful candidate output from ``work/<stage>/candidate`` to
   the stage's stable directory;
5. return a small JSON-compatible evidence dict.

Adapters never write completion markers. Marker validation, resume
traversal, final cross-stage validation, manifest construction, locking,
publication, and CLI execution remain deferred to the next C8.3B commit.
There is no base class, registry, DAG, or state machine here.

Failure behavior: any producer or gate failure raises
:class:`StageExecutionError` and promotes nothing. ``KeyboardInterrupt``
always propagates. An adapter cleans or replaces only its own candidate
and stable paths.

Design: docs/tmp/c8_3b_resumable_cold_backfill_design.md
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.data.security_types import classification_digest
from src.data.snapshot_foundation import (
    GateResult,
    adjusted_inventory_digest,
    digest_json,
    gate_spot_summary_reconciliation,
    resolve_adjusted_inventory,
    sha256_file,
    ticker_date_keys_digest,
)
from src.data.snapshot_orchestrator import SnapshotOrchestratorError
from src.data.paths import DEFAULT_SECURITY_TYPES_PATH
from src.data.split_adjuster import SplitAdjuster
from src.data.ticker_universe import load_ticker_universe
from src.data.trading_day import target_weekly_expiry_from_schedule
from src.features.option_surface_contract import (
    ContractCheckResult,
    check_expected_meta_keys,
    compute_overall_verdict,
    run_contract_checks,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Gate SP WARN text accepted for publication: the already reconciled
# ambiguous-key exclusion and its matching producer warning. Every other
# WARN and every FAIL is rejected.
_ACCEPTED_SPOT_WARNING_MARKERS = (
    "ambiguous ticker-date exclusion(s) present",
    "inconsistent repeated spot values",
)

SPLITS_FILENAME = "splits_hist_liquid.parquet"
SPOT_PARQUET_FILENAME = "spot_prices_adjusted.parquet"
SPOT_SUMMARY_FILENAME = "spot_summary.json"
# Durable Core dictionary filename (used only when composing paths in tests).
# Production default is ``DEFAULT_SECURITY_TYPES_PATH`` (shared reference root).
SECURITY_TYPES_FILENAME = "orats_security_types.parquet"


class StageExecutionError(SnapshotOrchestratorError):
    """Raised when a stage producer or acceptance gate fails (CLI exit 1)."""


# ── script producer access (scripts/ is not a package) ────────────────────────


def _script_module(name: str):
    """Load ``scripts/<name>.py`` once, cached under its plain module name.

    Tests may pre-load (or monkeypatch) the same module name to stub the
    producer boundary; this function then reuses that module.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = _PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StageExecutionError(f"cannot load producer script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _liquidity_module():
    return _script_module("build_liquidity_panel")


def _pit_audit_module():
    return _script_module("audit_pit_universe")


def _fetch_splits_module():
    return _script_module("fetch_splits")


def _adjusted_audit_module():
    return _script_module("audit_adjusted_liquid")


def _spot_module():
    return _script_module("extract_spot_prices")


def _surface_module():
    return _script_module("precompute_option_surface")


# ── shared small helpers ───────────────────────────────────────────────────────


def _config_date(config: Mapping[str, Any], key: str) -> date:
    try:
        return date.fromisoformat(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise StageExecutionError(f"run config field {key!r} is invalid") from exc


def _config_dates(config: Mapping[str, Any], key: str) -> list[date]:
    try:
        return [date.fromisoformat(v) for v in config[key]]
    except (KeyError, TypeError, ValueError) as exc:
        raise StageExecutionError(f"run config field {key!r} is invalid") from exc


def _require_inventory(run) -> Any:
    if run.inventory is None:
        raise StageExecutionError(
            "run has no frozen raw inventory in memory; open the resume with a "
            "raw rescan before executing stages"
        )
    return run.inventory


def _fresh_candidate_dir(path: Path) -> Path:
    """(Re)create a stage-owned candidate directory (restart-from-beginning)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _promote_candidate(candidate_dir: Path, stable_dir: Path) -> list[str]:
    """Atomically promote a stage candidate directory to its stable directory.

    Collects artifact names first, removes only this stage's uncertified
    ``stable_dir`` if present, then renames ``candidate_dir`` → ``stable_dir``
    with a single ``os.replace``. A failed rename leaves the candidate intact
    and never leaves a partially populated stable tree. No backup/rollback.
    """
    if not candidate_dir.is_dir():
        raise StageExecutionError(
            f"candidate directory is missing; nothing to promote: {candidate_dir}"
        )
    artifacts = sorted(p.name for p in candidate_dir.iterdir())
    if not artifacts:
        raise StageExecutionError(
            f"candidate directory is empty; nothing to promote: {candidate_dir}"
        )
    stable_dir.parent.mkdir(parents=True, exist_ok=True)
    if stable_dir.exists():
        shutil.rmtree(stable_dir)
    try:
        os.replace(candidate_dir, stable_dir)
    except OSError as exc:
        raise StageExecutionError(
            f"failed to promote {candidate_dir} -> {stable_dir}: {exc}"
        ) from exc
    return artifacts


def _tree_stats(root: Path, pattern: str = "**/*") -> tuple[int, int]:
    """Return (file_count, total_bytes) under ``root``."""
    count = 0
    total = 0
    for path in root.glob(pattern):
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return count, total


def _spot_warnings_accepted(gate: GateResult) -> bool:
    """True when a Gate SP WARN is exactly the reconciled ambiguous-key case."""
    if gate.failures or not gate.warnings:
        return False
    if not gate.metrics.get("ambiguous_exclusion_count"):
        return False
    return all(
        any(marker in warning for marker in _ACCEPTED_SPOT_WARNING_MARKERS)
        for warning in gate.warnings
    )


def _accepted_surface_warnings(
    checks: list[ContractCheckResult],
) -> list[str]:
    """Return accepted surface warning strings, or raise on any other WARN.

    The only publishable C6 WARN is the nonstructural ``a1_a2_join_integrity``
    informational case where ``surface_valid=false`` metadata still has quote
    rows (``invalid_meta_with_quotes_count > 0``). Every other WARN fails the
    stage before promotion.
    """
    accepted: list[str] = []
    for check in checks:
        if check.status != "WARN":
            continue
        if (
            check.name == "a1_a2_join_integrity"
            and int(check.metrics.get("invalid_meta_with_quotes_count") or 0) > 0
            and not check.failures
        ):
            accepted.extend(check.warnings)
            continue
        raise StageExecutionError(
            f"surface acceptance rejected nonstructural-policy WARN from "
            f"{check.name!r}: {'; '.join(check.warnings[:5]) or check.status}"
        )
    return accepted


# ── stage 1: liquidity ─────────────────────────────────────────────────────────


def run_liquidity_stage(
    run,
    *,
    security_types_path: Path | str | None = None,
    fetch_observation_fn: Callable[[str, date], pd.DataFrame] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Rebuild the C4 liquidity artifacts from the frozen raw inventory.

    Runs the accepted C4 backfill with a loader constrained to the frozen
    resolved trading dates and the accepted equity-only Core classification,
    writes candidate artifacts under ``work/liquidity/candidate``, requires a
    strict C7 PASS, and promotes to ``input/liquidity``.

    The Core security-types dictionary defaults to the durable shared
    reference file (:data:`DEFAULT_SECURITY_TYPES_PATH`). Existing tickers are
    never re-fetched; only missing tickers hit Core (with checkpoints every
    100 new classifications). Callers may inject ``security_types_path`` for
    tests.
    """
    building = Path(run.roots.building)
    config = run.run_config
    inventory = _require_inventory(run)

    from src.data.run_progress import write_run_progress

    write_run_progress(
        building,
        stage="liquidity",
        phase="preparing",
        message="Liquidity stage: preparing frozen-inventory backfill",
        build_id=config.get("build_id"),
    )

    if security_types_path is None:
        security_types_path = DEFAULT_SECURITY_TYPES_PATH
    security_types_path = Path(security_types_path)
    security_types_path.parent.mkdir(parents=True, exist_ok=True)

    start = _config_date(config, "requested_output_start")
    end = _config_date(config, "as_of_resolved_trading_day")
    resolved_dates = _config_dates(config, "resolved_trading_dates")
    c4 = config["c4_params"]

    mod = _liquidity_module()

    raw_root = Path(inventory.raw_root)
    frozen_paths = {
        record.trade_date: raw_root / record.rel_path
        for record in inventory.records
    }

    def _load_frozen_day(trade_date: date) -> pd.DataFrame:
        if trade_date not in frozen_paths:
            raise StageExecutionError(
                "liquidity loader asked for a date outside the frozen "
                f"inventory: {trade_date.isoformat()}"
            )
        return mod.load_raw_day_from_zip(raw_root, trade_date)

    classify_fn = mod.make_core_classifier(
        security_types_path,
        fetch_observation_fn=fetch_observation_fn,
        progress_path=building,
    )

    try:
        result = mod.run_backfill(
            raw_root,
            start,
            end,
            _load_frozen_day,
            resolved_dates,
            lookback_weeks=c4["lookback_weeks"],
            min_valid_quote_weeks=c4["min_valid_quote_weeks"],
            dte_min=c4["dte_min"],
            dte_max=c4["dte_max"],
            show_progress=show_progress,
            classify_fn=classify_fn,
        )
    except StageExecutionError:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise StageExecutionError(f"liquidity producer failed: {exc}") from exc

    result.panel = mod.stamp_panel_universe_params(
        result.panel,
        dvol_top_pct=c4["dvol_top_pct"],
        spread_bot_pct=c4["spread_bot_pct"],
    )
    liquid = mod.build_liquid_tickers(
        result.panel, c4["dvol_top_pct"], c4["spread_bot_pct"]
    )
    if liquid.empty:
        raise StageExecutionError(
            "liquidity producer yielded an empty liquid-ticker superset"
        )

    candidate = _fresh_candidate_dir(building / "work" / "liquidity" / "candidate")
    mod.write_artifacts(candidate, result, liquid_tickers=liquid)

    report_path = building / "reports" / "liquidity" / "pit_universe_audit.md"
    audit_code = _pit_audit_module().main(
        [
            "--panel-path", str(candidate / mod.PANEL_FILENAME),
            "--weekly-path", str(candidate / mod.WEEKLY_FILENAME),
            "--liquid-tickers-path", str(candidate / mod.LIQUID_TICKERS_FILENAME),
            "--discover-samples",
            "--strict",
            "--dvol-top-pct", str(c4["dvol_top_pct"]),
            "--spread-bottom-pct", str(c4["spread_bot_pct"]),
            "--output-report", str(report_path),
        ]
    )
    if audit_code != 0:
        raise StageExecutionError(
            f"strict C7 PIT universe audit did not PASS (exit {audit_code}); "
            f"report: {report_path}"
        )

    stable_dir = building / "input" / "liquidity"
    promoted = _promote_candidate(candidate, stable_dir)

    equity_universe = sorted(str(t).strip().upper() for t in liquid["Ticker"])
    return {
        "stage": "liquidity",
        "status": "PASS",
        "output_dir": str(stable_dir),
        "report_path": str(report_path),
        "artifacts": promoted,
        "files_read": result.files_read,
        "daily_row_count": len(result.daily),
        "weekly_row_count": len(result.weekly),
        "panel_row_count": len(result.panel),
        "classified_ticker_count": len(result.classification),
        "classification_digest": (
            classification_digest(result.classification)
            if not result.classification.empty
            else None
        ),
        "liquid_ticker_count": len(equity_universe),
        "equity_universe_digest": digest_json(equity_universe),
        "accepted_warnings": list(result.warnings),
    }


# ── stage 2: adjusted ──────────────────────────────────────────────────────────


def run_adjusted_stage(run, *, max_workers: int | None = None) -> dict[str, Any]:
    """Fetch scoped splits and adjust exactly the frozen physical ZIP set.

    Reuses the fixed C5 split-fetch path (token from ``ORATS_API_TOKEN`` only)
    and the exact-list :meth:`SplitAdjuster.process_zip_paths` entry point,
    requires a strict C5 PASS with the frozen physical dates as expected
    dates, and promotes to ``input/adjusted_liquid``.
    """
    building = Path(run.roots.building)
    config = run.run_config
    inventory = _require_inventory(run)

    liquid_csv = building / "input" / "liquidity" / "liquid_tickers.csv"
    if not liquid_csv.is_file():
        raise StageExecutionError(
            f"certified liquidity output is missing: {liquid_csv}"
        )
    universe = load_ticker_universe(liquid_csv)

    if not os.environ.get("ORATS_API_TOKEN"):
        raise StageExecutionError(
            "ORATS_API_TOKEN environment variable is required for the scoped "
            "split fetch (no token flag is supported)"
        )

    raw_root = Path(inventory.raw_root)
    physical_dates = sorted(record.trade_date for record in inventory.records)
    zip_paths = [
        raw_root / record.rel_path
        for record in sorted(inventory.records, key=lambda r: r.trade_date)
    ]

    work_dir = building / "work" / "adjusted"
    candidate = _fresh_candidate_dir(work_dir / "candidate")
    splits_path = candidate / SPLITS_FILENAME

    try:
        _fetch_splits_module().main(
            ["--ticker-universe", str(liquid_csv), "--out", str(splits_path)]
        )
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        raise StageExecutionError(f"scoped split fetch failed: {exc}") from exc
    except Exception as exc:
        raise StageExecutionError(f"scoped split fetch failed: {exc}") from exc
    # The fetch checkpoint is working state only; it must not be promoted.
    splits_path.with_name(
        f"{splits_path.stem}.checkpoint{splits_path.suffix}"
    ).unlink(missing_ok=True)

    adjuster = SplitAdjuster(
        raw_root=raw_root,
        adj_root=candidate,
        splits_path=splits_path,
        overwrite=False,
        ticker_universe=universe,
    )
    try:
        produced = adjuster.process_zip_paths(zip_paths, max_workers=max_workers)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise StageExecutionError(f"exact-list split adjustment failed: {exc}") from exc

    expected_dates_path = work_dir / "expected_dates.txt"
    expected_dates_path.write_text(
        "".join(f"{d.isoformat()}\n" for d in physical_dates), encoding="utf-8"
    )

    report_path = building / "reports" / "adjusted" / "adjusted_liquid_audit.md"
    years = sorted({d.year for d in physical_dates})
    try:
        _adjusted_audit_module().main(
            [
                "--raw-root", str(raw_root),
                "--adj-root", str(candidate),
                "--splits", str(splits_path),
                "--ticker-universe", str(liquid_csv),
                "--years", *[str(y) for y in years],
                "--report-path", str(report_path),
                "--expected-dates", str(expected_dates_path),
            ]
        )
    except SystemExit as exc:
        raise StageExecutionError(
            f"C5 adjusted-liquid audit FAILED (exit {exc.code}); "
            f"report: {report_path}"
        ) from exc

    verdict = _read_report_verdict(report_path)
    if verdict != "PASS":
        raise StageExecutionError(
            f"C5 adjusted-liquid audit verdict is {verdict!r}, strict PASS is "
            f"required; report: {report_path}"
        )

    split_metadata_hash = sha256_file(splits_path)
    stable_dir = building / "input" / "adjusted_liquid"
    promoted = _promote_candidate(candidate, stable_dir)

    stable_parquets = sorted(stable_dir.glob("*/ORATS_SMV_Strikes_*.parquet"))
    file_count, total_bytes = _tree_stats(stable_dir)
    return {
        "stage": "adjusted",
        "status": "PASS",
        "output_dir": str(stable_dir),
        "report_path": str(report_path),
        "artifacts": promoted,
        "expected_zip_count": len(zip_paths),
        "produced_file_count": len(produced),
        "date_min": physical_dates[0].isoformat(),
        "date_max": physical_dates[-1].isoformat(),
        "date_count": len(physical_dates),
        "output_file_count": file_count,
        "output_total_bytes": total_bytes,
        "split_metadata_hash": split_metadata_hash,
        "universe_ticker_count": len(universe),
        "universe_digest": digest_json(sorted(universe)),
        "adjusted_inventory_digest": adjusted_inventory_digest(
            stable_dir, stable_parquets
        ),
        "audit_verdict": verdict,
    }


def _read_report_verdict(report_path: Path) -> str:
    """Extract the overall verdict line from a C5 audit markdown report."""
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StageExecutionError(
            f"cannot read C5 audit report: {report_path}: {exc}"
        ) from exc
    for line in text.splitlines():
        if line.startswith("## Overall verdict:"):
            return line.split("**")[1] if "**" in line else line
    raise StageExecutionError(
        f"C5 audit report has no overall verdict line: {report_path}"
    )


# ── stage 3: spot ──────────────────────────────────────────────────────────────


def run_spot_stage(run) -> dict[str, Any]:
    """Extract spot prices from certified adjusted chains and run Gate SP.

    Consumes only ``input/adjusted_liquid``, produces the spot parquet and
    compact summary under ``work/spot/candidate``, reconciles them with
    :func:`gate_spot_summary_reconciliation`, and promotes to ``cache/spot``.
    PASS publishes; the only publishable WARN is the already reconciled
    ambiguous-key exclusion.
    """
    building = Path(run.roots.building)
    config = run.run_config

    adjusted_root = building / "input" / "adjusted_liquid"
    if not adjusted_root.is_dir():
        raise StageExecutionError(
            f"certified adjusted output is missing: {adjusted_root}"
        )

    resolved_dates = _config_dates(config, "resolved_trading_dates")
    start_year = min(d.year for d in resolved_dates)
    end_year = max(d.year for d in resolved_dates)

    candidate = _fresh_candidate_dir(building / "work" / "spot" / "candidate")
    output_path = candidate / SPOT_PARQUET_FILENAME
    summary_path = candidate / SPOT_SUMMARY_FILENAME

    exit_code = _spot_module().main(
        [
            "--data-root", str(adjusted_root),
            "--output", str(output_path),
            "--summary-path", str(summary_path),
            "--start-year", str(start_year),
            "--end-year", str(end_year),
        ]
    )
    if exit_code != 0:
        raise StageExecutionError(f"spot producer failed (exit {exit_code})")

    try:
        inventory = resolve_adjusted_inventory(adjusted_root, start_year, end_year)
    except Exception as exc:
        raise StageExecutionError(
            f"cannot resolve adjusted inventory for Gate SP: {exc}"
        ) from exc

    source_keys: set[tuple[date, str]] = set()
    for trade_date, path in inventory.resolved_paths_by_date.items():
        try:
            tickers = pd.read_parquet(path, columns=["ticker"])["ticker"]
        except Exception as exc:
            raise StageExecutionError(
                f"cannot derive source keys from {path}: {exc}"
            ) from exc
        for ticker in tickers.astype(str).str.strip().str.upper().unique():
            source_keys.add((trade_date, ticker))

    output_frame = pd.read_parquet(output_path, columns=["date", "ticker"])
    output_keys = {
        (d, str(t)) for d, t in zip(output_frame["date"], output_frame["ticker"])
    }

    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageExecutionError(f"cannot read spot summary: {exc}") from exc

    gate = gate_spot_summary_reconciliation(
        summary,
        source_keys=source_keys,
        output_keys=output_keys,
        resolved_trading_dates=inventory.resolved_trading_dates,
    )

    report_path = building / "reports" / "spot" / "gate_spot_reconciliation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if gate.status == "FAIL":
        raise StageExecutionError(
            f"Gate SP FAILED: {'; '.join(gate.failures[:5])}; report: {report_path}"
        )
    if gate.status == "WARN" and not _spot_warnings_accepted(gate):
        raise StageExecutionError(
            "Gate SP WARN is not the accepted reconciled ambiguous-key case: "
            f"{'; '.join(gate.warnings[:5])}; report: {report_path}"
        )

    stable_dir = building / "cache" / "spot"
    promoted = _promote_candidate(candidate, stable_dir)

    return {
        "stage": "spot",
        "status": gate.status,
        "output_dir": str(stable_dir),
        "output_path": str(stable_dir / SPOT_PARQUET_FILENAME),
        "summary_path": str(stable_dir / SPOT_SUMMARY_FILENAME),
        "report_path": str(report_path),
        "artifacts": promoted,
        "source_key_count": summary["source_ticker_date_key_count"],
        "source_key_digest": summary["source_ticker_date_key_digest"],
        "output_key_count": summary["output_ticker_date_key_count"],
        "output_key_digest": summary["output_ticker_date_key_digest"],
        "ambiguous_exclusion_count": summary["ambiguous_exclusion_count"],
        "output_row_count": summary["output_row_count"],
        "output_total_bytes": (stable_dir / SPOT_PARQUET_FILENAME).stat().st_size,
        "accepted_warnings": list(gate.warnings),
    }


# ── stage 4: surface ───────────────────────────────────────────────────────────


def run_surface_stage(run, *, workers: int | None = None) -> dict[str, Any]:
    """Precompute the weekly A1/A2 option surface and gate exact A1 coverage.

    Consumes the certified equity superset, adjusted chains, and spot data;
    derives the supported weekly entry schedule (successor present; entry and
    successor in both adjusted and spot inventories); runs the existing weekly
    producer into ``work/surface/candidate`` over that supported interval;
    requires exact equality between expected and actual A1 keys plus a C6
    contract-check suite with no FAIL and only the accepted informational
    ``a1_a2_join_integrity`` WARN; and promotes the A1/A2 pair together to
    ``cache/surface``.
    """
    building = Path(run.roots.building)
    config = run.run_config

    liquid_csv = building / "input" / "liquidity" / "liquid_tickers.csv"
    adjusted_root = building / "input" / "adjusted_liquid"
    spot_path = building / "cache" / "spot" / SPOT_PARQUET_FILENAME
    for required in (liquid_csv, spot_path):
        if not required.is_file():
            raise StageExecutionError(f"required stage input is missing: {required}")
    if not adjusted_root.is_dir():
        raise StageExecutionError(
            f"certified adjusted output is missing: {adjusted_root}"
        )

    tickers = load_ticker_universe(liquid_csv)
    start = _config_date(config, "requested_output_start")
    end = _config_date(config, "as_of_resolved_trading_day")
    start_year, end_year = start.year, end.year

    mod = _surface_module()
    trade_dates, weekly_schedule = mod.generate_trade_dates(
        start, end, "weekly", adjusted_root
    )
    if not weekly_schedule:
        raise StageExecutionError(
            "weekly surface schedule is empty; cannot resolve supported entries"
        )

    try:
        adjusted_inventory = resolve_adjusted_inventory(
            adjusted_root, start_year, end_year
        )
    except Exception as exc:
        raise StageExecutionError(
            f"cannot resolve adjusted inventory for surface schedule: {exc}"
        ) from exc
    adjusted_dates = set(adjusted_inventory.resolved_trading_dates)

    try:
        spot_dates = {
            d
            for d in pd.to_datetime(
                pd.read_parquet(spot_path, columns=["date"])["date"]
            ).dt.date
            if d is not None
        }
    except Exception as exc:
        raise StageExecutionError(
            f"cannot read spot date inventory from {spot_path}: {exc}"
        ) from exc

    supported_dates: list[date] = []
    for entry_date in trade_dates:
        successor = target_weekly_expiry_from_schedule(entry_date, weekly_schedule)
        if successor is None:
            continue
        if entry_date not in adjusted_dates or successor not in adjusted_dates:
            continue
        if entry_date not in spot_dates or successor not in spot_dates:
            continue
        supported_dates.append(entry_date)

    if not supported_dates:
        raise StageExecutionError(
            "no supported weekly surface entry dates: every candidate entry "
            "lacks a schedule successor present in both the resolved adjusted "
            "inventory and the spot parquet date inventory"
        )

    supported_start = supported_dates[0]
    supported_end = supported_dates[-1]
    expected_keys = {
        (ticker, entry_date)
        for ticker in tickers
        for entry_date in supported_dates
    }

    candidate = _fresh_candidate_dir(building / "work" / "surface" / "candidate")
    argv = [
        "--data-root", str(adjusted_root),
        "--output-root", str(candidate),
        "--spot-db-path", str(spot_path),
        "--tickers-file", str(liquid_csv),
        "--frequency", "weekly",
        "--start-year", str(start_year),
        "--end-year", str(end_year),
        "--start-date", supported_start.isoformat(),
        "--end-date", supported_end.isoformat(),
        "--log-file", "-",
    ]
    if workers is not None:
        argv.extend(["--workers", str(workers)])
    exit_code = mod.main(argv)
    if exit_code != 0:
        raise StageExecutionError(f"surface producer failed (exit {exit_code})")

    meta_path = candidate / (
        f"option_surface_meta_weekly_{start_year}_{end_year}.parquet"
    )
    quotes_path = candidate / (
        f"option_surface_quotes_weekly_{start_year}_{end_year}.parquet"
    )
    for produced in (meta_path, quotes_path):
        if not produced.is_file():
            raise StageExecutionError(f"surface producer output missing: {produced}")

    meta_df = pd.read_parquet(meta_path)
    quotes_df = pd.read_parquet(quotes_path)

    checks = [check_expected_meta_keys(meta_df, expected_keys)]
    checks.extend(
        run_contract_checks(
            meta_df,
            quotes_df,
            frequency="weekly",
            data_root=adjusted_root,
            start_date=supported_start,
            end_date=supported_end,
        )
    )
    verdict = compute_overall_verdict(checks)

    report_path = building / "reports" / "surface" / "surface_contract_checks.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "overall_verdict": verdict,
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "metrics": c.metrics,
                        "failures": c.failures,
                        "warnings": c.warnings,
                        "examples": c.examples,
                    }
                    for c in checks
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if verdict == "FAIL":
        failing = [c.name for c in checks if c.status == "FAIL"]
        raise StageExecutionError(
            f"surface acceptance FAILED ({', '.join(failing)}); "
            f"report: {report_path}"
        )
    try:
        accepted_warnings = _accepted_surface_warnings(checks)
    except StageExecutionError as exc:
        raise StageExecutionError(f"{exc}; report: {report_path}") from exc

    actual_keys = {
        (str(t), d)
        for t, d in zip(
            meta_df["ticker"].astype(str).str.strip().str.upper(),
            pd.to_datetime(meta_df["entry_date"]).dt.date,
        )
    }

    stable_dir = building / "cache" / "surface"
    promoted = _promote_candidate(candidate, stable_dir)
    stable_meta = stable_dir / meta_path.name
    stable_quotes = stable_dir / quotes_path.name

    a2_grain = sorted(
        [str(t), pd.Timestamp(e).date().isoformat(), pd.Timestamp(x).date().isoformat(),
         float(s), str(side)]
        for t, e, x, s, side in zip(
            quotes_df["ticker"],
            quotes_df["entry_date"],
            quotes_df["expiry_date"],
            quotes_df["strike"],
            quotes_df["side"],
        )
    )

    return {
        "stage": "surface",
        "status": verdict,
        "output_dir": str(stable_dir),
        "meta_path": str(stable_meta),
        "quotes_path": str(stable_quotes),
        "report_path": str(report_path),
        "artifacts": promoted,
        "supported_entry_dates": [d.isoformat() for d in supported_dates],
        "expected_a1_key_count": len(expected_keys),
        "expected_a1_key_digest": ticker_date_keys_digest(
            (d, t) for t, d in expected_keys
        ),
        "actual_a1_key_count": len(actual_keys),
        "actual_a1_key_digest": ticker_date_keys_digest(
            (d, t) for t, d in actual_keys
        ),
        "surface_valid_true_count": int(meta_df["surface_valid"].sum()),
        "surface_valid_false_count": int((~meta_df["surface_valid"]).sum()),
        "a2_row_count": len(quotes_df),
        "a2_grain_digest": digest_json(a2_grain),
        "meta_total_bytes": stable_meta.stat().st_size,
        "quotes_total_bytes": stable_quotes.stat().st_size,
        "accepted_warnings": accepted_warnings,
    }
