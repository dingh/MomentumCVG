"""
Sprint 007 D0 — read-only artifact readiness validation.

Verifies Sprint 006 official-run identity, schema sufficiency, unique trade keys,
included-key parity, and paired leg quote/settlement identity. No bridge math
or granular economics interpretation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.backtest.surface_decision_report import (
    PRIMARY_END,
    PRIMARY_START,
    count_date_classes,
    filter_to_window,
)

OFFICIAL_RUN_DIR = Path(
    "C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z"
)
OFFICIAL_RECEIPT_PATH = OFFICIAL_RUN_DIR / "run_receipt.json"
OFFICIAL_EXECUTION_REPO_SHA = "e205b9acc5d0400aa38169de721acb7fb8268f29"
OFFICIAL_CONTRACT_SHA256 = (
    "4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c"
)
ACCEPTED_PRIMARY_INCLUDED_KEYS = 9212
FILL_TOLERANCE = 1e-9

TRADE_KEY = ("trade_date", "ticker", "direction")
LEG_KEY = ("trade_date", "ticker", "direction", "expiry_date", "option_type", "strike", "leg_index")
LEG_QUOTE_COLS = ("bid", "ask", "mid")
LEG_SETTLEMENT_COLS = ("exit_spot", "expiry_payoff_per_unit")

TRADE_LOG_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "included_in_portfolio",
    "pnl_total",
    "quantity",
    "capital_at_risk_dollars",
    "return_on_max_loss",
    "entry_cost_per_share",
    "net_credit_per_share",
    "max_loss_per_share",
    "spread_cost_ratio",
    "leg_spread_to_credit_ratio",
    "exit_spot",
    "instrument_type",
    "fill_label",
)
LEG_LOG_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "expiry_date",
    "option_type",
    "strike",
    "leg_index",
    "unit_quantity",
    "portfolio_quantity",
    "bid",
    "ask",
    "mid",
    "fill_price",
    "entry_cash_per_unit",
    "pnl_per_unit",
    "pnl_total_leg",
    "exit_spot",
    "expiry_payoff_per_unit",
    "included_in_portfolio",
    "fill_label",
)
DATE_SUMMARY_COLUMNS = (
    "trade_date",
    "cycle_pnl_total",
    "cycle_return_on_capital_at_risk",
    "short_cycle_pnl_total",
    "long_cycle_pnl_total",
)
DATE_STATUS_COLUMNS = ("trade_date", "status", "reason")
CANDIDATE_VIEW_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "decision_status",
)
FUNNEL_SUMMARY_COLUMNS = ("trade_date", "n_included", "n_jointly_eligible")

DECISION_REPORT_KEYS = ("by_fill", "windows", "fill_assumption_sensitivity", "limitations")
DECISION_REPORT_ROLES = ("decision_report_json", "decision_report_md")
RUN_OUTPUT_NAMES = (
    "candidate_view",
    "date_status",
    "date_summary",
    "funnel_summary",
    "leg_log",
    "run_summary",
    "trade_log",
)


class ArtifactValidationError(Exception):
    """Raised when a D0 gate fails."""


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class D0ValidationResult:
    verdict: str
    gates: list[GateResult] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_current_repo_sha(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _float_equal(left: Any, right: Any, *, tol: float = FILL_TOLERANCE) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return left == right


def expected_mid_fill_price(bid: Any, ask: Any, unit_quantity: Any) -> float:
    """Authoritative mid fill from ``FillAssumption.mid()`` (alpha=0.5 both sides)."""
    _ = unit_quantity  # buy and sell both interpolate to the same midpoint at alpha=0.5
    spread = float(ask) - float(bid)
    return float(bid) + 0.5 * spread


def expected_cross_fill_price(bid: Any, ask: Any, unit_quantity: Any) -> float:
    qty = float(unit_quantity)
    if qty > 0:
        return float(ask)
    if qty < 0:
        return float(bid)
    raise ValueError("unit_quantity must be non-zero for fill validation")


def leg_fill_convention_ok(row: pd.Series, *, fill_label: str) -> bool:
    if str(row.get("fill_label")) != fill_label:
        return False
    fill_price = row["fill_price"]
    if fill_label == "mid":
        expected = expected_mid_fill_price(row["bid"], row["ask"], row["unit_quantity"])
        return _float_equal(fill_price, expected)
    if fill_label == "cross":
        expected = expected_cross_fill_price(row["bid"], row["ask"], row["unit_quantity"])
        return _float_equal(fill_price, expected)
    return False


def expected_run_output_path(run_dir: Path, role: str, fill_label: str) -> Path:
    if role == "run_summary":
        return run_dir / f"run_summary_sprint006_baseline_v1_{fill_label}.json"
    return run_dir / f"{role}_sprint006_baseline_v1_{fill_label}.parquet"


def expected_decision_report_path(run_dir: Path, role: str) -> Path:
    if role == "decision_report_json":
        return run_dir / "decision_report.json"
    if role == "decision_report_md":
        return run_dir / "decision_report.md"
    raise ValueError(f"unknown decision report role: {role}")


def _load_parquet_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(columns))
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    if "expiry_date" in frame.columns:
        frame["expiry_date"] = pd.to_datetime(frame["expiry_date"]).dt.date
    return frame.sort_values(list(columns[:3])).reset_index(drop=True)


def _parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _included_trade_keys(trade_log: pd.DataFrame) -> set[tuple[date, str, str]]:
    if trade_log.empty:
        return set()
    included = trade_log[trade_log["included_in_portfolio"] == True]  # noqa: E712
    return {
        (row["trade_date"], str(row["ticker"]), str(row["direction"]))
        for _, row in included.iterrows()
    }


def _duplicate_trade_keys(trade_log: pd.DataFrame) -> list[tuple[Any, ...]]:
    if trade_log.empty:
        return []
    grouped = trade_log.groupby(list(TRADE_KEY), dropna=False).size()
    return [tuple(key) for key, count in grouped.items() if int(count) > 1]


def _leg_identity_frame(leg_log: pd.DataFrame, trade_keys: set[tuple[date, str, str]]) -> pd.DataFrame:
    if leg_log.empty or not trade_keys:
        return leg_log.iloc[0:0].copy()
    mask = leg_log["included_in_portfolio"] == True  # noqa: E712
    legs = leg_log.loc[mask].copy()
    keys = legs.apply(
        lambda row: (row["trade_date"], str(row["ticker"]), str(row["direction"])),
        axis=1,
    )
    legs = legs[keys.isin(trade_keys)].copy()
    return legs.sort_values(list(LEG_KEY)).reset_index(drop=True)


def _compare_paired_legs(
    mid_legs: pd.DataFrame,
    cross_legs: pd.DataFrame,
    trade_keys: set[tuple[date, str, str]],
) -> dict[str, Any]:
    mid_frame = _leg_identity_frame(mid_legs, trade_keys)
    cross_frame = _leg_identity_frame(cross_legs, trade_keys)

    mid_leg_keys = {
        tuple(row[col] for col in LEG_KEY) for _, row in mid_frame.iterrows()
    }
    cross_leg_keys = {
        tuple(row[col] for col in LEG_KEY) for _, row in cross_frame.iterrows()
    }

    quote_mismatches = 0
    settlement_mismatches = 0
    unit_quantity_mismatches = 0
    mid_fill_convention_violations = 0
    cross_fill_convention_violations = 0
    mid_fill_label_mismatches = 0
    cross_fill_label_mismatches = 0

    mid_index = {
        tuple(row[col] for col in LEG_KEY): row for _, row in mid_frame.iterrows()
    }
    cross_index = {
        tuple(row[col] for col in LEG_KEY): row for _, row in cross_frame.iterrows()
    }

    for _, row in mid_frame.iterrows():
        if str(row.get("fill_label")) != "mid":
            mid_fill_label_mismatches += 1
        if not leg_fill_convention_ok(row, fill_label="mid"):
            mid_fill_convention_violations += 1

    for _, row in cross_frame.iterrows():
        if str(row.get("fill_label")) != "cross":
            cross_fill_label_mismatches += 1
        if not leg_fill_convention_ok(row, fill_label="cross"):
            cross_fill_convention_violations += 1

    for key in sorted(mid_leg_keys & cross_leg_keys):
        mid_row = mid_index[key]
        cross_row = cross_index[key]
        for col in LEG_QUOTE_COLS:
            if not _float_equal(mid_row[col], cross_row[col]):
                quote_mismatches += 1
                break
        for col in LEG_SETTLEMENT_COLS:
            if not _float_equal(mid_row[col], cross_row[col]):
                settlement_mismatches += 1
                break
        if int(mid_row["unit_quantity"]) != int(cross_row["unit_quantity"]):
            unit_quantity_mismatches += 1

    return {
        "n_included_trade_keys": len(trade_keys),
        "n_mid_legs": len(mid_frame),
        "n_cross_legs": len(cross_frame),
        "missing_leg_keys_in_cross": len(mid_leg_keys - cross_leg_keys),
        "missing_leg_keys_in_mid": len(cross_leg_keys - mid_leg_keys),
        "quote_mismatches": quote_mismatches,
        "settlement_mismatches": settlement_mismatches,
        "unit_quantity_mismatches": unit_quantity_mismatches,
        "mid_fill_convention_violations": mid_fill_convention_violations,
        "cross_fill_convention_violations": cross_fill_convention_violations,
        "mid_fill_label_mismatches": mid_fill_label_mismatches,
        "cross_fill_label_mismatches": cross_fill_label_mismatches,
    }


def _assert_columns(path: Path, required: tuple[str, ...]) -> list[str]:
    actual = _parquet_columns(path)
    return [col for col in required if col not in actual]


def _artifact_record(path: Path, *, role: str, fill_label: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "fill_label": fill_label,
    }
    if path.suffix == ".parquet":
        record["columns"] = _parquet_columns(path)
    return record


def collect_artifact_inventory(receipt: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    receipt_path = run_dir / "run_receipt.json"
    inventory.append(_artifact_record(receipt_path, role="run_receipt"))
    for run in receipt.get("runs", []):
        fill_label = run.get("fill_label")
        outputs = run.get("outputs", {})
        for output_name in RUN_OUTPUT_NAMES:
            artifact = outputs[output_name]
            path = Path(artifact["path"])
            inventory.append(
                _artifact_record(path, role=output_name, fill_label=fill_label)
            )
    for report_name in DECISION_REPORT_ROLES:
        artifact = receipt["decision_report"][report_name]
        path = Path(artifact["path"])
        inventory.append(
            _artifact_record(path, role=report_name, fill_label=None)
        )
    return inventory


def verify_receipt_integrity(receipt: dict[str, Any], run_dir: Path) -> GateResult:
    failures: list[str] = []
    if receipt.get("repo_sha") != OFFICIAL_EXECUTION_REPO_SHA:
        failures.append("repo_sha mismatch")
    if not receipt.get("result_complete"):
        failures.append("result_complete is false")
    if receipt.get("has_unresolved_failures"):
        failures.append("has_unresolved_failures is true")
    contract_sha = receipt.get("contract", {}).get("sha256")
    if contract_sha != OFFICIAL_CONTRACT_SHA256:
        failures.append("contract sha256 mismatch")

    runs = receipt.get("runs", [])
    fill_labels = [run.get("fill_label") for run in runs]
    if fill_labels.count("mid") != 1:
        failures.append(f"expected one mid run, found {fill_labels.count('mid')}")
    if fill_labels.count("cross") != 1:
        failures.append(f"expected one cross run, found {fill_labels.count('cross')}")

    decision_report = receipt.get("decision_report", {})
    if set(decision_report.keys()) != set(DECISION_REPORT_ROLES):
        failures.append("decision_report roles mismatch")

    seen_paths: set[str] = set()
    run_dir_resolved = run_dir.resolve()

    for run in runs:
        fill_label = run.get("fill_label")
        outputs = run.get("outputs", {})
        if set(outputs.keys()) != set(RUN_OUTPUT_NAMES):
            failures.append(f"{fill_label} run output roles mismatch")
            continue
        for role in RUN_OUTPUT_NAMES:
            artifact = outputs[role]
            path = Path(artifact["path"]).resolve()
            expected_path = expected_run_output_path(run_dir, role, str(fill_label)).resolve()
            if path != expected_path:
                failures.append(f"{fill_label}/{role} path mismatch")
            if path.parent != run_dir_resolved:
                failures.append(f"{fill_label}/{role} outside run dir")
            path_key = str(path)
            if path_key in seen_paths:
                failures.append(f"duplicate path {path.name}")
            seen_paths.add(path_key)
            if not path.exists():
                failures.append(f"missing artifact {path.name}")
                continue
            actual = sha256_file(path)
            if actual != artifact["sha256"]:
                failures.append(f"receipt hash mismatch {path.name}")

    for role in DECISION_REPORT_ROLES:
        if role not in decision_report:
            continue
        artifact = decision_report[role]
        path = Path(artifact["path"]).resolve()
        expected_path = expected_decision_report_path(run_dir, role).resolve()
        if path != expected_path:
            failures.append(f"{role} path mismatch")
        if path.parent != run_dir_resolved:
            failures.append(f"{role} outside run dir")
        path_key = str(path)
        if path_key in seen_paths:
            failures.append(f"duplicate path {path.name}")
        seen_paths.add(path_key)
        if not path.exists():
            failures.append(f"missing artifact {path.name}")
            continue
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            failures.append(f"receipt hash mismatch {path.name}")

    receipt_path = (run_dir / "run_receipt.json").resolve()
    if str(receipt_path) in seen_paths:
        failures.append("duplicate run_receipt path")
    seen_paths.add(str(receipt_path))
    if not receipt_path.exists():
        failures.append("missing run_receipt.json")

    expected_count = 1 + (len(RUN_OUTPUT_NAMES) * 2) + len(DECISION_REPORT_ROLES)
    if len(seen_paths) != expected_count:
        failures.append(f"unique path count {len(seen_paths)} != {expected_count}")

    inventory = collect_artifact_inventory(receipt, run_dir) if not failures else []
    passed = not failures
    detail = (
        f"inventory={len(inventory) or expected_count} all receipt hashes matched"
        if passed
        else "; ".join(failures[:5])
    )
    return GateResult("G1", passed, detail)


def verify_calendar_completeness_for_fill(run_dir: Path, fill_label: str) -> GateResult:
    status = _load_parquet_columns(
        run_dir / f"date_status_sprint006_baseline_v1_{fill_label}.parquet",
        DATE_STATUS_COLUMNS,
    )
    counts = count_date_classes(status)
    passed = (
        counts["n_expected_dates"] == 403
        and counts["n_failed_dates"] == 0
        and counts["n_valid_no_trade_dates"] == 0
    )
    detail = (
        f"{fill_label}: n_expected={counts['n_expected_dates']} "
        f"n_failed={counts['n_failed_dates']} "
        f"n_valid_no_trade={counts['n_valid_no_trade_dates']}"
    )
    return GateResult(f"G1b_{fill_label}", passed, detail)


def verify_schema_sufficiency(run_dir: Path) -> GateResult:
    checks = {
        "trade_log_mid": (run_dir / "trade_log_sprint006_baseline_v1_mid.parquet", TRADE_LOG_COLUMNS),
        "trade_log_cross": (run_dir / "trade_log_sprint006_baseline_v1_cross.parquet", TRADE_LOG_COLUMNS),
        "leg_log_mid": (run_dir / "leg_log_sprint006_baseline_v1_mid.parquet", LEG_LOG_COLUMNS),
        "leg_log_cross": (run_dir / "leg_log_sprint006_baseline_v1_cross.parquet", LEG_LOG_COLUMNS),
        "date_summary_mid": (run_dir / "date_summary_sprint006_baseline_v1_mid.parquet", DATE_SUMMARY_COLUMNS),
        "date_summary_cross": (run_dir / "date_summary_sprint006_baseline_v1_cross.parquet", DATE_SUMMARY_COLUMNS),
        "date_status_mid": (run_dir / "date_status_sprint006_baseline_v1_mid.parquet", DATE_STATUS_COLUMNS),
        "date_status_cross": (run_dir / "date_status_sprint006_baseline_v1_cross.parquet", DATE_STATUS_COLUMNS),
        "candidate_view_mid": (run_dir / "candidate_view_sprint006_baseline_v1_mid.parquet", CANDIDATE_VIEW_COLUMNS),
        "candidate_view_cross": (run_dir / "candidate_view_sprint006_baseline_v1_cross.parquet", CANDIDATE_VIEW_COLUMNS),
        "funnel_summary_mid": (run_dir / "funnel_summary_sprint006_baseline_v1_mid.parquet", FUNNEL_SUMMARY_COLUMNS),
        "funnel_summary_cross": (run_dir / "funnel_summary_sprint006_baseline_v1_cross.parquet", FUNNEL_SUMMARY_COLUMNS),
    }
    missing_by_file: dict[str, list[str]] = {}
    for label, (path, cols) in checks.items():
        missing = _assert_columns(path, cols)
        if missing:
            missing_by_file[label] = missing

    report_path = run_dir / "decision_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing_report = [key for key in DECISION_REPORT_KEYS if key not in report]

    passed = not missing_by_file and not missing_report
    if passed:
        detail = "required columns present"
    else:
        detail = f"missing columns in {len(missing_by_file)} files"
        if missing_report:
            detail += f"; decision_report missing {missing_report}"
    return GateResult("G2", passed, detail)


def verify_unique_trade_keys(run_dir: Path) -> GateResult:
    failures: list[str] = []
    for label in ("mid", "cross"):
        path = run_dir / f"trade_log_sprint006_baseline_v1_{label}.parquet"
        log = _load_parquet_columns(path, TRADE_LOG_COLUMNS)
        dupes = _duplicate_trade_keys(log)
        if dupes:
            failures.append(f"{label} has {len(dupes)} duplicate trade keys")
    passed = not failures
    return GateResult("G3", passed, failures[0] if failures else "unique trade keys per fill")


def verify_included_key_parity(run_dir: Path) -> GateResult:
    mid_status = _load_parquet_columns(
        run_dir / "date_status_sprint006_baseline_v1_mid.parquet",
        DATE_STATUS_COLUMNS,
    )
    mid_summary = _load_parquet_columns(
        run_dir / "date_summary_sprint006_baseline_v1_mid.parquet",
        DATE_SUMMARY_COLUMNS,
    )
    mid_log = _load_parquet_columns(
        run_dir / "trade_log_sprint006_baseline_v1_mid.parquet",
        TRADE_LOG_COLUMNS,
    )
    cross_log = _load_parquet_columns(
        run_dir / "trade_log_sprint006_baseline_v1_cross.parquet",
        TRADE_LOG_COLUMNS,
    )

    _, _, mid_window = filter_to_window(
        mid_status,
        mid_summary,
        mid_log,
        PRIMARY_START,
        PRIMARY_END,
    )
    cross_status = _load_parquet_columns(
        run_dir / "date_status_sprint006_baseline_v1_cross.parquet",
        DATE_STATUS_COLUMNS,
    )
    cross_summary = _load_parquet_columns(
        run_dir / "date_summary_sprint006_baseline_v1_cross.parquet",
        DATE_SUMMARY_COLUMNS,
    )
    _, _, cross_window = filter_to_window(
        cross_status,
        cross_summary,
        cross_log,
        PRIMARY_START,
        PRIMARY_END,
    )

    mid_keys = _included_trade_keys(mid_window)
    cross_keys = _included_trade_keys(cross_window)
    mid_only = sorted(mid_keys - cross_keys)
    cross_only = sorted(cross_keys - mid_keys)

    passed = not mid_only and not cross_only and len(mid_keys) == ACCEPTED_PRIMARY_INCLUDED_KEYS
    detail = (
        f"primary included keys={len(mid_keys)} mid-only={len(mid_only)} "
        f"cross-only={len(cross_only)} expected={ACCEPTED_PRIMARY_INCLUDED_KEYS}"
    )
    return GateResult("G4", passed, detail)


def verify_leg_quote_settlement_pairing(run_dir: Path) -> GateResult:
    mid_log = _load_parquet_columns(
        run_dir / "trade_log_sprint006_baseline_v1_mid.parquet",
        TRADE_LOG_COLUMNS,
    )
    cross_log = _load_parquet_columns(
        run_dir / "trade_log_sprint006_baseline_v1_cross.parquet",
        TRADE_LOG_COLUMNS,
    )
    mid_status = _load_parquet_columns(
        run_dir / "date_status_sprint006_baseline_v1_mid.parquet",
        DATE_STATUS_COLUMNS,
    )
    mid_summary = _load_parquet_columns(
        run_dir / "date_summary_sprint006_baseline_v1_mid.parquet",
        DATE_SUMMARY_COLUMNS,
    )
    _, _, mid_window = filter_to_window(
        mid_status, mid_summary, mid_log, PRIMARY_START, PRIMARY_END
    )
    cross_status = _load_parquet_columns(
        run_dir / "date_status_sprint006_baseline_v1_cross.parquet",
        DATE_STATUS_COLUMNS,
    )
    cross_summary = _load_parquet_columns(
        run_dir / "date_summary_sprint006_baseline_v1_cross.parquet",
        DATE_SUMMARY_COLUMNS,
    )
    _, _, cross_window = filter_to_window(
        cross_status, cross_summary, cross_log, PRIMARY_START, PRIMARY_END
    )

    trade_keys = _included_trade_keys(mid_window) & _included_trade_keys(cross_window)
    mid_legs = _load_parquet_columns(
        run_dir / "leg_log_sprint006_baseline_v1_mid.parquet",
        LEG_LOG_COLUMNS,
    )
    cross_legs = _load_parquet_columns(
        run_dir / "leg_log_sprint006_baseline_v1_cross.parquet",
        LEG_LOG_COLUMNS,
    )
    pairing = _compare_paired_legs(mid_legs, cross_legs, trade_keys)

    trade_exit_mismatches = 0
    mid_trades = mid_window[mid_window["included_in_portfolio"] == True]  # noqa: E712
    cross_trades = cross_window[cross_window["included_in_portfolio"] == True]  # noqa: E712
    mid_trade_index = {
        (row["trade_date"], str(row["ticker"]), str(row["direction"])): row
        for _, row in mid_trades.iterrows()
    }
    cross_trade_index = {
        (row["trade_date"], str(row["ticker"]), str(row["direction"])): row
        for _, row in cross_trades.iterrows()
    }
    for key in trade_keys:
        if not _float_equal(mid_trade_index[key]["exit_spot"], cross_trade_index[key]["exit_spot"]):
            trade_exit_mismatches += 1

    passed = (
        pairing["missing_leg_keys_in_cross"] == 0
        and pairing["missing_leg_keys_in_mid"] == 0
        and pairing["quote_mismatches"] == 0
        and pairing["settlement_mismatches"] == 0
        and pairing["unit_quantity_mismatches"] == 0
        and pairing["mid_fill_convention_violations"] == 0
        and pairing["cross_fill_convention_violations"] == 0
        and pairing["mid_fill_label_mismatches"] == 0
        and pairing["cross_fill_label_mismatches"] == 0
        and trade_exit_mismatches == 0
    )
    detail = (
        f"unit_quantity_mismatches={pairing['unit_quantity_mismatches']} "
        f"mid_fill_violations={pairing['mid_fill_convention_violations']} "
        f"cross_fill_violations={pairing['cross_fill_convention_violations']} "
        f"mid_fill_label_mismatches={pairing['mid_fill_label_mismatches']} "
        f"cross_fill_label_mismatches={pairing['cross_fill_label_mismatches']} "
        f"leg_quote_mismatches={pairing['quote_mismatches']} "
        f"leg_settlement_mismatches={pairing['settlement_mismatches']} "
        f"trade_exit_mismatches={trade_exit_mismatches}"
    )
    return GateResult("G5", passed, detail)


def run_d0_validation(
    *,
    run_dir: Path | None = None,
    receipt_path: Path | None = None,
    d0_code_commit_sha: str | None = None,
) -> D0ValidationResult:
    run_dir = run_dir or OFFICIAL_RUN_DIR
    receipt_path = receipt_path or (run_dir / "run_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    d0_code_commit_sha = d0_code_commit_sha or get_current_repo_sha()

    gates = [
        verify_receipt_integrity(receipt, run_dir),
        verify_calendar_completeness_for_fill(run_dir, "mid"),
        verify_calendar_completeness_for_fill(run_dir, "cross"),
        verify_schema_sufficiency(run_dir),
        verify_unique_trade_keys(run_dir),
        verify_included_key_parity(run_dir),
        verify_leg_quote_settlement_pairing(run_dir),
        GateResult("G7", True, "scope limited to readiness checks"),
    ]

    inventory = collect_artifact_inventory(receipt, run_dir)
    verdict = (
        "READY_WITH_NARROW_ENABLING_CHANGE"
        if all(g.passed for g in gates)
        else "BLOCKED_BY_SPECIFIC_EVIDENCE_GAP"
    )

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_run_dir": str(run_dir),
        "receipt_path": str(receipt_path),
        "sprint006_execution_repo_sha": OFFICIAL_EXECUTION_REPO_SHA,
        "d0_code_commit_sha": d0_code_commit_sha,
        "verdict": verdict,
        "gates": [{"gate_id": g.gate_id, "passed": g.passed, "detail": g.detail} for g in gates],
        "artifact_inventory": inventory,
        "required_columns": {
            "trade_log": list(TRADE_LOG_COLUMNS),
            "leg_log": list(LEG_LOG_COLUMNS),
        },
    }

    return D0ValidationResult(verdict=verdict, gates=gates, manifest=manifest)


def write_manifest(result: D0ValidationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    return output_path


def write_execution_receipt(
    *,
    evidence_dir: Path,
    executed_notebook: Path,
    html_export: Path,
    d0_code_commit_sha: str | None = None,
    sprint006_execution_repo_sha: str = OFFICIAL_EXECUTION_REPO_SHA,
) -> Path:
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "d0_code_commit_sha": d0_code_commit_sha or get_current_repo_sha(),
        "sprint006_execution_repo_sha": sprint006_execution_repo_sha,
        "executed_notebook": str(executed_notebook),
        "executed_notebook_sha256": sha256_file(executed_notebook),
        "html_export": str(html_export),
        "html_export_sha256": sha256_file(html_export),
    }
    path = evidence_dir / "execution_receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def export_d0_evidence(
    *,
    result: D0ValidationResult,
    clean_notebook: Path,
    evidence_dir: Path | None = None,
    d0_code_commit_sha: str | None = None,
) -> Path:
    """Execute notebook fresh, export HTML, and write manifest + execution receipt."""
    evidence_dir = evidence_dir or Path(
        f"C:/MomentumCVG_env/runs/sprint007_d0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    executed = evidence_dir / "d0_readiness.executed.ipynb"
    html_path = evidence_dir / "d0_readiness.html"
    d0_code_commit_sha = d0_code_commit_sha or get_current_repo_sha()

    import os

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    python = Path("C:/MomentumCVG_env/venv/Scripts/python.exe")
    if not python.exists():
        python = Path(sys.executable)
    jupyter = [str(python), "-m", "jupyter"]

    subprocess.run(
        [
            *jupyter,
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(clean_notebook),
            "--output",
            executed.name,
            "--output-dir",
            str(evidence_dir),
            "--ExecutePreprocessor.kernel_name=momentumcvg",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [
            *jupyter,
            "nbconvert",
            "--to",
            "html",
            str(executed),
            "--output",
            html_path.name,
            "--output-dir",
            str(evidence_dir),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    write_manifest(result, evidence_dir / "d0_artifact_manifest.json")
    write_execution_receipt(
        evidence_dir=evidence_dir,
        executed_notebook=executed,
        html_export=html_path,
        d0_code_commit_sha=d0_code_commit_sha,
    )
    return evidence_dir
