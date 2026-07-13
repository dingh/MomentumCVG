"""Standalone PIT-universe audit CLI (Sprint 004 C7.3).

Exposes the accepted pure audit functions in ``src.data.pit_universe_audit``
through a safe CLI that loads A3 liquidity artifacts, evaluates explicit and/or
discovered samples, and writes a deterministic Markdown report.

Does not mutate artifacts, wire into refresh, or run production evidence.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.pit_universe_audit import (  # noqa: E402
    FAIL,
    FULL_REQUIRED_COLUMNS,
    PASS,
    WARN,
    ArtifactValidationError,
    ArtifactCheckResult,
    ArtifactEnvelopeResult,
    FullHistorySupersetCoverageResult,
    PitResolutionResult,
    PitAuditError,
    PitUniverseAuditReport,
    RollingProvenanceResult,
    Status,
    SupersetCoverageResult,
    assemble_audit_report,
    check_artifact_envelope,
    check_build_param_homogeneity,
    check_full_history_superset_coverage,
    check_panel_grain,
    check_panel_metric_integrity,
    check_required_columns,
    check_rolling_provenance,
    check_superset_coverage,
    check_ticker_validity,
    check_weekly_artifact,
    classify_snapshot_membership,
    evaluate_pit_sample,
    extract_liquid_ticker_set,
    normalize_date_column,
    normalize_date_value,
    read_superset_build_params,
)

DEFAULT_PANEL_PATH = Path(
    "C:/MomentumCVG_env/input/liquidity/ticker_liquidity_panel.parquet"
)
DEFAULT_WEEKLY_PATH = Path(
    "C:/MomentumCVG_env/input/liquidity/ticker_liquidity_weekly_observations.parquet"
)
DEFAULT_LIQUID_TICKERS_PATH = Path(
    "C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv"
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DISCOVERY_LABEL_NORMAL = "normal"
DISCOVERY_LABEL_BOUNDARY = "boundary_or_gap"
DISCOVERY_LABEL_MISSING = "missing_or_new_liquidity"
DISCOVERY_LABEL_ORDER = (
    DISCOVERY_LABEL_NORMAL,
    DISCOVERY_LABEL_BOUNDARY,
    DISCOVERY_LABEL_MISSING,
)


class UsageError(Exception):
    """CLI usage / configuration / path / read / write error (exit 2)."""


class InternalError(Exception):
    """Unexpected internal programming failure (exit 2)."""


_ARTIFACT_RUNTIME_EXCEPTIONS = (
    ArtifactValidationError,
    PitAuditError,
    ValueError,
    TypeError,
    OverflowError,
)

_PROVENANCE_INT_COLUMNS = (
    "lookback_weeks",
    "min_valid_quote_weeks",
    "dte_min",
    "dte_max",
    "window_shortfall",
    "valid_quote_weeks",
    "zero_volume_weeks",
)
_PROVENANCE_DATE_COLUMNS = ("window_start_date", "window_end_date")


@dataclass(frozen=True)
class SampleSpec:
    trade_date: pd.Timestamp
    target_snapshot_date: Optional[pd.Timestamp]
    labels: tuple[str, ...] = ()


@dataclass
class AuditRunContext:
    panel: pd.DataFrame
    weekly: pd.DataFrame
    liquid_tickers: pd.DataFrame
    liquid_set: set[str]
    panel_path: Path
    weekly_path: Path
    liquid_tickers_path: Path
    dvol_top_pct: float
    spread_bottom_pct: float
    max_examples: int
    strict: bool
    sample_labels: dict[tuple[Optional[str], str], tuple[str, ...]] = field(
        default_factory=dict
    )
    discovery_notes: list[str] = field(default_factory=list)
    rolling_na_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit point-in-time liquidity universe (A3 / S1 trust gate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--panel-path",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help="Path to ticker_liquidity_panel.parquet.",
    )
    p.add_argument(
        "--weekly-path",
        type=Path,
        default=DEFAULT_WEEKLY_PATH,
        help="Path to ticker_liquidity_weekly_observations.parquet.",
    )
    p.add_argument(
        "--liquid-tickers-path",
        type=Path,
        default=DEFAULT_LIQUID_TICKERS_PATH,
        help="Path to liquid_tickers.csv.",
    )
    p.add_argument(
        "--sample-date",
        action="append",
        default=None,
        dest="sample_dates",
        metavar="YYYY-MM-DD",
        help="Operator trade date to audit (repeatable).",
    )
    p.add_argument(
        "--discover-samples",
        action="store_true",
        help="Deterministically discover up to three audit sample cases.",
    )
    p.add_argument(
        "--dvol-top-pct",
        type=float,
        default=0.20,
        help="Requested S1 dvol top percentile.",
    )
    p.add_argument(
        "--spread-bottom-pct",
        type=float,
        default=1.0,
        help="Requested S1 spread bottom percentile.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Cap on checked tickers / listed examples.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat overall WARN as exit code 1.",
    )
    p.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Markdown report output path (required; no repo default).",
    )
    return p.parse_args(argv)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if not args.sample_dates and not args.discover_samples:
        raise UsageError(
            "require at least one of --sample-date or --discover-samples"
        )
    if args.max_examples < 1:
        raise UsageError(f"--max-examples must be >= 1 (got {args.max_examples})")
    if not (0.0 < float(args.dvol_top_pct) <= 1.0):
        raise UsageError(
            f"--dvol-top-pct must be in (0, 1] (got {args.dvol_top_pct})"
        )
    if not (0.0 < float(args.spread_bottom_pct) <= 1.0):
        raise UsageError(
            f"--spread-bottom-pct must be in (0, 1] (got {args.spread_bottom_pct})"
        )
    if args.sample_dates:
        for raw in args.sample_dates:
            _parse_iso_sample_date(raw)


def _parse_iso_sample_date(raw: str) -> pd.Timestamp:
    text = str(raw).strip()
    if not _ISO_DATE_RE.fullmatch(text):
        raise UsageError(f"non-ISO sample date (expected YYYY-MM-DD): {raw!r}")
    try:
        return normalize_date_value(text, label="sample-date")
    except Exception as exc:  # noqa: BLE001 — surface as usage error
        raise UsageError(f"invalid sample date {raw!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Loading / preflight
# ---------------------------------------------------------------------------


def _normalize_path_for_safety(path: Path) -> Path:
    """Absolute normalized path; does not require the path to exist."""
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(os.path.abspath(str(path))))


def validate_output_path_safety(
    output_report: Path,
    input_paths: Sequence[Path],
) -> None:
    """Reject output paths that alias or overwrite input artifacts."""
    out = _normalize_path_for_safety(output_report)
    labels = ("panel", "weekly", "liquid-tickers")
    for label, raw in zip(labels, input_paths):
        inp = _normalize_path_for_safety(raw)
        if out == inp:
            raise UsageError(
                f"output-report conflicts with {label} path: output={out} input={inp}"
            )
        if out.exists() and inp.exists():
            try:
                if out.samefile(inp):
                    raise UsageError(
                        f"output-report aliases {label} path: output={out} input={inp}"
                    )
            except OSError:
                pass


def _require_readable_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise UsageError(f"missing {label} path: {path}")
    if not path.is_file():
        raise UsageError(f"{label} path is not a file: {path}")
    return path


def _load_artifacts(
    panel_path: Path,
    weekly_path: Path,
    liquid_tickers_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    panel_path = _require_readable_file(panel_path, label="panel")
    weekly_path = _require_readable_file(weekly_path, label="weekly")
    liquid_tickers_path = _require_readable_file(
        liquid_tickers_path, label="liquid-tickers"
    )

    try:
        panel = pd.read_parquet(panel_path)
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"unreadable panel parquet {panel_path}: {exc}") from exc
    try:
        weekly = pd.read_parquet(weekly_path)
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"unreadable weekly parquet {weekly_path}: {exc}") from exc
    try:
        liquid_tickers = pd.read_csv(liquid_tickers_path)
    except Exception as exc:  # noqa: BLE001
        raise UsageError(
            f"unreadable liquid tickers CSV {liquid_tickers_path}: {exc}"
        ) from exc

    if "Ticker" not in liquid_tickers.columns and "ticker" not in liquid_tickers.columns:
        raise UsageError(
            "missing required Ticker column in liquid_tickers.csv "
            f"(columns={list(liquid_tickers.columns)})"
        )
    # Spec requires the canonical Ticker column for this CLI contract.
    if "Ticker" not in liquid_tickers.columns:
        raise UsageError(
            "missing required Ticker column in liquid_tickers.csv "
            f"(columns={list(liquid_tickers.columns)})"
        )

    try:
        liquid_set = extract_liquid_ticker_set(liquid_tickers)
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"invalid liquid tickers CSV: {exc}") from exc
    if not liquid_set:
        raise UsageError("empty normalized liquid-ticker set")

    return panel, weekly, liquid_tickers, liquid_set


def _ensure_output_parent_writable(output_report: Path) -> None:
    parent = output_report.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise UsageError(
            f"unwritable output parent {parent}: {exc}"
        ) from exc
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=".audit_pit_universe_probe_",
            delete=True,
        ) as probe:
            probe.write("")
            probe.flush()
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"unwritable output parent {parent}: {exc}") from exc


def _is_strict_bool_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _parse_provenance_int(value: object) -> Optional[int]:
    """Parse a strict integer provenance scalar.

    Accepts only ``int``, ``float``, ``np.integer``, and ``np.floating`` values
    that are finite and integer-valued. Rejects strings, booleans, null, and
    non-finite values without coercion.
    """
    if isinstance(value, (str, bool, np.bool_)):
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def _provenance_invalid_details(
    invalid_count: int,
    examples: Sequence[str],
    *,
    max_examples: int,
) -> dict[str, object]:
    capped = sorted(examples[:max_examples])
    return {
        "invalid_count": invalid_count,
        "examples": capped,
        "examples_capped": invalid_count > len(capped),
    }


def _record_provenance_invalid(
    *,
    invalid_count: int,
    examples: list[str],
    max_examples: int,
    example: str,
) -> int:
    invalid_count += 1
    if len(examples) < max_examples:
        examples.append(example)
    return invalid_count


def _check_panel_provenance_integers(
    panel: pd.DataFrame, *, max_examples: int
) -> list[ArtifactCheckResult]:
    """CLI-level integer provenance validation before sample execution."""
    if len(panel) == 0:
        return []

    missing_cols = [c for c in _PROVENANCE_INT_COLUMNS if c not in panel.columns]
    if missing_cols:
        return [
            ArtifactCheckResult(
                name="panel_provenance_integers",
                status=FAIL,
                message=f"missing provenance integer columns: {missing_cols}",
                details={"missing": missing_cols},
            )
        ]

    examples: list[str] = []
    invalid_count = 0

    def _record(field: str, ticker: str, snap: str, value: object) -> None:
        nonlocal invalid_count
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"{ticker}/{snap}/{field}={value!r}",
        )

    def _parse_field(value: object, field: str, ticker: str, snap: str) -> Optional[int]:
        parsed = _parse_provenance_int(value)
        if parsed is None:
            _record(field, ticker, snap, value)
        return parsed

    # Stamp-level build params from first row.
    lb = _parse_field(panel["lookback_weeks"].iloc[0], "lookback_weeks", "-", "-")
    mvqw = _parse_field(
        panel["min_valid_quote_weeks"].iloc[0], "min_valid_quote_weeks", "-", "-"
    )
    dte_min = _parse_field(panel["dte_min"].iloc[0], "dte_min", "-", "-")
    dte_max = _parse_field(panel["dte_max"].iloc[0], "dte_max", "-", "-")

    if lb is not None and lb <= 0:
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"-/-/lookback_weeks={lb}",
        )
    if mvqw is not None and mvqw <= 0:
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"-/-/min_valid_quote_weeks={mvqw}",
        )
    if lb is not None and mvqw is not None and mvqw > lb:
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"-/-/min_valid_quote_weeks>{lb}",
        )
    if dte_min is not None and dte_min < 0:
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"-/-/dte_min={dte_min}",
        )
    if dte_min is not None and dte_max is not None and dte_min > dte_max:
        invalid_count = _record_provenance_invalid(
            invalid_count=invalid_count,
            examples=examples,
            max_examples=max_examples,
            example=f"-/-/dte_min>{dte_max}",
        )

    month = None
    if "month_date" in panel.columns:
        try:
            month = normalize_date_column(panel, "month_date")
        except ArtifactValidationError:
            month = None

    for idx, row in panel.iterrows():
        ticker = str(row.get("ticker", idx))
        snap = (
            month.loc[idx].date().isoformat()
            if month is not None
            else str(row.get("month_date", "-"))
        )
        for field in ("window_shortfall", "valid_quote_weeks", "zero_volume_weeks"):
            val = _parse_field(row[field], field, ticker, snap)
            if val is None:
                continue
            if val < 0:
                invalid_count = _record_provenance_invalid(
                    invalid_count=invalid_count,
                    examples=examples,
                    max_examples=max_examples,
                    example=f"{ticker}/{snap}/{field}={val}",
                )
            if lb is not None and val > lb:
                invalid_count = _record_provenance_invalid(
                    invalid_count=invalid_count,
                    examples=examples,
                    max_examples=max_examples,
                    example=f"{ticker}/{snap}/{field}>{lb}={val}",
                )

    return [
        ArtifactCheckResult(
            name="panel_provenance_integers",
            status=FAIL if invalid_count else PASS,
            message=(
                f"{invalid_count} invalid integer provenance value(s)"
                if invalid_count
                else "panel integer provenance values valid"
            ),
            details=_provenance_invalid_details(
                invalid_count, examples, max_examples=max_examples
            ),
        )
    ]


def _check_panel_provenance_dates(
    panel: pd.DataFrame, *, max_examples: int
) -> list[ArtifactCheckResult]:
    if len(panel) == 0:
        return []

    missing_cols = [c for c in _PROVENANCE_DATE_COLUMNS if c not in panel.columns]
    if missing_cols:
        return [
            ArtifactCheckResult(
                name="panel_provenance_dates",
                status=FAIL,
                message=f"missing provenance date columns: {missing_cols}",
                details={"missing": missing_cols},
            )
        ]

    examples: list[str] = []
    invalid_count = 0
    try:
        month = normalize_date_column(panel, "month_date")
    except ArtifactValidationError as exc:
        return [
            ArtifactCheckResult(
                name="panel_provenance_dates",
                status=FAIL,
                message=f"cannot normalize month_date: {exc}",
                details={},
            )
        ]

    for idx, row in panel.iterrows():
        ticker = str(row.get("ticker", idx))
        snap_ts = month.loc[idx]
        snap = snap_ts.date().isoformat()
        parsed: dict[str, pd.Timestamp] = {}
        for field in _PROVENANCE_DATE_COLUMNS:
            try:
                parsed[field] = normalize_date_value(row[field], label=field)
            except ArtifactValidationError:
                invalid_count = _record_provenance_invalid(
                    invalid_count=invalid_count,
                    examples=examples,
                    max_examples=max_examples,
                    example=f"{ticker}/{snap}/{field}=unparseable",
                )
                parsed = {}
                break
        if not parsed:
            continue
        if parsed["window_start_date"] > parsed["window_end_date"]:
            invalid_count = _record_provenance_invalid(
                invalid_count=invalid_count,
                examples=examples,
                max_examples=max_examples,
                example=(
                    f"{ticker}/{snap}/window_start>{parsed['window_end_date'].date()}"
                ),
            )
        if parsed["window_end_date"] != snap_ts.normalize():
            invalid_count = _record_provenance_invalid(
                invalid_count=invalid_count,
                examples=examples,
                max_examples=max_examples,
                example=(
                    f"{ticker}/{snap}/window_end!={snap} "
                    f"(got {parsed['window_end_date'].date()})"
                ),
            )

    return [
        ArtifactCheckResult(
            name="panel_provenance_dates",
            status=FAIL if invalid_count else PASS,
            message=(
                f"{invalid_count} invalid provenance date value(s)"
                if invalid_count
                else "panel provenance dates valid"
            ),
            details=_provenance_invalid_details(
                invalid_count, examples, max_examples=max_examples
            ),
        )
    ]


def _check_panel_provenance_boolean(
    panel: pd.DataFrame, *, max_examples: int
) -> list[ArtifactCheckResult]:
    if len(panel) == 0 or "has_valid_atm_pair" not in panel.columns:
        return []

    examples: list[str] = []
    invalid_count = 0
    month = None
    try:
        month = normalize_date_column(panel, "month_date")
    except ArtifactValidationError:
        month = None

    for idx, value in panel["has_valid_atm_pair"].items():
        if not _is_strict_bool_scalar(value):
            ticker = str(panel.at[idx, "ticker"]) if "ticker" in panel.columns else str(idx)
            snap = (
                month.loc[idx].date().isoformat()
                if month is not None
                else str(panel.at[idx, "month_date"])
            )
            invalid_count = _record_provenance_invalid(
                invalid_count=invalid_count,
                examples=examples,
                max_examples=max_examples,
                example=f"{ticker}/{snap}/has_valid_atm_pair={value!r}",
            )

    return [
        ArtifactCheckResult(
            name="panel_provenance_boolean",
            status=FAIL if invalid_count else PASS,
            message=(
                f"{invalid_count} non-boolean has_valid_atm_pair value(s)"
                if invalid_count
                else "has_valid_atm_pair is strictly boolean"
            ),
            details=_provenance_invalid_details(
                invalid_count, examples, max_examples=max_examples
            ),
        )
    ]


def _check_panel_provenance_fields(
    panel: pd.DataFrame, *, max_examples: int
) -> list[ArtifactCheckResult]:
    """CLI-level provenance validation before expensive sample work."""
    checks: list[ArtifactCheckResult] = []
    checks.extend(_check_panel_provenance_integers(panel, max_examples=max_examples))
    checks.extend(_check_panel_provenance_dates(panel, max_examples=max_examples))
    checks.extend(_check_panel_provenance_boolean(panel, max_examples=max_examples))
    return checks


# ---------------------------------------------------------------------------
# Artifact checks (CLI orchestration only)
# ---------------------------------------------------------------------------


def _cli_nonempty_checks(
    panel: pd.DataFrame, weekly: pd.DataFrame
) -> list[ArtifactCheckResult]:
    checks: list[ArtifactCheckResult] = []

    if len(panel) == 0:
        checks.append(
            ArtifactCheckResult(
                name="panel_nonempty",
                status=FAIL,
                message="panel is empty",
                details={"row_count": 0},
            )
        )
    else:
        checks.append(
            ArtifactCheckResult(
                name="panel_nonempty",
                status=PASS,
                message="panel has rows",
                details={"row_count": int(len(panel))},
            )
        )

    if "month_date" in panel.columns and len(panel) > 0:
        try:
            n_snap = int(normalize_date_column(panel, "month_date").nunique())
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ArtifactCheckResult(
                    name="panel_snapshots",
                    status=FAIL,
                    message=f"cannot count panel snapshots: {exc}",
                    details={},
                )
            )
            n_snap = 0
        else:
            checks.append(
                ArtifactCheckResult(
                    name="panel_snapshots",
                    status=FAIL if n_snap == 0 else PASS,
                    message=(
                        "panel has zero distinct snapshots"
                        if n_snap == 0
                        else f"panel has {n_snap} distinct snapshots"
                    ),
                    details={"snapshot_count": n_snap},
                )
            )
    elif len(panel) > 0:
        checks.append(
            ArtifactCheckResult(
                name="panel_snapshots",
                status=FAIL,
                message="cannot count snapshots without month_date",
                details={},
            )
        )

    if len(weekly) == 0:
        checks.append(
            ArtifactCheckResult(
                name="weekly_nonempty",
                status=FAIL,
                message="weekly artifact is empty",
                details={"row_count": 0},
            )
        )
    else:
        checks.append(
            ArtifactCheckResult(
                name="weekly_nonempty",
                status=PASS,
                message="weekly artifact has rows",
                details={"row_count": int(len(weekly))},
            )
        )

    if "week_end_date" in weekly.columns and len(weekly) > 0:
        try:
            n_weeks = int(normalize_date_column(weekly, "week_end_date").nunique())
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ArtifactCheckResult(
                    name="weekly_weeks",
                    status=FAIL,
                    message=f"cannot count weekly weeks: {exc}",
                    details={},
                )
            )
        else:
            checks.append(
                ArtifactCheckResult(
                    name="weekly_weeks",
                    status=FAIL if n_weeks == 0 else PASS,
                    message=(
                        "weekly artifact has zero distinct weeks"
                        if n_weeks == 0
                        else f"weekly artifact has {n_weeks} distinct weeks"
                    ),
                    details={"week_count": n_weeks},
                )
            )
    elif len(weekly) > 0:
        checks.append(
            ArtifactCheckResult(
                name="weekly_weeks",
                status=FAIL,
                message="cannot count weeks without week_end_date",
                details={},
            )
        )

    return checks


def _run_full_artifact_checks(
    panel: pd.DataFrame, weekly: pd.DataFrame, *, max_examples: int
) -> list[ArtifactCheckResult]:
    checks: list[ArtifactCheckResult] = []
    checks.extend(_cli_nonempty_checks(panel, weekly))
    checks.append(check_required_columns(panel, required=FULL_REQUIRED_COLUMNS))
    checks.append(check_panel_grain(panel))
    checks.append(check_ticker_validity(panel))
    # Empty frames can raise inside homogeneity/metric helpers; convert to FAIL evidence.
    if len(panel) == 0:
        checks.append(
            ArtifactCheckResult(
                name="build_param_homogeneity",
                status=FAIL,
                message="panel is empty; build-parameter homogeneity not evaluable",
                details={},
            )
        )
        checks.append(
            ArtifactCheckResult(
                name="panel_metric_integrity",
                status=FAIL,
                message="panel is empty; panel metric integrity not evaluable",
                details={},
            )
        )
    else:
        try:
            checks.append(check_build_param_homogeneity(panel))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ArtifactCheckResult(
                    name="build_param_homogeneity",
                    status=FAIL,
                    message=str(exc),
                    details={},
                )
            )
        try:
            checks.append(check_panel_metric_integrity(panel))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                ArtifactCheckResult(
                    name="panel_metric_integrity",
                    status=FAIL,
                    message=str(exc),
                    details={},
                )
            )
        checks.extend(_check_panel_provenance_fields(panel, max_examples=max_examples))
    try:
        checks.extend(check_weekly_artifact(weekly))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            ArtifactCheckResult(
                name="weekly_artifact",
                status=FAIL,
                message=str(exc),
                details={},
            )
        )
    return checks


def _has_blocking_fail(checks: Sequence[ArtifactCheckResult]) -> bool:
    return any(c.status == FAIL for c in checks)


# ---------------------------------------------------------------------------
# Deterministic sample discovery
# ---------------------------------------------------------------------------


def _panel_snapshots(panel: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(set(normalize_date_column(panel, "month_date").tolist()))


def _weekly_trade_candidates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(set(normalize_date_column(weekly, "week_end_date").tolist()))


def _resolve_prior_snapshot(
    trade_date: pd.Timestamp, snapshots: Sequence[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    prior = [s for s in snapshots if s < trade_date]
    return max(prior) if prior else None


def map_snapshot_to_trade_date(
    target_snapshot: pd.Timestamp,
    panel_snapshots: Sequence[pd.Timestamp],
    weekly_candidates: Sequence[pd.Timestamp],
) -> Optional[pd.Timestamp]:
    """Earliest weekly candidate T > S with max(panel month_date < T) == S."""
    s = normalize_date_value(target_snapshot, label="target_snapshot")
    for t in weekly_candidates:
        if t <= s:
            continue
        resolved = _resolve_prior_snapshot(t, panel_snapshots)
        if resolved == s:
            return t
    return None


def _eligible_row_count(snap_rows: pd.DataFrame) -> int:
    if snap_rows.empty:
        return 0
    valid = snap_rows["has_valid_atm_pair"] == True  # noqa: E712
    return int(
        (valid & snap_rows["atm_straddle_dollar_vol"].notna() & snap_rows["atm_spread_pct"].notna()).sum()
    )


def _middle_third(items: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    n = len(items)
    if n == 0:
        return []
    lo = n // 3
    hi = (2 * n) // 3
    if hi <= lo:
        # Small panels: use the single middle element.
        return [items[n // 2]]
    group = list(items[lo:hi])
    return group if group else [items[n // 2]]


def _discover_normal_snapshot(
    panel: pd.DataFrame, snapshots: Sequence[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    month = normalize_date_column(panel, "month_date")
    candidates = _middle_third(list(snapshots))
    if not candidates:
        return None
    counts = []
    for s in candidates:
        n = _eligible_row_count(panel.loc[month == s])
        counts.append((s, n))
    median = float(pd.Series([c for _, c in counts]).median())
    best = sorted(counts, key=lambda x: (abs(x[1] - median), x[0]))[0]
    return best[0]


def _discover_boundary_snapshot(
    snapshots: Sequence[pd.Timestamp],
) -> Optional[pd.Timestamp]:
    if len(snapshots) < 2:
        return None
    gap_hits: list[pd.Timestamp] = []
    month_hits: list[pd.Timestamp] = []
    for prev, cur in zip(snapshots, snapshots[1:]):
        days = int((cur - prev).days)
        if days > 7:
            gap_hits.append(cur)
        if (cur.year, cur.month) != (prev.year, prev.month):
            month_hits.append(cur)
    if gap_hits:
        return min(gap_hits)
    if month_hits:
        return min(month_hits)
    return None


def _discover_missing_or_new_snapshot(
    panel: pd.DataFrame, snapshots: Sequence[pd.Timestamp]
) -> tuple[Optional[pd.Timestamp], list[str]]:
    """Priority: liquidity defect → genuinely new ticker → baseline fallback."""
    notes: list[str] = []
    if not snapshots:
        return None, notes

    month = normalize_date_column(panel, "month_date")
    min_vqw = None
    if "min_valid_quote_weeks" in panel.columns and len(panel):
        try:
            min_vqw = int(panel["min_valid_quote_weeks"].iloc[0])
        except (TypeError, ValueError):
            min_vqw = None

    # Priority 1 — actual liquidity defect or insufficient history.
    for s in snapshots:
        snap = panel.loc[month == s]
        if snap.empty:
            continue
        invalid = (snap["has_valid_atm_pair"] != True).any()  # noqa: E712
        missing_dvol = snap["atm_straddle_dollar_vol"].isna().any()
        missing_spread = snap["atm_spread_pct"].isna().any()
        short_hist = False
        if min_vqw is not None and "valid_quote_weeks" in snap.columns:
            vqw = pd.to_numeric(snap["valid_quote_weeks"], errors="coerce")
            short_hist = bool((vqw < min_vqw).any())
        if invalid or missing_dvol or missing_spread or short_hist:
            return s, notes

    # Priority 2 — genuinely new ticker after the first snapshot.
    if len(snapshots) >= 2:
        seen: set[str] = set()
        first = snapshots[0]
        seen.update(panel.loc[month == first, "ticker"].astype(str).tolist())
        for s in snapshots[1:]:
            snap = panel.loc[month == s]
            if snap.empty:
                continue
            tickers = sorted(snap["ticker"].astype(str).tolist())
            new_tickers = [t for t in tickers if t not in seen]
            seen.update(tickers)
            if new_tickers:
                return s, notes

    # Priority 3 — baseline fallback on initial population only.
    notes.append("missing_or_new_liquidity used baseline_initial_population fallback")
    return snapshots[0], notes


def discover_audit_samples(
    panel: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[list[SampleSpec], ArtifactCheckResult]:
    """Deterministically discover up to three labeled sample cases."""
    snapshots = _panel_snapshots(panel)
    candidates = _weekly_trade_candidates(weekly)
    notes: list[str] = []

    chosen: dict[str, Optional[tuple[pd.Timestamp, pd.Timestamp]]] = {
        DISCOVERY_LABEL_NORMAL: None,
        DISCOVERY_LABEL_BOUNDARY: None,
        DISCOVERY_LABEL_MISSING: None,
    }

    normal_s = _discover_normal_snapshot(panel, snapshots)
    if normal_s is not None:
        t = map_snapshot_to_trade_date(normal_s, snapshots, candidates)
        if t is None:
            notes.append(f"normal: no trade-date mapping for snapshot {normal_s.date()}")
        else:
            chosen[DISCOVERY_LABEL_NORMAL] = (normal_s, t)

    boundary_s = _discover_boundary_snapshot(snapshots)
    if boundary_s is not None:
        t = map_snapshot_to_trade_date(boundary_s, snapshots, candidates)
        if t is None:
            notes.append(
                f"boundary_or_gap: no trade-date mapping for snapshot {boundary_s.date()}"
            )
        else:
            chosen[DISCOVERY_LABEL_BOUNDARY] = (boundary_s, t)

    missing_s, missing_notes = _discover_missing_or_new_snapshot(panel, snapshots)
    notes.extend(missing_notes)
    if missing_s is not None:
        t = map_snapshot_to_trade_date(missing_s, snapshots, candidates)
        if t is None:
            notes.append(
                f"missing_or_new_liquidity: no trade-date mapping for snapshot {missing_s.date()}"
            )
        else:
            chosen[DISCOVERY_LABEL_MISSING] = (missing_s, t)

    # Group by (target, trade); collect labels; evaluate once.
    grouped: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
    mapping_mismatch = False
    for label in DISCOVERY_LABEL_ORDER:
        pair = chosen[label]
        if pair is None:
            continue
        s, t = pair
        if t == s:
            mapping_mismatch = True
            notes.append(f"{label}: refuse trade_date == target snapshot {s.date()}")
            continue
        resolved = _resolve_prior_snapshot(t, snapshots)
        if resolved != s:
            mapping_mismatch = True
            notes.append(
                f"{label}: target {s.date()} != resolved {None if resolved is None else resolved.date()} "
                f"for trade_date {t.date()}"
            )
            continue
        grouped.setdefault((s, t), []).append(label)

    specs: list[SampleSpec] = []
    for (s, t), labels in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        specs.append(
            SampleSpec(
                trade_date=t,
                target_snapshot_date=s,
                labels=tuple(labels),
            )
        )

    n_mapped = len(specs)
    if mapping_mismatch:
        status: Status = FAIL
        message = "discovery mapping mismatch (target != resolved or T == S)"
    elif n_mapped >= 3:
        status = PASS
        message = f"{n_mapped} distinct mapped discovery cases found"
    elif n_mapped >= 1:
        status = WARN
        message = f"only {n_mapped} distinct mapped discovery case(s) found (expected 3)"
    else:
        status = FAIL
        message = "zero mapped discovery cases found"

    details = {
        "mapped_case_count": n_mapped,
        "labels": [list(s.labels) for s in specs],
        "pairs": [
            {
                "labels": list(s.labels),
                "target_snapshot_date": s.target_snapshot_date.date().isoformat()
                if s.target_snapshot_date is not None
                else None,
                "trade_date": s.trade_date.date().isoformat(),
            }
            for s in specs
        ],
        "notes": notes,
    }
    check = ArtifactCheckResult(
        name="sample_discovery",
        status=status,
        message=message,
        details=details,
    )
    return specs, check


def _dedupe_explicit_dates(raw_dates: Sequence[str]) -> list[pd.Timestamp]:
    seen: set[pd.Timestamp] = set()
    out: list[pd.Timestamp] = []
    parsed = sorted({_parse_iso_sample_date(r) for r in raw_dates})
    for ts in parsed:
        if ts not in seen:
            seen.add(ts)
            out.append(ts)
    return out


def _combine_sample_specs(
    explicit: Sequence[SampleSpec],
    discovered: Sequence[SampleSpec],
) -> list[SampleSpec]:
    """Deduplicate by (trade_date, target_snapshot_date); merge labels."""
    bucket: dict[tuple[pd.Timestamp, Optional[pd.Timestamp]], list[str]] = {}
    order: list[tuple[pd.Timestamp, Optional[pd.Timestamp]]] = []

    def _add(spec: SampleSpec) -> None:
        key = (spec.trade_date, spec.target_snapshot_date)
        if key not in bucket:
            bucket[key] = []
            order.append(key)
        for lab in spec.labels:
            if lab not in bucket[key]:
                bucket[key].append(lab)

    for spec in explicit:
        _add(spec)
    for spec in discovered:
        _add(spec)

    order_sorted = sorted(order, key=lambda k: (k[0], k[1] is not None, k[1] or pd.Timestamp.min))
    return [
        SampleSpec(
            trade_date=k[0],
            target_snapshot_date=k[1],
            labels=tuple(bucket[k]),
        )
        for k in order_sorted
    ]


# ---------------------------------------------------------------------------
# Checked tickers / rolling
# ---------------------------------------------------------------------------


def select_checked_tickers_for_sample(
    panel: pd.DataFrame,
    sample: PitResolutionResult,
    *,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    max_examples: int,
) -> tuple[Optional[list[str]], Optional[str]]:
    """Return (tickers, na_reason). ``None`` tickers means rolling N/A."""
    resolved = sample.resolved_snapshot_date
    if resolved is None:
        return None, "rolling provenance not applicable: no resolved snapshot (before-first-snapshot)"

    month = normalize_date_column(panel, "month_date")
    snap = panel.loc[month == resolved]
    if snap.empty:
        return None, f"rolling provenance not applicable: resolved snapshot {resolved.date()} has no rows"

    classification = classify_snapshot_membership(
        snap,
        dvol_top_pct,
        spread_bottom_pct,
        all_panel_tickers=sorted(set(panel["ticker"].astype(str))),
    )

    selected = sorted(classification.selected)
    extras: list[str] = []
    for group in (
        classification.invalid_atm_pair,
        classification.missing_or_nonfinite_dvol,
        classification.missing_or_nonfinite_spread,
        classification.below_dvol_threshold,
        classification.below_spread_threshold,
        classification.new_or_insufficient_history,
        # Intentionally omit missing_from_snapshot: those tickers are absent from
        # the resolved snapshot and cannot be proven via rolling recomputation.
    ):
        for t in group:
            if t not in extras:
                extras.append(t)
    extras = sorted(extras)

    ordered: list[str] = []
    for t in selected:
        if t not in ordered:
            ordered.append(t)
    for t in extras:
        if t not in ordered:
            ordered.append(t)

    # Safety: if classification somehow yields nothing but snap has rows, use snap tickers.
    if not ordered:
        ordered = sorted(set(snap["ticker"].astype(str).tolist()))

    capped = ordered[: int(max_examples)]
    if not capped:
        return [], None
    return capped, None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_escape(value: object) -> str:
    text = "None" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _iso(value: object) -> str:
    if value is None:
        return "None"
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "None"
        return ts.normalize().date().isoformat()
    except Exception:  # noqa: BLE001
        return _md_escape(value)


def _render_check_details(
    details: Mapping[str, object], *, max_examples: int, indent: str = "  "
) -> list[str]:
    lines: list[str] = []
    for key in sorted(details.keys()):
        value = details[key]
        if isinstance(value, Mapping):
            capped = {k: value[k] for k in sorted(value.keys())}
            lines.append(f"{indent}- {_md_escape(key)}:")
            for sub_k in sorted(capped.keys()):
                lines.append(
                    f"{indent}  - {_md_escape(sub_k)}: `{_md_escape(capped[sub_k])}`"
                )
        elif isinstance(value, (list, tuple)):
            seq = list(value)
            if len(seq) > max_examples:
                seq = seq[:max_examples]
            lines.append(f"{indent}- {_md_escape(key)}: `{_md_escape(seq)}`")
        else:
            lines.append(f"{indent}- {_md_escape(key)}: `{_md_escape(value)}`")
    return lines


def _status_line(status: object) -> str:
    return _md_escape(status)


def render_markdown_report(
    report: PitUniverseAuditReport,
    *,
    panel_path: Path,
    weekly_path: Path,
    liquid_tickers_path: Path,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    strict: bool,
    sample_labels: Mapping[tuple[Optional[str], str], tuple[str, ...]],
    panel_row_count: int,
    weekly_row_count: int,
    liquid_ticker_count: int,
    rolling_na_notes: Sequence[str] = (),
    discovery_check: Optional[ArtifactCheckResult] = None,
    max_examples: int = 20,
) -> str:
    lines: list[str] = []
    lines.append("# C7 PIT Universe Audit")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- overall status: `{_status_line(report.overall_status)}`")
    lines.append(f"- strict mode: `{strict}`")
    lines.append("")

    # Scope
    lines.append("## Scope and parameters")
    lines.append("")
    lines.append(f"- panel path: `{_md_escape(panel_path)}`")
    lines.append(f"- weekly path: `{_md_escape(weekly_path)}`")
    lines.append(f"- liquid-tickers path: `{_md_escape(liquid_tickers_path)}`")
    lines.append(f"- requested dvol_top_pct: `{dvol_top_pct}`")
    lines.append(f"- requested spread_bottom_pct: `{spread_bottom_pct}`")
    lines.append(f"- sample count: `{len(report.samples)}`")
    lines.append(
        f"- rolling-provenance result count: `{len(report.rolling_provenance)}`"
    )
    lines.append("")

    # Inventory
    lines.append("## Artifact inventory")
    lines.append("")
    lines.append(f"- panel row count: `{panel_row_count}`")
    lines.append(f"- weekly row count: `{weekly_row_count}`")
    lines.append(f"- liquid ticker count: `{liquid_ticker_count}`")
    lines.append("")

    # Artifact checks
    lines.append("## Artifact checks")
    lines.append("")
    if not report.artifact_checks:
        lines.append("_No artifact checks recorded._")
    else:
        for c in report.artifact_checks:
            lines.append(
                f"- `{_md_escape(c.name)}`: `{_status_line(c.status)}` — {_md_escape(c.message)}"
            )
            if c.status in (FAIL, WARN) and c.details:
                lines.extend(_render_check_details(c.details, max_examples=max_examples))
    lines.append("")

    # Envelope
    lines.append("## Supported parameter envelope")
    lines.append("")
    env = report.artifact_envelope
    if env is None:
        lines.append("_Envelope evidence unavailable (blocking artifact failure or skipped)._")
    else:
        lines.append(f"- status: `{_status_line(env.status)}`")
        lines.append(f"- supported: `{env.supported}`")
        lines.append(f"- reason: {_md_escape(env.reason)}")
        lines.append(f"- requested dvol_top_pct: `{env.requested_dvol_top_pct}`")
        lines.append(f"- requested spread_bottom_pct: `{env.requested_spread_bottom_pct}`")
        lines.append(f"- stamped dvol_top_pct: `{env.superset_build_dvol_top_pct}`")
        lines.append(f"- stamped spread_bot_pct: `{env.superset_build_spread_bot_pct}`")
    lines.append("")

    # Discovery
    lines.append("## Sample discovery")
    lines.append("")
    disc = discovery_check
    if disc is None:
        # Fall back to artifact check named sample_discovery if present.
        for c in report.artifact_checks:
            if c.name == "sample_discovery":
                disc = c
                break
    if disc is None:
        lines.append("_Sample discovery not requested._")
    else:
        lines.append(f"- status: `{_status_line(disc.status)}`")
        lines.append(f"- message: {_md_escape(disc.message)}")
        details = disc.details or {}
        pairs = details.get("pairs", [])
        if pairs:
            for pair in pairs:
                lines.append(
                    f"- case labels={_md_escape(pair.get('labels'))} "
                    f"target={_md_escape(pair.get('target_snapshot_date'))} "
                    f"trade_date={_md_escape(pair.get('trade_date'))}"
                )
        notes = details.get("notes", [])
        for note in notes:
            lines.append(f"- note: {_md_escape(note)}")
        if disc.status in (FAIL, WARN) and details:
            lines.extend(_render_check_details(details, max_examples=max_examples))
    lines.append("")

    # PIT samples
    lines.append("## PIT sample results")
    lines.append("")
    if not report.samples:
        lines.append("_No evaluable PIT samples._")
    else:
        for sample in report.samples:
            key = (
                None
                if sample.target_snapshot_date is None
                else sample.target_snapshot_date.normalize().date().isoformat(),
                sample.trade_date.normalize().date().isoformat(),
            )
            labels = sample_labels.get(key, ())
            label_txt = ",".join(labels) if labels else "(explicit)"
            lines.append(f"### Sample `{_md_escape(label_txt)}`")
            lines.append("")
            lines.append(f"- label or labels: `{_md_escape(label_txt)}`")
            lines.append(f"- target_snapshot_date: `{_iso(sample.target_snapshot_date)}`")
            lines.append(f"- trade_date: `{_iso(sample.trade_date)}`")
            lines.append(f"- resolved_snapshot_date: `{_iso(sample.resolved_snapshot_date)}`")
            lines.append(f"- snapshot_lag_days: `{_md_escape(sample.snapshot_lag_days)}`")
            lines.append(f"- eligible_count: `{sample.eligible_count}`")
            lines.append(f"- selected_count: `{sample.selected_count}`")
            lines.append(f"- dvol_threshold: `{sample.dvol_threshold}`")
            lines.append(f"- spread_threshold: `{sample.spread_threshold}`")
            lines.append(f"- window_start_date: `{_iso(sample.window_start_date)}`")
            lines.append(f"- window_end_date: `{_iso(sample.window_end_date)}`")
            lines.append(f"- window_shortfall: `{_md_escape(sample.window_shortfall)}`")
            lines.append(f"- membership_hash: `{_md_escape(sample.membership_hash)}`")
            lines.append(
                f"- production_reference_match: `{sample.production_reference_match}`"
            )
            lines.append(f"- status: `{_status_line(sample.status)}`")
            notes = "; ".join(sample.notes) if sample.notes else "(none)"
            lines.append(f"- notes: {_md_escape(notes)}")
            excl = dict(sample.exclusions) if sample.exclusions else {}
            lines.append(f"- exclusion counts: `{_md_escape(excl)}`")
            lines.append(
                f"- mismatch tickers: `{_md_escape(list(sample.mismatch_tickers))}`"
            )
            lines.append("")

    # Rolling
    lines.append("## Rolling provenance")
    lines.append("")
    if rolling_na_notes:
        for note in rolling_na_notes:
            lines.append(f"- N/A: {_md_escape(note)}")
    if not report.rolling_provenance:
        if not rolling_na_notes:
            lines.append("_No rolling-provenance evidence produced._")
    else:
        for r in report.rolling_provenance:
            lines.append(f"### Rolling `{_iso(r.target_snapshot_date)}`")
            lines.append("")
            lines.append(f"- target snapshot: `{_iso(r.target_snapshot_date)}`")
            lines.append(f"- checked ticker count: `{len(r.tickers_checked)}`")
            capped = list(r.tickers_checked)
            lines.append(f"- checked tickers, capped: `{_md_escape(capped)}`")
            week_range = (
                f"{_iso(r.expected_week_ends[0])} .. {_iso(r.expected_week_ends[-1])}"
                if r.expected_week_ends
                else "(empty)"
            )
            lines.append(
                f"- expected week range/count: `{week_range}` / `{len(r.expected_week_ends)}`"
            )
            lines.append(
                f"- stored-panel recomputation match: `{r.recomputed_matches_panel}`"
            )
            lines.append(f"- future-invariance result: `{r.future_invariance_pass}`")
            lines.append(f"- field mismatch count: `{len(r.field_mismatches)}`")
            examples = [
                f"{m[0]}/{m[1]}" for m in r.field_mismatches[:20]
            ]
            lines.append(f"- field mismatch examples (capped): `{_md_escape(examples)}`")
            lines.append(f"- status: `{_status_line(r.status)}`")
            lines.append("")
    lines.append("")

    # Sample superset
    lines.append("## Sample superset coverage")
    lines.append("")
    if not report.sample_superset_coverage:
        lines.append("_No sample superset coverage evidence._")
    else:
        for i, cov in enumerate(report.sample_superset_coverage):
            lines.append(
                f"- sample[{i}]: status=`{_status_line(cov.status)}` "
                f"selected_count=`{cov.selected_count}` "
                f"missing=`{_md_escape(list(cov.missing_from_superset))}`"
            )
    lines.append("")

    # Full history
    lines.append("## Full-history superset coverage")
    lines.append("")
    fh = report.full_history_superset_coverage
    if fh is None:
        lines.append(
            "_Full-history superset coverage not run "
            "(unsupported envelope or blocked earlier)._"
        )
    else:
        lines.append(f"- status: `{_status_line(fh.status)}`")
        lines.append(f"- snapshots_checked: `{fh.snapshots_checked}`")
        lines.append(f"- unique_selected_tickers: `{fh.unique_selected_tickers}`")
        lines.append(f"- missing_ticker_count: `{fh.missing_ticker_count}`")
        lines.append(
            f"- sample_missing_tickers: `{_md_escape(list(fh.sample_missing_tickers))}`"
        )
        lines.append(f"- canonical_params: `{_md_escape(fh.canonical_params)}`")
    lines.append("")

    # Blocking / warnings
    lines.append("## Blocking failures")
    lines.append("")
    if not report.blocking_failures:
        lines.append("_None._")
    else:
        for item in report.blocking_failures:
            lines.append(f"- {_md_escape(item)}")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not report.warnings:
        lines.append("_None._")
    else:
        for item in report.warnings:
            lines.append(f"- {_md_escape(item)}")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Audit orchestration
# ---------------------------------------------------------------------------


def _sample_label_key(spec: SampleSpec) -> tuple[Optional[str], str]:
    target = (
        None
        if spec.target_snapshot_date is None
        else spec.target_snapshot_date.normalize().date().isoformat()
    )
    trade = spec.trade_date.normalize().date().isoformat()
    return target, trade


def _selected_tickers_from_sample(
    panel: pd.DataFrame,
    sample: PitResolutionResult,
    dvol_top_pct: float,
    spread_bottom_pct: float,
) -> list[str]:
    if sample.resolved_snapshot_date is None:
        return []
    month = normalize_date_column(panel, "month_date")
    snap = panel.loc[month == sample.resolved_snapshot_date]
    if snap.empty:
        return []
    classification = classify_snapshot_membership(snap, dvol_top_pct, spread_bottom_pct)
    return list(classification.selected)


def run_audit(
    *,
    panel: pd.DataFrame,
    weekly: pd.DataFrame,
    liquid_tickers: pd.DataFrame,
    liquid_set: set[str],
    sample_dates: Optional[Sequence[str]],
    discover_samples: bool,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    max_examples: int,
) -> tuple[
    PitUniverseAuditReport,
    dict[tuple[Optional[str], str], tuple[str, ...]],
    list[str],
    Optional[ArtifactCheckResult],
]:
    artifact_checks = _run_full_artifact_checks(panel, weekly, max_examples=max_examples)
    discovery_check: Optional[ArtifactCheckResult] = None
    sample_labels: dict[tuple[Optional[str], str], tuple[str, ...]] = {}
    rolling_na_notes: list[str] = []

    if _has_blocking_fail(artifact_checks):
        # Short-circuit expensive sample work; still assemble a FAIL report.
        report = assemble_audit_report(
            artifact_checks,
            None,
            [],
            [],
            [],
            None,
        )
        return report, sample_labels, rolling_na_notes, discovery_check

    # Envelope
    try:
        stamped_dvol, stamped_spread = read_superset_build_params(panel)
        envelope: Optional[ArtifactEnvelopeResult] = check_artifact_envelope(
            dvol_top_pct,
            spread_bottom_pct,
            stamped_dvol,
            stamped_spread,
        )
    except Exception as exc:  # noqa: BLE001
        envelope = ArtifactEnvelopeResult(
            requested_dvol_top_pct=float(dvol_top_pct),
            requested_spread_bottom_pct=float(spread_bottom_pct),
            superset_build_dvol_top_pct=float("nan"),
            superset_build_spread_bot_pct=float("nan"),
            supported=False,
            reason=f"cannot read stamped build params: {exc}",
            status=FAIL,
        )

    if envelope is not None and envelope.status == FAIL:
        artifact_checks = list(artifact_checks)
        report = assemble_audit_report(
            artifact_checks,
            envelope,
            [],
            [],
            [],
            None,
        )
        return report, sample_labels, rolling_na_notes, discovery_check

    # Resolve samples
    explicit_specs: list[SampleSpec] = []
    if sample_dates:
        for ts in _dedupe_explicit_dates(list(sample_dates)):
            explicit_specs.append(
                SampleSpec(
                    trade_date=ts,
                    target_snapshot_date=None,
                    labels=("explicit",),
                )
            )

    discovered_specs: list[SampleSpec] = []
    if discover_samples:
        discovered_specs, discovery_check = discover_audit_samples(panel, weekly)
        artifact_checks = list(artifact_checks) + [discovery_check]

    specs = _combine_sample_specs(explicit_specs, discovered_specs)
    for spec in specs:
        sample_labels[_sample_label_key(spec)] = spec.labels

    if not specs:
        no_samples = ArtifactCheckResult(
            name="evaluable_samples",
            status=FAIL,
            message="no evaluable samples after valid discovery/input loading",
            details={},
        )
        artifact_checks = list(artifact_checks) + [no_samples]
        report = assemble_audit_report(
            artifact_checks,
            envelope,
            [],
            [],
            [],
            None,
        )
        return report, sample_labels, rolling_na_notes, discovery_check

    # Evaluate samples
    samples: list[PitResolutionResult] = []
    for spec in specs:
        samples.append(
            evaluate_pit_sample(
                spec.trade_date,
                panel,
                dvol_top_pct,
                spread_bottom_pct,
                target_snapshot_date=spec.target_snapshot_date,
            )
        )

    # Rolling provenance
    rolling: list[RollingProvenanceResult] = []
    ticker_selection_fail = False
    for sample in samples:
        checked, na_reason = select_checked_tickers_for_sample(
            panel,
            sample,
            dvol_top_pct=dvol_top_pct,
            spread_bottom_pct=spread_bottom_pct,
            max_examples=max_examples,
        )
        if na_reason is not None:
            rolling_na_notes.append(
                f"trade_date={_iso(sample.trade_date)}: {na_reason}"
            )
            continue
        assert checked is not None
        if len(checked) == 0:
            ticker_selection_fail = True
            artifact_checks = list(artifact_checks) + [
                ArtifactCheckResult(
                    name="rolling_ticker_selection",
                    status=FAIL,
                    message=(
                        "resolved snapshot has rows but no checked ticker could be selected "
                        f"(trade_date={_iso(sample.trade_date)}, "
                        f"snapshot={_iso(sample.resolved_snapshot_date)})"
                    ),
                    details={
                        "trade_date": _iso(sample.trade_date),
                        "resolved_snapshot_date": _iso(sample.resolved_snapshot_date),
                    },
                )
            ]
            continue
        rolling.append(
            check_rolling_provenance(
                sample.resolved_snapshot_date,
                checked,
                weekly,
                panel,
                run_future_invariance=True,
            )
        )

    if not rolling and not ticker_selection_fail:
        # No substantive rolling evidence across the audit → blocking FAIL via assemble,
        # plus an explicit CLI check for clarity.
        artifact_checks = list(artifact_checks) + [
            ArtifactCheckResult(
                name="rolling_evidence_present",
                status=FAIL,
                message="no substantive rolling-provenance evidence across samples",
                details={"rolling_na_notes": list(rolling_na_notes)},
            )
        ]

    # Sample-level superset coverage
    coverage: list[SupersetCoverageResult] = []
    for sample in samples:
        selected = _selected_tickers_from_sample(
            panel, sample, dvol_top_pct, spread_bottom_pct
        )
        coverage.append(check_superset_coverage(selected, liquid_tickers))

    # Full-history (supported envelope only)
    full_history: Optional[FullHistorySupersetCoverageResult] = None
    if envelope is not None and envelope.supported:
        full_history = check_full_history_superset_coverage(
            panel,
            liquid_tickers,
            dvol_top_pct=dvol_top_pct,
            spread_bottom_pct=spread_bottom_pct,
        )

    report = assemble_audit_report(
        artifact_checks,
        envelope,
        samples,
        rolling,
        coverage,
        full_history,
    )
    return report, sample_labels, rolling_na_notes, discovery_check


def _assemble_runtime_validation_fail(
    exc: Exception,
    stage: str,
    *,
    prior_checks: Optional[Sequence[ArtifactCheckResult]] = None,
) -> PitUniverseAuditReport:
    runtime_check = ArtifactCheckResult(
        name="audit_runtime_validation",
        status=FAIL,
        message=(
            f"stage={stage}; {type(exc).__name__}: {exc}"
        ),
        details={
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    )
    checks = list(prior_checks or []) + [runtime_check]
    return assemble_audit_report(checks, None, [], [], [], None)


def _exit_code(overall: Status, *, strict: bool) -> int:
    if overall == FAIL:
        return 1
    if overall == WARN:
        return 1 if strict else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    panel: Optional[pd.DataFrame] = None
    weekly: Optional[pd.DataFrame] = None
    liquid_tickers: Optional[pd.DataFrame] = None
    liquid_set: set[str] = set()
    try:
        args = parse_args(argv)
        _validate_cli_args(args)
        validate_output_path_safety(
            args.output_report,
            (args.panel_path, args.weekly_path, args.liquid_tickers_path),
        )
        _ensure_output_parent_writable(args.output_report)
        panel, weekly, liquid_tickers, liquid_set = _load_artifacts(
            args.panel_path,
            args.weekly_path,
            args.liquid_tickers_path,
        )
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    assert panel is not None and weekly is not None and liquid_tickers is not None

    try:
        report, sample_labels, rolling_na_notes, discovery_check = run_audit(
            panel=panel,
            weekly=weekly,
            liquid_tickers=liquid_tickers,
            liquid_set=liquid_set,
            sample_dates=args.sample_dates,
            discover_samples=bool(args.discover_samples),
            dvol_top_pct=float(args.dvol_top_pct),
            spread_bottom_pct=float(args.spread_bottom_pct),
            max_examples=int(args.max_examples),
        )
    except _ARTIFACT_RUNTIME_EXCEPTIONS as exc:
        report = _assemble_runtime_validation_fail(exc, "run_audit")
        sample_labels = {}
        rolling_na_notes = ()
        discovery_check = None
    except InternalError as exc:
        print(f"error: internal error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: internal error: {exc}", file=sys.stderr)
        return 2
    else:
        try:
            md = render_markdown_report(
                report,
                panel_path=args.panel_path,
                weekly_path=args.weekly_path,
                liquid_tickers_path=args.liquid_tickers_path,
                dvol_top_pct=float(args.dvol_top_pct),
                spread_bottom_pct=float(args.spread_bottom_pct),
                strict=bool(args.strict),
                sample_labels=sample_labels,
                panel_row_count=int(len(panel)),
                weekly_row_count=int(len(weekly)),
                liquid_ticker_count=len(liquid_set),
                rolling_na_notes=rolling_na_notes,
                discovery_check=discovery_check,
                max_examples=int(args.max_examples),
            )
        except _ARTIFACT_RUNTIME_EXCEPTIONS as exc:
            report = _assemble_runtime_validation_fail(
                exc,
                "render_markdown_report",
                prior_checks=report.artifact_checks,
            )
        except InternalError as exc:
            print(f"error: internal error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"error: internal error: {exc}", file=sys.stderr)
            return 2
        else:
            try:
                write_report(args.output_report, md)
            except Exception as exc:  # noqa: BLE001
                print(f"error: failed to write report: {exc}", file=sys.stderr)
                return 2
            return _exit_code(report.overall_status, strict=bool(args.strict))

    # run_audit artifact failure path
    try:
        md = render_markdown_report(
            report,
            panel_path=args.panel_path,
            weekly_path=args.weekly_path,
            liquid_tickers_path=args.liquid_tickers_path,
            dvol_top_pct=float(args.dvol_top_pct),
            spread_bottom_pct=float(args.spread_bottom_pct),
            strict=bool(args.strict),
            sample_labels={},
            panel_row_count=int(len(panel)),
            weekly_row_count=int(len(weekly)),
            liquid_ticker_count=len(liquid_set),
            rolling_na_notes=(),
            discovery_check=None,
            max_examples=int(args.max_examples),
        )
    except _ARTIFACT_RUNTIME_EXCEPTIONS as exc:
        print(
            f"error: internal error during report render: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: internal error during report render: {exc}", file=sys.stderr)
        return 2
    try:
        write_report(args.output_report, md)
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to write report: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
