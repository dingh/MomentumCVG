"""Point-in-time (PIT) liquidity-universe audit module (Sprint 004 C7.2).

Pure, importable trust gate for the A3 rolling liquidity panel and the S1
universe membership it feeds. This module answers one question, independently
of the production S1 code path:

    Can we prove that the weekly trading universe is selected point-in-time
    from the accepted C4 rolling liquidity panel, without future-data leakage,
    stale-snapshot ambiguity, nondeterminism, or silent acceptance of missing
    liquidity?

Design references:
    docs/tmp/c7_0_pit_universe_reality_map.md
    docs/tmp/c7_1_pit_universe_design_memo.md

Scope guardrails (C7.2):
    * No CLI parsing, no file writes, no printing, no artifact mutation.
    * The only production S1 code touched is ``step1_get_universe`` (called at
      most once per comparison, as the production reference).
    * The independent reference reproduces S1 mechanics without calling S1.

Canonical snapshot policy (strict prior snapshot):
    global_snapshot_date = max(month_date where month_date < trade_date)

Date normalization contract:
    Plain date / datetime / ISO string / pandas Timestamp normalize to a
    timezone-naive ``pandas.Timestamp`` at date precision. Null required dates,
    unparseable non-null values, and timezone-aware values all FAIL. Timezone-
    aware values are rejected rather than converted, because UTC conversion can
    silently shift a date-only snapshot onto a different calendar day.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest.pipeline import step1_get_universe

# ---------------------------------------------------------------------------
# Status vocabulary + exceptions
# ---------------------------------------------------------------------------

Status = Literal["PASS", "WARN", "FAIL"]

PASS: Status = "PASS"
WARN: Status = "WARN"
FAIL: Status = "FAIL"


class PitAuditError(Exception):
    """Structural programmer/configuration error in the audit itself.

    Raised for misuse (missing arguments, impossible inputs). Evidence-level
    failures about the *artifact* are represented as ``ArtifactCheckResult``
    objects with ``status == "FAIL"`` — not raised.
    """


class ArtifactValidationError(PitAuditError):
    """An artifact is structurally invalid (bad dates, duplicate grain, mixed
    build parameters, ...).

    Low-level helpers raise this; the ``check_*`` wrappers catch it and return a
    ``FAIL`` :class:`ArtifactCheckResult` so evidence failures are represented
    consistently in the report.
    """


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

COL_MONTH_DATE = "month_date"
COL_TICKER = "ticker"
COL_DVOL = "atm_straddle_dollar_vol"
COL_SPREAD = "atm_spread_pct"
COL_HAS_VALID = "has_valid_atm_pair"

COL_WEEK_END = "week_end_date"
COL_WEEKLY_DVOL = "weekly_atm_straddle_dollar_vol"
COL_WEEKLY_SPREAD = "weekly_atm_spread_pct"
COL_WEEKLY_VALID = "weekly_has_valid_quote"

# Columns required for the basic S1 reference universe.
REFERENCE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {COL_MONTH_DATE, COL_TICKER, COL_DVOL, COL_SPREAD, COL_HAS_VALID}
)

# Columns required for rolling-provenance + artifact-envelope checks.
PROVENANCE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "lookback_weeks",
        "min_valid_quote_weeks",
        "dte_min",
        "dte_max",
        "dvol_top_pct",
        "spread_bot_pct",
        "window_start_date",
        "window_end_date",
        "window_shortfall",
        "valid_quote_weeks",
        "zero_volume_weeks",
    }
)

# Full artifact schema (superset of the two groups above).
FULL_REQUIRED_COLUMNS: frozenset[str] = REFERENCE_REQUIRED_COLUMNS | PROVENANCE_REQUIRED_COLUMNS

# Interpretation-defining build parameters that must be homogeneous in one file.
BUILD_PARAM_COLUMNS: Tuple[str, ...] = (
    "lookback_weeks",
    "min_valid_quote_weeks",
    "dte_min",
    "dte_max",
    "dvol_top_pct",
    "spread_bot_pct",
)
OPTIONAL_BUILD_PARAM_COLUMNS: Tuple[str, ...] = ("liquidity_source",)

# Numerical tolerances.
RANK_TOL = 1e-9
FLOAT_REL_TOL = 1e-9
_MAX_EXAMPLES = 20


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactCheckResult:
    name: str
    status: Status
    message: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactEnvelopeResult:
    requested_dvol_top_pct: float
    requested_spread_bottom_pct: float
    superset_build_dvol_top_pct: float
    superset_build_spread_bot_pct: float
    supported: bool
    reason: str
    status: Status


@dataclass(frozen=True)
class ReferenceUniverseResult:
    trade_date: pd.Timestamp
    resolved_snapshot_date: Optional[pd.Timestamp]
    snapshot_lag_days: Optional[int]
    dvol_top_pct: float
    spread_bottom_pct: float
    dvol_threshold: float
    spread_threshold: float
    eligible_count: int
    selected_count: int
    selected: pd.DataFrame  # [ticker, dvol_rank_pct, spread_rank_pct], sorted by ticker
    exclusions: Mapping[str, int]
    empty_reason: Optional[str]
    status: Status


@dataclass(frozen=True)
class UniverseComparisonResult:
    match: bool
    row_count_match: bool
    production_count: int
    reference_count: int
    production_only: Tuple[str, ...]
    reference_only: Tuple[str, ...]
    duplicate_production_tickers: Tuple[str, ...]
    rank_mismatches: Tuple[Tuple[str, str, float, float], ...]  # ticker, field, prod, ref
    status: Status


@dataclass(frozen=True)
class PitResolutionResult:
    trade_date: pd.Timestamp
    target_snapshot_date: Optional[pd.Timestamp]
    resolved_snapshot_date: Optional[pd.Timestamp]
    snapshot_lag_days: Optional[int]
    eligible_count: int
    selected_count: int
    dvol_threshold: float
    spread_threshold: float
    window_start_date: Optional[pd.Timestamp]
    window_end_date: Optional[pd.Timestamp]
    window_shortfall: Optional[int]
    exclusions: Mapping[str, int]
    membership_hash: str
    production_reference_match: bool
    mismatch_tickers: Tuple[str, ...]
    status: Status
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RollingProvenanceResult:
    target_snapshot_date: pd.Timestamp
    tickers_checked: Tuple[str, ...]
    expected_week_ends: Tuple[pd.Timestamp, ...]
    recomputed_matches_panel: bool
    field_mismatches: Tuple[Tuple[str, str, object, object], ...]  # ticker, field, expected, actual
    future_invariance_pass: bool
    status: Status


@dataclass(frozen=True)
class SupersetCoverageResult:
    selected_count: int
    missing_from_superset: Tuple[str, ...]
    status: Status


@dataclass(frozen=True)
class FullHistorySupersetCoverageResult:
    snapshots_checked: int
    unique_selected_tickers: int
    missing_ticker_count: int
    sample_missing_tickers: Tuple[str, ...]
    canonical_params: Tuple[float, float]
    status: Status


@dataclass(frozen=True)
class TickerClassification:
    invalid_atm_pair: Tuple[str, ...]
    missing_or_nonfinite_dvol: Tuple[str, ...]
    missing_or_nonfinite_spread: Tuple[str, ...]
    below_dvol_threshold: Tuple[str, ...]
    below_spread_threshold: Tuple[str, ...]
    new_or_insufficient_history: Tuple[str, ...]
    missing_from_snapshot: Tuple[str, ...]
    selected: Tuple[str, ...]

    def counts(self) -> Mapping[str, int]:
        return {
            "invalid_atm_pair": len(self.invalid_atm_pair),
            "missing_or_nonfinite_dvol": len(self.missing_or_nonfinite_dvol),
            "missing_or_nonfinite_spread": len(self.missing_or_nonfinite_spread),
            "below_dvol_threshold": len(self.below_dvol_threshold),
            "below_spread_threshold": len(self.below_spread_threshold),
            "new_or_insufficient_history": len(self.new_or_insufficient_history),
            "missing_from_snapshot": len(self.missing_from_snapshot),
            "selected": len(self.selected),
        }


@dataclass(frozen=True)
class PitUniverseAuditReport:
    artifact_checks: Tuple[ArtifactCheckResult, ...]
    artifact_envelope: Optional[ArtifactEnvelopeResult]
    samples: Tuple[PitResolutionResult, ...]
    rolling_provenance: Tuple[RollingProvenanceResult, ...]
    sample_superset_coverage: Tuple[SupersetCoverageResult, ...]
    full_history_superset_coverage: Optional[FullHistorySupersetCoverageResult]
    overall_status: Status
    blocking_failures: Tuple[str, ...]
    warnings: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Part 4 — Date normalization contract
# ---------------------------------------------------------------------------

def _is_tz_aware(value: object) -> bool:
    tz = getattr(value, "tzinfo", None)
    if tz is not None:
        return True
    # numpy datetime64 and naive timestamps have no tzinfo.
    return False


def normalize_date_value(value: object, *, label: str = "date") -> pd.Timestamp:
    """Normalize a single date-like value to a tz-naive Timestamp at date precision.

    Raises :class:`ArtifactValidationError` for null, unparseable, or
    timezone-aware inputs.
    """
    if value is None:
        raise ArtifactValidationError(f"{label}: null date is not allowed")
    # Reject NaT / NaN scalars.
    try:
        if not isinstance(value, str) and pd.isna(value):
            raise ArtifactValidationError(f"{label}: null/NaT date is not allowed")
    except (TypeError, ValueError):
        pass

    if _is_tz_aware(value):
        raise ArtifactValidationError(
            f"{label}: timezone-aware value {value!r} rejected "
            "(UTC conversion could shift the calendar day)"
        )

    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ArtifactValidationError(f"{label}: unparseable date {value!r} ({exc})")

    if ts is pd.NaT or pd.isna(ts):
        raise ArtifactValidationError(f"{label}: value {value!r} parsed to NaT")

    if ts.tzinfo is not None:
        raise ArtifactValidationError(
            f"{label}: timezone-aware value {value!r} rejected "
            "(UTC conversion could shift the calendar day)"
        )

    return ts.normalize()


def normalize_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Normalize an entire date column, preserving index.

    Raises :class:`ArtifactValidationError` if the column is missing or contains
    any null, unparseable, or timezone-aware value. Bad rows are never silently
    dropped.
    """
    if column not in frame.columns:
        raise ArtifactValidationError(f"required date column {column!r} is missing")

    # Fast path: a tz-aware pandas datetime column is rejected wholesale.
    dtype = frame[column].dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise ArtifactValidationError(
            f"{column!r}: timezone-aware column rejected "
            "(UTC conversion could shift the calendar day)"
        )

    values = [
        normalize_date_value(v, label=column) for v in frame[column].tolist()
    ]
    return pd.Series(values, index=frame.index, name=column)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return float(value)
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: object) -> bool:
    f = _to_float(value)
    return f is not None and math.isfinite(f)


def _floats_equal(a: object, b: object, *, rel_tol: float = FLOAT_REL_TOL) -> bool:
    fa = _to_float(a)
    fb = _to_float(b)
    a_nan = fa is None or math.isnan(fa)
    b_nan = fb is None or math.isnan(fb)
    if a_nan or b_nan:
        return a_nan and b_nan  # NaN == NaN only
    tol = rel_tol * max(1.0, abs(fa), abs(fb))
    return abs(fa - fb) <= tol


def _cap(items: Sequence[str], cap: int = _MAX_EXAMPLES) -> Tuple[str, ...]:
    return tuple(items[:cap])


def _aggregate_status(statuses: Iterable[Status]) -> Status:
    seen = set(statuses)
    if FAIL in seen:
        return FAIL
    if WARN in seen:
        return WARN
    return PASS


# ---------------------------------------------------------------------------
# Part 5 — Artifact validation
# ---------------------------------------------------------------------------

def check_required_columns(
    panel: pd.DataFrame,
    *,
    required: Iterable[str] = FULL_REQUIRED_COLUMNS,
    name: str = "required_columns",
) -> ArtifactCheckResult:
    required_set = set(required)
    missing = sorted(required_set - set(panel.columns))
    if missing:
        return ArtifactCheckResult(
            name=name,
            status=FAIL,
            message=f"missing required columns: {missing}",
            details={"missing": missing},
        )
    return ArtifactCheckResult(
        name=name,
        status=PASS,
        message="all required columns present",
        details={"checked": sorted(required_set)},
    )


def check_panel_grain(panel: pd.DataFrame) -> ArtifactCheckResult:
    """FAIL on any duplicate ``(month_date, ticker)`` across the full panel."""
    for col in (COL_MONTH_DATE, COL_TICKER):
        if col not in panel.columns:
            return ArtifactCheckResult(
                name="grain",
                status=FAIL,
                message=f"missing grain column {col!r}",
                details={"missing": col},
            )
    try:
        month = normalize_date_column(panel, COL_MONTH_DATE)
    except ArtifactValidationError as exc:
        return ArtifactCheckResult(
            name="grain", status=FAIL, message=str(exc), details={}
        )
    key = pd.DataFrame({COL_MONTH_DATE: month, COL_TICKER: panel[COL_TICKER].values})
    dup_mask = key.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        examples = (
            key[dup_mask]
            .drop_duplicates()
            .astype(str)
            .agg(" / ".join, axis=1)
            .tolist()
        )
        return ArtifactCheckResult(
            name="grain",
            status=FAIL,
            message=f"{n_dup} duplicate (month_date, ticker) rows",
            details={"duplicate_row_count": n_dup, "examples": _cap(examples)},
        )
    return ArtifactCheckResult(
        name="grain",
        status=PASS,
        message="no duplicate (month_date, ticker) rows",
        details={"rows": int(len(panel))},
    )


def check_ticker_validity(panel: pd.DataFrame) -> ArtifactCheckResult:
    """FAIL on null or empty (after string strip) ticker values."""
    if COL_TICKER not in panel.columns:
        return ArtifactCheckResult(
            name="ticker_validity",
            status=FAIL,
            message="missing ticker column",
            details={},
        )
    series = panel[COL_TICKER]
    null_count = int(series.isna().sum())
    stripped = series.dropna().astype(str).str.strip()
    empty_count = int((stripped == "").sum())
    if null_count or empty_count:
        return ArtifactCheckResult(
            name="ticker_validity",
            status=FAIL,
            message=f"{null_count} null and {empty_count} empty ticker values",
            details={"null_count": null_count, "empty_count": empty_count},
        )
    return ArtifactCheckResult(
        name="ticker_validity",
        status=PASS,
        message="all ticker values non-null and non-empty",
        details={},
    )


def check_build_param_homogeneity(panel: pd.DataFrame) -> ArtifactCheckResult:
    """FAIL if any interpretation-defining build parameter is mixed or null."""
    columns = list(BUILD_PARAM_COLUMNS)
    columns += [c for c in OPTIONAL_BUILD_PARAM_COLUMNS if c in panel.columns]

    problems: dict[str, object] = {}
    for col in columns:
        if col not in panel.columns:
            problems[col] = "missing"
            continue
        series = panel[col]
        if series.isna().any():
            problems[col] = "null_values"
            continue
        n_unique = series.nunique(dropna=False)
        if n_unique > 1:
            problems[col] = f"mixed:{sorted(map(str, series.unique()))[:5]}"

    if problems:
        return ArtifactCheckResult(
            name="build_param_homogeneity",
            status=FAIL,
            message=f"heterogeneous or missing build parameters: {sorted(problems)}",
            details=problems,
        )
    return ArtifactCheckResult(
        name="build_param_homogeneity",
        status=PASS,
        message="build parameters homogeneous",
        details={c: panel[c].iloc[0] for c in columns},
    )


def read_superset_build_params(panel: pd.DataFrame) -> Tuple[float, float]:
    """Read the homogeneous stamped superset build params from the panel.

    Returns ``(dvol_top_pct, spread_bot_pct)``. Raises
    :class:`ArtifactValidationError` if either column is missing, null, or mixed.
    """
    out: list[float] = []
    for col in ("dvol_top_pct", "spread_bot_pct"):
        if col not in panel.columns:
            raise ArtifactValidationError(f"panel missing build-param column {col!r}")
        series = panel[col]
        if series.isna().any():
            raise ArtifactValidationError(f"build-param column {col!r} has null values")
        uniques = series.unique()
        if len(uniques) != 1:
            raise ArtifactValidationError(
                f"build-param column {col!r} is not homogeneous: {sorted(map(str, uniques))[:5]}"
            )
        value = _to_float(uniques[0])
        if value is None:
            raise ArtifactValidationError(f"build-param column {col!r} is non-numeric")
        out.append(value)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Part 6 — Supported artifact parameter envelope
# ---------------------------------------------------------------------------

def check_artifact_envelope(
    requested_dvol_top_pct: float,
    requested_spread_bottom_pct: float,
    superset_build_dvol_top_pct: float,
    superset_build_spread_bot_pct: float,
) -> ArtifactEnvelopeResult:
    """Compare a requested S1 configuration against the panel's superset stamp."""
    req_dvol = _to_float(requested_dvol_top_pct)
    req_spread = _to_float(requested_spread_bottom_pct)
    sup_dvol = _to_float(superset_build_dvol_top_pct)
    sup_spread = _to_float(superset_build_spread_bot_pct)

    def _fail(reason: str) -> ArtifactEnvelopeResult:
        return ArtifactEnvelopeResult(
            requested_dvol_top_pct=req_dvol if req_dvol is not None else float("nan"),
            requested_spread_bottom_pct=req_spread if req_spread is not None else float("nan"),
            superset_build_dvol_top_pct=sup_dvol if sup_dvol is not None else float("nan"),
            superset_build_spread_bot_pct=sup_spread if sup_spread is not None else float("nan"),
            supported=False,
            reason=reason,
            status=FAIL,
        )

    # Validate the stamped superset params are themselves sensible.
    if sup_dvol is None or not (0.0 < sup_dvol <= 1.0):
        return _fail(f"superset dvol_top_pct out of bounds (0, 1]: {superset_build_dvol_top_pct!r}")
    if sup_spread is None or not (0.0 < sup_spread <= 1.0):
        return _fail(f"superset spread_bot_pct out of bounds (0, 1]: {superset_build_spread_bot_pct!r}")

    # Validate requested ranges.
    if req_dvol is None or not (0.0 < req_dvol <= 1.0):
        return _fail(f"requested dvol_top_pct out of range (0, 1]: {requested_dvol_top_pct!r}")
    if req_spread is None or not (0.0 < req_spread <= 1.0):
        return _fail(
            f"requested spread_bottom_pct out of range (0, 1]: {requested_spread_bottom_pct!r}"
        )

    # Blocking envelope rule: requested dvol must not exceed superset build dvol.
    if req_dvol > sup_dvol + RANK_TOL:
        return _fail(
            f"requested dvol_top_pct {req_dvol} exceeds superset build "
            f"dvol_top_pct {sup_dvol}; current artifacts cannot certify this universe"
        )
    if req_spread > 1.0 + RANK_TOL:
        return _fail(
            f"requested spread_bottom_pct {req_spread} exceeds 1.0; "
            "superset spread filter is fully open at 1.0"
        )

    return ArtifactEnvelopeResult(
        requested_dvol_top_pct=req_dvol,
        requested_spread_bottom_pct=req_spread,
        superset_build_dvol_top_pct=sup_dvol,
        superset_build_spread_bot_pct=sup_spread,
        supported=True,
        reason="requested configuration within supported superset envelope",
        status=PASS,
    )


# ---------------------------------------------------------------------------
# Part 7 — Independent reference universe
# ---------------------------------------------------------------------------

EMPTY_UNIVERSE_COLUMNS = ["ticker", "dvol_rank_pct", "spread_rank_pct"]


def _eligibility_masks(snap: pd.DataFrame) -> dict[str, pd.Series]:
    dvol = pd.to_numeric(snap[COL_DVOL], errors="coerce")
    spread = pd.to_numeric(snap[COL_SPREAD], errors="coerce")
    valid_pair = snap[COL_HAS_VALID] == True  # noqa: E712
    finite_dvol = np.isfinite(dvol)
    finite_spread = np.isfinite(spread)
    return {
        "valid_pair": valid_pair,
        "finite_dvol": finite_dvol,
        "finite_spread": finite_spread,
        "eligible": valid_pair & finite_dvol & finite_spread,
    }


def compute_reference_universe(
    trade_date: object,
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bottom_pct: float,
) -> ReferenceUniverseResult:
    """Independently reproduce S1 mechanics without calling ``step1_get_universe``."""
    missing = REFERENCE_REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ArtifactValidationError(
            f"reference universe requires columns {sorted(missing)} (missing)"
        )

    trade_ts = normalize_date_value(trade_date, label="trade_date")
    dvol_top = float(dvol_top_pct)
    spread_bottom = float(spread_bottom_pct)
    dvol_threshold = 1.0 - dvol_top
    spread_threshold = 1.0 - spread_bottom

    month = normalize_date_column(panel, COL_MONTH_DATE)
    prior = month[month < trade_ts]
    empty_selected = pd.DataFrame(columns=EMPTY_UNIVERSE_COLUMNS)

    if prior.empty:
        return ReferenceUniverseResult(
            trade_date=trade_ts,
            resolved_snapshot_date=None,
            snapshot_lag_days=None,
            dvol_top_pct=dvol_top,
            spread_bottom_pct=spread_bottom,
            dvol_threshold=dvol_threshold,
            spread_threshold=spread_threshold,
            eligible_count=0,
            selected_count=0,
            selected=empty_selected,
            exclusions={},
            empty_reason="before_first_snapshot",
            status=PASS,
        )

    resolved = prior.max()
    lag_days = int((trade_ts - resolved).days)

    snap = panel.loc[month == resolved].copy()

    # Reject duplicate snapshot/ticker grain at the resolved snapshot.
    if snap[COL_TICKER].duplicated().any():
        dups = sorted(snap.loc[snap[COL_TICKER].duplicated(keep=False), COL_TICKER].unique())
        raise ArtifactValidationError(
            f"duplicate ticker rows at snapshot {resolved.date()}: {dups[:5]}"
        )

    masks = _eligibility_masks(snap)
    valid_pair = masks["valid_pair"]
    finite_dvol = masks["finite_dvol"]
    finite_spread = masks["finite_spread"]
    eligible_mask = masks["eligible"]

    exclusions = {
        "invalid_atm_pair": int((~valid_pair).sum()),
        "missing_or_nonfinite_dvol": int((valid_pair & ~finite_dvol).sum()),
        "missing_or_nonfinite_spread": int((valid_pair & finite_dvol & ~finite_spread).sum()),
    }

    elig = snap.loc[eligible_mask].copy()
    if elig.empty:
        return ReferenceUniverseResult(
            trade_date=trade_ts,
            resolved_snapshot_date=resolved,
            snapshot_lag_days=lag_days,
            dvol_top_pct=dvol_top,
            spread_bottom_pct=spread_bottom,
            dvol_threshold=dvol_threshold,
            spread_threshold=spread_threshold,
            eligible_count=0,
            selected_count=0,
            selected=empty_selected,
            exclusions=exclusions,
            empty_reason="no_eligible_rows",
            status=PASS,
        )

    elig["dvol_rank_pct"] = pd.to_numeric(elig[COL_DVOL], errors="coerce").rank(
        ascending=True, method="average", pct=True
    )
    elig["spread_rank_pct"] = pd.to_numeric(elig[COL_SPREAD], errors="coerce").rank(
        ascending=False, method="average", pct=True
    )

    selected_mask = (elig["dvol_rank_pct"] >= dvol_threshold) & (
        elig["spread_rank_pct"] >= spread_threshold
    )
    selected = (
        elig.loc[selected_mask, ["ticker", "dvol_rank_pct", "spread_rank_pct"]]
        .sort_values("ticker", kind="mergesort")
        .reset_index(drop=True)
    )

    return ReferenceUniverseResult(
        trade_date=trade_ts,
        resolved_snapshot_date=resolved,
        snapshot_lag_days=lag_days,
        dvol_top_pct=dvol_top,
        spread_bottom_pct=spread_bottom,
        dvol_threshold=dvol_threshold,
        spread_threshold=spread_threshold,
        eligible_count=int(len(elig)),
        selected_count=int(len(selected)),
        selected=selected,
        exclusions=exclusions,
        empty_reason=None if len(selected) else "no_rows_pass_thresholds",
        status=PASS,
    )


# ---------------------------------------------------------------------------
# Part 8 — Compare production S1 with independent reference
# ---------------------------------------------------------------------------

def compare_universe_to_reference(
    trade_date: object,
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    *,
    step1_fn=step1_get_universe,
) -> UniverseComparisonResult:
    """Call production S1 once and compare to the independent reference."""
    reference = compute_reference_universe(trade_date, panel, dvol_top_pct, spread_bottom_pct)

    # step1_get_universe only reads ``config.dvol_top_pct`` / ``config.spread_bottom_pct``.
    params = SimpleNamespace(dvol_top_pct=dvol_top_pct, spread_bottom_pct=spread_bottom_pct)
    production = step1_fn(
        normalize_date_value(trade_date, label="trade_date").date(),
        panel,
        params,
    )

    prod = production.copy()
    prod_tickers = list(prod["ticker"])
    dup_prod = sorted({t for t in prod_tickers if prod_tickers.count(t) > 1})

    prod_sorted = prod.sort_values("ticker", kind="mergesort").reset_index(drop=True)
    ref_sorted = reference.selected.sort_values("ticker", kind="mergesort").reset_index(drop=True)

    prod_set = set(prod_sorted["ticker"])
    ref_set = set(ref_sorted["ticker"])
    production_only = tuple(sorted(prod_set - ref_set))
    reference_only = tuple(sorted(ref_set - prod_set))

    ref_lookup = {
        row.ticker: (float(row.dvol_rank_pct), float(row.spread_rank_pct))
        for row in ref_sorted.itertuples(index=False)
    }
    rank_mismatches: list[Tuple[str, str, float, float]] = []
    for row in prod_sorted.itertuples(index=False):
        if row.ticker not in ref_lookup:
            continue
        ref_dvol, ref_spread = ref_lookup[row.ticker]
        if abs(float(row.dvol_rank_pct) - ref_dvol) > RANK_TOL:
            rank_mismatches.append((row.ticker, "dvol_rank_pct", float(row.dvol_rank_pct), ref_dvol))
        if abs(float(row.spread_rank_pct) - ref_spread) > RANK_TOL:
            rank_mismatches.append(
                (row.ticker, "spread_rank_pct", float(row.spread_rank_pct), ref_spread)
            )

    row_count_match = len(prod_sorted) == len(ref_sorted)
    match = (
        not production_only
        and not reference_only
        and not dup_prod
        and row_count_match
        and not rank_mismatches
    )
    return UniverseComparisonResult(
        match=match,
        row_count_match=row_count_match,
        production_count=int(len(prod_sorted)),
        reference_count=int(len(ref_sorted)),
        production_only=production_only,
        reference_only=reference_only,
        duplicate_production_tickers=tuple(dup_prod),
        rank_mismatches=tuple(rank_mismatches),
        status=PASS if match else FAIL,
    )


# ---------------------------------------------------------------------------
# Part 9 — Deterministic membership hash
# ---------------------------------------------------------------------------

def _stable_num(value: object) -> str:
    """Full-precision, platform-independent numeric representation."""
    f = _to_float(value)
    if f is None or math.isnan(f):
        return "nan"
    return float(f).hex()


def _iso_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.date().isoformat()


def membership_hash_full(
    trade_date: object,
    resolved_snapshot_date: object,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    members: object,
) -> str:
    """Full SHA-256 hex digest of the canonical membership state."""
    member_records = _members_to_records(members)
    member_records.sort(key=lambda r: r["ticker"])
    payload = {
        "trade_date": _iso_or_none(trade_date),
        "resolved_snapshot_date": _iso_or_none(resolved_snapshot_date),
        "dvol_top_pct": _stable_num(dvol_top_pct),
        "spread_bottom_pct": _stable_num(spread_bottom_pct),
        "members": [
            {
                "ticker": r["ticker"],
                "dvol_rank_pct": _stable_num(r["dvol_rank_pct"]),
                "spread_rank_pct": _stable_num(r["spread_rank_pct"]),
            }
            for r in member_records
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def membership_hash(
    trade_date: object,
    resolved_snapshot_date: object,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    members: object,
) -> str:
    """First 16 hex characters of :func:`membership_hash_full` (for display)."""
    return membership_hash_full(
        trade_date, resolved_snapshot_date, dvol_top_pct, spread_bottom_pct, members
    )[:16]


def _members_to_records(members: object) -> list[dict]:
    if isinstance(members, pd.DataFrame):
        return [
            {
                "ticker": str(row.ticker),
                "dvol_rank_pct": row.dvol_rank_pct,
                "spread_rank_pct": row.spread_rank_pct,
            }
            for row in members.itertuples(index=False)
        ]
    records: list[dict] = []
    for item in members:
        if isinstance(item, Mapping):
            records.append(
                {
                    "ticker": str(item["ticker"]),
                    "dvol_rank_pct": item["dvol_rank_pct"],
                    "spread_rank_pct": item["spread_rank_pct"],
                }
            )
        else:
            ticker, dvol_rank, spread_rank = item
            records.append(
                {
                    "ticker": str(ticker),
                    "dvol_rank_pct": dvol_rank,
                    "spread_rank_pct": spread_rank,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Part 10 — Independent rolling-panel recomputation
# ---------------------------------------------------------------------------

def _global_week_ends(weekly_obs: pd.DataFrame) -> list[pd.Timestamp]:
    weeks = normalize_date_column(weekly_obs, COL_WEEK_END)
    return sorted(set(weeks.tolist()))


def _expected_window(
    all_week_ends: Sequence[pd.Timestamp],
    snapshot: pd.Timestamp,
    lookback_weeks: int,
) -> list[pd.Timestamp]:
    eligible = [w for w in all_week_ends if w <= snapshot]
    return eligible[-lookback_weeks:] if eligible else []


def recompute_rolling_snapshot(
    target_snapshot: object,
    checked_tickers: Iterable[str],
    weekly_obs: pd.DataFrame,
    lookback_weeks: int,
    min_valid_quote_weeks: int,
) -> pd.DataFrame:
    """Independently recompute the C4 rolling panel row for each checked ticker.

    Derives the expected global window using ``week_end_date <= target_snapshot``,
    so later weekly rows can never influence the result. Missing ticker-weeks are
    zero-filled for volume and treated as invalid quotes; the denominator is the
    configured ``lookback_weeks`` even during early-history shortfall.

    Returns one row per checked ticker (index = ticker) with columns matching the
    panel provenance fields.
    """
    for col in (COL_WEEK_END, COL_TICKER, COL_WEEKLY_DVOL, COL_WEEKLY_SPREAD, COL_WEEKLY_VALID):
        if col not in weekly_obs.columns:
            raise ArtifactValidationError(f"weekly observations missing column {col!r}")

    snapshot = normalize_date_value(target_snapshot, label="target_snapshot")
    lookback_weeks = int(lookback_weeks)
    min_valid_quote_weeks = int(min_valid_quote_weeks)

    week_series = normalize_date_column(weekly_obs, COL_WEEK_END)
    all_week_ends = sorted(set(week_series.tolist()))
    window = _expected_window(all_week_ends, snapshot, lookback_weeks)
    window_set = set(window)
    window_shortfall = max(0, lookback_weeks - len(window))
    window_start = window[0] if window else snapshot

    weekly = weekly_obs.copy()
    weekly["weeknorm"] = week_series.values
    in_window = weekly[weekly["weeknorm"].isin(window_set)]
    lookup: dict[tuple[pd.Timestamp, str], object] = {
        (row.weeknorm, row.ticker): row for row in in_window.itertuples(index=False)
    }

    records: list[dict] = []
    for ticker in checked_tickers:
        vol_sum = 0.0
        spreads: list[float] = []
        valid_weeks = 0
        zero_vol_weeks = 0
        for w in window:
            row = lookup.get((w, ticker))
            if row is None:
                zero_vol_weeks += 1
                continue
            wvol = _to_float(getattr(row, COL_WEEKLY_DVOL))
            finite_vol = wvol is not None and math.isfinite(wvol)
            vol_sum += wvol if finite_vol else 0.0
            if (not finite_vol) or wvol == 0:
                zero_vol_weeks += 1
            wspread = getattr(row, COL_WEEKLY_SPREAD)
            if bool(getattr(row, COL_WEEKLY_VALID)) and _is_finite(wspread):
                spreads.append(_to_float(wspread))
                valid_weeks += 1

        atm_dvol = vol_sum / lookback_weeks
        atm_spread = float(np.mean(spreads)) if spreads else float("nan")
        records.append(
            {
                "ticker": ticker,
                "atm_straddle_dollar_vol": atm_dvol,
                "atm_spread_pct": atm_spread,
                "valid_quote_weeks": valid_weeks,
                "zero_volume_weeks": zero_vol_weeks,
                "has_valid_atm_pair": valid_weeks >= min_valid_quote_weeks,
                "window_start_date": window_start,
                "window_end_date": snapshot,
                "window_shortfall": window_shortfall,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.set_index("ticker", drop=False)
    return result


_PROVENANCE_FLOAT_FIELDS = ("atm_straddle_dollar_vol", "atm_spread_pct")
_PROVENANCE_INT_FIELDS = ("valid_quote_weeks", "zero_volume_weeks", "window_shortfall")
_PROVENANCE_BOOL_FIELDS = ("has_valid_atm_pair",)
_PROVENANCE_DATE_FIELDS = ("window_start_date", "window_end_date")


def _compare_provenance_row(
    ticker: str,
    recomputed: pd.Series,
    stored: pd.Series,
) -> list[Tuple[str, str, object, object]]:
    mismatches: list[Tuple[str, str, object, object]] = []
    for f in _PROVENANCE_FLOAT_FIELDS:
        if not _floats_equal(recomputed[f], stored.get(f)):
            mismatches.append((ticker, f, recomputed[f], stored.get(f)))
    for f in _PROVENANCE_INT_FIELDS:
        if int(recomputed[f]) != int(stored.get(f)):
            mismatches.append((ticker, f, int(recomputed[f]), stored.get(f)))
    for f in _PROVENANCE_BOOL_FIELDS:
        if bool(recomputed[f]) != bool(stored.get(f)):
            mismatches.append((ticker, f, bool(recomputed[f]), stored.get(f)))
    for f in _PROVENANCE_DATE_FIELDS:
        exp = pd.Timestamp(recomputed[f]).normalize()
        act = normalize_date_value(stored.get(f), label=f)
        if exp != act:
            mismatches.append((ticker, f, exp, act))
    return mismatches


def check_rolling_provenance(
    target_snapshot: object,
    checked_tickers: Sequence[str],
    weekly_obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    lookback_weeks: Optional[int] = None,
    min_valid_quote_weeks: Optional[int] = None,
    run_future_invariance: bool = True,
) -> RollingProvenanceResult:
    """Recompute rolling metrics for ``checked_tickers`` and compare to the panel."""
    snapshot = normalize_date_value(target_snapshot, label="target_snapshot")

    if lookback_weeks is None:
        lookback_weeks = int(panel["lookback_weeks"].iloc[0])
    if min_valid_quote_weeks is None:
        min_valid_quote_weeks = int(panel["min_valid_quote_weeks"].iloc[0])

    recomputed = recompute_rolling_snapshot(
        snapshot, checked_tickers, weekly_obs, lookback_weeks, min_valid_quote_weeks
    )

    month = normalize_date_column(panel, COL_MONTH_DATE)
    snap_panel = panel.loc[month == snapshot].copy()
    snap_panel = snap_panel.set_index(COL_TICKER, drop=False)

    field_mismatches: list[Tuple[str, str, object, object]] = []
    for ticker in checked_tickers:
        if ticker not in snap_panel.index:
            field_mismatches.append((ticker, "missing_from_panel_snapshot", True, False))
            continue
        recomputed_row = recomputed.loc[ticker]
        stored_row = snap_panel.loc[ticker]
        field_mismatches.extend(_compare_provenance_row(ticker, recomputed_row, stored_row))

    future_ok = True
    if run_future_invariance:
        future_ok = check_future_invariance(
            snapshot, checked_tickers, weekly_obs, lookback_weeks, min_valid_quote_weeks
        )

    matches = not field_mismatches
    status: Status = PASS if (matches and future_ok) else FAIL
    return RollingProvenanceResult(
        target_snapshot_date=snapshot,
        tickers_checked=tuple(checked_tickers),
        expected_week_ends=tuple(
            _expected_window(_global_week_ends(weekly_obs), snapshot, lookback_weeks)
        ),
        recomputed_matches_panel=matches,
        field_mismatches=tuple(field_mismatches),
        future_invariance_pass=future_ok,
        status=status,
    )


# ---------------------------------------------------------------------------
# Part 11 — Future-invariance
# ---------------------------------------------------------------------------

def check_future_invariance(
    target_snapshot: object,
    checked_tickers: Sequence[str],
    weekly_obs: pd.DataFrame,
    lookback_weeks: int,
    min_valid_quote_weeks: int,
) -> bool:
    """Recomputing snapshot S must not change when future weekly rows are present.

    Recompute from the full weekly artifact and from a copy restricted to
    ``week_end_date <= S``; the two must be identical under the established
    tolerances.
    """
    snapshot = normalize_date_value(target_snapshot, label="target_snapshot")

    full = recompute_rolling_snapshot(
        snapshot, checked_tickers, weekly_obs, lookback_weeks, min_valid_quote_weeks
    )

    weeks = normalize_date_column(weekly_obs, COL_WEEK_END)
    restricted_obs = weekly_obs.loc[weeks <= snapshot].copy()
    restricted = recompute_rolling_snapshot(
        snapshot, checked_tickers, restricted_obs, lookback_weeks, min_valid_quote_weeks
    )

    for ticker in checked_tickers:
        rf = full.loc[ticker]
        rr = restricted.loc[ticker]
        if _compare_provenance_row(ticker, rf, rr):
            return False
    return True


# ---------------------------------------------------------------------------
# Part 12 — Superset coverage
# ---------------------------------------------------------------------------

def extract_liquid_ticker_set(source: object) -> set[str]:
    """Extract a set of liquid tickers from a DataFrame (``Ticker``/``ticker``)
    or any iterable of ticker strings. Values are stripped but not uppercased."""
    if isinstance(source, pd.DataFrame):
        if "Ticker" in source.columns:
            col = "Ticker"
        elif "ticker" in source.columns:
            col = "ticker"
        else:
            raise ArtifactValidationError(
                f"liquid tickers frame has no 'Ticker'/'ticker' column: {list(source.columns)}"
            )
        series = source[col].dropna().astype(str).str.strip()
        return {t for t in series if t != ""}
    return {str(t).strip() for t in source if str(t).strip() != ""}


def check_superset_coverage(
    selected_tickers: Iterable[str],
    liquid_tickers: object,
) -> SupersetCoverageResult:
    """FAIL when any selected ticker is absent from the precompute superset."""
    liquid = extract_liquid_ticker_set(liquid_tickers)
    selected = [str(t) for t in selected_tickers]
    missing = sorted({t for t in selected if t not in liquid})
    return SupersetCoverageResult(
        selected_count=len(selected),
        missing_from_superset=_cap(missing),
        status=FAIL if missing else PASS,
    )


def check_full_history_superset_coverage(
    panel: pd.DataFrame,
    liquid_tickers: object,
    *,
    dvol_top_pct: Optional[float] = None,
    spread_bottom_pct: float = 1.0,
) -> FullHistorySupersetCoverageResult:
    """Artifact-level coverage for the canonical supported configuration.

    For every panel snapshot ``S`` independently computes S1 membership directly
    on that snapshot (grouped/vectorized), then asserts every selected ticker is
    present in ``liquid_tickers``. Includes the terminal snapshot (no later trade
    date required).
    """
    if dvol_top_pct is None:
        dvol_top_pct, _stamped_spread = read_superset_build_params(panel)
    dvol_top = float(dvol_top_pct)
    spread_bottom = float(spread_bottom_pct)
    dvol_threshold = 1.0 - dvol_top
    spread_threshold = 1.0 - spread_bottom

    liquid = extract_liquid_ticker_set(liquid_tickers)

    work = panel.copy()
    work[COL_MONTH_DATE] = normalize_date_column(work, COL_MONTH_DATE).values
    dvol = pd.to_numeric(work[COL_DVOL], errors="coerce")
    spread = pd.to_numeric(work[COL_SPREAD], errors="coerce")
    eligible = (work[COL_HAS_VALID] == True) & np.isfinite(dvol) & np.isfinite(spread)  # noqa: E712
    elig = work.loc[eligible].copy()

    snapshots_checked = int(work[COL_MONTH_DATE].nunique())

    if elig.empty:
        return FullHistorySupersetCoverageResult(
            snapshots_checked=snapshots_checked,
            unique_selected_tickers=0,
            missing_ticker_count=0,
            sample_missing_tickers=(),
            canonical_params=(dvol_top, spread_bottom),
            status=PASS,
        )

    grouped = elig.groupby(COL_MONTH_DATE, sort=False)
    elig["dvol_rank_pct"] = grouped[COL_DVOL].rank(ascending=True, method="average", pct=True)
    elig["spread_rank_pct"] = grouped[COL_SPREAD].rank(ascending=False, method="average", pct=True)

    selected = elig.loc[
        (elig["dvol_rank_pct"] >= dvol_threshold) & (elig["spread_rank_pct"] >= spread_threshold)
    ]
    selected_tickers = set(selected[COL_TICKER].astype(str))
    missing = sorted(t for t in selected_tickers if t not in liquid)

    return FullHistorySupersetCoverageResult(
        snapshots_checked=snapshots_checked,
        unique_selected_tickers=len(selected_tickers),
        missing_ticker_count=len(missing),
        sample_missing_tickers=_cap(missing),
        canonical_params=(dvol_top, spread_bottom),
        status=FAIL if missing else PASS,
    )


# ---------------------------------------------------------------------------
# Part 13 — Missing / new ticker classification
# ---------------------------------------------------------------------------

def classify_snapshot_membership(
    snapshot_rows: pd.DataFrame,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    *,
    all_panel_tickers: Optional[Iterable[str]] = None,
    min_valid_quote_weeks: Optional[int] = None,
) -> TickerClassification:
    """Classify each ticker on one resolved snapshot into audit categories.

    A ticker being absent from the selected universe is NOT an error; the audit
    error is admitting an ineligible ticker or claiming a silent PASS without
    showing why rows were excluded. Production S1 output is unchanged.
    """
    snap = snapshot_rows.copy()
    masks = _eligibility_masks(snap)
    valid_pair = masks["valid_pair"]
    finite_dvol = masks["finite_dvol"]
    finite_spread = masks["finite_spread"]
    eligible_mask = masks["eligible"]

    def _tickers(mask: pd.Series) -> Tuple[str, ...]:
        return _cap(sorted(snap.loc[mask, COL_TICKER].astype(str).tolist()))

    invalid_atm_pair = _tickers(~valid_pair)
    missing_dvol = _tickers(valid_pair & ~finite_dvol)
    missing_spread = _tickers(valid_pair & finite_dvol & ~finite_spread)

    dvol_threshold = 1.0 - float(dvol_top_pct)
    spread_threshold = 1.0 - float(spread_bottom_pct)

    below_dvol: list[str] = []
    below_spread: list[str] = []
    selected: list[str] = []
    elig = snap.loc[eligible_mask].copy()
    if not elig.empty:
        elig["dvol_rank_pct"] = pd.to_numeric(elig[COL_DVOL], errors="coerce").rank(
            ascending=True, method="average", pct=True
        )
        elig["spread_rank_pct"] = pd.to_numeric(elig[COL_SPREAD], errors="coerce").rank(
            ascending=False, method="average", pct=True
        )
        for row in elig.itertuples(index=False):
            ticker = str(getattr(row, COL_TICKER))
            if row.dvol_rank_pct < dvol_threshold:
                below_dvol.append(ticker)
            elif row.spread_rank_pct < spread_threshold:
                below_spread.append(ticker)
            else:
                selected.append(ticker)

    new_or_insufficient: list[str] = []
    if min_valid_quote_weeks is None and "min_valid_quote_weeks" in snap.columns and len(snap):
        min_valid_quote_weeks = int(snap["min_valid_quote_weeks"].iloc[0])
    if min_valid_quote_weeks is not None and "valid_quote_weeks" in snap.columns:
        vqw = pd.to_numeric(snap["valid_quote_weeks"], errors="coerce")
        new_or_insufficient = sorted(
            snap.loc[vqw < min_valid_quote_weeks, COL_TICKER].astype(str).tolist()
        )

    missing_from_snapshot: Tuple[str, ...] = ()
    if all_panel_tickers is not None:
        present = set(snap[COL_TICKER].astype(str))
        absent = sorted(str(t) for t in set(map(str, all_panel_tickers)) - present)
        missing_from_snapshot = _cap(absent)

    return TickerClassification(
        invalid_atm_pair=invalid_atm_pair,
        missing_or_nonfinite_dvol=missing_dvol,
        missing_or_nonfinite_spread=missing_spread,
        below_dvol_threshold=_cap(sorted(below_dvol)),
        below_spread_threshold=_cap(sorted(below_spread)),
        new_or_insufficient_history=_cap(tuple(new_or_insufficient)),
        missing_from_snapshot=missing_from_snapshot,
        selected=_cap(sorted(selected)),
    )


# ---------------------------------------------------------------------------
# PIT sample evaluation (ties reference + comparison + hash + window metadata)
# ---------------------------------------------------------------------------

def evaluate_pit_sample(
    trade_date: object,
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    *,
    target_snapshot_date: object = None,
    step1_fn=step1_get_universe,
) -> PitResolutionResult:
    """Evaluate a single trade-date sample end to end (no I/O).

    Resolves the strict prior snapshot, builds the independent reference,
    compares production S1 to it, computes a deterministic membership hash, and
    reads window metadata from the panel snapshot.
    """
    reference = compute_reference_universe(trade_date, panel, dvol_top_pct, spread_bottom_pct)
    comparison = compare_universe_to_reference(
        trade_date, panel, dvol_top_pct, spread_bottom_pct, step1_fn=step1_fn
    )

    resolved = reference.resolved_snapshot_date
    target_ts = (
        normalize_date_value(target_snapshot_date, label="target_snapshot_date")
        if target_snapshot_date is not None
        else None
    )

    notes: list[str] = []
    status: Status = PASS

    # Strict prior-snapshot invariant.
    if resolved is not None and resolved >= reference.trade_date:
        status = FAIL
        notes.append("resolved_snapshot_date >= trade_date (same-day/future prohibited)")
    if target_ts is not None and resolved is not None and target_ts != resolved:
        status = FAIL
        notes.append(f"target_snapshot_date {target_ts.date()} != resolved {resolved.date()}")

    if not comparison.match:
        status = FAIL
        notes.append("production S1 differs from independent reference")

    # Window metadata from the panel snapshot (if present).
    window_start = window_end = None
    window_shortfall = None
    if resolved is not None and "window_end_date" in panel.columns:
        month = normalize_date_column(panel, COL_MONTH_DATE)
        snap_rows = panel.loc[month == resolved]
        if not snap_rows.empty:
            if "window_start_date" in snap_rows.columns:
                window_start = normalize_date_value(
                    snap_rows["window_start_date"].iloc[0], label="window_start_date"
                )
            window_end = normalize_date_value(
                snap_rows["window_end_date"].iloc[0], label="window_end_date"
            )
            if "window_shortfall" in snap_rows.columns:
                window_shortfall = int(snap_rows["window_shortfall"].iloc[0])
                if window_shortfall > 0 and status != FAIL:
                    status = WARN
                    notes.append("window_shortfall > 0 (early-history WARN)")

    mhash = membership_hash(
        reference.trade_date,
        resolved,
        dvol_top_pct,
        spread_bottom_pct,
        reference.selected,
    )

    mismatch = tuple(sorted(set(comparison.production_only) | set(comparison.reference_only)))

    return PitResolutionResult(
        trade_date=reference.trade_date,
        target_snapshot_date=target_ts,
        resolved_snapshot_date=resolved,
        snapshot_lag_days=reference.snapshot_lag_days,
        eligible_count=reference.eligible_count,
        selected_count=reference.selected_count,
        dvol_threshold=reference.dvol_threshold,
        spread_threshold=reference.spread_threshold,
        window_start_date=window_start,
        window_end_date=window_end,
        window_shortfall=window_shortfall,
        exclusions=reference.exclusions,
        membership_hash=mhash,
        production_reference_match=comparison.match,
        mismatch_tickers=mismatch,
        status=status,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Report assembly (pure — no I/O)
# ---------------------------------------------------------------------------

def assemble_audit_report(
    artifact_checks: Sequence[ArtifactCheckResult],
    artifact_envelope: Optional[ArtifactEnvelopeResult],
    samples: Sequence[PitResolutionResult],
    rolling_provenance: Sequence[RollingProvenanceResult],
    sample_superset_coverage: Sequence[SupersetCoverageResult],
    full_history_superset_coverage: Optional[FullHistorySupersetCoverageResult],
) -> PitUniverseAuditReport:
    """Package audit results and compute overall status (pure, no I/O)."""
    all_statuses: list[Status] = [c.status for c in artifact_checks]
    if artifact_envelope is not None:
        all_statuses.append(artifact_envelope.status)
    all_statuses.extend(s.status for s in samples)
    all_statuses.extend(r.status for r in rolling_provenance)
    all_statuses.extend(c.status for c in sample_superset_coverage)
    if full_history_superset_coverage is not None:
        all_statuses.append(full_history_superset_coverage.status)

    blocking: list[str] = []
    warnings: list[str] = []
    for c in artifact_checks:
        if c.status == FAIL:
            blocking.append(f"artifact:{c.name}: {c.message}")
        elif c.status == WARN:
            warnings.append(f"artifact:{c.name}: {c.message}")
    if artifact_envelope is not None and artifact_envelope.status == FAIL:
        blocking.append(f"envelope: {artifact_envelope.reason}")
    for s in samples:
        if s.status == FAIL:
            blocking.append(f"sample:{_iso_or_none(s.trade_date)}: {'; '.join(s.notes)}")
        elif s.status == WARN:
            warnings.append(f"sample:{_iso_or_none(s.trade_date)}: {'; '.join(s.notes)}")
    for r in rolling_provenance:
        if r.status == FAIL:
            blocking.append(f"rolling:{_iso_or_none(r.target_snapshot_date)}")
    for c in sample_superset_coverage:
        if c.status == FAIL:
            blocking.append(f"superset_sample: missing {list(c.missing_from_superset)}")
    if full_history_superset_coverage is not None and full_history_superset_coverage.status == FAIL:
        blocking.append(
            f"superset_full_history: {full_history_superset_coverage.missing_ticker_count} missing"
        )

    return PitUniverseAuditReport(
        artifact_checks=tuple(artifact_checks),
        artifact_envelope=artifact_envelope,
        samples=tuple(samples),
        rolling_provenance=tuple(rolling_provenance),
        sample_superset_coverage=tuple(sample_superset_coverage),
        full_history_superset_coverage=full_history_superset_coverage,
        overall_status=_aggregate_status(all_statuses),
        blocking_failures=tuple(blocking),
        warnings=tuple(warnings),
    )
