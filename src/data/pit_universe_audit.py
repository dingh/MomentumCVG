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
    invalid_production_rank_tickers: Tuple[str, ...] = ()
    comparison_errors: Tuple[str, ...] = ()


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
    selected_tickers: Tuple[str, ...] = ()


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

    col = frame[column]
    dtype = col.dtype

    # A tz-aware pandas datetime column is rejected wholesale.
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise ArtifactValidationError(
            f"{column!r}: timezone-aware column rejected "
            "(UTC conversion could shift the calendar day)"
        )

    # Fast path: an already tz-naive datetime64 column (any resolution) normalizes
    # with one vectorized op instead of millions of Python-level conversions.
    if pd.api.types.is_datetime64_dtype(dtype):
        if col.isna().any():
            raise ArtifactValidationError(f"{column!r}: null/NaT date is not allowed")
        return col.dt.normalize().rename(column)

    # Object / heterogeneous fallback: strict per-value normalization.
    values = [normalize_date_value(v, label=column) for v in col.tolist()]
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


def compare_numeric_values(
    actual: object,
    expected: object,
    *,
    allow_both_nan: bool,
    rel_tol: float,
    abs_tol: float = 0.0,
) -> bool:
    """Single strict numeric comparison contract for the whole audit.

    Semantics (see C7.2A Fix 1):

    * Both values finite:
        ``abs(a - b) <= max(abs_tol, rel_tol * max(1, abs(a), abs(b)))``.
    * Both NaN (or null / unparseable → treated as NaN):
        equal only when ``allow_both_nan`` is True.
    * One-sided NaN: mismatch.
    * Infinity (either side, +inf or -inf): never an ordinary valid match —
      always returns ``False``. Infinity must be rejected by artifact validation
      *before* comparison; here it can only surface as a mismatch, never a PASS.

    The tolerance term ``rel_tol * max(1, |a|, |b|)`` can only be evaluated once
    both sides are known finite, so it can never blow up to infinity.
    """
    fa = _to_float(actual)
    fb = _to_float(expected)
    fa = float("nan") if fa is None else fa
    fb = float("nan") if fb is None else fb

    a_nan = math.isnan(fa)
    b_nan = math.isnan(fb)
    if a_nan or b_nan:
        return bool(allow_both_nan and a_nan and b_nan)

    # Infinity is never treated as an ordinary valid match.
    if math.isinf(fa) or math.isinf(fb):
        return False

    tol = max(abs_tol, rel_tol * max(1.0, abs(fa), abs(fb)))
    return abs(fa - fb) <= tol


def _floats_equal(a: object, b: object, *, rel_tol: float = FLOAT_REL_TOL) -> bool:
    """Provenance/value equality where a legitimately-missing value is NaN.

    Both-NaN is treated as equal (missing spread is an expected valid state);
    infinity is always a mismatch.
    """
    return compare_numeric_values(a, b, allow_both_nan=True, rel_tol=rel_tol)


def _cap(items: Sequence[str], cap: int = _MAX_EXAMPLES) -> Tuple[str, ...]:
    return tuple(items[:cap])


def _aggregate_status(statuses: Iterable[Status]) -> Status:
    """Aggregate statuses. An empty input is FAIL — no evidence is not PASS."""
    seen = set(statuses)
    if not seen:
        return FAIL
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


def check_panel_metric_integrity(panel: pd.DataFrame) -> ArtifactCheckResult:
    """Layer A — artifact-value integrity for panel metric columns.

    This is *separate* from S1 parity (Layer B). It rejects corrupt artifact
    values regardless of what production S1 would do with them:

    * ``atm_straddle_dollar_vol`` / ``atm_spread_pct`` must be true numeric
      values. Non-numeric object/text is rejected (not silently coerced to NaN
      and treated as ordinary missing liquidity).
    * NaN is permitted only as the contract's missing-liquidity marker.
    * Infinity always FAILs.
    * Negative values FAIL (dollar volume and spread are non-negative).
    """
    problems: dict[str, object] = {}
    for col in (COL_DVOL, COL_SPREAD):
        if col not in panel.columns:
            problems[col] = "missing"
            continue
        series = panel[col]
        numeric = pd.to_numeric(series, errors="coerce")
        # Non-null originals that fail numeric coercion are corrupt (text).
        non_numeric = int((series.notna() & numeric.isna()).sum())
        if non_numeric:
            problems[col] = f"non_numeric_values:{non_numeric}"
            continue
        finite = numeric[np.isfinite(numeric)]
        n_inf = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
        if n_inf:
            problems[col] = f"infinite_values:{n_inf}"
            continue
        n_negative = int((finite < 0).sum())
        if n_negative:
            problems[col] = f"negative_values:{n_negative}"

    if problems:
        return ArtifactCheckResult(
            name="panel_metric_integrity",
            status=FAIL,
            message=f"invalid panel metric values: {sorted(problems)}",
            details=problems,
        )
    return ArtifactCheckResult(
        name="panel_metric_integrity",
        status=PASS,
        message="panel metric columns numeric, finite-or-null, non-negative",
        details={},
    )


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
    """Eligibility masks.

    ``eligible`` reproduces production S1 exactly (Layer B parity):
    ``has_valid_atm_pair == True AND dvol.notna() AND spread.notna()`` — the
    same predicate as ``step1_get_universe``. Artifact-value integrity (rejecting
    infinity / non-numeric text) is handled separately by
    :func:`check_panel_metric_integrity` (Layer A). ``finite_*`` masks are
    retained for audit-layer classification/reporting only.
    """
    dvol = pd.to_numeric(snap[COL_DVOL], errors="coerce")
    spread = pd.to_numeric(snap[COL_SPREAD], errors="coerce")
    valid_pair = snap[COL_HAS_VALID] == True  # noqa: E712
    notna_dvol = snap[COL_DVOL].notna()
    notna_spread = snap[COL_SPREAD].notna()
    finite_dvol = np.isfinite(dvol)
    finite_spread = np.isfinite(spread)
    return {
        "valid_pair": valid_pair,
        "notna_dvol": notna_dvol,
        "notna_spread": notna_spread,
        "finite_dvol": finite_dvol,
        "finite_spread": finite_spread,
        # Layer B parity with production S1 (.notna(), not isfinite).
        "eligible": valid_pair & notna_dvol & notna_spread,
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
    notna_dvol = masks["notna_dvol"]
    notna_spread = masks["notna_spread"]
    eligible_mask = masks["eligible"]

    # Exclusion buckets mirror the S1 (.notna) eligibility predicate exactly,
    # so counts reconcile with parity. Infinite values are caught upstream by
    # check_panel_metric_integrity (Layer A).
    exclusions = {
        "invalid_atm_pair": int((~valid_pair).sum()),
        "missing_or_nonfinite_dvol": int((valid_pair & ~notna_dvol).sum()),
        "missing_or_nonfinite_spread": int((valid_pair & notna_dvol & ~notna_spread).sum()),
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

PRODUCTION_OUTPUT_COLUMNS = ("ticker", "dvol_rank_pct", "spread_rank_pct")


def _comparison_fail(
    reference: ReferenceUniverseResult,
    production: object,
    errors: Sequence[str],
) -> UniverseComparisonResult:
    prod_count = int(len(production)) if isinstance(production, pd.DataFrame) else 0
    return UniverseComparisonResult(
        match=False,
        row_count_match=False,
        production_count=prod_count,
        reference_count=int(len(reference.selected)),
        production_only=(),
        reference_only=(),
        duplicate_production_tickers=(),
        rank_mismatches=(),
        status=FAIL,
        invalid_production_rank_tickers=(),
        comparison_errors=tuple(errors),
    )


def compare_production_output_to_reference(
    production: object,
    reference: ReferenceUniverseResult,
) -> UniverseComparisonResult:
    """Pure comparison of a production S1 output frame to the independent
    reference. Does not call S1 and does not recompute the reference.

    Malformed production output (not a DataFrame, missing required columns,
    non-finite ranks) is represented as an audit FAIL with explicit
    ``comparison_errors`` / mismatch records — never an opaque pandas exception.
    """
    if not isinstance(production, pd.DataFrame):
        return _comparison_fail(
            reference, production, [f"production output is not a DataFrame: {type(production)!r}"]
        )

    missing_cols = [c for c in PRODUCTION_OUTPUT_COLUMNS if c not in production.columns]
    if missing_cols:
        return _comparison_fail(
            reference, production, [f"production output missing columns: {missing_cols}"]
        )

    prod = production.copy()
    prod_tickers = [str(t) for t in prod["ticker"].tolist()]
    dup_prod = sorted({t for t in prod_tickers if prod_tickers.count(t) > 1})

    # Non-finite (NaN / inf) production ranks are invalid regardless of parity.
    invalid_rank: list[str] = []
    for row in prod.itertuples(index=False):
        if not (_is_finite(row.dvol_rank_pct) and _is_finite(row.spread_rank_pct)):
            invalid_rank.append(str(row.ticker))
    invalid_rank = sorted(set(invalid_rank))

    prod_sorted = prod.sort_values("ticker", kind="mergesort").reset_index(drop=True)
    ref_sorted = reference.selected.sort_values("ticker", kind="mergesort").reset_index(drop=True)

    prod_set = set(str(t) for t in prod_sorted["ticker"])
    ref_set = set(str(t) for t in ref_sorted["ticker"])
    production_only = tuple(sorted(prod_set - ref_set))
    reference_only = tuple(sorted(ref_set - prod_set))

    ref_lookup = {
        str(row.ticker): (row.dvol_rank_pct, row.spread_rank_pct)
        for row in ref_sorted.itertuples(index=False)
    }
    rank_mismatches: list[Tuple[str, str, float, float]] = []
    for row in prod_sorted.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker not in ref_lookup:
            continue
        ref_dvol, ref_spread = ref_lookup[ticker]
        # Ranks must be finite and equal within abs tolerance; NaN/inf → mismatch.
        if not compare_numeric_values(
            row.dvol_rank_pct, ref_dvol, allow_both_nan=False, rel_tol=0.0, abs_tol=RANK_TOL
        ):
            rank_mismatches.append(
                (ticker, "dvol_rank_pct", _to_float(row.dvol_rank_pct), _to_float(ref_dvol))
            )
        if not compare_numeric_values(
            row.spread_rank_pct, ref_spread, allow_both_nan=False, rel_tol=0.0, abs_tol=RANK_TOL
        ):
            rank_mismatches.append(
                (ticker, "spread_rank_pct", _to_float(row.spread_rank_pct), _to_float(ref_spread))
            )

    row_count_match = len(prod_sorted) == len(ref_sorted)
    match = (
        not production_only
        and not reference_only
        and not dup_prod
        and not invalid_rank
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
        invalid_production_rank_tickers=tuple(invalid_rank),
        comparison_errors=(),
    )


def compare_universe_to_reference(
    trade_date: object,
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bottom_pct: float,
    *,
    step1_fn=step1_get_universe,
    reference: Optional[ReferenceUniverseResult] = None,
) -> UniverseComparisonResult:
    """Call production S1 **exactly once** and compare to the independent reference.

    If ``reference`` is supplied it is reused (the reference is not recomputed),
    so a caller such as :func:`evaluate_pit_sample` computes the reference once
    and S1 once per sample.
    """
    if reference is None:
        reference = compute_reference_universe(trade_date, panel, dvol_top_pct, spread_bottom_pct)

    # step1_get_universe only reads ``config.dvol_top_pct`` / ``config.spread_bottom_pct``.
    params = SimpleNamespace(dvol_top_pct=dvol_top_pct, spread_bottom_pct=spread_bottom_pct)
    production = step1_fn(
        normalize_date_value(trade_date, label="trade_date").date(),
        panel,
        params,
    )
    return compare_production_output_to_reference(production, reference)


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
# Fix 3 — Weekly artifact validation
# ---------------------------------------------------------------------------

WEEKLY_REQUIRED_COLUMNS: Tuple[str, ...] = (
    COL_WEEK_END,
    COL_TICKER,
    COL_WEEKLY_DVOL,
    COL_WEEKLY_SPREAD,
    COL_WEEKLY_VALID,
)

_PREPARED_WEEK_COL = "weeknorm"


def _is_bool_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def check_weekly_artifact(weekly_obs: pd.DataFrame) -> Tuple[ArtifactCheckResult, ...]:
    """Validate the full weekly observations artifact before rolling recompute.

    Returns granular :class:`ArtifactCheckResult` records. Any FAIL means the
    weekly artifact must not be consumed by :func:`recompute_rolling_snapshot`.
    """
    missing = [c for c in WEEKLY_REQUIRED_COLUMNS if c not in weekly_obs.columns]
    if missing:
        return (
            ArtifactCheckResult(
                name="weekly_required_columns",
                status=FAIL,
                message=f"missing required weekly columns: {missing}",
                details={"missing": missing},
            ),
        )

    results: list[ArtifactCheckResult] = [
        ArtifactCheckResult(
            name="weekly_required_columns",
            status=PASS,
            message="all required weekly columns present",
            details={},
        )
    ]

    # --- Grain: no duplicate (week_end_date, ticker) ---
    try:
        weeknorm = normalize_date_column(weekly_obs, COL_WEEK_END)
        key = pd.DataFrame(
            {COL_WEEK_END: weeknorm.values, COL_TICKER: weekly_obs[COL_TICKER].astype(str).values}
        )
        dup_mask = key.duplicated(keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            examples = (
                key[dup_mask].drop_duplicates().astype(str).agg(" / ".join, axis=1).tolist()
            )
            results.append(
                ArtifactCheckResult(
                    name="weekly_grain",
                    status=FAIL,
                    message=f"{n_dup} duplicate (week_end_date, ticker) rows",
                    details={"duplicate_row_count": n_dup, "examples": _cap(examples)},
                )
            )
        else:
            results.append(
                ArtifactCheckResult(
                    name="weekly_grain", status=PASS, message="no duplicate weekly grain", details={}
                )
            )
    except ArtifactValidationError as exc:
        results.append(
            ArtifactCheckResult(name="weekly_grain", status=FAIL, message=str(exc), details={})
        )

    # --- Ticker validity ---
    tick = weekly_obs[COL_TICKER]
    null_tick = int(tick.isna().sum())
    empty_tick = int((tick.dropna().astype(str).str.strip() == "").sum())
    results.append(
        ArtifactCheckResult(
            name="weekly_ticker_validity",
            status=FAIL if (null_tick or empty_tick) else PASS,
            message=f"{null_tick} null and {empty_tick} empty weekly ticker values",
            details={"null_count": null_tick, "empty_count": empty_tick},
        )
    )

    # --- Boolean domain for weekly_has_valid_quote ---
    non_bool = int(sum(1 for v in weekly_obs[COL_WEEKLY_VALID].tolist() if not _is_bool_scalar(v)))
    results.append(
        ArtifactCheckResult(
            name="weekly_has_valid_quote_domain",
            status=FAIL if non_bool else PASS,
            message=(
                f"{non_bool} non-boolean weekly_has_valid_quote values"
                if non_bool
                else "weekly_has_valid_quote is boolean"
            ),
            details={"non_boolean_count": non_bool},
        )
    )

    # --- Volume validity (existing rows must be numeric, finite, non-negative) ---
    vol = weekly_obs[COL_WEEKLY_DVOL]
    vol_num = pd.to_numeric(vol, errors="coerce")
    vol_non_numeric = int((vol.notna() & vol_num.isna()).sum())
    vol_arr = vol_num.to_numpy(dtype="float64", na_value=np.nan)
    vol_nan = int(np.isnan(vol_arr).sum())  # includes non-numeric coerced-to-NaN
    vol_inf = int(np.isinf(vol_arr).sum())
    vol_negative = int((vol_num[np.isfinite(vol_num)] < 0).sum())
    vol_problems = (vol_non_numeric > 0) or (vol_nan > 0) or (vol_inf > 0) or (vol_negative > 0)
    results.append(
        ArtifactCheckResult(
            name="weekly_volume_validity",
            status=FAIL if vol_problems else PASS,
            message=(
                "weekly volume must be numeric, finite, non-negative"
                if vol_problems
                else "weekly volume numeric, finite, non-negative"
            ),
            details={
                "non_numeric": vol_non_numeric,
                "nan_or_missing": vol_nan,
                "infinite": vol_inf,
                "negative": vol_negative,
            },
        )
    )

    # --- Spread consistency vs valid flag ---
    spread = weekly_obs[COL_WEEKLY_SPREAD]
    spread_num = pd.to_numeric(spread, errors="coerce")
    spread_non_numeric = int((spread.notna() & spread_num.isna()).sum())
    spread_arr = spread_num.to_numpy(dtype="float64", na_value=np.nan)
    spread_inf = int(np.isinf(spread_arr).sum())
    valid_flags = [_is_bool_scalar(v) and bool(v) for v in weekly_obs[COL_WEEKLY_VALID].tolist()]
    valid_mask = pd.Series(valid_flags, index=weekly_obs.index)
    # valid quote requires finite spread.
    valid_needs_finite = int((valid_mask & ~np.isfinite(spread_num)).sum())
    spread_problems = (spread_non_numeric > 0) or (spread_inf > 0) or (valid_needs_finite > 0)
    results.append(
        ArtifactCheckResult(
            name="weekly_spread_consistency",
            status=FAIL if spread_problems else PASS,
            message=(
                "valid quotes require finite spread; infinity/non-numeric spread rejected"
                if spread_problems
                else "weekly spread consistent with valid flag"
            ),
            details={
                "non_numeric": spread_non_numeric,
                "infinite": spread_inf,
                "valid_quote_with_nonfinite_spread": valid_needs_finite,
            },
        )
    )

    return tuple(results)


def validate_weekly_artifact(weekly_obs: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a *prepared* weekly frame for rolling recompute.

    Raises :class:`ArtifactValidationError` on any structural problem (see
    :func:`check_weekly_artifact`). On success returns a copy with a normalized
    ``weeknorm`` date column, an explicitly-normalized boolean quote flag, and
    numeric volume/spread columns — so recompute never depends on duplicate-row
    ordering or ``bool()`` on unvalidated data.
    """
    results = check_weekly_artifact(weekly_obs)
    failures = [r for r in results if r.status == FAIL]
    if failures:
        raise ArtifactValidationError(
            "invalid weekly artifact: "
            + "; ".join(f"{r.name}: {r.message}" for r in failures)
        )

    prepared = weekly_obs.copy()
    prepared[_PREPARED_WEEK_COL] = normalize_date_column(prepared, COL_WEEK_END).values
    prepared[COL_WEEKLY_VALID] = [bool(v) for v in prepared[COL_WEEKLY_VALID].tolist()]
    prepared[COL_WEEKLY_DVOL] = pd.to_numeric(prepared[COL_WEEKLY_DVOL], errors="coerce")
    prepared[COL_WEEKLY_SPREAD] = pd.to_numeric(prepared[COL_WEEKLY_SPREAD], errors="coerce")
    return prepared


# ---------------------------------------------------------------------------
# Part 10 — Independent rolling-panel recomputation
# ---------------------------------------------------------------------------

def _global_week_ends(weekly_obs: pd.DataFrame) -> list[pd.Timestamp]:
    if _PREPARED_WEEK_COL in weekly_obs.columns:
        return sorted(set(weekly_obs[_PREPARED_WEEK_COL].tolist()))
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
    *,
    assume_validated: bool = False,
) -> pd.DataFrame:
    """Independently recompute the C4 rolling panel row for each checked ticker.

    Derives the expected global window using ``week_end_date <= target_snapshot``,
    so later weekly rows can never influence the result. Missing ticker-weeks are
    zero-filled for volume and treated as invalid quotes; the denominator is the
    configured ``lookback_weeks`` even during early-history shortfall.

    The weekly artifact is fully validated first (see
    :func:`validate_weekly_artifact`) — it is never silently consumed on invalid
    input. Pass ``assume_validated=True`` with an already-prepared frame to skip
    revalidation within one audit run.

    Returns one row per checked ticker (index = ticker) with columns matching the
    panel provenance fields.
    """
    if assume_validated:
        weekly = weekly_obs
        if _PREPARED_WEEK_COL not in weekly.columns:
            raise PitAuditError("assume_validated=True requires a prepared weekly frame")
    else:
        weekly = validate_weekly_artifact(weekly_obs)

    snapshot = normalize_date_value(target_snapshot, label="target_snapshot")
    lookback_weeks = int(lookback_weeks)
    min_valid_quote_weeks = int(min_valid_quote_weeks)

    all_week_ends = sorted(set(weekly[_PREPARED_WEEK_COL].tolist()))
    window = _expected_window(all_week_ends, snapshot, lookback_weeks)
    window_set = set(window)
    window_shortfall = max(0, lookback_weeks - len(window))
    window_start = window[0] if window else snapshot

    in_window = weekly[weekly[_PREPARED_WEEK_COL].isin(window_set)]
    # Grain is validated upstream, so (weeknorm, ticker) keys are unique.
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

    # Validate the weekly artifact once, then reuse the prepared frame for both
    # the recompute and the future-invariance recompute (no repeated validation).
    prepared_weekly = validate_weekly_artifact(weekly_obs)

    recomputed = recompute_rolling_snapshot(
        snapshot,
        checked_tickers,
        prepared_weekly,
        lookback_weeks,
        min_valid_quote_weeks,
        assume_validated=True,
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
            snapshot,
            checked_tickers,
            prepared_weekly,
            lookback_weeks,
            min_valid_quote_weeks,
            assume_validated=True,
        )

    matches = not field_mismatches
    status: Status = PASS if (matches and future_ok) else FAIL
    return RollingProvenanceResult(
        target_snapshot_date=snapshot,
        tickers_checked=tuple(checked_tickers),
        expected_week_ends=tuple(
            _expected_window(_global_week_ends(prepared_weekly), snapshot, lookback_weeks)
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
    *,
    assume_validated: bool = False,
) -> bool:
    """Recomputing snapshot S must not change when future weekly rows are present.

    Recompute from the full weekly artifact and from a copy restricted to
    ``week_end_date <= S``; the two must be identical under the established
    tolerances.
    """
    snapshot = normalize_date_value(target_snapshot, label="target_snapshot")

    prepared = weekly_obs if assume_validated else validate_weekly_artifact(weekly_obs)

    full = recompute_rolling_snapshot(
        snapshot,
        checked_tickers,
        prepared,
        lookback_weeks,
        min_valid_quote_weeks,
        assume_validated=True,
    )

    restricted_obs = prepared.loc[prepared[_PREPARED_WEEK_COL] <= snapshot].copy()
    restricted = recompute_rolling_snapshot(
        snapshot,
        checked_tickers,
        restricted_obs,
        lookback_weeks,
        min_valid_quote_weeks,
        assume_validated=True,
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


def check_sample_superset_coverage_consistency(
    samples: Sequence[PitResolutionResult],
    coverage: Sequence[SupersetCoverageResult],
    *,
    max_examples: int = _MAX_EXAMPLES,
) -> ArtifactCheckResult:
    """Verify complete sample membership aligns with coverage certification counts."""
    if len(samples) != len(coverage):
        return ArtifactCheckResult(
            name="sample_superset_coverage_consistency",
            status=FAIL,
            message=(
                f"sample/coverage length mismatch: {len(samples)} samples vs "
                f"{len(coverage)} coverage results"
            ),
            details={
                "sample_count": len(samples),
                "coverage_count": len(coverage),
            },
        )

    mismatches: list[dict[str, object]] = []
    for i, (sample, cov) in enumerate(zip(samples, coverage)):
        n_tickers = len(sample.selected_tickers)
        unique_tickers = len(set(sample.selected_tickers))
        consistent = (
            n_tickers == sample.selected_count
            and cov.selected_count == sample.selected_count
            and unique_tickers == sample.selected_count
        )
        if not consistent:
            mismatches.append(
                {
                    "sample_index": i,
                    "trade_date": sample.trade_date.date().isoformat(),
                    "resolved_snapshot_date": (
                        None
                        if sample.resolved_snapshot_date is None
                        else sample.resolved_snapshot_date.date().isoformat()
                    ),
                    "sample_selected_count": sample.selected_count,
                    "selected_tickers_len": n_tickers,
                    "unique_selected_ticker_count": unique_tickers,
                    "coverage_selected_count": cov.selected_count,
                }
            )

    capped = sorted(mismatches, key=lambda m: int(m["sample_index"]))[:max_examples]
    if mismatches:
        return ArtifactCheckResult(
            name="sample_superset_coverage_consistency",
            status=FAIL,
            message=f"{len(mismatches)} sample(s) with membership/count mismatch",
            details={
                "mismatch_count": len(mismatches),
                "mismatches": capped,
                "examples_capped": len(mismatches) > len(capped),
            },
        )

    return ArtifactCheckResult(
        name="sample_superset_coverage_consistency",
        status=PASS,
        message=f"all {len(samples)} sample(s) have consistent superset certification counts",
        details={"sample_count": len(samples)},
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
    # Assign coerced numeric values back so ranking is numeric, never
    # lexicographic on object/string columns. Non-numeric values become NaN and
    # are excluded (finite check below).
    work[COL_DVOL] = pd.to_numeric(work[COL_DVOL], errors="coerce")
    work[COL_SPREAD] = pd.to_numeric(work[COL_SPREAD], errors="coerce")
    dvol = work[COL_DVOL]
    spread = work[COL_SPREAD]
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
    # Compute the reference exactly once, and reuse it for the comparison so
    # production S1 is called exactly once per sample.
    reference = compute_reference_universe(trade_date, panel, dvol_top_pct, spread_bottom_pct)
    comparison = compare_universe_to_reference(
        trade_date, panel, dvol_top_pct, spread_bottom_pct, step1_fn=step1_fn, reference=reference
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

    # Target-snapshot resolution: an expected target with no resolved snapshot is
    # a hard FAIL (not skipped merely because ``resolved`` is None).
    if target_ts is not None:
        if resolved is None:
            status = FAIL
            notes.append(
                f"target_snapshot_date {target_ts.date()} supplied but no snapshot resolved"
            )
        elif target_ts != resolved:
            status = FAIL
            notes.append(f"target_snapshot_date {target_ts.date()} != resolved {resolved.date()}")

    if not comparison.match:
        status = FAIL
        notes.append("production S1 differs from independent reference")

    # Empty-universe conditions are explicit non-blocking WARNs (never silent
    # PASS). A supplied target with no snapshot already forced FAIL above.
    if reference.empty_reason in (
        "before_first_snapshot",
        "no_eligible_rows",
        "no_rows_pass_thresholds",
    ):
        notes.append(f"empty universe: {reference.empty_reason}")
        if status == PASS:
            status = WARN

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

    selected_tickers = tuple(
        sorted(reference.selected["ticker"].astype(str).tolist())
    )
    if len(selected_tickers) != reference.selected_count:
        raise PitAuditError(
            "selected_tickers length != reference.selected_count "
            f"({len(selected_tickers)} != {reference.selected_count})"
        )
    if len(selected_tickers) != len(set(selected_tickers)):
        raise PitAuditError("selected_tickers contains duplicates")

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
        selected_tickers=selected_tickers,
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
    """Package audit results and compute overall status (pure, no I/O).

    No evidence is not PASS: if any mandatory section is missing or empty, the
    report records a blocking failure and ``overall_status`` is FAIL.
    """
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

    # --- Mandatory-evidence gate: absence of evidence is a blocking failure. ---
    mandatory_missing: list[str] = []
    if not artifact_checks:
        mandatory_missing.append("no artifact checks")
    if artifact_envelope is None:
        mandatory_missing.append("missing artifact envelope evidence")
    if not samples:
        mandatory_missing.append("no PIT samples evaluated")
    if not rolling_provenance:
        mandatory_missing.append("no rolling provenance evidence")
    if not sample_superset_coverage:
        mandatory_missing.append("no sample superset coverage evidence")
    if full_history_superset_coverage is None:
        mandatory_missing.append("no full-history superset coverage evidence")
    if mandatory_missing:
        all_statuses.append(FAIL)
        blocking.extend(mandatory_missing)

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
