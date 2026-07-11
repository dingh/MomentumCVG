"""Assembly-readiness audit for option surface A1/A2 artifacts (C6.3).

Pure functions over quote dicts and metadata fields.  Readiness metrics are
derived audit fields — they are not producer validity requirements and do not
change ``surface_valid`` semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Default iron-fly wing symmetry tolerance (strike units).  This is an
# informational structural metric; downstream iron-fly assembly does not require
# symmetric wing distances.
DEFAULT_IRONFLY_SYMMETRY_TOLERANCE = 0.0

SURFACE_KEY_COLUMNS = ("ticker", "entry_date", "expiry_date")

FAIL_SEVERITY = "FAIL"
WARN_SEVERITY = "WARN"


@dataclass(frozen=True)
class QuoteRecord:
    """Minimal quote fields used by readiness checks."""

    side: str
    strike: float
    body_strike: float
    bid: float
    ask: float
    mid: float
    is_body: bool
    is_otm: bool


@dataclass
class SurfaceReadinessRow:
    """Derived readiness for one surface key."""

    ticker: str
    entry_date: Any
    expiry_date: Any
    surface_valid: bool
    has_body_call: bool
    has_body_put: bool
    body_strike: float | None
    body_pair_ready: bool
    straddle_ready: bool
    quotable_body_call_count: int
    quotable_body_put_count: int
    quotable_otm_call_count: int
    quotable_otm_put_count: int
    otm_call_wing_available: bool
    otm_put_wing_available: bool
    otm_wing_pair_available: bool
    symmetric_ironfly_pair_count: int
    symmetric_ironfly_pair_available: bool
    ironfly_candidate_pair_count: int
    ironfly_candidate_ready: bool
    ironcondor_candidate_count: int
    ironcondor_candidate_ready: bool
    readiness_failure_reasons: list[str] = field(default_factory=list)
    consistency_failures: list[str] = field(default_factory=list)
    consistency_warnings: list[str] = field(default_factory=list)


@dataclass
class ReadinessAuditResult:
    """Aggregate readiness audit outcome."""

    rows: list[SurfaceReadinessRow]
    status: str  # PASS | WARN | FAIL
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    failure_reason_breakdown: dict[str, int] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str | None = None


def is_quotable(bid: Any, ask: Any, mid: Any) -> bool:
    """A quote is quotable when bid, ask, and mid are all strictly positive."""
    try:
        return float(bid) > 0 and float(ask) > 0 and float(mid) > 0
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quote_from_mapping(row: Mapping[str, Any]) -> QuoteRecord:
    """Build a ``QuoteRecord`` from an A2 row mapping."""
    body_strike = _to_float(row.get("body_strike"))
    if body_strike is None:
        body_strike = 0.0
    return QuoteRecord(
        side=str(row.get("side", "")).lower(),
        strike=_to_float(row.get("strike")) or 0.0,
        body_strike=body_strike,
        bid=row.get("bid", 0),
        ask=row.get("ask", 0),
        mid=row.get("mid", 0),
        is_body=bool(row.get("is_body")),
        is_otm=bool(row.get("is_otm")),
    )


def expected_is_body(strike: float, body_strike: float) -> bool:
    return strike == body_strike


def expected_is_otm(side: str, strike: float, body_strike: float) -> bool:
    side_l = side.lower()
    if side_l == "call":
        return strike > body_strike
    if side_l == "put":
        return strike < body_strike
    return False


def is_otm_call_wing(quote: QuoteRecord) -> bool:
    return (
        quote.side == "call"
        and quote.strike > quote.body_strike
        and quote.is_otm
        and is_quotable(quote.bid, quote.ask, quote.mid)
    )


def is_otm_put_wing(quote: QuoteRecord) -> bool:
    return (
        quote.side == "put"
        and quote.strike < quote.body_strike
        and quote.is_otm
        and is_quotable(quote.bid, quote.ask, quote.mid)
    )


def is_quotable_body_call(quote: QuoteRecord, body_strike: float) -> bool:
    return (
        quote.side == "call"
        and quote.strike == body_strike
        and quote.is_body
        and is_quotable(quote.bid, quote.ask, quote.mid)
    )


def is_quotable_body_put(quote: QuoteRecord, body_strike: float) -> bool:
    return (
        quote.side == "put"
        and quote.strike == body_strike
        and quote.is_body
        and is_quotable(quote.bid, quote.ask, quote.mid)
    )


def count_symmetric_ironfly_pairs(
    call_wings: Sequence[QuoteRecord],
    put_wings: Sequence[QuoteRecord],
    body_strike: float,
    *,
    symmetry_tolerance: float = DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
) -> int:
    """Count call/put wing pairs with approximately symmetric distance from body."""
    count = 0
    for call in call_wings:
        call_dist = call.strike - body_strike
        for put in put_wings:
            put_dist = body_strike - put.strike
            if abs(call_dist - put_dist) <= symmetry_tolerance:
                count += 1
    return count


def count_ironcondor_candidates(
    quotable_puts: Sequence[QuoteRecord],
    quotable_calls: Sequence[QuoteRecord],
) -> int:
    """Count structural iron-condor configurations (put vertical × call vertical).

    Matches S3 ``build_ironcondor_from_surface`` leg ordering without delta
    filters: long put at lower strike, short put at higher strike; short call
    at lower strike, long call at higher strike.  Body quotes may serve as
    inner short legs.
    """
    put_pairs = _vertical_put_pairs(quotable_puts)
    call_pairs = _vertical_call_pairs(quotable_calls)
    return len(put_pairs) * len(call_pairs)


def _vertical_put_pairs(quotable_puts: Sequence[QuoteRecord]) -> list[tuple[float, float]]:
    """Return (long_strike, short_strike) with long_strike < short_strike."""
    strikes = sorted({q.strike for q in quotable_puts})
    pairs: list[tuple[float, float]] = []
    for i, long_strike in enumerate(strikes):
        for short_strike in strikes[i + 1 :]:
            pairs.append((long_strike, short_strike))
    return pairs


def _vertical_call_pairs(quotable_calls: Sequence[QuoteRecord]) -> list[tuple[float, float]]:
    """Return (short_strike, long_strike) with short_strike < long_strike."""
    strikes = sorted({q.strike for q in quotable_calls})
    pairs: list[tuple[float, float]] = []
    for i, short_strike in enumerate(strikes):
        for long_strike in strikes[i + 1 :]:
            pairs.append((short_strike, long_strike))
    return pairs


def ironcondor_candidate_ready(
    body_pair_ready: bool,
    quotable_puts: Sequence[QuoteRecord],
    quotable_calls: Sequence[QuoteRecord],
) -> bool:
    """Conservative structural condor readiness derived from S3 assembly."""
    if not body_pair_ready:
        return False
    if not quotable_puts or not quotable_calls:
        return False
    return count_ironcondor_candidates(quotable_puts, quotable_calls) > 0


def compute_surface_readiness(
    meta_row: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]],
    *,
    ironfly_symmetry_tolerance: float = DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
) -> SurfaceReadinessRow:
    """Compute readiness metrics for one surface key."""
    ticker = str(meta_row.get("ticker", ""))
    entry_date = meta_row.get("entry_date")
    expiry_date = meta_row.get("expiry_date")
    surface_valid = bool(meta_row.get("surface_valid"))
    has_body_call = bool(meta_row.get("has_body_call"))
    has_body_put = bool(meta_row.get("has_body_put"))
    body_strike_val = _to_float(meta_row.get("body_strike"))

    quotes = [quote_from_mapping(r) for r in quote_rows]
    readiness_reasons: list[str] = []
    consistency_failures: list[str] = []
    consistency_warnings: list[str] = []

    if body_strike_val is None:
        consistency_failures.append("body_strike_mismatch")
        body_strike = 0.0
    else:
        body_strike = body_strike_val

    # Classification and body-strike consistency (per quote).
    for q in quotes:
        exp_body = expected_is_body(q.strike, body_strike)
        if q.is_body != exp_body:
            consistency_failures.append("classification_mismatch")
        exp_otm = expected_is_otm(q.side, q.strike, body_strike)
        if q.is_otm != exp_otm:
            consistency_failures.append("classification_mismatch")
        if q.is_body and q.strike != body_strike:
            consistency_failures.append("body_strike_mismatch")

    quotable_body_calls = [q for q in quotes if is_quotable_body_call(q, body_strike)]
    quotable_body_puts = [q for q in quotes if is_quotable_body_put(q, body_strike)]

    if len(quotable_body_calls) > 1:
        consistency_failures.append("duplicate_body_call")
    if len(quotable_body_puts) > 1:
        consistency_failures.append("duplicate_body_put")

    body_pair_ready = len(quotable_body_calls) >= 1 and len(quotable_body_puts) >= 1
    straddle_ready = body_pair_ready

    a2_has_body_call = len(quotable_body_calls) == 1
    a2_has_body_put = len(quotable_body_puts) == 1

    if has_body_call != a2_has_body_call:
        consistency_failures.append("body_flag_mismatch")
    if has_body_put != a2_has_body_put:
        consistency_failures.append("body_flag_mismatch")
    if surface_valid and not body_pair_ready:
        consistency_failures.append("surface_valid_body_contradiction")
    if not surface_valid and body_pair_ready:
        consistency_warnings.append("surface_invalid_but_body_pair_present")

    otm_call_wings = [q for q in quotes if is_otm_call_wing(q)]
    otm_put_wings = [q for q in quotes if is_otm_put_wing(q)]
    otm_call_wing_available = len(otm_call_wings) > 0
    otm_put_wing_available = len(otm_put_wings) > 0
    otm_wing_pair_available = otm_call_wing_available and otm_put_wing_available

    if not otm_call_wing_available:
        readiness_reasons.append("no_otm_call_wing")
    if not otm_put_wing_available:
        readiness_reasons.append("no_otm_put_wing")
    if not body_pair_ready:
        if not quotable_body_calls:
            readiness_reasons.append("missing_body_call")
        if not quotable_body_puts:
            readiness_reasons.append("missing_body_put")

    symmetric_ironfly_pairs = 0
    if body_pair_ready and otm_wing_pair_available:
        symmetric_ironfly_pairs = count_symmetric_ironfly_pairs(
            otm_call_wings,
            otm_put_wings,
            body_strike,
            symmetry_tolerance=ironfly_symmetry_tolerance,
        )
    symmetric_ironfly_pair_available = symmetric_ironfly_pairs > 0
    ironfly_ready = body_pair_ready and otm_wing_pair_available

    quotable_puts = [q for q in quotes if q.side == "put" and is_quotable(q.bid, q.ask, q.mid)]
    quotable_calls = [q for q in quotes if q.side == "call" and is_quotable(q.bid, q.ask, q.mid)]
    condor_count = count_ironcondor_candidates(quotable_puts, quotable_calls) if body_pair_ready else 0
    condor_ready = ironcondor_candidate_ready(body_pair_ready, quotable_puts, quotable_calls)
    if body_pair_ready and not condor_ready:
        readiness_reasons.append("insufficient_condor_legs")

    # Deduplicate reason lists while preserving order.
    consistency_failures = list(dict.fromkeys(consistency_failures))
    consistency_warnings = list(dict.fromkeys(consistency_warnings))
    readiness_reasons = list(dict.fromkeys(readiness_reasons))

    return SurfaceReadinessRow(
        ticker=ticker,
        entry_date=entry_date,
        expiry_date=expiry_date,
        surface_valid=surface_valid,
        has_body_call=has_body_call,
        has_body_put=has_body_put,
        body_strike=body_strike_val,
        body_pair_ready=body_pair_ready,
        straddle_ready=straddle_ready,
        quotable_body_call_count=len(quotable_body_calls),
        quotable_body_put_count=len(quotable_body_puts),
        quotable_otm_call_count=len(otm_call_wings),
        quotable_otm_put_count=len(otm_put_wings),
        otm_call_wing_available=otm_call_wing_available,
        otm_put_wing_available=otm_put_wing_available,
        otm_wing_pair_available=otm_wing_pair_available,
        symmetric_ironfly_pair_count=symmetric_ironfly_pairs,
        symmetric_ironfly_pair_available=symmetric_ironfly_pair_available,
        ironfly_candidate_pair_count=symmetric_ironfly_pairs,
        ironfly_candidate_ready=ironfly_ready,
        ironcondor_candidate_count=condor_count,
        ironcondor_candidate_ready=condor_ready,
        readiness_failure_reasons=readiness_reasons,
        consistency_failures=consistency_failures,
        consistency_warnings=consistency_warnings,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _aggregate_readiness_metrics(rows: Sequence[SurfaceReadinessRow]) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [r for r in rows if r.surface_valid]
    valid_count = len(valid_rows)

    def count_true(attr: str, subset: Sequence[SurfaceReadinessRow] | None = None) -> int:
        data = subset if subset is not None else rows
        return sum(1 for r in data if getattr(r, attr))

    metrics: dict[str, Any] = {
        "surface_count": total,
        "surface_valid_count": valid_count,
        "surface_valid_rate": _rate(valid_count, total),
        "body_pair_ready_count": count_true("body_pair_ready"),
        "body_pair_ready_rate": _rate(count_true("body_pair_ready"), total),
        "straddle_ready_count": count_true("straddle_ready"),
        "straddle_ready_rate": _rate(count_true("straddle_ready"), total),
        "otm_call_wing_available_count": count_true("otm_call_wing_available"),
        "otm_call_wing_available_rate": _rate(count_true("otm_call_wing_available"), total),
        "otm_put_wing_available_count": count_true("otm_put_wing_available"),
        "otm_put_wing_available_rate": _rate(count_true("otm_put_wing_available"), total),
        "otm_wing_pair_available_count": count_true("otm_wing_pair_available"),
        "otm_wing_pair_available_rate": _rate(count_true("otm_wing_pair_available"), total),
        "symmetric_ironfly_pair_available_count": count_true("symmetric_ironfly_pair_available"),
        "symmetric_ironfly_pair_available_rate": _rate(
            count_true("symmetric_ironfly_pair_available"), total
        ),
        "symmetric_ironfly_pair_count": sum(r.symmetric_ironfly_pair_count for r in rows),
        "ironfly_candidate_ready_count": count_true("ironfly_candidate_ready"),
        "ironfly_candidate_ready_rate": _rate(count_true("ironfly_candidate_ready"), total),
        "ironcondor_candidate_ready_count": count_true("ironcondor_candidate_ready"),
        "ironcondor_candidate_ready_rate": _rate(count_true("ironcondor_candidate_ready"), total),
        "straddle_ready_among_surface_valid_rate": _rate(
            count_true("straddle_ready", valid_rows), valid_count
        ),
        "ironfly_candidate_ready_among_surface_valid_rate": _rate(
            count_true("ironfly_candidate_ready", valid_rows), valid_count
        ),
        "ironcondor_candidate_ready_among_surface_valid_rate": _rate(
            count_true("ironcondor_candidate_ready", valid_rows), valid_count
        ),
    }
    return metrics


def _failure_reason_breakdown(rows: Sequence[SurfaceReadinessRow]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        for reason in row.readiness_failure_reasons:
            breakdown[reason] = breakdown.get(reason, 0) + 1
        for reason in row.consistency_failures:
            key = reason if reason in {
                "body_flag_mismatch",
                "body_strike_mismatch",
                "classification_mismatch",
            } else reason
            breakdown[key] = breakdown.get(key, 0) + 1
    return dict(sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])))


def compute_readiness_verdict(
    rows: Sequence[SurfaceReadinessRow],
    *,
    contract_passed: bool,
    low_coverage_threshold: float = 0.5,
) -> ReadinessAuditResult:
    """Aggregate per-surface readiness into PASS / WARN / FAIL."""
    if not contract_passed:
        return ReadinessAuditResult(
            rows=list(rows),
            status="FAIL",
            blocked=True,
            block_reason="C6.2 contract checks failed; readiness evaluation unreliable",
            failures=["readiness blocked because C6.2 contract checks failed"],
        )

    metrics = _aggregate_readiness_metrics(rows)
    failures: list[str] = []
    warnings: list[str] = []
    examples: list[str] = []

    consistency_fail_count = 0
    for row in rows:
        if row.consistency_failures:
            consistency_fail_count += 1
            if len(examples) < 5:
                examples.append(
                    f"{row.ticker} {row.entry_date} {row.expiry_date}: "
                    f"{', '.join(row.consistency_failures)}"
                )

    if consistency_fail_count:
        failures.append(
            f"{consistency_fail_count} surface(s) have A1/A2 body or classification inconsistencies"
        )

    for row in rows:
        if "surface_valid_body_contradiction" in row.consistency_failures:
            if len(examples) < 8:
                examples.append(
                    f"surface_valid=True but straddle_ready=False: "
                    f"{row.ticker} entry_date={row.entry_date}"
                )

    invalid_with_body = sum(1 for r in rows if r.consistency_warnings)
    if invalid_with_body:
        warnings.append(
            f"{invalid_with_body} surface_valid=False row(s) still have quotable body pairs"
        )

    valid_count = metrics.get("surface_valid_count", 0)
    if valid_count == 0 and rows:
        warnings.append("no surface_valid rows in audited sample")

    for metric_key, label in (
        ("straddle_ready_among_surface_valid_rate", "straddle"),
        ("ironfly_candidate_ready_among_surface_valid_rate", "iron-fly"),
        ("ironcondor_candidate_ready_among_surface_valid_rate", "iron-condor"),
    ):
        rate = metrics.get(metric_key)
        if rate is not None and valid_count >= 3 and rate < low_coverage_threshold:
            warnings.append(
                f"low {label} readiness among surface_valid: {rate:.1%} "
                f"(threshold {low_coverage_threshold:.0%})"
            )

    if metrics.get("ironfly_candidate_ready_count", 0) == 0 and valid_count > 0:
        warnings.append("no iron-fly candidates in audited sample")
    if metrics.get("ironcondor_candidate_ready_count", 0) == 0 and valid_count > 0:
        warnings.append("no iron-condor candidates in audited sample (conservative structural rule)")

    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    return ReadinessAuditResult(
        rows=list(rows),
        status=status,
        metrics=metrics,
        failures=failures,
        warnings=warnings,
        examples=examples,
        failure_reason_breakdown=_failure_reason_breakdown(rows),
    )


def run_readiness_audit(
    meta_rows: Sequence[Mapping[str, Any]],
    quotes_by_surface: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
    *,
    contract_passed: bool,
    ironfly_symmetry_tolerance: float = DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
) -> ReadinessAuditResult:
    """Run readiness for pre-grouped quote rows keyed by surface tuple."""
    readiness_rows: list[SurfaceReadinessRow] = []
    for meta_row in meta_rows:
        key = tuple(meta_row.get(col) for col in SURFACE_KEY_COLUMNS)
        quote_rows = quotes_by_surface.get(key, ())
        readiness_rows.append(
            compute_surface_readiness(
                meta_row,
                quote_rows,
                ironfly_symmetry_tolerance=ironfly_symmetry_tolerance,
            )
        )
    return compute_readiness_verdict(readiness_rows, contract_passed=contract_passed)


def group_quotes_by_surface(
    quote_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    """Group A2 rows by (ticker, entry_date, expiry_date)."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in quote_rows:
        key = (row.get("ticker"), row.get("entry_date"), row.get("expiry_date"))
        grouped.setdefault(key, []).append(row)
    return grouped
