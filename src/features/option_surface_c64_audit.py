"""C6.4 real-artifact audit helpers: bounded load, duplicate triage, coverage (read-only)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.trading_day import target_weekly_expiry_from_schedule, weekly_trade_dates_in_range
from src.features.option_surface_contract import (
    META_GRAIN_COLUMNS,
    QUOTE_GRAIN_COLUMNS,
    _to_date,
)
from src.features.option_surface_readiness import SurfaceReadinessRow

WEEKLY_SCHEDULE_TAIL_DAYS = 21


@dataclass
class DuplicateGroupSummary:
    """One duplicate grain group with identical/conflicting classification."""

    grain_label: str
    key: tuple[Any, ...]
    row_count: int
    classification: str  # IDENTICAL_DUPLICATE | CONFLICTING_DUPLICATE
    differing_columns: list[str] = field(default_factory=list)


@dataclass
class DuplicateTriageResult:
    """Aggregate duplicate triage for one artifact grain."""

    grain_label: str
    grain_columns: list[str]
    duplicate_key_count: int
    duplicate_row_count: int
    affected_ticker_count: int
    affected_date_count: int
    identical_key_count: int
    conflicting_key_count: int
    groups: list[DuplicateGroupSummary] = field(default_factory=list)
    example_keys: list[str] = field(default_factory=list)


@dataclass
class WeeklyExpiryEvidence:
    """Strict weekly expiry alignment vs schedule successor."""

    eligible_row_count: int
    exact_target_match_count: int
    silent_mismatch_count: int
    missing_target_failure_count: int
    target_not_listed_failure_count: int
    no_expiries_on_entry_chain_count: int
    mismatch_examples: list[str] = field(default_factory=list)


@dataclass
class CoverageMetrics:
    """Requested vs actual artifact coverage."""

    meta_row_count: int
    quote_row_count: int
    ticker_count: int
    entry_date_count: int
    requested_start: date | None
    requested_end: date | None
    schedule_min: date | None
    schedule_max: date | None
    meta_entry_min: date | None
    meta_entry_max: date | None
    quote_entry_min: date | None
    quote_entry_max: date | None
    expiry_min: date | None
    expiry_max: date | None
    surface_valid_count: int
    surface_valid_rate: float
    straddle_ready_count: int
    straddle_ready_rate: float
    ironfly_candidate_ready_count: int
    ironfly_candidate_ready_rate: float
    ironcondor_candidate_ready_count: int
    ironcondor_candidate_ready_rate: float
    straddle_ready_among_surface_valid_rate: float | None
    ironfly_ready_among_surface_valid_rate: float | None
    ironcondor_ready_among_surface_valid_rate: float | None
    requested_tickers: list[str] = field(default_factory=list)
    actual_tickers: list[str] = field(default_factory=list)
    absent_requested_tickers: list[str] = field(default_factory=list)


def _parquet_filters(
    *,
    sample_tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> list[tuple] | None:
    filters: list[tuple] = []
    if sample_tickers:
        tickers = sorted({t.upper() for t in sample_tickers})
        filters.append(("ticker", "in", tickers))
    if start_date is not None:
        filters.append(("entry_date", ">=", pd.Timestamp(start_date)))
    if end_date is not None:
        filters.append(("entry_date", "<=", pd.Timestamp(end_date)))
    return filters or None


def load_bounded_artifacts(
    meta_path: Path,
    quotes_path: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sample_tickers: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load A1/A2 parquets with optional pyarrow pushdown filters."""
    filters = _parquet_filters(
        sample_tickers=sample_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    try:
        meta_df = pd.read_parquet(meta_path, filters=filters)
        quotes_df = pd.read_parquet(quotes_path, filters=filters)
    except Exception:
        meta_df = pd.read_parquet(meta_path)
        quotes_df = pd.read_parquet(quotes_path)
        from src.features.option_surface_contract import filter_artifacts

        return filter_artifacts(
            meta_df,
            quotes_df,
            start_date=start_date,
            end_date=end_date,
            sample_tickers=sample_tickers,
        )
    return meta_df.reset_index(drop=True), quotes_df.reset_index(drop=True)


def _classify_duplicate_group(group: pd.DataFrame) -> tuple[str, list[str]]:
    if len(group) <= 1:
        return "PASS", []
    deduped = group.drop_duplicates()
    if len(deduped) == 1:
        return "IDENTICAL_DUPLICATE", []
    differing: list[str] = []
    for col in group.columns:
        if group[col].nunique(dropna=False) > 1:
            differing.append(str(col))
    return "CONFLICTING_DUPLICATE", sorted(differing)


def triage_duplicates(
    df: pd.DataFrame,
    grain_columns: Sequence[str],
    *,
    grain_label: str,
    max_examples: int = 10,
) -> DuplicateTriageResult:
    """Classify duplicate grain groups as identical or conflicting."""
    missing = [col for col in grain_columns if col not in df.columns]
    if missing:
        return DuplicateTriageResult(
            grain_label=grain_label,
            grain_columns=list(grain_columns),
            duplicate_key_count=0,
            duplicate_row_count=0,
            affected_ticker_count=0,
            affected_date_count=0,
            identical_key_count=0,
            conflicting_key_count=0,
        )

    normalized = df.copy()
    for col in grain_columns:
        if col.endswith("_date") or col == "entry_date":
            normalized[col] = normalized[col].map(_to_date)

    grouped = normalized.groupby(list(grain_columns), dropna=False)
    duplicate_groups: list[DuplicateGroupSummary] = []
    identical_count = 0
    conflicting_count = 0
    duplicate_row_count = 0
    affected_tickers: set[str] = set()
    affected_dates: set[date] = set()
    example_keys: list[str] = []

    for key, group in grouped:
        if len(group) <= 1:
            continue
        classification, differing = _classify_duplicate_group(group)
        key_tuple = key if isinstance(key, tuple) else (key,)
        duplicate_groups.append(
            DuplicateGroupSummary(
                grain_label=grain_label,
                key=key_tuple,
                row_count=len(group),
                classification=classification,
                differing_columns=differing,
            )
        )
        duplicate_row_count += len(group)
        if classification == "IDENTICAL_DUPLICATE":
            identical_count += 1
        else:
            conflicting_count += 1
        if "ticker" in grain_columns:
            idx = list(grain_columns).index("ticker")
            affected_tickers.add(str(key_tuple[idx]))
        if "entry_date" in grain_columns:
            idx = list(grain_columns).index("entry_date")
            entry_val = key_tuple[idx]
            if isinstance(entry_val, date):
                affected_dates.add(entry_val)
        if len(example_keys) < max_examples:
            example_keys.append(
                ", ".join(f"{col}={key_tuple[i]}" for i, col in enumerate(grain_columns))
            )

    return DuplicateTriageResult(
        grain_label=grain_label,
        grain_columns=list(grain_columns),
        duplicate_key_count=len(duplicate_groups),
        duplicate_row_count=duplicate_row_count,
        affected_ticker_count=len(affected_tickers),
        affected_date_count=len(affected_dates),
        identical_key_count=identical_count,
        conflicting_key_count=conflicting_count,
        groups=duplicate_groups,
        example_keys=example_keys,
    )


def compute_weekly_expiry_evidence(
    meta_df: pd.DataFrame,
    *,
    data_root: Path | str,
    start_date: date,
    end_date: date,
) -> WeeklyExpiryEvidence:
    """Compare expiry_date to strict schedule successor where available."""
    schedule_end = end_date + timedelta(days=WEEKLY_SCHEDULE_TAIL_DAYS)
    schedule = weekly_trade_dates_in_range(start_date, schedule_end, data_root)

    eligible = 0
    exact_match = 0
    silent_mismatch = 0
    missing_target = 0
    target_not_listed = 0
    no_expiries = 0
    mismatch_examples: list[str] = []

    for _, row in meta_df.iterrows():
        entry = _to_date(row.get("entry_date"))
        if entry is None:
            continue
        target = target_weekly_expiry_from_schedule(entry, schedule)
        if target is None:
            continue
        eligible += 1
        expiry = _to_date(row.get("expiry_date"))
        reason = row.get("failure_reason")
        reason_str = None if pd.isna(reason) else str(reason)

        if reason_str == "no_target_weekly_expiry":
            missing_target += 1
            continue
        if reason_str == "target_weekly_expiry_not_listed":
            target_not_listed += 1
            continue
        if reason_str == "no_expiries_on_entry_chain":
            no_expiries += 1
            continue

        if expiry == target:
            exact_match += 1
        else:
            silent_mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    f"ticker={row.get('ticker')} entry_date={entry} "
                    f"expiry_date={expiry} target={target} failure_reason={reason_str}"
                )

    return WeeklyExpiryEvidence(
        eligible_row_count=eligible,
        exact_target_match_count=exact_match,
        silent_mismatch_count=silent_mismatch,
        missing_target_failure_count=missing_target,
        target_not_listed_failure_count=target_not_listed,
        no_expiries_on_entry_chain_count=no_expiries,
        mismatch_examples=mismatch_examples,
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_coverage_metrics(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    readiness_rows: Sequence[SurfaceReadinessRow],
    *,
    requested_start: date | None,
    requested_end: date | None,
    requested_tickers: Sequence[str] | None,
    data_root: Path | str | None,
    frequency: str,
) -> CoverageMetrics:
    """Aggregate requested vs actual coverage and readiness rates."""
    entry_dates = [_to_date(v) for v in meta_df.get("entry_date", [])]
    entry_dates = [d for d in entry_dates if d is not None]
    quote_entry_dates = [_to_date(v) for v in quotes_df.get("entry_date", [])]
    quote_entry_dates = [d for d in quote_entry_dates if d is not None]
    expiry_dates = [_to_date(v) for v in meta_df.get("expiry_date", [])]
    expiry_dates = [d for d in expiry_dates if d is not None]

    schedule_min = schedule_max = None
    if frequency == "weekly" and data_root is not None and requested_start and requested_end:
        schedule = weekly_trade_dates_in_range(requested_start, requested_end, data_root)
        if schedule:
            schedule_min = schedule[0]
            schedule_max = schedule[-1]

    actual_tickers = sorted(meta_df["ticker"].astype(str).unique()) if "ticker" in meta_df.columns else []
    requested = sorted({t.upper() for t in requested_tickers}) if requested_tickers else []
    absent = [t for t in requested if t not in {x.upper() for x in actual_tickers}]

    valid_count = int(meta_df["surface_valid"].sum()) if "surface_valid" in meta_df.columns else 0
    meta_count = len(meta_df)

    straddle_count = sum(1 for r in readiness_rows if r.straddle_ready)
    ironfly_count = sum(1 for r in readiness_rows if r.ironfly_candidate_ready)
    ironcondor_count = sum(1 for r in readiness_rows if r.ironcondor_candidate_ready)
    valid_rows = [r for r in readiness_rows if r.surface_valid]
    valid_n = len(valid_rows)

    return CoverageMetrics(
        meta_row_count=meta_count,
        quote_row_count=len(quotes_df),
        ticker_count=len(actual_tickers),
        entry_date_count=len(set(entry_dates)),
        requested_start=requested_start,
        requested_end=requested_end,
        schedule_min=schedule_min,
        schedule_max=schedule_max,
        meta_entry_min=min(entry_dates) if entry_dates else None,
        meta_entry_max=max(entry_dates) if entry_dates else None,
        quote_entry_min=min(quote_entry_dates) if quote_entry_dates else None,
        quote_entry_max=max(quote_entry_dates) if quote_entry_dates else None,
        expiry_min=min(expiry_dates) if expiry_dates else None,
        expiry_max=max(expiry_dates) if expiry_dates else None,
        surface_valid_count=valid_count,
        surface_valid_rate=_rate(valid_count, meta_count),
        straddle_ready_count=straddle_count,
        straddle_ready_rate=_rate(straddle_count, meta_count),
        ironfly_candidate_ready_count=ironfly_count,
        ironfly_candidate_ready_rate=_rate(ironfly_count, meta_count),
        ironcondor_candidate_ready_count=ironcondor_count,
        ironcondor_candidate_ready_rate=_rate(ironcondor_count, meta_count),
        straddle_ready_among_surface_valid_rate=_rate(
            sum(1 for r in valid_rows if r.straddle_ready), valid_n
        )
        if valid_n
        else None,
        ironfly_ready_among_surface_valid_rate=_rate(
            sum(1 for r in valid_rows if r.ironfly_candidate_ready), valid_n
        )
        if valid_n
        else None,
        ironcondor_ready_among_surface_valid_rate=_rate(
            sum(1 for r in valid_rows if r.ironcondor_candidate_ready), valid_n
        )
        if valid_n
        else None,
        requested_tickers=requested,
        actual_tickers=actual_tickers,
        absent_requested_tickers=absent,
    )


def per_ticker_coverage_from_meta(
    meta_df: pd.DataFrame,
    readiness_rows: Sequence[SurfaceReadinessRow],
) -> list[dict[str, Any]]:
    """Per-ticker table including producer failure_reason for invalid rows."""
    readiness_by_key = {(str(r.ticker), _to_date(r.entry_date)): r for r in readiness_rows}
    by_ticker: dict[str, list[dict[str, Any]]] = {}

    for _, row in meta_df.iterrows():
        ticker = str(row["ticker"])
        entry = _to_date(row["entry_date"])
        key = (ticker, entry)
        readiness = readiness_by_key.get(key)
        by_ticker.setdefault(ticker, []).append(
            {
                "surface_valid": bool(row.get("surface_valid")),
                "failure_reason": row.get("failure_reason"),
                "straddle_ready": readiness.straddle_ready if readiness else False,
                "ironfly_ready": readiness.ironfly_candidate_ready if readiness else False,
                "ironcondor_ready": readiness.ironcondor_candidate_ready if readiness else False,
            }
        )

    table: list[dict[str, Any]] = []
    for ticker in sorted(by_ticker):
        rows = by_ticker[ticker]
        attempted = len(rows)
        valid = sum(1 for r in rows if r["surface_valid"])
        straddle = sum(1 for r in rows if r["straddle_ready"])
        ironfly = sum(1 for r in rows if r["ironfly_ready"])
        ironcondor = sum(1 for r in rows if r["ironcondor_ready"])
        failure_counts: dict[str, int] = {}
        for r in rows:
            if r["surface_valid"]:
                continue
            reason = r["failure_reason"]
            key = "(null)" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
            failure_counts[key] = failure_counts.get(key, 0) + 1
        top_failure = max(failure_counts, key=failure_counts.get) if failure_counts else "(none)"
        table.append(
            {
                "ticker": ticker,
                "attempted_surface_count": attempted,
                "surface_valid_count": valid,
                "surface_valid_rate": _rate(valid, attempted),
                "straddle_ready_count": straddle,
                "straddle_ready_rate": _rate(straddle, attempted),
                "ironfly_ready_count": ironfly,
                "ironfly_ready_rate": _rate(ironfly, attempted),
                "ironcondor_ready_count": ironcondor,
                "ironcondor_ready_rate": _rate(ironcondor, attempted),
                "top_failure_reason": top_failure,
            }
        )
    return table


def duplicate_verdict(
    meta_triage: DuplicateTriageResult,
    quote_triage: DuplicateTriageResult,
    *,
    legacy_mode: bool,
) -> tuple[str, list[str], list[str]]:
    """Return (status, blocking_failures, warnings) for duplicate triage."""
    blocking: list[str] = []
    warnings: list[str] = []

    for triage in (meta_triage, quote_triage):
        if triage.conflicting_key_count:
            blocking.append(
                f"{triage.grain_label}: {triage.conflicting_key_count} CONFLICTING_DUPLICATE key(s)"
            )
        if triage.identical_key_count:
            msg = (
                f"{triage.grain_label}: {triage.identical_key_count} IDENTICAL_DUPLICATE key(s) "
                f"({triage.duplicate_row_count} rows)"
            )
            if legacy_mode:
                warnings.append(msg)
            else:
                blocking.append(msg)

    if blocking:
        return "FAIL", blocking, warnings
    if warnings:
        return "WARN", blocking, warnings
    return "PASS", blocking, warnings


def weekly_expiry_verdict(
    evidence: WeeklyExpiryEvidence,
    *,
    legacy_mode: bool,
) -> tuple[str, list[str], list[str]]:
    """Verdict for strict weekly expiry alignment."""
    blocking: list[str] = []
    warnings: list[str] = []

    if evidence.silent_mismatch_count:
        msg = (
            f"{evidence.silent_mismatch_count} row(s) with expiry_date != "
            f"target_weekly_expiry_from_schedule (eligible={evidence.eligible_row_count})"
        )
        if legacy_mode:
            warnings.append(msg)
        else:
            blocking.append(msg)

    if legacy_mode and evidence.silent_mismatch_count:
        return "WARN", blocking, warnings
    if blocking:
        return "FAIL", blocking, warnings
    return "PASS", blocking, warnings


def triage_meta_duplicates(meta_df: pd.DataFrame) -> DuplicateTriageResult:
    return triage_duplicates(meta_df, META_GRAIN_COLUMNS, grain_label="A1")


def triage_quote_duplicates(quotes_df: pd.DataFrame) -> DuplicateTriageResult:
    return triage_duplicates(quotes_df, QUOTE_GRAIN_COLUMNS, grain_label="A2")
