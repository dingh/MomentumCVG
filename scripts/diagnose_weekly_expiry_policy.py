"""C6.1B read-only weekly expiry policy diagnostic (Sprint 004).

Compares current chain-scanned expiry selection against calendar-paired weekly
target expiry semantics. Writes a markdown report only; does not mutate producer
artifacts, cache, or parquet files.

C6.1B decides policy for C6.1C: strict calendar-paired weekly expiry with no
nearest-DTE fallback. Sample A (known-weekly) is the mechanical gate; Sample B
(broad C4 coverage) is informational only.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.orats_provider import ORATSDataProvider, OptionQuote
from src.data.paths import DEFAULT_ADJUSTED_LIQUID_ROOT, DEFAULT_LIQUID_TICKERS_PATH
from src.data.trading_day import (
    orats_daily_parquet_path,
    target_weekly_expiry_from_schedule,
    weekly_trade_dates_in_range,
)
from src.features.option_surface_analyzer import OptionSurfaceBuilder

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "SPY", "QQQ")
DEFAULT_START_DATE = date(2024, 1, 1)
DEFAULT_END_DATE = date(2024, 3, 31)
DEFAULT_MAX_ENTRY_DATES = 12
DEFAULT_COVERAGE_MAX_TICKERS = 50
DEFAULT_OUTPUT_REPORT = Path("docs/tmp/c6_1b_weekly_expiry_diagnostic.md")
WEEKLY_DTE_TARGET = 7
SAMPLE_A_SANITY_THRESHOLD = 0.90
SCHEDULE_TAIL_DAYS = 21
VERDICT_PASS_POLICY = "PASS WITH POLICY CLARIFICATION"
NO_PRODUCER_CHANGE_STATEMENT = (
    "No producer expiry behavior was changed in C6.1B."
)
NO_FALLBACK_STATEMENT = (
    "C6.1C must not silently substitute a nearby expiry. "
    "No fallback to nearest-DTE expiry is allowed."
)
PARQUET_COLUMNS = (
    "ticker",
    "expirDate",
    "adj_stkPx",
    "adj_strike",
    "adj_cBidPx",
    "adj_cAskPx",
    "adj_pBidPx",
    "adj_pAskPx",
)


@dataclass(frozen=True)
class TickerDaySnapshot:
    """In-memory per-ticker slice for one entry-date parquet (diagnostic cache)."""

    ticker: str
    entry_date: date
    spot: Decimal | None
    available_expiries: tuple[date, ...]
    rows_by_expiry: dict[date, pd.DataFrame]


@dataclass(frozen=True)
class DiagnosedObservation:
    ticker: str
    entry_date: date
    expiry_chain_scanned: date | None
    expiry_target_weekly: date | None
    target_listed_on_chain: bool
    target_body_call_quotable: bool
    target_body_put_quotable: bool
    target_body_pair_quotable: bool
    dte_chain: int | None
    dte_target: int | None
    dte_delta: int | None
    expiries_match: bool

    @property
    def weekly_tradable(self) -> bool:
        """Exact target listed and body call/put quotable (strict weekly rule)."""
        return self.target_listed_on_chain and self.target_body_pair_quotable


@dataclass(frozen=True)
class SkippedObservation:
    ticker: str
    entry_date: date
    reason: str


@dataclass(frozen=True)
class SampleResult:
    label: str
    tickers: list[str]
    entry_dates: list[date]
    diagnosed: list[DiagnosedObservation]
    skipped: list[SkippedObservation]
    metrics: dict[str, Any]
    sampling_method: str


def normalize_tickers(raw_tickers: Sequence[object]) -> list[str]:
    """Strip, uppercase, dedupe, and preserve first-seen order for ticker scope."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_tickers:
        if pd.isna(raw):
            continue
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    if not normalized:
        raise ValueError("Ticker scope is empty after normalization")
    return normalized


def is_body_quote_quotable(quote: OptionQuote | None) -> bool:
    """Match producer body-leg quotability in ``process_single_entry``."""
    if quote is None:
        return False
    return quote.bid > 0 and quote.ask > 0 and quote.mid > 0


def is_side_quotable(bid: float, ask: float) -> bool:
    """Match producer bid/ask/mid checks on raw ORATS wide-format prices."""
    if bid <= 0 or ask <= 0:
        return False
    return (bid + ask) / 2 > 0


def select_body_strike_from_strikes(
    strikes: Sequence[Decimal],
    entry_spot: Decimal,
) -> Decimal | None:
    """Match producer ATM body strike selection (nearest strike; ties -> lower)."""
    if not strikes:
        return None
    return min(strikes, key=lambda strike: (abs(strike - entry_spot), strike))


def compute_dte_fields(
    entry_date: date,
    expiry_chain_scanned: date | None,
    expiry_target_weekly: date | None,
) -> tuple[int | None, int | None, int | None]:
    """Return ``(dte_chain, dte_target, dte_delta)`` for mismatch diagnostics."""
    dte_chain = (
        (expiry_chain_scanned - entry_date).days if expiry_chain_scanned is not None else None
    )
    dte_target = (
        (expiry_target_weekly - entry_date).days if expiry_target_weekly is not None else None
    )
    dte_delta = (
        dte_chain - dte_target
        if dte_chain is not None and dte_target is not None
        else None
    )
    return dte_chain, dte_target, dte_delta


def evaluate_target_body_quotability(
    provider: ORATSDataProvider,
    ticker: str,
    entry_date: date,
    expiry_target_weekly: date,
) -> tuple[bool, bool, bool]:
    """Return call/put/pair quotability for target expiry using producer body rules."""
    chain = provider.get_option_chain(
        ticker=ticker,
        trade_date=entry_date,
        expiry_date=expiry_target_weekly,
    )
    entry_spot_raw = provider.get_spot_price(ticker, entry_date)
    if entry_spot_raw is None:
        return False, False, False

    entry_spot = Decimal(str(entry_spot_raw))
    strikes = sorted({quote.strike for quote in chain})
    body_strike = select_body_strike_from_strikes(strikes, entry_spot)
    if body_strike is None:
        return False, False, False

    body_call = next(
        (quote for quote in chain if quote.strike == body_strike and quote.option_type == "call"),
        None,
    )
    body_put = next(
        (quote for quote in chain if quote.strike == body_strike and quote.option_type == "put"),
        None,
    )
    call_ok = is_body_quote_quotable(body_call)
    put_ok = is_body_quote_quotable(body_put)
    return call_ok, put_ok, call_ok and put_ok


def evaluate_target_body_from_snapshot(
    snapshot: TickerDaySnapshot,
    expiry_target_weekly: date,
) -> tuple[bool, bool, bool]:
    """Evaluate target body quotability from a preloaded ticker-day snapshot."""
    if snapshot.spot is None:
        return False, False, False

    expiry_rows = snapshot.rows_by_expiry.get(expiry_target_weekly)
    if expiry_rows is None or expiry_rows.empty:
        return False, False, False

    strikes = [Decimal(str(value)) for value in expiry_rows["adj_strike"].tolist()]
    body_strike = select_body_strike_from_strikes(strikes, snapshot.spot)
    if body_strike is None:
        return False, False, False

    strike_rows = expiry_rows[
        expiry_rows["adj_strike"].apply(lambda value: Decimal(str(value)) == body_strike)
    ]
    if strike_rows.empty:
        return False, False, False

    row = strike_rows.iloc[0]
    call_ok = is_side_quotable(float(row["adj_cBidPx"]), float(row["adj_cAskPx"]))
    put_ok = is_side_quotable(float(row["adj_pBidPx"]), float(row["adj_pAskPx"]))
    return call_ok, put_ok, call_ok and put_ok


def load_entry_day_snapshots(
    input_root: Path,
    entry_date: date,
    tickers: Sequence[str],
) -> tuple[dict[str, TickerDaySnapshot] | None, str | None]:
    """Load one entry-date parquet once and slice it to the sample tickers."""
    parquet_path = orats_daily_parquet_path(input_root, entry_date)
    if not parquet_path.is_file():
        return None, "entry_parquet_missing"

    try:
        df = pd.read_parquet(
            parquet_path,
            columns=list(PARQUET_COLUMNS),
            filters=[("ticker", "in", list(tickers))],
        )
    except Exception as exc:
        return None, f"entry_chain_error: {exc}"

    if df.empty:
        return {
            ticker: TickerDaySnapshot(
                ticker=ticker,
                entry_date=entry_date,
                spot=None,
                available_expiries=(),
                rows_by_expiry={},
            )
            for ticker in tickers
        }, None

    df = df.copy()
    df["expirDate"] = pd.to_datetime(df["expirDate"]).dt.date

    snapshots: dict[str, TickerDaySnapshot] = {}
    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker]
        if ticker_df.empty:
            snapshots[ticker] = TickerDaySnapshot(
                ticker=ticker,
                entry_date=entry_date,
                spot=None,
                available_expiries=(),
                rows_by_expiry={},
            )
            continue

        spot_value = ticker_df["adj_stkPx"].iloc[0]
        spot = None if pd.isna(spot_value) else Decimal(str(spot_value))
        expiries = tuple(sorted(ticker_df["expirDate"].unique()))
        rows_by_expiry = {
            expiry: ticker_df[ticker_df["expirDate"] == expiry].copy()
            for expiry in expiries
        }
        snapshots[ticker] = TickerDaySnapshot(
            ticker=ticker,
            entry_date=entry_date,
            spot=spot,
            available_expiries=expiries,
            rows_by_expiry=rows_by_expiry,
        )

    return snapshots, None


def build_weekly_schedule(
    input_root: Path | str,
    sample_start: date,
    sample_end: date,
    *,
    schedule_tail_days: int = SCHEDULE_TAIL_DAYS,
) -> list[date]:
    """Build weekly entry schedule with tail room for ``schedule[i+1]`` lookups."""
    schedule_end = sample_end + timedelta(days=schedule_tail_days)
    return weekly_trade_dates_in_range(sample_start, schedule_end, input_root)


def select_entry_dates(
    schedule: Sequence[date],
    sample_start: date,
    sample_end: date,
    max_entry_dates: int | None,
) -> list[date]:
    """Select bounded weekly entry dates inside the sample window."""
    entry_dates = [day for day in schedule if sample_start <= day <= sample_end]
    if max_entry_dates is not None and len(entry_dates) > max_entry_dates:
        return entry_dates[:max_entry_dates]
    return entry_dates


def sample_coverage_tickers(
    ranked_tickers: Sequence[str],
    max_tickers: int,
) -> list[str]:
    """Deterministic stratified sample: top / middle / lower by liquidity rank order.

    ``ranked_tickers`` should already be sorted most-liquid → least-liquid.
    """
    if max_tickers <= 0:
        raise ValueError("max_tickers must be positive")
    universe = list(ranked_tickers)
    if not universe:
        raise ValueError("Coverage ticker universe is empty")
    if len(universe) <= max_tickers:
        return universe

    n_top = max_tickers // 3
    n_mid = max_tickers // 3
    n_low = max_tickers - n_top - n_mid
    mid_start = max(0, (len(universe) - n_mid) // 2)

    selected: list[str] = []
    seen: set[str] = set()

    def _extend(candidates: Sequence[str], limit: int) -> None:
        added = 0
        for symbol in candidates:
            if added >= limit or len(selected) >= max_tickers:
                return
            if symbol in seen:
                continue
            seen.add(symbol)
            selected.append(symbol)
            added += 1

    _extend(universe[:n_top], n_top)
    _extend(universe[mid_start : mid_start + n_mid], n_mid)
    _extend(list(reversed(universe)), n_low)
    if len(selected) < max_tickers:
        _extend(universe, max_tickers - len(selected))
    return selected


def load_ranked_coverage_tickers(tickers_path: Path) -> list[str]:
    """Load C4 liquid tickers ranked by ``snapshots_qualified`` descending."""
    if not tickers_path.exists():
        raise FileNotFoundError(f"Coverage ticker file not found: {tickers_path}")
    df = pd.read_csv(tickers_path)
    if "Ticker" not in df.columns:
        raise ValueError(f"Ticker column not found in {tickers_path}")
    if "snapshots_qualified" in df.columns:
        df = df.sort_values("snapshots_qualified", ascending=False, kind="mergesort")
    return normalize_tickers(df["Ticker"].tolist())


def diagnose_observation(
    builder: OptionSurfaceBuilder,
    provider: ORATSDataProvider,
    ticker: str,
    entry_date: date,
    schedule: Sequence[date],
    snapshot: TickerDaySnapshot | None = None,
) -> DiagnosedObservation | SkippedObservation:
    """Diagnose one `(ticker, entry_date)` without changing producer behavior."""
    if entry_date not in schedule:
        return SkippedObservation(ticker, entry_date, "entry_date_not_in_schedule")

    expiry_target_weekly = target_weekly_expiry_from_schedule(entry_date, schedule)
    if expiry_target_weekly is None:
        return SkippedObservation(ticker, entry_date, "no_successor_week")

    if snapshot is not None:
        available_expiries = list(snapshot.available_expiries)
        if not available_expiries:
            return SkippedObservation(ticker, entry_date, "no_expiries_on_entry_chain")

        provider.get_available_expiries = lambda _ticker, _trade_date: available_expiries  # type: ignore[method-assign]
        expiry_chain_scanned = builder._find_best_expiry(
            ticker, entry_date, WEEKLY_DTE_TARGET
        )
        target_listed_on_chain = expiry_target_weekly in available_expiries
        call_ok, put_ok, pair_ok = evaluate_target_body_from_snapshot(
            snapshot, expiry_target_weekly
        )
    else:
        try:
            available_expiries = provider.get_available_expiries(ticker, entry_date)
        except FileNotFoundError:
            return SkippedObservation(ticker, entry_date, "entry_parquet_missing")
        except Exception as exc:
            return SkippedObservation(ticker, entry_date, f"entry_chain_error: {exc}")

        if not available_expiries:
            return SkippedObservation(ticker, entry_date, "no_expiries_on_entry_chain")

        expiry_chain_scanned = builder._find_best_expiry(
            ticker, entry_date, WEEKLY_DTE_TARGET
        )
        target_listed_on_chain = expiry_target_weekly in available_expiries

        try:
            call_ok, put_ok, pair_ok = evaluate_target_body_quotability(
                provider, ticker, entry_date, expiry_target_weekly
            )
        except FileNotFoundError:
            return SkippedObservation(ticker, entry_date, "entry_parquet_missing")
        except Exception as exc:
            return SkippedObservation(ticker, entry_date, f"target_body_error: {exc}")

    dte_chain, dte_target, dte_delta = compute_dte_fields(
        entry_date, expiry_chain_scanned, expiry_target_weekly
    )
    expiries_match = (
        expiry_chain_scanned is not None
        and expiry_target_weekly is not None
        and expiry_chain_scanned == expiry_target_weekly
    )

    return DiagnosedObservation(
        ticker=ticker,
        entry_date=entry_date,
        expiry_chain_scanned=expiry_chain_scanned,
        expiry_target_weekly=expiry_target_weekly,
        target_listed_on_chain=target_listed_on_chain,
        target_body_call_quotable=call_ok,
        target_body_put_quotable=put_ok,
        target_body_pair_quotable=pair_ok,
        dte_chain=dte_chain,
        dte_target=dte_target,
        dte_delta=dte_delta,
        expiries_match=expiries_match,
    )


def run_diagnostic(
    *,
    input_root: Path,
    tickers: Sequence[str],
    sample_start: date,
    sample_end: date,
    max_entry_dates: int | None,
) -> tuple[list[date], list[DiagnosedObservation], list[SkippedObservation]]:
    """Run the bounded diagnostic over ticker × entry-date grid."""
    schedule = build_weekly_schedule(input_root, sample_start, sample_end)
    entry_dates = select_entry_dates(
        schedule, sample_start, sample_end, max_entry_dates
    )

    provider = ORATSDataProvider(
        data_root=input_root,
        min_volume=0,
        min_open_interest=0,
        min_bid=0.0,
        max_spread_pct=999.0,
    )
    builder = OptionSurfaceBuilder(
        data_root=str(input_root),
        spot_db=None,  # type: ignore[arg-type]
        dte_target=WEEKLY_DTE_TARGET,
        frequency="weekly",
    )
    builder.provider = provider

    diagnosed: list[DiagnosedObservation] = []
    skipped: list[SkippedObservation] = []

    for entry_date in entry_dates:
        logger.info("Loading entry date %s (one parquet read)", entry_date.isoformat())
        snapshots, load_error = load_entry_day_snapshots(input_root, entry_date, tickers)
        if snapshots is None:
            reason = load_error or "entry_parquet_missing"
            for ticker in tickers:
                skipped.append(SkippedObservation(ticker, entry_date, reason))
            continue

        for ticker in tickers:
            result = diagnose_observation(
                builder,
                provider,
                ticker,
                entry_date,
                schedule,
                snapshot=snapshots[ticker],
            )
            if isinstance(result, SkippedObservation):
                skipped.append(result)
            else:
                diagnosed.append(result)

    return entry_dates, diagnosed, skipped


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_metrics(
    diagnosed: Sequence[DiagnosedObservation],
    skipped: Sequence[SkippedObservation],
    attempted: int,
) -> dict[str, Any]:
    """Summarize core C6.1B metrics for report rendering."""
    diagnosed_count = len(diagnosed)
    skipped_count = len(skipped)
    skip_reasons = Counter(item.reason for item in skipped)

    match_count = sum(1 for item in diagnosed if item.expiries_match)
    listed_count = sum(1 for item in diagnosed if item.target_listed_on_chain)
    call_count = sum(1 for item in diagnosed if item.target_body_call_quotable)
    put_count = sum(1 for item in diagnosed if item.target_body_put_quotable)
    pair_count = sum(1 for item in diagnosed if item.target_body_pair_quotable)
    weekly_tradable_count = sum(1 for item in diagnosed if item.weekly_tradable)
    missing_target_count = sum(1 for item in diagnosed if not item.target_listed_on_chain)
    body_not_quotable_count = sum(
        1
        for item in diagnosed
        if item.target_listed_on_chain and not item.target_body_pair_quotable
    )

    dte_deltas = [item.dte_delta for item in diagnosed if item.dte_delta is not None]
    dte_delta_hist = Counter(dte_deltas)

    mismatches = [
        item
        for item in diagnosed
        if item.expiry_chain_scanned is not None
        and item.expiry_target_weekly is not None
        and not item.expiries_match
    ]
    missing_target = [item for item in diagnosed if not item.target_listed_on_chain]
    unquotable_target_body = [
        item
        for item in diagnosed
        if item.target_listed_on_chain and not item.target_body_pair_quotable
    ]

    return {
        "attempted": attempted,
        "diagnosed_count": diagnosed_count,
        "skipped_count": skipped_count,
        "skip_reasons": skip_reasons,
        "match_rate": _rate(match_count, diagnosed_count),
        "listed_rate": _rate(listed_count, diagnosed_count),
        "call_rate": _rate(call_count, diagnosed_count),
        "put_rate": _rate(put_count, diagnosed_count),
        "pair_rate": _rate(pair_count, diagnosed_count),
        "weekly_tradable_count": weekly_tradable_count,
        "weekly_tradable_rate": _rate(weekly_tradable_count, diagnosed_count),
        "weekly_tradable_rate_of_attempted": _rate(weekly_tradable_count, attempted),
        "missing_target_count": missing_target_count,
        "missing_target_rate": _rate(missing_target_count, diagnosed_count),
        "body_not_quotable_count": body_not_quotable_count,
        "body_not_quotable_rate": _rate(body_not_quotable_count, diagnosed_count),
        # Back-compat alias used by older tests / logs
        "readiness_rate": _rate(weekly_tradable_count, diagnosed_count),
        "readiness_count": weekly_tradable_count,
        "dte_delta_hist": dte_delta_hist,
        "mismatches": mismatches,
        "missing_target": missing_target,
        "unquotable_target_body": unquotable_target_body,
    }


def classify_sample_a_verdict(metrics: dict[str, Any]) -> str:
    """Classify known-weekly Sample A as PASS / CAUTION / FAIL (mechanical gate)."""
    diagnosed_count = metrics["diagnosed_count"]
    attempted = metrics["attempted"]
    weekly_tradable_rate = metrics["weekly_tradable_rate"]

    if attempted == 0 or diagnosed_count == 0:
        return "FAIL"
    if weekly_tradable_rate is None:
        return "FAIL"

    match_rate = metrics["match_rate"] or 0.0
    if weekly_tradable_rate >= SAMPLE_A_SANITY_THRESHOLD:
        if metrics["skipped_count"] == 0 and match_rate >= SAMPLE_A_SANITY_THRESHOLD:
            return "PASS"
        return "CAUTION"
    if weekly_tradable_rate >= 0.80:
        return "CAUTION"
    return "FAIL"


def classify_verdict(
    sample_a_metrics: dict[str, Any],
    *,
    sample_b_metrics: dict[str, Any] | None = None,
) -> str:
    """Overall C6.1B verdict: Sample A gates C6.1C; Sample B never blocks."""
    sample_a = classify_sample_a_verdict(sample_a_metrics)
    if sample_a == "FAIL":
        return "FAIL"
    if sample_a == "CAUTION":
        return "CAUTION"
    # Sample B coverage is informational — low coverage does not block C6.1C.
    _ = sample_b_metrics
    return VERDICT_PASS_POLICY


def recommendation_text(
    verdict: str,
    sample_a_metrics: dict[str, Any],
    sample_b_metrics: dict[str, Any] | None = None,
) -> str:
    a_rate = sample_a_metrics.get("weekly_tradable_rate")
    a_pct = f"{100.0 * a_rate:.1f}%" if a_rate is not None else "n/a"
    b_note = ""
    if sample_b_metrics is not None:
        b_rate = sample_b_metrics.get("weekly_tradable_rate")
        b_pct = f"{100.0 * b_rate:.1f}%" if b_rate is not None else "n/a"
        b_note = (
            f" Sample B weekly-tradable rate is {b_pct} (informational coverage only; "
            "does not block C6.1C)."
        )

    if verdict == VERDICT_PASS_POLICY:
        return (
            "Proceed to C6.1C with strict calendar-paired weekly expiry. "
            f"Sample A known-weekly sanity weekly-tradable rate is {a_pct}. "
            "Ticker-weeks without the exact next-week target expiry are skipped / "
            "not weekly-ready. No fallback to nearest-DTE expiry is allowed."
            f"{b_note}"
        )
    if verdict == "CAUTION":
        return (
            "HD review required before deciding. Sample A known-weekly sanity is not clean. "
            f"Sample A weekly-tradable rate: {a_pct}.{b_note}"
        )
    return (
        "Do not proceed to C6.1C yet. Sample A known-weekly sanity failed. "
        f"Sample A weekly-tradable rate: {a_pct}.{b_note}"
    )


def _format_pct(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{100.0 * rate:.1f}%"


def _example_rows(
    observations: Sequence[DiagnosedObservation],
    limit: int = 8,
) -> list[str]:
    lines: list[str] = []
    for item in observations[:limit]:
        lines.append(
            f"- `{item.ticker}` `{item.entry_date.isoformat()}`: "
            f"chain=`{item.expiry_chain_scanned}` target=`{item.expiry_target_weekly}` "
            f"listed={item.target_listed_on_chain} body_pair={item.target_body_pair_quotable} "
            f"dte_delta={item.dte_delta}"
        )
    if len(observations) > limit:
        lines.append(f"- … and {len(observations) - limit} more")
    return lines


def _render_sample_section(sample: SampleResult, *, is_sample_a: bool) -> list[str]:
    metrics = sample.metrics
    skip_lines = [
        f"- `{reason}`: {count}"
        for reason, count in sorted(metrics["skip_reasons"].items())
    ] or ["- (none)"]
    dte_lines = [
        f"- `{delta}`: {count}"
        for delta, count in sorted(metrics["dte_delta_hist"].items())
    ] or ["- (none)"]

    title = (
        "## Sample A — known-weekly sanity check"
        if is_sample_a
        else "## Sample B — broad-universe coverage check"
    )
    purpose = (
        "The AAPL/MSFT/NVDA/SPY/QQQ sample is a known-weekly sanity sample. "
        "The result confirms mechanical viability on known weekly-option names. "
        "This is the main C6.1C mechanical gate."
        if is_sample_a
        else (
            "The broader C4 liquid-universe sample estimates how much of the universe "
            "is actually weekly-tradable. This is informational coverage, not a C6.1C "
            "correctness blocker. Missing exact target weekly expiry should be counted "
            "as expected no-trade skip behavior."
        )
    )

    lines = [
        title,
        "",
        purpose,
        "",
        f"- **Tickers ({len(sample.tickers)}):** {', '.join(sample.tickers)}",
        f"- **Sampling method:** {sample.sampling_method}",
        f"- **Observations attempted:** {metrics['attempted']}",
        f"- **Successfully diagnosed:** {metrics['diagnosed_count']}",
        f"- **Skipped (data/load errors):** {metrics['skipped_count']}",
        "",
        "**Skip reasons (data/load):**",
        *skip_lines,
        "",
        "**Resolved entry dates:**",
        f"- {', '.join(day.isoformat() for day in sample.entry_dates) if sample.entry_dates else '(none)'}",
        "",
        "### Metrics",
        "",
        f"- **`expiry_chain_scanned == expiry_target_weekly` rate:** {_format_pct(metrics['match_rate'])}",
        f"- **`target_listed_on_chain` rate:** {_format_pct(metrics['listed_rate'])}",
        f"- **Target body call quotable rate:** {_format_pct(metrics['call_rate'])}",
        f"- **Target body put quotable rate:** {_format_pct(metrics['put_rate'])}",
        f"- **Target body pair quotable rate:** {_format_pct(metrics['pair_rate'])}",
        f"- **Weekly-tradable ticker-week rate (listed + body pair):** "
        f"{_format_pct(metrics['weekly_tradable_rate'])} "
        f"({metrics['weekly_tradable_count']}/{metrics['diagnosed_count']} diagnosed)",
        f"- **Missing exact target expiry rate (among diagnosed):** "
        f"{_format_pct(metrics['missing_target_rate'])} "
        f"({metrics['missing_target_count']}/{metrics['diagnosed_count']})",
        f"- **Target listed but body pair not quotable rate:** "
        f"{_format_pct(metrics['body_not_quotable_rate'])} "
        f"({metrics['body_not_quotable_count']}/{metrics['diagnosed_count']})",
        "",
    ]

    if not is_sample_a:
        lines.extend(
            [
                "**Coverage interpretation:** Missing exact target weekly expiry means no "
                "weekly trade. Missing exact target weekly expiry is expected for "
                "non-weekly-option names. Broad coverage affects opportunity count/capacity, "
                "not correctness.",
                "",
            ]
        )

    lines.extend(
        [
            "### Mismatch / skip examples",
            "",
            "#### Expiry mismatches (chain-scanned vs target weekly)",
            "",
            *(_example_rows(metrics["mismatches"]) or ["- (none)"]),
            "",
            "#### Missing target expiry on entry chain",
            "",
            *(_example_rows(metrics["missing_target"]) or ["- (none)"]),
            "",
            "#### Target listed but body pair not quotable",
            "",
            *(_example_rows(metrics["unquotable_target_body"]) or ["- (none)"]),
            "",
            "#### DTE delta distribution (`DTE_delta = DTE_chain - DTE_target`)",
            "",
            *dte_lines,
            "",
        ]
    )
    return lines


def render_markdown_report(
    *,
    repo_commit: str,
    input_root: Path,
    sample_start: date,
    sample_end: date,
    sample_a: SampleResult,
    sample_b: SampleResult | None,
    verdict: str,
) -> str:
    """Render the C6.1B markdown acceptance artifact with Sample A/B sections."""
    recommendation = recommendation_text(
        verdict, sample_a.metrics, sample_b.metrics if sample_b else None
    )

    lines = [
        "# C6.1B — Weekly Expiry Policy Diagnostic",
        "",
        f"**C6.1B weekly expiry diagnostic: {verdict}**",
        "",
        NO_PRODUCER_CHANGE_STATEMENT,
        "",
        NO_FALLBACK_STATEMENT,
        "",
        "---",
        "",
        "## 1. Scope",
        "",
        "- **Task:** C6.1B — weekly expiry policy diagnostic (read-only)",
        f"- **Repo commit reviewed:** `{repo_commit}`",
        f"- **Data root:** `{input_root}`",
        f"- **Sample date range:** `{sample_start.isoformat()}` … `{sample_end.isoformat()}`",
        "- **Entry date generation:** `weekly_trade_dates_in_range` on adjusted-liquid parquet "
        "presence (Friday anchor with Mon–Fri walk-back)",
        "- **Non-goal:** C6.1B is read-only — no producer expiry behavior changes, no parquet/cache writes",
        "- **Policy role:** C6.1B is deciding policy semantics, not proving broad weekly-option coverage.",
        "",
        "## Policy decision for C6.1C",
        "",
        "Proceed to C6.1C with strict calendar-paired weekly expiry.",
        "",
        "For weekly strategy semantics, ticker-weeks without the exact next-week target expiry "
        "are skipped / not weekly-ready.",
        "",
        "No fallback to nearest-DTE expiry is allowed.",
        "",
        "Missing exact target weekly expiry means no weekly trade.",
        "",
        "Missing exact target weekly expiry is expected for non-weekly-option names.",
        "",
        "C6.1C must not silently substitute a nearby expiry.",
        "",
        "The old chain-scanned `_find_best_expiry` behavior is permissive and should not "
        "define weekly strategy semantics.",
        "",
        "Broad coverage affects opportunity count/capacity, not correctness.",
        "",
    ]

    lines.extend(_render_sample_section(sample_a, is_sample_a=True))

    if sample_b is not None:
        lines.extend(_render_sample_section(sample_b, is_sample_a=False))
    else:
        lines.extend(
            [
                "## Sample B — broad-universe coverage check",
                "",
                "Sample B was not run in this invocation "
                "(pass `--include-coverage-sample` to enable).",
                "",
            ]
        )

    lines.extend(
        [
            "## Correct C6.1C gate",
            "",
            "C6.1C may proceed if:",
            "",
            "1. The strict weekly expiry policy is explicitly defined.",
            "2. Known-weekly sanity sample (Sample A) passes.",
            "3. C6.1C is required to skip ticker-weeks when exact target expiry is missing.",
            "4. C6.1C is forbidden from falling back to nearest-DTE expiry.",
            "5. Missing weekly expiry is treated as expected no-trade behavior, not as a producer bug.",
            "",
            "Broad-universe coverage metrics may still be reported, but they should not block "
            "C6.1C unless the diagnostic cannot reliably distinguish missing target expiry "
            "from data errors.",
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "---",
            "",
            "## Method notes",
            "",
            "- `expiry_chain_scanned` uses current `OptionSurfaceBuilder._find_best_expiry` "
            f"with `dte_target={WEEKLY_DTE_TARGET}` for comparison only.",
            "- `expiry_target_weekly` uses `target_weekly_expiry_from_schedule(entry_date, schedule)` only.",
            "- Target body quotability matches producer rules: ATM strike nearest spot (ties -> lower); "
            "`bid > 0`, `ask > 0`, `mid > 0` on body call/put.",
            "- Entry-date parquets are read once per sample date with ticker-column pruning.",
            "- Weekly-tradable = exact target listed AND body pair quotable. Otherwise not weekly-tradable.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_sample_a_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers is not None:
        return normalize_tickers(args.tickers)

    if args.tickers_file is not None:
        tickers_path = args.tickers_file
        if not tickers_path.exists():
            raise FileNotFoundError(f"Ticker universe file not found: {tickers_path}")
        df_tickers = pd.read_csv(tickers_path)
        if "Ticker" not in df_tickers.columns:
            raise ValueError(f"Ticker column not found in {tickers_path}")
        return normalize_tickers(df_tickers["Ticker"].tolist())

    return list(DEFAULT_TICKERS)


# Back-compat alias for existing unit tests
def resolve_tickers(args: argparse.Namespace) -> list[str]:
    return resolve_sample_a_tickers(args)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C6.1B read-only weekly expiry policy diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ADJUSTED_LIQUID_ROOT,
        help="Adjusted-liquid ORATS daily parquet root",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "Sample A inline ticker scope (normalized). "
            f"Default known-weekly sample: {', '.join(DEFAULT_TICKERS)}"
        ),
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="Sample A CSV with Ticker column (mutually exclusive with --tickers)",
    )
    parser.add_argument(
        "--include-coverage-sample",
        action="store_true",
        help="Also run Sample B broad C4 coverage sample (bounded)",
    )
    parser.add_argument(
        "--coverage-tickers-file",
        type=Path,
        default=DEFAULT_LIQUID_TICKERS_PATH,
        help="C4 liquid tickers CSV for Sample B",
    )
    parser.add_argument(
        "--coverage-max-tickers",
        type=int,
        default=DEFAULT_COVERAGE_MAX_TICKERS,
        help="Max tickers in Sample B stratified sample",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=DEFAULT_START_DATE,
        help="Sample window start (inclusive)",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=DEFAULT_END_DATE,
        help="Sample window end (inclusive)",
    )
    parser.add_argument(
        "--max-entry-dates",
        type=int,
        default=DEFAULT_MAX_ENTRY_DATES,
        help="Cap weekly entry dates inside the sample window",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Markdown report path (only file written)",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Repo commit hash for report scope (defaults to git HEAD)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    if args.tickers is not None and args.tickers_file is not None:
        parser.error("--tickers and --tickers-file are mutually exclusive")
    if args.start_date > args.end_date:
        parser.error("start-date must be on or before end-date")
    if args.max_entry_dates is not None and args.max_entry_dates <= 0:
        parser.error("max-entry-dates must be positive")
    if args.coverage_max_tickers <= 0:
        parser.error("coverage-max-tickers must be positive")
    return args


def resolve_repo_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_sample_result(
    *,
    label: str,
    tickers: Sequence[str],
    entry_dates: Sequence[date],
    diagnosed: Sequence[DiagnosedObservation],
    skipped: Sequence[SkippedObservation],
    sampling_method: str,
) -> SampleResult:
    attempted = len(tickers) * len(entry_dates)
    metrics = aggregate_metrics(diagnosed, skipped, attempted)
    return SampleResult(
        label=label,
        tickers=list(tickers),
        entry_dates=list(entry_dates),
        diagnosed=list(diagnosed),
        skipped=list(skipped),
        metrics=metrics,
        sampling_method=sampling_method,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        sample_a_tickers = resolve_sample_a_tickers(args)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    input_root = Path(args.input_root)
    if not input_root.exists():
        logger.error("Input root does not exist: %s", input_root)
        return 2

    logger.info("Running Sample A (known-weekly sanity): %s", ", ".join(sample_a_tickers))
    entry_dates_a, diagnosed_a, skipped_a = run_diagnostic(
        input_root=input_root,
        tickers=sample_a_tickers,
        sample_start=args.start_date,
        sample_end=args.end_date,
        max_entry_dates=args.max_entry_dates,
    )
    sample_a = _build_sample_result(
        label="sample_a",
        tickers=sample_a_tickers,
        entry_dates=entry_dates_a,
        diagnosed=diagnosed_a,
        skipped=skipped_a,
        sampling_method="fixed known-weekly defaults (AAPL/MSFT/NVDA/SPY/QQQ) or CLI override",
    )

    sample_b: SampleResult | None = None
    if args.include_coverage_sample:
        try:
            ranked = load_ranked_coverage_tickers(args.coverage_tickers_file)
            coverage_tickers = sample_coverage_tickers(
                ranked, args.coverage_max_tickers
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error("%s", exc)
            return 2

        logger.info(
            "Running Sample B (broad coverage): %s tickers from %s",
            len(coverage_tickers),
            args.coverage_tickers_file,
        )
        entry_dates_b, diagnosed_b, skipped_b = run_diagnostic(
            input_root=input_root,
            tickers=coverage_tickers,
            sample_start=args.start_date,
            sample_end=args.end_date,
            max_entry_dates=args.max_entry_dates,
        )
        sample_b = _build_sample_result(
            label="sample_b",
            tickers=coverage_tickers,
            entry_dates=entry_dates_b,
            diagnosed=diagnosed_b,
            skipped=skipped_b,
            sampling_method=(
                f"stratified top/middle/lower by snapshots_qualified from "
                f"{args.coverage_tickers_file} (max={args.coverage_max_tickers})"
            ),
        )

    verdict = classify_verdict(
        sample_a.metrics,
        sample_b_metrics=sample_b.metrics if sample_b else None,
    )
    report = render_markdown_report(
        repo_commit=resolve_repo_commit(args.commit),
        input_root=input_root,
        sample_start=args.start_date,
        sample_end=args.end_date,
        sample_a=sample_a,
        sample_b=sample_b,
        verdict=verdict,
    )

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")

    logger.info("C6.1B verdict: %s", verdict)
    logger.info("Wrote markdown report: %s", output_report)
    logger.info(
        "Sample A weekly-tradable %s (%s/%s diagnosed)",
        _format_pct(sample_a.metrics["weekly_tradable_rate"]),
        sample_a.metrics["weekly_tradable_count"],
        sample_a.metrics["diagnosed_count"],
    )
    if sample_b is not None:
        logger.info(
            "Sample B weekly-tradable %s (%s/%s diagnosed; informational)",
            _format_pct(sample_b.metrics["weekly_tradable_rate"]),
            sample_b.metrics["weekly_tradable_count"],
            sample_b.metrics["diagnosed_count"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
