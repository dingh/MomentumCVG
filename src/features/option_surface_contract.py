"""Read-only contract checks for option surface A1/A2 parquet artifacts (C6.2).

Pure functions over pandas DataFrames so unit tests and the audit CLI can
validate producer-row semantics without mutating on-disk artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from src.data.trading_day import weekly_trade_dates_in_range
from src.features.option_surface_analyzer import DOCUMENTED_SURFACE_FAILURE_TAGS

# Consumer-required columns (docs/surface_engine_data_contract.md § A1/A2).
SURFACE_META_REQUIRED_COLUMNS = frozenset({
    "ticker",
    "entry_date",
    "expiry_date",
    "dte_actual",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "surface_valid",
    "failure_reason",
})

SURFACE_QUOTES_REQUIRED_COLUMNS = frozenset({
    "ticker",
    "entry_date",
    "expiry_date",
    "entry_spot",
    "body_strike",
    "side",
    "is_body",
    "is_otm",
    "strike",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "iv",
    "delta",
    "abs_delta",
    "gamma",
    "vega",
    "theta",
    "volume",
    "open_interest",
})

QUOTE_GRAIN_COLUMNS = (
    "ticker",
    "entry_date",
    "expiry_date",
    "strike",
    "side",
)

META_GRAIN_COLUMNS = (
    "ticker",
    "entry_date",
)

SETTLEMENT_FIELDS = (
    "expiry_date",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "dte_actual",
)


@dataclass
class ContractCheckResult:
    """Outcome of one contract check category."""

    name: str
    status: str  # PASS | WARN | FAIL
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def _to_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _normalize_key_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns and (col.endswith("_date") or col == "entry_date"):
            out[col] = out[col].map(_to_date)
    return out


def check_required_columns(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
) -> ContractCheckResult:
    """Verify required A1 and A2 columns are present."""
    missing_meta = sorted(SURFACE_META_REQUIRED_COLUMNS - set(meta_df.columns))
    missing_quotes = sorted(SURFACE_QUOTES_REQUIRED_COLUMNS - set(quotes_df.columns))
    result = ContractCheckResult(
        name="schema_checks",
        status="PASS",
        metrics={
            "meta_columns_present": len(missing_meta) == 0,
            "quotes_columns_present": len(missing_quotes) == 0,
            "missing_meta_columns": missing_meta,
            "missing_quotes_columns": missing_quotes,
        },
    )
    if missing_meta:
        result.failures.append(f"meta missing required columns: {missing_meta}")
    if missing_quotes:
        result.failures.append(f"quotes missing required columns: {missing_quotes}")
    if result.failures:
        result.status = "FAIL"
    return result


def check_surface_valid_invariant(meta_df: pd.DataFrame) -> ContractCheckResult:
    """``surface_valid == has_body_call AND has_body_put AND n_surface_quotes > 0``."""
    required = {"surface_valid", "has_body_call", "has_body_put", "n_surface_quotes"}
    if not required <= set(meta_df.columns):
        return ContractCheckResult(
            name="surface_valid_invariant",
            status="FAIL",
            failures=[f"meta missing columns for invariant check: {sorted(required - set(meta_df.columns))}"],
        )

    expected = (
        meta_df["has_body_call"].astype(bool)
        & meta_df["has_body_put"].astype(bool)
        & (meta_df["n_surface_quotes"] > 0)
    )
    violations = meta_df[meta_df["surface_valid"].astype(bool) != expected]
    result = ContractCheckResult(
        name="surface_valid_invariant",
        status="PASS",
        metrics={
            "row_count": len(meta_df),
            "violation_count": len(violations),
            "pass_count": len(meta_df) - len(violations),
        },
    )
    if not violations.empty:
        result.status = "FAIL"
        result.failures.append(
            f"{len(violations)} metadata row(s) violate the surface_valid invariant"
        )
        for _, row in violations.head(5).iterrows():
            result.examples.append(
                f"ticker={row['ticker']} entry_date={row['entry_date']} "
                f"surface_valid={row['surface_valid']} has_body_call={row['has_body_call']} "
                f"has_body_put={row['has_body_put']} n_surface_quotes={row['n_surface_quotes']}"
            )
    return result


def check_failure_vocabulary(
    meta_df: pd.DataFrame,
    *,
    known_tags: Iterable[str] | None = None,
) -> ContractCheckResult:
    """Validate ``failure_reason`` tags on invalid rows; WARN on unknown/null tags."""
    if "surface_valid" not in meta_df.columns or "failure_reason" not in meta_df.columns:
        return ContractCheckResult(
            name="failure_vocabulary",
            status="FAIL",
            failures=["meta missing surface_valid or failure_reason columns"],
        )

    tags = frozenset(known_tags) if known_tags is not None else DOCUMENTED_SURFACE_FAILURE_TAGS
    invalid = meta_df[~meta_df["surface_valid"].astype(bool)].copy()
    valid = meta_df[meta_df["surface_valid"].astype(bool)].copy()

    null_invalid = invalid[invalid["failure_reason"].isna()]
    unknown = invalid[
        invalid["failure_reason"].notna()
        & ~invalid["failure_reason"].astype(str).isin(tags)
    ]
    valid_with_reason = valid[valid["failure_reason"].notna()]

    breakdown = (
        invalid["failure_reason"]
        .fillna("<null>")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    result = ContractCheckResult(
        name="failure_vocabulary",
        status="PASS",
        metrics={
            "known_tags": sorted(tags),
            "invalid_row_count": len(invalid),
            "failure_breakdown": breakdown,
            "unknown_tag_count": len(unknown),
            "null_reason_on_invalid_count": len(null_invalid),
            "reason_on_valid_count": len(valid_with_reason),
        },
    )

    if not null_invalid.empty:
        result.status = "WARN"
        result.warnings.append(
            f"{len(null_invalid)} invalid row(s) have null failure_reason "
            "(legacy pre-C6.1D behavior; WARN only)"
        )
    if not unknown.empty:
        result.status = "WARN"
        unknown_tags = sorted(unknown["failure_reason"].astype(str).unique())
        result.warnings.append(f"unknown failure_reason tag(s): {unknown_tags}")
        for tag in unknown_tags[:5]:
            result.examples.append(f"unknown_failure_reason:{tag}")
    if not valid_with_reason.empty:
        result.status = "WARN"
        result.warnings.append(
            f"{len(valid_with_reason)} surface_valid=True row(s) carry non-null failure_reason"
        )
    return result


def check_settlement_fields(meta_df: pd.DataFrame) -> ContractCheckResult:
    """Assert settlement fields on valid rows and ``dte_actual`` consistency."""
    needed = {"surface_valid", "entry_date", *SETTLEMENT_FIELDS}
    if not needed <= set(meta_df.columns):
        missing = sorted(needed - set(meta_df.columns))
        return ContractCheckResult(
            name="settlement_readiness",
            status="FAIL",
            failures=[f"meta missing columns for settlement check: {missing}"],
        )

    valid = _normalize_key_columns(meta_df, ("entry_date", "expiry_date"))
    valid = valid[valid["surface_valid"].astype(bool)]

    null_field_counts: dict[str, int] = {}
    for fld in SETTLEMENT_FIELDS:
        null_field_counts[f"null_{fld}"] = int(valid[fld].isna().sum())

    dte_mismatch = pd.DataFrame()
    if not valid.empty:
        computed = valid.apply(
            lambda row: (
                (_to_date(row["expiry_date"]) - _to_date(row["entry_date"])).days
                if _to_date(row["expiry_date"]) is not None and _to_date(row["entry_date"]) is not None
                else None
            ),
            axis=1,
        )
        dte_mismatch = valid[
            valid["dte_actual"].notna()
            & computed.notna()
            & (valid["dte_actual"] != computed)
        ]

    result = ContractCheckResult(
        name="settlement_readiness",
        status="PASS",
        metrics={
            "valid_row_count": len(valid),
            **null_field_counts,
            "dte_mismatch_count": len(dte_mismatch),
        },
    )

    for fld, count in null_field_counts.items():
        if count:
            result.status = "FAIL"
            result.failures.append(f"{count} valid row(s) missing {fld.removeprefix('null_')}")

    if not dte_mismatch.empty:
        result.status = "FAIL"
        result.failures.append(f"{len(dte_mismatch)} valid row(s) have dte_actual mismatch")
        for _, row in dte_mismatch.head(5).iterrows():
            result.examples.append(
                f"ticker={row['ticker']} entry_date={row['entry_date']} "
                f"dte_actual={row['dte_actual']} expiry_date={row['expiry_date']}"
            )
    return result


def check_a1_a2_join(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
) -> ContractCheckResult:
    """Detect orphan quotes and valid metadata rows without matching quote rows."""
    key_cols = ("ticker", "entry_date", "expiry_date")
    for col in ("surface_valid", *key_cols):
        if col not in meta_df.columns:
            return ContractCheckResult(
                name="a1_a2_join_integrity",
                status="FAIL",
                failures=[f"meta missing required column: {col}"],
            )
    for col in key_cols:
        if col not in quotes_df.columns:
            return ContractCheckResult(
                name="a1_a2_join_integrity",
                status="FAIL",
                failures=[f"quotes missing required column: {col}"],
            )

    meta = _normalize_key_columns(meta_df, key_cols)
    quotes = _normalize_key_columns(quotes_df, key_cols)

    meta_keys = {
        tuple(row)
        for row in meta[list(key_cols)].itertuples(index=False, name=None)
        if all(v is not None and not (isinstance(v, float) and pd.isna(v)) for v in row)
    }
    quote_keys = list(quotes[list(key_cols)].itertuples(index=False, name=None))

    orphan_count = 0
    orphan_examples: list[str] = []
    for key in quote_keys:
        if key not in meta_keys:
            orphan_count += 1
            if len(orphan_examples) < 5:
                orphan_examples.append(
                    f"orphan quote: ticker={key[0]} entry_date={key[1]} expiry_date={key[2]}"
                )

    quote_key_set = {
        key for key in quote_keys
        if all(v is not None and not (isinstance(v, float) and pd.isna(v)) for v in key)
    }

    valid_meta = meta[meta["surface_valid"].astype(bool)]
    valid_without_quotes = 0
    valid_missing_examples: list[str] = []
    for _, row in valid_meta.iterrows():
        key = (row["ticker"], row["entry_date"], row["expiry_date"])
        if key not in quote_key_set:
            valid_without_quotes += 1
            if len(valid_missing_examples) < 5:
                valid_missing_examples.append(
                    f"valid meta without quotes: ticker={key[0]} entry_date={key[1]} expiry_date={key[2]}"
                )

    invalid_with_quotes = 0
    invalid_quote_examples: list[str] = []
    invalid_meta = meta[~meta["surface_valid"].astype(bool)]
    for _, row in invalid_meta.iterrows():
        key = (row["ticker"], row["entry_date"], row["expiry_date"])
        if key in quote_key_set:
            invalid_with_quotes += 1
            if len(invalid_quote_examples) < 5:
                invalid_quote_examples.append(
                    f"invalid meta with quotes (informational): ticker={key[0]} "
                    f"entry_date={key[1]} expiry_date={key[2]}"
                )

    result = ContractCheckResult(
        name="a1_a2_join_integrity",
        status="PASS",
        metrics={
            "meta_key_count": len(meta_keys),
            "quote_row_count": len(quotes_df),
            "orphan_quote_count": orphan_count,
            "valid_meta_without_quotes_count": valid_without_quotes,
            "invalid_meta_with_quotes_count": invalid_with_quotes,
        },
        examples=orphan_examples + valid_missing_examples,
    )

    if orphan_count:
        result.status = "FAIL"
        result.failures.append(f"{orphan_count} orphan quote row(s) lack matching metadata")
    if valid_without_quotes:
        result.status = "FAIL"
        result.failures.append(
            f"{valid_without_quotes} surface_valid=True metadata row(s) have zero quote rows"
        )
    if invalid_with_quotes:
        result.warnings.append(
            f"{invalid_with_quotes} surface_valid=False metadata row(s) have quote rows "
            "(allowed partial-surface behavior; informational WARN)"
        )
        if result.status == "PASS":
            result.status = "WARN"
        result.examples.extend(invalid_quote_examples)
    return result


def check_meta_grain(meta_df: pd.DataFrame) -> ContractCheckResult:
    """Detect duplicate metadata rows at the A1 grain (one row per ticker-entry)."""
    missing = [col for col in META_GRAIN_COLUMNS if col not in meta_df.columns]
    if missing:
        return ContractCheckResult(
            name="meta_grain",
            status="FAIL",
            failures=[f"meta missing grain columns: {missing}"],
        )

    meta = _normalize_key_columns(meta_df, META_GRAIN_COLUMNS)
    grain = meta[list(META_GRAIN_COLUMNS)]
    dup_mask = grain.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    unique_violations = int(grain.duplicated(keep="first").sum())

    result = ContractCheckResult(
        name="meta_grain",
        status="PASS",
        metrics={
            "meta_row_count": len(meta_df),
            "duplicate_row_count": dup_count,
            "duplicate_key_count": unique_violations,
            "grain": list(META_GRAIN_COLUMNS),
        },
    )
    if unique_violations:
        result.status = "FAIL"
        result.failures.append(
            f"{unique_violations} duplicate metadata key(s) at grain {list(META_GRAIN_COLUMNS)}"
        )
        for _, row in grain[grain.duplicated(keep="first")].head(5).iterrows():
            result.examples.append(
                f"duplicate: ticker={row['ticker']} entry_date={row['entry_date']}"
            )
    return result


def check_quote_grain(quotes_df: pd.DataFrame) -> ContractCheckResult:
    """Detect duplicate quote rows at the stable A2 grain."""
    missing = [col for col in QUOTE_GRAIN_COLUMNS if col not in quotes_df.columns]
    if missing:
        return ContractCheckResult(
            name="quote_grain",
            status="FAIL",
            failures=[f"quotes missing grain columns: {missing}"],
        )

    grain = quotes_df[list(QUOTE_GRAIN_COLUMNS)]
    dup_mask = grain.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    unique_violations = int(grain.duplicated(keep="first").sum())

    result = ContractCheckResult(
        name="quote_grain",
        status="PASS",
        metrics={
            "quote_row_count": len(quotes_df),
            "duplicate_row_count": dup_count,
            "duplicate_key_count": unique_violations,
            "grain": list(QUOTE_GRAIN_COLUMNS),
        },
    )
    if unique_violations:
        result.status = "FAIL"
        result.failures.append(
            f"{unique_violations} duplicate quote key(s) at grain {list(QUOTE_GRAIN_COLUMNS)}"
        )
        for _, row in grain[grain.duplicated(keep="first")].head(5).iterrows():
            result.examples.append(
                f"duplicate: ticker={row['ticker']} entry_date={row['entry_date']} "
                f"expiry_date={row['expiry_date']} strike={row['strike']} side={row['side']}"
            )
    return result


def check_weekly_date_alignment(
    meta_df: pd.DataFrame,
    *,
    frequency: str,
    data_root: Path | str,
    start_date: date,
    end_date: date,
    schedule_fn: Callable[..., list[date]] = weekly_trade_dates_in_range,
) -> ContractCheckResult:
    """Verify weekly ``entry_date`` values lie on the resolved weekly schedule."""
    if frequency != "weekly":
        return ContractCheckResult(
            name="date_alignment",
            status="PASS",
            metrics={"skipped": True, "reason": f"frequency={frequency!r} is not weekly"},
        )
    if "entry_date" not in meta_df.columns:
        return ContractCheckResult(
            name="date_alignment",
            status="FAIL",
            failures=["meta missing entry_date column"],
        )

    schedule = set(schedule_fn(start_date, end_date, data_root))
    entry_dates = {
        d for d in (_to_date(v) for v in meta_df["entry_date"].unique()) if d is not None
    }
    misaligned = sorted(entry_dates - schedule)

    result = ContractCheckResult(
        name="date_alignment",
        status="PASS",
        metrics={
            "schedule_entry_count": len(schedule),
            "artifact_entry_count": len(entry_dates),
            "misaligned_entry_count": len(misaligned),
            "policy": "WARN on misaligned entry_date (legacy pre-C6.1C artifacts tolerated)",
        },
    )
    if misaligned:
        result.status = "WARN"
        result.warnings.append(
            f"{len(misaligned)} entry_date value(s) outside resolved weekly schedule"
        )
        for entry_day in misaligned[:10]:
            result.examples.append(f"misaligned entry_date: {entry_day.isoformat()}")
    return result


def compute_overall_verdict(results: Sequence[ContractCheckResult]) -> str:
    """Aggregate category statuses into PASS / WARN / FAIL."""
    if any(r.status == "FAIL" for r in results):
        return "FAIL"
    if any(r.status == "WARN" for r in results):
        return "WARN"
    return "PASS"


def run_contract_checks(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    frequency: str,
    data_root: Path | str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[ContractCheckResult]:
    """Run the full C6.2 contract check suite on in-memory artifacts."""
    results = [
        check_required_columns(meta_df, quotes_df),
        check_surface_valid_invariant(meta_df),
        check_failure_vocabulary(meta_df),
        check_settlement_fields(meta_df),
        check_meta_grain(meta_df),
        check_a1_a2_join(meta_df, quotes_df),
        check_quote_grain(quotes_df),
    ]
    if frequency == "weekly" and data_root is not None and start_date and end_date:
        results.append(
            check_weekly_date_alignment(
                meta_df,
                frequency=frequency,
                data_root=data_root,
                start_date=start_date,
                end_date=end_date,
            )
        )
    return results


def filter_artifacts(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sample_tickers: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply bounded audit filters without mutating source files."""
    meta = meta_df.copy()
    quotes = quotes_df.copy()

    if sample_tickers:
        tickers = {t.upper() for t in sample_tickers}
        meta = meta[meta["ticker"].astype(str).str.upper().isin(tickers)]
        quotes = quotes[quotes["ticker"].astype(str).str.upper().isin(tickers)]

    if start_date is not None:
        meta = meta[meta["entry_date"].map(_to_date) >= start_date]
        quotes = quotes[quotes["entry_date"].map(_to_date) >= start_date]
    if end_date is not None:
        meta = meta[meta["entry_date"].map(_to_date) <= end_date]
        quotes = quotes[quotes["entry_date"].map(_to_date) <= end_date]

    if max_rows is not None and len(meta) > max_rows:
        meta = meta.head(max_rows).copy()
        keep_keys = set(
            zip(
                meta["ticker"].astype(str),
                meta["entry_date"].map(_to_date),
                strict=False,
            )
        )
        quote_keys = list(
            zip(
                quotes["ticker"].astype(str),
                quotes["entry_date"].map(_to_date),
                strict=False,
            )
        )
        quotes = quotes[[key in keep_keys for key in quote_keys]].copy()

    return meta.reset_index(drop=True), quotes.reset_index(drop=True)
