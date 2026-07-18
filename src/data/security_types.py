"""Persistent ORATS security-type dictionary (ticker classification cache).

Maintains one durable Parquet dictionary with one deterministic row per
normalized ticker so ORATS Core history is fetched at most once per ticker
across liquidity backfills. Consumed by the C4 liquidity build to filter
candidate daily liquidity data to company equities before weekly / panel /
rank construction.

Classification policy (window-independent, persists across backfills)
---------------------------------------------------------------------
* Historical ORATS Core ``assetType`` is the single source (``orats_core``).
* Types 0, 1, 2, 3 classify as ``company_equity``.
* Any valid historical record with type 4-9 classifies the ticker as
  ``non_company_equity`` (mixed 0-3 / 4-9 history resolves non-company).
* Missing history, malformed values, values outside 0-9, unexpected tickers,
  or conflicting duplicate ``(ticker, tradeDate)`` records fail the update.
* Existing dictionary entries are authoritative and never re-fetched or
  altered by a missing-ticker append.

Cache behaviour
---------------
* Dictionary absent: fetch every candidate, validate all results, write the
  complete dictionary atomically.
* Dictionary present: load and validate it; fetch only
  ``candidates - dictionary``; when nothing is missing make no request and do
  not rewrite the file; otherwise validate the complete batch, merge
  deterministically, and atomically replace via temp-file + ``os.replace``.
* Any fetch or validation failure leaves the previously accepted dictionary
  byte-for-byte unchanged.

A future policy change must bump ``classification_version`` and perform an
explicit rebuild; there is no refresh, expiration, or watermark logic here.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from src.data.snapshot_foundation import digest_json

logger = logging.getLogger(__name__)

CLASSIFICATION_COMPANY_EQUITY = "company_equity"
CLASSIFICATION_NON_COMPANY_EQUITY = "non_company_equity"
_VALID_CLASSIFICATIONS = (
    CLASSIFICATION_COMPANY_EQUITY,
    CLASSIFICATION_NON_COMPANY_EQUITY,
)

SOURCE_ORATS_CORE = "orats_core"
CLASSIFICATION_VERSION = 1

COMPANY_EQUITY_ASSET_TYPES = frozenset({0, 1, 2, 3})
NON_COMPANY_ASSET_TYPES = frozenset({4, 5, 6, 7, 8, 9})
_VALID_ASSET_TYPES = COMPANY_EQUITY_ASSET_TYPES | NON_COMPANY_ASSET_TYPES

SECURITY_TYPES_COLUMNS = (
    "ticker",
    "classification",
    "observed_asset_types",
    "source",
    "source_date_min",
    "source_date_max",
    "classified_at_utc",
    "classification_version",
)

_HISTORY_REQUIRED_COLUMNS = ("ticker", "tradeDate", "assetType")


class SecurityTypesError(Exception):
    """Blocking failure classifying tickers or updating the dictionary."""


# ── normalization ─────────────────────────────────────────────────────────────


def normalize_ticker(raw: object) -> str:
    """Return the normalized (stripped, upper-cased) ticker or fail."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise SecurityTypesError(f"Ticker is missing/null: {raw!r}")
    ticker = str(raw).strip().upper()
    if not ticker:
        raise SecurityTypesError(f"Ticker is blank after normalization: {raw!r}")
    return ticker


def _encode_observed_asset_types(types: Iterable[int]) -> str:
    """Deterministic storage form: comma-joined sorted unique int types."""
    return ",".join(str(t) for t in sorted(set(types)))


def _decode_observed_asset_types(encoded: object) -> set[int]:
    """Parse the stored ``observed_asset_types`` string; fail on malformed."""
    if not isinstance(encoded, str) or not encoded:
        raise SecurityTypesError(
            f"observed_asset_types must be a nonempty string; got {encoded!r}"
        )
    try:
        types = {int(part) for part in encoded.split(",")}
    except ValueError as exc:
        raise SecurityTypesError(
            f"observed_asset_types is malformed: {encoded!r}"
        ) from exc
    return types


# ── per-ticker classification ─────────────────────────────────────────────────


def classify_asset_type_history(
    ticker: str,
    history: pd.DataFrame,
    *,
    classified_at_utc: str,
) -> dict:
    """Classify one ticker from its historical Core ``assetType`` records.

    Returns one dictionary-row dict. Fails closed on missing history,
    malformed / out-of-domain values, unexpected tickers, or conflicting
    duplicate ``(ticker, tradeDate)`` records.
    """
    expected = normalize_ticker(ticker)

    if history is None or history.empty:
        raise SecurityTypesError(
            f"No historical ORATS Core assetType records for ticker {expected}; "
            "cannot classify (missing history fails the update)."
        )
    missing_cols = [c for c in _HISTORY_REQUIRED_COLUMNS if c not in history.columns]
    if missing_cols:
        raise SecurityTypesError(
            f"Core history for ticker {expected} missing columns {missing_cols}."
        )

    rows = history.copy()

    rows["ticker"] = [normalize_ticker(t) for t in rows["ticker"]]
    unexpected = sorted(set(rows["ticker"]) - {expected})
    if unexpected:
        raise SecurityTypesError(
            f"Core history for ticker {expected} contains unexpected "
            f"ticker(s) {unexpected}."
        )

    trade_dates = pd.to_datetime(rows["tradeDate"], errors="coerce")
    if trade_dates.isna().any():
        bad = rows.loc[trade_dates.isna(), "tradeDate"].head(3).tolist()
        raise SecurityTypesError(
            f"Core history for ticker {expected} has unparseable tradeDate "
            f"value(s): {bad}."
        )
    rows["tradeDate"] = trade_dates.dt.date

    asset_types = pd.to_numeric(rows["assetType"], errors="coerce")
    if asset_types.isna().any():
        bad = rows.loc[asset_types.isna(), "assetType"].head(3).tolist()
        raise SecurityTypesError(
            f"Core history for ticker {expected} has missing/malformed "
            f"assetType value(s): {bad}."
        )
    if not (asset_types == asset_types.astype(int)).all():
        bad = asset_types[asset_types != asset_types.astype(int)].head(3).tolist()
        raise SecurityTypesError(
            f"Core history for ticker {expected} has non-integer assetType "
            f"value(s): {bad}."
        )
    rows["assetType"] = asset_types.astype(int)

    out_of_domain = sorted(set(rows["assetType"]) - _VALID_ASSET_TYPES)
    if out_of_domain:
        raise SecurityTypesError(
            f"Core history for ticker {expected} has assetType value(s) "
            f"outside 0-9: {out_of_domain}."
        )

    deduped = rows.drop_duplicates(subset=["ticker", "tradeDate", "assetType"])
    conflict_mask = deduped.duplicated(subset=["ticker", "tradeDate"], keep=False)
    if conflict_mask.any():
        examples = (
            deduped.loc[conflict_mask, ["tradeDate", "assetType"]]
            .head(4)
            .to_dict("records")
        )
        raise SecurityTypesError(
            f"Core history for ticker {expected} has conflicting duplicate "
            f"(ticker, tradeDate) records: {examples}."
        )

    observed = set(deduped["assetType"])
    if observed & NON_COMPANY_ASSET_TYPES:
        classification = CLASSIFICATION_NON_COMPANY_EQUITY
    else:
        classification = CLASSIFICATION_COMPANY_EQUITY

    return {
        "ticker": expected,
        "classification": classification,
        "observed_asset_types": _encode_observed_asset_types(observed),
        "source": SOURCE_ORATS_CORE,
        "source_date_min": min(deduped["tradeDate"]).isoformat(),
        "source_date_max": max(deduped["tradeDate"]).isoformat(),
        "classified_at_utc": classified_at_utc,
        "classification_version": CLASSIFICATION_VERSION,
    }


# ── dictionary schema validation ──────────────────────────────────────────────


def validate_security_types(df: pd.DataFrame) -> None:
    """Explicitly validate the complete dictionary schema and invariants."""
    missing = [c for c in SECURITY_TYPES_COLUMNS if c not in df.columns]
    if missing:
        raise SecurityTypesError(
            f"Security-type dictionary missing columns {missing}; "
            f"expected {list(SECURITY_TYPES_COLUMNS)}."
        )

    tickers = [normalize_ticker(t) for t in df["ticker"]]
    if tickers != list(df["ticker"]):
        raise SecurityTypesError(
            "Security-type dictionary contains non-normalized ticker values."
        )
    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise SecurityTypesError(
            f"Security-type dictionary has duplicate ticker rows: {dupes[:5]}."
        )

    bad_class = sorted(set(df["classification"]) - set(_VALID_CLASSIFICATIONS))
    if bad_class:
        raise SecurityTypesError(
            f"Security-type dictionary has invalid classification(s): {bad_class}."
        )

    bad_source = sorted(set(df["source"]) - {SOURCE_ORATS_CORE})
    if bad_source:
        raise SecurityTypesError(
            f"Security-type dictionary has invalid source value(s): {bad_source}."
        )

    for row in df.itertuples(index=False):
        observed = _decode_observed_asset_types(row.observed_asset_types)
        out_of_domain = sorted(observed - _VALID_ASSET_TYPES)
        if out_of_domain:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: observed_asset_types outside 0-9: "
                f"{out_of_domain}."
            )
        expected_class = (
            CLASSIFICATION_NON_COMPANY_EQUITY
            if observed & NON_COMPANY_ASSET_TYPES
            else CLASSIFICATION_COMPANY_EQUITY
        )
        if row.classification != expected_class:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: classification {row.classification!r} "
                f"inconsistent with observed_asset_types "
                f"{row.observed_asset_types!r}."
            )
        for field in ("source_date_min", "source_date_max"):
            value = getattr(row, field)
            try:
                datetime.strptime(str(value), "%Y-%m-%d")
            except ValueError as exc:
                raise SecurityTypesError(
                    f"Ticker {row.ticker}: {field} is not an ISO date: {value!r}."
                ) from exc
        if str(row.source_date_min) > str(row.source_date_max):
            raise SecurityTypesError(
                f"Ticker {row.ticker}: source_date_min > source_date_max."
            )
        version = row.classification_version
        try:
            version_int = int(version)
        except (TypeError, ValueError) as exc:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: invalid classification_version {version!r}."
            ) from exc
        if version_int != version or version_int < 1:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: invalid classification_version {version!r}."
            )


def load_security_types(path: Path | str) -> pd.DataFrame:
    """Load and validate the complete existing dictionary."""
    file_path = Path(path)
    if not file_path.is_file():
        raise SecurityTypesError(f"Security-type dictionary not found: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        raise SecurityTypesError(
            f"Failed to read security-type dictionary {file_path}: {exc}"
        ) from exc
    validate_security_types(df)
    return df


# ── atomic write ──────────────────────────────────────────────────────────────


def write_security_types_atomic(df: pd.DataFrame, path: Path | str) -> None:
    """Atomically write the dictionary (temp file + readback + ``os.replace``).

    Any failure before the final replace leaves an existing target
    byte-for-byte unchanged; a partially written dictionary is never published.
    """
    validate_security_types(df)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target.parent / f"{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        df.to_parquet(temp_path, index=False)
        readback = pd.read_parquet(temp_path)
        validate_security_types(readback)
        os.replace(temp_path, target)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ── dictionary update (cache behaviour) ───────────────────────────────────────


def _sorted_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("ticker", kind="mergesort").reset_index(drop=True)


def _classify_batch(
    tickers: Sequence[str],
    fetch_history_fn: Callable[[str], pd.DataFrame],
    *,
    now_fn: Callable[[], datetime],
) -> pd.DataFrame:
    """Fetch + classify every ticker; any single failure fails the batch."""
    classified_at = (
        now_fn().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    rows = [
        classify_asset_type_history(
            ticker,
            fetch_history_fn(ticker),
            classified_at_utc=classified_at,
        )
        for ticker in sorted(tickers)
    ]
    return pd.DataFrame(rows, columns=list(SECURITY_TYPES_COLUMNS))


def ensure_security_types(
    candidate_tickers: Iterable[str],
    path: Path | str,
    *,
    fetch_history_fn: Callable[[str], pd.DataFrame],
    now_fn: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    """Ensure the dictionary covers every candidate ticker; return it.

    Fetches ORATS Core history (via ``fetch_history_fn``) only for candidate
    tickers absent from the dictionary. See module docstring for the exact
    cache behaviour contract.
    """
    if now_fn is None:
        now_fn = lambda: datetime.now(timezone.utc)  # noqa: E731

    candidates = {normalize_ticker(t) for t in candidate_tickers}
    if not candidates:
        raise SecurityTypesError("Candidate ticker set is empty.")

    file_path = Path(path)
    if not file_path.is_file():
        logger.info(
            "Security-type dictionary absent at %s — classifying %d candidate ticker(s).",
            file_path, len(candidates),
        )
        dictionary = _sorted_dictionary(
            _classify_batch(sorted(candidates), fetch_history_fn, now_fn=now_fn)
        )
        write_security_types_atomic(dictionary, file_path)
        return dictionary

    existing = load_security_types(file_path)
    missing = sorted(candidates - set(existing["ticker"]))
    if not missing:
        logger.info(
            "Security-type dictionary covers all %d candidate ticker(s); "
            "no fetch, no rewrite.",
            len(candidates),
        )
        return existing

    logger.info(
        "Security-type dictionary missing %d of %d candidate ticker(s) — "
        "fetching only the missing set.",
        len(missing), len(candidates),
    )
    new_rows = _classify_batch(missing, fetch_history_fn, now_fn=now_fn)

    # Existing classifications are authoritative: append-only merge, then a
    # deterministic sort. Existing rows are never altered.
    merged = _sorted_dictionary(
        pd.concat([existing, new_rows], ignore_index=True)
    )
    write_security_types_atomic(merged, file_path)
    return merged


# ── snapshot-local subset and digest ──────────────────────────────────────────


def snapshot_classification(
    dictionary: pd.DataFrame,
    candidate_tickers: Iterable[str],
) -> pd.DataFrame:
    """Return the snapshot-local subset for exactly the candidate tickers.

    This subset is the immutable evidence consumed by one liquidity build;
    the shared dictionary remains the reusable cache.
    """
    candidates = {normalize_ticker(t) for t in candidate_tickers}
    covered = set(dictionary["ticker"])
    uncovered = sorted(candidates - covered)
    if uncovered:
        raise SecurityTypesError(
            f"Candidate ticker(s) missing from security-type dictionary: "
            f"{uncovered[:10]}."
        )
    subset = dictionary[dictionary["ticker"].isin(candidates)]
    return _sorted_dictionary(subset)


def classification_digest(classification: pd.DataFrame) -> str:
    """Deterministic content digest of a classification frame.

    Digest of the sorted ``[ticker, classification, observed_asset_types]``
    triples only — retrieval timestamps never contribute.
    """
    triples = sorted(
        [str(row.ticker), str(row.classification), str(row.observed_asset_types)]
        for row in classification.itertuples(index=False)
    )
    return digest_json(triples)


def company_equity_tickers(classification: pd.DataFrame) -> set[str]:
    """Tickers classified as company equities in a (validated) frame."""
    mask = classification["classification"] == CLASSIFICATION_COMPANY_EQUITY
    return set(classification.loc[mask, "ticker"])
