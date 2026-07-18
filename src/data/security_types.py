"""Persistent ORATS security-type dictionary (ticker classification cache).

Maintains one durable Parquet dictionary with one deterministic row per
normalized ticker so ORATS Core is queried at most once per ticker across
liquidity backfills. Consumed by the C4 liquidity build to filter candidate
daily liquidity data to company equities before weekly / panel / rank
construction.

Classification policy (version 2 — date-specific observation)
--------------------------------------------------------------
* A ticker is classified from **one** Core ``assetType`` observation fetched
  for a specific ``(ticker, tradeDate)`` pair — never from unrestricted full
  history.
* The requested date is the ticker's **latest observed trade date** in the
  candidate daily liquidity data. If that date returns a valid empty
  response, up to :data:`MAX_OBSERVED_DATE_ATTEMPTS` - 1 additional observed
  dates are attempted, newest first — only dates on which the ticker actually
  appeared in daily observations. Exhausting the bounded attempts fails.
* The accepted observation classifies the ticker:
  ``assetType`` 0-3 → ``company_equity``; 4-9 → ``non_company_equity``.
* Returned ticker and trade date are validated against the request.
  Identical duplicate rows are deduplicated; conflicting rows, unexpected
  ticker/date values, malformed or out-of-domain ``assetType`` values, and
  HTTP/parse failures fail the update.
* Existing dictionary entries are authoritative and never re-fetched or
  altered by a missing-ticker append.

Cache behaviour
---------------
* Dictionary absent: classify every candidate, validate all results, write
  the complete dictionary atomically.
* Dictionary present: load and validate it; classify only
  ``candidates - dictionary``; when nothing is missing make no request and do
  not rewrite the file; otherwise validate the complete batch, merge
  deterministically, and atomically replace via temp-file + ``os.replace``.
* Any fetch or validation failure leaves the previously accepted dictionary
  byte-for-byte unchanged.

Versioning
----------
``CLASSIFICATION_VERSION`` is 2. Validation requires the current version
exactly: a dictionary produced under the version-1 policy (full-history,
any-type-wins) is rejected with an explicit rebuild instruction. Versions are
never silently mixed and external data is never deleted by this module.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

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
CLASSIFICATION_VERSION = 2

# Latest observed date plus up to four fallback observed dates.
MAX_OBSERVED_DATE_ATTEMPTS = 5

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

_OBSERVATION_REQUIRED_COLUMNS = ("ticker", "tradeDate", "assetType")


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


def _normalize_observed_dates(raw_dates: Iterable[object]) -> list[date]:
    """Return distinct observed trade dates sorted newest first, or fail."""
    normalized: set[date] = set()
    for raw in raw_dates:
        if isinstance(raw, datetime):
            normalized.add(raw.date())
        elif isinstance(raw, date):
            normalized.add(raw)
        else:
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed):
                raise SecurityTypesError(
                    f"Unparseable observed trade date: {raw!r}"
                )
            normalized.add(parsed.date())
    if not normalized:
        raise SecurityTypesError("Observed trade-date list is empty.")
    return sorted(normalized, reverse=True)


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


def _classification_for_type(asset_type: int) -> str:
    if asset_type in NON_COMPANY_ASSET_TYPES:
        return CLASSIFICATION_NON_COMPANY_EQUITY
    return CLASSIFICATION_COMPANY_EQUITY


# ── per-observation validation and per-ticker classification ─────────────────


def accept_asset_type_observation(
    ticker: str,
    trade_date: date,
    frame: pd.DataFrame,
) -> int:
    """Validate one date-specific Core response and return its assetType.

    The response must be non-empty and consistent with the request: every row
    must carry the requested normalized ticker and the requested trade date.
    Identical duplicate rows are deduplicated; anything else (conflicting
    rows, unexpected ticker/date, malformed or out-of-domain values) fails.
    """
    expected = normalize_ticker(ticker)
    if frame is None or frame.empty:
        raise SecurityTypesError(
            f"Core observation for ticker {expected} on {trade_date.isoformat()} "
            "is empty; caller must treat valid-empty via fallback, not here."
        )
    missing_cols = [c for c in _OBSERVATION_REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise SecurityTypesError(
            f"Core observation for ticker {expected} missing columns {missing_cols}."
        )

    rows = frame.copy()

    rows["ticker"] = [normalize_ticker(t) for t in rows["ticker"]]
    unexpected = sorted(set(rows["ticker"]) - {expected})
    if unexpected:
        raise SecurityTypesError(
            f"Core observation for ticker {expected} contains unexpected "
            f"ticker(s) {unexpected}."
        )

    parsed_dates = pd.to_datetime(rows["tradeDate"], errors="coerce")
    if parsed_dates.isna().any():
        bad = rows.loc[parsed_dates.isna(), "tradeDate"].head(3).tolist()
        raise SecurityTypesError(
            f"Core observation for ticker {expected} has unparseable tradeDate "
            f"value(s): {bad}."
        )
    rows["tradeDate"] = parsed_dates.dt.date
    wrong_dates = sorted({d for d in rows["tradeDate"] if d != trade_date})
    if wrong_dates:
        raise SecurityTypesError(
            f"Core observation for ticker {expected} returned tradeDate(s) "
            f"{[d.isoformat() for d in wrong_dates]} that do not match the "
            f"requested date {trade_date.isoformat()}."
        )

    asset_types = pd.to_numeric(rows["assetType"], errors="coerce")
    if asset_types.isna().any():
        bad = rows.loc[asset_types.isna(), "assetType"].head(3).tolist()
        raise SecurityTypesError(
            f"Core observation for ticker {expected} has missing/malformed "
            f"assetType value(s): {bad}."
        )
    if not (asset_types == asset_types.astype(int)).all():
        bad = asset_types[asset_types != asset_types.astype(int)].head(3).tolist()
        raise SecurityTypesError(
            f"Core observation for ticker {expected} has non-integer assetType "
            f"value(s): {bad}."
        )
    rows["assetType"] = asset_types.astype(int)

    out_of_domain = sorted(set(rows["assetType"]) - _VALID_ASSET_TYPES)
    if out_of_domain:
        raise SecurityTypesError(
            f"Core observation for ticker {expected} has assetType value(s) "
            f"outside 0-9: {out_of_domain}."
        )

    deduped = rows.drop_duplicates(subset=["ticker", "tradeDate", "assetType"])
    if len(deduped) > 1:
        examples = deduped[["tradeDate", "assetType"]].head(4).to_dict("records")
        raise SecurityTypesError(
            f"Core observation for ticker {expected} on {trade_date.isoformat()} "
            f"has conflicting duplicate records: {examples}."
        )

    return int(deduped["assetType"].iloc[0])


def classify_ticker_with_fallback(
    ticker: str,
    observed_dates: Sequence[object],
    fetch_observation_fn: Callable[[str, date], pd.DataFrame],
    *,
    classified_at_utc: str,
) -> dict:
    """Classify one ticker from its latest successfully retrieved observed date.

    Attempts the ticker's latest observed trade date first. A valid empty
    response falls back to the next-newest observed date, up to
    :data:`MAX_OBSERVED_DATE_ATTEMPTS` total attempts. Any HTTP/parse/
    validation failure propagates immediately; exhausting the bounded
    attempts without a row fails classification.
    """
    expected = normalize_ticker(ticker)
    dates = _normalize_observed_dates(observed_dates)
    attempts = dates[:MAX_OBSERVED_DATE_ATTEMPTS]

    for trade_date in attempts:
        frame = fetch_observation_fn(expected, trade_date)
        if frame is None:
            raise SecurityTypesError(
                f"Core fetch for ticker {expected} on {trade_date.isoformat()} "
                "returned None; expected a DataFrame."
            )
        if frame.empty:
            # Valid-empty: no Core record on this observed date — fall back.
            continue
        asset_type = accept_asset_type_observation(expected, trade_date, frame)
        return {
            "ticker": expected,
            "classification": _classification_for_type(asset_type),
            "observed_asset_types": _encode_observed_asset_types([asset_type]),
            "source": SOURCE_ORATS_CORE,
            "source_date_min": trade_date.isoformat(),
            "source_date_max": trade_date.isoformat(),
            "classified_at_utc": classified_at_utc,
            "classification_version": CLASSIFICATION_VERSION,
        }

    raise SecurityTypesError(
        f"No Core assetType observation for ticker {expected} on any of "
        f"{len(attempts)} attempted observed date(s) "
        f"{[d.isoformat() for d in attempts]}; cannot classify."
    )


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
        version = row.classification_version
        try:
            version_int = int(version)
        except (TypeError, ValueError) as exc:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: invalid classification_version {version!r}."
            ) from exc
        if version_int != version or version_int != CLASSIFICATION_VERSION:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: classification_version {version!r} != "
                f"required {CLASSIFICATION_VERSION}. The dictionary was built "
                "under an older classification policy; rebuild it explicitly "
                "(remove the dictionary file and rerun the backfill). "
                "Refusing to mix classification versions."
            )

        observed = _decode_observed_asset_types(row.observed_asset_types)
        if len(observed) != 1:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: observed_asset_types "
                f"{row.observed_asset_types!r} must contain exactly one "
                "accepted assetType under classification version "
                f"{CLASSIFICATION_VERSION}."
            )
        asset_type = next(iter(observed))
        if asset_type not in _VALID_ASSET_TYPES:
            raise SecurityTypesError(
                f"Ticker {row.ticker}: observed_asset_types outside 0-9: "
                f"{sorted(observed)}."
            )
        if row.classification != _classification_for_type(asset_type):
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
        if str(row.source_date_min) != str(row.source_date_max):
            raise SecurityTypesError(
                f"Ticker {row.ticker}: source_date_min "
                f"({row.source_date_min!r}) must equal source_date_max "
                f"({row.source_date_max!r}) — both are the accepted query date."
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


def _normalize_candidates(
    candidates: Mapping[str, Sequence[object]],
) -> dict[str, list[date]]:
    """Normalize a ticker → observed-dates mapping; merge colliding keys."""
    normalized: dict[str, list[date]] = {}
    for raw_ticker, raw_dates in candidates.items():
        ticker = normalize_ticker(raw_ticker)
        dates = _normalize_observed_dates(raw_dates)
        if ticker in normalized:
            dates = sorted(set(normalized[ticker]) | set(dates), reverse=True)
        normalized[ticker] = dates
    if not normalized:
        raise SecurityTypesError("Candidate ticker set is empty.")
    return normalized


def _classify_batch(
    candidates: Mapping[str, Sequence[date]],
    fetch_observation_fn: Callable[[str, date], pd.DataFrame],
    *,
    now_fn: Callable[[], datetime],
) -> pd.DataFrame:
    """Classify every candidate ticker; any single failure fails the batch."""
    classified_at = (
        now_fn().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    rows = [
        classify_ticker_with_fallback(
            ticker,
            candidates[ticker],
            fetch_observation_fn,
            classified_at_utc=classified_at,
        )
        for ticker in sorted(candidates)
    ]
    return pd.DataFrame(rows, columns=list(SECURITY_TYPES_COLUMNS))


def ensure_security_types(
    candidates: Mapping[str, Sequence[object]],
    path: Path | str,
    *,
    fetch_observation_fn: Callable[[str, date], pd.DataFrame],
    now_fn: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    """Ensure the dictionary covers every candidate ticker; return it.

    ``candidates`` maps each normalized ticker to the distinct trade dates on
    which it appeared in candidate daily liquidity data. Only tickers absent
    from the dictionary are classified, one date-specific Core observation at
    a time (latest observed date first, bounded valid-empty fallback). See
    the module docstring for the exact cache-behaviour contract.
    """
    if now_fn is None:
        now_fn = lambda: datetime.now(timezone.utc)  # noqa: E731

    normalized = _normalize_candidates(candidates)

    file_path = Path(path)
    if not file_path.is_file():
        logger.info(
            "Security-type dictionary absent at %s — classifying %d candidate ticker(s).",
            file_path, len(normalized),
        )
        dictionary = _sorted_dictionary(
            _classify_batch(normalized, fetch_observation_fn, now_fn=now_fn)
        )
        write_security_types_atomic(dictionary, file_path)
        return dictionary

    existing = load_security_types(file_path)
    missing = sorted(set(normalized) - set(existing["ticker"]))
    if not missing:
        logger.info(
            "Security-type dictionary covers all %d candidate ticker(s); "
            "no fetch, no rewrite.",
            len(normalized),
        )
        return existing

    logger.info(
        "Security-type dictionary missing %d of %d candidate ticker(s) — "
        "classifying only the missing set.",
        len(missing), len(normalized),
    )
    new_rows = _classify_batch(
        {t: normalized[t] for t in missing}, fetch_observation_fn, now_fn=now_fn
    )

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
