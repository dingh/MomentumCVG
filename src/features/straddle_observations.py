"""Canonical surface -> straddle observation transform (Sprint 005 D2).

Converts the accepted Sprint 004 surface artifacts (A1 ``option_surface_meta``
and A2 ``option_surface_quotes``) into exactly one weekly long-straddle
observation per accepted A1 ``(ticker, entry_date)`` key. The emitted table is
loaded directly as the ``straddle_history`` data source of a
``FeatureDataContext``; no adapter, renaming, or unit conversion sits between
this artifact and ``MomentumCalculator`` / ``CVGCalculator``.

The complete A1 key grid is the contract. Both calculators look back by row
position rather than by calendar date, so dropping an unavailable ticker-week
would silently shift every later window for that ticker. Unavailable weeks are
therefore preserved as rows with null economics and an explicit missingness
reason.

Design of record: ``docs/surface_straddle_observation_transform_design.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.input_snapshot import (
    ARTIFACT_OPTION_SURFACE_META,
    ARTIFACT_OPTION_SURFACE_QUOTES,
    InputSnapshotManifest,
    default_manifest_path,
    read_manifest,
)
from src.data.snapshot_foundation import (
    resolve_under_root,
    sha256_file,
    ticker_date_keys_digest,
)

# ── Frozen transform identity ────────────────────────────────────────────────
# These constants define the one approved economic behavior. They are not
# runtime options: there is no CLI flag or config branch that selects an
# alternative, because a canonical artifact must have exactly one economic
# meaning. They are serialized into the lineage receipt so the artifact stays
# self-describing.

SNAPSHOT_STRADDLE_SCHEMA_VERSION = "1"
TRANSFORM_CONFIG_VERSION = "d2-1"

DIRECTION = "long"
FILL_MODEL = "mid"
MAX_LEG_SPREAD_PCT = 0.99
ENTRY_IV_RULE = "mean_body_call_put_iv"
VOL_GAP_RULE = "realized_volatility_minus_entry_iv"
# Decision D-1 (strategy owner, 2026-08-01): spread ineligibility does not by
# itself null the volatility fields. See ``build_observations``.
SPREAD_INELIGIBLE_VOLATILITY_RULE = "preserve"
RETURN_PCT_UNITS = "percentage_points"
VOLATILITY_UNITS = "annualized_decimal"

TRANSFORM_CONFIG: dict[str, Any] = {
    "direction": DIRECTION,
    "fill": FILL_MODEL,
    "max_leg_spread_pct": MAX_LEG_SPREAD_PCT,
    "entry_iv_rule": ENTRY_IV_RULE,
    "vol_gap_rule": VOL_GAP_RULE,
    "spread_ineligible_volatility_rule": SPREAD_INELIGIBLE_VOLATILITY_RULE,
    "return_pct_units": RETURN_PCT_UNITS,
    "volatility_units": VOLATILITY_UNITS,
    "transform_config_version": TRANSFORM_CONFIG_VERSION,
}

# ── Output contract ──────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_BODY_SPREAD_INELIGIBLE = "body_spread_ineligible"
STATUS_BODY_QUOTE_UNUSABLE = "body_quote_unusable"
STATUS_SURFACE_INVALID = "surface_invalid"

OBSERVATION_STATUSES = (
    STATUS_OK,
    STATUS_BODY_SPREAD_INELIGIBLE,
    STATUS_BODY_QUOTE_UNUSABLE,
    STATUS_SURFACE_INVALID,
)

REASON_SPREAD_ABOVE_THRESHOLD = "body_spread_above_threshold"
REASON_SPREAD_UNAVAILABLE = "body_spread_unavailable"
REASON_QUOTE_NOT_POSITIVE_FINITE = "body_quote_not_positive_finite"
REASON_SURFACE_INVALID_REASON_MISSING = "surface_invalid_reason_missing"

# Reasons D2 assigns itself. ``surface_invalid`` rows instead carry A1's own
# ``failure_reason`` tag unchanged, so that vocabulary is not duplicated here.
MISSING_REASONS = (
    REASON_SPREAD_ABOVE_THRESHOLD,
    REASON_SPREAD_UNAVAILABLE,
    REASON_QUOTE_NOT_POSITIVE_FINITE,
    REASON_SURFACE_INVALID_REASON_MISSING,
)

OBSERVATION_COLUMNS = (
    "ticker",
    "entry_date",
    "observation_status",
    "missing_reason",
    "surface_valid",
    # A1 passthroughs, preserved on every row including surface_invalid ones.
    "expiry_date",
    "dte_actual",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "spot_move_pct",
    # Stored body-leg quotes, retained so every derived field is recomputable
    # from this artifact alone during a D1 audit.
    "call_bid",
    "call_ask",
    "put_bid",
    "put_ask",
    "call_iv",
    "put_iv",
    "call_spread_pct",
    "put_spread_pct",
    # Long-straddle trade economics.
    "entry_cost",
    "exit_value",
    "pnl",
    "return_pct",
    # Calculator-facing volatility fields.
    "entry_iv",
    "realized_volatility",
    "vol_gap",
)

# Trade economics are populated together or not at all: they all describe the
# same claimed entry.
TRADE_ECONOMIC_COLUMNS = ("entry_cost", "exit_value", "pnl", "return_pct")

DATETIME_DTYPE = "datetime64[ns]"
_DATE_COLUMNS = ("entry_date", "expiry_date")

_FLOAT_COLUMNS = (
    "dte_actual",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "spot_move_pct",
    "call_bid",
    "call_ask",
    "put_bid",
    "put_ask",
    "call_iv",
    "put_iv",
    "call_spread_pct",
    "put_spread_pct",
    "entry_cost",
    "exit_value",
    "pnl",
    "return_pct",
    "entry_iv",
    "realized_volatility",
    "vol_gap",
)

# ── Artifact placement ───────────────────────────────────────────────────────
# The snapshot root is immutable and ``cache/`` is the mutable producer cache,
# so derived artifacts live under their own snapshot-keyed root.

DEFAULT_DERIVED_ROOT = Path("C:/MomentumCVG_env/derived")
OBSERVATIONS_FILENAME = "straddle_observations_weekly.parquet"
LINEAGE_FILENAME = "straddle_observations_weekly.lineage.json"

_META_COLUMNS = [
    "ticker",
    "entry_date",
    "expiry_date",
    "dte_actual",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "spot_move_pct",
    "realized_volatility",
    "surface_valid",
    "failure_reason",
]

# A2's ``mid`` must not be used for pricing, and volume/open interest must not
# gate anything, so neither is read.
_QUOTE_COLUMNS = [
    "ticker",
    "entry_date",
    "expiry_date",
    "strike",
    "side",
    "is_body",
    "bid",
    "ask",
    "spread_pct",
    "iv",
]


class StraddleObservationStructuralError(RuntimeError):
    """Raised when D2 cannot produce a trustworthy artifact.

    Covers input-identity failures (a snapshot that is not the accepted one, or
    an A1 whose key grid does not match the manifest), contradictions between
    A1 and A2 about which body contract exists, and violations of D2's own
    output contract. Ordinary market unavailability is never an error; it is
    recorded as a row.
    """


@dataclass(frozen=True)
class SurfaceInputs:
    """Manifest-resolved locations of the accepted A1 and A2 artifacts."""

    snapshot_root: Path
    manifest_path: Path
    manifest: InputSnapshotManifest
    meta_path: Path
    quotes_path: Path

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def build_id(self) -> str:
        return self.manifest.build_id


@dataclass(frozen=True)
class PublicationResult:
    """Outcome of publishing the observation table and its lineage receipt."""

    observations_path: Path
    lineage_path: Path
    receipt: dict[str, Any]
    written: bool  # False when an identical artifact was already published


# ── 1. Input resolution and guards ───────────────────────────────────────────


def _locate_manifest(snapshot_root: Path) -> Path:
    manifests_dir = snapshot_root / "manifests"
    if not manifests_dir.is_dir():
        raise StraddleObservationStructuralError(
            f"snapshot root has no manifests directory: {manifests_dir}"
        )
    candidates = sorted(manifests_dir.glob("input_snapshot_*.json"))
    if len(candidates) != 1:
        raise StraddleObservationStructuralError(
            f"expected exactly one input snapshot manifest under {manifests_dir}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def resolve_surface_inputs(snapshot_root: Path | str) -> SurfaceInputs:
    """Resolve A1/A2 from an accepted snapshot manifest and guard acceptance.

    The only input parameter is the snapshot root: A1 and A2 are never taken
    from direct path overrides. ``C:/MomentumCVG_env/cache/`` holds files with
    the same names at different sizes, so manifest-only resolution plus the
    key-set guard in :func:`load_surface_frames` is what keeps a mutable cache
    from silently standing in for the accepted input.
    """
    root = Path(snapshot_root)
    manifest_path = _locate_manifest(root)
    manifest = read_manifest(manifest_path)

    expected_path = default_manifest_path(root, manifest.snapshot_id)
    if manifest_path.name != expected_path.name:
        raise StraddleObservationStructuralError(
            f"manifest filename {manifest_path.name!r} does not match its "
            f"snapshot_id {manifest.snapshot_id!r}"
        )
    if manifest.production_accepted is not True:
        raise StraddleObservationStructuralError(
            f"snapshot {manifest.snapshot_id} is not production_accepted "
            f"(production_accepted={manifest.production_accepted!r})"
        )
    if manifest.blocking_failures:
        raise StraddleObservationStructuralError(
            f"snapshot {manifest.snapshot_id} has blocking failures: "
            f"{manifest.blocking_failures}"
        )
    # The accepted snapshot is WARN, so WARN must be tolerated; only FAIL aborts.
    if manifest.overall_status not in ("PASS", "WARN"):
        raise StraddleObservationStructuralError(
            f"snapshot {manifest.snapshot_id} overall_status is "
            f"{manifest.overall_status!r}; expected PASS or WARN"
        )

    paths: dict[str, Path] = {}
    for artifact_key in (ARTIFACT_OPTION_SURFACE_META, ARTIFACT_OPTION_SURFACE_QUOTES):
        if artifact_key not in manifest.artifacts:
            raise StraddleObservationStructuralError(
                f"manifest does not publish artifact {artifact_key!r}"
            )
        resolved = resolve_under_root(
            root, manifest.artifacts[artifact_key], label=artifact_key
        )
        if not resolved.is_file():
            raise StraddleObservationStructuralError(
                f"artifact {artifact_key!r} is missing at {resolved}"
            )
        paths[artifact_key] = resolved

    return SurfaceInputs(
        snapshot_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        meta_path=paths[ARTIFACT_OPTION_SURFACE_META],
        quotes_path=paths[ARTIFACT_OPTION_SURFACE_QUOTES],
    )


def a1_key_digest(meta_df: pd.DataFrame) -> str:
    """Digest the A1 ``(ticker, entry_date)`` key set the way the manifest did."""
    keys = {
        (pd.Timestamp(entry).date(), str(ticker).strip().upper())
        for ticker, entry in zip(meta_df["ticker"], meta_df["entry_date"])
    }
    return ticker_date_keys_digest(keys)


def _normalize_surface_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case tickers and convert A1/A2 ``date32`` columns to datetime64."""
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    for column in _DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column]).astype(DATETIME_DTYPE)
    return df


def load_surface_frames(inputs: SurfaceInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the A1 grid and the A2 body legs, then verify the A1 key identity.

    Only the columns the transform consumes are read, and A2 is restricted to
    ``is_body`` rows at the Parquet level: the OTM wings can never contribute to
    a body straddle, so they are unreachable rather than merely unused.

    The digest check proves this A1 carries the accepted key grid. It is not a
    full-file content proof; content trust rests on the snapshot's immutability,
    and the receipt records both source SHA-256 digests for later comparison.
    """
    meta_df = pd.read_parquet(inputs.meta_path, columns=_META_COLUMNS)
    quotes_df = pd.read_parquet(
        inputs.quotes_path,
        columns=_QUOTE_COLUMNS,
        filters=[("is_body", "==", True)],
    )

    meta_df = _normalize_surface_frame(meta_df)
    quotes_df = _normalize_surface_frame(quotes_df)

    recomputed = a1_key_digest(meta_df)
    published = inputs.manifest.params.get("surface_actual_a1_key_digest")
    if recomputed != published:
        raise StraddleObservationStructuralError(
            "A1 key-set digest does not match the manifest: "
            f"recomputed={recomputed!r} manifest={published!r}. "
            f"Resolved A1 was {inputs.meta_path}."
        )
    return meta_df, quotes_df


# ── 2. Body-leg join ─────────────────────────────────────────────────────────


def _body_side_frame(body_quotes: pd.DataFrame, side: str) -> pd.DataFrame:
    """Rename one body side's columns to their ``call_*`` / ``put_*`` form."""
    side_frame = body_quotes[body_quotes["side"] == side]
    renamed = side_frame.rename(
        columns={
            "bid": f"{side}_bid",
            "ask": f"{side}_ask",
            "iv": f"{side}_iv",
            "spread_pct": f"{side}_spread_pct",
            "strike": f"{side}_strike",
            "expiry_date": f"{side}_expiry_date",
        }
    )
    renamed[f"{side}_leg_count"] = renamed.groupby(
        ["ticker", "entry_date"], sort=False
    )["ticker"].transform("size")
    columns = [
        "ticker",
        "entry_date",
        f"{side}_bid",
        f"{side}_ask",
        f"{side}_iv",
        f"{side}_spread_pct",
        f"{side}_strike",
        f"{side}_expiry_date",
        f"{side}_leg_count",
    ]
    # Keeping the first row per key would silently hide a duplicate; the extra
    # rows are preserved here so validate_structural_integrity can see them.
    return renamed[columns].drop_duplicates(subset=["ticker", "entry_date"], keep="first")


def join_body_legs(meta_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Attach each valid A1 key's stored body call and put to its grid row.

    Leg selection is a lookup, never a search: there is no nearest-strike,
    nearest-delta, nearest-expiry, or nearest-week fallback. Partial A2 rows
    belonging to ``surface_valid=False`` A1 rows are ignored entirely, since
    harvesting them would substitute a contract the producer rejected.
    """
    joined = meta_df.copy()
    joined["surface_valid"] = joined["surface_valid"].astype(bool)

    body_quotes = quotes_df[quotes_df["is_body"].astype(bool)]
    # De-duplicating the key filter keeps a duplicated A1 key from inflating the
    # leg counts; that condition belongs to the output key post-condition.
    valid_keys = joined.loc[
        joined["surface_valid"], ["ticker", "entry_date"]
    ].drop_duplicates()
    body_quotes = body_quotes.merge(valid_keys, on=["ticker", "entry_date"], how="inner")

    for side in ("call", "put"):
        joined = joined.merge(
            _body_side_frame(body_quotes, side),
            on=["ticker", "entry_date"],
            how="left",
        )
        joined[f"{side}_leg_count"] = joined[f"{side}_leg_count"].fillna(0).astype(int)
    return joined


# ── 3. Structural validation ─────────────────────────────────────────────────


def _example_keys(frame: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list[str]:
    sample = frame.loc[mask, ["ticker", "entry_date"]].head(limit)
    return [
        f"{ticker}@{pd.Timestamp(entry).date().isoformat()}"
        for ticker, entry in sample.itertuples(index=False, name=None)
    ]


def _values_disagree(left: pd.Series, right: pd.Series) -> pd.Series:
    """True where two aligned columns hold different values (NaN == NaN)."""
    both_null = left.isna() & right.isna()
    return ~(both_null | (left == right))


def validate_structural_integrity(joined: pd.DataFrame) -> None:
    """Fail the run when A1 and A2 contradict each other about the body legs.

    A structural error means the two artifacts disagree about what exists, so
    no observation for that key can be trusted. It is deliberately distinct
    from a leg whose own values are unusable, which is ordinary data quality
    and is recorded as a row instead. All categories are reported together so
    an operator sees the full shape of a failure at once.

    The accepted Sprint 004 input certification is trusted rather than
    repeated: A1/A2 schemas, grain, coverage, settlement fields, and join
    integrity were certified at snapshot acceptance and are not rechecked here.
    """
    valid = joined["surface_valid"].astype(bool)
    problems: list[str] = []

    checks: list[tuple[str, pd.Series]] = [
        ("valid A1 row with no body call", valid & (joined["call_leg_count"] == 0)),
        ("valid A1 row with no body put", valid & (joined["put_leg_count"] == 0)),
        ("valid A1 row with duplicate body call", valid & (joined["call_leg_count"] > 1)),
        ("valid A1 row with duplicate body put", valid & (joined["put_leg_count"] > 1)),
    ]
    for side in ("call", "put"):
        present = valid & (joined[f"{side}_leg_count"] == 1)
        checks.append(
            (
                f"body {side} strike disagrees with A1 body_strike",
                present & _values_disagree(joined[f"{side}_strike"], joined["body_strike"]),
            )
        )
        checks.append(
            (
                f"body {side} expiry disagrees with A1 expiry_date",
                present
                & _values_disagree(joined[f"{side}_expiry_date"], joined["expiry_date"]),
            )
        )

    for label, mask in checks:
        count = int(mask.sum())
        if count:
            problems.append(
                f"{label}: {count} key(s); examples {_example_keys(joined, mask)}"
            )

    if problems:
        raise StraddleObservationStructuralError(
            "A1/A2 body-leg contradictions detected; no artifact written.\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


# ── 4. Observation construction ──────────────────────────────────────────────


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _positive_finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def build_observations(joined: pd.DataFrame) -> pd.DataFrame:
    """Build one straddle observation per A1 key from the joined body legs.

    Economics follow ``build_straddle_from_surface(direction="long",
    fill=FillAssumption.mid())`` plus ``settle``: a long call and long put at
    ``body_strike``, each priced at the bid/ask midpoint (never the stored
    ``mid`` column, which ORATS may round), held to expiry against A1's
    ``exit_spot``. ``return_pct`` is in percentage points with a floor of -100.
    """
    call_bid = _as_float(joined["call_bid"])
    call_ask = _as_float(joined["call_ask"])
    put_bid = _as_float(joined["put_bid"])
    put_ask = _as_float(joined["put_ask"])
    call_spread = _as_float(joined["call_spread_pct"])
    put_spread = _as_float(joined["put_spread_pct"])
    call_iv = _as_float(joined["call_iv"])
    put_iv = _as_float(joined["put_iv"])
    body_strike = _as_float(joined["body_strike"])
    exit_spot = _as_float(joined["exit_spot"])
    source_rv = _as_float(joined["realized_volatility"])

    surface_valid = joined["surface_valid"].to_numpy(dtype=bool)

    # Named masks rather than one dense expression, so the precedence below can
    # be read directly against the design's status table.
    quotes_usable = (
        _positive_finite(call_bid)
        & _positive_finite(call_ask)
        & _positive_finite(put_bid)
        & _positive_finite(put_ask)
    )
    spread_known = np.isfinite(call_spread) & np.isfinite(put_spread)
    spread_within_threshold = (
        spread_known
        & (call_spread <= MAX_LEG_SPREAD_PCT)
        & (put_spread <= MAX_LEG_SPREAD_PCT)
    )

    # Frozen precedence: surface invalidity outranks an unusable quote, which
    # outranks spread ineligibility. Fixing the order here stops overlapping bad
    # conditions from being resolved by incidental branch order.
    is_surface_invalid = ~surface_valid
    is_quote_unusable = surface_valid & ~quotes_usable
    is_spread_unavailable = surface_valid & quotes_usable & ~spread_known
    is_spread_above = (
        surface_valid & quotes_usable & spread_known & ~spread_within_threshold
    )
    is_ok = surface_valid & quotes_usable & spread_within_threshold

    # A1's own failure vocabulary is passed through unchanged rather than
    # reclassified; the placeholder only covers a null tag, which keeps
    # missing_reason non-null by construction on every non-ok row.
    a1_reason = (
        joined["failure_reason"]
        .astype(object)
        .where(joined["failure_reason"].notna(), REASON_SURFACE_INVALID_REASON_MISSING)
        .to_numpy(dtype=object)
    )

    status = np.full(len(joined), STATUS_OK, dtype=object)
    missing_reason = np.full(len(joined), None, dtype=object)
    for mask, status_value, reason_value in (
        (is_surface_invalid, STATUS_SURFACE_INVALID, None),
        (is_quote_unusable, STATUS_BODY_QUOTE_UNUSABLE, REASON_QUOTE_NOT_POSITIVE_FINITE),
        (is_spread_unavailable, STATUS_BODY_SPREAD_INELIGIBLE, REASON_SPREAD_UNAVAILABLE),
        (is_spread_above, STATUS_BODY_SPREAD_INELIGIBLE, REASON_SPREAD_ABOVE_THRESHOLD),
    ):
        status[mask] = status_value
        missing_reason[mask] = a1_reason[mask] if reason_value is None else reason_value

    entry_cost = np.where(is_ok, (call_bid + call_ask) / 2 + (put_bid + put_ask) / 2, np.nan)
    exit_value = np.where(
        is_ok,
        np.maximum(exit_spot - body_strike, 0.0) + np.maximum(body_strike - exit_spot, 0.0),
        np.nan,
    )
    pnl = exit_value - entry_cost
    # abs() keeps the expression literally identical to ``Position.pnl_pct``;
    # a long straddle is always a debit, so the two coincide.
    with np.errstate(invalid="ignore", divide="ignore"):
        return_pct = pnl / np.abs(entry_cost) * 100.0

    # Decision D-1: a wide spread governs whether the position could be taken,
    # not whether the volatility gap was seen, so entry_iv survives on
    # spread-ineligible rows. An unusable quote does not support a
    # calculator-facing market observation, so it does not inherit that rule.
    iv_usable = _positive_finite(call_iv) & _positive_finite(put_iv)
    iv_eligible_status = is_ok | is_spread_unavailable | is_spread_above
    entry_iv = np.where(iv_eligible_status & iv_usable, (call_iv + put_iv) / 2, np.nan)

    # A1's realized volatility is preserved wherever it is usable, including on
    # surface_invalid rows: one missing input must not erase unrelated valid
    # information, and D2 never recomputes it.
    realized_volatility = np.where(
        np.isfinite(source_rv) & (source_rv >= 0), source_rv, np.nan
    )
    vol_gap = realized_volatility - entry_iv

    observations = pd.DataFrame(
        {
            "ticker": joined["ticker"].to_numpy(),
            "entry_date": joined["entry_date"].to_numpy(),
            "observation_status": status,
            "missing_reason": missing_reason,
            "surface_valid": surface_valid,
            "expiry_date": joined["expiry_date"].to_numpy(),
            "dte_actual": _as_float(joined["dte_actual"]),
            "entry_spot": _as_float(joined["entry_spot"]),
            "exit_spot": exit_spot,
            "body_strike": body_strike,
            # A1 stores spot_move_pct in percent while the volatilities are
            # decimals; D2 preserves A1's units rather than silently rescaling.
            "spot_move_pct": _as_float(joined["spot_move_pct"]),
            "call_bid": call_bid,
            "call_ask": call_ask,
            "put_bid": put_bid,
            "put_ask": put_ask,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "call_spread_pct": call_spread,
            "put_spread_pct": put_spread,
            "entry_cost": entry_cost,
            "exit_value": exit_value,
            "pnl": pnl,
            "return_pct": return_pct,
            "entry_iv": entry_iv,
            "realized_volatility": realized_volatility,
            "vol_gap": vol_gap,
        }
    )

    observations["ticker"] = observations["ticker"].astype(str)
    observations["observation_status"] = observations["observation_status"].astype(str)
    # A1 stores dates as date32, from which pandas infers second resolution.
    # Pinning nanoseconds keeps the emitted dtype identical to the dtype a
    # consumer reads back from Parquet, so the content digest survives the
    # round trip and the calculators always see one date type.
    for column in ("entry_date", "expiry_date"):
        observations[column] = pd.to_datetime(observations[column]).astype(DATETIME_DTYPE)
    for column in _FLOAT_COLUMNS:
        observations[column] = observations[column].astype("float64")

    observations = observations[list(OBSERVATION_COLUMNS)]
    # Sorting by the unique key makes the row order total, and the default
    # RangeIndex is required by MomentumCalculator's index.get_loc lookup.
    return observations.sort_values(["ticker", "entry_date"]).reset_index(drop=True)


# ── 5. Output contract ───────────────────────────────────────────────────────


def validate_output_contract(observations: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    """Verify what D2 itself promises, before anything is published.

    These post-conditions prove the key grid, the status/reason vocabulary, and
    the dependent-null rules. They replace an input-side re-audit of A1/A2: a
    duplicated input key, for example, cannot satisfy the one-unique-row-per-key
    contract and surfaces here at a fraction of the cost.
    """
    problems: list[str] = []

    if list(observations.columns) != list(OBSERVATION_COLUMNS):
        problems.append(
            f"column order {list(observations.columns)} != {list(OBSERVATION_COLUMNS)}"
        )
    if not isinstance(observations.index, pd.RangeIndex) or not observations.index.is_unique:
        problems.append("output index must be a unique RangeIndex")
    for column in _DATE_COLUMNS:
        if observations[column].dtype != np.dtype(DATETIME_DTYPE):
            problems.append(
                f"{column} must be {DATETIME_DTYPE}; got {observations[column].dtype}"
            )

    output_keys = set(zip(observations["ticker"], observations["entry_date"]))
    expected_keys = set(
        zip(
            meta_df["ticker"].astype(str).str.strip().str.upper(),
            pd.to_datetime(meta_df["entry_date"]).astype(DATETIME_DTYPE),
        )
    )
    if len(observations) != len(meta_df):
        problems.append(f"row count {len(observations)} != A1 row count {len(meta_df)}")
    if len(output_keys) != len(observations):
        problems.append("duplicate (ticker, entry_date) key in output")
    if output_keys != expected_keys:
        problems.append(
            f"output key set differs from A1: {len(expected_keys - output_keys)} missing, "
            f"{len(output_keys - expected_keys)} unexpected"
        )

    status = observations["observation_status"]
    unknown_statuses = sorted(set(status.unique()) - set(OBSERVATION_STATUSES))
    if unknown_statuses:
        problems.append(f"unknown observation_status value(s): {unknown_statuses}")

    is_ok = status == STATUS_OK
    reason_missing = observations["missing_reason"].isna()
    if not np.array_equal(reason_missing.to_numpy(), is_ok.to_numpy()):
        problems.append("missing_reason must be null exactly on ok rows")
    for status_value, allowed in (
        (STATUS_BODY_SPREAD_INELIGIBLE, {REASON_SPREAD_ABOVE_THRESHOLD, REASON_SPREAD_UNAVAILABLE}),
        (STATUS_BODY_QUOTE_UNUSABLE, {REASON_QUOTE_NOT_POSITIVE_FINITE}),
    ):
        rows = observations.loc[status == status_value, "missing_reason"]
        unexpected = sorted(set(rows.dropna().unique()) - allowed)
        if unexpected:
            problems.append(f"{status_value} rows carry unexpected reason(s): {unexpected}")

    for column in TRADE_ECONOMIC_COLUMNS:
        values = observations[column]
        if not values[is_ok].notna().all():
            problems.append(f"{column} must be non-null on every ok row")
        if not np.isfinite(values[is_ok].to_numpy(dtype="float64")).all():
            problems.append(f"{column} must be finite on every ok row")
        if values[~is_ok].notna().any():
            problems.append(f"{column} must be null on every non-ok row")
    if (observations.loc[is_ok, "entry_cost"] <= 0).any():
        problems.append("entry_cost must be positive on every ok row (long straddle debit)")

    entry_iv = observations["entry_iv"]
    iv_allowed = is_ok | (status == STATUS_BODY_SPREAD_INELIGIBLE)
    if entry_iv[~iv_allowed].notna().any():
        problems.append(
            "entry_iv may only be populated on ok or body_spread_ineligible rows"
        )
    populated_iv = entry_iv.dropna().to_numpy(dtype="float64")
    if populated_iv.size and not (np.isfinite(populated_iv) & (populated_iv > 0)).all():
        problems.append("populated entry_iv must be positive and finite")

    realized = observations["realized_volatility"]
    populated_rv = realized.dropna().to_numpy(dtype="float64")
    if populated_rv.size and not (np.isfinite(populated_rv) & (populated_rv >= 0)).all():
        problems.append("populated realized_volatility must be non-negative and finite")

    expected_gap_present = (entry_iv.notna() & realized.notna()).to_numpy()
    if not np.array_equal(observations["vol_gap"].notna().to_numpy(), expected_gap_present):
        problems.append(
            "vol_gap must be populated exactly where entry_iv and realized_volatility are"
        )

    for column in ("return_pct", "entry_iv", "realized_volatility", "vol_gap"):
        values = observations[column].dropna().to_numpy(dtype="float64")
        if values.size and not np.isfinite(values).all():
            problems.append(f"{column} must not contain infinite values")

    if problems:
        raise StraddleObservationStructuralError(
            "output contract violated; no artifact written.\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def transform_surface_frames(
    meta_df: pd.DataFrame, quotes_df: pd.DataFrame
) -> pd.DataFrame:
    """Run the full A1/A2 -> observation pipeline over in-memory frames."""
    joined = join_body_legs(meta_df, quotes_df)
    validate_structural_integrity(joined)
    observations = build_observations(joined)
    validate_output_contract(observations, meta_df)
    return observations


def observation_coverage(observations: pd.DataFrame) -> dict[str, Any]:
    """Summarize row counts by status and reason plus populated-field counts."""
    return {
        "row_count": int(len(observations)),
        "key_count": int(
            observations[["ticker", "entry_date"]].drop_duplicates().shape[0]
        ),
        "status_counts": {
            str(status): int(count)
            for status, count in observations["observation_status"]
            .value_counts()
            .sort_index()
            .items()
        },
        "missing_reason_counts": {
            str(reason): int(count)
            for reason, count in observations["missing_reason"]
            .dropna()
            .value_counts()
            .sort_index()
            .items()
        },
        "non_null_counts": {
            column: int(observations[column].notna().sum())
            for column in ("return_pct", "entry_iv", "realized_volatility", "vol_gap")
        },
    }


# ── 6. Determinism, lineage, and publication ─────────────────────────────────


def content_digest(observations: pd.DataFrame) -> str:
    """SHA-256 over the frame's value content in canonical row order.

    Parquet bytes are not the determinism criterion because writer version
    strings and compression internals vary across environments; the value
    content does not.
    """
    hashed = pd.util.hash_pandas_object(observations, index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _current_repo_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_lineage_receipt(
    *,
    inputs: SurfaceInputs,
    observations: pd.DataFrame,
    meta_row_count: int,
    quote_row_count: int,
    observations_digest: str,
    file_sha256: str,
    repo_sha: str | None,
    created_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the lineage receipt, modeled on the snapshot manifest idiom."""
    created = created_at_utc or datetime.now(timezone.utc)
    recomputed_key_digest = a1_key_digest(observations)
    manifest_key_digest = inputs.manifest.params.get("surface_actual_a1_key_digest")
    return {
        "schema_version": SNAPSHOT_STRADDLE_SCHEMA_VERSION,
        "artifact": "straddle_observations_weekly",
        "created_at_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_sha": repo_sha,
        "snapshot_id": inputs.snapshot_id,
        "build_id": inputs.build_id,
        "snapshot_root": str(inputs.snapshot_root),
        "manifest_path": str(inputs.manifest_path),
        "manifest_overall_status": inputs.manifest.overall_status,
        "production_accepted": inputs.manifest.production_accepted,
        "sources": {
            ARTIFACT_OPTION_SURFACE_META: {
                "manifest_path": inputs.manifest.artifacts[ARTIFACT_OPTION_SURFACE_META],
                "absolute_path": str(inputs.meta_path),
                "sha256": sha256_file(inputs.meta_path),
                "row_count": meta_row_count,
            },
            ARTIFACT_OPTION_SURFACE_QUOTES: {
                "manifest_path": inputs.manifest.artifacts[ARTIFACT_OPTION_SURFACE_QUOTES],
                "absolute_path": str(inputs.quotes_path),
                "sha256": sha256_file(inputs.quotes_path),
                "body_row_count": quote_row_count,
            },
        },
        "a1_key_count": meta_row_count,
        "a1_key_digest": recomputed_key_digest,
        "manifest_a1_key_digest": manifest_key_digest,
        "a1_key_digest_matches_manifest": recomputed_key_digest == manifest_key_digest,
        "transform_config": dict(TRANSFORM_CONFIG),
        "output": {
            "row_count": int(len(observations)),
            "key_count": int(
                observations[["ticker", "entry_date"]].drop_duplicates().shape[0]
            ),
            "output_key_digest": recomputed_key_digest,
            "content_digest": observations_digest,
            "file_sha256": file_sha256,
            "column_order": list(OBSERVATION_COLUMNS),
        },
        "coverage": observation_coverage(observations),
    }


def derived_dir_for_snapshot(snapshot_id: str, output_root: Path | str | None = None) -> Path:
    """Return ``<output_root>/<snapshot_id>``, defaulting to the derived root."""
    root = Path(output_root) if output_root is not None else DEFAULT_DERIVED_ROOT
    return root / snapshot_id


def _write_parquet_atomic(observations: pd.DataFrame, target: Path) -> str:
    """Write the Parquet to a sibling temp file and return its SHA-256."""
    temp_path = target.parent / f"{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        observations.to_parquet(temp_path, index=False, compression="snappy")
        digest = sha256_file(temp_path)
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return digest


def _write_json_atomic(payload: dict[str, Any], target: Path) -> None:
    temp_path = target.parent / f"{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _check_republish(
    existing_receipt_path: Path,
    observations_path: Path,
    observations_digest: str,
) -> dict[str, Any]:
    """Refuse to change a canonical path's meaning; allow an identical rerun.

    D3, D4, and D5 treat the snapshot-keyed path as fixed, so a code change or
    a change to a frozen rule must never leave different bytes there. There is
    no override flag: a new transform version must use a different output root.
    """
    with existing_receipt_path.open(encoding="utf-8") as handle:
        existing = json.load(handle)

    existing_version = existing.get("transform_config", {}).get("transform_config_version")
    existing_digest = existing.get("output", {}).get("content_digest")
    if existing_version != TRANSFORM_CONFIG_VERSION:
        raise StraddleObservationStructuralError(
            f"refusing to republish {observations_path}: existing receipt records "
            f"transform_config_version {existing_version!r}, this run is "
            f"{TRANSFORM_CONFIG_VERSION!r}. Use a different --output-root."
        )
    if existing_digest != observations_digest:
        raise StraddleObservationStructuralError(
            f"refusing to republish {observations_path}: existing receipt records "
            f"content_digest {existing_digest!r}, this run produced "
            f"{observations_digest!r}. Use a different --output-root."
        )
    if not observations_path.is_file():
        raise StraddleObservationStructuralError(
            f"lineage receipt exists at {existing_receipt_path} but its artifact "
            f"{observations_path} is missing; treat as corruption, not a rerun."
        )
    actual_sha = sha256_file(observations_path)
    if actual_sha != existing.get("output", {}).get("file_sha256"):
        raise StraddleObservationStructuralError(
            f"published artifact {observations_path} does not match the SHA-256 in "
            f"{existing_receipt_path}; treat as corruption, not a rerun."
        )
    return existing


def publish_observations(
    observations: pd.DataFrame,
    *,
    inputs: SurfaceInputs,
    destination_dir: Path,
    meta_row_count: int,
    quote_row_count: int,
    repo_sha: str | None = None,
) -> PublicationResult:
    """Publish the Parquet atomically, then the lineage receipt last.

    Publication order matters: the receipt is the completion marker, so a
    receipt at the canonical path always implies a complete artifact beside it.
    A killed run can leave a stray temp file but never a truncated Parquet or a
    receipt describing an artifact that was not fully written.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    observations_path = destination / OBSERVATIONS_FILENAME
    lineage_path = destination / LINEAGE_FILENAME

    observations_digest = content_digest(observations)
    if lineage_path.is_file():
        existing = _check_republish(lineage_path, observations_path, observations_digest)
        return PublicationResult(
            observations_path=observations_path,
            lineage_path=lineage_path,
            receipt=existing,
            written=False,
        )

    file_sha256 = _write_parquet_atomic(observations, observations_path)
    receipt = build_lineage_receipt(
        inputs=inputs,
        observations=observations,
        meta_row_count=meta_row_count,
        quote_row_count=quote_row_count,
        observations_digest=observations_digest,
        file_sha256=file_sha256,
        repo_sha=repo_sha if repo_sha is not None else _current_repo_sha(),
    )
    _write_json_atomic(receipt, lineage_path)
    return PublicationResult(
        observations_path=observations_path,
        lineage_path=lineage_path,
        receipt=receipt,
        written=True,
    )
