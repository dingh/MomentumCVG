"""Weekly input snapshot receipt — Stage A input layer (Sprint 004 C1).

This module records *which weekly input artifacts were used* for a given as-of
date. It is intentionally minimal: a receipt, not pipeline telemetry.

Answers one question:
    What input snapshot did I use, for what as-of date, with which logical
    artifacts and key params?

Two IDs (do not confuse them):
    snapshot_id  — deterministic receipt identity (16-hex sha256 prefix).
                   Same identity fields → same id across re-runs.
    build_id     — unique per CLI execution (UTC timestamp + short hash).

snapshot_id is a *logical* receipt id, not a byte-level content fingerprint.
Callers compute snapshot_id via compute_snapshot_id() before write_manifest();
the write path does not recompute it silently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ── Schema constants ──────────────────────────────────────────────────────────

ValidationStatus = Literal["PASS", "WARN", "FAIL"]

DEFAULT_DATA_SOURCE = "orats_adjusted_cache"
INPUT_SNAPSHOT_SCHEMA_VERSION = "1"
SNAPSHOT_ID_PREFIX_LEN = 16

# Fixed artifact keys written into manifests. Values are cache-relative paths.
ARTIFACT_SPLITS = "splits"
ARTIFACT_SPOT_PRICES = "spot_prices"
ARTIFACT_LIQUIDITY_PANEL = "liquidity_panel"
ARTIFACT_OPTION_SURFACE_META = "option_surface_meta"
ARTIFACT_OPTION_SURFACE_QUOTES = "option_surface_quotes"

# Fields that feed snapshot_id hashing (see _identity_dict / _hash_identity).
_IDENTITY_KEYS = (
    "schema_version",
    "as_of_resolved_trading_day",
    "data_source",
    "artifacts",
    "params",
)

# Full manifest JSON requires every key below. C3+ may patch reports/status/notes.
_REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "snapshot_id",
    "build_id",
    "created_at_utc",
    "as_of_requested",
    "as_of_resolved_trading_day",
    "data_source",
    "cache_dir",
    "artifacts",
    "params",
    "reports",
    "overall_status",
    "blocking_failures",
    "notes",
)


@dataclass
class InputSnapshotManifest:
    """In-memory weekly input receipt (schema version 1)."""

    schema_version: str
    snapshot_id: str
    build_id: str
    created_at_utc: datetime
    as_of_requested: date  # user/CLI input date
    as_of_resolved_trading_day: date  # last trading day <= as_of_requested (HD-004-2)
    data_source: str
    cache_dir: str  # absolute/local root; excluded from snapshot_id hash
    artifacts: dict[str, str]  # ARTIFACT_* key → cache-relative path
    params: dict[str, Any]  # identity-relevant params only (e.g. rolling_months)
    reports: dict[str, str | None]  # markdown report paths; filled by C3+ audits
    overall_status: ValidationStatus | None  # aggregate PASS/WARN/FAIL; C3+
    blocking_failures: list[str]
    notes: list[str]


# ── Field validation helpers ──────────────────────────────────────────────────
# Strict parse/validate on both load and serialize paths so bad in-memory objects
# cannot write invalid JSON that would fail on read.


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string; got {value!r}")
    return value


def _parse_artifacts(artifacts: Mapping[Any, Any], *, field_name: str = "artifacts") -> dict[str, str]:
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    parsed: dict[str, str] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings; got {key!r}")
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} values must be strings; got {value!r} for key {key!r}"
            )
        # Normalize before hash and JSON so Windows paths do not fork snapshot_id.
        parsed[key] = value.replace("\\", "/")
    return parsed


def _parse_reports(reports: Mapping[Any, Any]) -> dict[str, str | None]:
    if not isinstance(reports, Mapping):
        raise ValueError("reports must be a mapping")

    parsed: dict[str, str | None] = {}
    for key, value in reports.items():
        if not isinstance(key, str):
            raise ValueError(f"reports keys must be strings; got {key!r}")
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"reports values must be str or None; got {value!r} for key {key!r}"
            )
        parsed[key] = value
    return parsed


def _format_date(value: date) -> str:
    # datetime is a subclass of date; reject it so JSON stays date-only (YYYY-MM-DD).
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"Expected date, got {value!r}")
    return value.isoformat()


def _parse_status(value: Any) -> ValidationStatus | None:
    if value is not None and value not in ("PASS", "WARN", "FAIL"):
        raise ValueError(f"Invalid overall_status: {value!r}")
    return value


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid date for {field_name}: {value!r}") from exc
    raise ValueError(f"Invalid date for {field_name}: {value!r}")


def _parse_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    return dict(params)


def _format_utc_datetime(value: datetime) -> str:
    # Naive datetimes are treated as UTC (consistent with generate_build_id).
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid datetime for {field_name}: {value!r}") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"Invalid datetime for {field_name}: {value!r}")


# ── snapshot_id identity + hashing ────────────────────────────────────────────
# Only _IDENTITY_KEYS participate in the hash. Everything else (build_id,
# created_at_utc, cache_dir, reports, validation fields, etc.) is excluded.


def _identity_dict(
    *,
    schema_version: str,
    as_of_resolved_trading_day: date | str,
    data_source: str,
    artifacts: Mapping[Any, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_day = _format_date(
        _parse_date(as_of_resolved_trading_day, "as_of_resolved_trading_day")
    )

    return {
        "schema_version": schema_version,
        "as_of_resolved_trading_day": resolved_day,
        "data_source": data_source,
        "artifacts": _parse_artifacts(artifacts),
        "params": _parse_params(params),
    }


def _hash_identity(identity: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError as exc:
        raise ValueError(
            "snapshot_id identity fields must be JSON-serializable; "
            "check params and artifacts for non-serializable values"
        ) from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:SNAPSHOT_ID_PREFIX_LEN]


def compute_snapshot_id(
    manifest_or_fields: InputSnapshotManifest | Mapping[str, Any],
) -> str:
    """Hash the receipt identity subset; accepts a manifest or an identity mapping."""
    if isinstance(manifest_or_fields, InputSnapshotManifest):
        identity = _identity_dict(
            schema_version=manifest_or_fields.schema_version,
            as_of_resolved_trading_day=manifest_or_fields.as_of_resolved_trading_day,
            data_source=manifest_or_fields.data_source,
            artifacts=manifest_or_fields.artifacts,
            params=manifest_or_fields.params,
        )
    else:
        missing = [key for key in _IDENTITY_KEYS if key not in manifest_or_fields]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing identity fields for snapshot_id: {joined}")
        params = _parse_params(manifest_or_fields["params"])
        identity = _identity_dict(
            schema_version=_require_str(
                manifest_or_fields["schema_version"],
                "schema_version",
            ),
            as_of_resolved_trading_day=manifest_or_fields["as_of_resolved_trading_day"],
            data_source=_require_str(manifest_or_fields["data_source"], "data_source"),
            artifacts=manifest_or_fields["artifacts"],
            params=params,
        )
    return _hash_identity(identity)


def generate_build_id(*, now: datetime, command: str) -> str:
    """Return a unique execution id: {UTC ts}_{6-hex hash of ts + command}."""
    if now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now.astimezone(timezone.utc)
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{ts}\0{command}".encode()).hexdigest()[:6]
    return f"{ts}_{suffix}"


# ── JSON serialization ────────────────────────────────────────────────────────


def manifest_to_dict(manifest: InputSnapshotManifest) -> dict[str, Any]:
    """Convert manifest to a JSON-ready dict; validates fields on the way out."""
    return {
        "schema_version": manifest.schema_version,
        "snapshot_id": manifest.snapshot_id,
        "build_id": manifest.build_id,
        "created_at_utc": _format_utc_datetime(manifest.created_at_utc),
        "as_of_requested": _format_date(manifest.as_of_requested),
        "as_of_resolved_trading_day": _format_date(manifest.as_of_resolved_trading_day),
        "data_source": manifest.data_source,
        "cache_dir": manifest.cache_dir,
        "artifacts": _parse_artifacts(manifest.artifacts),
        "params": _parse_params(manifest.params),
        "reports": _parse_reports(manifest.reports),
        "overall_status": _parse_status(manifest.overall_status),
        "blocking_failures": list(manifest.blocking_failures),
        "notes": list(manifest.notes),
    }


def manifest_from_dict(data: Mapping[str, Any]) -> InputSnapshotManifest:
    """Parse and validate a manifest dict; raises ValueError on schema mismatch."""
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in data]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required manifest fields: {joined}")

    schema_version = _require_str(data["schema_version"], "schema_version")
    if schema_version != INPUT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {INPUT_SNAPSHOT_SCHEMA_VERSION!r}"
        )

    overall_status = _parse_status(data["overall_status"])

    blocking_failures = data["blocking_failures"]
    if not isinstance(blocking_failures, list):
        raise ValueError("blocking_failures must be a list")

    notes = data["notes"]
    if not isinstance(notes, list):
        raise ValueError("notes must be a list")

    artifacts = data["artifacts"]
    reports = data["reports"]

    return InputSnapshotManifest(
        schema_version=schema_version,
        snapshot_id=_require_str(data["snapshot_id"], "snapshot_id"),
        build_id=_require_str(data["build_id"], "build_id"),
        created_at_utc=_parse_utc_datetime(data["created_at_utc"], "created_at_utc"),
        as_of_requested=_parse_date(data["as_of_requested"], "as_of_requested"),
        as_of_resolved_trading_day=_parse_date(
            data["as_of_resolved_trading_day"],
            "as_of_resolved_trading_day",
        ),
        data_source=_require_str(data["data_source"], "data_source"),
        cache_dir=_require_str(data["cache_dir"], "cache_dir"),
        artifacts=_parse_artifacts(artifacts),
        params=_parse_params(data["params"]),
        reports=_parse_reports(reports),
        overall_status=overall_status,
        blocking_failures=[str(item) for item in blocking_failures],
        notes=[str(item) for item in notes],
    )


# ── File I/O ──────────────────────────────────────────────────────────────────


def write_manifest(path: Path | str, manifest: InputSnapshotManifest) -> None:
    """Write manifest JSON (indent=2). Does not recompute snapshot_id."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_to_dict(manifest)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_manifest(path: Path | str) -> InputSnapshotManifest:
    """Load manifest JSON from disk."""
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Manifest JSON must be an object")
    return manifest_from_dict(data)


def default_manifest_path(cache_dir: Path | str, snapshot_id: str) -> Path:
    """Return {cache_dir}/manifests/input_snapshot_{snapshot_id}.json."""
    root = Path(cache_dir)
    return root / "manifests" / f"input_snapshot_{snapshot_id}.json"
