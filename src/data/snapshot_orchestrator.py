"""Resumable cold-backfill snapshot orchestration foundation (Sprint 004 C8.3B).

This module holds the *foundation* half of the C8.3B orchestrator as direct
functions and small dataclasses:

* frozen raw ORATS ZIP inventory discovery (central-directory evidence, no
  full-byte hashing of large archives);
* immutable ``run_config.json`` / ``raw_inventory.json`` freeze, load, and
  validation with a self-verifying ``run_config_id``;
* new-run preparation (fresh ``<BUILD_ID>.building`` root plus the owned
  directory layout) and resume-open of an explicitly named ``.building`` run;
* raw-inventory rescan with digest-drift rejection on resume.

The four producer stages (liquidity, adjusted, spot, surface), completion
markers, cross-stage validation, manifest construction, and atomic
publication are deferred to the next C8.3B commit. There is no framework,
DAG, plugin, receipt, or state-machine abstraction here — and no ``.failed``
lifecycle.

Design: docs/tmp/c8_3b_resumable_cold_backfill_design.md
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from src.data.snapshot_foundation import (
    SNAPSHOT_BUILD_ID_RE,
    SnapshotFoundationError,
    SnapshotRoots,
    canonical_json_bytes,
    create_fresh_staging_root,
    derive_snapshot_roots,
    digest_json,
    generate_snapshot_build_id,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RUN_CONFIG_FILENAME = "run_config.json"
RAW_INVENTORY_FILENAME = "raw_inventory.json"
RUN_CONFIG_SCHEMA_VERSION = "1"
RAW_INVENTORY_SCHEMA_VERSION = "1"

# Mirrors scripts/build_liquidity_panel.py DEFAULT_LOOKBACK_WEEKS (C4). A unit
# test cross-checks the two so they cannot silently diverge.
DEFAULT_LOOKBACK_WEEKS = 12
# C4 discovery pad: raw history begins (lookback_weeks + 2) weeks before the
# requested output start (see run_build in scripts/build_liquidity_panel.py).
# This module owns the single named form of that formula for snapshot runs.
RAW_DEPENDENCY_PAD_WEEKS = 2

# C4 stage parameters frozen into the run config for later stages. Values
# mirror scripts/build_liquidity_panel.py defaults (cross-checked by tests).
DEFAULT_C4_PARAMS: dict[str, Any] = {
    "lookback_weeks": DEFAULT_LOOKBACK_WEEKS,
    "min_valid_quote_weeks": 3,
    "dte_min": 5,
    "dte_max": 60,
    "dvol_top_pct": 0.20,
    "spread_bot_pct": 1.0,
}

# Existing surface/expiry policy identity (C6.1C strict calendar-paired weekly
# expiry via target_weekly_expiry_from_schedule; schedule successor tail from
# scripts/precompute_option_surface.py WEEKLY_SCHEDULE_TAIL_DAYS).
DEFAULT_SURFACE_POLICY: dict[str, Any] = {
    "entry_schedule": "weekly_last_trading_day",
    "expiry_policy": "strict_next_week_from_schedule",
    "weekly_schedule_tail_days": 21,
}

# Owned directory layout created inside a fresh <BUILD_ID>.building root.
SNAPSHOT_OWNED_DIRS: tuple[str, ...] = (
    "markers",
    "work/liquidity",
    "work/adjusted",
    "work/spot",
    "work/surface",
    "input/liquidity",
    "input/adjusted_liquid",
    "cache/spot",
    "cache/surface",
    "reports/liquidity",
    "reports/adjusted",
    "reports/spot",
    "reports/surface",
    "reports/final",
    "manifests",
)

_RAW_ZIP_NAME_RE = re.compile(r"^ORATS_SMV_Strikes_(\d{8})\.zip$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Content-level emptiness verification is only attempted on small archives;
# anything larger is trivially nonempty and is never read member-by-member.
_EMPTY_CONTENT_MAX_BYTES = 1 << 16

_RUN_CONFIG_REQUIRED_KEYS = (
    "schema_version",
    "build_id",
    "mode",
    "snapshots_root",
    "raw_root",
    "requested_output_start",
    "raw_dependency_start",
    "as_of_requested",
    "as_of_resolved_trading_day",
    "lookback_weeks",
    "physical_raw_dates",
    "resolved_trading_dates",
    "c4_params",
    "surface_policy",
    "inventory_digest",
    "repo_sha_at_freeze",
    "run_config_id",
)

_RAW_INVENTORY_REQUIRED_KEYS = (
    "schema_version",
    "raw_dependency_start",
    "as_of_requested",
    "archives",
    "physical_raw_dates",
    "resolved_trading_dates",
    "as_of_resolved_trading_day",
    "missing_weekday_dates_diagnostic",
    "inventory_digest",
)

# Keys of the raw-inventory document that form the digest identity. Retrieval
# timestamps and absolute machine paths are deliberately absent.
_RAW_INVENTORY_IDENTITY_KEYS = (
    "schema_version",
    "raw_dependency_start",
    "as_of_requested",
    "archives",
)


# ── Errors ────────────────────────────────────────────────────────────────────


class SnapshotOrchestratorError(SnapshotFoundationError):
    """Base class for orchestrator foundation failures."""


class RawInventoryError(SnapshotOrchestratorError):
    """Raised when the frozen raw inventory cannot be built or validated."""


class RunConfigError(SnapshotOrchestratorError):
    """Raised for immutable run-config / resume corruption failures.

    The CLI maps this class of failure to exit code 2 (config/corruption).
    """


# ── Dependency-start formula (single named form of the C4 pad) ────────────────


def compute_raw_dependency_start(
    requested_output_start: date,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> date:
    """Return ``requested_output_start - (lookback_weeks + 2) weeks``.

    This is the accepted C4 discovery pad (default 12 → 14 weeks). Do not
    introduce a competing formula elsewhere.
    """
    if not isinstance(lookback_weeks, int) or isinstance(lookback_weeks, bool):
        raise RunConfigError(f"lookback_weeks must be an int; got {lookback_weeks!r}")
    if lookback_weeks <= 0:
        raise RunConfigError(f"lookback_weeks must be positive; got {lookback_weeks}")
    if isinstance(requested_output_start, datetime) or not isinstance(
        requested_output_start, date
    ):
        raise RunConfigError(
            f"requested_output_start must be a date; got {requested_output_start!r}"
        )
    return requested_output_start - timedelta(
        weeks=lookback_weeks + RAW_DEPENDENCY_PAD_WEEKS
    )


# ── Raw inventory discovery ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RawArchiveRecord:
    """Canonical evidence for one selected raw ORATS daily ZIP archive."""

    rel_path: str  # raw-root-relative POSIX path
    trade_date: date
    file_size: int
    # Deterministic ZIP central-directory evidence, sorted by member name:
    # (member_name, crc32, uncompressed_size). Large bodies are never hashed.
    members: tuple[tuple[str, int, int], ...]
    verified_empty: bool


@dataclass(frozen=True)
class RawInventory:
    """Frozen raw inventory for one backfill run."""

    raw_root: Path
    raw_dependency_start: date
    as_of_requested: date
    records: tuple[RawArchiveRecord, ...]
    physical_raw_dates: tuple[date, ...]
    resolved_trading_dates: tuple[date, ...]
    as_of_resolved_trading_day: date
    missing_weekday_dates: tuple[date, ...]  # diagnostic list only — never a gate
    inventory_digest: str


def _archive_verified_empty(
    archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> bool:
    """Deterministically decide whether an archive is proven empty.

    Empty means: no members, all members have zero uncompressed size, or every
    (small) member decodes to at most one non-blank line (a bare CSV header).
    Content is only read when the total uncompressed size is tiny, so large
    archives are classified from the central directory alone.
    """
    if not infos:
        return True
    if all(info.file_size == 0 for info in infos):
        return True
    if sum(info.file_size for info in infos) > _EMPTY_CONTENT_MAX_BYTES:
        return False
    for info in infos:
        text = archive.read(info).decode("utf-8", errors="replace")
        nonblank = [line for line in text.splitlines() if line.strip()]
        if len(nonblank) > 1:
            return False
    return True


def scan_raw_inventory(
    raw_root: Path | str,
    raw_dependency_start: date,
    as_of_requested: date,
) -> RawInventory:
    """Scan ``*/ORATS_SMV_Strikes_YYYYMMDD.zip`` archives into a frozen inventory.

    Selection is bounded inclusively by ``[raw_dependency_start, as_of_requested]``.
    Fails closed on a missing/invalid raw root, malformed archive filename or
    date anywhere under the root, an unreadable or malformed ZIP, duplicate
    archives for one trade date, a nonempty Saturday/Sunday archive, or
    inability to resolve the run bounds. Observed missing Monday–Friday dates
    are recorded as a diagnostic list only.
    """
    root = Path(raw_root)
    if not root.is_dir():
        raise RawInventoryError(f"raw root does not exist or is not a directory: {root}")
    if isinstance(raw_dependency_start, datetime) or not isinstance(
        raw_dependency_start, date
    ):
        raise RawInventoryError(
            f"raw_dependency_start must be a date; got {raw_dependency_start!r}"
        )
    if isinstance(as_of_requested, datetime) or not isinstance(as_of_requested, date):
        raise RawInventoryError(f"as_of_requested must be a date; got {as_of_requested!r}")
    if raw_dependency_start > as_of_requested:
        raise RawInventoryError(
            "cannot resolve run bounds: raw_dependency_start "
            f"{raw_dependency_start.isoformat()} is after as_of_requested "
            f"{as_of_requested.isoformat()}"
        )

    records_by_date: dict[date, RawArchiveRecord] = {}
    for path in sorted(root.glob("*/ORATS_SMV_Strikes_*.zip")):
        match = _RAW_ZIP_NAME_RE.match(path.name)
        if match is None:
            raise RawInventoryError(f"malformed raw archive filename: {path}")
        try:
            trade_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError as exc:
            raise RawInventoryError(
                f"invalid trade date in raw archive filename: {path}"
            ) from exc
        if not (raw_dependency_start <= trade_date <= as_of_requested):
            continue
        if trade_date in records_by_date:
            raise RawInventoryError(
                f"duplicate raw archives for trade date {trade_date.isoformat()}: "
                f"{records_by_date[trade_date].rel_path} and "
                f"{path.relative_to(root).as_posix()}"
            )

        try:
            with zipfile.ZipFile(path, "r") as archive:
                infos = archive.infolist()
                members = tuple(
                    sorted((info.filename, info.CRC, info.file_size) for info in infos)
                )
                verified_empty = _archive_verified_empty(archive, infos)
        except RawInventoryError:
            raise
        except Exception as exc:
            raise RawInventoryError(
                f"unreadable or malformed ZIP archive {path}: {exc}"
            ) from exc

        if trade_date.weekday() >= 5 and not verified_empty:
            raise RawInventoryError(
                f"nonempty weekend archive: {path.relative_to(root).as_posix()}"
            )

        records_by_date[trade_date] = RawArchiveRecord(
            rel_path=path.relative_to(root).as_posix(),
            trade_date=trade_date,
            file_size=path.stat().st_size,
            members=members,
            verified_empty=verified_empty,
        )

    if not records_by_date:
        raise RawInventoryError(
            "cannot resolve run bounds: no raw archives discovered under "
            f"{root} in [{raw_dependency_start.isoformat()}, "
            f"{as_of_requested.isoformat()}]"
        )

    records = tuple(records_by_date[d] for d in sorted(records_by_date))
    physical_raw_dates = tuple(r.trade_date for r in records)
    resolved_trading_dates = tuple(
        r.trade_date
        for r in records
        if not (r.trade_date.weekday() >= 5 and r.verified_empty)
    )

    resolvable = [d for d in resolved_trading_dates if d <= as_of_requested]
    if not resolvable:
        raise RawInventoryError(
            "cannot resolve run bounds: no resolved trading date on or before "
            f"as_of_requested {as_of_requested.isoformat()}"
        )
    as_of_resolved_trading_day = max(resolvable)

    # Diagnostic only: weekdays with no physical archive between the observed
    # inventory endpoints. Not a PASS/WARN/FAIL outcome and never a blocker.
    missing_weekdays: list[date] = []
    cursor = physical_raw_dates[0]
    physical_set = set(physical_raw_dates)
    while cursor <= physical_raw_dates[-1]:
        if cursor.weekday() < 5 and cursor not in physical_set:
            missing_weekdays.append(cursor)
        cursor += timedelta(days=1)

    inventory = RawInventory(
        raw_root=root,
        raw_dependency_start=raw_dependency_start,
        as_of_requested=as_of_requested,
        records=records,
        physical_raw_dates=physical_raw_dates,
        resolved_trading_dates=resolved_trading_dates,
        as_of_resolved_trading_day=as_of_resolved_trading_day,
        missing_weekday_dates=tuple(missing_weekdays),
        inventory_digest="",
    )
    digest = digest_json(raw_inventory_identity_payload(inventory))
    return RawInventory(
        raw_root=inventory.raw_root,
        raw_dependency_start=inventory.raw_dependency_start,
        as_of_requested=inventory.as_of_requested,
        records=inventory.records,
        physical_raw_dates=inventory.physical_raw_dates,
        resolved_trading_dates=inventory.resolved_trading_dates,
        as_of_resolved_trading_day=inventory.as_of_resolved_trading_day,
        missing_weekday_dates=inventory.missing_weekday_dates,
        inventory_digest=digest,
    )


def raw_inventory_identity_payload(inventory: RawInventory) -> dict[str, Any]:
    """Canonical digest identity for a frozen raw inventory.

    Contains only run bounds and per-archive canonical evidence (raw-root-
    relative POSIX path, ISO date, size, central-directory members, verified-
    empty status). Retrieval timestamps and absolute machine paths are excluded.
    """
    return {
        "schema_version": RAW_INVENTORY_SCHEMA_VERSION,
        "raw_dependency_start": inventory.raw_dependency_start.isoformat(),
        "as_of_requested": inventory.as_of_requested.isoformat(),
        "archives": [
            {
                "rel_path": record.rel_path,
                "trade_date": record.trade_date.isoformat(),
                "file_size": record.file_size,
                "verified_empty": record.verified_empty,
                "members": [list(member) for member in record.members],
            }
            for record in inventory.records
        ],
    }


def raw_inventory_document(inventory: RawInventory) -> dict[str, Any]:
    """Full ``raw_inventory.json`` payload: identity plus derived evidence."""
    document = raw_inventory_identity_payload(inventory)
    document.update(
        {
            "physical_raw_dates": [d.isoformat() for d in inventory.physical_raw_dates],
            "resolved_trading_dates": [
                d.isoformat() for d in inventory.resolved_trading_dates
            ],
            "as_of_resolved_trading_day": inventory.as_of_resolved_trading_day.isoformat(),
            "missing_weekday_dates_diagnostic": [
                d.isoformat() for d in inventory.missing_weekday_dates
            ],
            "inventory_digest": inventory.inventory_digest,
        }
    )
    return document


# ── Immutable run configuration ───────────────────────────────────────────────


def _normalize_root(path: Path | str) -> str:
    """Normalized (absolute, forward-slashed) root representation for configs."""
    return Path(path).resolve().as_posix()


def current_repo_sha(repo_root: Path | str | None = None) -> str:
    """Best-effort git HEAD SHA. Diagnostic evidence only — never run identity."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=10,
        )
    except Exception:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def compute_run_config_id(config: Mapping[str, Any]) -> str:
    """SHA-256 of canonical run_config JSON with ``run_config_id`` omitted."""
    body = {key: value for key, value in config.items() if key != "run_config_id"}
    return digest_json(body)


def _parse_config_date(config: Mapping[str, Any], key: str) -> date:
    value = config.get(key)
    if not isinstance(value, str):
        raise RunConfigError(f"run config {key} must be an ISO date string; got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RunConfigError(f"run config {key} is not a valid ISO date: {value!r}") from exc


def validate_run_config(config: Mapping[str, Any]) -> None:
    """Validate an immutable run configuration, including ``run_config_id``.

    Any failure is corruption/configuration failure (CLI exit code 2). The
    stored ``repo_sha_at_freeze`` is diagnostic only and is never compared
    against the current repository state.
    """
    if not isinstance(config, Mapping):
        raise RunConfigError(f"run config must be a JSON object; got {type(config)!r}")
    missing = [key for key in _RUN_CONFIG_REQUIRED_KEYS if key not in config]
    if missing:
        raise RunConfigError(f"run config missing required fields: {', '.join(missing)}")

    if config["schema_version"] != RUN_CONFIG_SCHEMA_VERSION:
        raise RunConfigError(
            f"unsupported run config schema_version {config['schema_version']!r}"
        )
    build_id = config["build_id"]
    if not isinstance(build_id, str) or not SNAPSHOT_BUILD_ID_RE.match(build_id):
        raise RunConfigError(f"run config build_id is malformed: {build_id!r}")
    if config["mode"] != "backfill":
        raise RunConfigError(f"unsupported run config mode: {config['mode']!r}")

    requested_output_start = _parse_config_date(config, "requested_output_start")
    raw_dependency_start = _parse_config_date(config, "raw_dependency_start")
    as_of_requested = _parse_config_date(config, "as_of_requested")
    as_of_resolved = _parse_config_date(config, "as_of_resolved_trading_day")
    if raw_dependency_start > as_of_requested:
        raise RunConfigError("run config raw_dependency_start is after as_of_requested")
    if requested_output_start > as_of_requested:
        raise RunConfigError("run config requested_output_start is after as_of_requested")

    lookback_weeks = config["lookback_weeks"]
    if (
        not isinstance(lookback_weeks, int)
        or isinstance(lookback_weeks, bool)
        or lookback_weeks <= 0
    ):
        raise RunConfigError(f"run config lookback_weeks is invalid: {lookback_weeks!r}")
    expected_dep_start = compute_raw_dependency_start(
        requested_output_start, lookback_weeks
    )
    if raw_dependency_start != expected_dep_start:
        raise RunConfigError(
            "run config raw_dependency_start does not match the C4 dependency "
            f"formula: stored {raw_dependency_start.isoformat()}, expected "
            f"{expected_dep_start.isoformat()}"
        )

    for key in ("physical_raw_dates", "resolved_trading_dates"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise RunConfigError(f"run config {key} must be a nonempty list")
    physical = set(config["physical_raw_dates"])
    resolved = list(config["resolved_trading_dates"])
    if not set(resolved).issubset(physical):
        raise RunConfigError(
            "run config resolved_trading_dates is not a subset of physical_raw_dates"
        )
    if as_of_resolved.isoformat() not in resolved:
        raise RunConfigError(
            "run config as_of_resolved_trading_day is not a resolved trading date"
        )

    for key in ("c4_params", "surface_policy"):
        if not isinstance(config[key], Mapping):
            raise RunConfigError(f"run config {key} must be a mapping")

    digest = config["inventory_digest"]
    if not isinstance(digest, str) or not _HEX64_RE.match(digest):
        raise RunConfigError(f"run config inventory_digest is malformed: {digest!r}")

    stored_id = config["run_config_id"]
    expected_id = compute_run_config_id(config)
    if stored_id != expected_id:
        raise RunConfigError(
            "run_config_id mismatch — run configuration is corrupt or was "
            f"modified (stored {stored_id!r}, recomputed {expected_id!r})"
        )


def build_run_config(
    *,
    build_id: str,
    snapshots_root: Path | str,
    raw_root: Path | str,
    requested_output_start: date,
    lookback_weeks: int,
    inventory: RawInventory,
    c4_params: Mapping[str, Any],
    surface_policy: Mapping[str, Any],
    repo_sha_at_freeze: str,
) -> dict[str, Any]:
    """Assemble the immutable run configuration dict (with ``run_config_id``)."""
    c4 = dict(c4_params)
    if c4.get("lookback_weeks") != lookback_weeks:
        raise RunConfigError(
            "c4_params lookback_weeks must equal the run lookback_weeks "
            f"({c4.get('lookback_weeks')!r} != {lookback_weeks!r}); one formula only"
        )
    config: dict[str, Any] = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "build_id": build_id,
        "mode": "backfill",
        "snapshots_root": _normalize_root(snapshots_root),
        "raw_root": _normalize_root(raw_root),
        "requested_output_start": requested_output_start.isoformat(),
        "raw_dependency_start": inventory.raw_dependency_start.isoformat(),
        "as_of_requested": inventory.as_of_requested.isoformat(),
        "as_of_resolved_trading_day": inventory.as_of_resolved_trading_day.isoformat(),
        "lookback_weeks": lookback_weeks,
        "physical_raw_dates": [d.isoformat() for d in inventory.physical_raw_dates],
        "resolved_trading_dates": [
            d.isoformat() for d in inventory.resolved_trading_dates
        ],
        "c4_params": c4,
        "surface_policy": dict(surface_policy),
        "inventory_digest": inventory.inventory_digest,
        # Diagnostic evidence only. Repository SHA changes never invalidate a run.
        "repo_sha_at_freeze": repo_sha_at_freeze,
    }
    config["run_config_id"] = compute_run_config_id(config)
    return config


def _atomic_write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a frozen JSON file; refuse to overwrite an existing one."""
    if path.exists():
        raise RunConfigError(f"refusing to overwrite existing frozen file: {path}")
    # Canonicalize first so a non-serializable payload fails before touching disk.
    canonical_check = canonical_json_bytes(dict(payload))
    temp_path = path.parent / f"{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        with temp_path.open(encoding="utf-8") as handle:
            readback = json.load(handle)
        if readback != json.loads(canonical_check.decode("utf-8")):
            raise RunConfigError(f"frozen file readback mismatch for {path}")
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_run_config(building_root: Path | str) -> dict[str, Any]:
    """Load and validate the immutable run configuration of a ``.building`` run."""
    path = Path(building_root) / RUN_CONFIG_FILENAME
    if not path.is_file():
        raise RunConfigError(f"missing run configuration: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"malformed run configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunConfigError(f"run configuration must be a JSON object: {path}")
    validate_run_config(data)
    return data


def load_raw_inventory_document(building_root: Path | str) -> dict[str, Any]:
    """Load and internally validate the frozen ``raw_inventory.json`` document."""
    path = Path(building_root) / RAW_INVENTORY_FILENAME
    if not path.is_file():
        raise RunConfigError(f"missing frozen raw inventory: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"malformed frozen raw inventory {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunConfigError(f"frozen raw inventory must be a JSON object: {path}")
    missing = [key for key in _RAW_INVENTORY_REQUIRED_KEYS if key not in data]
    if missing:
        raise RunConfigError(
            f"frozen raw inventory missing required fields: {', '.join(missing)}"
        )
    identity = {key: data[key] for key in _RAW_INVENTORY_IDENTITY_KEYS}
    recomputed = digest_json(identity)
    if data["inventory_digest"] != recomputed:
        raise RunConfigError(
            "frozen raw inventory digest mismatch — inventory file is corrupt "
            f"(stored {data['inventory_digest']!r}, recomputed {recomputed!r})"
        )
    return data


def rescan_and_verify_raw_inventory(config: Mapping[str, Any]) -> RawInventory:
    """Rescan the raw root from the frozen config and reject digest drift."""
    inventory = scan_raw_inventory(
        config["raw_root"],
        _parse_config_date(config, "raw_dependency_start"),
        _parse_config_date(config, "as_of_requested"),
    )
    if inventory.inventory_digest != config["inventory_digest"]:
        raise RunConfigError(
            f"raw inventory drift detected for build {config['build_id']}: the raw "
            "root no longer matches the frozen inventory "
            f"(frozen {config['inventory_digest']!r}, rescanned "
            f"{inventory.inventory_digest!r})"
        )
    return inventory


# ── New-run preparation and resume lifecycle ──────────────────────────────────


@dataclass(frozen=True)
class PreparedRun:
    """A freshly prepared (frozen, not yet executed) backfill run."""

    build_id: str
    snapshots_root: Path
    roots: SnapshotRoots
    run_config: dict[str, Any]
    inventory: RawInventory


@dataclass(frozen=True)
class ResumedRun:
    """An explicitly named ``.building`` run opened for resume."""

    build_id: str
    snapshots_root: Path
    roots: SnapshotRoots
    run_config: dict[str, Any]
    inventory: RawInventory | None  # rescanned inventory when rescan requested


def create_snapshot_layout(building_root: Path | str) -> None:
    """Create the documented owned directory layout inside a fresh building root."""
    root = Path(building_root)
    for rel in SNAPSHOT_OWNED_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


def prepare_new_backfill_run(
    *,
    snapshots_root: Path | str,
    raw_root: Path | str,
    requested_output_start: date,
    as_of_requested: date,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    c4_params: Mapping[str, Any] | None = None,
    surface_policy: Mapping[str, Any] | None = None,
    repo_sha_at_freeze: str | None = None,
    build_id: str | None = None,
) -> PreparedRun:
    """Freeze a new cold-backfill run: inventory, layout, and immutable config.

    Scans and freezes the raw inventory, generates the build id (unless a test
    injects one), creates only a fresh ``<BUILD_ID>.building`` root (refusing
    reuse of an existing building or final root), creates the owned directory
    layout, and atomically writes ``run_config.json`` and ``raw_inventory.json``
    — validating both after writing. Does not run producers, acquire the lock,
    or publish anything.
    """
    if requested_output_start > as_of_requested:
        raise RunConfigError(
            "cannot resolve run bounds: requested_output_start "
            f"{requested_output_start.isoformat()} is after as_of_requested "
            f"{as_of_requested.isoformat()}"
        )
    if c4_params is None:
        c4_params = {**DEFAULT_C4_PARAMS, "lookback_weeks": lookback_weeks}
    if surface_policy is None:
        surface_policy = DEFAULT_SURFACE_POLICY

    raw_dependency_start = compute_raw_dependency_start(
        requested_output_start, lookback_weeks
    )
    inventory = scan_raw_inventory(raw_root, raw_dependency_start, as_of_requested)

    if build_id is None:
        build_id = generate_snapshot_build_id()
    if repo_sha_at_freeze is None:
        repo_sha_at_freeze = current_repo_sha()

    config = build_run_config(
        build_id=build_id,
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=requested_output_start,
        lookback_weeks=lookback_weeks,
        inventory=inventory,
        c4_params=c4_params,
        surface_policy=surface_policy,
        repo_sha_at_freeze=repo_sha_at_freeze,
    )

    roots = create_fresh_staging_root(snapshots_root, build_id)
    create_snapshot_layout(roots.building)

    _atomic_write_frozen_json(roots.building / RUN_CONFIG_FILENAME, config)
    _atomic_write_frozen_json(
        roots.building / RAW_INVENTORY_FILENAME, raw_inventory_document(inventory)
    )

    # Post-write validation: the frozen files must load and self-verify.
    loaded_config = load_run_config(roots.building)
    loaded_inventory = load_raw_inventory_document(roots.building)
    if loaded_inventory["inventory_digest"] != loaded_config["inventory_digest"]:
        raise RunConfigError(
            "frozen inventory digest does not match run configuration after write"
        )

    return PreparedRun(
        build_id=build_id,
        snapshots_root=Path(snapshots_root),
        roots=roots,
        run_config=loaded_config,
        inventory=inventory,
    )


def open_resume_run(
    snapshots_root: Path | str,
    build_id: str,
    *,
    rescan_raw: bool = True,
) -> ResumedRun:
    """Open only the explicitly named ``<BUILD_ID>.building`` run for resume.

    Never opens a final snapshot for mutation and never rewrites frozen files.
    Validates the immutable run configuration, the frozen raw inventory, and
    (by default) rescans the raw root and rejects digest drift. All validation
    failures are corruption/configuration failures (``RunConfigError``, CLI
    exit code 2). A changed repository SHA alone never blocks resume.
    """
    if not isinstance(build_id, str) or not SNAPSHOT_BUILD_ID_RE.match(build_id):
        raise RunConfigError(f"malformed build id for resume: {build_id!r}")

    roots = derive_snapshot_roots(snapshots_root, build_id)
    if not roots.building.is_dir():
        if roots.final.exists():
            raise RunConfigError(
                f"refusing to resume {build_id}: only a final snapshot exists at "
                f"{roots.final}; resume opens only the named .building root and "
                "never mutates a published snapshot"
            )
        raise RunConfigError(f"no .building run to resume: {roots.building}")
    if roots.final.exists():
        raise RunConfigError(
            f"corrupt lifecycle for {build_id}: both {roots.building} and "
            f"{roots.final} exist"
        )

    config = load_run_config(roots.building)
    normalized_snapshots_root = _normalize_root(snapshots_root)
    if normalized_snapshots_root != config["snapshots_root"]:
        raise RunConfigError(
            f"snapshot root mismatch for build {build_id}: supplied "
            f"{normalized_snapshots_root!r}, frozen {config['snapshots_root']!r}; "
            "relocating or copying a .building run to another snapshot root is "
            "not permitted"
        )
    if config["build_id"] != build_id:
        raise RunConfigError(
            f"run configuration build_id {config['build_id']!r} does not match "
            f"the resumed root {build_id!r}"
        )
    inventory_doc = load_raw_inventory_document(roots.building)
    if inventory_doc["inventory_digest"] != config["inventory_digest"]:
        raise RunConfigError(
            "frozen raw inventory digest does not match the run configuration "
            f"for build {build_id}"
        )

    inventory = rescan_and_verify_raw_inventory(config) if rescan_raw else None

    return ResumedRun(
        build_id=build_id,
        snapshots_root=Path(snapshots_root),
        roots=roots,
        run_config=config,
        inventory=inventory,
    )
