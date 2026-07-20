"""Resumable cold-backfill snapshot orchestration (Sprint 004 C8.3B).

This module holds direct functions and small dataclasses for:

* frozen raw ORATS ZIP inventory discovery (central-directory evidence, no
  full-byte hashing of large archives);
* immutable ``run_config.json`` / ``raw_inventory.json`` freeze, load, and
  validation with a self-verifying ``run_config_id``;
* new-run preparation (fresh ``<BUILD_ID>.building`` root plus the owned
  directory layout) and resume-open of an explicitly named ``.building`` run;
* raw-inventory rescan with digest-drift rejection on resume;
* four completion markers and stage-boundary resume over the fixed order
  liquidity → adjusted → spot → surface;
* final cross-stage validation and schema-v1 candidate-manifest construction
  while the root remains named ``.building``;
* atomic publication of a finalized candidate (``work/`` removal + rename to
  the final root) under an already-held sibling lock.

CLI argument parsing and lock acquisition/release remain in
``scripts/refresh_weekly_inputs.py``. There is no framework, DAG, plugin,
receipt, or state-machine abstraction here — and no ``.failed`` lifecycle.

Design: docs/tmp/c8_3b_resumable_cold_backfill_design.md
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.data.input_snapshot import (
    ARTIFACT_ADJUSTED_CHAINS_ROOT,
    ARTIFACT_LIQUID_TICKERS,
    ARTIFACT_LIQUIDITY_DAILY,
    ARTIFACT_LIQUIDITY_PANEL,
    ARTIFACT_LIQUIDITY_WEEKLY,
    ARTIFACT_OPTION_SURFACE_META,
    ARTIFACT_OPTION_SURFACE_QUOTES,
    ARTIFACT_SPOT_PRICES,
    ARTIFACT_SPLITS,
    INPUT_SNAPSHOT_SCHEMA_VERSION,
    InputSnapshotManifest,
    compute_snapshot_id,
    read_manifest,
    write_manifest,
)
from src.data.pit_universe_audit import compute_reference_universe
from src.data.security_types import classification_digest
from src.data.snapshot_foundation import (
    SNAPSHOT_BUILD_ID_RE,
    SiblingBuildLock,
    SnapshotFoundationError,
    SnapshotLifecycleError,
    SnapshotLockError,
    SnapshotPathError,
    SnapshotRoots,
    adjusted_inventory_digest,
    canonical_json_bytes,
    create_fresh_staging_root,
    derive_snapshot_roots,
    digest_json,
    generate_snapshot_build_id,
    resolve_under_root,
    sha256_file,
    sibling_lock_path,
    ticker_date_keys_digest,
    validate_same_volume_publication,
)
from src.data.ticker_universe import load_ticker_universe

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


# ── Completion markers and stage-boundary resume ───────────────────────────────


STAGE_ORDER: tuple[str, ...] = ("liquidity", "adjusted", "spot", "surface")
STAGE_CONTRACT_VERSION = "1"

STAGE_OWNED_DIRS: dict[str, tuple[str, ...]] = {
    "liquidity": ("work/liquidity", "input/liquidity", "reports/liquidity"),
    "adjusted": ("work/adjusted", "input/adjusted_liquid", "reports/adjusted"),
    "spot": ("work/spot", "cache/spot", "reports/spot"),
    "surface": ("work/surface", "cache/surface", "reports/surface"),
}

_LIQUIDITY_ARTIFACTS = (
    "input/liquidity/ticker_liquidity_daily_observations.parquet",
    "input/liquidity/ticker_liquidity_weekly_observations.parquet",
    "input/liquidity/ticker_liquidity_panel.parquet",
    "input/liquidity/liquid_tickers.csv",
    "input/liquidity/security_classification.parquet",
)
_LIQUIDITY_REPORT = "reports/liquidity/pit_universe_audit.md"
_ADJUSTED_SPLITS = "input/adjusted_liquid/splits_hist_liquid.parquet"
_ADJUSTED_REPORT = "reports/adjusted/adjusted_liquid_audit.md"
_SPOT_OUTPUT = "cache/spot/spot_prices_adjusted.parquet"
_SPOT_SUMMARY = "cache/spot/spot_summary.json"
_SPOT_REPORT = "reports/spot/gate_spot_reconciliation.json"
_SURFACE_REPORT = "reports/surface/surface_contract_checks.json"

_MARKER_REQUIRED_FIELDS = (
    "stage",
    "stage_contract_version",
    "run_config_id",
    "stage_accepted",
    "completed_at_utc",
    "producer_repo_sha",
    "gate_status",
    "required_paths",
    "accepted_warnings",
    "evidence",
)

_SPOT_WARN_MARKERS = (
    "ambiguous ticker-date exclusion(s) present",
    "inconsistent repeated spot values",
)
_SPOT_WARN_REASON = "reconciled_ambiguous_ticker_date_exclusion"
_SURFACE_WARN_REASON = "informational_invalid_meta_with_quotes"
_SURFACE_WARN_MARKER = "surface_valid=False metadata row(s) have quote rows"


@dataclass(frozen=True)
class StageResumeState:
    """Result of inspecting completion markers in fixed stage order."""

    completed_stages: tuple[str, ...]
    next_stage: str | None
    validated_markers: dict[str, dict[str, Any]] = field(default_factory=dict)


def stage_marker_path(building_root: Path | str, stage: str) -> Path:
    """Return ``markers/<stage>.done.json`` under the building root."""
    if stage not in STAGE_ORDER:
        raise RunConfigError(f"unknown stage name: {stage!r}")
    return Path(building_root) / "markers" / f"{stage}.done.json"


def _building_rel(building: Path, path: Path | str, *, label: str) -> str:
    """Return a forward-slashed path relative to ``building``, fail-closed."""
    absolute = Path(path).resolve(strict=False)
    root = building.resolve(strict=False)
    try:
        rel = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise RunConfigError(
            f"{label} is not under .building root {root}: {path}"
        ) from exc
    # Round-trip through the shared containment helper.
    resolve_under_root(building, rel, label=label)
    return rel


def _require_rel_file(building: Path, rel: str, *, label: str) -> Path:
    try:
        path = resolve_under_root(building, rel, label=label)
    except SnapshotPathError as exc:
        raise RunConfigError(str(exc)) from exc
    if not path.is_file():
        raise RunConfigError(f"{label} missing required file: {rel}")
    return path


def _as_building_rel(building: Path, value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunConfigError(f"{label} must be a nonempty string")
    path = Path(value)
    if path.is_absolute():
        return _building_rel(building, path, label=label)
    return value.replace("\\", "/")


def _require_marker_fields(marker: Mapping[str, Any], stage: str) -> None:
    missing = [k for k in _MARKER_REQUIRED_FIELDS if k not in marker]
    if missing:
        raise RunConfigError(
            f"{stage} marker missing required fields: {', '.join(missing)}"
        )
    if marker["stage"] != stage:
        raise RunConfigError(
            f"{stage} marker stage field is {marker['stage']!r}"
        )
    if marker["stage_contract_version"] != STAGE_CONTRACT_VERSION:
        raise RunConfigError(
            f"{stage} marker has incompatible stage_contract_version "
            f"{marker['stage_contract_version']!r}"
        )
    if marker["stage_accepted"] is not True:
        raise RunConfigError(f"{stage} marker stage_accepted is not true")
    if not isinstance(marker["required_paths"], list) or not marker["required_paths"]:
        raise RunConfigError(f"{stage} marker required_paths must be a nonempty list")
    if not isinstance(marker["accepted_warnings"], list):
        raise RunConfigError(f"{stage} marker accepted_warnings must be a list")
    if not isinstance(marker["evidence"], Mapping):
        raise RunConfigError(f"{stage} marker evidence must be an object")
    if not isinstance(marker["producer_repo_sha"], str) or not marker["producer_repo_sha"]:
        raise RunConfigError(f"{stage} marker producer_repo_sha must be a nonempty string")
    if not isinstance(marker["completed_at_utc"], str) or not marker["completed_at_utc"]:
        raise RunConfigError(f"{stage} marker completed_at_utc must be a nonempty string")


def _normalize_accepted_warnings(stage: str, evidence: Mapping[str, Any]) -> list[dict[str, str]]:
    """Map adapter warnings to the narrow marker acceptance policy."""
    raw = list(evidence.get("accepted_warnings") or [])
    status = evidence.get("status")
    if stage in ("liquidity", "adjusted"):
        if raw:
            raise RunConfigError(
                f"{stage} evidence must not contain accepted_warnings"
            )
        return []
    if status == "PASS":
        if raw:
            raise RunConfigError(
                f"{stage} PASS evidence must not contain accepted_warnings"
            )
        return []
    if status != "WARN":
        raise RunConfigError(
            f"{stage} evidence status must be PASS or WARN; got {status!r}"
        )
    if stage == "spot":
        if not raw or not evidence.get("ambiguous_exclusion_count"):
            raise RunConfigError(
                "spot WARN evidence must describe only the reconciled "
                "ambiguous-key case"
            )
        if not all(
            any(marker in str(w) for marker in _SPOT_WARN_MARKERS) for w in raw
        ):
            raise RunConfigError(
                "spot accepted_warnings are not the reconciled ambiguous-key case"
            )
        return [{"warning": str(w), "reason": _SPOT_WARN_REASON} for w in raw]
    if stage == "surface":
        if not raw or not all(_SURFACE_WARN_MARKER in str(w) for w in raw):
            raise RunConfigError(
                "surface WARN evidence must describe only the "
                "a1_a2_join_integrity informational case"
            )
        return [{"warning": str(w), "reason": _SURFACE_WARN_REASON} for w in raw]
    raise RunConfigError(f"unknown stage for warning normalization: {stage!r}")


def _bind_marker_envelope(stage: str, marker: Mapping[str, Any]) -> None:
    """Require marker envelope fields to match the embedded evidence."""
    evidence = marker["evidence"]
    if not isinstance(evidence, Mapping):
        raise RunConfigError(f"{stage} marker evidence must be an object")
    if evidence.get("stage") != stage:
        raise RunConfigError(
            f"{stage} marker evidence.stage is {evidence.get('stage')!r}"
        )
    if marker.get("gate_status") != evidence.get("status"):
        raise RunConfigError(
            f"{stage} marker gate_status {marker.get('gate_status')!r} does not "
            f"match evidence.status {evidence.get('status')!r}"
        )
    expected = _normalize_accepted_warnings(stage, evidence)
    if marker.get("accepted_warnings") != expected:
        raise RunConfigError(
            f"{stage} marker accepted_warnings do not match evidence warnings"
        )


def _validate_acceptance_report(
    building: Path, stage: str, evidence: Mapping[str, Any]
) -> None:
    """Read the stage acceptance report and bind it to evidence (no re-audit)."""
    if stage == "liquidity":
        text = _require_rel_file(
            building, _LIQUIDITY_REPORT, label="liquidity report"
        ).read_text(encoding="utf-8")
        if "- overall status: `PASS`" not in text:
            raise RunConfigError("liquidity C7 report does not record overall PASS")
        if "- strict mode: `True`" not in text:
            raise RunConfigError("liquidity C7 report does not record strict mode")
        return
    if stage == "adjusted":
        text = _require_rel_file(
            building, _ADJUSTED_REPORT, label="adjusted report"
        ).read_text(encoding="utf-8")
        if "**PASS WITH WARNINGS**" in text:
            raise RunConfigError(
                "adjusted C5 report is PASS WITH WARNINGS; exact PASS is required"
            )
        if "## Overall verdict: **PASS**" not in text:
            raise RunConfigError("adjusted C5 report does not record exact PASS")
        return
    if stage == "spot":
        path = _require_rel_file(building, _SPOT_REPORT, label="spot report")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunConfigError(f"malformed spot report: {exc}") from exc
        if not isinstance(report, Mapping):
            raise RunConfigError("spot report must be a JSON object")
        if report.get("status") != evidence.get("status"):
            raise RunConfigError("spot report status does not match evidence.status")
        if report.get("failures"):
            raise RunConfigError("spot report failures must be empty")
        if list(report.get("warnings") or []) != list(
            evidence.get("accepted_warnings") or []
        ):
            raise RunConfigError(
                "spot report warnings do not match evidence.accepted_warnings"
            )
        return
    if stage == "surface":
        path = _require_rel_file(building, _SURFACE_REPORT, label="surface report")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunConfigError(f"malformed surface report: {exc}") from exc
        if not isinstance(report, Mapping):
            raise RunConfigError("surface report must be a JSON object")
        if report.get("overall_verdict") != evidence.get("status"):
            raise RunConfigError(
                "surface report overall_verdict does not match evidence.status"
            )
        checks = report.get("checks") or []
        if not isinstance(checks, list):
            raise RunConfigError("surface report checks must be a list")
        if any(isinstance(c, Mapping) and c.get("status") == "FAIL" for c in checks):
            raise RunConfigError("surface report contains a FAIL check")
        failures = [
            f
            for c in checks
            if isinstance(c, Mapping)
            for f in (c.get("failures") or [])
        ]
        if failures:
            raise RunConfigError("surface report failures must be empty")
        flattened = [
            w
            for c in checks
            if isinstance(c, Mapping)
            for w in (c.get("warnings") or [])
        ]
        if flattened != list(evidence.get("accepted_warnings") or []):
            raise RunConfigError(
                "surface report warnings do not match evidence.accepted_warnings"
            )
        return
    raise RunConfigError(f"unknown stage for report validation: {stage!r}")


def _evidence_required_paths(
    stage: str, evidence: Mapping[str, Any], building: Path
) -> list[str]:
    """Derive the marker required_paths list (building-relative)."""
    if stage == "liquidity":
        return list(_LIQUIDITY_ARTIFACTS) + [_LIQUIDITY_REPORT]
    if stage == "spot":
        return [_SPOT_OUTPUT, _SPOT_SUMMARY, _SPOT_REPORT]
    if stage == "surface":
        meta = evidence.get("meta_path")
        quotes = evidence.get("quotes_path")
        if not isinstance(meta, str) or not isinstance(quotes, str):
            raise RunConfigError("surface evidence missing meta_path or quotes_path")
        return [meta, quotes, _SURFACE_REPORT]
    if stage != "adjusted":
        raise RunConfigError(f"unknown stage: {stage!r}")
    paths = [_ADJUSTED_SPLITS, _ADJUSTED_REPORT]
    adj_root = building / "input" / "adjusted_liquid"
    for parquet in sorted(adj_root.glob("*/ORATS_SMV_Strikes_*.parquet")):
        paths.append(_building_rel(building, parquet, label="adjusted artifact"))
    return paths


def _relativize_evidence(building: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(evidence)
    for key in (
        "output_dir",
        "report_path",
        "output_path",
        "summary_path",
        "meta_path",
        "quotes_path",
    ):
        if key in out and out[key]:
            out[key] = _as_building_rel(building, out[key], label=key)
    return out


def _validate_liquidity_evidence(building: Path, evidence: Mapping[str, Any]) -> None:
    if evidence.get("status") != "PASS":
        raise RunConfigError("liquidity evidence gate status must be PASS")
    for rel in _LIQUIDITY_ARTIFACTS:
        _require_rel_file(building, rel, label="liquidity")
    _require_rel_file(building, _LIQUIDITY_REPORT, label="liquidity report")
    universe = load_ticker_universe(building / "input" / "liquidity" / "liquid_tickers.csv")
    if evidence.get("equity_universe_digest") != digest_json(sorted(universe)):
        raise RunConfigError("liquidity equity_universe_digest mismatch")
    classification = pd.read_parquet(
        building / "input" / "liquidity" / "security_classification.parquet"
    )
    if evidence.get("classification_digest") != classification_digest(classification):
        raise RunConfigError("liquidity classification_digest mismatch")


def _validate_adjusted_evidence(building: Path, evidence: Mapping[str, Any]) -> None:
    if evidence.get("status") != "PASS" or evidence.get("audit_verdict") != "PASS":
        raise RunConfigError("adjusted evidence must record strict C5 PASS")
    splits = _require_rel_file(building, _ADJUSTED_SPLITS, label="adjusted splits")
    _require_rel_file(building, _ADJUSTED_REPORT, label="adjusted report")
    if evidence.get("split_metadata_hash") != sha256_file(splits):
        raise RunConfigError("adjusted split_metadata_hash mismatch")
    liquid = building / "input" / "liquidity" / "liquid_tickers.csv"
    if not liquid.is_file():
        raise RunConfigError("adjusted validation requires certified liquid_tickers.csv")
    universe = load_ticker_universe(liquid)
    if evidence.get("universe_digest") != digest_json(sorted(universe)):
        raise RunConfigError("adjusted universe_digest mismatch")
    adj_root = building / "input" / "adjusted_liquid"
    parquets = sorted(adj_root.glob("*/ORATS_SMV_Strikes_*.parquet"))
    if not parquets:
        raise RunConfigError("adjusted stable output has no daily parquet files")
    if evidence.get("adjusted_inventory_digest") != adjusted_inventory_digest(
        adj_root, parquets
    ):
        raise RunConfigError("adjusted_inventory_digest mismatch")
    total_bytes = sum(p.stat().st_size for p in adj_root.rglob("*") if p.is_file())
    file_count = sum(1 for p in adj_root.rglob("*") if p.is_file())
    if evidence.get("output_total_bytes") != total_bytes:
        raise RunConfigError("adjusted output_total_bytes mismatch")
    if evidence.get("output_file_count") != file_count:
        raise RunConfigError("adjusted output_file_count mismatch")
    if evidence.get("date_count") != len(parquets):
        raise RunConfigError("adjusted date_count mismatch")


def _validate_spot_evidence(building: Path, evidence: Mapping[str, Any]) -> None:
    status = evidence.get("status")
    if status not in ("PASS", "WARN"):
        raise RunConfigError(f"spot evidence status invalid: {status!r}")
    output = _require_rel_file(building, _SPOT_OUTPUT, label="spot output")
    summary_path = _require_rel_file(building, _SPOT_SUMMARY, label="spot summary")
    _require_rel_file(building, _SPOT_REPORT, label="spot report")
    if evidence.get("output_total_bytes") != output.stat().st_size:
        raise RunConfigError("spot output_total_bytes mismatch")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    for key, summary_key in (
        ("source_key_count", "source_ticker_date_key_count"),
        ("source_key_digest", "source_ticker_date_key_digest"),
        ("output_key_count", "output_ticker_date_key_count"),
        ("output_key_digest", "output_ticker_date_key_digest"),
        ("ambiguous_exclusion_count", "ambiguous_exclusion_count"),
        ("output_row_count", "output_row_count"),
    ):
        if evidence.get(key) != summary.get(summary_key):
            raise RunConfigError(f"spot evidence {key} mismatch versus summary")
    frame = pd.read_parquet(output, columns=["date", "ticker"])
    output_keys = {
        (d, str(t)) for d, t in zip(frame["date"], frame["ticker"])
    }
    if evidence.get("output_key_digest") != ticker_date_keys_digest(output_keys):
        raise RunConfigError("spot output_key_digest mismatch versus parquet")
    if evidence.get("output_key_count") != len(output_keys):
        raise RunConfigError("spot output_key_count mismatch versus parquet")


def _validate_surface_evidence(building: Path, evidence: Mapping[str, Any]) -> None:
    status = evidence.get("status")
    if status not in ("PASS", "WARN"):
        raise RunConfigError(f"surface evidence status invalid: {status!r}")
    meta_rel = _as_building_rel(building, evidence.get("meta_path"), label="meta_path")
    quotes_rel = _as_building_rel(
        building, evidence.get("quotes_path"), label="quotes_path"
    )
    meta_path = _require_rel_file(building, meta_rel, label="surface meta")
    quotes_path = _require_rel_file(building, quotes_rel, label="surface quotes")
    _require_rel_file(building, _SURFACE_REPORT, label="surface report")
    if evidence.get("meta_total_bytes") != meta_path.stat().st_size:
        raise RunConfigError("surface meta_total_bytes mismatch")
    if evidence.get("quotes_total_bytes") != quotes_path.stat().st_size:
        raise RunConfigError("surface quotes_total_bytes mismatch")
    meta_df = pd.read_parquet(meta_path)
    quotes_df = pd.read_parquet(quotes_path)
    actual_keys = {
        (pd.Timestamp(d).date(), str(t).strip().upper())
        for t, d in zip(meta_df["ticker"], meta_df["entry_date"])
    }
    if evidence.get("actual_a1_key_count") != len(actual_keys):
        raise RunConfigError("surface actual_a1_key_count mismatch")
    if evidence.get("actual_a1_key_digest") != ticker_date_keys_digest(actual_keys):
        raise RunConfigError("surface actual_a1_key_digest mismatch")
    if evidence.get("expected_a1_key_count") != evidence.get("actual_a1_key_count"):
        raise RunConfigError("surface expected/actual A1 counts disagree")
    if evidence.get("expected_a1_key_digest") != evidence.get("actual_a1_key_digest"):
        raise RunConfigError("surface expected/actual A1 digests disagree")
    if evidence.get("a2_row_count") != len(quotes_df):
        raise RunConfigError("surface a2_row_count mismatch")
    a2_grain = sorted(
        [
            str(t),
            pd.Timestamp(e).date().isoformat(),
            pd.Timestamp(x).date().isoformat(),
            float(s),
            str(side),
        ]
        for t, e, x, s, side in zip(
            quotes_df["ticker"],
            quotes_df["entry_date"],
            quotes_df["expiry_date"],
            quotes_df["strike"],
            quotes_df["side"],
        )
    )
    if evidence.get("a2_grain_digest") != digest_json(a2_grain):
        raise RunConfigError("surface a2_grain_digest mismatch")


def validate_stage_evidence(
    building_root: Path | str,
    stage: str,
    evidence: Mapping[str, Any],
) -> None:
    """Validate compact adapter evidence against stable on-disk artifacts."""
    if stage not in STAGE_ORDER:
        raise RunConfigError(f"unknown stage: {stage!r}")
    building = Path(building_root)
    if stage == "liquidity":
        _validate_liquidity_evidence(building, evidence)
    elif stage == "adjusted":
        _validate_adjusted_evidence(building, evidence)
    elif stage == "spot":
        _validate_spot_evidence(building, evidence)
    else:
        _validate_surface_evidence(building, evidence)
    _normalize_accepted_warnings(stage, evidence)
    _validate_acceptance_report(building, stage, evidence)


def build_stage_marker(
    *,
    building_root: Path | str,
    stage: str,
    run_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    producer_repo_sha: str,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a marker document from accepted adapter evidence (no write)."""
    if stage not in STAGE_ORDER:
        raise RunConfigError(f"unknown stage: {stage!r}")
    building = Path(building_root)
    if evidence.get("stage") != stage:
        raise RunConfigError(
            f"evidence stage {evidence.get('stage')!r} does not match {stage!r}"
        )
    relative_evidence = _relativize_evidence(building, evidence)
    validate_stage_evidence(building, stage, relative_evidence)
    warnings = _normalize_accepted_warnings(stage, relative_evidence)
    required = _evidence_required_paths(stage, relative_evidence, building)
    for rel in required:
        _require_rel_file(building, rel, label=f"{stage} required_paths")
    gate_status = evidence.get("status")
    if stage in ("liquidity", "adjusted") and gate_status != "PASS":
        raise RunConfigError(f"{stage} marker gate_status must be PASS")
    if stage in ("spot", "surface") and gate_status not in ("PASS", "WARN"):
        raise RunConfigError(f"{stage} marker gate_status must be PASS or WARN")
    if completed_at_utc is None:
        completed_at_utc = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
    return {
        "stage": stage,
        "stage_contract_version": STAGE_CONTRACT_VERSION,
        "run_config_id": run_config["run_config_id"],
        "stage_accepted": True,
        "completed_at_utc": completed_at_utc,
        "producer_repo_sha": producer_repo_sha,
        "gate_status": gate_status,
        "required_paths": required,
        "accepted_warnings": warnings,
        "evidence": relative_evidence,
    }


def write_stage_marker(
    building_root: Path | str,
    marker: Mapping[str, Any],
) -> Path:
    """Atomically write a stage marker; refuse to overwrite an existing one."""
    stage = marker.get("stage")
    if stage not in STAGE_ORDER:
        raise RunConfigError(f"marker has unknown stage: {stage!r}")
    path = stage_marker_path(building_root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_frozen_json(path, marker)
    return path


def load_and_validate_stage_marker(
    building_root: Path | str,
    stage: str,
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one marker and validate contract + compact evidence (corruption → 2)."""
    path = stage_marker_path(building_root, stage)
    if not path.is_file():
        raise RunConfigError(f"missing stage marker: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"malformed stage marker {path}: {exc}") from exc
    if not isinstance(marker, dict):
        raise RunConfigError(f"stage marker must be a JSON object: {path}")
    _require_marker_fields(marker, stage)
    if marker["run_config_id"] != run_config["run_config_id"]:
        raise RunConfigError(
            f"{stage} marker run_config_id mismatch: marker "
            f"{marker['run_config_id']!r} vs run {run_config['run_config_id']!r}"
        )
    _bind_marker_envelope(stage, marker)
    for rel in marker["required_paths"]:
        if not isinstance(rel, str):
            raise RunConfigError(f"{stage} marker required_paths entries must be strings")
        _require_rel_file(Path(building_root), rel, label=f"{stage} required_paths")
    validate_stage_evidence(building_root, stage, marker["evidence"])
    # producer_repo_sha is diagnostic only — never compared to current HEAD.
    return marker


def inspect_stage_markers(
    building_root: Path | str,
    run_config: Mapping[str, Any],
) -> StageResumeState:
    """Inspect markers in fixed order and return the accepted resume prefix."""
    building = Path(building_root)
    completed: list[str] = []
    validated: dict[str, dict[str, Any]] = {}
    next_stage: str | None = None
    saw_missing = False

    for stage in STAGE_ORDER:
        path = stage_marker_path(building, stage)
        if not path.is_file():
            if next_stage is None:
                next_stage = stage
            saw_missing = True
            continue
        if saw_missing:
            raise RunConfigError(
                f"corrupt marker layout: found {stage} marker after a missing "
                "predecessor stage"
            )
        validated[stage] = load_and_validate_stage_marker(building, stage, run_config)
        completed.append(stage)

    return StageResumeState(
        completed_stages=tuple(completed),
        next_stage=next_stage,
        validated_markers=validated,
    )


def _reset_stage_owned_dirs(building: Path, stage: str) -> None:
    for rel in STAGE_OWNED_DIRS[stage]:
        path = building / rel
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def execute_backfill_stages(
    run: PreparedRun | ResumedRun,
    lock: SiblingBuildLock,
    *,
    producer_repo_sha: str | None = None,
    max_workers: int | None = None,
    surface_workers: int | None = None,
    stage_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run incomplete producer stages under an already-held sibling lock.

    Inspects and validates markers before any cleanup. Skips the valid accepted
    prefix. For each remaining stage: resets only that stage's owned directories,
    invokes the existing adapter, validates evidence, and atomically writes the
    marker. Does not acquire or release ``lock``.
    """
    if not lock.held:
        raise SnapshotLockError(
            "execute_backfill_stages requires the sibling lock to already be held"
        )
    building = Path(run.roots.building)
    config = run.run_config
    state = inspect_stage_markers(building, config)
    if state.next_stage is None:
        return dict(state.validated_markers)

    if producer_repo_sha is None:
        producer_repo_sha = current_repo_sha()

    def _default_runner(stage: str) -> dict[str, Any]:
        # Lazy import avoids a circular import with snapshot_stage_adapters.
        from src.data import snapshot_stage_adapters as adapters

        if stage == "liquidity":
            return adapters.run_liquidity_stage(run)
        if stage == "adjusted":
            return adapters.run_adjusted_stage(run, max_workers=max_workers)
        if stage == "spot":
            return adapters.run_spot_stage(run)
        return adapters.run_surface_stage(run, workers=surface_workers)

    runner = stage_runner or _default_runner
    validated = dict(state.validated_markers)
    start_index = STAGE_ORDER.index(state.next_stage)

    for stage in STAGE_ORDER[start_index:]:
        _reset_stage_owned_dirs(building, stage)
        evidence = runner(stage)
        marker = build_stage_marker(
            building_root=building,
            stage=stage,
            run_config=config,
            evidence=evidence,
            producer_repo_sha=producer_repo_sha,
        )
        write_stage_marker(building, marker)
        validated[stage] = load_and_validate_stage_marker(building, stage, config)

    return validated


# ── Final cross-stage validation and candidate manifest ────────────────────────


_FINAL_REPORT_REL = "reports/final/final_validation.json"
_ORATS_RAW_REBUILD = "orats_raw_rebuild"


def _adjusted_dates(building: Path) -> list[date]:
    dates: list[date] = []
    for path in sorted(
        (building / "input" / "adjusted_liquid").glob("*/ORATS_SMV_Strikes_*.parquet")
    ):
        try:
            dates.append(datetime.strptime(path.stem.rsplit("_", 1)[-1], "%Y%m%d").date())
        except ValueError as exc:
            raise RunConfigError(
                f"adjusted parquet filename is not a trade date: {path.name}"
            ) from exc
    return dates


def _spot_dates(building: Path) -> list[date]:
    frame = pd.read_parquet(building / _SPOT_OUTPUT, columns=["date"])
    return sorted({pd.Timestamp(d).date() for d in frame["date"]})


def _cross_check_identity(
    building: Path,
    config: Mapping[str, Any],
    markers: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], str]:
    universe = sorted(
        load_ticker_universe(building / "input" / "liquidity" / "liquid_tickers.csv")
    )
    digest = digest_json(universe)
    liq = markers["liquidity"]["evidence"].get("equity_universe_digest")
    adj = markers["adjusted"]["evidence"].get("universe_digest")
    if liq != digest or adj != digest:
        raise RunConfigError(
            "cross-stage universe identity disagreement "
            f"(liquidity={liq!r}, adjusted={adj!r}, recomputed={digest!r})"
        )
    supported = [
        date.fromisoformat(d)
        for d in markers["surface"]["evidence"]["supported_entry_dates"]
    ]
    expected = {(d, t) for t in universe for d in supported}
    surf = markers["surface"]["evidence"]
    if (
        surf.get("expected_a1_key_count") != len(expected)
        or surf.get("expected_a1_key_digest") != ticker_date_keys_digest(expected)
    ):
        raise RunConfigError("recomputed Surface expected A1 count/digest mismatch")
    physical = [date.fromisoformat(d) for d in config["physical_raw_dates"]]
    resolved = [date.fromisoformat(d) for d in config["resolved_trading_dates"]]
    if _adjusted_dates(building) != physical:
        raise RunConfigError(
            "adjusted daily parquet dates disagree with frozen physical_raw_dates"
        )
    if _spot_dates(building) != resolved:
        raise RunConfigError(
            "spot resolved dates disagree with frozen resolved_trading_dates"
        )
    if any(d not in set(resolved) for d in supported):
        raise RunConfigError(
            "surface supported_entry_dates are outside frozen resolved inventory"
        )
    return universe, digest


def _select_feature_ready(
    building: Path,
    markers: Mapping[str, Mapping[str, Any]],
    universe: list[str],
    config: Mapping[str, Any],
) -> tuple[date | None, date | None, int]:
    schedule = [
        date.fromisoformat(d)
        for d in markers["surface"]["evidence"]["supported_entry_dates"]
    ]
    panel = pd.read_parquet(
        building / "input" / "liquidity" / "ticker_liquidity_panel.parquet"
    )
    c4 = config["c4_params"]
    dvol_top_pct = float(c4["dvol_top_pct"])
    spread_bot_pct = float(c4["spread_bot_pct"])
    adjusted_set, spot_set = set(_adjusted_dates(building)), set(_spot_dates(building))
    meta = pd.read_parquet(
        building / markers["surface"]["evidence"]["meta_path"],
        columns=["ticker", "entry_date", "expiry_date"],
    )
    rows = [
        (str(t).strip().upper(), pd.Timestamp(e).date(), pd.Timestamp(x).date())
        for t, e, x in zip(meta["ticker"], meta["entry_date"], meta["expiry_date"])
    ]
    universe_set = set(universe)
    ready: set[date] = set()
    for entry in schedule:
        pit = compute_reference_universe(
            entry, panel, dvol_top_pct, spread_bot_pct
        )
        resolved = pit.resolved_snapshot_date
        pit_ok = (
            resolved is not None
            and pd.Timestamp(resolved).date() < entry
            and pit.selected_count > 0
        )
        expiries = {x for _t, e, x in rows if e == entry}
        present = {t for t, e, _x in rows if e == entry}
        if (
            pit_ok
            and expiries
            and entry in adjusted_set
            and entry in spot_set
            and all(x in adjusted_set and x in spot_set for x in expiries)
            and present == universe_set
        ):
            ready.add(entry)
    best_start = best_end = None
    best_count = 0
    i = 0
    while i < len(schedule):
        if schedule[i] not in ready:
            i += 1
            continue
        j = i
        while j < len(schedule) and schedule[j] in ready:
            j += 1
        if (j - i) > best_count:
            best_start, best_end, best_count = schedule[i], schedule[j - 1], j - i
        i = j
    return best_start, best_end, best_count


def _manifest_rel_ok(building: Path, rel: str, *, label: str, directory: bool = False) -> str:
    if not isinstance(rel, str) or not rel:
        raise RunConfigError(f"{label} must be a nonempty relative path")
    normalized = rel.replace("\\", "/")
    parts = Path(normalized).parts
    if any(p in ("work", "candidate") or p.startswith(".tmp") for p in parts):
        raise RunConfigError(f"{label} may not reference work/, candidate, or temps: {rel}")
    if any(p.endswith(".building") for p in parts):
        raise RunConfigError(f"{label} may not reference a .building path: {rel}")
    try:
        path = resolve_under_root(building, normalized, label=label)
    except SnapshotPathError as exc:
        raise RunConfigError(str(exc)) from exc
    if directory:
        if not path.is_dir():
            raise RunConfigError(f"{label} missing required directory: {rel}")
    elif not path.is_file():
        raise RunConfigError(f"{label} missing required file: {rel}")
    return normalized


def finalize_candidate_snapshot(
    run: PreparedRun | ResumedRun,
    lock: SiblingBuildLock,
) -> tuple[InputSnapshotManifest, Path]:
    """Validate four complete stages, write final report + schema-v1 manifest.

    Requires an already-held sibling lock for this build. Leaves the root named
    ``.building``; does not acquire/release the lock or publish.
    """
    if not lock.held:
        raise SnapshotLockError(
            "finalize_candidate_snapshot requires the sibling lock to already be held"
        )
    expected_lock = sibling_lock_path(run.snapshots_root, run.build_id)
    if Path(lock.path).resolve() != Path(expected_lock).resolve():
        raise SnapshotLockError(
            f"sibling lock {lock.path} does not belong to build {run.build_id}"
        )
    building = Path(run.roots.building)
    if not str(building).endswith(".building"):
        raise RunConfigError(f"candidate root must remain .building: {building}")
    config = run.run_config
    state = inspect_stage_markers(building, config)
    if state.next_stage is not None or set(state.completed_stages) != set(STAGE_ORDER):
        raise RunConfigError(
            "finalization requires all four stages complete; next="
            f"{state.next_stage!r} completed={state.completed_stages!r}"
        )
    markers = state.validated_markers
    surface_report = json.loads(
        _require_rel_file(building, _SURFACE_REPORT, label="surface report").read_text(
            encoding="utf-8"
        )
    )
    if not (surface_report.get("checks") or []):
        raise RunConfigError(
            "surface report checks list is empty; cannot certify candidate"
        )
    universe, universe_digest = _cross_check_identity(building, config, markers)
    ready_start, ready_end, ready_count = _select_feature_ready(
        building, markers, universe, config
    )
    if ready_count == 0 or ready_start is None or ready_end is None:
        raise RunConfigError(
            "no feature-ready interval: no contiguous Surface-supported "
            "entry dates satisfy strict-prior PIT membership and input coverage"
        )

    for rel in ("reports/final", "manifests"):
        path = building / rel
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    weekday_gaps = list(
        load_raw_inventory_document(building).get("missing_weekday_dates_diagnostic")
        or []
    )
    accepted_warnings = [
        w
        for stage in ("spot", "surface")
        for w in markers[stage].get("accepted_warnings") or []
    ]
    overall = "WARN" if accepted_warnings else "PASS"
    ready_start_s = None if ready_start is None else ready_start.isoformat()
    ready_end_s = None if ready_end is None else ready_end.isoformat()
    gap_notes = [f"missing weekday (diagnostic): {d}" for d in weekday_gaps]
    final_report = {
        "status": overall,
        "run_config_id": config["run_config_id"],
        "stages": [
            {
                "stage": stage,
                "stage_contract_version": markers[stage]["stage_contract_version"],
            }
            for stage in STAGE_ORDER
        ],
        "universe_count": len(universe),
        "universe_digest": universe_digest,
        "physical_date_count": len(config["physical_raw_dates"]),
        "resolved_date_count": len(config["resolved_trading_dates"]),
        "feature_ready_start": ready_start_s,
        "feature_ready_end": ready_end_s,
        "feature_ready_entry_count": ready_count,
        "accepted_warnings": accepted_warnings,
        "weekday_gap_notes": gap_notes,
    }
    (building / _FINAL_REPORT_REL).write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    surf = markers["surface"]["evidence"]
    artifacts = {
        ARTIFACT_LIQUIDITY_DAILY: _LIQUIDITY_ARTIFACTS[0],
        ARTIFACT_LIQUIDITY_WEEKLY: _LIQUIDITY_ARTIFACTS[1],
        ARTIFACT_LIQUIDITY_PANEL: _LIQUIDITY_ARTIFACTS[2],
        ARTIFACT_LIQUID_TICKERS: _LIQUIDITY_ARTIFACTS[3],
        ARTIFACT_SPLITS: _ADJUSTED_SPLITS,
        ARTIFACT_ADJUSTED_CHAINS_ROOT: "input/adjusted_liquid",
        ARTIFACT_SPOT_PRICES: _SPOT_OUTPUT,
        ARTIFACT_OPTION_SURFACE_META: surf["meta_path"],
        ARTIFACT_OPTION_SURFACE_QUOTES: surf["quotes_path"],
    }
    reports = {
        "liquidity": _LIQUIDITY_REPORT,
        "adjusted": _ADJUSTED_REPORT,
        "spot": _SPOT_REPORT,
        "surface": _SURFACE_REPORT,
        "final": _FINAL_REPORT_REL,
    }
    for label, rel in {**artifacts, **reports}.items():
        _manifest_rel_ok(
            building,
            rel,
            label=label,
            directory=label == ARTIFACT_ADJUSTED_CHAINS_ROOT,
        )

    planned_final = str(Path(run.roots.final).resolve())
    params = {
        "scope": "full",
        "run_config_id": config["run_config_id"],
        "inventory_digest": config["inventory_digest"],
        "equity_universe_digest": universe_digest,
        "classification_digest": markers["liquidity"]["evidence"]["classification_digest"],
        "adjusted_inventory_digest": markers["adjusted"]["evidence"][
            "adjusted_inventory_digest"
        ],
        "spot_output_key_digest": markers["spot"]["evidence"]["output_key_digest"],
        "surface_expected_a1_key_digest": surf["expected_a1_key_digest"],
        "surface_actual_a1_key_digest": surf["actual_a1_key_digest"],
        "c4_params": dict(config["c4_params"]),
        "surface_policy": dict(config["surface_policy"]),
        "feature_ready_start": ready_start_s,
        "feature_ready_end": ready_end_s,
    }
    snapshot_id = compute_snapshot_id(
        {
            "schema_version": INPUT_SNAPSHOT_SCHEMA_VERSION,
            "as_of_resolved_trading_day": config["as_of_resolved_trading_day"],
            "data_source": _ORATS_RAW_REBUILD,
            "artifacts": artifacts,
            "params": params,
        }
    )
    manifest = InputSnapshotManifest(
        schema_version=INPUT_SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        build_id=run.build_id,
        created_at_utc=datetime.now(timezone.utc),
        as_of_requested=date.fromisoformat(config["as_of_requested"]),
        as_of_resolved_trading_day=date.fromisoformat(
            config["as_of_resolved_trading_day"]
        ),
        data_source=_ORATS_RAW_REBUILD,
        cache_dir=planned_final,
        artifacts=artifacts,
        params=params,
        reports=reports,
        overall_status=overall,
        blocking_failures=[],
        notes=gap_notes,
        production_accepted=True,
    )
    manifest_path = building / "manifests" / f"input_snapshot_{snapshot_id}.json"
    write_manifest(manifest_path, manifest)
    loaded = read_manifest(manifest_path)
    if (
        compute_snapshot_id(loaded) != loaded.snapshot_id
        or loaded.build_id != run.build_id
        or Path(loaded.cache_dir).resolve() != Path(planned_final).resolve()
        or loaded.data_source != _ORATS_RAW_REBUILD
        or loaded.params.get("scope") != "full"
        or loaded.production_accepted is not True
    ):
        raise RunConfigError("manifest read-back identity/publication checks failed")
    for label, rel in {
        **loaded.artifacts,
        **{k: v for k, v in loaded.reports.items() if v},
    }.items():
        _manifest_rel_ok(
            building,
            rel,
            label=f"readback {label}",
            directory=label == ARTIFACT_ADJUSTED_CHAINS_ROOT,
        )
    for stage in STAGE_ORDER:
        load_and_validate_stage_marker(building, stage, config)
    return loaded, manifest_path


def publish_candidate_snapshot(
    run: PreparedRun | ResumedRun,
    lock: SiblingBuildLock,
) -> Path:
    """Publish a finalized ``.building`` candidate by removing ``work/`` and renaming.

    Requires the correct sibling lock to already be held. Does not re-validate
    the manifest, release the lock, or inspect the snapshot after rename.
    """
    if not lock.held:
        raise SnapshotLockError(
            "publish_candidate_snapshot requires the sibling lock to already be held"
        )
    expected_lock = sibling_lock_path(run.snapshots_root, run.build_id)
    if Path(lock.path).resolve() != Path(expected_lock).resolve():
        raise SnapshotLockError(
            f"sibling lock {lock.path} does not belong to build {run.build_id}"
        )
    building = Path(run.roots.building)
    final = Path(run.roots.final)
    if not building.is_dir():
        raise SnapshotLifecycleError(f"missing .building root for publication: {building}")
    if final.exists():
        raise SnapshotLifecycleError(
            f"refusing to overwrite existing final root: {final}"
        )
    validate_same_volume_publication(building, final)
    work = building / "work"
    if work.exists():
        shutil.rmtree(work)
    os.replace(building, final)
    return final
