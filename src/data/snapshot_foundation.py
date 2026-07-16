"""Reusable snapshot foundation primitives (Sprint 004 C8.3A).

This module holds the smallest set of durable, reviewable primitives that
C8.3B (snapshot orchestration) and Sprint 005 (single-manifest consumption)
need, without any orchestration, copying, or publication logic:

* canonical JSON serialization and SHA-256 digests;
* a collision-resistant snapshot ``build_id`` generator;
* snapshot-root lifecycle helpers (``.building`` / final / ``.failed``) with
  fresh-root and same-volume publication preconditions;
* a root-relative artifact/report path validator;
* the accepted C8.2 adjusted-inventory resolution semantics;
* adjusted-inventory and upstream-bundle identity digests;
* a small JSON-compatible gate-result type plus a handful of pure gates.

The module deliberately avoids frameworks, subprocess execution, and any
producer wiring. Those belong to C8.3B.

Production-shaped, not production-complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

import pandas as pd

# ── Errors ────────────────────────────────────────────────────────────────────


class SnapshotFoundationError(RuntimeError):
    """Base class for snapshot foundation failures."""


class SnapshotLifecycleError(SnapshotFoundationError):
    """Raised for staging/final/failed root lifecycle violations."""


class SnapshotPathError(SnapshotFoundationError):
    """Raised when an artifact/report path is unsafe or escapes the root."""


class AdjustedInventoryError(SnapshotFoundationError):
    """Raised for adjusted-inventory resolution failures."""


# ── 1.1 Canonical serialization and digests ───────────────────────────────────


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for ``value``.

    Canonical form uses sorted keys and compact separators. Non-JSON-
    serializable inputs fail with a clear ``ValueError`` rather than being
    silently stringified. Callers that need stable path handling across
    Windows and POSIX should pass already-normalized (forward-slashed)
    string values; this function does not rewrite path separators.
    """
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError as exc:
        raise ValueError(
            f"value is not JSON-serializable for canonical encoding: {exc}"
        ) from exc
    return text.encode("utf-8")


def sha256_file(path: Path | str) -> str:
    """Return the full lowercase SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    """Return the full lowercase SHA-256 hex digest of canonical JSON.

    Full hex is returned unless a caller explicitly slices a shortened
    logical id (see :func:`upstream_bundle_id`).
    """
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# ── 1.2 Collision-resistant build_id ──────────────────────────────────────────

_BUILD_ID_UUID_RE = re.compile(r"[0-9a-f]{8,}")
SNAPSHOT_BUILD_ID_RE = re.compile(r"^\d{8}T\d{12}Z_[0-9a-f]{8}$")


def generate_snapshot_build_id(
    *,
    now: datetime | None = None,
    uuid_source: Callable[[], uuid.UUID | str] | None = None,
) -> str:
    """Return a collision-resistant snapshot build id.

    Format: ``YYYYMMDDTHHMMSSffffffZ_<8 lowercase UUID hex>`` (UTC).

    ``now`` and ``uuid_source`` are injectable for tests. The id is
    independent of any command text. Two invocations sharing the same
    timestamp still differ when their UUIDs differ. A malformed injected
    UUID output is rejected rather than producing an invalid id.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now.astimezone(timezone.utc)
    timestamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")

    if uuid_source is None:
        uuid_source = uuid.uuid4
    raw = uuid_source()
    if isinstance(raw, uuid.UUID):
        hex_full = raw.hex
    elif isinstance(raw, str):
        hex_full = raw
    else:
        raise SnapshotFoundationError(
            f"uuid_source must return uuid.UUID or hex str; got {raw!r}"
        )
    hex_full = hex_full.lower()
    if not _BUILD_ID_UUID_RE.fullmatch(hex_full):
        raise SnapshotFoundationError(
            f"uuid_source produced malformed uuid hex: {raw!r}"
        )
    return f"{timestamp}_{hex_full[:8]}"


# ── 1.3 Snapshot roots and lifecycle safety ───────────────────────────────────


@dataclass(frozen=True)
class SnapshotRoots:
    """The three lifecycle roots derived from one ``build_id``."""

    build_id: str
    building: Path
    final: Path
    failed: Path


def derive_snapshot_roots(snapshots_root: Path | str, build_id: str) -> SnapshotRoots:
    """Derive ``.building`` / final / ``.failed`` roots for a build id."""
    base = Path(snapshots_root)
    return SnapshotRoots(
        build_id=build_id,
        building=base / f"{build_id}.building",
        final=base / build_id,
        failed=base / f"{build_id}.failed",
    )


def create_fresh_staging_root(
    snapshots_root: Path | str, build_id: str
) -> SnapshotRoots:
    """Create only the fresh staging root, refusing any reuse.

    Refuses if any of the staging, final, or failed roots already exists.
    Never deletes or cleans an existing root, and never opens an existing
    final root for writing.
    """
    roots = derive_snapshot_roots(snapshots_root, build_id)
    for existing in (roots.building, roots.final, roots.failed):
        if existing.exists():
            raise SnapshotLifecycleError(
                f"refusing to reuse existing snapshot path: {existing}"
            )
    roots.building.mkdir(parents=True, exist_ok=False)
    return roots


def validate_same_volume_publication(
    building: Path | str, final: Path | str
) -> None:
    """Validate that staging and final roots share the same local volume.

    Publication is a same-volume directory rename. This only validates
    eligibility; it does not perform the rename. On POSIX (no drive
    letters) both drives are empty and the check trivially passes.
    """
    building_drive = os.path.splitdrive(os.path.abspath(str(building)))[0]
    final_drive = os.path.splitdrive(os.path.abspath(str(final)))[0]
    if building_drive.lower() != final_drive.lower():
        raise SnapshotLifecycleError(
            "staging and final roots must be on the same volume for "
            f"publication: {building} vs {final}"
        )


# ── 1.4 Root-relative path safety ─────────────────────────────────────────────


def resolve_under_root(
    root: Path | str, rel_path: str, *, label: str = "path"
) -> Path:
    """Validate ``rel_path`` and resolve it under ``root``.

    Rejects non-string, empty, absolute (Windows or POSIX), drive-qualified,
    and ``..``-escaping values, normalizes separators, and requires the
    resolved path to remain under ``root``. Returns the resolved path.
    """
    if not isinstance(rel_path, str):
        raise SnapshotPathError(f"{label} must be a string; got {rel_path!r}")
    if rel_path == "":
        raise SnapshotPathError(f"{label} must not be empty")

    win = PureWindowsPath(rel_path)
    if win.drive:
        raise SnapshotPathError(
            f"{label} must not be drive-qualified: {rel_path!r}"
        )
    if win.anchor or PurePosixPath(rel_path).is_absolute():
        raise SnapshotPathError(
            f"{label} must be relative, not absolute: {rel_path!r}"
        )

    normalized = rel_path.replace("\\", "/")
    if ".." in PurePosixPath(normalized).parts:
        raise SnapshotPathError(
            f"{label} must not contain '..' segments: {rel_path!r}"
        )

    root_path = Path(root)
    candidate = (root_path / normalized).resolve(strict=False)
    root_resolved = root_path.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise SnapshotPathError(
            f"{label} escapes snapshot root {root_resolved}: {rel_path!r}"
        ) from exc
    return candidate


# ── 1.5 Adjusted inventory resolver ───────────────────────────────────────────

_ADJUSTED_FILENAME_RE = re.compile(r"^ORATS_SMV_Strikes_(\d{8})\.parquet$")


@dataclass(frozen=True)
class AdjustedInventory:
    """Resolved adjusted daily inventory (C8.2 semantics).

    ``physical_dates`` are all well-formed, year-checked adjusted parquet
    filename dates (including verified-empty weekend files).
    ``resolved_trading_dates`` are the physical dates minus verified-empty
    weekend files. ``date_min``/``date_max`` bound the physical inventory.
    The split-history parquet is never part of this daily inventory.
    """

    physical_dates: tuple[date, ...]
    resolved_trading_dates: tuple[date, ...]
    weekend_excluded_dates: tuple[date, ...]
    physical_paths_by_date: dict[date, Path]
    resolved_paths_by_date: dict[date, Path]
    date_min: date
    date_max: date


def resolve_adjusted_inventory(
    data_root: Path | str,
    start_year: int,
    end_year: int,
) -> AdjustedInventory:
    """Resolve the adjusted daily inventory, preserving accepted C8.2 rules.

    Fails closed on a missing data root, a missing requested year directory,
    a malformed filename, a file date outside its containing year directory
    (checked before weekend exclusion), a non-empty weekend file, a duplicate
    date, or an empty physical inventory. Verified-empty weekend files are
    included in the physical inventory but excluded from the resolved trading
    inventory.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise AdjustedInventoryError(f"data root does not exist: {data_root}")

    physical: list[tuple[date, Path]] = []
    resolved: list[tuple[date, Path]] = []
    weekend_excluded: list[date] = []

    for year in range(start_year, end_year + 1):
        year_dir = data_root / str(year)
        if not year_dir.is_dir():
            raise AdjustedInventoryError(
                f"requested year directory missing: {year_dir}"
            )
        for file_path in sorted(year_dir.glob("ORATS_SMV_Strikes_*.parquet")):
            match = _ADJUSTED_FILENAME_RE.match(file_path.name)
            if match is None:
                raise AdjustedInventoryError(
                    f"malformed adjusted filename: {file_path}"
                )
            try:
                trade_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError as exc:
                raise AdjustedInventoryError(
                    f"invalid date in adjusted filename: {file_path}"
                ) from exc
            # Year membership is checked before weekend exclusion so an empty
            # weekend file in the wrong year directory still fails.
            if trade_date.year != year:
                raise AdjustedInventoryError(
                    f"adjusted file date {trade_date.isoformat()} does not belong "
                    f"to its containing year directory {year}"
                )
            physical.append((trade_date, file_path))
            if trade_date.weekday() >= 5:  # Saturday/Sunday: not a trading day
                try:
                    weekend_frame = pd.read_parquet(file_path)
                except Exception as exc:
                    raise AdjustedInventoryError(
                        f"failed to read weekend-dated file {file_path}: {exc}"
                    ) from exc
                if not weekend_frame.empty:
                    raise AdjustedInventoryError(
                        f"weekend-dated file contains data: {file_path}"
                    )
                weekend_excluded.append(trade_date)
                continue
            resolved.append((trade_date, file_path))

    if not physical:
        raise AdjustedInventoryError(
            f"no adjusted dates discovered under {data_root} "
            f"for years {start_year}-{end_year}"
        )

    seen: set[date] = set()
    for trade_date, _ in physical:
        if trade_date in seen:
            raise AdjustedInventoryError(
                f"duplicate adjusted date discovered: {trade_date.isoformat()}"
            )
        seen.add(trade_date)

    physical.sort(key=lambda item: item[0])
    resolved.sort(key=lambda item: item[0])
    physical_dates = tuple(d for d, _ in physical)
    resolved_dates = tuple(d for d, _ in resolved)

    return AdjustedInventory(
        physical_dates=physical_dates,
        resolved_trading_dates=resolved_dates,
        weekend_excluded_dates=tuple(sorted(weekend_excluded)),
        physical_paths_by_date={d: p for d, p in physical},
        resolved_paths_by_date={d: p for d, p in resolved},
        date_min=physical_dates[0],
        date_max=physical_dates[-1],
    )


# ── 1.6 Inventory and bundle identity ─────────────────────────────────────────


def adjusted_inventory_digest(
    adjusted_root: Path | str,
    physical_paths: Iterable[Path | str],
) -> str:
    """Digest the adjusted daily inventory by root-relative path and size.

    Returns ``sha256`` of canonical JSON of the sorted list of
    ``[root-relative forward-slashed path, file_size_bytes]`` entries. The
    same relative-path representation must be used for source and copy. Only
    physical daily parquets are passed in; the split-history file is excluded.
    Large daily parquets are never full-byte hashed.
    """
    root = Path(adjusted_root)
    entries: list[list[Any]] = []
    for raw_path in physical_paths:
        path = Path(raw_path)
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SnapshotPathError(
                f"adjusted inventory path {path} is not under root {root}"
            ) from exc
        entries.append([rel, path.stat().st_size])
    entries.sort()
    return digest_json(entries)


def upstream_bundle_id(
    *,
    c4_evidence_id: str,
    c5_evidence_id: str,
    liquid_tickers_sha256: str,
    splits_sha256: str,
    adjusted_inventory_digest: str,
) -> str:
    """Return the 16-hex upstream bundle identity for the accepted C4/C5 bundle."""
    payload = {
        "c4_evidence_id": c4_evidence_id,
        "c5_evidence_id": c5_evidence_id,
        "liquid_tickers_sha256": liquid_tickers_sha256,
        "splits_sha256": splits_sha256,
        "adjusted_inventory_digest": adjusted_inventory_digest,
    }
    return digest_json(payload)[:16]


def ticker_date_keys_digest(keys: Iterable[tuple[Any, Any]]) -> str:
    """Digest a set of ticker-date keys using a documented canonical form.

    Each key is ``(date_or_iso, ticker)``. The canonical representation is a
    de-duplicated, sorted list of ``[iso_date, ticker]`` pairs. Order and
    duplicates in the input do not affect the digest.
    """
    normalized: set[tuple[str, str]] = set()
    for raw_date, ticker in keys:
        if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
            iso = raw_date.isoformat()
        elif isinstance(raw_date, datetime):
            iso = raw_date.date().isoformat()
        else:
            iso = str(raw_date)
        normalized.add((iso, str(ticker)))
    return digest_json([[iso, ticker] for iso, ticker in sorted(normalized)])


# ── 1.7 Gate result type and pure gates ───────────────────────────────────────

GATE_PASS = "PASS"
GATE_WARN = "WARN"
GATE_FAIL = "FAIL"
_GATE_STATUSES = (GATE_PASS, GATE_WARN, GATE_FAIL)


@dataclass
class GateResult:
    """A small JSON-compatible gate outcome."""

    name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in _GATE_STATUSES:
            raise ValueError(
                f"invalid gate status {self.status!r}; expected one of {_GATE_STATUSES}"
            )

    @property
    def passed(self) -> bool:
        return self.status == GATE_PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": dict(self.metrics),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def _status_for(failures: list[str], warnings: list[str] | None = None) -> str:
    if failures:
        return GATE_FAIL
    if warnings:
        return GATE_WARN
    return GATE_PASS


def gate_source_copy_identity(
    source: Mapping[str, Any],
    copied: Mapping[str, Any],
    *,
    name: str = "source_copy_identity",
) -> GateResult:
    """Require every source component to equal its copied counterpart."""
    failures: list[str] = []
    all_keys = sorted(set(source) | set(copied))
    for key in all_keys:
        if key not in source:
            failures.append(f"component {key!r} present in copy but missing from source")
        elif key not in copied:
            failures.append(f"component {key!r} present in source but missing from copy")
        elif source[key] != copied[key]:
            failures.append(
                f"component {key!r} mismatch: source={source[key]!r} copy={copied[key]!r}"
            )
    return GateResult(
        name=name,
        status=_status_for(failures),
        metrics={"component_count": len(all_keys)},
        failures=failures,
    )


def gate_required_paths_exist(
    root: Path | str,
    rel_paths: Mapping[str, str] | Iterable[str],
    *,
    name: str = "required_paths_exist",
) -> GateResult:
    """Require every listed root-relative path to resolve and exist."""
    if isinstance(rel_paths, Mapping):
        items = list(rel_paths.items())
    else:
        items = [(rel, rel) for rel in rel_paths]

    failures: list[str] = []
    present = 0
    for label, rel in items:
        try:
            resolved = resolve_under_root(root, rel, label=str(label))
        except SnapshotPathError as exc:
            failures.append(str(exc))
            continue
        if resolved.exists():
            present += 1
        else:
            failures.append(f"required path missing: {label} -> {rel} ({resolved})")
    return GateResult(
        name=name,
        status=_status_for(failures),
        metrics={"required_count": len(items), "present_count": present},
        failures=failures,
    )


def gate_manifest_path_containment(
    root: Path | str,
    rel_paths: Mapping[str, str] | Iterable[str],
    *,
    name: str = "manifest_path_containment",
) -> GateResult:
    """Require every artifact/report path to resolve safely under the root."""
    if isinstance(rel_paths, Mapping):
        items = list(rel_paths.items())
    else:
        items = [(rel, rel) for rel in rel_paths]

    failures: list[str] = []
    for label, rel in items:
        try:
            resolve_under_root(root, rel, label=str(label))
        except SnapshotPathError as exc:
            failures.append(str(exc))
    return GateResult(
        name=name,
        status=_status_for(failures),
        metrics={"checked_count": len(items)},
        failures=failures,
    )


_SPOT_SUMMARY_REQUIRED_FIELDS = (
    "source_ticker_date_key_count",
    "source_ticker_date_key_digest",
    "output_ticker_date_key_count",
    "output_ticker_date_key_digest",
    "ambiguous_exclusion_count",
    "ambiguous_exclusions",
    "output_row_count",
)


def gate_spot_summary_reconciliation(
    summary: Mapping[str, Any],
    *,
    name: str = "spot_summary_reconciliation",
) -> GateResult:
    """Reconcile the compact spot summary's counts and digests.

    Requires: output keys == source keys minus ambiguous exclusions; the
    ambiguous count equals the listed exclusions; the output row count equals
    the output key count; and both key digests are non-empty strings.
    """
    missing = [f for f in _SPOT_SUMMARY_REQUIRED_FIELDS if f not in summary]
    if missing:
        return GateResult(
            name=name,
            status=GATE_FAIL,
            metrics={},
            failures=[f"spot summary missing field {f!r}" for f in missing],
        )

    failures: list[str] = []
    source_count = summary["source_ticker_date_key_count"]
    output_count = summary["output_ticker_date_key_count"]
    ambiguous_count = summary["ambiguous_exclusion_count"]
    output_rows = summary["output_row_count"]

    if source_count - ambiguous_count != output_count:
        failures.append(
            "reconciliation mismatch: source_keys "
            f"({source_count}) - ambiguous ({ambiguous_count}) "
            f"!= output_keys ({output_count})"
        )

    listed = len(summary["ambiguous_exclusions"])
    if listed != ambiguous_count:
        failures.append(
            f"ambiguous_exclusion_count ({ambiguous_count}) != listed "
            f"ambiguous_exclusions ({listed})"
        )

    if output_rows != output_count:
        failures.append(
            f"output_row_count ({output_rows}) != output_ticker_date_key_count "
            f"({output_count})"
        )

    for digest_field in (
        "source_ticker_date_key_digest",
        "output_ticker_date_key_digest",
    ):
        value = summary[digest_field]
        if not isinstance(value, str) or not value:
            failures.append(f"{digest_field} must be a non-empty string")

    return GateResult(
        name=name,
        status=_status_for(failures),
        metrics={
            "source_ticker_date_key_count": source_count,
            "output_ticker_date_key_count": output_count,
            "ambiguous_exclusion_count": ambiguous_count,
        },
        failures=failures,
    )
