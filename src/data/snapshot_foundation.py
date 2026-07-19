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

if os.name == "nt":
    import msvcrt
else:
    import fcntl
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


class SnapshotLockError(SnapshotFoundationError):
    """Raised when the sibling snapshot build lock cannot be acquired/used."""


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


# ── 1.3b Sibling snapshot build lock (C8.3B) ──────────────────────────────────


def sibling_lock_path(snapshots_root: Path | str, build_id: str) -> Path:
    """Path of the sibling build lock: ``<snapshots-root>/<BUILD_ID>.lock``.

    The lock file is a *sibling* of ``<BUILD_ID>.building`` — never inside it —
    so it survives (and never travels with) atomic publication.
    """
    return Path(snapshots_root) / f"{build_id}.lock"


class SiblingBuildLock:
    """OS-level exclusive lock on the sibling ``<BUILD_ID>.lock`` file.

    Semantics:

    * a real OS-level exclusive lock (``msvcrt.locking`` on Windows,
      ``fcntl.flock`` elsewhere) held on an open file handle for the entire
      lock lifetime;
    * acquisition fails immediately and clearly when another holder exists;
    * process exit or crash releases the OS lock automatically;
    * normal release unlocks and closes the handle but never moves or deletes
      the sibling lock file;
    * no stale-lock cleanup, and ownership is never inferred from mere file
      existence — only from holding the OS lock.

    Usable as a context manager or via explicit ``acquire()`` / ``release()``
    so later orchestration can hold it through atomic publication.
    """

    def __init__(self, snapshots_root: Path | str, build_id: str) -> None:
        self.path = sibling_lock_path(snapshots_root, build_id)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "SiblingBuildLock":
        if self._handle is not None:
            raise SnapshotLockError(f"lock already held by this object: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "a+b" creates the file when missing without truncating an existing
        # file that another process may currently hold locked.
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SnapshotLockError(
                f"another process holds the snapshot build lock: {self.path}"
            ) from exc
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "SiblingBuildLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False


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


def _canonical_iso_date(raw_date: Any) -> str:
    """Return the documented canonical ISO string for a date-like value."""
    if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
        return raw_date.isoformat()
    if isinstance(raw_date, datetime):
        return raw_date.date().isoformat()
    return str(raw_date)


def _canonical_ticker_date_key(raw_date: Any, ticker: Any) -> tuple[str, str]:
    """Return the canonical ``(iso_date, ticker)`` representation of one key."""
    return (_canonical_iso_date(raw_date), str(ticker))


def ticker_date_keys_digest(keys: Iterable[tuple[Any, Any]]) -> str:
    """Digest a set of ticker-date keys using a documented canonical form.

    Each key is ``(date_or_iso, ticker)``. The canonical representation is a
    de-duplicated, sorted list of ``[iso_date, ticker]`` pairs. Order and
    duplicates in the input do not affect the digest.
    """
    normalized = {_canonical_ticker_date_key(raw_date, ticker) for raw_date, ticker in keys}
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
    "resolved_date_count",
    "resolved_date_min",
    "resolved_date_max",
    "source_ticker_date_key_count",
    "source_ticker_date_key_digest",
    "output_ticker_date_key_count",
    "output_ticker_date_key_digest",
    "ambiguous_exclusion_count",
    "ambiguous_exclusions",
    "output_row_count",
    "producer_status",
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}")


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_spot_summary_schema(
    summary: Mapping[str, Any],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Validate summary field types before reconciliation.

    Returns ``(failures, parsed_exclusions)``. Reconciliation must not proceed
    while failures are present; the parsed exclusion list is only trustworthy
    when no failures are returned.
    """
    failures: list[str] = []

    for field in (
        "resolved_date_count",
        "source_ticker_date_key_count",
        "output_ticker_date_key_count",
        "ambiguous_exclusion_count",
        "output_row_count",
    ):
        if not _is_nonneg_int(summary[field]):
            failures.append(f"{field} must be a nonnegative integer; got {summary[field]!r}")

    for field in ("source_ticker_date_key_digest", "output_ticker_date_key_digest"):
        if not _is_hex64(summary[field]):
            failures.append(
                f"{field} must be a 64-char lowercase hex string; got {summary[field]!r}"
            )

    if summary["producer_status"] not in (GATE_PASS, GATE_WARN):
        failures.append(
            f"producer_status must be {GATE_PASS!r} or {GATE_WARN!r}; "
            f"got {summary['producer_status']!r}"
        )

    parsed_exclusions: list[tuple[str, str]] = []
    exclusions_raw = summary["ambiguous_exclusions"]
    if not isinstance(exclusions_raw, list):
        failures.append("ambiguous_exclusions must be a list")
    else:
        for entry in exclusions_raw:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                failures.append(
                    f"ambiguous_exclusions entry must be [YYYY-MM-DD, ticker]; got {entry!r}"
                )
                continue
            iso_value, ticker_value = entry[0], entry[1]
            if not _is_iso_date(iso_value):
                failures.append(
                    f"ambiguous_exclusions date must be ISO YYYY-MM-DD; got {iso_value!r}"
                )
                continue
            if (
                not isinstance(ticker_value, str)
                or not ticker_value
                or ticker_value != ticker_value.strip().upper()
            ):
                failures.append(
                    "ambiguous_exclusions ticker must be a normalized nonempty string; "
                    f"got {ticker_value!r}"
                )
                continue
            parsed_exclusions.append((iso_value, ticker_value))

    if _is_nonneg_int(summary["resolved_date_count"]):
        count = summary["resolved_date_count"]
        min_value = summary["resolved_date_min"]
        max_value = summary["resolved_date_max"]
        if count == 0:
            if min_value is not None or max_value is not None:
                failures.append(
                    "resolved_date_min/max must be null when resolved_date_count is zero"
                )
        else:
            for field, value in (
                ("resolved_date_min", min_value),
                ("resolved_date_max", max_value),
            ):
                if not _is_iso_date(value):
                    failures.append(f"{field} must be a valid ISO date; got {value!r}")

    return failures, parsed_exclusions


def gate_spot_summary_reconciliation(
    summary: Mapping[str, Any],
    *,
    source_keys: Iterable[tuple[Any, Any]],
    output_keys: Iterable[tuple[Any, Any]],
    resolved_trading_dates: Iterable[Any],
    name: str = "spot_summary_reconciliation",
) -> GateResult:
    """Reconcile the compact spot summary against independently derived keys.

    The gate is pure: it never reads files. C8.3B derives the three key inputs
    from the copied adjusted source (``source_keys``, before ambiguous
    exclusions), the published spot parquet (``output_keys``), and the resolved
    ``AdjustedInventory`` (``resolved_trading_dates``), then passes them here.

    Keys are normalized to a single canonical ``(iso_date, ticker)`` form. The
    gate verifies every count/digest, that the explicit ambiguous exclusions are
    unique members of the source that are absent from the output, and that the
    output equals ``source_keys - ambiguous_exclusions``.

    FAIL on any schema or reconciliation problem. WARN when reconciliation
    succeeds but ambiguous exclusions or documented producer warnings exist.
    PASS only when reconciliation succeeds with no exclusions or warnings.
    """
    missing = [f for f in _SPOT_SUMMARY_REQUIRED_FIELDS if f not in summary]
    if missing:
        return GateResult(
            name=name,
            status=GATE_FAIL,
            metrics={},
            failures=[f"spot summary missing field {f!r}" for f in missing],
        )

    schema_failures, parsed_exclusions = _validate_spot_summary_schema(summary)
    if schema_failures:
        return GateResult(name=name, status=GATE_FAIL, metrics={}, failures=schema_failures)

    source_norm = {_canonical_ticker_date_key(d, t) for d, t in source_keys}
    output_norm = {_canonical_ticker_date_key(d, t) for d, t in output_keys}
    resolved_norm = sorted({_canonical_iso_date(d) for d in resolved_trading_dates})
    exclusion_set = set(parsed_exclusions)

    failures: list[str] = []

    if summary["resolved_date_count"] != len(resolved_norm):
        failures.append(
            f"resolved_date_count ({summary['resolved_date_count']}) != independently "
            f"derived resolved dates ({len(resolved_norm)})"
        )
    if resolved_norm:
        if summary["resolved_date_min"] != resolved_norm[0]:
            failures.append(
                f"resolved_date_min ({summary['resolved_date_min']!r}) != {resolved_norm[0]!r}"
            )
        if summary["resolved_date_max"] != resolved_norm[-1]:
            failures.append(
                f"resolved_date_max ({summary['resolved_date_max']!r}) != {resolved_norm[-1]!r}"
            )

    if summary["source_ticker_date_key_count"] != len(source_norm):
        failures.append(
            f"source_ticker_date_key_count ({summary['source_ticker_date_key_count']}) "
            f"!= independently derived source keys ({len(source_norm)})"
        )
    if summary["source_ticker_date_key_digest"] != ticker_date_keys_digest(source_norm):
        failures.append("source_ticker_date_key_digest does not match derived source keys")

    if summary["output_ticker_date_key_count"] != len(output_norm):
        failures.append(
            f"output_ticker_date_key_count ({summary['output_ticker_date_key_count']}) "
            f"!= independently derived output keys ({len(output_norm)})"
        )
    if summary["output_ticker_date_key_digest"] != ticker_date_keys_digest(output_norm):
        failures.append("output_ticker_date_key_digest does not match derived output keys")

    if summary["ambiguous_exclusion_count"] != len(parsed_exclusions):
        failures.append(
            f"ambiguous_exclusion_count ({summary['ambiguous_exclusion_count']}) != listed "
            f"ambiguous_exclusions ({len(parsed_exclusions)})"
        )
    if len(exclusion_set) != len(parsed_exclusions):
        failures.append("ambiguous_exclusions contain duplicate keys")

    not_in_source = sorted(k for k in exclusion_set if k not in source_norm)
    if not_in_source:
        failures.append(f"ambiguous exclusions absent from source keys: {not_in_source[:5]}")
    in_output = sorted(k for k in exclusion_set if k in output_norm)
    if in_output:
        failures.append(f"ambiguous exclusions present in output keys: {in_output[:5]}")

    expected_output = source_norm - exclusion_set
    if output_norm != expected_output:
        missing_out = sorted(expected_output - output_norm)
        extra_out = sorted(output_norm - expected_output)
        failures.append(
            "output keys != source_keys - ambiguous_exclusions "
            f"(missing {missing_out[:5]}, extra {extra_out[:5]})"
        )

    if summary["output_row_count"] != len(output_norm):
        failures.append(
            f"output_row_count ({summary['output_row_count']}) != output keys "
            f"({len(output_norm)})"
        )

    output_dates = {iso for iso, _ in output_norm}
    if output_dates != set(resolved_norm):
        only_output = sorted(output_dates - set(resolved_norm))
        only_resolved = sorted(set(resolved_norm) - output_dates)
        failures.append(
            "dates in output keys != resolved trading dates "
            f"(output-only {only_output[:5]}, resolved-only {only_resolved[:5]})"
        )

    producer_warnings = summary.get("warnings", [])
    if not isinstance(producer_warnings, list):
        producer_warnings = []

    warnings_out: list[str] = []
    metrics = {
        "resolved_date_count": len(resolved_norm),
        "source_ticker_date_key_count": len(source_norm),
        "output_ticker_date_key_count": len(output_norm),
        "ambiguous_exclusion_count": len(parsed_exclusions),
    }

    if failures:
        return GateResult(name=name, status=GATE_FAIL, metrics=metrics, failures=failures)

    if parsed_exclusions or producer_warnings or summary["producer_status"] == GATE_WARN:
        if parsed_exclusions:
            warnings_out.append(
                f"{len(parsed_exclusions)} ambiguous ticker-date exclusion(s) present"
            )
        warnings_out.extend(str(w) for w in producer_warnings)
        return GateResult(
            name=name, status=GATE_WARN, metrics=metrics, failures=failures, warnings=warnings_out
        )

    return GateResult(name=name, status=GATE_PASS, metrics=metrics, failures=failures)
