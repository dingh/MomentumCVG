"""Durable in-progress status for long snapshot backfills.

Writes ``run_progress.json`` under the ``.building`` root so an operator (or
agent) can poll stage/phase percent without relying on redirected stdout.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "run_progress.json"


def run_progress_path(building_root: Path | str) -> Path:
    """Return ``<building>/run_progress.json``."""
    return Path(building_root) / PROGRESS_FILENAME


def write_run_progress(
    building_root: Path | str,
    *,
    stage: str,
    phase: str,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
    **extra: Any,
) -> Path:
    """Atomically write a small JSON progress snapshot.

    ``pct`` is derived from ``current/total`` when both are present and
    ``total > 0``. Extra keys (e.g. ``build_id``, ``stage_index``) are stored
    as provided.
    """
    building = Path(building_root)
    building.mkdir(parents=True, exist_ok=True)
    path = run_progress_path(building)

    pct: float | None = None
    if current is not None and total is not None and total > 0:
        pct = round(100.0 * float(current) / float(total), 1)

    payload: dict[str, Any] = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "phase": phase,
        "current": current,
        "total": total,
        "pct": pct,
        "message": message,
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value

    temp_path = building / f"{PROGRESS_FILENAME}.tmp-{uuid.uuid4().hex}"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path
