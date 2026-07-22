"""Unit tests for durable backfill run_progress.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.run_progress import PROGRESS_FILENAME, run_progress_path, write_run_progress


def test_write_run_progress_derives_pct_and_updates_atomically(tmp_path: Path):
    building = tmp_path / "build.building"
    path = write_run_progress(
        building,
        stage="liquidity",
        phase="core_classification",
        current=50,
        total=100,
        message="halfway",
        build_id="abc",
    )
    assert path == building / PROGRESS_FILENAME
    assert path == run_progress_path(building)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "liquidity"
    assert payload["phase"] == "core_classification"
    assert payload["current"] == 50
    assert payload["total"] == 100
    assert payload["pct"] == 50.0
    assert payload["message"] == "halfway"
    assert payload["build_id"] == "abc"
    assert "updated_at_utc" in payload
    assert not list(building.glob(f"{PROGRESS_FILENAME}.tmp-*"))


def test_write_run_progress_omits_pct_when_total_unknown(tmp_path: Path):
    path = write_run_progress(
        tmp_path / "b.building",
        stage="adjusted",
        phase="starting",
        message="go",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pct"] is None
    assert payload["current"] is None
    assert payload["total"] is None
