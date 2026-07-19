"""Unit tests for the C8.3B snapshot orchestration foundation.

Deterministic tests only: temporary directories and tiny synthetic ZIP
fixtures. No real ORATS data, no network, no producer execution.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.data.snapshot_orchestrator import (
    DEFAULT_C4_PARAMS,
    DEFAULT_LOOKBACK_WEEKS,
    RAW_DEPENDENCY_PAD_WEEKS,
    RAW_INVENTORY_FILENAME,
    RUN_CONFIG_FILENAME,
    RawInventoryError,
    RunConfigError,
    SNAPSHOT_OWNED_DIRS,
    compute_raw_dependency_start,
    compute_run_config_id,
    load_raw_inventory_document,
    load_run_config,
    open_resume_run,
    prepare_new_backfill_run,
    scan_raw_inventory,
    validate_run_config,
)

ROOT = Path(__file__).resolve().parents[2]

BUILD_ID_A = "20260717T220000123456Z_aaaaaaaa"
BUILD_ID_B = "20260717T220000123456Z_bbbbbbbb"

# Deterministic week: Mon 2024-01-01 .. Fri 2024-01-05, Sat 06, Sun 07.
MON, TUE, WED, THU, FRI = (date(2024, 1, d) for d in range(1, 6))
SAT, SUN = date(2024, 1, 6), date(2024, 1, 7)


# ── fixtures ───────────────────────────────────────────────────────────────────


def _zip_path(root: Path, day: date) -> Path:
    return root / f"{day.year:04d}" / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.zip"


def _write_zip(root: Path, day: date, *, rows: int = 1, salt: str = "") -> Path:
    """Write one synthetic raw archive; ``rows=0`` gives a header-only (empty) CSV."""
    path = _zip_path(root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ticker,stkPx"] + [f"AAPL,{100 + i}{salt}" for i in range(rows)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.csv",
            "\n".join(lines) + "\n",
        )
    return path


def _seed_week(root: Path, *, empty_weekend: bool = True) -> None:
    for day in (MON, TUE, WED, THU, FRI):
        _write_zip(root, day, rows=2)
    if empty_weekend:
        _write_zip(root, SAT, rows=0)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    return root


@pytest.fixture
def snapshots_root(tmp_path: Path) -> Path:
    root = tmp_path / "snapshots"
    root.mkdir()
    return root


def _prepare(
    snapshots_root: Path,
    raw_root: Path,
    *,
    build_id: str = BUILD_ID_A,
    repo_sha: str = "f" * 40,
    lookback_weeks: int = 1,
):
    """Prepare a small run whose dependency window covers the seeded week."""
    return prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=FRI,
        as_of_requested=SUN,
        lookback_weeks=lookback_weeks,
        repo_sha_at_freeze=repo_sha,
        build_id=build_id,
    )


# ── dependency-start formula ───────────────────────────────────────────────────


def test_dependency_start_uses_lookback_plus_two():
    start = date(2024, 6, 7)
    assert RAW_DEPENDENCY_PAD_WEEKS == 2
    assert compute_raw_dependency_start(start, 12) == start - timedelta(weeks=14)
    assert compute_raw_dependency_start(start, 1) == start - timedelta(weeks=3)
    # Default lookback produces the documented 14-week dependency pad.
    assert compute_raw_dependency_start(start) == start - timedelta(
        weeks=DEFAULT_LOOKBACK_WEEKS + 2
    )


def test_dependency_start_matches_c4_constant_and_formula():
    """The orchestrator mirrors C4 — no competing lookback default or pad."""
    spec = importlib.util.spec_from_file_location(
        "build_liquidity_panel_for_orch_test",
        ROOT / "scripts" / "build_liquidity_panel.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert DEFAULT_LOOKBACK_WEEKS == module.DEFAULT_LOOKBACK_WEEKS
    # C4 run_build discovery pad: start_date - (lookback_weeks + 2) weeks.
    start = date(2024, 6, 7)
    for lookback in (1, 12, 26):
        assert compute_raw_dependency_start(start, lookback) == start - timedelta(
            weeks=lookback + 2
        )
    # Frozen C4 params match the producer defaults for later stages.
    assert DEFAULT_C4_PARAMS["min_valid_quote_weeks"] == module.DEFAULT_MIN_VALID_QUOTE_WEEKS
    assert DEFAULT_C4_PARAMS["dte_min"] == module.DEFAULT_DTE_MIN
    assert DEFAULT_C4_PARAMS["dte_max"] == module.DEFAULT_DTE_MAX
    assert DEFAULT_C4_PARAMS["dvol_top_pct"] == module.DEFAULT_DVOL_TOP_PCT
    assert DEFAULT_C4_PARAMS["spread_bot_pct"] == module.DEFAULT_SPREAD_BOT_PCT


def test_dependency_start_rejects_bad_lookback():
    with pytest.raises(RunConfigError, match="positive"):
        compute_raw_dependency_start(date(2024, 6, 7), 0)


# ── raw inventory scan ─────────────────────────────────────────────────────────


def test_valid_inventory_and_deterministic_digest(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        root.mkdir()
        _seed_week(root)

    inv_a = scan_raw_inventory(root_a, MON, SUN)
    inv_a_again = scan_raw_inventory(root_a, MON, SUN)
    inv_b = scan_raw_inventory(root_b, MON, SUN)

    assert inv_a.inventory_digest == inv_a_again.inventory_digest
    # Identical relative structure under a different absolute root → same
    # digest: no machine-specific absolute paths or timestamps participate.
    assert inv_a.inventory_digest == inv_b.inventory_digest
    assert len(inv_a.inventory_digest) == 64
    assert inv_a.records[0].rel_path == "2024/ORATS_SMV_Strikes_20240101.zip"
    assert all(record.members for record in inv_a.records)


def test_member_content_change_changes_digest(raw_root):
    _seed_week(raw_root)
    before = scan_raw_inventory(raw_root, MON, SUN).inventory_digest
    # Rewrite one archive with different member content (CRC changes).
    _write_zip(raw_root, WED, rows=2, salt="x")
    after = scan_raw_inventory(raw_root, MON, SUN).inventory_digest
    assert before != after


def test_selection_is_bounded_inclusively(raw_root):
    _seed_week(raw_root)
    _write_zip(raw_root, date(2023, 12, 22), rows=1)  # before dependency start
    inv = scan_raw_inventory(raw_root, MON, SUN)
    assert inv.physical_raw_dates[0] == MON
    inv_wide = scan_raw_inventory(raw_root, date(2023, 12, 22), SUN)
    assert inv_wide.physical_raw_dates[0] == date(2023, 12, 22)


def test_malformed_zip_fails(raw_root):
    _seed_week(raw_root)
    bad = _zip_path(raw_root, date(2024, 1, 4))
    bad.write_bytes(b"this is not a zip archive")
    with pytest.raises(RawInventoryError, match="unreadable or malformed ZIP"):
        scan_raw_inventory(raw_root, MON, SUN)


def test_malformed_filename_fails(raw_root):
    _seed_week(raw_root)
    stray = raw_root / "2024" / "ORATS_SMV_Strikes_2024010.zip"
    stray.write_bytes(b"x")
    with pytest.raises(RawInventoryError, match="malformed raw archive filename"):
        scan_raw_inventory(raw_root, MON, SUN)


def test_invalid_calendar_date_in_filename_fails(raw_root):
    _seed_week(raw_root)
    stray = raw_root / "2024" / "ORATS_SMV_Strikes_20240230.zip"
    with zipfile.ZipFile(stray, "w") as archive:
        archive.writestr("x.csv", "ticker\n")
    with pytest.raises(RawInventoryError, match="invalid trade date"):
        scan_raw_inventory(raw_root, MON, SUN)


def test_duplicate_archive_date_fails(raw_root):
    _seed_week(raw_root)
    # Same trade date under a second directory level.
    dup = raw_root / "misc" / "ORATS_SMV_Strikes_20240103.zip"
    dup.parent.mkdir()
    with zipfile.ZipFile(dup, "w") as archive:
        archive.writestr("d.csv", "ticker\nAAPL\n")
    with pytest.raises(RawInventoryError, match="duplicate raw archives"):
        scan_raw_inventory(raw_root, MON, SUN)


def test_empty_weekend_is_physical_but_not_resolved(raw_root):
    _seed_week(raw_root, empty_weekend=True)
    inv = scan_raw_inventory(raw_root, MON, SUN)
    assert SAT in inv.physical_raw_dates
    assert SAT not in inv.resolved_trading_dates
    record = next(r for r in inv.records if r.trade_date == SAT)
    assert record.verified_empty


def test_truly_empty_weekend_zip_is_verified_empty(raw_root):
    _seed_week(raw_root, empty_weekend=False)
    path = _zip_path(raw_root, SUN)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w"):
        pass  # zero members
    inv = scan_raw_inventory(raw_root, MON, SUN)
    assert SUN in inv.physical_raw_dates
    assert SUN not in inv.resolved_trading_dates


def test_nonempty_weekend_archive_fails(raw_root):
    _seed_week(raw_root, empty_weekend=False)
    _write_zip(raw_root, SAT, rows=3)
    with pytest.raises(RawInventoryError, match="nonempty weekend archive"):
        scan_raw_inventory(raw_root, MON, SUN)


def test_weekday_gaps_are_diagnostic_only(raw_root):
    for day in (MON, TUE, THU, FRI):  # Wednesday missing
        _write_zip(raw_root, day, rows=2)
    inv = scan_raw_inventory(raw_root, MON, SUN)  # no exception: not a blocker
    assert inv.missing_weekday_dates == (WED,)
    assert WED not in inv.physical_raw_dates
    assert inv.as_of_resolved_trading_day == FRI


def test_as_of_resolves_from_frozen_raw_inventory(raw_root):
    _seed_week(raw_root, empty_weekend=True)
    # Sunday request resolves past the proven-empty Saturday to Friday.
    inv = scan_raw_inventory(raw_root, MON, SUN)
    assert inv.as_of_resolved_trading_day == FRI
    # Mid-week request resolves to the latest resolved date <= request.
    inv_wed = scan_raw_inventory(raw_root, MON, WED)
    assert inv_wed.as_of_resolved_trading_day == WED


def test_missing_bounds_fail_closed(raw_root):
    with pytest.raises(RawInventoryError, match="no raw archives"):
        scan_raw_inventory(raw_root, MON, SUN)

    with pytest.raises(RawInventoryError, match="does not exist"):
        scan_raw_inventory(raw_root / "nope", MON, SUN)

    _seed_week(raw_root)
    with pytest.raises(RawInventoryError, match="is after as_of_requested"):
        scan_raw_inventory(raw_root, SUN, MON)

    # Only a proven-empty weekend archive in range: no resolvable trading day.
    weekend_only = raw_root.parent / "weekend_only"
    weekend_only.mkdir()
    _write_zip(weekend_only, SAT, rows=0)
    with pytest.raises(RawInventoryError, match="no resolved trading date"):
        scan_raw_inventory(weekend_only, SAT, SUN)


# ── new-run preparation ────────────────────────────────────────────────────────


def test_new_run_writes_immutable_config_inventory_and_layout(
    snapshots_root, raw_root
):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)

    building = snapshots_root / f"{BUILD_ID_A}.building"
    assert prepared.roots.building == building
    assert (building / RUN_CONFIG_FILENAME).is_file()
    assert (building / RAW_INVENTORY_FILENAME).is_file()
    for rel in SNAPSHOT_OWNED_DIRS:
        assert (building / rel).is_dir(), rel
    assert not (snapshots_root / BUILD_ID_A).exists()  # nothing published
    assert not (snapshots_root / f"{BUILD_ID_A}.lock").exists()  # no lock taken

    config = prepared.run_config
    assert config["build_id"] == BUILD_ID_A
    assert config["mode"] == "backfill"
    assert config["requested_output_start"] == FRI.isoformat()
    assert config["raw_dependency_start"] == (FRI - timedelta(weeks=3)).isoformat()
    assert config["as_of_requested"] == SUN.isoformat()
    assert config["as_of_resolved_trading_day"] == FRI.isoformat()
    assert config["inventory_digest"] == prepared.inventory.inventory_digest
    assert config["repo_sha_at_freeze"] == "f" * 40
    assert SAT.isoformat() in config["physical_raw_dates"]
    assert SAT.isoformat() not in config["resolved_trading_dates"]
    assert config["c4_params"]["lookback_weeks"] == 1
    assert config["surface_policy"]["expiry_policy"] == "strict_next_week_from_schedule"

    inventory_doc = load_raw_inventory_document(building)
    assert inventory_doc["inventory_digest"] == config["inventory_digest"]


def test_new_run_refuses_existing_building_or_final_root(snapshots_root, raw_root):
    _seed_week(raw_root)
    _prepare(snapshots_root, raw_root)
    with pytest.raises(Exception, match="refusing to reuse"):
        _prepare(snapshots_root, raw_root)  # .building exists

    (snapshots_root / BUILD_ID_B).mkdir()
    with pytest.raises(Exception, match="refusing to reuse"):
        _prepare(snapshots_root, raw_root, build_id=BUILD_ID_B)  # final exists


def test_new_run_rejects_inverted_bounds(snapshots_root, raw_root):
    _seed_week(raw_root)
    with pytest.raises(RunConfigError, match="is after as_of_requested"):
        prepare_new_backfill_run(
            snapshots_root=snapshots_root,
            raw_root=raw_root,
            requested_output_start=SUN,
            as_of_requested=FRI,
            lookback_weeks=1,
            repo_sha_at_freeze="f" * 40,
            build_id=BUILD_ID_A,
        )
    assert not (snapshots_root / f"{BUILD_ID_A}.building").exists()


# ── run_config_id validation ───────────────────────────────────────────────────


def test_run_config_id_round_trip_validation(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    config = load_run_config(prepared.roots.building)
    validate_run_config(config)  # no raise
    assert config["run_config_id"] == compute_run_config_id(config)


def test_tampered_run_config_fails_validation(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    path = prepared.roots.building / RUN_CONFIG_FILENAME
    config = json.loads(path.read_text(encoding="utf-8"))
    config["repo_sha_at_freeze"] = "0" * 40  # id-covered field silently changed
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with pytest.raises(RunConfigError, match="run_config_id mismatch"):
        load_run_config(prepared.roots.building)


def test_frozen_files_refuse_overwrite(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    from src.data.snapshot_orchestrator import _atomic_write_frozen_json

    with pytest.raises(RunConfigError, match="refusing to overwrite"):
        _atomic_write_frozen_json(
            prepared.roots.building / RUN_CONFIG_FILENAME, {"x": 1}
        )


# ── resume ─────────────────────────────────────────────────────────────────────


def test_resume_validates_and_does_not_rewrite_frozen_files(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    config_path = prepared.roots.building / RUN_CONFIG_FILENAME
    inventory_path = prepared.roots.building / RAW_INVENTORY_FILENAME
    config_bytes = config_path.read_bytes()
    inventory_bytes = inventory_path.read_bytes()

    resumed = open_resume_run(snapshots_root, BUILD_ID_A)
    assert resumed.build_id == BUILD_ID_A
    assert resumed.run_config["run_config_id"] == prepared.run_config["run_config_id"]
    assert resumed.inventory is not None
    assert resumed.inventory.inventory_digest == prepared.inventory.inventory_digest
    # Frozen intent is validated, never rewritten or "repaired".
    assert config_path.read_bytes() == config_bytes
    assert inventory_path.read_bytes() == inventory_bytes


def test_raw_drift_on_resume_fails(snapshots_root, raw_root):
    _seed_week(raw_root)
    _prepare(snapshots_root, raw_root)
    _write_zip(raw_root, WED, rows=5, salt="drift")  # mutate a frozen archive
    with pytest.raises(RunConfigError, match="raw inventory drift"):
        open_resume_run(snapshots_root, BUILD_ID_A)


def test_repo_sha_change_does_not_block_resume(snapshots_root, raw_root):
    _seed_week(raw_root)
    _prepare(snapshots_root, raw_root, repo_sha="a" * 40)
    # The current repository SHA differs from the frozen diagnostic value;
    # resume must still validate and open the run.
    resumed = open_resume_run(snapshots_root, BUILD_ID_A)
    assert resumed.run_config["repo_sha_at_freeze"] == "a" * 40


def test_resume_opens_only_the_named_building_root(snapshots_root, raw_root):
    _seed_week(raw_root)
    _prepare(snapshots_root, raw_root, build_id=BUILD_ID_A)
    _prepare(snapshots_root, raw_root, build_id=BUILD_ID_B)
    resumed = open_resume_run(snapshots_root, BUILD_ID_B)
    assert resumed.roots.building == snapshots_root / f"{BUILD_ID_B}.building"
    assert resumed.run_config["build_id"] == BUILD_ID_B


def test_resume_rejects_final_root_and_missing_building(snapshots_root):
    with pytest.raises(RunConfigError, match="no .building run to resume"):
        open_resume_run(snapshots_root, BUILD_ID_A)

    (snapshots_root / BUILD_ID_A).mkdir()  # only a published final snapshot
    with pytest.raises(RunConfigError, match="never mutates a published snapshot"):
        open_resume_run(snapshots_root, BUILD_ID_A)


def test_resume_rejects_ambiguous_building_plus_final(snapshots_root, raw_root):
    _seed_week(raw_root)
    _prepare(snapshots_root, raw_root)
    (snapshots_root / BUILD_ID_A).mkdir()
    with pytest.raises(RunConfigError, match="corrupt lifecycle"):
        open_resume_run(snapshots_root, BUILD_ID_A)


def test_resume_rejects_malformed_build_id(snapshots_root):
    with pytest.raises(RunConfigError, match="malformed build id"):
        open_resume_run(snapshots_root, "not-a-build-id")


def test_resume_rejects_tampered_inventory(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    path = prepared.roots.building / RAW_INVENTORY_FILENAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["archives"][0]["file_size"] += 1
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    with pytest.raises(RunConfigError, match="digest mismatch"):
        open_resume_run(snapshots_root, BUILD_ID_A)
