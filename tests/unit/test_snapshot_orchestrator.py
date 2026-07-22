"""Unit tests for the C8.3B snapshot orchestration foundation.

Deterministic tests only: temporary directories and tiny synthetic ZIP
fixtures. No real ORATS data, no network, no producer execution.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.data.security_types import classification_digest
from src.data.snapshot_foundation import (
    SiblingBuildLock,
    SnapshotLockError,
    adjusted_inventory_digest,
    digest_json,
    sha256_file,
    ticker_date_keys_digest,
)
from src.data.snapshot_orchestrator import (
    DEFAULT_C4_PARAMS,
    DEFAULT_LOOKBACK_WEEKS,
    RAW_DEPENDENCY_PAD_WEEKS,
    RAW_INVENTORY_FILENAME,
    RUN_CONFIG_FILENAME,
    SCOPE_BOUNDED,
    SCOPE_FULL,
    STAGE_CONTRACT_VERSION,
    STAGE_ORDER,
    RawInventoryError,
    RunConfigError,
    SNAPSHOT_OWNED_DIRS,
    build_stage_marker,
    compute_raw_dependency_start,
    compute_run_config_id,
    effective_run_scope,
    execute_backfill_stages,
    finalize_candidate_snapshot,
    inspect_stage_markers,
    load_and_validate_stage_marker,
    load_raw_inventory_document,
    load_run_config,
    open_resume_run,
    prepare_new_backfill_run,
    scan_raw_inventory,
    stage_marker_path,
    validate_run_config,
    write_stage_marker,
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
    assert config["scope"] == SCOPE_FULL
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


def test_bounded_evidence_freezes_scope_bounded(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=FRI,
        as_of_requested=SUN,
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_A,
        scope=SCOPE_BOUNDED,
    )
    assert prepared.run_config["scope"] == SCOPE_BOUNDED
    assert effective_run_scope(prepared.run_config) == SCOPE_BOUNDED


def test_scope_changes_run_config_id_and_manifest_snapshot_id(snapshots_root, raw_root):
    from src.data.input_snapshot import compute_snapshot_id

    _seed_week(raw_root)
    full = _prepare(snapshots_root, raw_root, build_id=BUILD_ID_A)
    bounded = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=FRI,
        as_of_requested=SUN,
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_B,
        scope=SCOPE_BOUNDED,
    )
    assert full.run_config["run_config_id"] != bounded.run_config["run_config_id"]

    _install_finalizable(full)
    _install_finalizable(bounded)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        full_manifest, _ = finalize_candidate_snapshot(full, lock)
    with SiblingBuildLock(snapshots_root, BUILD_ID_B) as lock:
        bounded_manifest, _ = finalize_candidate_snapshot(bounded, lock)
    assert full_manifest.snapshot_id != bounded_manifest.snapshot_id
    assert full_manifest.params["scope"] == SCOPE_FULL
    assert bounded_manifest.params["scope"] == SCOPE_BOUNDED
    assert compute_snapshot_id(full_manifest) == full_manifest.snapshot_id
    assert compute_snapshot_id(bounded_manifest) == bounded_manifest.snapshot_id


def test_legacy_config_without_scope_validates_as_full(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    config = dict(prepared.run_config)
    del config["scope"]
    config["run_config_id"] = compute_run_config_id(config)
    validate_run_config(config)
    assert effective_run_scope(config) == SCOPE_FULL


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


def test_resume_rejects_building_run_copied_to_another_snapshot_root(
    tmp_path, raw_root
):
    snapshots_a = tmp_path / "snapshots-a"
    snapshots_b = tmp_path / "snapshots-b"
    snapshots_a.mkdir()
    snapshots_b.mkdir()
    _seed_week(raw_root)
    prepared = _prepare(snapshots_a, raw_root)

    copied_building = snapshots_b / prepared.roots.building.name
    shutil.copytree(prepared.roots.building, copied_building)

    with pytest.raises(
        RunConfigError,
        match=r"relocating or copying a \.building run.*not permitted",
    ):
        open_resume_run(snapshots_b, BUILD_ID_A)

    resumed = open_resume_run(snapshots_a, BUILD_ID_A)
    assert resumed.roots.building == prepared.roots.building


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


# ── completion markers and stage-boundary resume ───────────────────────────────


def _write_liquidity_artifacts(
    building: Path,
    tickers: list[str] = ("AAA", "BBB"),
    *,
    panel_month: date = date(2023, 12, 29),
    panel_eligible: bool = True,
):
    liquid_dir = building / "input" / "liquidity"
    liquid_dir.mkdir(parents=True, exist_ok=True)
    reports = building / "reports" / "liquidity"
    reports.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    for name in (
        "ticker_liquidity_daily_observations.parquet",
        "ticker_liquidity_weekly_observations.parquet",
        "ticker_liquidity_panel.parquet",
    ):
        if name == "ticker_liquidity_panel.parquet":
            n = len(tickers)
            pd.DataFrame(
                {
                    "ticker": list(tickers),
                    "month_date": [panel_month] * n,
                    "atm_straddle_dollar_vol": (
                        [100.0 - i for i in range(n)] if panel_eligible else [None] * n
                    ),
                    "atm_spread_pct": (
                        [0.01 + 0.001 * i for i in range(n)]
                        if panel_eligible
                        else [None] * n
                    ),
                    "has_valid_atm_pair": [panel_eligible] * n,
                }
            ).to_parquet(liquid_dir / name, index=False)
        else:
            pd.DataFrame({"ticker": list(tickers)}).to_parquet(
                liquid_dir / name, index=False
            )
    pd.DataFrame(
        {
            "Ticker": list(tickers),
            "snapshots_qualified": [3] * len(tickers),
            "months_qualified": [3] * len(tickers),
        }
    ).to_csv(liquid_dir / "liquid_tickers.csv", index=False)
    classification = pd.DataFrame(
        {
            "ticker": list(tickers),
            "classification": ["company_equity"] * len(tickers),
            "observed_asset_types": ["[0]"] * len(tickers),
        }
    )
    classification.to_parquet(liquid_dir / "security_classification.parquet", index=False)
    (reports / "pit_universe_audit.md").write_text(
        "# C7 PIT Universe Audit\n\n## Verdict\n\n"
        "- overall status: `PASS`\n"
        "- strict mode: `True`\n",
        encoding="utf-8",
    )
    return {
        "stage": "liquidity",
        "status": "PASS",
        "output_dir": str(liquid_dir),
        "report_path": str(reports / "pit_universe_audit.md"),
        "artifacts": sorted(p.name for p in liquid_dir.iterdir()),
        "files_read": 2,
        "daily_row_count": 2,
        "weekly_row_count": 2,
        "panel_row_count": 2,
        "classified_ticker_count": len(tickers),
        "classification_digest": classification_digest(classification),
        "liquid_ticker_count": len(tickers),
        "equity_universe_digest": digest_json(sorted(tickers)),
        "accepted_warnings": [],
    }


def _write_adjusted_artifacts(building: Path, days: list[date]):
    import pandas as pd

    liquid = building / "input" / "liquidity" / "liquid_tickers.csv"
    if not liquid.is_file():
        _write_liquidity_artifacts(building)
    adj = building / "input" / "adjusted_liquid"
    adj.mkdir(parents=True, exist_ok=True)
    reports = building / "reports" / "adjusted"
    reports.mkdir(parents=True, exist_ok=True)
    splits = adj / "splits_hist_liquid.parquet"
    pd.DataFrame(
        {"ticker": ["AAA"], "split_date": [date(2024, 2, 1)], "divisor": [2.0]}
    ).to_parquet(splits, index=False)
    parquets = []
    for day in days:
        path = adj / str(day.year) / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": ["AAA", "BBB"], "stkPx": [1.0, 2.0]}).to_parquet(
            path, index=False
        )
        parquets.append(path)
    (reports / "adjusted_liquid_audit.md").write_text(
        "## Overall verdict: **PASS**\n", encoding="utf-8"
    )
    file_count = sum(1 for p in adj.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in adj.rglob("*") if p.is_file())
    return {
        "stage": "adjusted",
        "status": "PASS",
        "output_dir": str(adj),
        "report_path": str(reports / "adjusted_liquid_audit.md"),
        "artifacts": ["splits_hist_liquid.parquet", "2024"],
        "expected_zip_count": len(days),
        "produced_file_count": len(days),
        "date_min": days[0].isoformat(),
        "date_max": days[-1].isoformat(),
        "date_count": len(days),
        "output_file_count": file_count,
        "output_total_bytes": total_bytes,
        "split_metadata_hash": sha256_file(splits),
        "universe_ticker_count": 2,
        "universe_digest": digest_json(["AAA", "BBB"]),
        "adjusted_inventory_digest": adjusted_inventory_digest(adj, parquets),
        "audit_verdict": "PASS",
    }


def _write_spot_artifacts(building: Path, days: list[date] | date = FRI):
    import pandas as pd

    if isinstance(days, date):
        days = [days]
    spot_dir = building / "cache" / "spot"
    spot_dir.mkdir(parents=True, exist_ok=True)
    reports = building / "reports" / "spot"
    reports.mkdir(parents=True, exist_ok=True)
    keys = [(day, t) for day in days for t in ("AAA", "BBB")]
    frame = pd.DataFrame(
        {
            "date": [d for d, _ in keys],
            "ticker": [t for _, t in keys],
            "adj_spot_price": [10.0] * len(keys),
            "spot_price": [10.0] * len(keys),
        }
    )
    out = spot_dir / "spot_prices_adjusted.parquet"
    frame.to_parquet(out, index=False)
    summary = {
        "resolved_date_count": len(days),
        "resolved_date_min": days[0].isoformat(),
        "resolved_date_max": days[-1].isoformat(),
        "weekend_excluded_dates": [],
        "source_ticker_date_key_count": len(keys),
        "source_ticker_date_key_digest": ticker_date_keys_digest(keys),
        "output_ticker_date_key_count": len(keys),
        "output_ticker_date_key_digest": ticker_date_keys_digest(keys),
        "ambiguous_exclusion_count": 0,
        "ambiguous_exclusions": [],
        "output_row_count": len(keys),
        "producer_status": "PASS",
        "warnings": [],
    }
    (spot_dir / "spot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (reports / "gate_spot_reconciliation.json").write_text(
        json.dumps(
            {
                "name": "spot_summary_reconciliation",
                "status": "PASS",
                "failures": [],
                "warnings": [],
                "metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "stage": "spot",
        "status": "PASS",
        "output_dir": str(spot_dir),
        "output_path": str(out),
        "summary_path": str(spot_dir / "spot_summary.json"),
        "report_path": str(reports / "gate_spot_reconciliation.json"),
        "artifacts": ["spot_prices_adjusted.parquet", "spot_summary.json"],
        "source_key_count": len(keys),
        "source_key_digest": summary["source_ticker_date_key_digest"],
        "output_key_count": len(keys),
        "output_key_digest": summary["output_ticker_date_key_digest"],
        "ambiguous_exclusion_count": 0,
        "output_row_count": len(keys),
        "output_total_bytes": out.stat().st_size,
        "accepted_warnings": [],
    }


def _write_surface_artifacts(
    building: Path,
    entries: list[date] | date = FRI,
    *,
    expiry_by_entry: dict[date, date] | None = None,
    empty_checks: bool = False,
):
    import pandas as pd

    if isinstance(entries, date):
        entries = [entries]
    if expiry_by_entry is None:
        expiry_by_entry = {e: e for e in entries}
    surface = building / "cache" / "surface"
    surface.mkdir(parents=True, exist_ok=True)
    reports = building / "reports" / "surface"
    reports.mkdir(parents=True, exist_ok=True)
    tickers = ["AAA", "BBB"]
    meta_rows = []
    quote_rows = []
    for entry in entries:
        expiry = expiry_by_entry[entry]
        for t in tickers:
            meta_rows.append(
                {
                    "ticker": t,
                    "entry_date": entry,
                    "expiry_date": expiry,
                    "surface_valid": True,
                }
            )
            for side in ("call", "put"):
                quote_rows.append(
                    {
                        "ticker": t,
                        "entry_date": entry,
                        "expiry_date": expiry,
                        "strike": 10.0,
                        "side": side,
                    }
                )
    meta = pd.DataFrame(meta_rows)
    quotes = pd.DataFrame(quote_rows)
    meta_path = surface / "option_surface_meta_weekly_2024_2024.parquet"
    quotes_path = surface / "option_surface_quotes_weekly_2024_2024.parquet"
    meta.to_parquet(meta_path, index=False)
    quotes.to_parquet(quotes_path, index=False)
    checks = [] if empty_checks else [
        {
            "name": "expected_meta_keys",
            "status": "PASS",
            "metrics": {},
            "failures": [],
            "warnings": [],
        }
    ]
    (reports / "surface_contract_checks.json").write_text(
        json.dumps({"overall_verdict": "PASS", "checks": checks}) + "\n",
        encoding="utf-8",
    )
    keys = [(e, t) for e in entries for t in tickers]
    digest = ticker_date_keys_digest(keys)
    a2_grain = sorted(
        [
            str(t),
            pd.Timestamp(e).date().isoformat(),
            pd.Timestamp(x).date().isoformat(),
            float(s),
            str(side),
        ]
        for t, e, x, s, side in zip(
            quotes["ticker"],
            quotes["entry_date"],
            quotes["expiry_date"],
            quotes["strike"],
            quotes["side"],
        )
    )
    return {
        "stage": "surface",
        "status": "PASS",
        "output_dir": str(surface),
        "meta_path": str(meta_path),
        "quotes_path": str(quotes_path),
        "report_path": str(reports / "surface_contract_checks.json"),
        "artifacts": [meta_path.name, quotes_path.name],
        "supported_entry_dates": [e.isoformat() for e in entries],
        "expected_a1_key_count": len(keys),
        "expected_a1_key_digest": digest,
        "actual_a1_key_count": len(keys),
        "actual_a1_key_digest": digest,
        "surface_valid_true_count": len(keys),
        "surface_valid_false_count": 0,
        "a2_row_count": len(quotes),
        "a2_grain_digest": digest_json(a2_grain),
        "meta_total_bytes": meta_path.stat().st_size,
        "quotes_total_bytes": quotes_path.stat().st_size,
        "accepted_warnings": [],
    }


def _install_marker(prepared, stage: str, evidence: dict, *, repo_sha: str = "a" * 40):
    marker = build_stage_marker(
        building_root=prepared.roots.building,
        stage=stage,
        run_config=prepared.run_config,
        evidence=evidence,
        producer_repo_sha=repo_sha,
        completed_at_utc="2024-01-05T12:00:00.000000Z",
    )
    write_stage_marker(prepared.roots.building, marker)
    return marker


def test_inspect_no_markers_selects_liquidity(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    state = inspect_stage_markers(prepared.roots.building, prepared.run_config)
    assert state.completed_stages == ()
    assert state.next_stage == "liquidity"
    assert state.validated_markers == {}


def test_inspect_valid_liquidity_selects_adjusted(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    evidence = _write_liquidity_artifacts(prepared.roots.building)
    _install_marker(prepared, "liquidity", evidence)

    state = inspect_stage_markers(prepared.roots.building, prepared.run_config)
    assert state.completed_stages == ("liquidity",)
    assert state.next_stage == "adjusted"
    assert "liquidity" in state.validated_markers


def test_inspect_liquidity_adjusted_prefix_starts_spot(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(prepared.roots.building))
    _install_marker(
        prepared,
        "adjusted",
        _write_adjusted_artifacts(prepared.roots.building, [MON, TUE, WED, THU, FRI]),
    )

    state = inspect_stage_markers(prepared.roots.building, prepared.run_config)
    assert state.completed_stages == ("liquidity", "adjusted")
    assert state.next_stage == "spot"


def test_inspect_four_valid_markers_skips_all(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(building))
    _install_marker(
        prepared, "adjusted", _write_adjusted_artifacts(building, [MON, TUE, WED, THU, FRI])
    )
    _install_marker(prepared, "spot", _write_spot_artifacts(building))
    _install_marker(prepared, "surface", _write_surface_artifacts(building))

    state = inspect_stage_markers(building, prepared.run_config)
    assert state.completed_stages == STAGE_ORDER
    assert state.next_stage is None


def test_malformed_marker_raises_run_config_error(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    path = stage_marker_path(prepared.roots.building, "liquidity")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RunConfigError, match="malformed stage marker"):
        inspect_stage_markers(prepared.roots.building, prepared.run_config)


def test_evidence_mismatch_raises_run_config_error(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    evidence = _write_liquidity_artifacts(prepared.roots.building)
    evidence["equity_universe_digest"] = "0" * 64
    with pytest.raises(RunConfigError, match="equity_universe_digest mismatch"):
        build_stage_marker(
            building_root=prepared.roots.building,
            stage="liquidity",
            run_config=prepared.run_config,
            evidence=evidence,
            producer_repo_sha="a" * 40,
        )


def test_wrong_contract_version_or_run_config_id_raises(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    evidence = _write_liquidity_artifacts(prepared.roots.building)
    marker = build_stage_marker(
        building_root=prepared.roots.building,
        stage="liquidity",
        run_config=prepared.run_config,
        evidence=evidence,
        producer_repo_sha="a" * 40,
    )
    marker["stage_contract_version"] = "999"
    write_stage_marker(prepared.roots.building, marker)
    with pytest.raises(RunConfigError, match="incompatible stage_contract_version"):
        load_and_validate_stage_marker(
            prepared.roots.building, "liquidity", prepared.run_config
        )

    path = stage_marker_path(prepared.roots.building, "liquidity")
    path.unlink()
    marker["stage_contract_version"] = STAGE_CONTRACT_VERSION
    marker["run_config_id"] = "0" * 64
    write_stage_marker(prepared.roots.building, marker)
    with pytest.raises(RunConfigError, match="run_config_id mismatch"):
        load_and_validate_stage_marker(
            prepared.roots.building, "liquidity", prepared.run_config
        )


def test_later_marker_without_predecessor_is_corruption(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    # Write an adjusted marker without liquidity.
    _write_liquidity_artifacts(prepared.roots.building)
    evidence = _write_adjusted_artifacts(
        prepared.roots.building, [MON, TUE, WED, THU, FRI]
    )
    _install_marker(prepared, "adjusted", evidence)
    with pytest.raises(RunConfigError, match="after a missing predecessor"):
        inspect_stage_markers(prepared.roots.building, prepared.run_config)


def test_output_without_marker_is_cleaned_and_stage_reruns(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    stale = building / "input" / "liquidity" / "stale.txt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale", encoding="utf-8")

    calls: list[str] = []

    def runner(stage: str):
        calls.append(stage)
        if stage == "liquidity":
            return _write_liquidity_artifacts(building)
        if stage == "adjusted":
            return _write_adjusted_artifacts(building, [MON, TUE, WED, THU, FRI])
        if stage == "spot":
            return _write_spot_artifacts(building)
        return _write_surface_artifacts(building)

    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        execute_backfill_stages(
            prepared, lock, producer_repo_sha="b" * 40, stage_runner=runner
        )

    assert calls[0] == "liquidity"
    assert not stale.exists()
    assert stage_marker_path(building, "liquidity").is_file()
    progress = json.loads((building / "run_progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "surface"
    assert progress["phase"] == "complete"
    assert progress["pct"] == 100.0


def test_accepted_predecessor_output_not_modified(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    liq_evidence = _write_liquidity_artifacts(building)
    _install_marker(prepared, "liquidity", liq_evidence)
    panel = building / "input" / "liquidity" / "ticker_liquidity_panel.parquet"
    before = panel.read_bytes()

    def runner(stage: str):
        assert stage != "liquidity"
        if stage == "adjusted":
            return _write_adjusted_artifacts(building, [MON, TUE, WED, THU, FRI])
        if stage == "spot":
            return _write_spot_artifacts(building)
        return _write_surface_artifacts(building)

    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        execute_backfill_stages(
            prepared, lock, producer_repo_sha="b" * 40, stage_runner=runner
        )

    assert panel.read_bytes() == before


def test_failed_stage_writes_no_marker(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)

    def runner(stage: str):
        raise RuntimeError("producer failed")

    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RuntimeError, match="producer failed"):
            execute_backfill_stages(
                prepared, lock, producer_repo_sha="b" * 40, stage_runner=runner
            )

    assert not stage_marker_path(prepared.roots.building, "liquidity").exists()


def test_keyboard_interrupt_writes_no_marker_preserves_prefix(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(building))

    def runner(stage: str):
        raise KeyboardInterrupt

    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(KeyboardInterrupt):
            execute_backfill_stages(
                prepared, lock, producer_repo_sha="b" * 40, stage_runner=runner
            )

    assert stage_marker_path(building, "liquidity").is_file()
    assert not stage_marker_path(building, "adjusted").exists()


def test_changed_repo_sha_does_not_invalidate_marker(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    evidence = _write_liquidity_artifacts(prepared.roots.building)
    _install_marker(prepared, "liquidity", evidence, repo_sha="oldsha00000000000000000000000000000001")
    # Validation must succeed even though producer_repo_sha != current HEAD.
    marker = load_and_validate_stage_marker(
        prepared.roots.building, "liquidity", prepared.run_config
    )
    assert marker["producer_repo_sha"].startswith("oldsha")


def test_marker_write_is_atomic_and_refuses_overwrite(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    evidence = _write_liquidity_artifacts(prepared.roots.building)
    marker = build_stage_marker(
        building_root=prepared.roots.building,
        stage="liquidity",
        run_config=prepared.run_config,
        evidence=evidence,
        producer_repo_sha="a" * 40,
    )
    path = write_stage_marker(prepared.roots.building, marker)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp-*"))
    with pytest.raises(RunConfigError, match="refusing to overwrite"):
        write_stage_marker(prepared.roots.building, marker)


def test_execute_requires_held_lock(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    lock = SiblingBuildLock(snapshots_root, BUILD_ID_A)
    with pytest.raises(SnapshotLockError, match="already be held"):
        execute_backfill_stages(prepared, lock, stage_runner=lambda stage: {})


def _rewrite_marker(building: Path, stage: str, mutate) -> None:
    path = stage_marker_path(building, stage)
    marker = json.loads(path.read_text(encoding="utf-8"))
    mutate(marker)
    path.unlink()
    write_stage_marker(building, marker)


def test_report_changed_to_fail_rejects_marker(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(building))
    (
        building / "reports" / "liquidity" / "pit_universe_audit.md"
    ).write_text(
        "# C7 PIT Universe Audit\n\n## Verdict\n\n"
        "- overall status: `FAIL`\n"
        "- strict mode: `True`\n",
        encoding="utf-8",
    )
    with pytest.raises(RunConfigError, match="overall PASS"):
        load_and_validate_stage_marker(building, "liquidity", prepared.run_config)


def test_marker_gate_status_disagreeing_with_evidence_rejects(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(building))

    def mutate(marker):
        marker["gate_status"] = "WARN"

    _rewrite_marker(building, "liquidity", mutate)
    with pytest.raises(RunConfigError, match="gate_status"):
        load_and_validate_stage_marker(building, "liquidity", prepared.run_config)


def test_marker_warnings_disagreeing_with_evidence_rejects(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "spot", _write_spot_artifacts(building))

    def mutate(marker):
        marker["accepted_warnings"] = [
            {
                "warning": "1 ambiguous ticker-date exclusion(s) present",
                "reason": "reconciled_ambiguous_ticker_date_exclusion",
            }
        ]

    _rewrite_marker(building, "spot", mutate)
    with pytest.raises(RunConfigError, match="accepted_warnings"):
        load_and_validate_stage_marker(building, "spot", prepared.run_config)


def test_liquidity_or_adjusted_evidence_with_warning_rejects(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    liq = _write_liquidity_artifacts(building)
    liq["accepted_warnings"] = ["unexpected warning"]
    with pytest.raises(RunConfigError, match="must not contain accepted_warnings"):
        build_stage_marker(
            building_root=building,
            stage="liquidity",
            run_config=prepared.run_config,
            evidence=liq,
            producer_repo_sha="a" * 40,
        )

    adj = _write_adjusted_artifacts(building, [MON, TUE, WED, THU, FRI])
    adj["accepted_warnings"] = ["unexpected warning"]
    with pytest.raises(RunConfigError, match="must not contain accepted_warnings"):
        build_stage_marker(
            building_root=building,
            stage="adjusted",
            run_config=prepared.run_config,
            evidence=adj,
            producer_repo_sha="a" * 40,
        )


# ── final cross-stage validation and candidate manifest ────────────────────────


def _seed_trading_weeks(raw_root: Path, fridays: list[date]) -> None:
    for fri in fridays:
        monday = fri - timedelta(days=4)
        for offset in range(5):
            _write_zip(raw_root, monday + timedelta(days=offset), rows=2)
        _write_zip(raw_root, fri + timedelta(days=1), rows=0)


def _install_finalizable(
    prepared,
    *,
    surface_entries: list[date] | None = None,
    expiry_by_entry: dict[date, date] | None = None,
    empty_checks: bool = False,
    adjusted_days: list[date] | None = None,
    spot_days: list[date] | None = None,
    panel_month: date = date(2023, 12, 29),
    panel_eligible: bool = True,
):
    building = prepared.roots.building
    physical = [date.fromisoformat(d) for d in prepared.run_config["physical_raw_dates"]]
    resolved = [date.fromisoformat(d) for d in prepared.run_config["resolved_trading_dates"]]
    if surface_entries is None:
        surface_entries = [resolved[-1]]
    _install_marker(
        prepared,
        "liquidity",
        _write_liquidity_artifacts(
            building, panel_month=panel_month, panel_eligible=panel_eligible
        ),
    )
    _install_marker(
        prepared,
        "adjusted",
        _write_adjusted_artifacts(building, adjusted_days or physical),
    )
    _install_marker(
        prepared,
        "spot",
        _write_spot_artifacts(building, spot_days or resolved),
    )
    _install_marker(
        prepared,
        "surface",
        _write_surface_artifacts(
            building,
            surface_entries,
            expiry_by_entry=expiry_by_entry,
            empty_checks=empty_checks,
        ),
    )


def _fingerprints(building: Path) -> dict[str, str]:
    out = {}
    for path in sorted(building.rglob("*")):
        if path.is_file():
            rel = path.relative_to(building).as_posix()
            out[rel] = sha256_file(path)
    return out


def test_finalize_four_markers_writes_valid_manifest(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        manifest, path = finalize_candidate_snapshot(prepared, lock)
    assert path.is_file()
    assert path.parent.name == "manifests"
    assert prepared.roots.building.name.endswith(".building")
    assert not prepared.roots.final.exists()
    assert manifest.production_accepted is True
    assert manifest.data_source == "orats_raw_rebuild"
    assert Path(manifest.cache_dir).resolve() == prepared.roots.final.resolve()
    assert manifest.params["scope"] == SCOPE_FULL
    assert (prepared.roots.building / "reports/final/final_validation.json").is_file()


def test_bounded_finalize_sets_production_accepted_false(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=FRI,
        as_of_requested=SUN,
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_A,
        scope=SCOPE_BOUNDED,
    )
    _install_finalizable(prepared)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        manifest, _ = finalize_candidate_snapshot(prepared, lock)
    assert manifest.params["scope"] == SCOPE_BOUNDED
    assert manifest.production_accepted is False
    assert manifest.overall_status == "PASS"
    assert manifest.data_source == "orats_raw_rebuild"


def test_bounded_pass_can_publish_without_production_accepted(
    snapshots_root, raw_root
):
    from src.data.snapshot_orchestrator import publish_candidate_snapshot

    _seed_week(raw_root)
    prepared = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=FRI,
        as_of_requested=SUN,
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_A,
        scope=SCOPE_BOUNDED,
    )
    _install_finalizable(prepared)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        manifest, _ = finalize_candidate_snapshot(prepared, lock)
        final = publish_candidate_snapshot(prepared, lock)
        assert lock.held
    assert final.is_dir()
    assert not prepared.roots.building.exists()
    assert manifest.production_accepted is False
    assert manifest.overall_status == "PASS"


def test_finalize_missing_marker_writes_no_manifest(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    building = prepared.roots.building
    _install_marker(prepared, "liquidity", _write_liquidity_artifacts(building))
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RunConfigError, match="all four stages"):
            finalize_candidate_snapshot(prepared, lock)
    assert list((building / "manifests").glob("input_snapshot_*.json")) == []


def test_finalize_universe_or_date_disagreement_blocks(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared, adjusted_days=[MON, TUE, WED, THU, FRI])
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RunConfigError, match="physical_raw_dates"):
            finalize_candidate_snapshot(prepared, lock)

    prepared2 = _prepare(snapshots_root, raw_root, build_id=BUILD_ID_B)
    _install_finalizable(prepared2)
    liquid = prepared2.roots.building / "input/liquidity/liquid_tickers.csv"
    liquid.write_text(
        "Ticker,snapshots_qualified,months_qualified\nAAA,3,3\nCCC,3,3\n",
        encoding="utf-8",
    )
    with SiblingBuildLock(snapshots_root, BUILD_ID_B) as lock:
        with pytest.raises(RunConfigError, match="universe identity|equity_universe"):
            finalize_candidate_snapshot(prepared2, lock)


def test_finalize_readiness_gap_selects_largest_earliest(snapshots_root, raw_root):
    fri_a, fri_b, fri_c = FRI, date(2024, 1, 12), date(2024, 1, 19)
    _seed_trading_weeks(raw_root, [fri_a, fri_b, fri_c])
    prepared = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=fri_a,
        as_of_requested=date(2024, 1, 21),
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_A,
    )
    outside = date(2024, 2, 1)
    _install_finalizable(
        prepared,
        surface_entries=[fri_a, fri_b, fri_c],
        expiry_by_entry={fri_a: outside, fri_b: fri_b, fri_c: fri_c},
    )
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        manifest, _ = finalize_candidate_snapshot(prepared, lock)
    assert manifest.params["feature_ready_start"] == fri_b.isoformat()
    assert manifest.params["feature_ready_end"] == fri_c.isoformat()

    prepared2 = prepare_new_backfill_run(
        snapshots_root=snapshots_root,
        raw_root=raw_root,
        requested_output_start=fri_a,
        as_of_requested=date(2024, 1, 21),
        lookback_weeks=1,
        repo_sha_at_freeze="f" * 40,
        build_id=BUILD_ID_B,
    )
    _install_finalizable(
        prepared2,
        surface_entries=[fri_a, fri_b, fri_c],
        expiry_by_entry={fri_a: fri_a, fri_b: outside, fri_c: fri_c},
    )
    with SiblingBuildLock(snapshots_root, BUILD_ID_B) as lock:
        manifest2, _ = finalize_candidate_snapshot(prepared2, lock)
    assert manifest2.params["feature_ready_start"] == fri_a.isoformat()
    assert manifest2.params["feature_ready_end"] == fri_a.isoformat()


def test_finalize_manifest_paths_resolve_in_building(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        manifest, _ = finalize_candidate_snapshot(prepared, lock)
    building = prepared.roots.building
    assert Path(manifest.cache_dir).resolve() == prepared.roots.final.resolve()
    assert not str(Path(manifest.cache_dir).resolve()).endswith(".building")
    for rel in manifest.artifacts.values():
        assert (building / rel).exists()
        assert "work" not in Path(rel).parts
        assert "candidate" not in Path(rel).parts
    for rel in manifest.reports.values():
        assert rel is not None
        assert (building / rel).is_file()


def test_finalize_rejects_empty_surface_checks(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared, empty_checks=True)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RunConfigError, match="checks list is empty"):
            finalize_candidate_snapshot(prepared, lock)


def test_finalize_rerun_only_touches_final_and_manifests(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared)
    building = prepared.roots.building
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        finalize_candidate_snapshot(prepared, lock)
        before = _fingerprints(building)
        (building / "reports/final/sentinel.txt").write_text("x", encoding="utf-8")
        finalize_candidate_snapshot(prepared, lock)
        after = _fingerprints(building)
    assert "reports/final/sentinel.txt" not in after
    stable_before = {
        rel: digest
        for rel, digest in before.items()
        if not rel.startswith("reports/final/") and not rel.startswith("manifests/")
    }
    stable_after = {
        rel: digest
        for rel, digest in after.items()
        if not rel.startswith("reports/final/") and not rel.startswith("manifests/")
    }
    assert stable_before == stable_after
    assert any(rel.startswith("manifests/") for rel in after)
    assert any(rel.startswith("reports/final/") for rel in after)


def _assert_no_final_outputs(building: Path) -> None:
    assert list((building / "manifests").glob("input_snapshot_*.json")) == []
    assert not (building / "reports/final/final_validation.json").is_file()


def test_finalize_rejects_ineligible_prior_pit_snapshot(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    _install_finalizable(prepared, panel_eligible=False)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RunConfigError, match="no feature-ready interval"):
            finalize_candidate_snapshot(prepared, lock)
    _assert_no_final_outputs(prepared.roots.building)


def test_finalize_rejects_missing_strict_prior_pit_snapshot(snapshots_root, raw_root):
    _seed_week(raw_root)
    prepared = _prepare(snapshots_root, raw_root)
    # Same-day panel date is not a strict prior for the supported entry.
    _install_finalizable(prepared, panel_month=FRI)
    with SiblingBuildLock(snapshots_root, BUILD_ID_A) as lock:
        with pytest.raises(RunConfigError, match="no feature-ready interval"):
            finalize_candidate_snapshot(prepared, lock)
    _assert_no_final_outputs(prepared.roots.building)


# ── atomic publication ─────────────────────────────────────────────────────────


def test_publish_removes_work_renames_and_keeps_sibling_lock(tmp_path):
    from src.data.snapshot_foundation import derive_snapshot_roots
    from src.data.snapshot_orchestrator import publish_candidate_snapshot
    from types import SimpleNamespace

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    roots = derive_snapshot_roots(snapshots, BUILD_ID_A)
    roots.building.mkdir()
    (roots.building / "work" / "liquidity" / "candidate").mkdir(parents=True)
    (roots.building / "markers").mkdir()
    (roots.building / "input").mkdir()
    (roots.building / "work" / "liquidity" / "candidate" / "tmp.txt").write_text(
        "x", encoding="utf-8"
    )
    run = SimpleNamespace(
        build_id=BUILD_ID_A, snapshots_root=snapshots, roots=roots
    )
    with SiblingBuildLock(snapshots, BUILD_ID_A) as lock:
        final = publish_candidate_snapshot(run, lock)
        assert lock.held
        assert final == roots.final
        assert final.is_dir()
        assert not roots.building.exists()
        assert not (final / "work").exists()
        assert (final / "markers").is_dir()
        lock_path = snapshots / f"{BUILD_ID_A}.lock"
        assert lock_path.is_file()
        assert lock_path.parent == snapshots
        assert lock_path.parent == final.parent


def test_publish_refuses_existing_final_root(tmp_path):
    from src.data.snapshot_foundation import (
        SnapshotLifecycleError,
        derive_snapshot_roots,
    )
    from src.data.snapshot_orchestrator import publish_candidate_snapshot
    from types import SimpleNamespace

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    roots = derive_snapshot_roots(snapshots, BUILD_ID_A)
    roots.building.mkdir()
    (roots.building / "work").mkdir()
    roots.final.mkdir()
    run = SimpleNamespace(
        build_id=BUILD_ID_A, snapshots_root=snapshots, roots=roots
    )
    with SiblingBuildLock(snapshots, BUILD_ID_A) as lock:
        with pytest.raises(SnapshotLifecycleError, match="overwrite"):
            publish_candidate_snapshot(run, lock)
        assert roots.building.is_dir()
        assert (roots.building / "work").is_dir()
        assert lock.held


def test_publish_rename_failure_leaves_building(tmp_path, monkeypatch):
    from src.data.snapshot_foundation import derive_snapshot_roots
    from src.data.snapshot_orchestrator import publish_candidate_snapshot
    from types import SimpleNamespace
    import src.data.snapshot_orchestrator as orch

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    roots = derive_snapshot_roots(snapshots, BUILD_ID_A)
    roots.building.mkdir()
    (roots.building / "work").mkdir()
    run = SimpleNamespace(
        build_id=BUILD_ID_A, snapshots_root=snapshots, roots=roots
    )

    def boom(*_a, **_k):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(orch.os, "replace", boom)
    with SiblingBuildLock(snapshots, BUILD_ID_A) as lock:
        with pytest.raises(OSError, match="simulated rename failure"):
            publish_candidate_snapshot(run, lock)
        assert roots.building.is_dir()
        assert not roots.final.exists()
        # work/ may already be removed before rename; building itself remains.
        assert lock.held
