"""Unit tests for the snapshot foundation primitives (Sprint 004 C8.3A).

No real data. All fixtures use ``tmp_path`` with tiny synthetic files.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.snapshot_foundation import (
    AdjustedInventoryError,
    GATE_FAIL,
    GATE_PASS,
    GateResult,
    SnapshotLifecycleError,
    SnapshotPathError,
    SNAPSHOT_BUILD_ID_RE,
    adjusted_inventory_digest,
    canonical_json_bytes,
    create_fresh_staging_root,
    derive_snapshot_roots,
    digest_json,
    gate_source_copy_identity,
    gate_spot_summary_reconciliation,
    generate_snapshot_build_id,
    resolve_adjusted_inventory,
    resolve_under_root,
    sha256_file,
    ticker_date_keys_digest,
    upstream_bundle_id,
    validate_same_volume_publication,
)


# ── adjusted-inventory fixture helpers ─────────────────────────────────────────

_FILENAME = "ORATS_SMV_Strikes_{}.parquet"


def _write_day(root: Path, day: date, frame: pd.DataFrame) -> Path:
    path = root / str(day.year) / _FILENAME.format(day.strftime("%Y%m%d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame([{"ticker": "AAPL", "stkPx": 100.0, "adj_stkPx": 100.0}])


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "stkPx", "adj_stkPx"])


# ── 1.1 canonical serialization and digests ────────────────────────────────────


def test_canonical_digest_is_stable_and_key_order_independent():
    a = {"b": 1, "a": [3, 2, 1], "c": "x"}
    b = {"c": "x", "a": [3, 2, 1], "b": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert digest_json(a) == digest_json(b)
    assert len(digest_json(a)) == 64  # full sha256 hex


def test_non_serializable_canonical_value_fails_clearly():
    with pytest.raises(ValueError, match="not JSON-serializable"):
        canonical_json_bytes({"x": {1, 2, 3}})  # sets are not JSON-serializable


def test_sha256_file_matches_known_content(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(path) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


# ── 1.2 build_id ───────────────────────────────────────────────────────────────


def test_build_id_format():
    now = datetime(2026, 7, 15, 22, 0, 0, 123456, tzinfo=timezone.utc)
    build_id = generate_snapshot_build_id(
        now=now, uuid_source=lambda: uuid.UUID(int=0x1234567890)
    )
    assert SNAPSHOT_BUILD_ID_RE.match(build_id)
    assert build_id.startswith("20260715T220000123456Z_")


def test_build_id_uniqueness_with_same_injected_timestamp():
    now = datetime(2026, 7, 15, 22, 0, 0, 123456, tzinfo=timezone.utc)
    first = generate_snapshot_build_id(
        now=now, uuid_source=lambda: uuid.UUID("11111111-0000-0000-0000-000000000000")
    )
    second = generate_snapshot_build_id(
        now=now, uuid_source=lambda: uuid.UUID("22222222-0000-0000-0000-000000000000")
    )
    assert first != second
    assert first.split("_")[0] == second.split("_")[0]  # same timestamp


def test_build_id_rejects_malformed_injected_uuid():
    now = datetime(2026, 7, 15, 22, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(Exception):
        generate_snapshot_build_id(now=now, uuid_source=lambda: "zzzz")


# ── 1.3 lifecycle roots ────────────────────────────────────────────────────────


def test_fresh_staging_creation(tmp_path):
    roots = create_fresh_staging_root(tmp_path, "20260715T220000123456Z_abcdef01")
    assert roots.building.is_dir()
    assert not roots.final.exists()
    assert not roots.failed.exists()


@pytest.mark.parametrize("suffix", [".building", "", ".failed"])
def test_existing_root_refusal(tmp_path, suffix):
    build_id = "20260715T220000123456Z_abcdef01"
    roots = derive_snapshot_roots(tmp_path, build_id)
    target = {".building": roots.building, "": roots.final, ".failed": roots.failed}[suffix]
    target.mkdir(parents=True)
    with pytest.raises(SnapshotLifecycleError, match="refusing to reuse"):
        create_fresh_staging_root(tmp_path, build_id)


def test_same_volume_validation_passes_for_shared_root(tmp_path):
    roots = derive_snapshot_roots(tmp_path, "20260715T220000123456Z_abcdef01")
    validate_same_volume_publication(roots.building, roots.final)  # no raise


@pytest.mark.skipif(os.name != "nt", reason="drive letters only on Windows")
def test_same_volume_validation_rejects_cross_volume():
    with pytest.raises(SnapshotLifecycleError, match="same volume"):
        validate_same_volume_publication(
            "C:/snap/build.building", "D:/snap/build"
        )


# ── 1.4 path safety ─────────────────────────────────────────────────────────────


def test_valid_root_relative_resolution(tmp_path):
    resolved = resolve_under_root(tmp_path, "input/liquidity/liquid_tickers.csv")
    assert resolved == (tmp_path / "input" / "liquidity" / "liquid_tickers.csv").resolve()
    # backslash separators normalize
    resolved_win = resolve_under_root(tmp_path, "cache\\spot_prices_adjusted.parquet")
    assert resolved_win == (tmp_path / "cache" / "spot_prices_adjusted.parquet").resolve()


def test_absolute_posix_path_rejected(tmp_path):
    with pytest.raises(SnapshotPathError, match="absolute"):
        resolve_under_root(tmp_path, "/etc/passwd")


def test_windows_drive_path_rejected(tmp_path):
    with pytest.raises(SnapshotPathError, match="drive-qualified"):
        resolve_under_root(tmp_path, "C:/Windows/system32")


def test_dotdot_escape_rejected(tmp_path):
    with pytest.raises(SnapshotPathError, match=r"\.\."):
        resolve_under_root(tmp_path, "input/../../secret.txt")


# ── 1.5 adjusted inventory resolver ────────────────────────────────────────────


def test_adjusted_inventory_resolves_and_excludes_empty_weekend(tmp_path):
    root = tmp_path / "adj"
    _write_day(root, date(2024, 1, 2), _good_frame())
    _write_day(root, date(2024, 1, 3), _good_frame())
    _write_day(root, date(2024, 1, 7), _empty_frame())  # Sunday, empty

    inv = resolve_adjusted_inventory(root, 2024, 2024)
    assert inv.physical_dates == (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 7))
    assert inv.resolved_trading_dates == (date(2024, 1, 2), date(2024, 1, 3))
    assert inv.weekend_excluded_dates == (date(2024, 1, 7),)
    assert inv.date_min == date(2024, 1, 2)
    assert inv.date_max == date(2024, 1, 7)


def test_nonempty_weekend_fails(tmp_path):
    root = tmp_path / "adj"
    _write_day(root, date(2024, 1, 2), _good_frame())
    _write_day(root, date(2024, 1, 6), _good_frame())  # Saturday with data
    with pytest.raises(AdjustedInventoryError, match="weekend-dated file contains data"):
        resolve_adjusted_inventory(root, 2024, 2024)


def test_wrong_year_weekend_fails_before_exclusion(tmp_path):
    root = tmp_path / "adj"
    _write_day(root, date(2024, 1, 2), _good_frame())
    # A 2024 Sunday file placed under the 2023 directory.
    stray = root / "2023" / _FILENAME.format("20240107")
    stray.parent.mkdir(parents=True, exist_ok=True)
    _empty_frame().to_parquet(stray, index=False)
    with pytest.raises(AdjustedInventoryError, match="does not belong"):
        resolve_adjusted_inventory(root, 2023, 2024)


def test_malformed_adjusted_filename_fails(tmp_path):
    root = tmp_path / "adj"
    _write_day(root, date(2024, 1, 2), _good_frame())
    bad = root / "2024" / "ORATS_SMV_Strikes_2024010.parquet"
    bad.write_bytes(b"x")
    with pytest.raises(AdjustedInventoryError, match="malformed adjusted filename"):
        resolve_adjusted_inventory(root, 2024, 2024)


def test_empty_inventory_fails(tmp_path):
    root = tmp_path / "adj"
    (root / "2024").mkdir(parents=True)
    with pytest.raises(AdjustedInventoryError, match="no adjusted dates"):
        resolve_adjusted_inventory(root, 2024, 2024)


def test_missing_data_root_fails(tmp_path):
    with pytest.raises(AdjustedInventoryError, match="data root"):
        resolve_adjusted_inventory(tmp_path / "nope", 2024, 2024)


def test_split_history_file_not_in_inventory(tmp_path):
    root = tmp_path / "adj"
    _write_day(root, date(2024, 1, 2), _good_frame())
    # Split history lives at the adjusted root, not a year dir, and does not
    # match the daily filename pattern.
    _good_frame().to_parquet(root / "splits_hist_liquid.parquet", index=False)
    inv = resolve_adjusted_inventory(root, 2024, 2024)
    assert inv.physical_dates == (date(2024, 1, 2),)


# ── 1.6 inventory and bundle identity ──────────────────────────────────────────


def test_adjusted_inventory_digest_stability(tmp_path):
    root = tmp_path / "adj"
    p1 = _write_day(root, date(2024, 1, 2), _good_frame())
    p2 = _write_day(root, date(2024, 1, 3), _good_frame())
    # Order of paths must not matter.
    d1 = adjusted_inventory_digest(root, [p1, p2])
    d2 = adjusted_inventory_digest(root, [p2, p1])
    assert d1 == d2
    assert len(d1) == 64


def test_upstream_bundle_id_stability_and_length():
    kwargs = dict(
        c4_evidence_id="c4",
        c5_evidence_id="c5",
        liquid_tickers_sha256="a" * 64,
        splits_sha256="b" * 64,
        adjusted_inventory_digest="c" * 64,
    )
    first = upstream_bundle_id(**kwargs)
    second = upstream_bundle_id(**kwargs)
    assert first == second
    assert len(first) == 16
    changed = dict(kwargs, splits_sha256="d" * 64)
    assert upstream_bundle_id(**changed) != first


def test_ticker_date_keys_digest_order_independent():
    keys_a = [(date(2024, 1, 2), "AAPL"), (date(2024, 1, 3), "MSFT")]
    keys_b = [(date(2024, 1, 3), "MSFT"), (date(2024, 1, 2), "AAPL"), (date(2024, 1, 2), "AAPL")]
    assert ticker_date_keys_digest(keys_a) == ticker_date_keys_digest(keys_b)


# ── source/copy identity gate ──────────────────────────────────────────────────


def test_source_copy_identity_match_and_mismatch():
    source = {"liquid_tickers_sha256": "a", "splits_sha256": "b"}
    ok = gate_source_copy_identity(source, dict(source))
    assert ok.status == GATE_PASS

    bad = gate_source_copy_identity(source, {"liquid_tickers_sha256": "a", "splits_sha256": "X"})
    assert bad.status == GATE_FAIL
    assert any("splits_sha256" in f for f in bad.failures)


# ── 1.7 gate result and reconciliation ─────────────────────────────────────────


def test_gate_result_json_compatible_structure():
    result = GateResult(name="g", status=GATE_PASS, metrics={"n": 1})
    payload = result.to_dict()
    assert payload == {
        "name": "g",
        "status": "PASS",
        "metrics": {"n": 1},
        "failures": [],
        "warnings": [],
    }
    # Must be JSON-encodable.
    canonical_json_bytes(payload)

    with pytest.raises(ValueError, match="invalid gate status"):
        GateResult(name="g", status="MAYBE")


def test_spot_summary_reconciliation_pass_and_fail():
    good = {
        "source_ticker_date_key_count": 10,
        "source_ticker_date_key_digest": "a" * 64,
        "output_ticker_date_key_count": 8,
        "output_ticker_date_key_digest": "b" * 64,
        "ambiguous_exclusion_count": 2,
        "ambiguous_exclusions": [["2024-01-02", "SPX"], ["2024-01-03", "DJX"]],
        "output_row_count": 8,
    }
    assert gate_spot_summary_reconciliation(good).status == GATE_PASS

    bad = dict(good, output_row_count=7)
    result = gate_spot_summary_reconciliation(bad)
    assert result.status == GATE_FAIL
    assert any("output_row_count" in f for f in result.failures)

    missing = {k: v for k, v in good.items() if k != "ambiguous_exclusions"}
    assert gate_spot_summary_reconciliation(missing).status == GATE_FAIL
