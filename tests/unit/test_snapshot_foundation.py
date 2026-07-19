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
    GATE_WARN,
    GateResult,
    SiblingBuildLock,
    SnapshotLifecycleError,
    SnapshotLockError,
    SnapshotPathError,
    SNAPSHOT_BUILD_ID_RE,
    sibling_lock_path,
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


_D1 = date(2024, 1, 2)
_D2 = date(2024, 1, 3)


def _make_spot_summary(
    source_keys,
    output_keys,
    resolved_dates,
    exclusions,
    *,
    producer_status=None,
    warnings=None,
):
    """Build a fully-populated compact spot summary consistent with the keys."""
    resolved = sorted(set(resolved_dates))
    excl = sorted([d.isoformat(), t] for d, t in exclusions)
    status = producer_status
    if status is None:
        status = "WARN" if excl else "PASS"
    warn = warnings
    if warn is None:
        warn = (
            [f"dropped {len(excl)} ticker-date keys with inconsistent repeated spot values"]
            if excl
            else []
        )
    return {
        "resolved_date_count": len(resolved),
        "resolved_date_min": resolved[0].isoformat() if resolved else None,
        "resolved_date_max": resolved[-1].isoformat() if resolved else None,
        "weekend_excluded_dates": [],
        "source_ticker_date_key_count": len(set(source_keys)),
        "source_ticker_date_key_digest": ticker_date_keys_digest(source_keys),
        "output_ticker_date_key_count": len(set(output_keys)),
        "output_ticker_date_key_digest": ticker_date_keys_digest(output_keys),
        "ambiguous_exclusion_count": len(excl),
        "ambiguous_exclusions": excl,
        "output_row_count": len(set(output_keys)),
        "producer_status": status,
        "warnings": warn,
    }


def _clean_source_output():
    source = {(_D1, "AAPL"), (_D1, "MSFT"), (_D2, "AAPL"), (_D2, "MSFT")}
    return source, set(source), [_D1, _D2]


def _reconcile(summary, source_keys, output_keys, resolved_dates):
    return gate_spot_summary_reconciliation(
        summary,
        source_keys=source_keys,
        output_keys=output_keys,
        resolved_trading_dates=resolved_dates,
    )


def test_spot_summary_reconciliation_pass_clean():
    source, output, resolved = _clean_source_output()
    summary = _make_spot_summary(source, output, resolved, [])
    assert _reconcile(summary, source, output, resolved).status == GATE_PASS


def test_spot_summary_reconciliation_warns_on_exclusions():
    source = {(_D1, "AAPL"), (_D1, "MSFT"), (_D2, "AAPL"), (_D2, "DJX")}
    output = source - {(_D2, "DJX")}
    resolved = [_D1, _D2]
    summary = _make_spot_summary(source, output, resolved, [(_D2, "DJX")])
    result = _reconcile(summary, source, output, resolved)
    assert result.status == GATE_WARN
    assert any("ambiguous" in w for w in result.warnings)


def test_spot_summary_reconciliation_rejects_arbitrary_digest():
    """A digest that is not derived from the keys must FAIL, not pass."""
    source, output, resolved = _clean_source_output()
    summary = _make_spot_summary(source, output, resolved, [])
    summary["output_ticker_date_key_digest"] = "b" * 64
    result = _reconcile(summary, source, output, resolved)
    assert result.status == GATE_FAIL
    assert any("output_ticker_date_key_digest" in f for f in result.failures)


def test_spot_summary_reconciliation_output_must_equal_source_minus_exclusions():
    source = {(_D1, "AAPL"), (_D1, "MSFT"), (_D2, "AAPL"), (_D2, "DJX")}
    # Output drops DJX but summary claims no exclusion -> reconciliation fails.
    output = source - {(_D2, "DJX")}
    resolved = [_D1, _D2]
    summary = _make_spot_summary(source, output, resolved, [])
    result = _reconcile(summary, source, output, resolved)
    assert result.status == GATE_FAIL
    assert any("source_keys - ambiguous_exclusions" in f for f in result.failures)


def test_spot_summary_reconciliation_exclusion_must_be_in_source_absent_from_output():
    source, output, resolved = _clean_source_output()
    # Claim an exclusion that is not in the source at all.
    summary = _make_spot_summary(source, output, resolved, [(_D2, "NOPE")])
    result = _reconcile(summary, source, output, resolved)
    assert result.status == GATE_FAIL
    assert any("absent from source keys" in f for f in result.failures)


def test_spot_summary_reconciliation_resolved_dates_must_match():
    source, output, resolved = _clean_source_output()
    summary = _make_spot_summary(source, output, resolved, [])
    # Independently derived resolved dates disagree with the summary.
    result = _reconcile(summary, source, output, [_D1])
    assert result.status == GATE_FAIL
    assert any("resolved_date_count" in f for f in result.failures)


def test_spot_summary_reconciliation_missing_field_fails():
    source, output, resolved = _clean_source_output()
    summary = _make_spot_summary(source, output, resolved, [])
    del summary["ambiguous_exclusions"]
    assert _reconcile(summary, source, output, resolved).status == GATE_FAIL


def test_spot_summary_reconciliation_schema_bad_types_fail_clearly():
    source, output, resolved = _clean_source_output()

    bad_count = _make_spot_summary(source, output, resolved, [])
    bad_count["output_row_count"] = "4"
    res = _reconcile(bad_count, source, output, resolved)
    assert res.status == GATE_FAIL
    assert any("nonnegative integer" in f for f in res.failures)

    bad_digest = _make_spot_summary(source, output, resolved, [])
    bad_digest["source_ticker_date_key_digest"] = "NOTHEX"
    res = _reconcile(bad_digest, source, output, resolved)
    assert res.status == GATE_FAIL
    assert any("64-char lowercase hex" in f for f in res.failures)

    bad_excl = _make_spot_summary(source, output, resolved, [])
    bad_excl["ambiguous_exclusions"] = [["2024-01-02"]]
    bad_excl["ambiguous_exclusion_count"] = 1
    res = _reconcile(bad_excl, source, output, resolved)
    assert res.status == GATE_FAIL
    assert any("YYYY-MM-DD" in f for f in res.failures)

    bad_status = _make_spot_summary(source, output, resolved, [])
    bad_status["producer_status"] = "MAYBE"
    res = _reconcile(bad_status, source, output, resolved)
    assert res.status == GATE_FAIL
    assert any("producer_status" in f for f in res.failures)


# ── 1.3b sibling snapshot build lock (C8.3B) ───────────────────────────────────

_LOCK_BUILD_ID = "20260717T220000123456Z_abcdef01"


def test_sibling_lock_path_is_sibling_of_building(tmp_path):
    path = sibling_lock_path(tmp_path, _LOCK_BUILD_ID)
    assert path == tmp_path / f"{_LOCK_BUILD_ID}.lock"
    building = derive_snapshot_roots(tmp_path, _LOCK_BUILD_ID).building
    # The lock never lives inside the candidate snapshot root.
    assert building not in path.parents


def test_exclusive_sibling_lock_blocks_second_holder(tmp_path):
    first = SiblingBuildLock(tmp_path, _LOCK_BUILD_ID)
    second = SiblingBuildLock(tmp_path, _LOCK_BUILD_ID)
    with first:
        # Held for the whole context: a second holder must fail clearly.
        assert first.held
        with pytest.raises(SnapshotLockError, match="another process holds"):
            second.acquire()
        assert first.held  # still held after the contention attempt
    assert not first.held


def test_lock_release_allows_new_holder_and_leaves_sibling_file(tmp_path):
    lock_file = sibling_lock_path(tmp_path, _LOCK_BUILD_ID)
    with SiblingBuildLock(tmp_path, _LOCK_BUILD_ID):
        assert lock_file.exists()
    # Normal release closes the handle but never moves or deletes the file.
    assert lock_file.exists()
    with SiblingBuildLock(tmp_path, _LOCK_BUILD_ID) as again:
        assert again.held
    assert lock_file.exists()


def test_lock_release_leaves_no_lock_inside_snapshot_roots(tmp_path):
    roots = create_fresh_staging_root(tmp_path, _LOCK_BUILD_ID)
    with SiblingBuildLock(tmp_path, _LOCK_BUILD_ID):
        pass
    assert list(roots.building.rglob("*.lock")) == []
    assert sibling_lock_path(tmp_path, _LOCK_BUILD_ID).exists()


def test_lock_double_acquire_on_same_object_fails(tmp_path):
    lock = SiblingBuildLock(tmp_path, _LOCK_BUILD_ID)
    lock.acquire()
    try:
        with pytest.raises(SnapshotLockError, match="already held"):
            lock.acquire()
    finally:
        lock.release()
    lock.release()  # idempotent release is safe
