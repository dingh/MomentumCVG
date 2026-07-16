"""Unit tests for weekly input snapshot receipt (Sprint 004 C1)."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.data import input_snapshot
from src.data.input_snapshot import (
    ARTIFACT_ADJUSTED_CHAINS_ROOT,
    ARTIFACT_LIQUID_TICKERS,
    ARTIFACT_LIQUIDITY_DAILY,
    ARTIFACT_LIQUIDITY_PANEL,
    ARTIFACT_LIQUIDITY_WEEKLY,
    ARTIFACT_OPTION_SURFACE_META,
    ARTIFACT_OPTION_SURFACE_QUOTES,
    ARTIFACT_SPLITS,
    ARTIFACT_SPOT_PRICES,
    DEFAULT_DATA_SOURCE,
    INPUT_SNAPSHOT_SCHEMA_VERSION,
    InputSnapshotManifest,
    compute_snapshot_id,
    default_manifest_path,
    generate_build_id,
    manifest_from_dict,
    manifest_to_dict,
    read_manifest,
    write_manifest,
)


def _identity_fields(**overrides):
    base = {
        "schema_version": INPUT_SNAPSHOT_SCHEMA_VERSION,
        "as_of_resolved_trading_day": date(2026, 6, 26),
        "data_source": DEFAULT_DATA_SOURCE,
        "artifacts": {
            ARTIFACT_SPLITS: "splits_hist.parquet",
            ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
            ARTIFACT_LIQUIDITY_PANEL: "ticker_liquidity_panel.parquet",
            ARTIFACT_OPTION_SURFACE_META: "option_surface_meta_weekly_2018_2026.parquet",
            ARTIFACT_OPTION_SURFACE_QUOTES: "option_surface_quotes_weekly_2018_2026.parquet",
        },
        "params": {
            "rolling_months": 3,
            "universe_rule": "top_20_pct_and_filter",
            "feature_branch": "deferred_to_sprint005",
        },
    }
    base.update(overrides)
    return base


def _sample_manifest(**overrides) -> InputSnapshotManifest:
    identity = _identity_fields()
    defaults = {
        "schema_version": identity["schema_version"],
        "snapshot_id": compute_snapshot_id(identity),
        "build_id": "20260621T143022Z_a1b2c3",
        "created_at_utc": datetime(2026, 6, 21, 14, 30, 22, tzinfo=timezone.utc),
        "as_of_requested": date(2026, 6, 26),
        "as_of_resolved_trading_day": identity["as_of_resolved_trading_day"],
        "data_source": identity["data_source"],
        "cache_dir": "C:/MomentumCVG_env/cache",
        "artifacts": identity["artifacts"],
        "params": identity["params"],
        "reports": {
            "validate": None,
            "split_audit": None,
            "surface_audit": None,
        },
        "overall_status": None,
        "blocking_failures": [],
        "notes": [],
    }
    defaults.update(overrides)
    return InputSnapshotManifest(**defaults)


class TestComputeSnapshotId:
    def test_same_identity_fields_produce_same_snapshot_id(self):
        first = compute_snapshot_id(_identity_fields())
        second = compute_snapshot_id(_identity_fields())
        assert first == second
        assert len(first) == 16

    def test_different_as_of_resolved_trading_day_changes_snapshot_id(self):
        base = _identity_fields()
        other = _identity_fields(as_of_resolved_trading_day=date(2026, 6, 20))
        assert compute_snapshot_id(base) != compute_snapshot_id(other)

    def test_different_artifact_path_changes_snapshot_id(self):
        base = _identity_fields()
        artifacts = dict(base["artifacts"])
        artifacts[ARTIFACT_SPLITS] = "other_splits.parquet"
        other = _identity_fields(artifacts=artifacts)
        assert compute_snapshot_id(base) != compute_snapshot_id(other)

    def test_different_rolling_months_changes_snapshot_id(self):
        base = _identity_fields()
        params = dict(base["params"])
        params["rolling_months"] = 6
        other = _identity_fields(params=params)
        assert compute_snapshot_id(base) != compute_snapshot_id(other)

    def test_excluded_fields_do_not_change_snapshot_id(self):
        identity = _identity_fields()
        base_id = compute_snapshot_id(identity)

        manifest_a = _sample_manifest(
            build_id="build_a",
            created_at_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            cache_dir="C:/cache/a",
            as_of_requested=date(2026, 6, 25),
            reports={"validate": "reports/a.md", "split_audit": None, "surface_audit": None},
            overall_status="WARN",
            blocking_failures=["x"],
            notes=["note"],
        )
        manifest_b = _sample_manifest(
            build_id="build_b",
            created_at_utc=datetime(2026, 2, 2, 12, 0, 0, tzinfo=timezone.utc),
            cache_dir="C:/cache/b",
            as_of_requested=date(2026, 6, 27),
            reports={"validate": None, "split_audit": "reports/b.md", "surface_audit": None},
            overall_status="FAIL",
            blocking_failures=["y", "z"],
            notes=["other"],
        )

        assert compute_snapshot_id(manifest_a) == base_id
        assert compute_snapshot_id(manifest_b) == base_id

    def test_artifact_dict_key_order_does_not_affect_snapshot_id(self):
        artifacts_a = {
            ARTIFACT_SPLITS: "splits_hist.parquet",
            ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
            ARTIFACT_LIQUIDITY_PANEL: "ticker_liquidity_panel.parquet",
            ARTIFACT_OPTION_SURFACE_META: "option_surface_meta_weekly_2018_2026.parquet",
            ARTIFACT_OPTION_SURFACE_QUOTES: "option_surface_quotes_weekly_2018_2026.parquet",
        }
        artifacts_b = {
            ARTIFACT_OPTION_SURFACE_QUOTES: "option_surface_quotes_weekly_2018_2026.parquet",
            ARTIFACT_OPTION_SURFACE_META: "option_surface_meta_weekly_2018_2026.parquet",
            ARTIFACT_LIQUIDITY_PANEL: "ticker_liquidity_panel.parquet",
            ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
            ARTIFACT_SPLITS: "splits_hist.parquet",
        }
        first = compute_snapshot_id(_identity_fields(artifacts=artifacts_a))
        second = compute_snapshot_id(_identity_fields(artifacts=artifacts_b))
        assert first == second


class TestManifestSerialization:
    def test_round_trip_preserves_values(self):
        manifest = _sample_manifest(
            overall_status="PASS",
            blocking_failures=["none"],
            notes=["ok"],
            reports={
                "validate": "manifests/reports/validate_x.md",
                "split_audit": None,
                "surface_audit": "manifests/reports/surface_audit_x.md",
            },
        )
        restored = manifest_from_dict(manifest_to_dict(manifest))

        assert restored.schema_version == manifest.schema_version
        assert restored.snapshot_id == manifest.snapshot_id
        assert restored.build_id == manifest.build_id
        assert restored.created_at_utc == manifest.created_at_utc
        assert restored.as_of_requested == manifest.as_of_requested
        assert restored.as_of_resolved_trading_day == manifest.as_of_resolved_trading_day
        assert restored.data_source == manifest.data_source
        assert restored.cache_dir == manifest.cache_dir
        assert restored.artifacts == manifest.artifacts
        assert restored.params == manifest.params
        assert restored.reports == manifest.reports
        assert restored.overall_status == manifest.overall_status
        assert restored.blocking_failures == manifest.blocking_failures
        assert restored.notes == manifest.notes

    def test_unknown_schema_version_raises_clear_error(self):
        payload = manifest_to_dict(_sample_manifest())
        payload["schema_version"] = "2"
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            manifest_from_dict(payload)

    def test_none_artifact_path_raises(self):
        payload = manifest_to_dict(_sample_manifest())
        payload["artifacts"][ARTIFACT_SPLITS] = None
        with pytest.raises(ValueError, match="artifacts values must be strings"):
            manifest_from_dict(payload)

    def test_invalid_report_value_raises(self):
        payload = manifest_to_dict(_sample_manifest())
        payload["reports"]["validate"] = 123
        with pytest.raises(ValueError, match="reports values must be str or None"):
            manifest_from_dict(payload)

    def test_artifact_paths_serialize_with_forward_slashes(self):
        manifest = _sample_manifest(
            artifacts={
                ARTIFACT_SPLITS: "subdir\\splits_hist.parquet",
                ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
                ARTIFACT_LIQUIDITY_PANEL: "ticker_liquidity_panel.parquet",
                ARTIFACT_OPTION_SURFACE_META: "option_surface_meta_weekly_2018_2026.parquet",
                ARTIFACT_OPTION_SURFACE_QUOTES: "option_surface_quotes_weekly_2018_2026.parquet",
            }
        )
        serialized = manifest_to_dict(manifest)
        assert serialized["artifacts"][ARTIFACT_SPLITS] == "subdir/splits_hist.parquet"

    def test_manifest_to_dict_rejects_invalid_report_value(self):
        manifest = _sample_manifest(
            reports={"validate": 123, "split_audit": None, "surface_audit": None}
        )
        with pytest.raises(ValueError, match="reports values must be str or None"):
            manifest_to_dict(manifest)

    def test_manifest_to_dict_rejects_invalid_overall_status(self):
        manifest = _sample_manifest(overall_status="OK")
        with pytest.raises(ValueError, match="Invalid overall_status"):
            manifest_to_dict(manifest)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", 1),
            ("snapshot_id", None),
            ("build_id", 123),
            ("data_source", None),
            ("cache_dir", Path("C:/MomentumCVG_env/cache")),
        ],
    )
    def test_manifest_from_dict_rejects_non_string_core_fields(self, field, value):
        payload = manifest_to_dict(_sample_manifest())
        payload[field] = value
        with pytest.raises(ValueError, match=f"{field} must be a string"):
            manifest_from_dict(payload)

    @pytest.mark.parametrize("field", ["as_of_requested", "as_of_resolved_trading_day"])
    def test_manifest_to_dict_rejects_datetime_date_fields(self, field):
        manifest = _sample_manifest(**{field: datetime(2026, 6, 26, 10, 30)})
        with pytest.raises(ValueError, match="Expected date"):
            manifest_to_dict(manifest)


class TestComputeSnapshotIdValidation:
    def test_non_json_serializable_params_raise_clear_error(self):
        identity = _identity_fields()
        identity["params"] = {"rolling_months": date(2026, 1, 1)}
        with pytest.raises(ValueError, match="snapshot_id identity fields must be JSON-serializable"):
            compute_snapshot_id(identity)

    def test_non_mapping_params_raises(self):
        identity = _identity_fields()
        identity["params"] = [("rolling_months", 3)]
        with pytest.raises(ValueError, match="params must be a mapping"):
            compute_snapshot_id(identity)

    def test_datetime_as_of_resolved_trading_day_raises(self):
        identity = _identity_fields(
            as_of_resolved_trading_day=datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)
        )
        with pytest.raises(ValueError, match="Invalid date for as_of_resolved_trading_day"):
            compute_snapshot_id(identity)

    def test_non_string_schema_version_raises(self):
        identity = _identity_fields(schema_version=1)
        with pytest.raises(ValueError, match="schema_version must be a string"):
            compute_snapshot_id(identity)

    def test_non_string_data_source_raises(self):
        identity = _identity_fields(data_source=None)
        with pytest.raises(ValueError, match="data_source must be a string"):
            compute_snapshot_id(identity)

    def test_mapping_artifacts_accepted(self):
        from types import MappingProxyType

        identity = _identity_fields()
        identity["artifacts"] = MappingProxyType(dict(identity["artifacts"]))
        snapshot_id = compute_snapshot_id(identity)
        assert len(snapshot_id) == 16


class TestManifestIO:
    def test_write_and_read_manifest(self, tmp_path):
        manifest = _sample_manifest()
        path = tmp_path / "manifests" / "input_snapshot_test.json"
        write_manifest(path, manifest)
        restored = read_manifest(path)
        assert restored == manifest

    def test_atomic_write_success_leaves_no_temp_file(self, tmp_path):
        manifest = _sample_manifest()
        path = tmp_path / "manifests" / "input_snapshot_test.json"
        write_manifest(path, manifest)
        assert path.exists()
        assert list(path.parent.glob("*.tmp-*")) == []
        assert read_manifest(path) == manifest

    def test_failed_write_preserves_existing_manifest(self, tmp_path, monkeypatch):
        path = tmp_path / "manifests" / "input_snapshot_test.json"
        write_manifest(path, _sample_manifest())
        original_bytes = path.read_bytes()

        # A serialization failure must not damage the existing target or leave
        # a temp file behind.
        def _boom(*args, **kwargs):
            raise OSError("simulated serialization failure")

        monkeypatch.setattr(input_snapshot.json, "dump", _boom)

        changed = _sample_manifest(notes=["changed"])
        with pytest.raises(OSError):
            write_manifest(path, changed)

        assert path.read_bytes() == original_bytes
        assert list(path.parent.glob("*.tmp-*")) == []


class TestSchemaV1Compatibility:
    def test_existing_schema_v1_manifest_still_loads(self):
        # A pre-C8 manifest with no production_accepted / new artifact keys.
        legacy = {
            "schema_version": "1",
            "snapshot_id": "abc123def4567890",
            "build_id": "20260621T143022Z_a1b2c3",
            "created_at_utc": "2026-06-21T14:30:22Z",
            "as_of_requested": "2026-06-26",
            "as_of_resolved_trading_day": "2026-06-26",
            "data_source": DEFAULT_DATA_SOURCE,
            "cache_dir": "C:/MomentumCVG_env/cache",
            "artifacts": {
                ARTIFACT_SPLITS: "splits_hist.parquet",
                ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
            },
            "params": {"rolling_months": 3},
            "reports": {"validate": None},
            "overall_status": None,
            "blocking_failures": [],
            "notes": [],
        }
        manifest = manifest_from_dict(legacy)
        assert manifest.schema_version == "1"
        assert manifest.production_accepted is None
        # Round-trips without introducing a production_accepted key.
        assert "production_accepted" not in manifest_to_dict(manifest)

    def test_existing_snapshot_id_vector_is_unchanged(self):
        # Golden vector pinned against the accepted C1 identity.
        assert compute_snapshot_id(_identity_fields()) == "4a42436f29f77a4a"


class TestC8IdentityAndArtifacts:
    def test_scope_in_params_changes_snapshot_id(self):
        base = _identity_fields()
        params = dict(base["params"])
        params["scope"] = "full"
        with_scope = _identity_fields(params=params)
        assert compute_snapshot_id(with_scope) != compute_snapshot_id(base)

    def test_execution_outcome_changes_do_not_change_snapshot_id(self):
        base_id = compute_snapshot_id(_identity_fields())
        accepted = _sample_manifest(
            overall_status="PASS",
            production_accepted=True,
            blocking_failures=[],
            notes=["published"],
        )
        rejected = _sample_manifest(
            overall_status="FAIL",
            production_accepted=False,
            blocking_failures=["gate SF failed"],
            notes=["not published"],
        )
        assert compute_snapshot_id(accepted) == base_id
        assert compute_snapshot_id(rejected) == base_id

    def test_new_artifact_constants_round_trip(self):
        artifacts = {
            ARTIFACT_LIQUIDITY_DAILY: "input/liquidity/ticker_liquidity_daily_observations.parquet",
            ARTIFACT_LIQUIDITY_WEEKLY: "input/liquidity/ticker_liquidity_weekly_observations.parquet",
            ARTIFACT_LIQUIDITY_PANEL: "input/liquidity/ticker_liquidity_panel.parquet",
            ARTIFACT_LIQUID_TICKERS: "input/liquidity/liquid_tickers.csv",
            ARTIFACT_SPLITS: "input/adjusted_liquid/splits_hist_liquid.parquet",
            ARTIFACT_ADJUSTED_CHAINS_ROOT: "input/adjusted_liquid",
            ARTIFACT_SPOT_PRICES: "cache/spot_prices_adjusted.parquet",
            ARTIFACT_OPTION_SURFACE_META: "cache/option_surface_meta_weekly_2018_2026.parquet",
            ARTIFACT_OPTION_SURFACE_QUOTES: "cache/option_surface_quotes_weekly_2018_2026.parquet",
        }
        manifest = _sample_manifest(artifacts=artifacts)
        restored = manifest_from_dict(manifest_to_dict(manifest))
        assert restored.artifacts == artifacts

    def test_new_c8_manifest_round_trips_without_field_loss(self, tmp_path):
        params = {
            "scope": "full",
            "upstream_bundle_id": "0123456789abcdef",
            "effective_ticker_universe_sha256": "e" * 64,
            "frequency": "weekly",
            "adjusted_inventory_digest": "d" * 64,
            "feature_ready_start": "2018-01-05",
            "feature_ready_end": "2026-02-20",
        }
        artifacts = {
            ARTIFACT_LIQUID_TICKERS: "input/liquidity/liquid_tickers.csv",
            ARTIFACT_SPLITS: "input/adjusted_liquid/splits_hist_liquid.parquet",
            ARTIFACT_ADJUSTED_CHAINS_ROOT: "input/adjusted_liquid",
            ARTIFACT_SPOT_PRICES: "cache/spot_prices_adjusted.parquet",
        }
        reports = {
            "copy_identity": "reports/copy_identity.json",
            "c5_audit": "reports/c5_audit.md",
            "spot_summary": "reports/spot_summary.json",
        }
        manifest = _sample_manifest(
            params=params,
            artifacts=artifacts,
            reports=reports,
            overall_status="PASS",
            production_accepted=True,
            notes=["scope=full"],
        )
        path = tmp_path / "manifests" / "input_snapshot_c8.json"
        write_manifest(path, manifest)
        restored = read_manifest(path)

        assert restored.params == params
        assert restored.artifacts == artifacts
        assert restored.reports == reports
        assert restored.production_accepted is True
        assert restored.overall_status == "PASS"
        assert restored == manifest


class TestGenerateBuildId:
    BUILD_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{6}$")

    def test_format_matches_expected_pattern(self):
        now = datetime(2026, 6, 21, 14, 30, 22, tzinfo=timezone.utc)
        build_id = generate_build_id(now=now, command="refresh --as-of 2026-06-26")
        assert self.BUILD_ID_PATTERN.match(build_id)

    def test_same_now_different_command_gives_different_build_id(self):
        now = datetime(2026, 6, 21, 14, 30, 22, tzinfo=timezone.utc)
        first = generate_build_id(now=now, command="plan")
        second = generate_build_id(now=now, command="validate")
        assert first != second


class TestDefaultManifestPath:
    def test_returns_expected_path(self):
        path = default_manifest_path("C:/MomentumCVG_env/cache", "abc123def4567890")
        expected = Path("C:/MomentumCVG_env/cache/manifests/input_snapshot_abc123def4567890.json")
        assert path == expected
