"""Sprint 006 D1 — frozen-contract adapter tests.

Covers contract identity, recognized-field mapping, accepted-path behavior,
dry-run/preflight, overwrite refusal, and end-to-end writing through the real
``SurfaceRunner`` on synthetic fixtures. No accepted real-data artifacts are read
and no economic P&L is asserted.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest import sprint006_baseline as sb
from src.backtest.option_surface import FillAssumption
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner, SurfaceRunResult
from tests.unit.test_surface_runner_data_flow import (
    TRADE_DATE,
    _build_features,
    _build_liquidity_panel,
    _build_surface_parquets,
    _make_config,
)

FROZEN_CONTRACT_PATH = Path("configs/sprint006_baseline_v1.json")


@pytest.fixture
def synthetic_runner(tmp_path: Path) -> SurfaceRunner:
    """Same synthetic surface the canonical runner tests use."""
    meta_path, quotes_path = _build_surface_parquets(tmp_path)
    liquidity_path = _build_liquidity_panel(tmp_path)
    features_dir = _build_features(tmp_path).parent
    return SurfaceRunner(
        data_paths=SurfaceDataPaths(
            cache_dir=tmp_path,
            features_dir=features_dir,
            liquidity_panel_path=liquidity_path,
            surface_meta_path=meta_path,
            surface_quotes_path=quotes_path,
            earnings_path=None,
        )
    )


# =============================================================================
# Synthetic contract fixtures
# =============================================================================

def _synthetic_contract_payload(paths: dict) -> dict:
    """A contract with the frozen shape but synthetic paths and loose thresholds.

    Thresholds are chosen so the tiny synthetic cross-section trades; they are
    not Sprint 006 economic parameters.
    """
    return {
        "contract_id": "sprint006_baseline_v1",
        "contract_version": 1,
        "status": "accepted",
        "accepted_inputs": {
            "features_dir": str(paths["features_dir"]),
            "baseline_feature_file": str(paths["feature_file"]),
            "a1_surface_meta": str(paths["meta"]),
            "a2_surface_quotes": str(paths["quotes"]),
            "liquidity_panel": str(paths["liquidity"]),
            "manifest": str(paths["manifest"]),
            "d3_receipt": str(paths["receipt"]),
            "earnings_path": None,
            "mutable_cache_forbidden": True,
        },
        "feature_window": {
            "max_lag": 42,
            "min_lag": 8,
            "momentum_col": "mom_42_8_mean",
            "cvg_col": "cvg_42_8",
            "count_col": "mom_42_8_count",
            "cvg_count_col": "cvg_count_42_8",
            "search": False,
        },
        "periods": {
            "run_start_date": "2024-01-05",
            "run_end_date": "2024-01-06",
        },
        "shared_run_config": {
            "min_count_pct": 0.5,
            "long_top_pct": 0.25,
            "short_bottom_pct": 0.5,
            "cvg_filter_pct": 1.0,
            "dvol_top_pct": 1.0,
            "spread_bottom_pct": 1.0,
            "short_structure": "ironfly",
            "wing_selection_rule": "closest_delta",
            "wing_selection_rule_note": "prose that must not be mapped",
            "wing_delta_target": 0.25,
            "max_names_per_side": 25,
            "max_loss_budget_per_trade": 500.0,
            "max_loss_budget_per_trade_note": "prose that must not be mapped",
            "earnings_exclusion_days": 0,
            "cost_model": "mid",
            "cost_model_note": "prose that must not be mapped",
            "start_date": "2024-01-05",
            "end_date": "2024-01-06",
            "max_leg_spread_pct": 0.5,
            "max_leg_spread_pct_intent": "prose that must not be mapped",
            "max_spread_cost_ratio": None,
            "condor_short_delta_target": None,
            "condor_long_delta_target": None,
            "include_diagnostics": True,
            "sizing_mode": "conceptual",
            "tier_a_mode": "equal_max_loss",
            "tier_a_short_budget": 10000.0,
            "tier_a_long_budget": 10000.0,
            "tier_a_long_budget_role": "fallback_only",
            "tier_a_fallback_rule": "prose that must not be mapped",
            "tier_b_short_max_loss_budget": None,
            "contract_multiplier": 100.0,
            "deployable_capital": None,
        },
        "runs": [
            {
                "run_id": "sprint006_baseline_v1_mid",
                "role": "diagnostic",
                "fill": {"label": "mid", "buy_alpha": 0.5, "sell_alpha": 0.5},
                "cost_model": "mid",
                "primary_decision_view": False,
            },
            {
                "run_id": "sprint006_baseline_v1_cross",
                "role": "primary",
                "fill": {"label": "cross", "buy_alpha": 1.0, "sell_alpha": 1.0},
                "cost_model": "mid",
                "primary_decision_view": True,
            },
        ],
    }


def _build_synthetic_inputs(root: Path) -> dict:
    meta, quotes = _build_surface_parquets(root)
    liquidity = _build_liquidity_panel(root)
    features_dir = _build_features(root).parent
    manifest = root / "input_snapshot_synthetic.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = root / "features_backfill_v1.lineage.json"
    receipt.write_text("{}", encoding="utf-8")
    return {
        "meta": meta,
        "quotes": quotes,
        "liquidity": liquidity,
        "features_dir": features_dir,
        "feature_file": features_dir / "features_42_8.parquet",
        "manifest": manifest,
        "receipt": receipt,
    }


def _write_contract(root: Path, mutate=None) -> Path:
    payload = _synthetic_contract_payload(_build_synthetic_inputs(root))
    if mutate is not None:
        mutate(payload)
    path = root / "synthetic_contract.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_contract_path(tmp_path: Path) -> Path:
    return _write_contract(tmp_path)


# =============================================================================
# Contract identity
# =============================================================================

class TestContractIdentity:
    def test_frozen_contract_loads_with_expected_identity(self):
        contract = sb.load_contract(FROZEN_CONTRACT_PATH)
        assert contract.contract_id == "sprint006_baseline_v1"
        assert contract.contract_version == 1
        assert contract.status == "accepted"
        assert len(contract.sha256) == 64

    def test_wrong_contract_id_rejected(self, tmp_path: Path):
        path = _write_contract(tmp_path, lambda p: p.update(contract_id="other_experiment"))
        with pytest.raises(sb.ContractError, match="contract_id"):
            sb.load_contract(path)

    def test_unaccepted_status_rejected(self, tmp_path: Path):
        path = _write_contract(tmp_path, lambda p: p.update(status="proposed"))
        with pytest.raises(sb.ContractError, match="status"):
            sb.load_contract(path)

    def test_missing_contract_file_rejected(self, tmp_path: Path):
        with pytest.raises(sb.ContractError, match="not found"):
            sb.load_contract(tmp_path / "absent.json")


# =============================================================================
# Recognized-field mapping / twin configs
# =============================================================================

class TestRunConfigMapping:
    def test_frozen_contract_builds_both_runs(self):
        configs = sb.build_run_configs(sb.load_contract(FROZEN_CONTRACT_PATH))
        assert [c.run_id for c in configs] == [
            "sprint006_baseline_v1_mid",
            "sprint006_baseline_v1_cross",
        ]
        assert [c.fill.label for c in configs] == ["mid", "cross"]

    def test_frozen_contract_values_reach_the_config(self):
        mid, cross = sb.build_run_configs(sb.load_contract(FROZEN_CONTRACT_PATH))
        assert mid.momentum_col == "mom_42_8_mean"
        assert mid.cvg_col == "cvg_42_8"
        assert mid.count_col == "mom_42_8_count"
        assert mid.cvg_count_col == "cvg_count_42_8"
        assert mid.min_count_pct == 0.80
        assert mid.max_names_per_side == 25
        assert mid.spread_bottom_pct == 1.0
        assert mid.short_structure == "ironfly"
        assert mid.wing_delta_target == 0.15
        assert mid.sizing_mode == "conceptual"
        assert mid.tier_a_mode == "equal_max_loss"
        assert mid.tier_a_short_budget == 10000.0
        assert mid.tier_a_long_budget == 10000.0
        assert mid.tier_b_short_max_loss_budget is None
        assert mid.earnings_exclusion_days == 0
        assert mid.max_leg_spread_pct == 0.5
        assert mid.start_date == date(2018, 10, 26)
        assert mid.end_date == date(2026, 7, 10)
        assert cross.cost_model == mid.cost_model == "mid"

    def test_twin_runs_differ_only_by_run_id_and_fill(self):
        mid, cross = sb.build_run_configs(sb.load_contract(FROZEN_CONTRACT_PATH))
        mid_dump = sb.effective_config_dump(mid)
        cross_dump = sb.effective_config_dump(cross)
        differing = {k for k in mid_dump if mid_dump[k] != cross_dump[k]}
        assert differing == {"run_id", "fill"}
        assert mid_dump["fill"] == {"buy_alpha": 0.5, "sell_alpha": 0.5, "label": "mid"}
        assert cross_dump["fill"] == {"buy_alpha": 1.0, "sell_alpha": 1.0, "label": "cross"}

    def test_note_and_intent_keys_are_not_mapped(self, synthetic_contract_path: Path):
        configs = sb.build_run_configs(sb.load_contract(synthetic_contract_path))
        for config in configs:
            assert not hasattr(config, "cost_model_note")
            assert not hasattr(config, "tier_a_fallback_rule")
            assert not hasattr(config, "max_leg_spread_pct_intent")
            assert config.cvg_count_col == "cvg_count_42_8"

    def test_missing_cross_run_rejected(self, tmp_path: Path):
        path = _write_contract(tmp_path, lambda p: p.__setitem__("runs", p["runs"][:1]))
        with pytest.raises(sb.ContractError, match="fill labels"):
            sb.build_run_configs(sb.load_contract(path))

    def test_two_primary_runs_rejected(self, tmp_path: Path):
        def mutate(payload):
            payload["runs"][0]["primary_decision_view"] = True

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="primary_decision_view"):
            sb.build_run_configs(sb.load_contract(path))

    def test_search_enabled_contract_rejected(self, tmp_path: Path):
        def mutate(payload):
            payload["feature_window"]["search"] = True

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="search"):
            sb.build_run_configs(sb.load_contract(path))

    def test_period_disagreement_rejected(self, tmp_path: Path):
        def mutate(payload):
            payload["periods"]["run_end_date"] = "2024-02-01"

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="run dates disagree"):
            sb.build_run_configs(sb.load_contract(path))


# =============================================================================
# Accepted path behavior
# =============================================================================

class TestAcceptedPaths:
    def test_preflight_resolves_accepted_paths(self, synthetic_contract_path: Path):
        checked = sb.preflight(sb.load_contract(synthetic_contract_path))
        accepted = checked.accepted_inputs
        assert checked.data_paths.resolved_features_dir == accepted["features_dir"]
        assert checked.data_paths.resolved_surface_meta_path == accepted["a1_surface_meta"]
        assert checked.data_paths.resolved_surface_quotes_path == accepted["a2_surface_quotes"]
        assert checked.data_paths.resolved_liquidity_panel_path == accepted["liquidity_panel"]
        assert checked.data_paths.earnings_path is None

    def test_snapshot_cache_surface_segment_is_accepted(self, tmp_path: Path):
        """Accepted A1/A2 live under ``<snapshot>/cache/surface`` — that must pass."""
        snapshot_like = tmp_path / "snapshots" / "build_id" / "cache" / "surface"
        snapshot_like.mkdir(parents=True)
        inputs = _build_synthetic_inputs(tmp_path)
        moved = {}
        for key in ("meta", "quotes"):
            target = snapshot_like / inputs[key].name
            inputs[key].replace(target)
            moved[key] = target
        payload = _synthetic_contract_payload({**inputs, **moved})
        path = tmp_path / "snapshot_cache_contract.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        checked = sb.preflight(sb.load_contract(path))
        assert "cache" in checked.data_paths.resolved_surface_meta_path.parts
        assert checked.data_paths.resolved_surface_meta_path == moved["meta"].resolve()

    def test_mutable_producer_cache_root_refused(self, tmp_path: Path):
        def mutate(payload):
            payload["accepted_inputs"]["features_dir"] = str(
                sb.MUTABLE_CACHE_ROOT / "features"
            )

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="mutable producer cache root"):
            sb.preflight(sb.load_contract(path))

    def test_missing_accepted_input_refused(self, tmp_path: Path):
        path = _write_contract(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        Path(payload["accepted_inputs"]["liquidity_panel"]).unlink()
        with pytest.raises(sb.ContractError, match="liquidity_panel does not exist"):
            sb.preflight(sb.load_contract(path))

    def test_earnings_path_must_be_null(self, tmp_path: Path):
        def mutate(payload):
            payload["accepted_inputs"]["earnings_path"] = "C:/somewhere/earnings.parquet"

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="earnings_path"):
            sb.preflight(sb.load_contract(path))

    def test_feature_file_must_match_accepted_baseline(self, tmp_path: Path):
        def mutate(payload):
            payload["feature_window"].update(
                max_lag=60,
                min_lag=24,
                momentum_col="mom_60_24_mean",
                cvg_col="cvg_60_24",
                count_col="mom_60_24_count",
            )

        path = _write_contract(tmp_path, mutate)
        with pytest.raises(sb.ContractError, match="accepted baseline feature file"):
            sb.preflight(sb.load_contract(path))

    def test_frozen_contract_paths_are_not_under_mutable_cache_root(self):
        """Read-only check of the frozen contract; no artifact access."""
        accepted = sb.load_contract(FROZEN_CONTRACT_PATH).payload["accepted_inputs"]
        mutable_root = sb.MUTABLE_CACHE_ROOT.resolve()
        for key in ("features_dir", "baseline_feature_file", "a1_surface_meta",
                    "a2_surface_quotes", "liquidity_panel", "manifest", "d3_receipt"):
            assert not sb._is_inside(Path(accepted[key]).resolve(), mutable_root)


# =============================================================================
# Overwrite refusal
# =============================================================================

class TestOutputLocationRefusal:
    def test_output_dir_inside_repo_refused(self):
        with pytest.raises(sb.ContractError, match="Git repository root"):
            sb.create_run_dir(sb._REPO_ROOT / "runs" / "d1_probe")

    def test_output_dir_inside_mutable_cache_refused(self):
        with pytest.raises(sb.ContractError, match="mutable producer cache root"):
            sb.create_run_dir(sb.MUTABLE_CACHE_ROOT / "runs" / "d1_probe")


class TestOverwriteRefusal:
    def test_existing_run_dir_refused(self, tmp_path: Path):
        existing = tmp_path / "run"
        existing.mkdir()
        with pytest.raises(sb.ContractError, match="existing run output directory"):
            sb.create_run_dir(existing)

    def test_new_run_dir_created(self, tmp_path: Path):
        run_dir = sb.create_run_dir(tmp_path / "run")
        assert run_dir.is_dir()

    def test_existing_artifact_refused(self, tmp_path: Path):
        run_dir = sb.create_run_dir(tmp_path / "run")
        result = SurfaceRunResult(
            config=_make_config(run_id="overwrite_probe"),
            trade_log=pd.DataFrame({"trade_date": [TRADE_DATE], "ticker": ["LONG1"]}),
            date_summary=pd.DataFrame({"trade_date": [TRADE_DATE], "n_traded": [1]}),
            run_summary={"run_id": "overwrite_probe"},
            date_status=pd.DataFrame(
                {
                    "trade_date": [TRADE_DATE],
                    "status": ["traded"],
                    "reason": [None],
                }
            ),
        )
        sb.write_run_outputs(result, run_dir)
        with pytest.raises(sb.ContractError, match="refusing to overwrite existing artifact"):
            sb.write_run_outputs(result, run_dir)


# =============================================================================
# End-to-end adapter execution on synthetic fixtures
# =============================================================================

class TestRunBaselineOnSyntheticFixtures:
    @pytest.fixture
    def outcome(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        contract_path = _write_contract(tmp_path)
        monkeypatch.setattr(sb, "clean_repo_sha", lambda: "a" * 40)
        return sb.run_baseline(
            contract_path=contract_path,
            output_dir=tmp_path / "run",
            command=["run_sprint006_baseline.py", "--dry-run=false"],
        )

    def test_both_runs_execute_and_write_artifacts(self, outcome):
        assert [run["run_id"] for run in outcome["runs"]] == [
            "sprint006_baseline_v1_mid",
            "sprint006_baseline_v1_cross",
        ]
        for run in outcome["runs"]:
            assert run["trade_log_rows"] > 0
            for path in run["outputs"].values():
                assert path.is_file()

    def test_receipt_records_proportional_identity(self, outcome):
        receipt = json.loads(outcome["receipt_path"].read_text(encoding="utf-8"))
        assert receipt["experiment_id"] == "sprint006_baseline_v1"
        assert receipt["repo_sha"] == "a" * 40
        assert receipt["contract"]["contract_id"] == "sprint006_baseline_v1"
        assert receipt["contract"]["contract_version"] == 1
        assert receipt["contract"]["status"] == "accepted"
        assert len(receipt["contract"]["sha256"]) == 64
        assert receipt["command"][0] == "run_sprint006_baseline.py"
        assert [run["fill_label"] for run in receipt["runs"]] == ["mid", "cross"]
        assert receipt["deferred"] == [
            "decision-quality report and dual return views (D3)",
            "real-data smoke, manual trade sample, and full-history execution (D4)",
        ]
        for run in receipt["runs"]:
            assert "n_failed_dates" in run
            assert "has_unresolved_failures" in run
            assert "feature_dates_absent_from_a1" in run
            assert "n_feature_dates_absent_from_a1" in run
            assert set(run["outputs"]) >= {
                "trade_log",
                "date_summary",
                "date_status",
                "run_summary",
            }

    def test_receipt_dumps_effective_configs_and_output_digests(self, outcome):
        receipt = json.loads(outcome["receipt_path"].read_text(encoding="utf-8"))
        for run in receipt["runs"]:
            assert run["effective_config"]["run_id"] == run["run_id"]
            assert run["effective_config"]["sizing_mode"] == "conceptual"
            assert run["effective_config"]["cvg_count_col"] == "cvg_count_42_8"
            assert set(run["outputs"]) == {
                "trade_log",
                "date_summary",
                "date_status",
                "run_summary",
            }
            for artifact in run["outputs"].values():
                assert len(artifact["sha256"]) == 64

    def test_written_date_status_schema(self, outcome):
        for run in outcome["runs"]:
            status = pd.read_parquet(run["outputs"]["date_status"])
            assert list(status.columns) == ["trade_date", "status", "reason"]
            assert not status.empty

    def test_written_trade_log_reloads(self, outcome):
        for run in outcome["runs"]:
            reloaded = pd.read_parquet(run["outputs"]["trade_log"])
            assert len(reloaded) == run["trade_log_rows"]
            assert (reloaded["fill_label"].dropna() == run["fill_label"]).all()

    def test_dirty_repo_blocks_writing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        contract_path = _write_contract(tmp_path)

        def dirty():
            raise sb.ContractError("refusing to write baseline artifacts from a dirty working tree")

        monkeypatch.setattr(sb, "clean_repo_sha", dirty)
        with pytest.raises(sb.ContractError, match="dirty working tree"):
            sb.run_baseline(contract_path=contract_path, output_dir=tmp_path / "run")
        assert not (tmp_path / "run").exists()

    def test_preflight_alone_touches_no_output_and_no_runner(self, tmp_path: Path):
        """Dry-run semantics: validation only, nothing executed or written."""
        contract_path = _write_contract(tmp_path)
        checked = sb.preflight(sb.load_contract(contract_path))
        assert len(checked.configs) == 2
        assert not (tmp_path / "run").exists()


# =============================================================================
# Fill authority (behavioral) — inactive cost_model must not change economics
# =============================================================================

class TestFillControlsEconomics:
    @staticmethod
    def _traded(result) -> pd.DataFrame:
        traded = result.trade_log[result.trade_log["included_in_portfolio"] == True]  # noqa: E712
        assert not traded.empty
        return traded.set_index(["ticker", "direction"]).sort_index()

    def test_cross_fill_costs_more_than_mid(self, synthetic_runner):
        mid = self._traded(synthetic_runner.run_single_config(
            _make_config(run_id="mid", fill=FillAssumption.mid())
        ))
        cross = self._traded(synthetic_runner.run_single_config(
            _make_config(run_id="cross", fill=FillAssumption.cross())
        ))
        assert (cross["spread_cost_per_share"] > 0).any()
        assert (mid["spread_cost_per_share"].abs() < 1e-9).all()
        assert not mid["pnl_per_share"].equals(cross["pnl_per_share"])
        assert (mid["fill_label"] == "mid").all()
        assert (cross["fill_label"] == "cross").all()

    @pytest.mark.parametrize("fill_factory", [FillAssumption.mid, FillAssumption.cross])
    @pytest.mark.parametrize("cost_model", ["half_spread_per_leg", "full_spread_per_leg"])
    def test_inactive_cost_model_does_not_change_economics(
        self, synthetic_runner, fill_factory, cost_model
    ):
        baseline = self._traded(synthetic_runner.run_single_config(
            _make_config(run_id="baseline", fill=fill_factory(), cost_model="mid")
        ))
        altered = self._traded(synthetic_runner.run_single_config(
            _make_config(run_id="altered", fill=fill_factory(), cost_model=cost_model)
        ))
        for column in (
            "entry_cost_per_share",
            "net_credit_per_share",
            "spread_cost_per_share",
            "pnl_per_share",
            "pnl_total",
            "capital_at_risk_dollars",
        ):
            pd.testing.assert_series_equal(
                baseline[column], altered[column], check_names=False
            )
