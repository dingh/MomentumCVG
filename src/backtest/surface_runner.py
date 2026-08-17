"""Surface-first backtest runner.

High-level purpose
------------------
Run one *fixed* configuration over the precomputed option surface and emit a
flat trade log that is suitable for:

- comparing live-plausible configurations
- generating a weekly manual execution sheet later
- understanding which assumptions preserve alpha after structure / fill / sizing

This runner intentionally does NOT integrate with the legacy BacktestEngine.
It uses the surface directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.backtest.option_surface import OptionSurfaceDB
from src.backtest.run_config import BacktestRunConfig
from src.backtest.pipeline import (
    step1_get_universe,
    step2_score_signals,
    step3_get_eligible_structures,
    step4_apply_exclusions,
    step5_select_and_size,
    validate_feature_count_columns,
)
from src.backtest.surface_run_config import (
    SurfaceDataPaths,
    SurfaceRunnerSettings,
)
from src.backtest.surface_metrics import build_date_summary, summarize_trade_log

DATE_STATUS_COLUMNS = ["trade_date", "status", "reason"]


@dataclass
class SurfaceRunResult:
    config: BacktestRunConfig
    trade_log: pd.DataFrame
    date_summary: pd.DataFrame
    run_summary: Dict[str, object]
    date_status: pd.DataFrame


class SurfaceRunner:
    """
    Execute one BacktestRunConfig on the precomputed option surface.

    Thin S1→S8 orchestrator: universe → signals → structures → exclusions →
    ``pipeline.step5_select_and_size`` → ``surface_metrics`` date/run summaries.
    """

    def __init__(
        self,
        data_paths: SurfaceDataPaths = SurfaceDataPaths(),
        settings: SurfaceRunnerSettings = SurfaceRunnerSettings(),
    ):
        self.data_paths = data_paths
        self.settings = settings

        self.surface_db = OptionSurfaceDB.load(
            str(self.data_paths.resolved_surface_meta_path),
            str(self.data_paths.resolved_surface_quotes_path),
        )
        self.liquidity_panel = pd.read_parquet(self.data_paths.resolved_liquidity_panel_path)
        if "month_date" in self.liquidity_panel.columns:
            self.liquidity_panel["month_date"] = pd.to_datetime(self.liquidity_panel["month_date"])

        self.earnings = None
        if self.data_paths.earnings_path is not None and Path(self.data_paths.earnings_path).exists():
            self.earnings = pd.read_parquet(self.data_paths.earnings_path)
            if "earnings_date" in self.earnings.columns:
                self.earnings["earnings_date"] = pd.to_datetime(self.earnings["earnings_date"])

        self._features_cache: Dict[Path, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single_config(self, config: BacktestRunConfig) -> SurfaceRunResult:
        features = self._load_features_for_config(config)
        validate_feature_count_columns(features, config)

        expected_dates = self._get_expected_dates_from_a1(config)
        feature_dates = set(self._get_feature_dates(features, config))
        feature_dates_absent_from_a1 = sorted(feature_dates - set(expected_dates))

        trade_rows: List[Dict[str, object]] = []
        date_status_rows: List[Dict[str, object]] = []

        for trade_date in expected_dates:
            if trade_date not in feature_dates:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "failed",
                        "reason": "missing_features",
                    }
                )
                continue

            universe = self._step1_universe(trade_date, config)
            signals = self._step2_signals(trade_date, features, universe, config)

            if signals.empty:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "valid_no_trade",
                        "reason": "empty_signals",
                    }
                )
                continue

            structures = step3_get_eligible_structures(
                trade_date, signals, self.surface_db, config
            )
            structures = step4_apply_exclusions(structures, self.earnings, config)
            s5_out = step5_select_and_size(
                signals=signals,
                structures=structures,
                config=config,
            )
            if "_assembly" in s5_out.columns:
                s5_out = s5_out.drop(columns=["_assembly"])
            trade_rows.extend(s5_out.to_dict(orient="records"))

            included = False
            if not s5_out.empty and "included_in_portfolio" in s5_out.columns:
                included = bool((s5_out["included_in_portfolio"] == True).any())  # noqa: E712

            if included:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "traded",
                        "reason": None,
                    }
                )
            else:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "valid_no_trade",
                        "reason": "no_included_names",
                    }
                )

        trade_log = pd.DataFrame(trade_rows)
        if not trade_log.empty and "trade_date" in trade_log.columns:
            trade_log["trade_date"] = pd.to_datetime(trade_log["trade_date"]).dt.date
            trade_log = trade_log.sort_values(
                ["trade_date", "included_in_portfolio", "direction", "ticker"],
                ascending=[True, False, True, True],
            ).reset_index(drop=True)

        date_status = pd.DataFrame(date_status_rows, columns=DATE_STATUS_COLUMNS)
        if not date_status.empty:
            date_status["trade_date"] = pd.to_datetime(date_status["trade_date"]).dt.date
            date_status = date_status.sort_values("trade_date").reset_index(drop=True)
        self._assert_date_status_partition(expected_dates, date_status)

        date_summary = build_date_summary(trade_log)
        n_failed = int((date_status["status"] == "failed").sum()) if not date_status.empty else 0
        n_traded = int((date_status["status"] == "traded").sum()) if not date_status.empty else 0
        n_vnt = int((date_status["status"] == "valid_no_trade").sum()) if not date_status.empty else 0
        run_summary = {
            "run_id": config.run_id,
            "short_structure": config.short_structure,
            "momentum_col": config.momentum_col,
            "cvg_col": config.cvg_col,
            "fill_label": config.fill.label,
            **summarize_trade_log(trade_log),
            "n_expected_dates": len(expected_dates),
            "n_traded_dates": n_traded,
            "n_valid_no_trade_dates": n_vnt,
            "n_failed_dates": n_failed,
            "has_unresolved_failures": n_failed > 0,
            "n_feature_dates_absent_from_a1": len(feature_dates_absent_from_a1),
            "feature_dates_absent_from_a1": [
                d.isoformat() for d in feature_dates_absent_from_a1
            ],
        }
        return SurfaceRunResult(
            config=config,
            trade_log=trade_log,
            date_summary=date_summary,
            run_summary=run_summary,
            date_status=date_status,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_features_for_config(self, config: BacktestRunConfig) -> pd.DataFrame:
        path = self.data_paths.features_path_for_config(config)
        if path not in self._features_cache:
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            self._features_cache[path] = df
        return self._features_cache[path]

    def _get_expected_dates_from_a1(self, config: BacktestRunConfig) -> List[date]:
        """Sorted unique A1 entry_date values in [start, end], any surface_valid."""
        meta = self.surface_db.meta_df
        keys = meta["entry_date_key"]
        mask = (keys >= config.start_date) & (keys <= config.end_date)
        return sorted(keys.loc[mask].unique().tolist())

    @staticmethod
    def _get_feature_dates(features: pd.DataFrame, config: BacktestRunConfig) -> List[date]:
        mask = (
            (features["date"].dt.date >= config.start_date)
            & (features["date"].dt.date <= config.end_date)
        )
        return sorted(features.loc[mask, "date"].dt.date.unique().tolist())

    @staticmethod
    def _assert_date_status_partition(
        expected_dates: List[date],
        date_status: pd.DataFrame,
    ) -> None:
        expected_set = set(expected_dates)
        if date_status.empty:
            if expected_set:
                raise RuntimeError(
                    "date_status is empty but expected dates are non-empty"
                )
            return
        if list(date_status.columns) != DATE_STATUS_COLUMNS:
            raise RuntimeError(
                f"date_status columns must be {DATE_STATUS_COLUMNS}, "
                f"got {list(date_status.columns)}"
            )
        if date_status["trade_date"].duplicated().any():
            raise RuntimeError("date_status has duplicate trade_date values")
        statuses = set(date_status["status"].unique())
        allowed = {"traded", "valid_no_trade", "failed"}
        if not statuses.issubset(allowed):
            raise RuntimeError(f"date_status has unknown status values: {statuses - allowed}")
        observed = set(date_status["trade_date"].tolist())
        if observed != expected_set:
            raise RuntimeError(
                "date_status partition invariant failed: "
                f"missing={sorted(expected_set - observed)!r} "
                f"extra={sorted(observed - expected_set)!r}"
            )

    # ------------------------------------------------------------------
    # Step 1 / Step 2 wrappers
    # ------------------------------------------------------------------

    def _step1_universe(self, trade_date: date, config: BacktestRunConfig) -> pd.DataFrame:
        universe = step1_get_universe(trade_date, self.liquidity_panel, config)
        if universe is None:
            raise RuntimeError(
                f"step1_get_universe returned None for trade_date={trade_date}. "
                "Check that the liquidity panel covers this date."
            )
        return universe

    def _step2_signals(
        self,
        trade_date: date,
        features: pd.DataFrame,
        universe: pd.DataFrame,
        config: BacktestRunConfig,
    ) -> pd.DataFrame:
        signals = step2_score_signals(trade_date, features, universe, config)
        if signals is None:
            raise RuntimeError(
                f"step2_score_signals returned None for trade_date={trade_date}. "
                "Ensure the features DataFrame and universe are populated correctly."
            )
        return signals
