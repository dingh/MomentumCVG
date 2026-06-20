
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
)
from src.backtest.surface_run_config import (
    SurfaceDataPaths,
    SurfaceRunnerSettings,
)
from src.backtest.surface_metrics import build_date_summary, summarize_trade_log


@dataclass
class SurfaceRunResult:
    config: BacktestRunConfig
    trade_log: pd.DataFrame
    date_summary: pd.DataFrame
    run_summary: Dict[str, object]


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
        trade_dates = self._get_trade_dates(features, config)
        trade_rows: List[Dict[str, object]] = []

        for trade_date in trade_dates:
            universe = self._step1_universe(trade_date, config)
            signals = self._step2_signals(trade_date, features, universe, config)

            if signals.empty:
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

        trade_log = pd.DataFrame(trade_rows)
        if not trade_log.empty and "trade_date" in trade_log.columns:
            trade_log["trade_date"] = pd.to_datetime(trade_log["trade_date"]).dt.date
            trade_log = trade_log.sort_values(
                ["trade_date", "included_in_portfolio", "direction", "ticker"],
                ascending=[True, False, True, True],
            ).reset_index(drop=True)

        date_summary = build_date_summary(trade_log)
        run_summary = {
            "run_id": config.run_id,
            "short_structure": config.short_structure,
            "momentum_col": config.momentum_col,
            "cvg_col": config.cvg_col,
            "fill_label": config.fill.label,
            **summarize_trade_log(trade_log),
        }
        return SurfaceRunResult(
            config=config,
            trade_log=trade_log,
            date_summary=date_summary,
            run_summary=run_summary,
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

    @staticmethod
    def _get_trade_dates(features: pd.DataFrame, config: BacktestRunConfig) -> List[date]:
        mask = (
            (features["date"].dt.date >= config.start_date)
            & (features["date"].dt.date <= config.end_date)
        )
        dates = sorted(features.loc[mask, "date"].dt.date.unique().tolist())
        return dates

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
