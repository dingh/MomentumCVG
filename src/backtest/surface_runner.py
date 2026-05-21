
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
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.option_surface import (
    OptionSurfaceDB,
    StrategyAssemblyResult,
    build_ironcondor_from_surface,
    build_ironfly_from_surface,
    build_straddle_from_surface,
)
from src.backtest.run_config import BacktestRunConfig
from src.backtest.pipeline import step1_get_universe, step2_score_signals
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

    Design choices
    --------------
    - Step 1 universe and Step 2 signal ranking are reused from pipeline.py where possible.
    - Step 3 is surface-native: structures are assembled directly from OptionSurfaceDB.
    - Position sizing is equal max-loss per trade with INTEGER contract counts.
    - Cost/fill assumptions are embedded in structure assembly through FillAssumption.
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

            structures = self._build_structures_for_date(trade_date, signals, config)
            trade_rows.extend(
                self._select_size_and_settle(
                    trade_date=trade_date,
                    signals=signals,
                    structures=structures,
                    config=config,
                )
            )

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

    # ------------------------------------------------------------------
    # Step 3 — build structures directly from surface
    # ------------------------------------------------------------------

    def _build_structures_for_date(
        self,
        trade_date: date,
        signals: pd.DataFrame,
        config: BacktestRunConfig,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for rec in signals.to_dict(orient="records"):
            ticker = rec["ticker"]
            direction = rec["direction"]
            row = dict(rec)
            row.update(
                {
                    "trade_date": trade_date,
                    "instrument_type": None,
                    "structure_ok": False,
                    "failure_reason": None,
                    "had_earnings_nearby": False,
                }
            )

            try:
                meta = self.surface_db.get_metadata(ticker, trade_date)
                row["entry_spot"] = float(meta["entry_spot"])
                row["exit_spot"] = float(meta["exit_spot"])
                row["body_strike"] = float(meta["body_strike"])
                row["expiry_date"] = pd.Timestamp(meta["expiry_date"]).date()
                row["dte_actual"] = int(meta["dte_actual"])
            except Exception as exc:
                row["failure_reason"] = f"metadata_error:{exc}"
                rows.append(row)
                continue

            try:
                assembly = self._assemble_structure(ticker, trade_date, direction, config)
                # _assembly_to_row overwrites expiry_date/entry_spot/body_strike
                # with values from the assembly object (not from surface metadata).
                # The assembly is the authoritative source of truth for these fields.
                row.update(self._assembly_to_row(assembly))
                row["structure_ok"] = True
                # Store the live assembly object so _select_size_and_settle can
                # call .settle() on it later.  This private key is popped before
                # the row is returned to avoid leaking a non-serialisable object.
                row["_assembly"] = assembly
            except Exception as exc:
                row["failure_reason"] = str(exc)

            row["had_earnings_nearby"] = self._has_earnings_nearby(
                ticker=ticker,
                expiry_date=row.get("expiry_date"),
                exclusion_days=config.earnings_exclusion_days,
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def _assemble_structure(
        self,
        ticker: str,
        trade_date: date,
        direction: str,
        config: BacktestRunConfig,
    ) -> StrategyAssemblyResult:
        if direction == "long":
            return build_straddle_from_surface(
                surface_db=self.surface_db,
                ticker=ticker,
                entry_date=trade_date,
                direction="long",
                fill=config.fill,
                max_leg_spread_pct=config.max_leg_spread_pct,
            )

        if config.short_structure == "ironfly":
            return build_ironfly_from_surface(
                surface_db=self.surface_db,
                ticker=ticker,
                entry_date=trade_date,
                wing_target_delta=config.wing_delta_target,
                fill=config.fill,
                max_leg_spread_pct=config.max_leg_spread_pct,
                max_spread_cost_ratio=config.max_spread_cost_ratio,
            )
        elif config.short_structure == "ironcondor":
            return build_ironcondor_from_surface(
                surface_db=self.surface_db,
                ticker=ticker,
                entry_date=trade_date,
                short_delta_target=config.condor_short_delta_target,
                long_delta_target=config.condor_long_delta_target,
                fill=config.fill,
                max_leg_spread_pct=config.max_leg_spread_pct,
                max_spread_cost_ratio=config.max_spread_cost_ratio,
            )
        elif config.short_structure == "straddle":
            return build_straddle_from_surface(
                surface_db=self.surface_db,
                ticker=ticker,
                entry_date=trade_date,
                direction="short",
                fill=config.fill,
                max_leg_spread_pct=config.max_leg_spread_pct,
            )
        raise ValueError(f"Unsupported short_structure: {config.short_structure!r}")

    def _assembly_to_row(self, assembly: StrategyAssemblyResult) -> Dict[str, object]:
        d = {
            "instrument_type": assembly.strategy_name,
            "entry_cost_per_share": float(assembly.entry_cost),
            "entry_cost_mid_per_share": float(assembly.entry_cost_mid),
            "net_credit_per_share": float(assembly.net_credit),
            "max_loss_per_share": float(assembly.max_loss_per_share) if assembly.max_loss_per_share is not None else None,
            "spread_cost_per_share": float(assembly.spread_cost),
            "spread_cost_ratio": assembly.spread_cost_ratio,
            "total_leg_spread_per_share": float(assembly.total_leg_spread),
            "leg_spread_to_credit_ratio": assembly.leg_spread_to_credit_ratio,
            "strategy_net_delta": float(assembly.strategy.net_delta),
            "strategy_net_vega": float(assembly.strategy.net_vega),
            "strategy_net_gamma": float(assembly.strategy.net_gamma),
            "strategy_net_theta": float(assembly.strategy.net_theta),
            "theoretical_return_on_max_loss": assembly.return_on_max_loss,
            "expiry_date": assembly.expiry_date,
            "entry_spot": float(assembly.entry_spot),
            "body_strike": float(assembly.body_strike),
        }
        d.update(assembly.diagnostics)
        return d

    def _has_earnings_nearby(
        self,
        ticker: str,
        expiry_date: Optional[date],
        exclusion_days: int,
    ) -> bool:
        if self.earnings is None or expiry_date is None or exclusion_days <= 0:
            return False
        expiry_ts = pd.Timestamp(expiry_date)
        start_ts = expiry_ts - pd.Timedelta(days=exclusion_days)
        mask = (
            (self.earnings["ticker"] == ticker)
            & (self.earnings["earnings_date"] >= start_ts)
            & (self.earnings["earnings_date"] <= expiry_ts)
        )
        return bool(mask.any())

    # ------------------------------------------------------------------
    # Step 4 — select, size, settle
    # (labelled 5/6 originally; renaming kept for reference)
    # ------------------------------------------------------------------
    # Flow:
    #   1. Mark rows excluded due to missing structure or earnings.
    #   2. From eligible rows, select up to max_names_per_side per direction
    #      (ranked by signal_rank_pct).
    #   3. For each selected row: compute integer contract count from budget,
    #      call assembly.settle(exit_spot) to get realised P&L.
    #   4. When include_diagnostics=False, excluded rows are DROPPED entirely
    #      from the returned list (trade log only contains traded positions).
    # ------------------------------------------------------------------

    def _select_size_and_settle(
        self,
        trade_date: date,
        signals: pd.DataFrame,  # NOTE: not used directly; signal columns are already merged into structures
        structures: pd.DataFrame,
        config: BacktestRunConfig,
    ) -> List[Dict[str, object]]:
        if structures.empty:
            return []

        rows: List[Dict[str, object]] = []
        work = structures.copy()

        # exclusion priority 1: no structure
        work["included_in_portfolio"] = False
        work["exclusion_reason"] = None

        missing_mask = work["structure_ok"] != True  # noqa: E712
        work.loc[missing_mask, "exclusion_reason"] = "no_tradeable_structure"

        earnings_mask = (work["structure_ok"] == True) & (work["had_earnings_nearby"] == True)  # noqa: E712
        work.loc[earnings_mask, "exclusion_reason"] = "earnings_exclusion"

        eligible = work[
            (work["structure_ok"] == True)  # noqa: E712
            & (work["had_earnings_nearby"] == False)  # noqa: E712
        ].copy()

        selected_idx = []
        for direction, g in eligible.groupby("direction", sort=False):
            if direction == "long":
                g_sorted = g.sort_values("signal_rank_pct", ascending=False)
            else:
                g_sorted = g.sort_values("signal_rank_pct", ascending=True)
            selected_idx.extend(g_sorted.head(config.max_names_per_side).index.tolist())
            cap_excluded = g_sorted.iloc[config.max_names_per_side:]
            if not cap_excluded.empty:
                work.loc[cap_excluded.index, "exclusion_reason"] = "max_names_cap"

        if selected_idx:
            work.loc[selected_idx, "included_in_portfolio"] = True

        # size and settle included rows
        for idx, row in work.iterrows():
            out = row.to_dict()
            max_loss_per_share = out.get("max_loss_per_share")
            assembly = out.pop("_assembly", None)

            if not bool(out["included_in_portfolio"]):
                if config.include_diagnostics:
                    out.update({"pnl_per_share": None})
                    rows.append(out)
                continue

            max_loss_per_share = self._resolve_max_loss_per_share(out, max_loss_per_share)
            if max_loss_per_share is None or max_loss_per_share <= 0:
                out["included_in_portfolio"] = False
                out["exclusion_reason"] = "invalid_max_loss"
                if config.include_diagnostics:
                    rows.append(out)
                continue

            exit_spot = Decimal(str(out["exit_spot"]))
            position = assembly.settle(exit_spot=exit_spot)
            pnl_per_share = float(position.pnl) if position.pnl is not None else None

            out.update({"pnl_per_share": pnl_per_share})
            rows.append(out)

        return rows

    def _resolve_max_loss_per_share(
        self,
        row: Dict[str, object],
        max_loss_per_share: Optional[float],
    ) -> Optional[float]:
        if max_loss_per_share is not None:
            return float(max_loss_per_share)

        instrument_type = row.get("instrument_type")

        # short straddle has no bounded max loss; use a configurable risk proxy.
        # Default multiplier is 2.0x the net credit received, meaning the position
        # is treated as having a "max loss" of 2x the initial premium collected.
        # This is an approximation: tune short_straddle_risk_multiplier carefully.
        if instrument_type == "short_straddle":
            net_credit = max(float(row.get("net_credit_per_share", 0.0)), 0.0)
            return net_credit * self.settings.short_straddle_risk_multiplier

        return None
