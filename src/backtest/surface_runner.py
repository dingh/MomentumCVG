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

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest.option_surface import OptionSurfaceDB
from src.backtest.run_config import BacktestRunConfig
from src.backtest.pipeline import (
    eligible_feature_cross_section,
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

FUNNEL_SUMMARY_COLUMNS = [
    "run_id",
    "fill_label",
    "trade_date",
    "n_expected",
    "n_feature_covered",
    "n_universe",
    "n_jointly_eligible",
    "n_post_signal",
    "n_post_signal_long",
    "n_post_signal_short",
    "n_constructable",
    "n_constructable_long",
    "n_constructable_short",
    "n_included",
    "n_included_long",
    "n_included_short",
    "date_status",
    "date_reason",
]

LEG_LOG_COLUMNS = [
    "run_id",
    "fill_label",
    "trade_date",
    "ticker",
    "direction",
    "expiry_date",
    "option_type",
    "strike",
    "leg_index",
    "unit_quantity",
    "bid",
    "ask",
    "mid",
    "fill_price",
    "included_in_portfolio",
    "portfolio_quantity",
    "exit_spot",
    "expiry_payoff_per_unit",
    "entry_cash_per_unit",
    "pnl_per_unit",
    "pnl_total_leg",
]


def _empty_funnel_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FUNNEL_SUMMARY_COLUMNS)


def _empty_leg_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LEG_LOG_COLUMNS)


def _count_direction(frame: pd.DataFrame, direction: str, mask: Optional[pd.Series] = None) -> int:
    if frame is None or frame.empty or "direction" not in frame.columns:
        return 0
    side = frame["direction"] == direction
    if mask is not None:
        side = side & mask
    return int(side.sum())


def _structure_ok_mask(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "structure_ok" not in frame.columns:
        return pd.Series(dtype=bool)
    return frame["structure_ok"] == True  # noqa: E712


def _included_mask(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "included_in_portfolio" not in frame.columns:
        return pd.Series(dtype=bool)
    return frame["included_in_portfolio"] == True  # noqa: E712


def _funnel_row(
    *,
    config: BacktestRunConfig,
    trade_date: date,
    date_status: str,
    date_reason: Optional[str],
    n_feature_covered: int,
    n_universe: Optional[int],
    n_jointly_eligible: Optional[int],
    n_post_signal: Optional[int],
    n_post_signal_long: Optional[int],
    n_post_signal_short: Optional[int],
    n_constructable: Optional[int],
    n_constructable_long: Optional[int],
    n_constructable_short: Optional[int],
    n_included: Optional[int],
    n_included_long: Optional[int],
    n_included_short: Optional[int],
) -> Dict[str, Any]:
    return {
        "run_id": config.run_id,
        "fill_label": config.fill.label,
        "trade_date": trade_date,
        "n_expected": 1,
        "n_feature_covered": n_feature_covered,
        "n_universe": n_universe,
        "n_jointly_eligible": n_jointly_eligible,
        "n_post_signal": n_post_signal,
        "n_post_signal_long": n_post_signal_long,
        "n_post_signal_short": n_post_signal_short,
        "n_constructable": n_constructable,
        "n_constructable_long": n_constructable_long,
        "n_constructable_short": n_constructable_short,
        "n_included": n_included,
        "n_included_long": n_included_long,
        "n_included_short": n_included_short,
        "date_status": date_status,
        "date_reason": date_reason,
    }


def serialize_constructable_legs(
    s5_out: pd.DataFrame,
    config: BacktestRunConfig,
) -> List[Dict[str, Any]]:
    """Serialize unit legs for constructable S5 rows before ``_assembly`` is dropped."""
    if s5_out is None or s5_out.empty or "_assembly" not in s5_out.columns:
        return []

    fill = config.fill
    rows: List[Dict[str, Any]] = []
    for _, row in s5_out.iterrows():
        if row.get("structure_ok") != True:  # noqa: E712
            continue
        assembly = row.get("_assembly")
        if assembly is None or (isinstance(assembly, float) and pd.isna(assembly)):
            continue

        included = bool(row.get("included_in_portfolio") == True)  # noqa: E712
        quantity = row.get("quantity")
        try:
            qty_mag = abs(float(quantity)) if quantity is not None and not pd.isna(quantity) else None
        except (TypeError, ValueError):
            qty_mag = None
        if not included:
            qty_mag = None

        exit_spot_raw = row.get("exit_spot")
        try:
            exit_spot = (
                None
                if exit_spot_raw is None or pd.isna(exit_spot_raw)
                else Decimal(str(exit_spot_raw))
            )
        except (TypeError, ValueError):
            exit_spot = None

        trade_date = row.get("trade_date")
        ticker = row.get("ticker")
        direction = row.get("direction")
        expiry_date = assembly.expiry_date

        for leg_index, leg in enumerate(assembly.strategy.legs):
            unit_quantity = int(leg.quantity)
            quote = leg.option
            if unit_quantity > 0:
                fill_price = fill.buy_price(quote)
                entry_cash = fill_price * abs(unit_quantity)
            else:
                fill_price = fill.sell_price(quote)
                entry_cash = -fill_price * abs(unit_quantity)

            if exit_spot is None:
                expiry_payoff = None
            else:
                expiry_payoff = leg.calculate_intrinsic_value(exit_spot) * unit_quantity

            pnl_per_unit = (
                None if expiry_payoff is None else expiry_payoff - entry_cash
            )
            portfolio_quantity = (
                None if qty_mag is None else qty_mag * unit_quantity
            )
            pnl_total_leg = (
                None if qty_mag is None or pnl_per_unit is None else qty_mag * float(pnl_per_unit)
            )

            rows.append(
                {
                    "run_id": config.run_id,
                    "fill_label": fill.label,
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "direction": direction,
                    "expiry_date": expiry_date,
                    "option_type": str(quote.option_type),
                    "strike": float(quote.strike),
                    "leg_index": int(leg_index),
                    "unit_quantity": unit_quantity,
                    "bid": float(quote.bid),
                    "ask": float(quote.ask),
                    "mid": float(quote.mid),
                    "fill_price": float(fill_price),
                    "included_in_portfolio": included,
                    "portfolio_quantity": portfolio_quantity,
                    "exit_spot": None if exit_spot is None else float(exit_spot),
                    "expiry_payoff_per_unit": (
                        None if expiry_payoff is None else float(expiry_payoff)
                    ),
                    "entry_cash_per_unit": float(entry_cash),
                    "pnl_per_unit": None if pnl_per_unit is None else float(pnl_per_unit),
                    "pnl_total_leg": pnl_total_leg,
                }
            )
    return rows


@dataclass
class SurfaceRunResult:
    config: BacktestRunConfig
    trade_log: pd.DataFrame
    date_summary: pd.DataFrame
    run_summary: Dict[str, object]
    date_status: pd.DataFrame
    funnel_summary: pd.DataFrame = field(default_factory=_empty_funnel_summary)
    leg_log: pd.DataFrame = field(default_factory=_empty_leg_log)


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
        funnel_rows: List[Dict[str, object]] = []
        leg_rows: List[Dict[str, object]] = []

        for trade_date in expected_dates:
            if trade_date not in feature_dates:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "failed",
                        "reason": "missing_features",
                    }
                )
                funnel_rows.append(
                    _funnel_row(
                        config=config,
                        trade_date=trade_date,
                        date_status="failed",
                        date_reason="missing_features",
                        n_feature_covered=0,
                        n_universe=None,
                        n_jointly_eligible=None,
                        n_post_signal=None,
                        n_post_signal_long=None,
                        n_post_signal_short=None,
                        n_constructable=None,
                        n_constructable_long=None,
                        n_constructable_short=None,
                        n_included=None,
                        n_included_long=None,
                        n_included_short=None,
                    )
                )
                continue

            universe = self._step1_universe(trade_date, config)
            eligible = eligible_feature_cross_section(
                trade_date, features, universe, config
            )
            signals = self._step2_signals(trade_date, features, universe, config)
            n_universe = int(len(universe)) if universe is not None else 0
            n_jointly_eligible = int(len(eligible))

            if signals.empty:
                date_status_rows.append(
                    {
                        "trade_date": trade_date,
                        "status": "valid_no_trade",
                        "reason": "empty_signals",
                    }
                )
                funnel_rows.append(
                    _funnel_row(
                        config=config,
                        trade_date=trade_date,
                        date_status="valid_no_trade",
                        date_reason="empty_signals",
                        n_feature_covered=1,
                        n_universe=n_universe,
                        n_jointly_eligible=n_jointly_eligible,
                        n_post_signal=0,
                        n_post_signal_long=0,
                        n_post_signal_short=0,
                        n_constructable=0,
                        n_constructable_long=0,
                        n_constructable_short=0,
                        n_included=0,
                        n_included_long=0,
                        n_included_short=0,
                    )
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
            leg_rows.extend(serialize_constructable_legs(s5_out, config))
            if "_assembly" in s5_out.columns:
                s5_out = s5_out.drop(columns=["_assembly"])
            trade_rows.extend(s5_out.to_dict(orient="records"))

            included = False
            if not s5_out.empty and "included_in_portfolio" in s5_out.columns:
                included = bool((s5_out["included_in_portfolio"] == True).any())  # noqa: E712

            if included:
                status_label = "traded"
                reason_label: Optional[str] = None
            else:
                status_label = "valid_no_trade"
                reason_label = "no_included_names"
            date_status_rows.append(
                {
                    "trade_date": trade_date,
                    "status": status_label,
                    "reason": reason_label,
                }
            )

            ok_mask = _structure_ok_mask(s5_out)
            inc_mask = _included_mask(s5_out)
            funnel_rows.append(
                _funnel_row(
                    config=config,
                    trade_date=trade_date,
                    date_status=status_label,
                    date_reason=reason_label,
                    n_feature_covered=1,
                    n_universe=n_universe,
                    n_jointly_eligible=n_jointly_eligible,
                    n_post_signal=int(len(signals)),
                    n_post_signal_long=_count_direction(signals, "long"),
                    n_post_signal_short=_count_direction(signals, "short"),
                    n_constructable=int(ok_mask.sum()) if len(ok_mask) else 0,
                    n_constructable_long=_count_direction(s5_out, "long", ok_mask),
                    n_constructable_short=_count_direction(s5_out, "short", ok_mask),
                    n_included=int(inc_mask.sum()) if len(inc_mask) else 0,
                    n_included_long=_count_direction(s5_out, "long", inc_mask),
                    n_included_short=_count_direction(s5_out, "short", inc_mask),
                )
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
        funnel_summary = pd.DataFrame(funnel_rows, columns=FUNNEL_SUMMARY_COLUMNS)
        if not funnel_summary.empty:
            funnel_summary["trade_date"] = pd.to_datetime(
                funnel_summary["trade_date"]
            ).dt.date
            funnel_summary = funnel_summary.sort_values("trade_date").reset_index(
                drop=True
            )

        leg_log = pd.DataFrame(leg_rows, columns=LEG_LOG_COLUMNS)
        if not leg_log.empty:
            leg_log["trade_date"] = pd.to_datetime(leg_log["trade_date"]).dt.date
            if "expiry_date" in leg_log.columns:
                leg_log["expiry_date"] = pd.to_datetime(leg_log["expiry_date"]).dt.date
            leg_log = leg_log.sort_values(
                ["trade_date", "ticker", "direction", "leg_index"]
            ).reset_index(drop=True)

        return SurfaceRunResult(
            config=config,
            trade_log=trade_log,
            date_summary=date_summary,
            run_summary=run_summary,
            date_status=date_status,
            funnel_summary=funnel_summary,
            leg_log=leg_log,
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
