
"""CLI entrypoint for surface-first config search.

Default mode is FULL-SAMPLE grid search.

Example
-------
python scripts/run_surface_search.py \
    --mode full_sample \
    --start-date 2018-01-01 \
    --end-date 2026-12-31 \
    --momentum-cols mom_42_8_mean \
    --fills mid,cross \
    --short-structures ironfly,ironcondor \
    --wing-deltas 0.10,0.15,0.20 \
    --condor-short-deltas 0.25,0.30 \
    --condor-long-deltas 0.10,0.15
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.option_surface import FillAssumption
from src.backtest.run_config import BacktestRunConfig
from src.backtest.surface_run_config import (
    SurfaceDataPaths,
    SurfaceRunnerSettings,
    SearchProtocolConfig,
    SurfaceSearchSpec,
    derive_cvg_and_count_cols,
)
from src.backtest.surface_runner import SurfaceRunner
from src.backtest.surface_search import SurfaceSearch


def _parse_date(raw: str) -> date:
    return pd.Timestamp(raw).date()


def _parse_list(raw: str, cast=str) -> List:
    if raw is None or raw == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_fill(label: str) -> FillAssumption:
    label = label.strip().lower()
    if label == "mid":
        return FillAssumption.mid()
    if label == "cross":
        return FillAssumption.cross()
    raise ValueError(f"Unsupported fill label: {label!r}")


def _make_run_id(
    momentum_col: str,
    short_structure: str,
    fill_label: str,
    wing_delta: float | None = None,
    condor_short: float | None = None,
    condor_long: float | None = None,
) -> str:
    base = f"{momentum_col}__{short_structure}__{fill_label}"
    if short_structure == "ironfly" and wing_delta is not None:
        return f"{base}__wing{wing_delta:.3f}"
    if short_structure == "ironcondor" and condor_short is not None and condor_long is not None:
        return f"{base}__short{condor_short:.3f}__long{condor_long:.3f}"
    return base


def build_configs_from_args(args) -> List[BacktestRunConfig]:
    fills = [_parse_fill(x) for x in _parse_list(args.fills, str)]
    short_structures = _parse_list(args.short_structures, str)
    momentum_cols = _parse_list(args.momentum_cols, str)

    wing_deltas = _parse_list(args.wing_deltas, float)
    condor_short_deltas = _parse_list(args.condor_short_deltas, float)
    condor_long_deltas = _parse_list(args.condor_long_deltas, float)

    configs: List[BacktestRunConfig] = []
    for momentum_col in momentum_cols:
        cvg_col, count_col = derive_cvg_and_count_cols(momentum_col)

        for fill in fills:
            for short_structure in short_structures:
                if short_structure == "ironfly":
                    for wing_delta in wing_deltas:
                        configs.append(
                            BacktestRunConfig(
                                run_id=_make_run_id(momentum_col, short_structure, fill.label, wing_delta=wing_delta),
                                momentum_col=momentum_col,
                                cvg_col=cvg_col,
                                count_col=count_col,
                                min_count_pct=args.min_count_pct,
                                long_top_pct=args.long_top_pct,
                                short_bottom_pct=args.short_bottom_pct,
                                cvg_filter_pct=args.cvg_filter_pct,
                                dvol_top_pct=args.dvol_top_pct,
                                spread_bottom_pct=args.spread_bottom_pct,
                                short_structure=short_structure,
                                wing_selection_rule="closest_delta",
                                wing_delta_target=wing_delta,
                                max_names_per_side=args.max_names_per_side,
                                max_loss_budget_per_trade=args.max_loss_budget_per_trade,
                                earnings_exclusion_days=args.earnings_exclusion_days,
                                cost_model="mid",  # legacy field kept valid; surface runner uses fill directly
                                start_date=_parse_date(args.start_date),
                                end_date=_parse_date(args.end_date),
                                fill=fill,
                                max_leg_spread_pct=args.max_leg_spread_pct,
                                max_spread_cost_ratio=args.max_spread_cost_ratio,
                                include_diagnostics=args.include_diagnostics,
                            )
                        )
                elif short_structure == "ironcondor":
                    for short_delta in condor_short_deltas:
                        for long_delta in condor_long_deltas:
                            configs.append(
                                BacktestRunConfig(
                                    run_id=_make_run_id(
                                        momentum_col, short_structure, fill.label,
                                        condor_short=short_delta, condor_long=long_delta
                                    ),
                                    momentum_col=momentum_col,
                                    cvg_col=cvg_col,
                                    count_col=count_col,
                                    min_count_pct=args.min_count_pct,
                                    long_top_pct=args.long_top_pct,
                                    short_bottom_pct=args.short_bottom_pct,
                                    cvg_filter_pct=args.cvg_filter_pct,
                                    dvol_top_pct=args.dvol_top_pct,
                                    spread_bottom_pct=args.spread_bottom_pct,
                                    short_structure=short_structure,
                                    wing_selection_rule="closest_delta",
                                    # BUG: wing_delta_target=0.15 is a dummy value for ironcondor.
                                    # The condor uses condor_short_delta_target / condor_long_delta_target
                                    # instead, but BacktestRunConfig requires this field regardless.
                                    wing_delta_target=0.15,
                                    max_names_per_side=args.max_names_per_side,
                                    max_loss_budget_per_trade=args.max_loss_budget_per_trade,
                                    earnings_exclusion_days=args.earnings_exclusion_days,
                                    # cost_model is a legacy field kept for schema compatibility;
                                    # the surface runner uses config.fill (FillAssumption) instead.
                                    cost_model="mid",
                                    start_date=_parse_date(args.start_date),
                                    end_date=_parse_date(args.end_date),
                                    fill=fill,
                                    max_leg_spread_pct=args.max_leg_spread_pct,
                                    max_spread_cost_ratio=args.max_spread_cost_ratio,
                                    condor_short_delta_target=short_delta,
                                    condor_long_delta_target=long_delta,
                                    include_diagnostics=args.include_diagnostics,
                                )
                            )
                elif short_structure == "straddle":
                    configs.append(
                        BacktestRunConfig(
                            run_id=_make_run_id(momentum_col, short_structure, fill.label),
                            momentum_col=momentum_col,
                            cvg_col=cvg_col,
                            count_col=count_col,
                            min_count_pct=args.min_count_pct,
                            long_top_pct=args.long_top_pct,
                            short_bottom_pct=args.short_bottom_pct,
                            cvg_filter_pct=args.cvg_filter_pct,
                            dvol_top_pct=args.dvol_top_pct,
                            spread_bottom_pct=args.spread_bottom_pct,
                            short_structure=short_structure,
                            wing_selection_rule="closest_delta",
                            wing_delta_target=0.15,
                            max_names_per_side=args.max_names_per_side,
                            max_loss_budget_per_trade=args.max_loss_budget_per_trade,
                            earnings_exclusion_days=args.earnings_exclusion_days,
                            cost_model="mid",
                            start_date=_parse_date(args.start_date),
                            end_date=_parse_date(args.end_date),
                            fill=fill,
                            max_leg_spread_pct=args.max_leg_spread_pct,
                            max_spread_cost_ratio=args.max_spread_cost_ratio,
                            include_diagnostics=args.include_diagnostics,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported short_structure: {short_structure!r}")
    return configs


def main():
    parser = argparse.ArgumentParser(description="Search live-plausible surface configs.")
    parser.add_argument("--cache-dir", type=str, default=r"C:\MomentumCVG_env\cache")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)

    parser.add_argument("--mode", type=str, default="full_sample", choices=["full_sample", "walk_forward"])
    parser.add_argument("--score-metric", type=str, default="robust_score")
    parser.add_argument("--train-window-dates", type=int, default=52)
    parser.add_argument("--test-window-dates", type=int, default=13)
    parser.add_argument("--step-dates", type=int, default=13)

    parser.add_argument("--momentum-cols", type=str, default="mom_42_8_mean")
    parser.add_argument("--fills", type=str, default="mid,cross")
    parser.add_argument("--short-structures", type=str, default="ironfly,ironcondor")

    parser.add_argument("--wing-deltas", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--condor-short-deltas", type=str, default="0.25,0.30,0.35")
    parser.add_argument("--condor-long-deltas", type=str, default="0.10,0.15")

    parser.add_argument("--min-count-pct", type=float, default=0.80)
    parser.add_argument("--long-top-pct", type=float, default=0.10)
    parser.add_argument("--short-bottom-pct", type=float, default=0.10)
    parser.add_argument("--cvg-filter-pct", type=float, default=0.50)

    parser.add_argument("--dvol-top-pct", type=float, default=0.20)
    parser.add_argument("--spread-bottom-pct", type=float, default=0.20)

    parser.add_argument("--max-names-per-side", type=int, default=3)
    parser.add_argument("--max-loss-budget-per-trade", type=float, default=500.0)
    parser.add_argument("--earnings-exclusion-days", type=int, default=5)

    parser.add_argument("--max-leg-spread-pct", type=float, default=0.50)
    parser.add_argument("--max-spread-cost-ratio", type=float, default=None)
    parser.add_argument("--include-diagnostics", action="store_true")

    parser.add_argument("--contract-multiplier", type=int, default=100)
    parser.add_argument("--short-straddle-risk-multiplier", type=float, default=2.0)

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir / "surface_search_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = SurfaceDataPaths(cache_dir=cache_dir, contract_multiplier=args.contract_multiplier)
    runner_settings = SurfaceRunnerSettings(
        short_straddle_risk_multiplier=args.short_straddle_risk_multiplier
    )

    runner = SurfaceRunner(data_paths=data_paths, settings=runner_settings)
    configs = build_configs_from_args(args)

    protocol = SearchProtocolConfig(
        mode=args.mode,
        score_metric=args.score_metric,
        train_window_dates=args.train_window_dates,
        test_window_dates=args.test_window_dates,
        step_dates=args.step_dates,
    )
    spec = SurfaceSearchSpec(configs=configs, protocol=protocol)

    search = SurfaceSearch(runner)
    result = search.run(spec)

    ranked_path = output_dir / f"ranked_summaries_{args.mode}.parquet"
    result.ranked_summaries.to_parquet(ranked_path, index=False)

    for run_id, trade_log in result.trade_logs.items():
        if trade_log is None or trade_log.empty:
            continue
        safe_run_id = run_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        trade_log.to_parquet(output_dir / f"trade_log_{safe_run_id}.parquet", index=False)

    for run_id, date_summary in result.date_summaries.items():
        if date_summary is None or date_summary.empty:
            continue
        safe_run_id = run_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        date_summary.to_parquet(output_dir / f"date_summary_{safe_run_id}.parquet", index=False)

    print(f"Wrote ranked summaries to: {ranked_path}")
    print(f"Configs tested: {len(configs)}")
    if not result.ranked_summaries.empty:
        print("Top 10 configs:")
        print(result.ranked_summaries.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
