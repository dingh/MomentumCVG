
"""Config search utilities for surface-first backtests."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

import pandas as pd

from src.backtest.run_config import BacktestRunConfig
from src.backtest.surface_metrics import rank_run_summaries
from src.backtest.surface_run_config import SurfaceSearchSpec
from src.backtest.surface_runner import SurfaceRunner


@dataclass
class SurfaceSearchResult:
    ranked_summaries: pd.DataFrame
    trade_logs: Dict[str, pd.DataFrame]
    date_summaries: Dict[str, pd.DataFrame]


class SurfaceSearch:
    """
    Run many surface configs and rank them.

    Supports:
    - full-sample ranking
    - simple walk-forward config selection
    """

    def __init__(self, runner: SurfaceRunner):
        self.runner = runner

    def run(self, spec: SurfaceSearchSpec) -> SurfaceSearchResult:
        if spec.protocol.mode == "full_sample":
            return self._run_full_sample(spec)
        return self._run_walk_forward(spec)

    def _run_full_sample(self, spec: SurfaceSearchSpec) -> SurfaceSearchResult:
        summaries = []
        trade_logs = {}
        date_summaries = {}

        for config in spec.configs:
            result = self.runner.run_single_config(config)
            summaries.append(result.run_summary)
            trade_logs[config.run_id] = result.trade_log
            date_summaries[config.run_id] = result.date_summary

        ranked = rank_run_summaries(
            pd.DataFrame(summaries),
            metric=spec.protocol.score_metric,
        )
        return SurfaceSearchResult(
            ranked_summaries=ranked,
            trade_logs=trade_logs,
            date_summaries=date_summaries,
        )

    def _run_walk_forward(self, spec: SurfaceSearchSpec) -> SurfaceSearchResult:
        protocol = spec.protocol
        # These were validated in SearchProtocolConfig.__post_init__;
        # assert here to satisfy static type checkers.
        assert protocol.train_window_dates is not None
        assert protocol.test_window_dates is not None
        assert protocol.step_dates is not None

        if not spec.configs:
            return SurfaceSearchResult(
                ranked_summaries=pd.DataFrame(),
                trade_logs={},
                date_summaries={},
            )

        # ASSUMPTION: all configs in the spec share the same feature file
        # (i.e. the same max_lag / min_lag window).  The trade-date calendar is
        # derived from the first config only.  If you mix feature windows in one
        # spec, the OOS calendar may not cover the other configs correctly.
        base = spec.configs[0]
        features = self.runner._load_features_for_config(base)
        all_dates = self.runner._get_trade_dates(features, base)

        all_oos_trade_logs = []
        all_oos_summaries = []

        start_idx = protocol.train_window_dates
        segment = 0

        while start_idx + protocol.test_window_dates <= len(all_dates):
            train_dates = all_dates[start_idx - protocol.train_window_dates : start_idx]
            test_dates = all_dates[start_idx : start_idx + protocol.test_window_dates]

            if len(train_dates) < protocol.min_train_dates:
                start_idx += protocol.step_dates
                continue

            # Rank configs on the trailing train block.
            train_summaries = []
            for config in spec.configs:
                train_cfg = replace(
                    config,
                    start_date=train_dates[0],
                    end_date=train_dates[-1],
                    run_id=f"{config.run_id}__train_seg{segment}",
                )
                train_result = self.runner.run_single_config(train_cfg)
                summary = dict(train_result.run_summary)
                summary["base_run_id"] = config.run_id
                train_summaries.append(summary)

            ranked_train = rank_run_summaries(
                pd.DataFrame(train_summaries),
                metric=protocol.score_metric,
            )
            best_base_run_id = ranked_train.iloc[0]["base_run_id"]
            best_cfg = next(cfg for cfg in spec.configs if cfg.run_id == best_base_run_id)

            # Run the chosen config out of sample on the next block.
            test_cfg = replace(
                best_cfg,
                start_date=test_dates[0],
                end_date=test_dates[-1],
                run_id=f"{best_cfg.run_id}__oos_seg{segment}",
            )
            test_result = self.runner.run_single_config(test_cfg)
            trade_log = test_result.trade_log.copy()
            if not trade_log.empty:
                trade_log["segment"] = segment
                trade_log["selected_from_train_metric"] = protocol.score_metric
                trade_log["selected_base_run_id"] = best_base_run_id
                all_oos_trade_logs.append(trade_log)

            summary = dict(test_result.run_summary)
            summary["segment"] = segment
            summary["selected_base_run_id"] = best_base_run_id
            summary["train_metric"] = protocol.score_metric
            all_oos_summaries.append(summary)

            start_idx += protocol.step_dates
            segment += 1

        if all_oos_trade_logs:
            combined_trade_log = pd.concat(all_oos_trade_logs, ignore_index=True)
        else:
            combined_trade_log = pd.DataFrame()

        ranked = rank_run_summaries(
            pd.DataFrame(all_oos_summaries),
            metric=protocol.score_metric,
        ) if all_oos_summaries else pd.DataFrame()

        # All OOS segments are concatenated into a single trade log keyed
        # "walk_forward_oos".  Per-segment detail is preserved via the
        # "segment" column.  date_summaries is intentionally left empty here;
        # callers can build it from combined_trade_log using build_date_summary.
        return SurfaceSearchResult(
            ranked_summaries=ranked,
            trade_logs={"walk_forward_oos": combined_trade_log},
            date_summaries={},
        )
