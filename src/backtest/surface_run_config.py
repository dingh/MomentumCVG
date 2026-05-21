
"""Configuration helpers for surface-based backtests and config search.

This module sits *above* `src/backtest/run_config.py`.

- `BacktestRunConfig` remains the single-run trading template.
- The classes here provide:
    * data-path resolution for the surface runner
    * feature-file inference from momentum / cvg / count column names
    * search-protocol settings for full-sample and walk-forward config search

The goal is to keep the runner lightweight and focused on:
    signal → structure assembly from surface → sizing → settle → score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence
import re

from src.backtest.run_config import BacktestRunConfig


# Matches patterns like: mom_42_8_mean, cvg_42_8, mom_42_8_count
# Captures the two numeric window bounds between underscores.
# Matches patterns like: mom_42_8_mean, cvg_42_8, mom_42_8_count
# Captures the two numeric window bounds between underscores.
_FEATURE_WINDOW_RE = re.compile(r"_(\d+)_(\d+)(?:_|$)")


def infer_feature_window(*column_names: str) -> tuple[int, int]:
    """
    Infer (max_lag, min_lag) from feature column names such as:
        mom_42_8_mean
        cvg_42_8
        mom_42_8_count

    Returns
    -------
    (max_lag, min_lag)

    Raises
    ------
    ValueError if no parsable column name is supplied.
    """
    for col in column_names:
        if not col:
            continue
        match = _FEATURE_WINDOW_RE.search(col)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(
        "Could not infer feature window from column names: "
        + ", ".join(repr(c) for c in column_names if c)
    )


def derive_cvg_and_count_cols(momentum_col: str) -> tuple[str, str]:
    """
    Convenience helper for common naming convention:
        mom_42_8_mean  -> cvg_42_8, mom_42_8_count
    """
    max_lag, min_lag = infer_feature_window(momentum_col)
    return f"cvg_{max_lag}_{min_lag}", f"mom_{max_lag}_{min_lag}_count"


@dataclass(frozen=True)
class SurfaceDataPaths:
    """
    Standardised path bundle for the surface runner.

    Defaults match the layout you described:
        C:\\MomentumCVG_env\\cache
            ├── features/
            │   └── features_<max>_<min>.parquet
            ├── ticker_liquidity_panel.parquet
            ├── option_surface_meta_weekly_2018_2026.parquet
            └── option_surface_quotes_weekly_2018_2026.parquet
    """
    cache_dir: Path = Path(r"C:\MomentumCVG_env\cache")
    features_dir: Optional[Path] = None
    liquidity_panel_path: Optional[Path] = None
    surface_meta_path: Optional[Path] = None
    surface_quotes_path: Optional[Path] = None
    earnings_path: Optional[Path] = None

    @property
    def resolved_features_dir(self) -> Path:
        return self.features_dir or (self.cache_dir / "features")

    @property
    def resolved_liquidity_panel_path(self) -> Path:
        return self.liquidity_panel_path or (self.cache_dir / "ticker_liquidity_panel.parquet")

    @property
    def resolved_surface_meta_path(self) -> Path:
        return self.surface_meta_path or (
            self.cache_dir / "option_surface_meta_weekly_2018_2026.parquet"
        )

    @property
    def resolved_surface_quotes_path(self) -> Path:
        return self.surface_quotes_path or (
            self.cache_dir / "option_surface_quotes_weekly_2018_2026.parquet"
        )

    def features_path_for_config(self, config: BacktestRunConfig) -> Path:
        # Infer the window from whichever column name contains the pattern.
        # All three column names should encode the same window; the first
        # successful parse wins.  The returned Path is used as a cache key
        # in SurfaceRunner._features_cache (Path equality is by value).
        max_lag, min_lag = infer_feature_window(
            config.momentum_col, config.cvg_col, config.count_col
        )
        return self.resolved_features_dir / f"features_{max_lag}_{min_lag}.parquet"


@dataclass(frozen=True)
class SurfaceRunnerSettings:
    """
    Runner-level settings that are NOT part of the single-run signal/structure template.

    These settings are about unit conversion and guardrails in the runner itself.
    """
    short_straddle_risk_multiplier: float = 2.0
    min_contracts: int = 1

    def __post_init__(self):
        if self.short_straddle_risk_multiplier <= 0:
            raise ValueError(
                "short_straddle_risk_multiplier must be > 0, "
                f"got {self.short_straddle_risk_multiplier}"
            )
        if self.min_contracts < 1:
            raise ValueError(f"min_contracts must be >= 1, got {self.min_contracts}")


@dataclass(frozen=True)
class SearchProtocolConfig:
    """
    Search protocol for ranking configurations.

    mode
    ----
    full_sample:
        Run every config on the full requested date range and compare summaries.

    walk_forward:
        Rolling protocol:
            train_window_dates   historical trade dates used to rank configs
            test_window_dates    next block used out-of-sample
            step_dates           how far to roll forward each iteration
    """
    mode: Literal["full_sample", "walk_forward"] = "full_sample"
    score_metric: str = "robust_score"

    train_window_dates: Optional[int] = None
    test_window_dates: Optional[int] = None
    step_dates: Optional[int] = None
    min_train_dates: int = 26

    def __post_init__(self):
        if self.mode not in ("full_sample", "walk_forward"):
            raise ValueError(f"Unsupported mode: {self.mode!r}")
        if self.min_train_dates < 1:
            raise ValueError(f"min_train_dates must be >= 1, got {self.min_train_dates}")

        if self.mode == "walk_forward":
            if self.train_window_dates is None or self.train_window_dates < 1:
                raise ValueError(
                    "train_window_dates must be set and >= 1 for walk_forward mode"
                )
            if self.test_window_dates is None or self.test_window_dates < 1:
                raise ValueError(
                    "test_window_dates must be set and >= 1 for walk_forward mode"
                )
            if self.step_dates is None:
                # Default: roll forward by one full test window (non-overlapping OOS blocks).
                # frozen=True prevents normal attribute assignment, so we bypass
                # the immutability guard via object.__setattr__ — a standard pattern
                # for conditional defaults in frozen dataclasses.
                object.__setattr__(self, "step_dates", self.test_window_dates)
            elif self.step_dates < 1:
                raise ValueError("step_dates must be >= 1 for walk_forward mode")


# NOTE: SurfaceSearchSpec is intentionally NOT frozen because Sequence[...] is
# mutable in practice (usually a list).  All other config dataclasses are frozen.
@dataclass
class SurfaceSearchSpec:
    """
    Bundles together the configs to test and the protocol used to rank them.
    """
    configs: Sequence[BacktestRunConfig]
    protocol: SearchProtocolConfig = field(default_factory=SearchProtocolConfig)

    def __post_init__(self):
        if not self.configs:
            raise ValueError("SurfaceSearchSpec.configs must not be empty")
