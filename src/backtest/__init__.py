"""Backtesting engine

Exports (v2 — pre-computed pipeline):
    BacktestRunConfig  : configuration dataclass for one backtest run
    BacktestEngineV2   : orchestrator that calls the six-step pipeline
    pipeline           : six pure per-date step functions

Original engine (kept for reference, not used by v2):
    # BacktestEngine, BacktestConfig
"""

from src.backtest.run_config import BacktestRunConfig
from src.backtest.engine_v2 import BacktestEngineV2

__all__ = [
    'BacktestRunConfig',
    'BacktestEngineV2',
]

# Original engine — preserved, not exported until v2 is validated
# from src.backtest.engine import BacktestEngine, BacktestConfig

