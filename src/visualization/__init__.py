"""
Visualization module for ORATS option chain tradability diagnostics.

Provides interactive plotly-based plots for analyzing liquidity, spreads,
depth, and tradability of option chains — helping calibrate backtest filters.

Column mapping from ORATS parquet format to canonical plot-friendly names
is maintained here as the single source of truth.
"""

# Canonical column mapping: ORATS parquet column -> plot-friendly name
# Used by ChainLoader to rename at load time so all downstream code reads cleanly.
COLUMN_MAP: dict[str, str] = {
    # Price
    "adj_stkPx": "stockPrice",
    "adj_strike": "strike",
    # Call quotes
    "adj_cBidPx": "callBidPrice",
    "adj_cAskPx": "callAskPrice",
    # Put quotes
    "adj_pBidPx": "putBidPrice",
    "adj_pAskPx": "putAskPrice",
    # IV
    "smoothSmvVol": "smvVol",
    "cMidIv": "callMidIv",
    "pMidIv": "putMidIv",
    # Depth
    "cOi": "callOpenInterest",
    "pOi": "putOpenInterest",
    "cVolu": "callVolume",
    "pVolu": "putVolume",
    # Greeks (keep same names)
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    # Expiry
    "expirDate": "expirDate",
    # Ticker (pass-through)
    "ticker": "ticker",
}

# Inverse mapping for reference (canonical -> parquet)
COLUMN_MAP_INV: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}

from .chain_slice import ChainSlice
from .chain_loader import ChainLoader
from .chain_diagnostics import (
    plot_iv_surface,
    plot_spread_heatmap,
    plot_oi_volume,
    plot_smile,
    plot_term_structure,
    plot_theta_per_capital,
    plot_straddle_friction,
    plot_tradability_distribution,
    plot_tradability_timeseries,
    plot_signal_tradability,
    compute_atm_friction,
    compute_universe_friction,
)

__all__ = [
    "COLUMN_MAP",
    "COLUMN_MAP_INV",
    "ChainSlice",
    "ChainLoader",
    "plot_iv_surface",
    "plot_spread_heatmap",
    "plot_oi_volume",
    "plot_smile",
    "plot_term_structure",
    "plot_theta_per_capital",
    "plot_straddle_friction",
    "plot_tradability_distribution",
    "plot_tradability_timeseries",
    "plot_signal_tradability",
    "compute_atm_friction",
    "compute_universe_friction",
]
