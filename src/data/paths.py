"""Canonical on-disk paths for ORATS raw and adjusted data stores (Sprint 004 C5.11A / C6.1A)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ADJUSTED_LIQUID_ROOT = Path("C:/MomentumCVG_env/input/adjusted_liquid")
LEGACY_ORATS_ADJUSTED_ROOT = Path("C:/ORATS/data/ORATS_Adjusted")
RAW_ORATS_ROOT = Path("C:/ORATS/data/ORATS_Data")

# Stage A output cache (C6.1A)
DEFAULT_CACHE_ROOT = Path("C:/MomentumCVG_env/cache")
DEFAULT_SPOT_PRICES_PATH = DEFAULT_CACHE_ROOT / "spot_prices_adjusted.parquet"
DEFAULT_LIQUID_TICKERS_PATH = Path("C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv")

# Durable reference data (persistent across snapshot builds)
DEFAULT_REFERENCE_ROOT = Path("C:/MomentumCVG_env/reference")
DEFAULT_SECURITY_TYPES_PATH = DEFAULT_REFERENCE_ROOT / "orats_security_types.parquet"
DEFAULT_PRECOMPUTE_OPTION_SURFACE_LOG = Path(
    "C:/MomentumCVG_env/log/precompute_option_surface.log"
)
