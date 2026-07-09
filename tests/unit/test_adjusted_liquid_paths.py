"""Unit tests for C5.11A adjusted-liquid path wiring."""

from __future__ import annotations

from pathlib import Path

import src.data.orats_provider as orats_provider_module
from src.backtest.config import DEFAULT_CONFIG
from src.data.orats_provider import ORATSDataProvider
from src.data.paths import (
    DEFAULT_ADJUSTED_LIQUID_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_LIQUID_TICKERS_PATH,
    DEFAULT_SPOT_PRICES_PATH,
    LEGACY_ORATS_ADJUSTED_ROOT,
    RAW_ORATS_ROOT,
)


def test_central_path_constants_exist() -> None:
    assert DEFAULT_ADJUSTED_LIQUID_ROOT == Path(
        "C:/MomentumCVG_env/input/adjusted_liquid"
    )
    assert LEGACY_ORATS_ADJUSTED_ROOT == Path("C:/ORATS/data/ORATS_Adjusted")
    assert RAW_ORATS_ROOT == Path("C:/ORATS/data/ORATS_Data")
    assert DEFAULT_CACHE_ROOT == Path("C:/MomentumCVG_env/cache")
    assert DEFAULT_SPOT_PRICES_PATH == DEFAULT_CACHE_ROOT / "spot_prices_adjusted.parquet"
    assert DEFAULT_LIQUID_TICKERS_PATH == Path(
        "C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv"
    )


def test_orats_provider_default_data_root() -> None:
    provider = ORATSDataProvider()
    assert provider.data_root == DEFAULT_ADJUSTED_LIQUID_ROOT
    assert provider.data_root != LEGACY_ORATS_ADJUSTED_ROOT


def test_backtest_default_config_data_root() -> None:
    data_root = DEFAULT_CONFIG["data_provider"]["params"]["data_root"]
    assert data_root == DEFAULT_ADJUSTED_LIQUID_ROOT.as_posix()


def test_active_defaults_import_central_paths_module() -> None:
    assert orats_provider_module.DEFAULT_ADJUSTED_LIQUID_ROOT is DEFAULT_ADJUSTED_LIQUID_ROOT
