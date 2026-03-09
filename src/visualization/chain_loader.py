"""
ChainLoader — loads raw (unfiltered) ORATS chain data and returns ChainSlice objects.

Uses ORATSDataProvider internally for cached parquet access, but does NOT apply
any liquidity filters.  The point is to let the user visualize the full chain
and decide where to set thresholds.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from ..data.orats_provider import ORATSDataProvider
from . import COLUMN_MAP
from .chain_slice import ChainSlice


class ChainLoader:
    """
    Load raw ORATS chain data and wrap it in a ``ChainSlice``.

    Parameters
    ----------
    data_root : str
        Path to ORATS_Adjusted folder.
    cache_size : int
        Number of daily files kept in the LRU cache.
    """

    def __init__(
        self,
        data_root: str = "c:/ORATS/data/ORATS_Adjusted",
        cache_size: int = 5,
    ) -> None:
        # We only need the raw file-loading capability of ORATSDataProvider.
        # Set filter thresholds to zero so _load_day_data gives everything.
        self._provider = ORATSDataProvider(
            data_root=data_root,
            min_volume=0,
            min_open_interest=0,
            min_bid=0.0,
            max_spread_pct=99.0,
            cache_size=cache_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename ORATS parquet columns to canonical names via COLUMN_MAP.

        Drop the unadjusted originals that would collide with their adj_*
        counterparts after renaming (e.g. raw 'strike' conflicts with
        adj_strike → strike).
        """
        # Columns whose unadjusted original must be dropped before renaming
        UNADJUSTED_TO_DROP = ["strike", "stkPx", "cBidPx", "cAskPx", "pBidPx", "pAskPx"]
        cols_to_drop = [c for c in UNADJUSTED_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        return df.rename(columns=rename)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_chain(self, ticker: str, trade_date: date) -> ChainSlice:
        """
        Load the full (unfiltered) chain for *ticker* on *trade_date*.

        Returns a ``ChainSlice`` with canonical column names and derived cols.
        """
        raw_df = self._provider._load_day_data(trade_date)

        # Filter to requested ticker
        df = raw_df[raw_df["ticker"] == ticker].copy()
        if df.empty:
            raise ValueError(
                f"No data for ticker={ticker!r} on {trade_date}. "
                f"Available tickers (sample): {sorted(raw_df['ticker'].unique()[:10])}"
            )

        df = self._rename_columns(df)
        return ChainSlice(ticker=ticker, trade_date=trade_date, df=df)

    def load_chain_multi(
        self,
        tickers: List[str],
        trade_date: date,
    ) -> Dict[str, ChainSlice]:
        """
        Load chains for multiple tickers on the same date (single file read).

        Returns dict ``{ticker: ChainSlice}``.  Tickers not found in the
        file are silently skipped.
        """
        raw_df = self._provider._load_day_data(trade_date)
        results: Dict[str, ChainSlice] = {}

        for tkr in tickers:
            sub = raw_df[raw_df["ticker"] == tkr].copy()
            if sub.empty:
                continue
            sub = self._rename_columns(sub)
            results[tkr] = ChainSlice(ticker=tkr, trade_date=trade_date, df=sub)

        return results

    def available_tickers(self, trade_date: date) -> List[str]:
        """Return sorted list of tickers present in the file for *trade_date*."""
        raw_df = self._provider._load_day_data(trade_date)
        return sorted(raw_df["ticker"].unique())
