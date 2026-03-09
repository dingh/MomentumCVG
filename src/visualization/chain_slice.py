"""
ChainSlice — preprocessed option chain data for a single (ticker, trade_date).

All derived columns (mid prices, spreads, spread%, moneyness, DTE) are
computed once at construction time and reused across all plot functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List

import numpy as np
import pandas as pd


EPS = 1e-9  # guard against division by zero in spread% calcs


@dataclass
class ChainSlice:
    """
    Preprocessed option chain for one (ticker, trade_date).

    The ``df`` DataFrame uses canonical column names (see ``COLUMN_MAP`` in
    ``__init__.py``) and is enriched with derived columns at construction time.

    Derived columns added by ``_enrich``:
        callMid, putMid        — (bid + ask) / 2
        callSpr, putSpr        — absolute bid-ask spread
        callSprPct, putSprPct  — spread / max(mid, eps)
        mny                    — strike / stockPrice
        logMny                 — ln(strike / stockPrice)
        dte                    — (expirDate - trade_date).days
    """

    ticker: str
    trade_date: date
    df: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        """Compute derived columns in place after init."""
        self._enrich()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich(self) -> None:
        """Add derived columns to *self.df* (mutates in place)."""
        # Reset index to avoid pandas alignment errors from duplicate labels
        # (common after filtering a larger DataFrame by ticker)
        self.df.reset_index(drop=True, inplace=True)
        df = self.df

        # Extract as numpy arrays to avoid pandas column-alignment errors
        # (ORATS parquet files can have duplicate column names which break
        #  Series / DataFrame arithmetic via pandas' align machinery)
        call_bid = df["callBidPrice"].values.astype(float)
        call_ask = df["callAskPrice"].values.astype(float)
        put_bid  = df["putBidPrice"].values.astype(float)
        put_ask  = df["putAskPrice"].values.astype(float)
        strike   = df["strike"].values.astype(float)
        stock_px = df["stockPrice"].values.astype(float)

        # Mid prices
        call_mid = 0.5 * (call_bid + call_ask)
        put_mid  = 0.5 * (put_bid  + put_ask)
        df["callMid"] = call_mid
        df["putMid"]  = put_mid

        # Absolute spreads
        df["callSpr"] = call_ask - call_bid
        df["putSpr"]  = put_ask  - put_bid

        # Percent spreads
        df["callSprPct"] = (call_ask - call_bid) / np.maximum(call_mid, EPS)
        df["putSprPct"]  = (put_ask  - put_bid)  / np.maximum(put_mid,  EPS)

        # Moneyness
        spot_safe = np.maximum(stock_px, EPS)
        df["mny"]    = strike / spot_safe
        df["logMny"] = np.log(strike / spot_safe)

        # DTE (days to each expiry from trade_date)
        df["expirDate"] = pd.to_datetime(df["expirDate"])
        td = pd.Timestamp(self.trade_date)
        df["dte"] = (df["expirDate"] - td).dt.days

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def available_expiries(self) -> List[date]:
        """Sorted list of unique expiry dates in this chain."""
        expiries = self.df["expirDate"].dropna().dt.date.unique()
        return sorted(expiries)

    def filter_expiry(self, expiry_date: date) -> pd.DataFrame:
        """Return rows matching *expiry_date*."""
        mask = self.df["expirDate"].dt.date == expiry_date
        return self.df[mask].copy()

    def spot_price(self) -> float:
        """Representative spot price (first non-NaN stockPrice)."""
        return float(self.df["stockPrice"].dropna().iloc[0])

    def summary(self) -> dict:
        """Quick overview dict for display / logging."""
        expiries = self.available_expiries()
        dtes = self.df["dte"].dropna()
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "spot_price": self.spot_price(),
            "total_rows": len(self.df),
            "num_expiries": len(expiries),
            "dte_range": (int(dtes.min()), int(dtes.max())) if len(dtes) else None,
            "expiry_dates": expiries,
        }
