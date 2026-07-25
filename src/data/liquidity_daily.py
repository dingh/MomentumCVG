"""Independent per-day liquidity calculations for process-pool execution.

This module intentionally contains no cross-day state.  Keeping the worker
entry point in an importable package module is required by Windows
``ProcessPoolExecutor`` spawn semantics.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_DTE_MIN = 5
DEFAULT_DTE_MAX = 60
LIQUIDITY_SOURCE = "raw_option_bid_x_volume_sum_dte_5_60"

RAW_REQUIRED_COLS = (
    "ticker",
    "expirDate",
    "stkPx",
    "strike",
    "cBidPx",
    "cAskPx",
    "pBidPx",
    "pAskPx",
    "cVolu",
    "pVolu",
)

DAILY_REQUIRED_COLS = (
    "trade_date",
    "ticker",
    "daily_atm_straddle_dollar_vol",
    "daily_atm_spread_pct",
    "daily_has_valid_quote",
    "n_candidate_expiries",
    "n_expiries_total",
    "no_expiry_in_band",
    "liquidity_source",
)


class LiquidityPanelError(Exception):
    """Blocking failure reading or calculating daily liquidity."""


def load_raw_day_from_zip_path(zip_path: Path | str) -> pd.DataFrame:
    """Load one wide-format ORATS chain from an exact frozen ZIP path."""
    path = Path(zip_path)
    if not path.is_file():
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(path, "r") as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.endswith((".csv", ".txt"))
            ]
            if not csv_names:
                raise LiquidityPanelError(f"No CSV/TXT inside {path.name}")
            with archive.open(csv_names[0]) as handle:
                return pd.read_csv(handle, dtype={"ticker": str})
    except LiquidityPanelError:
        raise
    except Exception as exc:
        raise LiquidityPanelError(f"Failed to read {path}: {exc}") from exc


def _valid_leg_quote(bid: float, ask: float) -> bool:
    return bool(
        np.isfinite(bid)
        and np.isfinite(ask)
        and bid > 0
        and ask > 0
        and ask >= bid
    )


def _leg_spread_pct(bid: float, ask: float) -> float:
    if not _valid_leg_quote(bid, ask):
        return float("nan")
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def validate_raw_columns(day_df: pd.DataFrame) -> None:
    missing = [column for column in RAW_REQUIRED_COLS if column not in day_df.columns]
    if missing:
        raise LiquidityPanelError(
            f"ORATS raw ZIP missing columns required for liquidity: {missing}. "
            "Expected native ORATS wide-format columns (stkPx, cBidPx, …); "
            "do not use adj_* or ORATS_Adjusted as input."
        )


def candidate_expiries(
    expiries: Sequence[date],
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> list[date]:
    return sorted(
        expiry
        for expiry in expiries
        if dte_min <= (expiry - trade_date).days <= dte_max
    )


def select_atm_row(expiry_frame: pd.DataFrame) -> pd.Series:
    """Select the raw strike closest to raw spot; lower strike breaks ties."""
    frame = expiry_frame.copy()
    frame["_dist"] = (frame["strike"] - frame["stkPx"]).abs()
    return frame.sort_values(["_dist", "strike"], kind="mergesort").iloc[0]


def compute_expiry_atm_liquidity(atm: pd.Series) -> tuple[float, float]:
    """Return ATM straddle dollar volume and worst-leg spread for one expiry."""
    call_bid = float(atm["cBidPx"])
    put_bid = float(atm["pBidPx"])
    call_ask = float(atm["cAskPx"])
    put_ask = float(atm["pAskPx"])
    call_volume = 0.0 if pd.isna(atm["cVolu"]) else float(atm["cVolu"])
    put_volume = 0.0 if pd.isna(atm["pVolu"]) else float(atm["pVolu"])

    if not (
        _valid_leg_quote(call_bid, call_ask)
        and _valid_leg_quote(put_bid, put_ask)
    ):
        return 0.0, float("nan")

    expiry_volume = min(
        100.0 * call_bid * call_volume,
        100.0 * put_bid * put_volume,
    )
    expiry_spread = float(
        max(
            _leg_spread_pct(call_bid, call_ask),
            _leg_spread_pct(put_bid, put_ask),
        )
    )
    return expiry_volume, expiry_spread


def compute_ticker_daily_observation(
    all_ticker_rows: pd.DataFrame,
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> dict:
    expiries = sorted(all_ticker_rows["expirDate"].dropna().unique())
    candidates = candidate_expiries(
        expiries, trade_date, dte_min=dte_min, dte_max=dte_max
    )
    if not candidates:
        return {
            "daily_atm_straddle_dollar_vol": 0.0,
            "daily_atm_spread_pct": np.nan,
            "daily_has_valid_quote": False,
            "n_candidate_expiries": 0,
            "n_expiries_total": len(expiries),
            "no_expiry_in_band": True,
            "liquidity_source": LIQUIDITY_SOURCE,
        }

    total_volume = 0.0
    spread_numerator = 0.0
    spread_denominator = 0.0
    for expiry in candidates:
        expiry_frame = all_ticker_rows[all_ticker_rows["expirDate"] == expiry]
        if expiry_frame.empty:
            continue
        expiry_volume, expiry_spread = compute_expiry_atm_liquidity(
            select_atm_row(expiry_frame)
        )
        total_volume += expiry_volume
        if expiry_volume > 0 and np.isfinite(expiry_spread):
            spread_numerator += expiry_spread * expiry_volume
            spread_denominator += expiry_volume

    daily_spread = (
        spread_numerator / spread_denominator
        if spread_denominator > 0
        else float("nan")
    )
    return {
        "daily_atm_straddle_dollar_vol": total_volume,
        "daily_atm_spread_pct": daily_spread,
        "daily_has_valid_quote": total_volume > 0 and np.isfinite(daily_spread),
        "n_candidate_expiries": len(candidates),
        "n_expiries_total": len(expiries),
        "no_expiry_in_band": False,
        "liquidity_source": LIQUIDITY_SOURCE,
    }


def compute_daily_liquidity_observations(
    day_df: pd.DataFrame,
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> pd.DataFrame:
    validate_raw_columns(day_df)
    frame = day_df.copy()
    frame["expirDate"] = pd.to_datetime(frame["expirDate"]).dt.date

    records: list[dict] = []
    for ticker, ticker_rows in frame.groupby("ticker", sort=False):
        observation = compute_ticker_daily_observation(
            ticker_rows,
            trade_date,
            dte_min=dte_min,
            dte_max=dte_max,
        )
        records.append(
            {"trade_date": trade_date, "ticker": ticker, **observation}
        )
    if not records:
        return pd.DataFrame(columns=list(DAILY_REQUIRED_COLS))
    return pd.DataFrame(records)


def process_daily_zip(
    zip_path: Path | str,
    trade_date: date,
    dte_min: int,
    dte_max: int,
) -> pd.DataFrame:
    """Process one exact ZIP path; importable process-pool worker entry point."""
    day_df = load_raw_day_from_zip_path(zip_path)
    if day_df.empty:
        return pd.DataFrame(columns=list(DAILY_REQUIRED_COLS))
    return compute_daily_liquidity_observations(
        day_df,
        trade_date,
        dte_min=dte_min,
        dte_max=dte_max,
    )
