"""
Build the ticker liquidity panel used by step1_get_universe() in the backtest pipeline.

For each calendar month in the scan range, this script:
  1. Finds the first available trading day in the ORATS adjusted parquet store.
  2. Looks up the liquid monthly expiry for that month from liquid_expiry_dates.csv
     (the third-Friday expiry dates with 3 000+ tickers present).
  3. For every ticker on that trade date, computes ATM straddle liquidity metrics
     measured on the monthly expiry:
       - atm_straddle_dollar_vol  = min(call_vol, put_vol) × straddle_mid
       - atm_spread_pct           = max(call_spread_pct, put_spread_pct)
       - n_expiries_5_60dte       = structural depth (# weekly expiry dates in range)
  4. Writes one row per (month_date, ticker) to ticker_liquidity_panel.parquet.
  5. Derives a one-time liquid_tickers.csv snapshot: tickers that meet the
     top-dvol / bottom-spread thresholds in >= MIN_MONTHS_QUALIFIED months.
     (Used by precompute scripts that need a static ticker list.)

Pipeline contract (src/backtest/pipeline.py step1_get_universe):
    The panel is loaded once at engine init.  At each trade_date, step1 performs
    a point-in-time lookup (most recent month_date <= trade_date) and ranks
    tickers cross-sectionally on dollar_vol and effective_spread.
    Required columns consumed by step1:
        month_date, ticker, atm_straddle_dollar_vol, atm_spread_pct

Usage
-----
    # Default paths
    python scripts/build_liquidity_panel.py

    # Custom paths / date range
    python scripts/build_liquidity_panel.py \\
        --data-root  C:/ORATS/data/ORATS_Adjusted \\
        --cache-dir  C:/MomentumCVG_env/cache \\
        --start-year 2017 \\
        --end-year   2026 \\
        --dvol-top-pct    0.20 \\
        --spread-bot-pct  0.20 \\
        --min-months      10
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve project root so src/ imports work regardless of cwd ──────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent   # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                # MomentumCVG/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.orats_provider import ORATSDataProvider  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT      = Path("C:/ORATS/data/ORATS_Adjusted")
DEFAULT_CACHE_DIR      = Path("C:/MomentumCVG_env/cache")
DEFAULT_START_YEAR     = 2017
DEFAULT_END_YEAR       = 2026
DEFAULT_DVOL_TOP_PCT   = 0.20   # keep top 20% by ATM dollar volume
DEFAULT_SPREAD_BOT_PCT = 0.20   # keep bottom 20% by ATM spread pct

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ticker_liquidity_panel.parquet for the backtest pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root",      type=Path,  default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-dir",      type=Path,  default=DEFAULT_CACHE_DIR)
    p.add_argument("--start-year",     type=int,   default=DEFAULT_START_YEAR)
    p.add_argument("--end-year",       type=int,   default=DEFAULT_END_YEAR)
    p.add_argument(
        "--dvol-top-pct",
        type=float,
        default=DEFAULT_DVOL_TOP_PCT,
        help="Fraction used to compute the dvol threshold for liquid_tickers.csv (e.g. 0.20 = top 20%%).",
    )
    p.add_argument(
        "--spread-bot-pct",
        type=float,
        default=DEFAULT_SPREAD_BOT_PCT,
        help="Fraction used to compute the spread threshold for liquid_tickers.csv (e.g. 0.20 = bottom 20%%).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mid(bid: float, ask: float) -> float:
    """Return mid price, or NaN if either input is invalid."""
    if not (np.isfinite(bid) and np.isfinite(ask)):
        return float("nan")
    return (bid + ask) / 2.0


def _safe_spread_pct(bid: float, ask: float) -> float:
    """Return (ask - bid) / mid, or NaN if mid is non-positive."""
    mid = _safe_mid(bid, ask)
    if not (np.isfinite(mid) and mid > 0):
        return float("nan")
    return (ask - bid) / mid


def _compute_ticker_metrics(
    g_all: pd.DataFrame,
    target_expiry: date,
    trade_date: date,
) -> dict:
    """
    Compute ATM straddle liquidity metrics for one ticker on one trade_date.

    Parameters
    ----------
    g_all        : All rows for this ticker on trade_date (all expiries).
    target_expiry: The liquid monthly expiry to measure ATM liquidity on.
    trade_date   : The observation date (used for DTE and depth window).

    Returns
    -------
    Flat dict matching the panel schema.  All numeric values are Python floats
    (NaN where unavailable) so pd.DataFrame(records) produces the right dtypes.
    """
    # structural depth: count unique expiry dates within 5–60 DTE window
    n_expiries_5_60 = int(sum(
        5 <= (e - trade_date).days <= 60
        for e in g_all["expirDate"].dropna().unique()
    ))

    # slice to target expiry only
    g_exp = g_all[g_all["expirDate"] == target_expiry]
    if g_exp.empty:
        return {
            "has_valid_atm_pair":      False,
            "atm_strike":              np.nan,
            "underlying_spot":         float(g_all["adj_stkPx"].iloc[0]) if not g_all.empty else np.nan,
            "target_dte":              (target_expiry - trade_date).days,
            "n_expiries_5_60dte":      n_expiries_5_60,
            "atm_straddle_dollar_vol": np.nan,
            "atm_spread_pct":          np.nan,
            "call_spread_pct":         np.nan,
            "put_spread_pct":          np.nan,
            "straddle_mid":            np.nan,
            "pair_oi_min":             np.nan,
        }

    # find ATM strike: closest to spot; break ties by strike value
    g_exp = g_exp.copy()
    g_exp["_spot_dist"] = (g_exp["adj_strike"] - g_exp["adj_stkPx"]).abs()
    atm = g_exp.sort_values(["_spot_dist", "adj_strike"]).iloc[0]

    c_bid = float(atm.get("adj_cBidPx", np.nan))
    c_ask = float(atm.get("adj_cAskPx", np.nan))
    p_bid = float(atm.get("adj_pBidPx", np.nan))
    p_ask = float(atm.get("adj_pAskPx", np.nan))
    c_mid = _safe_mid(c_bid, c_ask)
    p_mid = _safe_mid(p_bid, p_ask)

    straddle_mid = (
        (c_mid if np.isfinite(c_mid) else 0.0)
        + (p_mid if np.isfinite(p_mid) else 0.0)
    )
    if straddle_mid <= 0:
        straddle_mid = np.nan

    call_sp = _safe_spread_pct(c_bid, c_ask)
    put_sp  = _safe_spread_pct(p_bid, p_ask)
    atm_sp  = (
        float(np.nanmax([call_sp, put_sp]))
        if any(np.isfinite(x) for x in [call_sp, put_sp])
        else np.nan
    )

    c_vol = float(atm.get("cVolu", np.nan))
    p_vol = float(atm.get("pVolu", np.nan))
    c_oi  = float(atm.get("cOi",   np.nan))
    p_oi  = float(atm.get("pOi",   np.nan))

    # conservative dollar volume: min(call_vol, put_vol) × straddle_mid
    min_vol = (
        float(np.nanmin([c_vol, p_vol]))
        if any(np.isfinite(x) for x in [c_vol, p_vol])
        else 0.0
    )
    atm_dollar_vol = (
        min_vol * straddle_mid
        if (np.isfinite(straddle_mid) and np.isfinite(min_vol))
        else np.nan
    )

    has_valid = bool(
        np.isfinite(c_bid) and c_bid > 0
        and np.isfinite(c_ask) and c_ask >= c_bid
        and np.isfinite(p_bid) and p_bid > 0
        and np.isfinite(p_ask) and p_ask >= p_bid
    )

    return {
        "has_valid_atm_pair":      has_valid,
        "atm_strike":              float(atm.get("adj_strike", np.nan)),
        "underlying_spot":         float(atm.get("adj_stkPx",  np.nan)),
        "target_dte":              (target_expiry - trade_date).days,
        "n_expiries_5_60dte":      n_expiries_5_60,
        "atm_straddle_dollar_vol": atm_dollar_vol,
        "atm_spread_pct":          atm_sp,
        "call_spread_pct":         call_sp,
        "put_spread_pct":          put_sp,
        "straddle_mid":            straddle_mid if np.isfinite(straddle_mid) else np.nan,
        "pair_oi_min":             (
            float(np.nanmin([c_oi, p_oi]))
            if any(np.isfinite(x) for x in [c_oi, p_oi])
            else np.nan
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(
    data_root: Path,
    cache_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Scan ORATS adjusted parquet files and build the liquidity panel DataFrame.

    Returns the full panel (all months × all tickers) before any filtering.
    Columns: month_date, ticker, expiry_date, target_dte, underlying_spot,
             atm_strike, atm_straddle_dollar_vol, atm_spread_pct,
             call_spread_pct, put_spread_pct, straddle_mid, pair_oi_min,
             n_expiries_5_60dte, has_valid_atm_pair
    """
    provider = ORATSDataProvider(data_root=str(data_root))

    # ── load liquid expiry dates ──────────────────────────────────────────────
    expiry_csv = cache_dir / "liquid_expiry_dates.csv"
    if not expiry_csv.exists():
        raise FileNotFoundError(
            f"liquid_expiry_dates.csv not found at {expiry_csv}.\n"
            "Run the expiry_coverage notebook first to generate it."
        )
    liquid_expiries = pd.read_csv(expiry_csv, parse_dates=["expirDate"])
    liquid_expiries["expirDate"] = liquid_expiries["expirDate"].dt.date
    month_to_expiry: dict[tuple[int, int], date] = {
        (row.expirDate.year, row.expirDate.month): row.expirDate
        for row in liquid_expiries.itertuples()
    }
    logger.info(
        "Loaded %d liquid expiry dates  (%s → %s)",
        len(liquid_expiries),
        liquid_expiries["expirDate"].min(),
        liquid_expiries["expirDate"].max(),
    )

    # ── discover all trading dates; keep first of each month ─────────────────
    all_dates: list[date] = []
    for fp in sorted(data_root.glob("*/ORATS_SMV_Strikes_*.parquet")):
        try:
            d = datetime.strptime(fp.stem.split("_")[-1], "%Y%m%d").date()
            if start_date <= d <= end_date:
                all_dates.append(d)
        except Exception:
            continue

    dates_df = pd.DataFrame({"trade_date": all_dates})
    dates_df["year_month"] = dates_df["trade_date"].apply(
        lambda d: (d.year, d.month)
    )
    first_of_month: list[date] = (
        dates_df.groupby("year_month")["trade_date"]
        .min()
        .reset_index(drop=True)
        .tolist()
    )

    # pair each first-of-month with its liquid expiry (drop months with no expiry)
    scannable: list[tuple[date, date]] = [
        (d, month_to_expiry[(d.year, d.month)])
        for d in first_of_month
        if (d.year, d.month) in month_to_expiry
    ]
    logger.info("Months to scan: %d", len(scannable))

    # ── main scan loop ────────────────────────────────────────────────────────
    records: list[dict] = []
    total = len(scannable)

    for i, (trade_date, expiry_date) in enumerate(scannable, start=1):
        try:
            df = provider._load_day_data(trade_date)
        except Exception as exc:
            logger.warning("[%d/%d] %s — LOAD ERROR: %s", i, total, trade_date, exc)
            continue

        if df is None or df.empty:
            logger.warning("[%d/%d] %s — empty file, skipping", i, total, trade_date)
            continue

        # normalize expirDate to Python date (parquet stores as datetime64)
        df = df.assign(expirDate=pd.to_datetime(df["expirDate"]).dt.date)

        for ticker, g_all in df.groupby("ticker", sort=False):
            m = _compute_ticker_metrics(g_all, expiry_date, trade_date)
            records.append({
                "month_date":              trade_date,
                "ticker":                  ticker,
                "expiry_date":             expiry_date,
                "target_dte":              m["target_dte"],
                "underlying_spot":         m["underlying_spot"],
                "atm_strike":              m["atm_strike"],
                "atm_straddle_dollar_vol": m["atm_straddle_dollar_vol"],
                "atm_spread_pct":          m["atm_spread_pct"],
                "call_spread_pct":         m["call_spread_pct"],
                "put_spread_pct":          m["put_spread_pct"],
                "straddle_mid":            m["straddle_mid"],
                "pair_oi_min":             m["pair_oi_min"],
                "n_expiries_5_60dte":      m["n_expiries_5_60dte"],
                "has_valid_atm_pair":      m["has_valid_atm_pair"],
            })

        if i % 12 == 0 or i == total:
            logger.info(
                "[%3d/%d] %s → expiry %s | rows: %d",
                i, total, trade_date, expiry_date, len(records),
            )

    panel = pd.DataFrame(records)
    logger.info("Scan complete. Panel shape: %s", panel.shape)
    return panel


def build_liquid_tickers(
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bot_pct: float,
) -> pd.DataFrame:
    """
    Derive the static liquid ticker list from the panel.

    A ticker qualifies in a given month if it simultaneously ranks in:
      - top    dvol_top_pct   by atm_straddle_dollar_vol
      - bottom spread_bot_pct by atm_spread_pct

    All tickers that qualified in at least one month are included.
    months_qualified is recorded for reference but no minimum threshold is applied.

    Returns a DataFrame [Ticker, months_qualified] sorted by Ticker.

    This is the input used by precompute scripts that need a static ticker
    universe (e.g. precompute_straddle_history.py, precompute_ironfly_history.py).
    The backtest pipeline does NOT use this file — it uses the full panel for
    point-in-time cross-sectional ranking at each trade date.
    """
    valid = panel[panel["has_valid_atm_pair"]].copy()
    qual_counts: dict[str, int] = {}

    for _, grp in valid.groupby("month_date"):
        # Require both metrics to be present — same population used for both thresholds,
        # consistent with step1_get_universe which ranks on tickers with BOTH values non-NaN.
        both_valid = grp[
            grp["atm_straddle_dollar_vol"].notna()
            & grp["atm_spread_pct"].notna()
        ]

        # skip months with too few tickers to rank meaningfully
        if len(both_valid) < 5:
            continue

        dvol_thresh = both_valid["atm_straddle_dollar_vol"].quantile(1.0 - dvol_top_pct)
        sp_thresh   = both_valid["atm_spread_pct"].quantile(spread_bot_pct)

        qualifiers = both_valid.loc[
            (both_valid["atm_straddle_dollar_vol"] >= dvol_thresh)
            & (both_valid["atm_spread_pct"] <= sp_thresh),
            "ticker",
        ]
        for t in qualifiers:
            qual_counts[t] = qual_counts.get(t, 0) + 1

    qual_df = (
        pd.Series(qual_counts, name="months_qualified")
        .sort_values(ascending=False)
        .rename_axis("ticker")
        .reset_index()
    )

    universe_df = (
        qual_df
        .rename(columns={"ticker": "Ticker"})
        [["Ticker", "months_qualified"]]
        .sort_values("Ticker")
        .reset_index(drop=True)
    )
    return universe_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    start_date = date(args.start_year, 1, 1)
    end_date   = date(args.end_year, 12, 31)

    logger.info(
        "Building liquidity panel: %s → %s  |  data_root=%s",
        start_date, end_date, args.data_root,
    )

    # ── build and save the panel ──────────────────────────────────────────────
    panel = build_panel(args.data_root, args.cache_dir, start_date, end_date)

    panel_path = args.cache_dir / "ticker_liquidity_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    logger.info("Saved panel → %s  (%d rows, %d tickers, %d months)",
                panel_path,
                len(panel),
                panel["ticker"].nunique(),
                panel["month_date"].nunique())

    # ── build and save the static liquid ticker list ──────────────────────────
    universe_df = build_liquid_tickers(
        panel,
        dvol_top_pct=args.dvol_top_pct,
        spread_bot_pct=args.spread_bot_pct,
    )

    universe_path = args.cache_dir / "liquid_tickers.csv"
    universe_df.to_csv(universe_path, index=False)
    logger.info(
        "Saved liquid tickers → %s  (%d tickers)",
        universe_path, len(universe_df),
    )

    # ── summary stats ─────────────────────────────────────────────────────────
    valid_pct = panel["has_valid_atm_pair"].mean()
    logger.info(
        "Panel summary: date range %s → %s  |  valid ATM pairs: %.1f%%",
        panel["month_date"].min(),
        panel["month_date"].max(),
        valid_pct * 100,
    )
    logger.info("Top 10 tickers by months qualified:\n%s",
                universe_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
