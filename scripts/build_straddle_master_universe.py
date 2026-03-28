from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional
import sys
import calendar

import numpy as np
import pandas as pd

# Make "src" importable when this script is run from repo root or copied nearby.
THIS_FILE = Path(__file__).resolve()
for candidate in [Path.cwd(), THIS_FILE.parent, THIS_FILE.parent.parent]:
    if (candidate / "src").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.data.orats_provider import ORATSDataProvider  # type: ignore


@dataclass(frozen=True)
class TargetDTEBucket:
    min_dte: int
    max_dte: int

    @property
    def center(self) -> float:
        return (self.min_dte + self.max_dte) / 2.0


DEFAULT_BUCKET = TargetDTEBucket(min_dte=5, max_dte=9)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def is_third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def list_available_trade_dates(
    data_root: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    every_nth_date: int = 1,
) -> list[date]:
    """Scan ORATS parquet files and return available trade dates.

    Expected file pattern from your provider:
      {data_root}/YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet
    """
    dates: list[date] = []
    for fp in sorted(data_root.glob("*/ORATS_SMV_Strikes_*.parquet")):
        try:
            date_str = fp.stem.split("_")[-1]
            d = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        dates.append(d)

    if every_nth_date > 1:
        dates = dates[::every_nth_date]
    return dates


def choose_target_expiry(expiries: Iterable[date], trade_date: date, bucket: TargetDTEBucket) -> Optional[date]:
    candidates = []
    for exp in expiries:
        dte = (exp - trade_date).days
        if bucket.min_dte <= dte <= bucket.max_dte:
            candidates.append((abs(dte - bucket.center), dte, exp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][2]


def _safe_mid(bid: float, ask: float) -> float:
    if pd.isna(bid) or pd.isna(ask):
        return float("nan")
    return (float(bid) + float(ask)) / 2.0


def _safe_spread_pct(bid: float, ask: float) -> float:
    mid = _safe_mid(bid, ask)
    if not np.isfinite(mid) or mid <= 0:
        return float("nan")
    return (float(ask) - float(bid)) / mid


def compute_atm_pair_metrics(df_ticker_expiry: pd.DataFrame) -> dict:
    """Compute ATM call/put pair metrics from a wide-format ORATS slice.

    Each row is one strike with both call and put fields.
    ATM is defined as the strike closest to adjusted spot price.
    """
    if df_ticker_expiry.empty:
        return {
            "has_valid_atm_pair": False,
            "atm_strike": np.nan,
            "pair_oi_min": np.nan,
            "pair_oi_sum": np.nan,
            "pair_volume_min": np.nan,
            "pair_volume_sum": np.nan,
            "pair_dollar_volume": np.nan,
            "straddle_mid": np.nan,
            "pair_spread_pct": np.nan,
            "call_spread_pct": np.nan,
            "put_spread_pct": np.nan,
        }

    work = df_ticker_expiry.copy()
    work["spot_distance"] = (work["adj_strike"] - work["adj_stkPx"]).abs()
    atm_row = work.sort_values(["spot_distance", "adj_strike"], ascending=[True, True]).iloc[0]

    c_bid = float(atm_row.get("adj_cBidPx", np.nan))
    c_ask = float(atm_row.get("adj_cAskPx", np.nan))
    p_bid = float(atm_row.get("adj_pBidPx", np.nan))
    p_ask = float(atm_row.get("adj_pAskPx", np.nan))
    c_mid = _safe_mid(c_bid, c_ask)
    p_mid = _safe_mid(p_bid, p_ask)

    call_spread_pct = _safe_spread_pct(c_bid, c_ask)
    put_spread_pct = _safe_spread_pct(p_bid, p_ask)

    has_valid_atm_pair = (
        np.isfinite(c_bid) and np.isfinite(c_ask) and np.isfinite(p_bid) and np.isfinite(p_ask)
        and c_bid > 0 and c_ask >= c_bid
        and p_bid > 0 and p_ask >= p_bid
    )

    c_vol = float(atm_row.get("cVolu", np.nan))
    p_vol = float(atm_row.get("pVolu", np.nan))
    c_oi = float(atm_row.get("cOi", np.nan))
    p_oi = float(atm_row.get("pOi", np.nan))

    return {
        "has_valid_atm_pair": bool(has_valid_atm_pair),
        "atm_strike": float(atm_row.get("adj_strike", np.nan)),
        "pair_oi_min": np.nanmin([c_oi, p_oi]),
        "pair_oi_sum": np.nansum([c_oi, p_oi]),
        "pair_volume_min": np.nanmin([c_vol, p_vol]),
        "pair_volume_sum": np.nansum([c_vol, p_vol]),
        "pair_dollar_volume": np.nansum([c_mid * c_vol, p_mid * p_vol]),
        "straddle_mid": np.nansum([c_mid, p_mid]),
        "pair_spread_pct": np.nanmax([call_spread_pct, put_spread_pct]),
        "call_spread_pct": call_spread_pct,
        "put_spread_pct": put_spread_pct,
    }


def scan_trade_date(
    provider: ORATSDataProvider,
    trade_date: date,
    bucket: TargetDTEBucket,
) -> pd.DataFrame:
    """Return one row per ticker for the given trade date."""
    df = provider._load_day_data(trade_date).copy()  # uses the provider's cached daily loader
    if df.empty:
        return pd.DataFrame()

    required_cols = {
        "ticker", "expirDate", "adj_strike", "adj_stkPx",
        "adj_cBidPx", "adj_cAskPx", "adj_pBidPx", "adj_pAskPx",
        "cVolu", "pVolu", "cOi", "pOi",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required ORATS columns for {trade_date}: {sorted(missing)}")

    df = df.dropna(subset=["ticker", "expirDate", "adj_strike", "adj_stkPx"]).copy()
    df["trade_date"] = pd.Timestamp(trade_date)
    df["expirDate"] = pd.to_datetime(df["expirDate"]).dt.date
    df["dte"] = (pd.to_datetime(df["expirDate"]) - pd.Timestamp(trade_date)).dt.days
    df = df[df["dte"] >= 0].copy()

    rows: list[dict] = []

    for ticker, g in df.groupby("ticker", sort=False):
        expiries = sorted(g["expirDate"].dropna().unique())
        target_expiry = choose_target_expiry(expiries, trade_date, bucket)

        row = {
            "trade_date": trade_date,
            "ticker": ticker,
            "n_expiries_total": int(len(expiries)),
            "n_unique_expiries_total": int(len(set(expiries))),
            "n_target_bucket_expiries": int(sum(bucket.min_dte <= (e - trade_date).days <= bucket.max_dte for e in expiries)),
            "has_target_expiry": target_expiry is not None,
            "target_expiry": target_expiry,
            "target_dte": (target_expiry - trade_date).days if target_expiry is not None else np.nan,
            "target_is_monthly": bool(is_third_friday(target_expiry)) if target_expiry is not None else False,
            "target_is_weekly": bool(target_expiry is not None and not is_third_friday(target_expiry)),
            "underlying_spot": float(g["adj_stkPx"].iloc[0]),
        }

        if target_expiry is not None:
            g_target = g[g["expirDate"] == target_expiry].copy()
            row.update(compute_atm_pair_metrics(g_target))
        else:
            row.update(compute_atm_pair_metrics(pd.DataFrame()))

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_tickers(
    daily_df: pd.DataFrame,
    min_pair_oi: int = 100,
    min_pair_volume: int = 10,
    max_pair_spread_pct: float = 0.50,
    top_n: int = 300,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    work = daily_df.copy()
    work["basic_pass"] = (
        work["has_valid_atm_pair"].fillna(False)
        & (work["pair_oi_min"].fillna(-1) >= min_pair_oi)
        & (work["pair_volume_min"].fillna(-1) >= min_pair_volume)
        & (work["pair_spread_pct"].fillna(np.inf) <= max_pair_spread_pct)
    )

    def q50(x: pd.Series) -> float:
        return float(x.median()) if len(x) else float("nan")

    grouped = work.groupby("ticker", sort=False)
    summary = grouped.agg(
        n_trade_dates_seen=("trade_date", "nunique"),
        presence_rate_target_dte=("has_target_expiry", "mean"),
        atm_pair_rate=("has_valid_atm_pair", "mean"),
        weekly_support_rate=("target_is_weekly", "mean"),
        monthly_support_rate=("target_is_monthly", "mean"),
        median_n_expiries_total=("n_expiries_total", q50),
        median_n_target_bucket_expiries=("n_target_bucket_expiries", q50),
        median_pair_oi_min=("pair_oi_min", q50),
        median_pair_volume_min=("pair_volume_min", q50),
        median_pair_dollar_volume=("pair_dollar_volume", q50),
        median_straddle_mid=("straddle_mid", q50),
        median_pair_spread_pct=("pair_spread_pct", q50),
        p75_pair_spread_pct=("pair_spread_pct", lambda s: float(s.quantile(0.75)) if len(s) else float("nan")),
        basic_pass_rate=("basic_pass", "mean"),
    ).reset_index()

    # Broad-universe score: focus on ATM pair tradability, not strict strategy eligibility.
    def rank01(s: pd.Series, ascending: bool = True) -> pd.Series:
        return s.rank(pct=True, ascending=ascending, method="average")

    summary["master_score"] = (
        0.30 * rank01(np.log1p(summary["median_pair_dollar_volume"].fillna(0.0)))
        + 0.25 * rank01(np.log1p(summary["median_pair_oi_min"].fillna(0.0)))
        + 0.20 * rank01(summary["basic_pass_rate"].fillna(0.0))
        + 0.15 * rank01(summary["atm_pair_rate"].fillna(0.0))
        + 0.10 * rank01(summary["weekly_support_rate"].fillna(0.0))
        - 0.10 * rank01(summary["median_pair_spread_pct"].fillna(np.inf), ascending=False)
    )

    summary = summary.sort_values(["master_score", "median_pair_dollar_volume"], ascending=[False, False]).reset_index(drop=True)
    summary["master_rank"] = np.arange(1, len(summary) + 1)
    summary["in_top_n"] = summary["master_rank"] <= top_n
    return summary


def build_universe(
    data_root: str,
    start_date: Optional[date],
    end_date: Optional[date],
    min_dte: int,
    max_dte: int,
    every_nth_date: int,
    min_pair_oi: int,
    min_pair_volume: int,
    max_pair_spread_pct: float,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    provider = ORATSDataProvider(data_root=data_root)
    bucket = TargetDTEBucket(min_dte=min_dte, max_dte=max_dte)

    trade_dates = list_available_trade_dates(
        data_root=Path(data_root),
        start_date=start_date,
        end_date=end_date,
        every_nth_date=every_nth_date,
    )
    if not trade_dates:
        raise ValueError("No trade dates found under the requested range.")

    daily_parts: list[pd.DataFrame] = []
    for i, d in enumerate(trade_dates, start=1):
        try:
            part = scan_trade_date(provider, d, bucket)
            if not part.empty:
                daily_parts.append(part)
        except FileNotFoundError:
            continue
        if i % 50 == 0:
            print(f"Scanned {i}/{len(trade_dates)} dates...", flush=True)

    daily_df = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    summary_df = summarize_tickers(
        daily_df=daily_df,
        min_pair_oi=min_pair_oi,
        min_pair_volume=min_pair_volume,
        max_pair_spread_pct=max_pair_spread_pct,
        top_n=top_n,
    )
    return daily_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broad ATM-straddle master universe from ORATS files.")
    parser.add_argument("--data-root", type=str, default="c:/ORATS/data/ORATS_Adjusted")
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument("--min-dte", type=int, default=5)
    parser.add_argument("--max-dte", type=int, default=9)
    parser.add_argument("--every-nth-date", type=int, default=1, help="Optional sampling knob. 1 = scan all available trade dates.")
    parser.add_argument("--min-pair-oi", type=int, default=100)
    parser.add_argument("--min-pair-volume", type=int, default=10)
    parser.add_argument("--max-pair-spread-pct", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="universe_output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    daily_df, summary_df = build_universe(
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        every_nth_date=args.every_nth_date,
        min_pair_oi=args.min_pair_oi,
        min_pair_volume=args.min_pair_volume,
        max_pair_spread_pct=args.max_pair_spread_pct,
        top_n=args.top_n,
    )

    daily_csv = outdir / "ticker_daily_expiry_profile.csv"
    summary_csv = outdir / "ticker_master_universe_summary.csv"
    tickers_txt = outdir / "master_universe_top_tickers.txt"

    daily_df.to_csv(daily_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_df.loc[summary_df["in_top_n"], ["ticker"]].to_csv(tickers_txt, index=False, header=False)

    print(f"Wrote daily profile: {daily_csv}")
    print(f"Wrote ticker summary: {summary_csv}")
    print(f"Wrote top tickers:   {tickers_txt}")
    if not summary_df.empty:
        print("\nTop 20 tickers by master_score:")
        print(summary_df[["master_rank", "ticker", "master_score", "median_pair_dollar_volume", "median_pair_oi_min", "basic_pass_rate"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
