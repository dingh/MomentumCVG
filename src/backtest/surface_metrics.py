
"""Metrics and scoring helpers for the surface runner."""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def build_date_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the flat trade log to one row per trade_date.

    Expected columns in trade_log:
        trade_date, included_in_portfolio, pnl_per_share, body_credit_per_share
    """
    if trade_log.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "n_candidates",
                "n_traded",
                "date_return_on_body_credit",
                "long_n_candidates",
                "short_n_candidates",
                "long_n_traded",
                "short_n_traded",
                "long_date_return_on_body_credit",
                "short_date_return_on_body_credit",
            ]
        )

    df = trade_log.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def _robc(subset: pd.DataFrame) -> float:
        """Mean pnl_per_share / body_credit_per_share for rows with body_credit > 0."""
        if subset.empty or "pnl_per_share" not in subset.columns or "body_credit_per_share" not in subset.columns:
            return np.nan
        valid = subset[subset["body_credit_per_share"] > 0]
        return float((valid["pnl_per_share"] / valid["body_credit_per_share"]).mean()) if not valid.empty else np.nan

    grouped = []
    for trade_date, g in df.groupby("trade_date", sort=True):
        traded       = g[g["included_in_portfolio"] == True]  # noqa: E712
        long_g       = g[g["direction"] == "long"]
        short_g      = g[g["direction"] == "short"]
        long_traded  = traded[traded["direction"] == "long"]
        short_traded = traded[traded["direction"] == "short"]
        grouped.append(
            {
                "trade_date": trade_date,
                "n_candidates": int(len(g)),
                "n_traded": int(len(traded)),
                "date_return_on_body_credit": _robc(traded),
                "long_n_candidates": int(len(long_g)),
                "short_n_candidates": int(len(short_g)),
                "long_n_traded": int(len(long_traded)),
                "short_n_traded": int(len(short_traded)),
                "long_date_return_on_body_credit": _robc(long_traded),
                "short_date_return_on_body_credit": _robc(short_traded),
            }
        )

    return pd.DataFrame(grouped).sort_values("trade_date").reset_index(drop=True)


def _compute_max_drawdown(returns: pd.Series) -> float:
    """
    Compute max drawdown on cumulative simple returns.
    """
    if returns.empty:
        return np.nan
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    running_peak = curve.cummax()
    drawdown = curve / running_peak - 1.0
    return float(drawdown.min())


def summarize_trade_log(trade_log: pd.DataFrame) -> Dict[str, float]:
    """
    Compute run-level metrics for config comparison.

    The summary is intentionally biased toward live relevance:
    - return on allocated risk
    - robustness to exclusions and structure availability
    - drawdown and hit rate
    """
    if trade_log.empty:
        return {
            "n_trade_dates": 0,
            "n_candidate_rows": 0,
            "n_traded_rows": 0,
            "availability_rate": 0.0,
            "avg_trades_per_date": 0.0,
            "mean_trade_return_on_body_credit": np.nan,
            "median_trade_return_on_body_credit": np.nan,
            "hit_rate": np.nan,
            "annualized_sharpe": np.nan,
            "max_drawdown": np.nan,
            "avg_spread_cost_ratio": np.nan,
            "avg_leg_spread_to_credit_ratio": np.nan,
            "robust_score": np.nan,
            "long_n_traded_rows": 0,
            "short_n_traded_rows": 0,
            "avg_long_return_on_body_credit": np.nan,
            "avg_short_return_on_body_credit": np.nan,
        }

    date_summary = build_date_summary(trade_log)
    traded = trade_log[trade_log["included_in_portfolio"] == True].copy()  # noqa: E712

    if "pnl_per_share" in traded.columns and "body_credit_per_share" in traded.columns:
        traded["_robc"] = np.where(
            traded["body_credit_per_share"] > 0,
            traded["pnl_per_share"] / traded["body_credit_per_share"],
            np.nan,
        )
    else:
        traded["_robc"] = np.nan

    mean_trade_return = float(traded["_robc"].mean()) if not traded.empty else np.nan
    median_trade_return = float(traded["_robc"].median()) if not traded.empty else np.nan
    hit_rate = float((traded["pnl_per_share"] > 0).mean()) if "pnl_per_share" in traded.columns and not traded.empty else np.nan

    long_n_traded_rows  = int(date_summary["long_n_traded"].sum())  if "long_n_traded"  in date_summary.columns else 0
    short_n_traded_rows = int(date_summary["short_n_traded"].sum()) if "short_n_traded" in date_summary.columns else 0
    avg_long_robc  = float(date_summary["long_date_return_on_body_credit"].dropna().mean())  if "long_date_return_on_body_credit"  in date_summary.columns else np.nan
    avg_short_robc = float(date_summary["short_date_return_on_body_credit"].dropna().mean()) if "short_date_return_on_body_credit" in date_summary.columns else np.nan

    n_candidate_rows = int(len(trade_log))
    n_traded_rows = int(len(traded))
    availability_rate = n_traded_rows / n_candidate_rows if n_candidate_rows else 0.0

    per_date_returns = date_summary["date_return_on_body_credit"].dropna()
    # Annualise assuming ~52 trade dates per year (i.e. weekly frequency).
    # If the actual trade-date density is lower (e.g. monthly), this will
    # overstate the Sharpe ratio — verify the frequency assumption for your data.
    if len(per_date_returns) >= 2 and per_date_returns.std(ddof=1) > 0:
        annualized_sharpe = float(
            per_date_returns.mean() / per_date_returns.std(ddof=1) * np.sqrt(52.0)
        )
    else:
        annualized_sharpe = np.nan

    avg_spread_cost_ratio = (
        float(traded["spread_cost_ratio"].mean())
        if "spread_cost_ratio" in traded and not traded["spread_cost_ratio"].dropna().empty
        else np.nan
    )
    avg_leg_spread_to_credit_ratio = (
        float(traded["leg_spread_to_credit_ratio"].mean())
        if "leg_spread_to_credit_ratio" in traded and not traded["leg_spread_to_credit_ratio"].dropna().empty
        else np.nan
    )
    max_drawdown = _compute_max_drawdown(per_date_returns)

    # Live-first ranking heuristic:
    #   robust_score = annualized_sharpe * availability_rate
    # This penalises configs that rarely produce a tradeable structure even
    # if their per-trade Sharpe looks good.  A config with Sharpe=2.0 but
    # only 50% availability scores the same as Sharpe=1.0 at 100% availability.
    # Limitation: does not account for convexity — a config with near-zero
    # availability can produce a high Sharpe on the few dates it does trade.
    robust_score = (
        annualized_sharpe * availability_rate
        if np.isfinite(annualized_sharpe)
        else np.nan
    )

    return {
        "n_trade_dates": int(len(date_summary)),
        "n_candidate_rows": n_candidate_rows,
        "n_traded_rows": n_traded_rows,
        "availability_rate": availability_rate,
        "avg_trades_per_date": float(n_traded_rows / len(date_summary)) if len(date_summary) else 0.0,
        "mean_trade_return_on_body_credit": mean_trade_return,
        "median_trade_return_on_body_credit": median_trade_return,
        "hit_rate": hit_rate,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "avg_spread_cost_ratio": avg_spread_cost_ratio,
        "avg_leg_spread_to_credit_ratio": avg_leg_spread_to_credit_ratio,
        "robust_score": robust_score,
        "long_n_traded_rows": long_n_traded_rows,
        "short_n_traded_rows": short_n_traded_rows,
        "avg_long_return_on_body_credit": avg_long_robc,
        "avg_short_return_on_body_credit": avg_short_robc,
    }


def rank_run_summaries(summary_df: pd.DataFrame, metric: str = "robust_score") -> pd.DataFrame:
    """
    Rank config summaries by a chosen metric (descending).
    """
    if summary_df.empty:
        return summary_df.copy()
    out = summary_df.copy()
    if metric not in out.columns:
        raise KeyError(f"Metric {metric!r} not found in summary_df columns")
    return out.sort_values(metric, ascending=False).reset_index(drop=True)
