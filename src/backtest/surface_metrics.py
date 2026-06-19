
"""
Metrics and scoring helpers for the surface runner.

Return metrics (two families — do not mix for go/no-go)
-------------------------------------------------------
**Primary (v1):** ``cycle_return_on_capital_at_risk`` and side splits
(``short_cycle_return``, ``long_cycle_return``).

    cycle_return = Σ pnl_total / Σ capital_at_risk_dollars

Dollar-weighted over included rows on a ``trade_date``. All included rows must
have finite ``pnl_total`` and ``capital_at_risk_dollars``; otherwise the cycle
return is ``NaN`` (sums may still report partial finite totals). Sharpe,
drawdown, and ``robust_score`` use this series. See
``docs/surface_engine_portfolio_metrics_design.md`` § Portfolio return.

**Legacy (interim):** ``date_return_on_body_credit`` and related
``*_return_on_body_credit`` columns.

    date_robc = mean( pnl_per_share / body_credit_per_share )

Equal-weight **per trade/name** on a date — *not* ``Σ pnl_total /
Σ (abs(quantity) × body_credit_per_share)``. Useful for relative config
search and signal-level diagnostics; **not** portfolio return and **not**
used for Sharpe/drawdown after Sprint 003 S8. Pre-dates S5 sizing columns;
kept for backward compatibility until body-credit search is retired.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def _safe_cycle_return(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when the denominator is unusable."""
    if denominator is None or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    if numerator is None or not np.isfinite(numerator):
        return np.nan
    return float(numerator / denominator)


_CYCLE_PNL_COL = "pnl_total"
_CYCLE_CAR_COL = "capital_at_risk_dollars"


def _cycle_economics_complete(subset: pd.DataFrame) -> bool:
    """
    True when ``subset`` is empty or every row has finite ``pnl_total`` and
    ``capital_at_risk_dollars``. Missing columns or any NaN/non-finite value
    on a non-empty subset → False.
    """
    if subset.empty:
        return True
    for col in (_CYCLE_PNL_COL, _CYCLE_CAR_COL):
        if col not in subset.columns:
            return False
        values = pd.to_numeric(subset[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            return False
    return True


def _sum_cycle_column(subset: pd.DataFrame, column: str) -> float:
    """Sum a cycle column for reporting; missing column or empty subset → 0."""
    if subset.empty or column not in subset.columns:
        return 0.0
    return float(pd.to_numeric(subset[column], errors="coerce").sum(skipna=True))


def _cycle_aggregate(subset: pd.DataFrame) -> tuple[float, float, float]:
    """
    Return (pnl_sum, capital_at_risk_sum, cycle_return) for included rows in ``subset``.

    Sums include finite values only (NaN skipped). ``cycle_return`` is NaN when
    any included row lacks valid cycle economics, or when the capital sum is
    not positive. Empty ``subset`` → sums 0, return NaN.
    """
    pnl_total = _sum_cycle_column(subset, _CYCLE_PNL_COL)
    capital_at_risk = _sum_cycle_column(subset, _CYCLE_CAR_COL)
    if not _cycle_economics_complete(subset):
        cycle_return = np.nan
    else:
        cycle_return = _safe_cycle_return(pnl_total, capital_at_risk)
    return pnl_total, capital_at_risk, cycle_return


def _cycle_side_metrics(traded: pd.DataFrame, direction: str) -> Dict[str, float]:
    """Aggregate cycle PnL, capital at risk, and return for one direction."""
    side = traded[traded["direction"] == direction]
    prefix = "short" if direction == "short" else "long"
    pnl_total, capital_at_risk, cycle_return = _cycle_aggregate(side)
    return {
        f"{prefix}_cycle_pnl_total": pnl_total,
        f"{prefix}_cycle_capital_at_risk": capital_at_risk,
        f"{prefix}_cycle_return": cycle_return,
    }


def _cycle_book_metrics(traded: pd.DataFrame) -> Dict[str, float]:
    """Aggregate whole-book cycle PnL, capital at risk, and return."""
    pnl_total, capital_at_risk, cycle_return = _cycle_aggregate(traded)
    return {
        "cycle_pnl_total": pnl_total,
        "cycle_capital_at_risk": capital_at_risk,
        "cycle_return_on_capital_at_risk": cycle_return,
    }


def _mean_return_on_body_credit(subset: pd.DataFrame) -> float:
    """
    Legacy interim return: equal-weight mean of per-trade body-credit ratios.

    Formula (included rows in ``subset`` only)::

        mean( pnl_per_share / body_credit_per_share )
        for rows where body_credit_per_share > 0

    This is **not** dollar-weighted portfolio return. It does **not** use
    ``pnl_total``, ``quantity``, or ``Σ / Σ``. A large position and a small
    position count equally. For portfolio aggregation use
    ``cycle_return_on_capital_at_risk`` (``Σ pnl_total / Σ capital_at_risk_dollars``).

    Rows with ``body_credit_per_share <= 0`` or missing are excluded from the mean.
    Empty ``subset`` or missing columns → ``NaN``.
    """
    if subset.empty or "pnl_per_share" not in subset.columns or "body_credit_per_share" not in subset.columns:
        return np.nan
    valid = subset[subset["body_credit_per_share"] > 0]
    return float((valid["pnl_per_share"] / valid["body_credit_per_share"]).mean()) if not valid.empty else np.nan


def build_date_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the flat trade log to one row per trade_date.

    Expected columns in trade_log:
        trade_date, included_in_portfolio, direction, pnl_per_share,
        body_credit_per_share, pnl_total, capital_at_risk_dollars

    Primary return (per ``trade_date``, included rows only)::

        cycle_return_on_capital_at_risk = Σ pnl_total / Σ capital_at_risk_dollars

    Every included row must have finite ``pnl_total`` and ``capital_at_risk_dollars``;
    otherwise the cycle return (book and affected side) is ``NaN``. Sums still report
    finite values where present (NaN rows skipped in the sum).

    Legacy body-credit columns (``date_return_on_body_credit``,
    ``long_date_return_on_body_credit``, ``short_date_return_on_body_credit``)
    use :func:`_mean_return_on_body_credit` — equal-weight per name, not ``Σ / Σ``.
    See module docstring.
    """
    base_columns = [
        "trade_date",
        "n_candidates",
        "n_traded",
        "cycle_pnl_total",
        "cycle_capital_at_risk",
        "cycle_return_on_capital_at_risk",
        "short_cycle_pnl_total",
        "short_cycle_capital_at_risk",
        "short_cycle_return",
        "long_cycle_pnl_total",
        "long_cycle_capital_at_risk",
        "long_cycle_return",
        "date_return_on_body_credit",
        "long_n_candidates",
        "short_n_candidates",
        "long_n_traded",
        "short_n_traded",
        "long_date_return_on_body_credit",
        "short_date_return_on_body_credit",
    ]
    if trade_log.empty:
        return pd.DataFrame(columns=base_columns)

    df = trade_log.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

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
                **_cycle_book_metrics(traded),
                **_cycle_side_metrics(traded, "short"),
                **_cycle_side_metrics(traded, "long"),
                "date_return_on_body_credit": _mean_return_on_body_credit(traded),
                "long_n_candidates": int(len(long_g)),
                "short_n_candidates": int(len(short_g)),
                "long_n_traded": int(len(long_traded)),
                "short_n_traded": int(len(short_traded)),
                "long_date_return_on_body_credit": _mean_return_on_body_credit(long_traded),
                "short_date_return_on_body_credit": _mean_return_on_body_credit(short_traded),
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

    Primary Sharpe / drawdown / robust_score use the per-date
    ``cycle_return_on_capital_at_risk`` series (``Σ / Σ`` dollar-weighted).

    Legacy body-credit fields (``mean_trade_return_on_body_credit``,
    ``median_trade_return_on_body_credit``, ``avg_*_return_on_body_credit``)
    are equal-weight per-trade or per-date means of
    ``pnl_per_share / body_credit_per_share`` — see
    :func:`_mean_return_on_body_credit`. Not used for Sharpe/drawdown.
    """
    empty_summary = {
        "n_trade_dates": 0,
        "n_candidate_rows": 0,
        "n_traded_rows": 0,
        "availability_rate": 0.0,
        "avg_trades_per_date": 0.0,
        "mean_cycle_return_on_capital_at_risk": np.nan,
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
        "avg_long_cycle_return": np.nan,
        "avg_short_cycle_return": np.nan,
        "avg_long_return_on_body_credit": np.nan,
        "avg_short_return_on_body_credit": np.nan,
    }
    if trade_log.empty:
        return empty_summary

    date_summary = build_date_summary(trade_log)
    traded = trade_log[trade_log["included_in_portfolio"] == True].copy()  # noqa: E712

    if "pnl_per_share" in traded.columns and "body_credit_per_share" in traded.columns:
        # Legacy: equal-weight mean per trade row (not Σ pnl_total / Σ body-credit dollars).
        traded["_robc"] = np.where(
            traded["body_credit_per_share"] > 0,
            traded["pnl_per_share"] / traded["body_credit_per_share"],
            np.nan,
        )
    else:
        traded["_robc"] = np.nan

    mean_trade_return = float(traded["_robc"].mean()) if not traded.empty else np.nan
    median_trade_return = float(traded["_robc"].median()) if not traded.empty else np.nan
    if "pnl_per_share" in traded.columns and not traded.empty:
        valid_pnl = traded["pnl_per_share"].notna()
        if not valid_pnl.any():
            hit_rate = np.nan
        else:
            hit_rate = float((traded.loc[valid_pnl, "pnl_per_share"] > 0).mean())
    else:
        hit_rate = np.nan

    long_n_traded_rows  = int(date_summary["long_n_traded"].sum())  if "long_n_traded"  in date_summary.columns else 0
    short_n_traded_rows = int(date_summary["short_n_traded"].sum()) if "short_n_traded" in date_summary.columns else 0
    avg_long_cycle_return = (
        float(date_summary["long_cycle_return"].dropna().mean())
        if "long_cycle_return" in date_summary.columns
        else np.nan
    )
    avg_short_cycle_return = (
        float(date_summary["short_cycle_return"].dropna().mean())
        if "short_cycle_return" in date_summary.columns
        else np.nan
    )
    avg_long_robc  = float(date_summary["long_date_return_on_body_credit"].dropna().mean())  if "long_date_return_on_body_credit"  in date_summary.columns else np.nan
    avg_short_robc = float(date_summary["short_date_return_on_body_credit"].dropna().mean()) if "short_date_return_on_body_credit" in date_summary.columns else np.nan

    n_candidate_rows = int(len(trade_log))
    n_traded_rows = int(len(traded))
    availability_rate = n_traded_rows / n_candidate_rows if n_candidate_rows else 0.0

    per_date_returns = date_summary["cycle_return_on_capital_at_risk"].dropna()
    mean_cycle_return = float(per_date_returns.mean()) if not per_date_returns.empty else np.nan
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
        "mean_cycle_return_on_capital_at_risk": mean_cycle_return,
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
        "avg_long_cycle_return": avg_long_cycle_return,
        "avg_short_cycle_return": avg_short_cycle_return,
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
