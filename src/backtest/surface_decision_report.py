"""
Surface-path decision-report calculations (Sprint 006 D3 Commit 1).

Pure post-pass metrics over existing ``date_status`` / ``date_summary`` /
``trade_log`` tables. Does not select, price, size, or settle trades.
Does not build JSON/Markdown report files (Commit 3) or check leg logs
(Commit 2).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.backtest.surface_metrics import _compute_max_drawdown

FULL_HISTORY_START: date = date(2018, 10, 26)
FULL_HISTORY_END: date = date(2026, 7, 10)
PRIMARY_START: date = date(2020, 1, 1)
PRIMARY_END: date = date(2026, 7, 10)

_CAR_COL = "cycle_return_on_capital_at_risk"
_PNL_COL = "cycle_pnl_total"
_CAP_COL = "cycle_capital_at_risk"


class DecisionMetricsError(ValueError):
    """Raised when traded-date economics or no-trade inclusion preconditions fail."""


def _to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _ensure_trade_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        out = frame.copy() if frame is not None else pd.DataFrame()
        if "trade_date" not in out.columns:
            out = pd.DataFrame(columns=["trade_date"])
        return out
    out = frame.copy()
    out["trade_date"] = [_to_date(v) for v in out["trade_date"]]
    return out


def filter_to_window(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inclusive closed-window filter. Calendar authority is ``date_status``."""
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    log = _ensure_trade_date_column(trade_log)

    status = status[(status["trade_date"] >= start) & (status["trade_date"] <= end)].copy()
    status = status.sort_values("trade_date").reset_index(drop=True)

    expected = set(status["trade_date"].tolist())
    if not summary.empty:
        summary = summary[summary["trade_date"].isin(expected)].copy()
        summary = summary.sort_values("trade_date").reset_index(drop=True)
    if not log.empty:
        log = log[log["trade_date"].isin(expected)].copy()
        log = log.sort_values("trade_date").reset_index(drop=True)
    return status, summary, log


def count_date_classes(date_status: pd.DataFrame) -> dict[str, Any]:
    status = _ensure_trade_date_column(date_status)
    n_expected = int(len(status))
    n_traded = int((status["status"] == "traded").sum()) if n_expected else 0
    n_vnt = int((status["status"] == "valid_no_trade").sum()) if n_expected else 0
    n_failed = int((status["status"] == "failed").sum()) if n_expected else 0
    return {
        "n_expected_dates": n_expected,
        "n_traded_dates": n_traded,
        "n_valid_no_trade_dates": n_vnt,
        "n_failed_dates": n_failed,
        "result_complete": n_failed == 0,
        "has_unresolved_failures": n_failed > 0,
        "first_date": status["trade_date"].iloc[0] if n_expected else None,
        "last_date": status["trade_date"].iloc[-1] if n_expected else None,
    }


def assert_report_preconditions(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> None:
    """
    Abort on broken traded-date economics or included rows on valid_no_trade.

    Failed dates are not an evaluator defect; they only mark incompleteness.
    """
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    log = _ensure_trade_date_column(trade_log)

    if not summary.empty and "trade_date" in summary.columns:
        counts = summary.groupby("trade_date", sort=False).size()
    else:
        counts = pd.Series(dtype=int)

    traded = status.loc[status["status"] == "traded", "trade_date"]
    for trade_date in traded.tolist():
        n = int(counts[trade_date]) if trade_date in counts.index else 0
        if n == 0:
            raise DecisionMetricsError(
                f"traded date {trade_date} is missing a matching date_summary row"
            )
        if n != 1:
            raise DecisionMetricsError(
                f"traded date {trade_date} has {n} date_summary rows; expected exactly 1"
            )
        row = summary.loc[summary["trade_date"] == trade_date].iloc[0]
        pnl = float(pd.to_numeric(row.get(_PNL_COL), errors="coerce"))
        cap = float(pd.to_numeric(row.get(_CAP_COL), errors="coerce"))
        car = float(pd.to_numeric(row.get(_CAR_COL), errors="coerce"))
        if not np.isfinite(pnl):
            raise DecisionMetricsError(
                f"traded date {trade_date} has non-finite {_PNL_COL}={pnl!r}"
            )
        if not np.isfinite(cap) or not (cap > 0.0):
            raise DecisionMetricsError(
                f"traded date {trade_date} has non-positive or non-finite "
                f"{_CAP_COL}={cap!r}"
            )
        if not np.isfinite(car):
            raise DecisionMetricsError(
                f"traded date {trade_date} has non-finite {_CAR_COL}={car!r}"
            )

    if log.empty or "included_in_portfolio" not in log.columns:
        return

    included = log[log["included_in_portfolio"] == True]  # noqa: E712
    if included.empty:
        return
    vnt_dates = set(status.loc[status["status"] == "valid_no_trade", "trade_date"].tolist())
    bad = included[included["trade_date"].isin(vnt_dates)]
    if not bad.empty:
        trade_date = bad["trade_date"].iloc[0]
        raise DecisionMetricsError(
            f"valid_no_trade date {trade_date} has included_in_portfolio=True trade_log rows"
        )


def _annualized_sharpe(returns: Sequence[float] | pd.Series) -> float:
    series = pd.Series(list(returns), dtype=float)
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return float(series.mean() / std * np.sqrt(52.0))


def _compounded_return(returns: Sequence[float] | pd.Series) -> float:
    series = pd.Series(list(returns), dtype=float)
    if series.empty:
        return float("nan")
    return float(np.prod(1.0 + series.to_numpy(dtype=float)) - 1.0)


def _annualized_return(compounded: float, n_expected_dates: int) -> float:
    if n_expected_dates < 1 or not np.isfinite(compounded) or compounded <= -1.0:
        return float("nan")
    return float((1.0 + compounded) ** (52.0 / float(n_expected_dates)) - 1.0)


def _traded_summary_join(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> pd.DataFrame:
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    traded = status.loc[status["status"] == "traded", ["trade_date"]].copy()
    if traded.empty:
        return traded
    joined = traded.merge(summary, on="trade_date", how="left", validate="one_to_one")
    return joined.sort_values("trade_date").reset_index(drop=True)


def compute_view_a(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Conditional-on-traded View A. Excludes valid_no_trade; does not 0-fill them."""
    classes = count_date_classes(date_status)
    traded = _traded_summary_join(date_status, date_summary)
    if traded.empty:
        returns = pd.Series(dtype=float)
        mean_car = float("nan")
        sharpe = float("nan")
        drawdown = float("nan")
    else:
        returns = pd.to_numeric(traded[_CAR_COL], errors="coerce")
        mean_car = float(returns.mean())
        sharpe = _annualized_sharpe(returns)
        drawdown = _compute_max_drawdown(returns)
    return {
        "label": "conditional_on_traded",
        "n_expected_dates": classes["n_expected_dates"],
        "n_traded_dates": classes["n_traded_dates"],
        "n_valid_no_trade_dates": classes["n_valid_no_trade_dates"],
        "n_failed_dates": classes["n_failed_dates"],
        "mean_cycle_car": mean_car,
        "annualized_sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def compute_view_b(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calendar-aligned View B.

    ``valid_no_trade`` → 0. Failed dates are never zero-filled; when any failed
    date exists, ``complete`` is False and compounded metrics are omitted.
    """
    status = _ensure_trade_date_column(date_status).sort_values("trade_date")
    classes = count_date_classes(status)
    base = {
        "label": "calendar_aligned",
        "complete": classes["result_complete"],
        "n_expected_dates": classes["n_expected_dates"],
        "n_traded_dates": classes["n_traded_dates"],
        "n_valid_no_trade_dates": classes["n_valid_no_trade_dates"],
        "n_failed_dates": classes["n_failed_dates"],
        "first_date": classes["first_date"],
        "last_date": classes["last_date"],
    }
    if not classes["result_complete"]:
        base.update(
            {
                "compounded_return": None,
                "annualized_return": None,
                "annualized_sharpe": None,
                "max_drawdown": None,
            }
        )
        return base

    summary = _ensure_trade_date_column(date_summary)
    summary_map = {
        row["trade_date"]: float(row[_CAR_COL])
        for _, row in summary.iterrows()
    }
    returns: list[float] = []
    for _, row in status.iterrows():
        st = row["status"]
        if st == "traded":
            returns.append(float(summary_map[row["trade_date"]]))
        elif st == "valid_no_trade":
            returns.append(0.0)
        else:
            # Defensive: result_complete should already be False.
            base.update(
                {
                    "complete": False,
                    "compounded_return": None,
                    "annualized_return": None,
                    "annualized_sharpe": None,
                    "max_drawdown": None,
                }
            )
            return base

    series = pd.Series(returns, dtype=float)
    compounded = _compounded_return(series)
    base.update(
        {
            "compounded_return": compounded,
            "annualized_return": _annualized_return(
                compounded, classes["n_expected_dates"]
            ),
            "annualized_sharpe": _annualized_sharpe(series),
            "max_drawdown": _compute_max_drawdown(series),
        }
    )
    return base


def compute_weekly_outcomes(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> dict[str, Any]:
    classes = count_date_classes(date_status)
    traded = _traded_summary_join(date_status, date_summary)
    if traded.empty:
        win_rate = float("nan")
        profit_factor = float("nan")
    else:
        cars = pd.to_numeric(traded[_CAR_COL], errors="coerce")
        win_rate = float((cars > 0.0).mean())
        pnls = pd.to_numeric(traded[_PNL_COL], errors="coerce")
        pos = float(pnls[pnls > 0.0].sum())
        neg = float(pnls[pnls < 0.0].sum())
        if neg == 0.0 and pos > 0.0:
            profit_factor = float("inf")
        elif neg == 0.0 and pos == 0.0:
            profit_factor = float("nan")
        else:
            profit_factor = float(pos / abs(neg))

    n_expected = classes["n_expected_dates"]
    no_trade_frequency = (
        float(classes["n_valid_no_trade_dates"] / n_expected) if n_expected else float("nan")
    )
    return {
        "win_rate": win_rate,
        "no_trade_frequency": no_trade_frequency,
        "profit_factor": profit_factor,
    }


def compute_yearly_metrics(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    if status.empty:
        return []
    years = sorted({d.year for d in status["trade_date"].tolist()})
    out: list[dict[str, Any]] = []
    for year in years:
        year_status = status[status["trade_date"].map(lambda d: d.year) == year].copy()
        year_summary = summary[
            summary["trade_date"].isin(set(year_status["trade_date"].tolist()))
        ].copy()
        assert_report_preconditions(year_status, year_summary, pd.DataFrame())
        out.append(
            {
                "year": year,
                "date_class_counts": count_date_classes(year_status),
                "view_a": compute_view_a(year_status, year_summary),
                "view_b": compute_view_b(year_status, year_summary),
                "weekly": compute_weekly_outcomes(year_status, year_summary),
            }
        )
    return out


def compute_long_short_attribution(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> dict[str, Any]:
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    log = _ensure_trade_date_column(trade_log)
    traded_dates = set(status.loc[status["status"] == "traded", "trade_date"].tolist())

    included = pd.DataFrame()
    if not log.empty and "included_in_portfolio" in log.columns:
        included = log[
            (log["included_in_portfolio"] == True)  # noqa: E712
            & (log["trade_date"].isin(traded_dates))
        ].copy()

    def _side_from_log(direction: str) -> dict[str, Any]:
        side = included[included["direction"] == direction] if not included.empty else included
        n_rows = int(len(side))
        if n_rows == 0:
            return {
                "n_traded_rows": 0,
                "pnl_total": 0.0,
                "capital_at_risk_dollars": 0.0,
            }
        return {
            "n_traded_rows": n_rows,
            "pnl_total": float(pd.to_numeric(side["pnl_total"], errors="coerce").sum()),
            "capital_at_risk_dollars": float(
                pd.to_numeric(side["capital_at_risk_dollars"], errors="coerce").sum()
            ),
        }

    traded_summary = _traded_summary_join(status, summary)

    def _mean_side_cycle(col: str) -> float:
        if traded_summary.empty or col not in traded_summary.columns:
            return float("nan")
        values = pd.to_numeric(traded_summary[col], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.empty:
            return float("nan")
        return float(finite.mean())

    return {
        "long": {
            **_side_from_log("long"),
            "mean_cycle_return": _mean_side_cycle("long_cycle_return"),
        },
        "short": {
            **_side_from_log("short"),
            "mean_cycle_return": _mean_side_cycle("short_cycle_return"),
        },
    }


def _included_keys(trade_log: pd.DataFrame) -> set[tuple[date, str, str]]:
    log = _ensure_trade_date_column(trade_log)
    if log.empty or "included_in_portfolio" not in log.columns:
        return set()
    included = log[log["included_in_portfolio"] == True]  # noqa: E712
    keys: set[tuple[date, str, str]] = set()
    for _, row in included.iterrows():
        keys.add((row["trade_date"], str(row["ticker"]), str(row["direction"])))
    return keys


def _mean_included_ratio(trade_log: pd.DataFrame, column: str) -> float:
    log = _ensure_trade_date_column(trade_log)
    if log.empty or "included_in_portfolio" not in log.columns or column not in log.columns:
        return float("nan")
    included = log[log["included_in_portfolio"] == True]  # noqa: E712
    values = pd.to_numeric(included[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def compute_fill_assumption_sensitivity(
    *,
    cross_date_status: pd.DataFrame,
    cross_date_summary: pd.DataFrame,
    cross_trade_log: pd.DataFrame,
    mid_date_status: pd.DataFrame,
    mid_date_summary: pd.DataFrame,
    mid_trade_log: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, Any]:
    """
    Mid-versus-cross fill-assumption sensitivity (not pure transaction-cost).

    Unmatched dates and included candidates are disclosed; never silently dropped.
    """
    cross_s, cross_sum, cross_log = filter_to_window(
        cross_date_status, cross_date_summary, cross_trade_log, start, end
    )
    mid_s, mid_sum, mid_log = filter_to_window(
        mid_date_status, mid_date_summary, mid_trade_log, start, end
    )
    assert_report_preconditions(cross_s, cross_sum, cross_log)
    assert_report_preconditions(mid_s, mid_sum, mid_log)

    cross_traded = set(cross_s.loc[cross_s["status"] == "traded", "trade_date"].tolist())
    mid_traded = set(mid_s.loc[mid_s["status"] == "traded", "trade_date"].tolist())
    both = sorted(cross_traded & mid_traded)
    cross_only = sorted(cross_traded - mid_traded)
    mid_only = sorted(mid_traded - cross_traded)

    cross_map = {
        row["trade_date"]: row
        for _, row in _ensure_trade_date_column(cross_sum).iterrows()
    }
    mid_map = {
        row["trade_date"]: row
        for _, row in _ensure_trade_date_column(mid_sum).iterrows()
    }

    if both:
        car_deltas = [
            float(cross_map[d][_CAR_COL]) - float(mid_map[d][_CAR_COL]) for d in both
        ]
        pnl_deltas = [
            float(cross_map[d][_PNL_COL]) - float(mid_map[d][_PNL_COL]) for d in both
        ]
        mean_car_delta = float(np.mean(car_deltas))
        mean_pnl_delta = float(np.mean(pnl_deltas))
    else:
        mean_car_delta = float("nan")
        mean_pnl_delta = float("nan")

    cross_keys = _included_keys(cross_log)
    mid_keys = _included_keys(mid_log)
    both_keys = sorted(cross_keys & mid_keys)
    cross_only_keys = sorted(cross_keys - mid_keys)
    mid_only_keys = sorted(mid_keys - cross_keys)

    return {
        "label": "mid_versus_cross_fill_assumption_sensitivity",
        "n_dates_both_traded": len(both),
        "n_dates_cross_only": len(cross_only),
        "n_dates_mid_only": len(mid_only),
        "dates_both_traded": both,
        "dates_cross_only": cross_only,
        "dates_mid_only": mid_only,
        "mean_cross_minus_mid_car_both_traded": mean_car_delta,
        "mean_cross_minus_mid_pnl_both_traded": mean_pnl_delta,
        "mean_spread_cost_ratio_cross": _mean_included_ratio(cross_log, "spread_cost_ratio"),
        "mean_spread_cost_ratio_mid": _mean_included_ratio(mid_log, "spread_cost_ratio"),
        "mean_leg_spread_to_credit_ratio_cross": _mean_included_ratio(
            cross_log, "leg_spread_to_credit_ratio"
        ),
        "mean_leg_spread_to_credit_ratio_mid": _mean_included_ratio(
            mid_log, "leg_spread_to_credit_ratio"
        ),
        "n_candidates_both_included": len(both_keys),
        "n_candidates_cross_only": len(cross_only_keys),
        "n_candidates_mid_only": len(mid_only_keys),
        "candidates_cross_only": cross_only_keys,
        "candidates_mid_only": mid_only_keys,
    }


def compute_top5_abs_pnl_concentration(trade_log: pd.DataFrame) -> dict[str, Any]:
    log = _ensure_trade_date_column(trade_log)
    empty = {
        "top5": [],
        "top5_share_sum": 0.0,
        "total_abs_pnl": 0.0,
    }
    if log.empty or "included_in_portfolio" not in log.columns:
        return empty
    included = log[log["included_in_portfolio"] == True].copy()  # noqa: E712
    if included.empty:
        return empty
    included["abs_pnl"] = pd.to_numeric(included["pnl_total"], errors="coerce").abs()
    by_ticker = (
        included.groupby("ticker", sort=True)["abs_pnl"].sum().reset_index()
    )
    total = float(by_ticker["abs_pnl"].sum())
    if total <= 0.0 or not np.isfinite(total):
        return {
            "top5": [],
            "top5_share_sum": 0.0,
            "total_abs_pnl": total if np.isfinite(total) else 0.0,
        }
    by_ticker["share"] = by_ticker["abs_pnl"] / total
    by_ticker = by_ticker.sort_values(
        ["share", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    top = by_ticker.head(5)
    top5 = [
        {"ticker": str(row["ticker"]), "share": float(row["share"]), "abs_pnl": float(row["abs_pnl"])}
        for _, row in top.iterrows()
    ]
    share_sum = float(sum(item["share"] for item in top5))
    return {
        "top5": top5,
        "top5_share_sum": share_sum,
        "total_abs_pnl": total,
    }


def compute_activity(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
) -> dict[str, Any]:
    """
    Activity and turnover using accepted date semantics.

    Turnover is complete only when there are no failed dates. Failed dates are
    never entered as zero selected names.
    """
    status = _ensure_trade_date_column(date_status)
    summary = _ensure_trade_date_column(date_summary)
    classes = count_date_classes(status)
    traded = _traded_summary_join(status, summary)

    def _mean_traded_col(col: str) -> float:
        if traded.empty or col not in traded.columns:
            return float("nan")
        return float(pd.to_numeric(traded[col], errors="coerce").mean())

    avg_names = _mean_traded_col("n_traded")
    avg_long = _mean_traded_col("long_n_traded")
    avg_short = _mean_traded_col("short_n_traded")

    vnt_dates = set(status.loc[status["status"] == "valid_no_trade", "trade_date"].tolist())
    traded_map = {
        row["trade_date"]: float(row["n_traded"]) if "n_traded" in traded.columns else 0.0
        for _, row in traded.iterrows()
    }

    complete_values: list[float] = []
    diagnostic_values: list[float] = []
    for _, row in status.iterrows():
        st = row["status"]
        d = row["trade_date"]
        if st == "traded":
            value = float(traded_map.get(d, 0.0))
            complete_values.append(value)
            diagnostic_values.append(value)
        elif st == "valid_no_trade":
            complete_values.append(0.0)
            diagnostic_values.append(0.0)
        # failed: never contribute 0 to turnover

    if classes["result_complete"]:
        turnover_complete = True
        mean_turnover = (
            float(np.mean(complete_values)) if complete_values else float("nan")
        )
    else:
        turnover_complete = False
        mean_turnover = None

    diagnostic_mean = (
        float(np.mean(diagnostic_values)) if diagnostic_values else float("nan")
    )

    return {
        "avg_included_names_per_traded_date": avg_names,
        "avg_long_names_per_traded_date": avg_long,
        "avg_short_names_per_traded_date": avg_short,
        "turnover": {
            "complete": turnover_complete,
            "mean_included_names": mean_turnover,
            "diagnostic_mean_included_names_traded_and_vnt": diagnostic_mean,
            "n_failed_dates_excluded": classes["n_failed_dates"],
            "n_valid_no_trade_dates_as_zero": classes["n_valid_no_trade_dates"],
        },
        "date_class_counts": classes,
    }


def evaluate_fill_window(
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    """
    Run Commit-1 single-fill metrics for one closed reporting window.

    Validates traded-date preconditions, then returns plain dict metrics.
    Does not serialize a report or inspect leg/funnel artifacts.
    """
    status, summary, log = filter_to_window(
        date_status, date_summary, trade_log, start, end
    )
    assert_report_preconditions(status, summary, log)
    classes = count_date_classes(status)
    return {
        "window_start": start,
        "window_end": end,
        "date_class_counts": classes,
        "result_complete": classes["result_complete"],
        "has_unresolved_failures": classes["has_unresolved_failures"],
        "view_a": compute_view_a(status, summary),
        "view_b": compute_view_b(status, summary),
        "weekly": compute_weekly_outcomes(status, summary),
        "yearly": compute_yearly_metrics(status, summary),
        "long_short": compute_long_short_attribution(status, summary, log),
        "concentration": compute_top5_abs_pnl_concentration(log),
        "activity": compute_activity(status, summary),
    }
