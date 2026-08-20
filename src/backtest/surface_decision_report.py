"""
Surface-path decision-report calculations (Sprint 006 D3).

Pure post-pass metrics over existing ``date_status`` / ``date_summary`` /
``trade_log`` / funnel / leg-log tables. Does not select, price, size, or
settle trades. Commit 3 adds candidate-view derivation and deterministic
JSON/Markdown report assembly.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence

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


_STRADDLE_INSTRUMENTS = {"long_straddle", "short_straddle"}
_IRON_FLY_INSTRUMENTS = {"iron_fly"}
_LEG_REL_TOL = 1e-8
_LEG_ABS_TOL = 1e-6


def _values_close(left: float, right: float) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    return abs(left - right) <= max(_LEG_ABS_TOL, _LEG_REL_TOL * max(abs(left), abs(right)))


def _finite_leg_numeric_series(
    matched: pd.DataFrame,
    *,
    trade_date: date,
    ticker: str,
    direction: str,
    column: str,
) -> pd.Series:
    values = pd.to_numeric(matched[column], errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype=float))
    if not bool(finite_mask.all()):
        raise DecisionMetricsError(
            f"included trade {trade_date} {ticker} {direction} has non-finite {column}"
        )
    return values


def _trade_key(
    row: pd.Series,
    *,
    run_id: str | None,
    fill_label: str | None,
) -> tuple:
    rid = row["run_id"] if "run_id" in row.index and pd.notna(row.get("run_id")) else run_id
    fill = (
        row["fill_label"]
        if "fill_label" in row.index and pd.notna(row.get("fill_label"))
        else fill_label
    )
    return (
        rid,
        fill,
        _to_date(row["trade_date"]),
        str(row["ticker"]),
        str(row["direction"]),
    )


def assert_included_trade_legs(
    trade_log: pd.DataFrame,
    leg_log: pd.DataFrame,
    *,
    run_id: str | None = None,
    fill_label: str | None = None,
) -> None:
    """
    Abort when an included trade has missing, incomplete, duplicate, or
    unexpected legs, or when unit economics fail to reconcile.

    Does not change D2 ``result_complete`` (failed-date coverage) semantics.
    """
    log = _ensure_trade_date_column(trade_log)
    legs = _ensure_trade_date_column(leg_log) if leg_log is not None else pd.DataFrame()

    if log.empty or "included_in_portfolio" not in log.columns:
        return

    included = log[log["included_in_portfolio"] == True].copy()  # noqa: E712
    if included.empty:
        return

    if not legs.empty:
        legs = legs.copy()
        legs["trade_date"] = [_to_date(v) for v in legs["trade_date"]]

    included_keys = set()
    for _, trade in included.iterrows():
        key = _trade_key(trade, run_id=run_id, fill_label=fill_label)
        included_keys.add(key)
        trade_date = key[2]
        ticker = key[3]
        direction = key[4]

        if trade.get("structure_ok") != True:  # noqa: E712
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has structure_ok!=True"
            )

        instrument = str(trade.get("instrument_type") or "")
        if instrument in _STRADDLE_INSTRUMENTS:
            required = {0, 1}
        elif instrument in _IRON_FLY_INSTRUMENTS:
            required = {0, 1, 2, 3}
        else:
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has "
                f"unsupported instrument_type={instrument!r}"
            )

        if legs.empty:
            matched = legs
        else:
            mask = (
                (legs["trade_date"] == trade_date)
                & (legs["ticker"].astype(str) == ticker)
                & (legs["direction"].astype(str) == direction)
            )
            if "run_id" in legs.columns and key[0] is not None:
                mask = mask & (legs["run_id"].astype(str) == str(key[0]))
            if "fill_label" in legs.columns and key[1] is not None:
                mask = mask & (legs["fill_label"].astype(str) == str(key[1]))
            matched = legs.loc[mask]

        if matched.empty:
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has no matching leg rows"
            )

        indices = [int(v) for v in matched["leg_index"].tolist()]
        if len(indices) != len(set(indices)):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has duplicate leg_index values"
            )
        observed = set(indices)
        if observed != required:
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has unexpected "
                f"leg_index set {sorted(observed)}; expected {sorted(required)}"
            )
        if len(matched) != len(required):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} has {len(matched)} "
                f"leg rows; expected {len(required)}"
            )

        entry_sum = float(
            _finite_leg_numeric_series(
                matched,
                trade_date=trade_date,
                ticker=ticker,
                direction=direction,
                column="entry_cash_per_unit",
            ).sum()
        )
        payoff_sum = float(
            _finite_leg_numeric_series(
                matched,
                trade_date=trade_date,
                ticker=ticker,
                direction=direction,
                column="expiry_payoff_per_unit",
            ).sum()
        )
        pnl_unit_sum = float(
            _finite_leg_numeric_series(
                matched,
                trade_date=trade_date,
                ticker=ticker,
                direction=direction,
                column="pnl_per_unit",
            ).sum()
        )
        pnl_total_sum = float(
            _finite_leg_numeric_series(
                matched,
                trade_date=trade_date,
                ticker=ticker,
                direction=direction,
                column="pnl_total_leg",
            ).sum()
        )

        entry_cost = float(pd.to_numeric(trade.get("entry_cost_per_share"), errors="coerce"))
        pnl_share = float(pd.to_numeric(trade.get("pnl_per_share"), errors="coerce"))
        pnl_total = float(pd.to_numeric(trade.get("pnl_total"), errors="coerce"))
        if "exit_value" in trade.index and pd.notna(trade.get("exit_value")):
            exit_value = float(pd.to_numeric(trade.get("exit_value"), errors="coerce"))
        else:
            exit_value = entry_cost + pnl_share

        if not _values_close(entry_sum, entry_cost):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} entry cash "
                f"{entry_sum} != entry_cost_per_share {entry_cost}"
            )
        if not _values_close(payoff_sum, exit_value):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} expiry payoff "
                f"{payoff_sum} != exit_value {exit_value}"
            )
        if not _values_close(pnl_unit_sum, pnl_share):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} pnl_per_unit "
                f"{pnl_unit_sum} != pnl_per_share {pnl_share}"
            )
        if not _values_close(pnl_total_sum, pnl_total):
            raise DecisionMetricsError(
                f"included trade {trade_date} {ticker} {direction} pnl_total_leg "
                f"{pnl_total_sum} != pnl_total {pnl_total}"
            )

    if legs.empty:
        return
    extra_included = legs
    if "included_in_portfolio" in legs.columns:
        extra_included = legs[legs["included_in_portfolio"] == True]  # noqa: E712
    for _, leg in extra_included.iterrows():
        key = _trade_key(leg, run_id=run_id, fill_label=fill_label)
        if key not in included_keys:
            raise DecisionMetricsError(
                f"leg log has included legs for unmatched trade "
                f"{key[2]} {key[3]} {key[4]}"
            )


# ---------------------------------------------------------------------------
# Commit 3 — candidate view, funnel totals, decision report JSON/MD
# ---------------------------------------------------------------------------

CANDIDATE_VIEW_COLUMNS = [
    "run_id",
    "fill_label",
    "trade_date",
    "ticker",
    "direction",
    "decision_status",
    "stage",
    "reason_code",
    "reason_raw",
]

REPORT_WINDOWS: tuple[tuple[str, date, date], ...] = (
    ("full_history", FULL_HISTORY_START, FULL_HISTORY_END),
    ("primary", PRIMARY_START, PRIMARY_END),
)

STRUCTURE_REASON_CODES = (
    "metadata_error",
    "missing_quotes_or_body",
    "wing_or_liquidity_selection",
    "other_structure",
)

_PORTFOLIO_REASON_CODES = frozenset(
    {
        "max_names_cap",
        "invalid_max_loss",
        "premium_exceeds_fair_share",
        "max_loss_exceeds_fair_share",
        "no_short_credit",
        "earnings_exclusion",
    }
)

_STRUCTURE_REASON_PREFIXES: tuple[tuple[str, str], ...] = (
    ("metadata_error:", "metadata_error"),
    ("No quote surface rows", "missing_quotes_or_body"),
    ("No eligible quotes", "missing_quotes_or_body"),
    ("Missing body call/put", "missing_quotes_or_body"),
    ("Missing tradeable body call/put", "missing_quotes_or_body"),
    ("No quotes with abs_delta", "wing_or_liquidity_selection"),
    ("Iron fly spread_cost_ratio=", "wing_or_liquidity_selection"),
)

SELECTION_BIAS_NOTICE = (
    "Post-signal candidate means the name already passed the Momentum-tail and "
    "within-side CVG filters; these artifacts cannot support full-universe "
    "Momentum IC or CVG increment tests."
)

REPORT_LIMITATIONS: tuple[str, ...] = (
    "Hold-to-expiry; positions are not managed intra-week.",
    "No earnings filter.",
    "Iron-fly wings use below-nearest 0.15-delta selection.",
    "Tier A sizing is not integer lots.",
    "Long-only fallback dates are possible.",
    "Mid is a fill-assumption diagnostic, not a pure transaction-cost attribution.",
    "robust_score is not a decision metric and is not used for go/no-go.",
    SELECTION_BIAS_NOTICE,
)


def classify_structure_reason_code(failure_reason: Any) -> Optional[str]:
    """Map a structure ``failure_reason`` prefix to one of four stable codes."""
    if failure_reason is None or (isinstance(failure_reason, float) and pd.isna(failure_reason)):
        return "other_structure"
    text = str(failure_reason)
    if text == "" or text.lower() == "nan":
        return "other_structure"
    for prefix, code in _STRUCTURE_REASON_PREFIXES:
        if text.startswith(prefix):
            return code
    return "other_structure"


def classify_portfolio_reason_code(exclusion_reason: Any) -> Optional[str]:
    """Map an S5 ``exclusion_reason`` onto the frozen portfolio vocabulary."""
    if exclusion_reason is None or (isinstance(exclusion_reason, float) and pd.isna(exclusion_reason)):
        return "other_exclusion"
    text = str(exclusion_reason)
    if text in _PORTFOLIO_REASON_CODES:
        return text
    return "other_exclusion"


def build_candidate_view(
    trade_log: pd.DataFrame,
    *,
    run_id: str,
    fill_label: str,
) -> pd.DataFrame:
    """Derive one candidate-view row per ``trade_log`` row (post-signal grain)."""
    empty = pd.DataFrame(columns=CANDIDATE_VIEW_COLUMNS)
    if trade_log is None or trade_log.empty:
        return empty

    frame = trade_log.copy()
    if "trade_date" not in frame.columns:
        return empty

    rows: list[dict[str, Any]] = []
    for _, raw in frame.iterrows():
        included = bool(raw.get("included_in_portfolio") is True)
        structure_ok = raw.get("structure_ok")
        if included:
            stage = "traded"
            decision_status = "traded"
            reason_code: Optional[str] = None
            reason_raw: Optional[str] = None
        elif structure_ok is not True:
            stage = "structure_failed"
            decision_status = "no_trade"
            reason_raw_val = raw.get("failure_reason")
            reason_raw = None if pd.isna(reason_raw_val) else str(reason_raw_val)
            reason_code = classify_structure_reason_code(reason_raw_val)
        else:
            stage = "portfolio_excluded"
            decision_status = "no_trade"
            reason_raw_val = raw.get("exclusion_reason")
            reason_raw = None if pd.isna(reason_raw_val) else str(reason_raw_val)
            reason_code = classify_portfolio_reason_code(reason_raw_val)

        row_fill = raw.get("fill_label", fill_label)
        if pd.isna(row_fill) or row_fill is None or str(row_fill) == "":
            row_fill = fill_label
        row_run = raw.get("run_id", run_id)
        if pd.isna(row_run) or row_run is None or str(row_run) == "":
            row_run = run_id

        rows.append(
            {
                "run_id": str(row_run),
                "fill_label": str(row_fill),
                "trade_date": _to_date(raw["trade_date"]),
                "ticker": raw.get("ticker"),
                "direction": raw.get("direction"),
                "decision_status": decision_status,
                "stage": stage,
                "reason_code": reason_code,
                "reason_raw": reason_raw,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_VIEW_COLUMNS)


def structure_failure_counts(candidate_view: pd.DataFrame) -> dict[str, int]:
    """Histogram of the four structure ``reason_code`` values (zeros for absent)."""
    counts = {code: 0 for code in STRUCTURE_REASON_CODES}
    if candidate_view is None or candidate_view.empty:
        return counts
    failed = candidate_view[candidate_view["stage"] == "structure_failed"]
    if failed.empty or "reason_code" not in failed.columns:
        return counts
    for code, n in failed["reason_code"].value_counts(dropna=False).items():
        key = str(code) if code is not None and not (isinstance(code, float) and pd.isna(code)) else "other_structure"
        if key in counts:
            counts[key] = int(n)
        else:
            counts["other_structure"] += int(n)
    return counts


def _null_aware_sum(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return None
    return float(valid.sum())


def _null_aware_mean(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _null_aware_n_evaluated(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.notna().sum())


def summarize_funnel(funnel_summary: pd.DataFrame) -> dict[str, Any]:
    """Aggregate funnel counts; averages skip null/unexecuted stage values."""
    if funnel_summary is None or funnel_summary.empty:
        return {
            "n_expected_dates": 0,
            "n_feature_covered_dates": 0,
            "joint_coverage_rate": float("nan"),
            "n_dates_with_universe": 0,
            "sum_universe": None,
            "mean_universe": None,
            "n_dates_with_jointly_eligible": 0,
            "sum_jointly_eligible": None,
            "mean_jointly_eligible": None,
            "n_dates_with_post_signal": 0,
            "sum_post_signal": None,
            "mean_post_signal": None,
            "sum_post_signal_long": None,
            "sum_post_signal_short": None,
            "n_dates_with_constructable": 0,
            "sum_constructable": None,
            "mean_constructable": None,
            "sum_constructable_long": None,
            "sum_constructable_short": None,
            "n_dates_with_included": 0,
            "sum_included": None,
            "mean_included": None,
            "sum_included_long": None,
            "sum_included_short": None,
            "selection_bias_notice": SELECTION_BIAS_NOTICE,
        }

    frame = funnel_summary.copy()
    if "trade_date" in frame.columns:
        frame["trade_date"] = [_to_date(v) for v in frame["trade_date"]]

    n_expected = int(pd.to_numeric(frame["n_expected"], errors="coerce").fillna(0).sum())
    n_feature = int(pd.to_numeric(frame["n_feature_covered"], errors="coerce").fillna(0).sum())
    coverage = float(n_feature / n_expected) if n_expected > 0 else float("nan")

    def _block(col: str) -> tuple[int, Optional[float], Optional[float]]:
        series = frame[col] if col in frame.columns else pd.Series(dtype=float)
        return (
            _null_aware_n_evaluated(series),
            _null_aware_sum(series),
            _null_aware_mean(series),
        )

    n_u, sum_u, mean_u = _block("n_universe")
    n_j, sum_j, mean_j = _block("n_jointly_eligible")
    n_p, sum_p, mean_p = _block("n_post_signal")
    n_c, sum_c, mean_c = _block("n_constructable")
    n_i, sum_i, mean_i = _block("n_included")

    return {
        "n_expected_dates": n_expected,
        "n_feature_covered_dates": n_feature,
        "joint_coverage_rate": coverage,
        "n_dates_with_universe": n_u,
        "sum_universe": sum_u,
        "mean_universe": mean_u,
        "n_dates_with_jointly_eligible": n_j,
        "sum_jointly_eligible": sum_j,
        "mean_jointly_eligible": mean_j,
        "n_dates_with_post_signal": n_p,
        "sum_post_signal": sum_p,
        "mean_post_signal": mean_p,
        "sum_post_signal_long": _null_aware_sum(frame.get("n_post_signal_long", pd.Series(dtype=float))),
        "sum_post_signal_short": _null_aware_sum(frame.get("n_post_signal_short", pd.Series(dtype=float))),
        "n_dates_with_constructable": n_c,
        "sum_constructable": sum_c,
        "mean_constructable": mean_c,
        "sum_constructable_long": _null_aware_sum(frame.get("n_constructable_long", pd.Series(dtype=float))),
        "sum_constructable_short": _null_aware_sum(frame.get("n_constructable_short", pd.Series(dtype=float))),
        "n_dates_with_included": n_i,
        "sum_included": sum_i,
        "mean_included": mean_i,
        "sum_included_long": _null_aware_sum(frame.get("n_included_long", pd.Series(dtype=float))),
        "sum_included_short": _null_aware_sum(frame.get("n_included_short", pd.Series(dtype=float))),
        "selection_bias_notice": SELECTION_BIAS_NOTICE,
    }


def report_jsonable(value: Any) -> Any:
    """JSON-ready conversion: NaN→null, +Infinity→\"Infinity\", stable types."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): report_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_jsonable(item) for item in value]
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == float("inf"):
            return "Infinity"
        if not np.isfinite(number):
            return None
        return number
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def dumps_decision_report(report: Mapping[str, Any]) -> str:
    """Deterministic standards-compliant JSON text for the decision report."""
    payload = report_jsonable(dict(report))
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _filter_frame_to_window(
    frame: pd.DataFrame,
    start: date,
    end: date,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    if frame is None or frame.empty or date_col not in frame.columns:
        return frame.copy() if frame is not None else pd.DataFrame()
    out = frame.copy()
    out[date_col] = [_to_date(v) for v in out[date_col]]
    mask = (out[date_col] >= start) & (out[date_col] <= end)
    return out.loc[mask].reset_index(drop=True)


def _window_fill_block(
    *,
    date_status: pd.DataFrame,
    date_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    funnel_summary: pd.DataFrame,
    candidate_view: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, Any]:
    metrics = evaluate_fill_window(
        date_status, date_summary, trade_log, start=start, end=end
    )
    funnel_w = _filter_frame_to_window(funnel_summary, start, end)
    candidates_w = _filter_frame_to_window(candidate_view, start, end)
    return {
        "window_start": metrics["window_start"],
        "window_end": metrics["window_end"],
        "date_class_counts": metrics["date_class_counts"],
        "result_complete": metrics["result_complete"],
        "has_unresolved_failures": metrics["has_unresolved_failures"],
        "view_a_conditional": {
            "mean_cycle_car": metrics["view_a"]["mean_cycle_car"],
            "sharpe": metrics["view_a"]["annualized_sharpe"],
            "drawdown": metrics["view_a"]["max_drawdown"],
            "n_traded": metrics["view_a"]["n_traded_dates"],
        },
        "view_b_calendar": {
            "compounded": metrics["view_b"]["compounded_return"],
            "annualized_return": metrics["view_b"]["annualized_return"],
            "sharpe": metrics["view_b"]["annualized_sharpe"],
            "drawdown": metrics["view_b"]["max_drawdown"],
            "complete": metrics["view_b"]["complete"],
        },
        "weekly": metrics["weekly"],
        "yearly": metrics["yearly"],
        "long_short": metrics["long_short"],
        "activity": metrics["activity"],
        "concentration": metrics["concentration"],
        "structure_failure_counts": structure_failure_counts(candidates_w),
        "funnel_totals": summarize_funnel(funnel_w),
    }


def build_decision_report(
    *,
    mid: Mapping[str, Any],
    cross: Mapping[str, Any],
    experiment_id: str,
    contract_id: str,
    repo_sha: str,
) -> dict[str, Any]:
    """
    Assemble the dual-fill / dual-window decision report.

    Aborts with ``DecisionMetricsError`` on broken traded-date economics or
    included-trade leg mismatches. Failed dates do not abort; they mark the
    pack incomplete.
    """
    packs = {"mid": dict(mid), "cross": dict(cross)}
    for label, pack in packs.items():
        if pack.get("fill_label") != label:
            raise DecisionMetricsError(
                f"fill pack labeled {label!r} has fill_label={pack.get('fill_label')!r}"
            )
        assert_report_preconditions(
            pack["date_status"], pack["date_summary"], pack["trade_log"]
        )
        assert_included_trade_legs(
            pack["trade_log"],
            pack["leg_log"],
            run_id=str(pack["run_id"]),
            fill_label=str(pack["fill_label"]),
        )
        pack["candidate_view"] = build_candidate_view(
            pack["trade_log"],
            run_id=str(pack["run_id"]),
            fill_label=str(pack["fill_label"]),
        )

    by_fill: dict[str, Any] = {}
    for fill_key, pack in packs.items():
        by_fill[fill_key] = {}
        for window_name, start, end in REPORT_WINDOWS:
            by_fill[fill_key][window_name] = _window_fill_block(
                date_status=pack["date_status"],
                date_summary=pack["date_summary"],
                trade_log=pack["trade_log"],
                funnel_summary=pack["funnel_summary"],
                candidate_view=pack["candidate_view"],
                start=start,
                end=end,
            )

    sensitivity: dict[str, Any] = {}
    for window_name, start, end in REPORT_WINDOWS:
        sensitivity[window_name] = compute_fill_assumption_sensitivity(
            cross_date_status=packs["cross"]["date_status"],
            cross_date_summary=packs["cross"]["date_summary"],
            cross_trade_log=packs["cross"]["trade_log"],
            mid_date_status=packs["mid"]["date_status"],
            mid_date_summary=packs["mid"]["date_summary"],
            mid_trade_log=packs["mid"]["trade_log"],
            start=start,
            end=end,
        )

    # Report completeness follows the official full-history calendars for both fills.
    result_complete = (
        by_fill["cross"]["full_history"]["result_complete"]
        and by_fill["mid"]["full_history"]["result_complete"]
    )
    has_unresolved = not result_complete

    return {
        "experiment_id": experiment_id,
        "contract_id": contract_id,
        "repo_sha": repo_sha,
        "result_complete": result_complete,
        "has_unresolved_failures": has_unresolved,
        "windows": {
            "full_history": {
                "start": FULL_HISTORY_START,
                "end": FULL_HISTORY_END,
            },
            "primary": {
                "start": PRIMARY_START,
                "end": PRIMARY_END,
            },
        },
        "fills": {
            "cross": {
                "role": "primary",
                "run_id": packs["cross"]["run_id"],
                "fill_label": "cross",
            },
            "mid": {
                "role": "diagnostic",
                "run_id": packs["mid"]["run_id"],
                "fill_label": "mid",
            },
        },
        "by_fill": by_fill,
        "fill_assumption_sensitivity": sensitivity,
        "concentration_primary_cross_top5": by_fill["cross"]["primary"]["concentration"],
        "limitations": list(REPORT_LIMITATIONS),
    }


def _fmt_metric(value: Any) -> str:
    converted = report_jsonable(value)
    if converted is None:
        return "n/a"
    if isinstance(converted, bool):
        return "true" if converted else "false"
    if isinstance(converted, float):
        return f"{converted:.6g}"
    return str(converted)


def render_decision_report_markdown(report: Mapping[str, Any]) -> str:
    """Compact human rendering of the same numbers as ``decision_report.json``."""
    lines: list[str] = []
    lines.append("# Sprint 006 baseline decision report")
    lines.append("")
    lines.append(
        "This pack summarizes the frozen mid+cross Surface baseline for the "
        "contract-defined full-history and primary windows. **Cross** is the "
        "primary economic view; **mid** is a fill-assumption diagnostic. "
        "No go/no-go conclusion is declared here."
    )
    lines.append("")
    if not report.get("result_complete", False):
        lines.append(
            "> **INCOMPLETE RESULT.** One or more expected dates are `failed`. "
            "Do not treat this pack as a complete backtest result (including turnover)."
        )
        lines.append("")

    lines.append(
        f"experiment_id=`{report.get('experiment_id')}` · "
        f"contract_id=`{report.get('contract_id')}` · "
        f"repo_sha=`{report.get('repo_sha')}` · "
        f"result_complete=`{report.get('result_complete')}`"
    )
    lines.append("")

    def _headline(fill_key: str, title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| window | view | mean_CAR / compounded | sharpe | drawdown | n_traded / complete |"
        )
        lines.append("|---|---|---|---|---|---|")
        for window_name, _, _ in REPORT_WINDOWS:
            block = report["by_fill"][fill_key][window_name]
            a = block["view_a_conditional"]
            b = block["view_b_calendar"]
            lines.append(
                f"| {window_name} | A conditional | {_fmt_metric(a['mean_cycle_car'])} | "
                f"{_fmt_metric(a['sharpe'])} | {_fmt_metric(a['drawdown'])} | "
                f"{_fmt_metric(a['n_traded'])} |"
            )
            lines.append(
                f"| {window_name} | B calendar | {_fmt_metric(b['compounded'])} | "
                f"{_fmt_metric(b['sharpe'])} | {_fmt_metric(b['drawdown'])} | "
                f"{_fmt_metric(b['complete'])} |"
            )
        lines.append("")

    _headline("cross", "Cross (primary)")
    _headline("mid", "Mid (diagnostic)")

    lines.append("## Completeness and date classes")
    lines.append("")
    lines.append("| fill | window | expected | traded | valid_no_trade | failed | complete |")
    lines.append("|---|---|---|---|---|---|---|")
    for fill_key in ("cross", "mid"):
        for window_name, _, _ in REPORT_WINDOWS:
            c = report["by_fill"][fill_key][window_name]["date_class_counts"]
            lines.append(
                f"| {fill_key} | {window_name} | {c['n_expected_dates']} | "
                f"{c['n_traded_dates']} | {c['n_valid_no_trade_dates']} | "
                f"{c['n_failed_dates']} | {c['result_complete']} |"
            )
    lines.append("")

    lines.append("## Weekly diagnostics (cross)")
    lines.append("")
    lines.append("| window | win_rate | profit_factor | no_trade_frequency |")
    lines.append("|---|---|---|---|")
    for window_name, _, _ in REPORT_WINDOWS:
        w = report["by_fill"]["cross"][window_name]["weekly"]
        lines.append(
            f"| {window_name} | {_fmt_metric(w.get('win_rate'))} | "
            f"{_fmt_metric(w.get('profit_factor'))} | "
            f"{_fmt_metric(w.get('no_trade_frequency'))} |"
        )
    lines.append("")

    lines.append("## Yearly diagnostics (cross, primary window)")
    lines.append("")
    yearly = report["by_fill"]["cross"]["primary"]["yearly"]
    if yearly:
        lines.append("| year | n_expected | n_traded | compounded | sharpe |")
        lines.append("|---|---|---|---|---|")
        for row in yearly:
            counts = row.get("date_class_counts", {})
            view_b = row.get("view_b", {})
            lines.append(
                f"| {row.get('year')} | {counts.get('n_expected_dates')} | "
                f"{counts.get('n_traded_dates')} | "
                f"{_fmt_metric(view_b.get('compounded_return'))} | "
                f"{_fmt_metric(view_b.get('annualized_sharpe'))} |"
            )
    else:
        lines.append("_No yearly rows in the primary window._")
    lines.append("")

    lines.append("## Long / short attribution (cross, primary)")
    lines.append("")
    ls = report["by_fill"]["cross"]["primary"]["long_short"]
    long_side = ls.get("long", {})
    short_side = ls.get("short", {})
    lines.append(
        f"- long mean cycle return: {_fmt_metric(long_side.get('mean_cycle_return'))}; "
        f"short mean cycle return: {_fmt_metric(short_side.get('mean_cycle_return'))}"
    )
    lines.append(
        f"- long PnL: {_fmt_metric(long_side.get('pnl_total'))}; "
        f"short PnL: {_fmt_metric(short_side.get('pnl_total'))}"
    )
    lines.append("")

    lines.append("## Activity and concentration")
    lines.append("")
    act = report["by_fill"]["cross"]["primary"]["activity"]
    turnover = act.get("turnover", {})
    conc = report["concentration_primary_cross_top5"]
    lines.append(
        f"- primary cross activity: mean included names/traded date="
        f"{_fmt_metric(act.get('avg_included_names_per_traded_date'))}, "
        f"turnover_complete={_fmt_metric(turnover.get('complete'))}, "
        f"mean_turnover_names={_fmt_metric(turnover.get('mean_included_names'))}"
    )
    lines.append(
        f"- top-5 |PnL| share (primary cross): {_fmt_metric(conc.get('top5_share_sum'))}"
    )
    lines.append("")

    lines.append("## Structure-failure counts")
    lines.append("")
    lines.append("| fill | window | metadata_error | missing_quotes_or_body | wing_or_liquidity_selection | other_structure |")
    lines.append("|---|---|---|---|---|---|")
    for fill_key in ("cross", "mid"):
        for window_name, _, _ in REPORT_WINDOWS:
            s = report["by_fill"][fill_key][window_name]["structure_failure_counts"]
            lines.append(
                f"| {fill_key} | {window_name} | {s['metadata_error']} | "
                f"{s['missing_quotes_or_body']} | {s['wing_or_liquidity_selection']} | "
                f"{s['other_structure']} |"
            )
    lines.append("")

    lines.append("## Funnel totals (cross)")
    lines.append("")
    lines.append("| window | expected | feature_covered | mean jointly eligible | sum included |")
    lines.append("|---|---|---|---|---|")
    for window_name, _, _ in REPORT_WINDOWS:
        f = report["by_fill"]["cross"][window_name]["funnel_totals"]
        lines.append(
            f"| {window_name} | {f['n_expected_dates']} | {f['n_feature_covered_dates']} | "
            f"{_fmt_metric(f['mean_jointly_eligible'])} | {_fmt_metric(f['sum_included'])} |"
        )
    lines.append("")
    lines.append(f"_Selection-bias notice:_ {SELECTION_BIAS_NOTICE}")
    lines.append("")

    lines.append("## Mid-versus-cross fill-assumption sensitivity")
    lines.append("")
    lines.append(
        "Cross-minus-mid is **not** a pure transaction-cost number: fills can also "
        "change sizing, inclusion, and selected structures."
    )
    lines.append("")
    lines.append(
        "| window | both traded | mid-only dates | cross-only dates | "
        "mid-only candidates | cross-only candidates |"
    )
    lines.append("|---|---|---|---|---|---|")
    for window_name, _, _ in REPORT_WINDOWS:
        sens = report["fill_assumption_sensitivity"][window_name]
        lines.append(
            f"| {window_name} | {_fmt_metric(sens.get('n_dates_both_traded'))} | "
            f"{_fmt_metric(sens.get('n_dates_mid_only'))} | "
            f"{_fmt_metric(sens.get('n_dates_cross_only'))} | "
            f"{_fmt_metric(sens.get('n_candidates_mid_only'))} | "
            f"{_fmt_metric(sens.get('n_candidates_cross_only'))} |"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in report.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
