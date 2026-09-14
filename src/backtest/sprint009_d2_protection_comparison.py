"""Sprint 009 D2 — fixed-quantity protection comparison on the accepted D1 panel.

Does not call SurfaceRunner, rerun selection, or read D0 for economics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.sprint007_artifact_validation import sha256_file
from src.backtest.sprint009_d0_body_wing_readiness import within_dollars

ACCEPTED_D1_DIR = Path("C:/MomentumCVG_env/runs/sprint009_d1_20260914T025142Z")
SUPERSEDED_D1_DIR = Path("C:/MomentumCVG_env/runs/sprint009_d1_20260914T001504Z")
ACCEPTED_D1_CODE_SHA = "5669773f356f6c33cef86bd0da30ce4051709a6b"
ACCEPTED_D1_VERDICT = "READY"
ACCEPTED_D1_HASHES = {
    "execution_receipt.json": "6acdc0c54732766352732bab1a91c47d53b0d84e4b641614ac660f53f457283d",
    "trade_decomposition.parquet": "72da77c10d3e8ba403012ffad785e5338d69be1f2e180135e3414e03f3a7213b",
    "date_decomposition.parquet": "49db7b233ee628551ec6fffd05ac76fc07e2b115e9d03019cb55ad054c6cf611",
    "annual_decomposition.parquet": "62be4856076fa5485d51a945dc15b83b358fc37cf513df71fad0abd1eabec607",
}
DEVELOPMENT_START = date(2020, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
ZERO_SHORT_DATE = date(2020, 3, 13)
EXPECTED_TRADES = 2087
EXPECTED_DATES = 209
EXPECTED_POSITIVE_DATES = 208
ANNUAL_YEARS = (2020, 2021, 2022, 2023)
WORST_N = 10
SIGN_EPS = 1e-6
FORBIDDEN_REPORT_KEYS = (
    "later_period_pnl",
    "filter_result",
    "margin_call",
    "uncovered_authorization",
    "primary_window_anchor_as_development",
)
TRADE_DOLLAR_COLUMNS = (
    "b_mid",
    "h_body",
    "w_mid",
    "h_wing",
    "w_pay",
    "p_body_cross",
    "p_fly_cross",
    "pnl_cross_official",
    "pnl_body_cross",
    "pnl_wing_cross",
    "pnl_legs_sum",
    "pnl_mid_at_cross_q",
)
DATE_DOLLAR_COLUMNS = ("b_mid", "h_body", "w_mid", "h_wing", "w_pay", "p_body_cross", "p_fly_cross")
ANNUAL_COUNT_COLUMNS = ("n_dates", "n_zero_short_dates", "n_trades")
DATE_COMPONENT_PAIRS = tuple((f"residual_{name}_vs_d1", f"saved_{name}") for name in DATE_DOLLAR_COLUMNS)
DATE_RECONCILE_PAIRS = DATE_COMPONENT_PAIRS + (("residual_mid_vs_trades", "trade_sum_fly_mid"),)
TRADE_COMPONENT_PAIRS = (
    ("residual_body_vs_d1", "pnl_body_cross"),
    ("residual_fly_vs_official", "pnl_cross_official"),
    ("residual_fly_vs_legs", "pnl_legs_sum"),
    ("residual_wing_vs_d1", "pnl_wing_cross"),
    ("residual_mid_vs_d1", "pnl_mid_at_cross_q"),
)
IDENTITY_PAIRS = (
    ("residual_identity_cross", "p_fly_cross"),
    ("residual_identity_mid", "p_fly_mid"),
)


class D2ComparisonError(Exception):
    """Raised when the helper cannot emit a blocked report."""


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class ComparisonResult:
    verdict: str
    gates: list[GateResult] = field(default_factory=list)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    worst: pd.DataFrame = field(default_factory=pd.DataFrame)
    frequency: dict[str, Any] = field(default_factory=dict)
    concentration: dict[str, Any] = field(default_factory=dict)
    drawdown: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def readiness_verdict(gates: list[GateResult]) -> str:
    if not gates or any(not gate.passed for gate in gates):
        return "BLOCKED"
    return "READY"


def d1_handoff_problems(receipt: dict[str, Any], d1_dir: Path, hashes: dict[str, str]) -> list[str]:
    """Named blockers for the accepted D1 handoff. Does not interpret economics."""
    problems: list[str] = []
    if Path(d1_dir).resolve() != ACCEPTED_D1_DIR.resolve():
        problems.append(f"D1 directory {d1_dir} is not the accepted directory {ACCEPTED_D1_DIR}")
    if Path(d1_dir).resolve() == SUPERSEDED_D1_DIR.resolve():
        problems.append("refusing the superseded D1 directory")
    observed_sha = receipt.get("code_sha")
    if observed_sha != ACCEPTED_D1_CODE_SHA:
        problems.append(f"D1 receipt code SHA {observed_sha} != {ACCEPTED_D1_CODE_SHA}")
    observed_verdict = receipt.get("verdict")
    if observed_verdict != ACCEPTED_D1_VERDICT:
        problems.append(f"D1 receipt verdict {observed_verdict!r} != {ACCEPTED_D1_VERDICT}")
    for name, expected in ACCEPTED_D1_HASHES.items():
        observed = hashes.get(name)
        if observed != expected:
            problems.append(f"{name} hash {observed} != {expected}")
    return problems


def accepted_d1_hashes(d1_dir: Path) -> dict[str, str]:
    return {name: sha256_file(d1_dir / name) for name in ACCEPTED_D1_HASHES}


def loss_avoided(body_pnl: float, fly_pnl: float) -> float:
    return max(-float(body_pnl), 0.0) - max(-float(fly_pnl), 0.0)


def drawdown_path(pnls: list[float]) -> tuple[list[float], list[float], float, float]:
    """Cumulative and drawdown from an initial peak of zero. Max drawdown is <= 0."""
    cumulative: list[float] = []
    drawdowns: list[float] = []
    level = 0.0
    peak = 0.0
    for pnl in pnls:
        level += float(pnl)
        peak = max(peak, level)
        cumulative.append(level)
        drawdowns.append(level - peak)
    ending = cumulative[-1] if cumulative else 0.0
    deepest = min(drawdowns) if drawdowns else 0.0
    return cumulative, drawdowns, ending, deepest


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_true_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value) is True
    return False


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _in_development(day: date) -> bool:
    return DEVELOPMENT_START <= day <= DEVELOPMENT_END


def _positive(value: float) -> bool:
    return _finite(value) and float(value) > SIGN_EPS


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10, method="linear")),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p90": float(np.percentile(array, 90, method="linear")),
    }


def _rate(count: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(count) / float(denominator)


def _share(numerator: float, denominator: float) -> tuple[float | None, str]:
    if not _finite(denominator) or abs(float(denominator)) <= SIGN_EPS:
        return None, "no gross losing dollars"
    return float(numerator) / float(denominator), ""


def _gross_loss(values: list[float]) -> float:
    return float(sum(max(-float(value), 0.0) for value in values))


def _scope_problems(trades: pd.DataFrame, dates: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if not trades.empty:
        for _, row in trades.iterrows():
            day = _as_date(row["trade_date"])
            label = str(row.get("window_label", ""))
            if label == "development" and not _in_development(day):
                problems.append(f"{day} {row.get('ticker')}: window_label development outside development window")
            elif _in_development(day) and label != "development":
                problems.append(f"{day} {row.get('ticker')}: trade_date in development window but window_label is {label}")
    if not dates.empty:
        for _, row in dates.iterrows():
            day = _as_date(row["trade_date"])
            label = str(row.get("window_label", "development"))
            if "window_label" in dates.columns:
                if label == "development" and not _in_development(day):
                    problems.append(f"{day}: calendar window_label development outside development window")
                elif _in_development(day) and label != "development":
                    problems.append(f"{day}: calendar date in development window but window_label is {label}")
            elif not _in_development(day):
                problems.append(f"{day}: calendar date outside development window")
    return problems


def _development_slice(trades: pd.DataFrame, dates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        kept_trades = trades
    else:
        days = trades["trade_date"].map(_as_date)
        kept_trades = trades.loc[(trades["window_label"].astype(str) == "development") & days.map(_in_development)].copy()
    if dates.empty:
        kept_dates = dates
    elif "window_label" in dates.columns:
        days = dates["trade_date"].map(_as_date)
        kept_dates = dates.loc[(dates["window_label"].astype(str) == "development") & days.map(_in_development)].copy()
    else:
        days = dates["trade_date"].map(_as_date)
        kept_dates = dates.loc[days.map(_in_development)].copy()
    return kept_trades, kept_dates


def _validate_trade(row: pd.Series, issues: list[str]) -> bool:
    key = f"{_as_date(row['trade_date'])} {row.get('ticker')} {row.get('direction')}"
    ok = True
    if str(row.get("direction", "")) != "short":
        issues.append(f"{key}: direction is not short")
        ok = False
    if not _is_true_bool(row.get("input_ok")):
        issues.append(f"{key}: input_ok is not an actual true boolean")
        ok = False
    if not _finite(row.get("Q")) or not float(row.get("Q")) > 0:
        issues.append(f"{key}: Q is not a positive finite quantity")
        ok = False
    if not _finite(row.get("quantity_cross_signed")) or not float(row.get("quantity_cross_signed")) < 0:
        issues.append(f"{key}: quantity_cross_signed is not negative")
        ok = False
    for name in TRADE_DOLLAR_COLUMNS:
        if not _finite(row.get(name)):
            issues.append(f"{key}: {name} missing or non-finite")
            ok = False
    return ok


def _annotate_trade(row: pd.Series) -> dict[str, Any]:
    w_mid = float(row["w_mid"])
    h_wing = float(row["h_wing"])
    w_pay = float(row["w_pay"])
    body = float(row["p_body_cross"])
    fly = float(row["p_fly_cross"])
    b_mid = float(row["b_mid"])
    cost = w_mid + h_wing
    net = w_pay - cost
    fly_mid = b_mid + w_pay - w_mid
    return {
        "trade_date": _as_date(row["trade_date"]),
        "ticker": str(row["ticker"]),
        "direction": "short",
        "window_label": "development",
        "Q": float(row["Q"]),
        "quantity_cross_signed": float(row["quantity_cross_signed"]),
        "input_ok": True,
        "b_mid": b_mid,
        "h_body": float(row["h_body"]),
        "w_mid": w_mid,
        "h_wing": h_wing,
        "w_pay": w_pay,
        "p_body_cross": body,
        "p_fly_cross": fly,
        "cost_cross": cost,
        "net_cross": net,
        "p_fly_mid": fly_mid,
        "net_mid": w_pay - w_mid,
        "pnl_cross_official": float(row["pnl_cross_official"]),
        "pnl_body_cross": float(row["pnl_body_cross"]),
        "pnl_wing_cross": float(row["pnl_wing_cross"]),
        "pnl_legs_sum": float(row["pnl_legs_sum"]),
        "pnl_mid_at_cross_q": float(row["pnl_mid_at_cross_q"]),
        "loss_avoided_trade": loss_avoided(body, fly),
        "residual_body_vs_d1": body - float(row["pnl_body_cross"]),
        "residual_fly_vs_official": fly - float(row["pnl_cross_official"]),
        "residual_fly_vs_legs": fly - float(row["pnl_legs_sum"]),
        "residual_wing_vs_d1": net - float(row["pnl_wing_cross"]),
        "residual_mid_vs_d1": fly_mid - float(row["pnl_mid_at_cross_q"]),
        "residual_identity_cross": fly - (body + net),
        "residual_identity_mid": fly_mid - (b_mid + w_pay - w_mid),
        "payout_positive": _positive(w_pay),
        "net_positive": _positive(net),
    }


def _build_dates(trades: pd.DataFrame, calendar: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    grouped = {_as_date(day): frame for day, frame in trades.groupby("trade_date")} if not trades.empty else {}
    calendar_days = [_as_date(value) for value in calendar["trade_date"]] if not calendar.empty else []
    if len(calendar_days) != len(set(calendar_days)):
        issues.append("duplicate development calendar dates")
    by_day = {}
    if not calendar.empty:
        for _, row in calendar.iterrows():
            by_day[_as_date(row["trade_date"])] = row
    extra = sorted(set(grouped) - set(calendar_days))
    if extra:
        issues.append(f"short trades missing from development calendar: {extra[:5]}")
    rows: list[dict[str, Any]] = []
    for day in sorted(set(calendar_days) | set(grouped)):
        status = by_day.get(day)
        frame = grouped.get(day, pd.DataFrame())
        n_trades = int(len(frame))
        klass = str(status["short_book_class"]) if status is not None else "blocked"
        if status is None:
            issues.append(f"{day}: date missing from development calendar")
            klass = "blocked"
        elif klass == "verified_zero_short" and n_trades != 0:
            issues.append(f"{day}: verified zero-short date has included trades")
        elif klass == "verified_positive_short" and n_trades == 0:
            issues.append(f"{day}: verified positive date has no trades")
        saved = {name: 0.0 for name in DATE_DOLLAR_COLUMNS}
        saved_n_trades: float | None = 0.0 if status is not None else None
        derived = {name: 0.0 for name in ("b_mid", "h_body", "w_mid", "h_wing", "w_pay", "p_body_cross", "p_fly_cross", "cost_cross", "net_cross", "p_fly_mid", "net_mid")}
        official_mid = 0.0
        if n_trades:
            for name in derived:
                derived[name] = float(frame[name].sum())
            official_mid = float(frame["pnl_mid_at_cross_q"].sum())
        if status is not None:
            for name in DATE_DOLLAR_COLUMNS:
                if not _finite(status.get(name)):
                    issues.append(f"{day}: saved {name} missing or non-finite")
                    saved[name] = float("nan")
                else:
                    saved[name] = float(status[name])
            if not _finite(status.get("n_trades")):
                issues.append(f"{day}: saved n_trades missing or non-finite")
                saved_n_trades = None
            else:
                saved_n_trades = float(status["n_trades"])
        body = derived["p_body_cross"]
        fly = derived["p_fly_cross"]
        fly_mid = derived["p_fly_mid"]
        component_residuals = {
            name: (derived[name] - saved[name] if _finite(saved[name]) else None) for name in DATE_DOLLAR_COLUMNS
        }
        rows.append(
            {
                "trade_date": day,
                "short_book_class": klass,
                "n_trades": n_trades,
                **derived,
                "loss_avoided_date": loss_avoided(body, fly),
                **{f"saved_{name}": saved[name] for name in DATE_DOLLAR_COLUMNS},
                "saved_n_trades": saved_n_trades,
                **{f"residual_{name}_vs_d1": component_residuals[name] for name in DATE_DOLLAR_COLUMNS},
                "residual_n_trades_vs_d1": None if saved_n_trades is None else float(n_trades) - saved_n_trades,
                "residual_body_vs_d1": component_residuals["p_body_cross"],
                "residual_fly_vs_d1": component_residuals["p_fly_cross"],
                "residual_mid_vs_trades": fly_mid - official_mid,
                "trade_sum_fly_mid": official_mid,
                "residual_identity_cross": fly - (body + derived["net_cross"]),
                "residual_identity_mid": fly_mid - (derived["b_mid"] + derived["w_pay"] - derived["w_mid"]),
                "payout_positive": _positive(derived["w_pay"]),
                "net_positive": _positive(derived["net_cross"]),
            }
        )
    return pd.DataFrame(rows)


def _build_annual(trades: pd.DataFrame, dates: pd.DataFrame, annual: pd.DataFrame | None, issues: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if dates.empty:
        return pd.DataFrame(rows)
    ordered = dates.sort_values("trade_date")
    body_path, _, _, _ = drawdown_path([float(value) for value in ordered["p_body_cross"]])
    fly_path, _, _, _ = drawdown_path([float(value) for value in ordered["p_fly_cross"]])
    path_by_date = {
        _as_date(day): (body_path[index], fly_path[index])
        for index, day in enumerate(ordered["trade_date"])
    }
    saved_by_year = {}
    if annual is not None and not annual.empty:
        for _, row in annual.iterrows():
            saved_by_year[int(row["year"])] = row
    dated = dates.copy()
    dated["year"] = dated["trade_date"].map(lambda value: _as_date(value).year)
    for year in ANNUAL_YEARS:
        date_frame = dated.loc[dated["year"] == year]
        if date_frame.empty:
            continue
        year_days = {_as_date(value) for value in date_frame["trade_date"]}
        year_trades = trades.loc[trades["trade_date"].map(_as_date).isin(year_days)] if not trades.empty else trades
        last_day = max(year_days)
        ending_body, ending_fly = path_by_date[last_day]
        body = float(date_frame["p_body_cross"].sum())
        fly = float(date_frame["p_fly_cross"].sum())
        w_pay = float(date_frame["w_pay"].sum())
        w_mid = float(date_frame["w_mid"].sum())
        h_wing = float(date_frame["h_wing"].sum())
        b_mid = float(date_frame["b_mid"].sum())
        h_body = float(date_frame["h_body"].sum())
        net = float(date_frame["net_cross"].sum())
        fly_mid = float(date_frame["p_fly_mid"].sum())
        row: dict[str, Any] = {
            "year": year,
            "n_dates": int(len(date_frame)),
            "n_zero_short_dates": int((date_frame["short_book_class"] == "verified_zero_short").sum()),
            "n_trades": int(date_frame["n_trades"].sum()),
            "p_body_cross": body,
            "p_fly_cross": fly,
            "net_cross": net,
            "w_pay": w_pay,
            "cost_cross": float(date_frame["cost_cross"].sum()),
            "w_mid": w_mid,
            "h_wing": h_wing,
            "h_body": h_body,
            "b_mid": b_mid,
            "p_fly_mid": fly_mid,
            "payout_positive_trades": int(year_trades["payout_positive"].sum()) if not year_trades.empty else 0,
            "net_positive_trades": int(year_trades["net_positive"].sum()) if not year_trades.empty else 0,
            "ending_cumulative_body_cross": ending_body,
            "ending_cumulative_fly_cross": ending_fly,
            "residual_identity_cross": fly - (body + (w_pay - w_mid - h_wing)),
            "residual_identity_mid": fly_mid - (b_mid + (w_pay - w_mid)),
        }
        saved = saved_by_year.get(year)
        if saved is None:
            issues.append(f"{year}: missing from annual_decomposition")
            for name in DATE_DOLLAR_COLUMNS:
                row[f"saved_{name}"] = None
                row[f"residual_{name}_vs_d1"] = None
            for name in ANNUAL_COUNT_COLUMNS:
                row[f"residual_{name}_vs_d1"] = None
        else:
            for name in DATE_DOLLAR_COLUMNS:
                observed = saved.get(name)
                row[f"saved_{name}"] = None if not _finite(observed) else float(observed)
                row[f"residual_{name}_vs_d1"] = None if not _finite(observed) else float(row[name]) - float(observed)
            for name in ANNUAL_COUNT_COLUMNS:
                observed = saved.get(name)
                row[f"residual_{name}_vs_d1"] = None if not _finite(observed) else float(row[name]) - float(observed)
                if not _finite(observed) or int(observed) != int(row[name]):
                    issues.append(f"{year}: {name} {int(row[name])} != saved {observed}")
        rows.append(row)
    if annual is not None and not annual.empty:
        extra_years = sorted(set(int(value) for value in annual["year"]) - set(ANNUAL_YEARS))
        if extra_years:
            issues.append(f"annual table has years outside 2020-2023: {extra_years}")
    return pd.DataFrame(rows)


def _rank_trades(trades: pd.DataFrame, column: str, list_id: str) -> pd.DataFrame:
    ordered = trades.sort_values([column, "trade_date", "ticker", "direction"], ascending=True, kind="mergesort")
    chosen = ordered.head(WORST_N)
    rows = []
    for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
        rows.append(
            {
                "list_id": list_id,
                "rank": rank,
                "trade_date": row["trade_date"],
                "ticker": row["ticker"],
                "direction": row["direction"],
                "n_trades": 1,
                "p_body_cross": float(row["p_body_cross"]),
                "p_fly_cross": float(row["p_fly_cross"]),
                "w_pay": float(row["w_pay"]),
                "cost_cross": float(row["cost_cross"]),
                "net_cross": float(row["net_cross"]),
                "loss_avoided": float(row["loss_avoided_trade"]),
            }
        )
    return pd.DataFrame(rows)


def _rank_dates(dates: pd.DataFrame, column: str, list_id: str) -> pd.DataFrame:
    ordered = dates.sort_values([column, "trade_date"], ascending=True, kind="mergesort")
    chosen = ordered.head(WORST_N)
    rows = []
    for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
        rows.append(
            {
                "list_id": list_id,
                "rank": rank,
                "trade_date": row["trade_date"],
                "ticker": None,
                "direction": None,
                "n_trades": int(row["n_trades"]),
                "p_body_cross": float(row["p_body_cross"]),
                "p_fly_cross": float(row["p_fly_cross"]),
                "w_pay": float(row["w_pay"]),
                "cost_cross": float(row["cost_cross"]),
                "net_cross": float(row["net_cross"]),
                "loss_avoided": loss_avoided(float(row["p_body_cross"]), float(row["p_fly_cross"])),
            }
        )
    return pd.DataFrame(rows)


def _concentration_block(label: str, ranked: pd.DataFrame, ranked_column: str, universe: list[float]) -> dict[str, Any]:
    denominator = _gross_loss(universe)
    list_values = [float(value) for value in ranked[ranked_column]] if not ranked.empty else []
    numerator = _gross_loss(list_values)
    top_value = list_values[0] if list_values else 0.0
    share, reason = _share(numerator, denominator)
    top_share, top_reason = _share(max(-top_value, 0.0), denominator)
    return {
        "list_id": label,
        "denominator_gross_losing_dollars": denominator,
        "worst10_gross_losing_dollars": numerator,
        "worst10_share": share,
        "top1_gross_losing_dollars": max(-top_value, 0.0),
        "top1_share": top_share,
        "reason": reason or top_reason,
    }


def _frequency(trades: pd.DataFrame, dates: pd.DataFrame) -> dict[str, Any]:
    n_trades = int(len(trades))
    n_dates = int(len(dates))
    payout_count = int(trades["payout_positive"].sum()) if n_trades else 0
    net_count = int(trades["net_positive"].sum()) if n_trades else 0
    date_payout = int(dates["payout_positive"].sum()) if n_dates else 0
    date_net = int(dates["net_positive"].sum()) if n_dates else 0
    trade_reason = "" if n_trades else "no development trades"
    date_reason = "" if n_dates else "no development dates"
    conditional = trades.loc[trades["payout_positive"]] if n_trades else trades

    def _fields(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"gross_payout": None, "cross_purchase_cost": None, "net_contribution": None}
        return {
            "gross_payout": _distribution([float(value) for value in frame["w_pay"]]),
            "cross_purchase_cost": _distribution([float(value) for value in frame["cost_cross"]]),
            "net_contribution": _distribution([float(value) for value in frame["net_cross"]]),
        }

    mid_positive = int(trades["net_mid"].map(_positive).sum()) if n_trades else 0
    return {
        "trade_counts": {
            "n_trades": n_trades,
            "payout_positive": payout_count,
            "payout_exceeded_cost": net_count,
            "net_contribution_positive": net_count,
            "rate_reason": trade_reason,
            "payout_positive_rate": _rate(payout_count, n_trades),
            "payout_exceeded_cost_rate": _rate(net_count, n_trades),
            "net_contribution_positive_rate": _rate(net_count, n_trades),
        },
        "date_counts": {
            "n_dates": n_dates,
            "payout_positive": date_payout,
            "payout_exceeded_cost": date_net,
            "net_contribution_positive": date_net,
            "rate_reason": date_reason,
            "payout_positive_rate": _rate(date_payout, n_dates),
            "payout_exceeded_cost_rate": _rate(date_net, n_dates),
            "net_contribution_positive_rate": _rate(date_net, n_dates),
        },
        "all_trades": {"n": n_trades, **_fields(trades)},
        "conditional_positive_payout": {
            "n": int(len(conditional)),
            "reason": "" if len(conditional) else "no positive wing payout",
            **_fields(conditional),
        },
        "midpoint": {
            "purchase_cost_sum": float(trades["w_mid"].sum()) if n_trades else 0.0,
            "net_sum": float(trades["net_mid"].sum()) if n_trades else 0.0,
            "net_positive_count": mid_positive,
            "net_positive_rate": _rate(mid_positive, n_trades),
            "rate_reason": trade_reason,
        },
    }


def _attach_paths(dates: pd.DataFrame) -> pd.DataFrame:
    if dates.empty:
        return dates
    ordered = dates.sort_values("trade_date").copy()
    body_c, body_dd, _, _ = drawdown_path([float(value) for value in ordered["p_body_cross"]])
    fly_c, fly_dd, _, _ = drawdown_path([float(value) for value in ordered["p_fly_cross"]])
    body_mid_c, body_mid_dd, _, _ = drawdown_path([float(value) for value in ordered["b_mid"]])
    fly_mid_c, fly_mid_dd, _, _ = drawdown_path([float(value) for value in ordered["p_fly_mid"]])
    ordered["cumulative_body_cross"] = body_c
    ordered["drawdown_body_cross"] = body_dd
    ordered["cumulative_fly_cross"] = fly_c
    ordered["drawdown_fly_cross"] = fly_dd
    ordered["cumulative_body_mid"] = body_mid_c
    ordered["drawdown_body_mid"] = body_mid_dd
    ordered["cumulative_fly_mid"] = fly_mid_c
    ordered["drawdown_fly_mid"] = fly_mid_dd
    return ordered.sort_values("trade_date").reset_index(drop=True)


def _drawdown_summary(dates: pd.DataFrame) -> dict[str, Any]:
    if dates.empty:
        empty = {"ending_cumulative": 0.0, "max_drawdown": 0.0}
        return {"body_cross": empty, "fly_cross": empty, "body_mid": empty, "fly_mid": empty}
    dates = dates.sort_values("trade_date")
    return {
        "body_cross": {
            "ending_cumulative": float(dates["cumulative_body_cross"].iloc[-1] if len(dates) else 0.0),
            "max_drawdown": float(dates["drawdown_body_cross"].min()),
        },
        "fly_cross": {
            "ending_cumulative": float(dates["cumulative_fly_cross"].iloc[-1]),
            "max_drawdown": float(dates["drawdown_fly_cross"].min()),
        },
        "body_mid": {
            "ending_cumulative": float(dates["cumulative_body_mid"].iloc[-1]),
            "max_drawdown": float(dates["drawdown_body_mid"].min()),
        },
        "fly_mid": {
            "ending_cumulative": float(dates["cumulative_fly_mid"].iloc[-1]),
            "max_drawdown": float(dates["drawdown_fly_mid"].min()),
        },
    }


def _official_coverage_problems(trades: pd.DataFrame, dates: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if len(trades) != EXPECTED_TRADES:
        problems.append(f"development trades {len(trades)} != {EXPECTED_TRADES}")
    if len(dates) != EXPECTED_DATES:
        problems.append(f"development dates {len(dates)} != {EXPECTED_DATES}")
    if dates.empty:
        problems.append("development calendar is empty")
        return problems
    positive = int((dates["short_book_class"] == "verified_positive_short").sum())
    zero = dates.loc[dates["short_book_class"] == "verified_zero_short", "trade_date"].map(_as_date).tolist()
    if positive != EXPECTED_POSITIVE_DATES:
        problems.append(f"verified_positive_short dates {positive} != {EXPECTED_POSITIVE_DATES}")
    if zero != [ZERO_SHORT_DATE]:
        problems.append(f"verified zero-short dates {zero} != {[ZERO_SHORT_DATE]}")
    years = {_as_date(value).year for value in dates["trade_date"]}
    missing = [year for year in ANNUAL_YEARS if year not in years]
    if missing:
        problems.append(f"development calendar missing years {missing}")
    return problems


def _quantity_preservation_problems(source: pd.DataFrame, output: pd.DataFrame) -> list[str]:
    """Output keys and quantities must be the source rows that were emitted, not a rescaled book."""
    problems: list[str] = []
    source_by_key: dict[tuple[date, str, str], pd.Series] = {}
    if not source.empty:
        for _, row in source.iterrows():
            key = (_as_date(row["trade_date"]), str(row["ticker"]), str(row["direction"]))
            source_by_key[key] = row
    output_by_key: dict[tuple[date, str, str], pd.Series] = {}
    if not output.empty:
        for _, row in output.iterrows():
            key = (_as_date(row["trade_date"]), str(row["ticker"]), str(row["direction"]))
            if key in output_by_key:
                problems.append(f"{key[0]} {key[1]} {key[2]}: duplicate output trade key")
            output_by_key[key] = row
    missing = sorted(set(source_by_key) - set(output_by_key))
    extra = sorted(set(output_by_key) - set(source_by_key))
    if missing:
        shown = ", ".join(f"{day} {ticker} {direction}" for day, ticker, direction in missing[:3])
        problems.append(f"output trade keys missing source rows: {shown}")
    if extra:
        shown = ", ".join(f"{day} {ticker} {direction}" for day, ticker, direction in extra[:3])
        problems.append(f"output trade keys absent from source rows: {shown}")
    for key, row in output_by_key.items():
        source_row = source_by_key.get(key)
        if source_row is None:
            continue
        label = f"{key[0]} {key[1]} {key[2]}"
        if not _finite(row.get("Q")) or not _finite(source_row.get("Q")) or float(row["Q"]) != float(source_row["Q"]):
            problems.append(f"{label}: output Q {row.get('Q')} != source Q {source_row.get('Q')}")
        signed = row.get("quantity_cross_signed")
        source_signed = source_row.get("quantity_cross_signed")
        if not _finite(signed) or not _finite(source_signed) or float(signed) != float(source_signed):
            problems.append(f"{label}: output quantity_cross_signed {signed} != source {source_signed}")
        if len(problems) >= 6:
            break
    return problems


def _residual_text(value: Any) -> str:
    if value is None or not _finite(value):
        return "undefined"
    return f"{float(value):.6g}"


def _dollar_residual_failures(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    *,
    limit: int = 8,
) -> list[str]:
    """Residual must be within the dollar tolerance of zero, using the matching reference column."""
    failed: list[str] = []
    if frame.empty:
        return ["empty frame"]
    for _, row in frame.iterrows():
        label = row.get("trade_date", row.get("year"))
        for residual_name, reference_name in pairs:
            residual = row.get(residual_name)
            reference = row.get(reference_name)
            if residual is None or not _finite(residual) or reference is None or not _finite(reference):
                failed.append(f"{label} {residual_name} residual={_residual_text(residual)}")
            elif not within_dollars(float(residual), 0.0, float(reference)):
                failed.append(
                    f"{label} {residual_name} residual={_residual_text(residual)} reference={_residual_text(reference)}"
                )
            else:
                continue
            if len(failed) >= limit:
                return failed
    return failed


def _count_residual_failures(frame: pd.DataFrame, columns: tuple[str, ...], *, limit: int = 8) -> list[str]:
    failed: list[str] = []
    if frame.empty:
        return ["empty frame"]
    for _, row in frame.iterrows():
        label = row.get("trade_date", row.get("year"))
        for name in columns:
            value = row.get(name)
            if value is None or not _finite(value) or float(value) != 0.0:
                failed.append(f"{label} {name} residual={_residual_text(value)}")
                if len(failed) >= limit:
                    return failed
    return failed


def _development_identity(trades: pd.DataFrame) -> dict[str, float | None]:
    if trades.empty:
        return {
            "residual_identity_cross": None,
            "residual_identity_mid": None,
            "reference_cross": None,
            "reference_mid": None,
        }
    body = float(trades["p_body_cross"].sum())
    fly = float(trades["p_fly_cross"].sum())
    net = float(trades["net_cross"].sum())
    b_mid = float(trades["b_mid"].sum())
    w_pay = float(trades["w_pay"].sum())
    w_mid = float(trades["w_mid"].sum())
    fly_mid = float(trades["p_fly_mid"].sum())
    return {
        "residual_identity_cross": fly - (body + net),
        "residual_identity_mid": fly_mid - (b_mid + (w_pay - w_mid)),
        "reference_cross": fly,
        "reference_mid": fly_mid,
    }


def _development_identity_failures(trades: pd.DataFrame) -> list[str]:
    values = _development_identity(trades)
    failed: list[str] = []
    cross = values["residual_identity_cross"]
    mid = values["residual_identity_mid"]
    if cross is None or not _finite(cross) or not within_dollars(float(cross), 0.0, float(values["reference_cross"] or 0.0)):
        failed.append(
            f"development residual_identity_cross residual={_residual_text(cross)} reference={_residual_text(values['reference_cross'])}"
        )
    if mid is None or not _finite(mid) or not within_dollars(float(mid), 0.0, float(values["reference_mid"] or 0.0)):
        failed.append(
            f"development residual_identity_mid residual={_residual_text(mid)} reference={_residual_text(values['reference_mid'])}"
        )
    return failed


def residual_summary(trades: pd.DataFrame, dates: pd.DataFrame, annual: pd.DataFrame) -> dict[str, Any]:
    trade_cols = tuple(name for name, _ in TRADE_COMPONENT_PAIRS) + tuple(name for name, _ in IDENTITY_PAIRS)
    date_cols = tuple(name for name, _ in DATE_RECONCILE_PAIRS) + ("residual_n_trades_vs_d1",) + tuple(name for name, _ in IDENTITY_PAIRS)
    annual_cols = tuple(name for name, _ in DATE_COMPONENT_PAIRS) + tuple(
        f"residual_{name}_vs_d1" for name in ANNUAL_COUNT_COLUMNS
    ) + tuple(name for name, _ in IDENTITY_PAIRS)
    return {
        "trade_max_abs": _max_abs_residuals(trades, trade_cols),
        "date_max_abs": _max_abs_residuals(dates, date_cols),
        "annual_max_abs": _max_abs_residuals(annual, annual_cols),
        "development_identity": _development_identity(trades),
    }


def _max_abs_residuals(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for name in columns:
        if frame.empty or name not in frame.columns:
            summary[name] = None
            continue
        finite = [abs(float(value)) for value in frame[name] if value is not None and _finite(value)]
        summary[name] = max(finite) if finite else None
    return summary


def evaluate_gates(
    *,
    trades: pd.DataFrame,
    dates: pd.DataFrame,
    annual: pd.DataFrame,
    worst: pd.DataFrame,
    issues: list[str],
    report: dict[str, Any],
    require_official_coverage: bool,
    handoff_problems: list[str] | None = None,
) -> list[GateResult]:
    gates: list[GateResult] = []
    coverage = list(handoff_problems or [])
    if require_official_coverage:
        coverage.extend(_official_coverage_problems(trades, dates))
    gates.append(
        GateResult(
            "provenance",
            not coverage,
            "development coverage matches the accepted D1 evidence" if not coverage else "; ".join(coverage[:6]),
        )
    )
    gates.append(
        GateResult(
            "coverage",
            not issues and (trades.empty or bool(trades["input_ok"].map(_is_true_bool).all())),
            "development inputs are finite and paired" if not issues else "; ".join(issues[:6]),
        )
    )
    identity_failed = _dollar_residual_failures(trades, IDENTITY_PAIRS) if not trades.empty else ["no trades"]
    if not dates.empty:
        identity_failed.extend(_dollar_residual_failures(dates, IDENTITY_PAIRS))
    if not annual.empty:
        identity_failed.extend(_dollar_residual_failures(annual, IDENTITY_PAIRS))
    identity_failed.extend(_development_identity_failures(trades))
    gates.append(
        GateResult(
            "identity",
            not identity_failed,
            "cross and midpoint identities hold" if not identity_failed else "; ".join(identity_failed[:6]),
        )
    )
    reconcile_failed = _dollar_residual_failures(trades, TRADE_COMPONENT_PAIRS) if not trades.empty else ["no trades"]
    if not dates.empty:
        reconcile_failed.extend(_dollar_residual_failures(dates, DATE_RECONCILE_PAIRS))
        reconcile_failed.extend(_count_residual_failures(dates, ("residual_n_trades_vs_d1",)))
    if not annual.empty:
        reconcile_failed.extend(_dollar_residual_failures(annual, DATE_COMPONENT_PAIRS))
        reconcile_failed.extend(_count_residual_failures(annual, tuple(f"residual_{name}_vs_d1" for name in ANNUAL_COUNT_COLUMNS)))
    gates.append(
        GateResult(
            "reconciliation",
            not reconcile_failed,
            "reconciled to saved D1 dollars" if not reconcile_failed else "; ".join(reconcile_failed[:6]),
        )
    )
    ranking_failed: list[str] = []
    for list_id in ("body_trades", "body_dates", "fly_trades", "fly_dates"):
        block = worst.loc[worst["list_id"] == list_id] if not worst.empty else worst
        if len(block) != WORST_N:
            ranking_failed.append(f"{list_id} has {len(block)} rows")
            continue
        if list_id.endswith("trades") and block["ticker"].isna().any():
            ranking_failed.append(f"{list_id} missing paired keys")
        if list_id.endswith("dates") and block["n_trades"].isna().any():
            ranking_failed.append(f"{list_id} missing n_trades")
        if not block["loss_avoided"].map(_finite).all():
            ranking_failed.append(f"{list_id} missing loss_avoided")
    gates.append(
        GateResult(
            "ranking",
            not ranking_failed,
            "four worst-10 lists are paired and ordered" if not ranking_failed else "; ".join(ranking_failed),
        )
    )
    calendar_failed: list[str] = []
    if dates.empty or not (dates["trade_date"].map(_as_date) == ZERO_SHORT_DATE).any():
        if require_official_coverage:
            calendar_failed.append("missing 2020-03-13")
    else:
        zero = dates.loc[dates["trade_date"].map(_as_date) == ZERO_SHORT_DATE].iloc[0]
        dollar_names = ("p_body_cross", "p_fly_cross", "w_pay", "cost_cross", "net_cross", "b_mid", "p_fly_mid")
        if int(zero["n_trades"]) != 0 or any(not _finite(zero[name]) or abs(float(zero[name])) > SIGN_EPS for name in dollar_names):
            calendar_failed.append("2020-03-13 is not a zero-dollar zero-short date")
        if "cumulative_body_cross" not in dates.columns or "cumulative_fly_cross" not in dates.columns:
            calendar_failed.append("zero-short date is not a cumulative step")
    gates.append(
        GateResult(
            "calendar",
            not calendar_failed,
            "development calendar retained, including the zero-short date" if not calendar_failed else "; ".join(calendar_failed),
        )
    )
    forbidden = [key for key in FORBIDDEN_REPORT_KEYS if key in report]
    gates.append(
        GateResult(
            "scope",
            not forbidden,
            "no later-period or margin field" if not forbidden else f"forbidden keys {forbidden}",
        )
    )
    return gates


def compare_development(
    trades: pd.DataFrame,
    dates: pd.DataFrame,
    annual: pd.DataFrame | None = None,
    aggregate: dict[str, Any] | None = None,
    *,
    require_official_coverage: bool = False,
    handoff_problems: list[str] | None = None,
) -> ComparisonResult:
    issues = _scope_problems(trades, dates)
    dev_trades, dev_dates = _development_slice(trades, dates)
    records: list[dict[str, Any]] = []
    if not dev_trades.empty:
        keys = list(zip(dev_trades["trade_date"].map(_as_date), dev_trades["ticker"].astype(str), dev_trades["direction"].astype(str)))
        if len(keys) != len(set(keys)):
            issues.append("duplicate trade keys")
        for _, row in dev_trades.iterrows():
            if _validate_trade(row, issues):
                records.append(_annotate_trade(row))
    compared = pd.DataFrame(records)
    date_table = _build_dates(compared, dev_dates, issues)
    date_table = _attach_paths(date_table)
    annual_table = _build_annual(compared, date_table, annual, issues)
    if aggregate is not None and not compared.empty:
        checks = {
            "development_official_pnl": float(compared["pnl_cross_official"].sum()),
            "p_body_cross": float(compared["p_body_cross"].sum()),
            "p_fly_cross": float(compared["p_fly_cross"].sum()),
            "w_mid": float(compared["w_mid"].sum()),
            "h_wing": float(compared["h_wing"].sum()),
            "w_pay": float(compared["w_pay"].sum()),
            "b_mid": float(compared["b_mid"].sum()),
        }
        for name, observed in checks.items():
            reference = aggregate.get(name)
            if reference is None or not _finite(reference) or not within_dollars(observed, float(reference), float(reference)):
                residual = None if reference is None or not _finite(reference) else observed - float(reference)
                issues.append(
                    f"aggregate {name} residual={_residual_text(residual)} reference={_residual_text(reference)}"
                )
    issues.extend(_quantity_preservation_problems(dev_trades, compared))
    worst_frames = []
    if not compared.empty:
        worst_frames.append(_rank_trades(compared, "p_body_cross", "body_trades"))
        worst_frames.append(_rank_trades(compared, "p_fly_cross", "fly_trades"))
    if not date_table.empty:
        worst_frames.append(_rank_dates(date_table, "p_body_cross", "body_dates"))
        worst_frames.append(_rank_dates(date_table, "p_fly_cross", "fly_dates"))
    worst = pd.concat(worst_frames, ignore_index=True) if worst_frames else pd.DataFrame()
    frequency = _frequency(compared, date_table) if not compared.empty else {}
    concentration = {}
    if not compared.empty and not date_table.empty and not worst.empty:
        concentration = {
            "body_trades": _concentration_block(
                "body_trades",
                worst.loc[worst["list_id"] == "body_trades"],
                "p_body_cross",
                [float(value) for value in compared["p_body_cross"]],
            ),
            "fly_trades": _concentration_block(
                "fly_trades",
                worst.loc[worst["list_id"] == "fly_trades"],
                "p_fly_cross",
                [float(value) for value in compared["p_fly_cross"]],
            ),
            "body_dates": _concentration_block(
                "body_dates",
                worst.loc[worst["list_id"] == "body_dates"],
                "p_body_cross",
                [float(value) for value in date_table["p_body_cross"]],
            ),
            "fly_dates": _concentration_block(
                "fly_dates",
                worst.loc[worst["list_id"] == "fly_dates"],
                "p_fly_cross",
                [float(value) for value in date_table["p_fly_cross"]],
            ),
        }
    drawdown = _drawdown_summary(date_table)
    totals = {}
    if not compared.empty:
        net = float(compared["net_cross"].sum())
        totals = {
            "n_trades": int(len(compared)),
            "n_dates": int(len(date_table)),
            "body_cross": float(compared["p_body_cross"].sum()),
            "wing_cross": net,
            "fly_cross": float(compared["p_fly_cross"].sum()),
            "body_mid": float(compared["b_mid"].sum()),
            "wing_mid": float(compared["net_mid"].sum()),
            "fly_mid": float(compared["p_fly_mid"].sum()),
            "cross_purchase_cost": float(compared["cost_cross"].sum()),
            "midpoint_purchase_cost": float(compared["w_mid"].sum()),
            "gross_payout": float(compared["w_pay"].sum()),
            "pnl_gained_by_removing_wings": -net,
            "loss_avoided_trade_sum": float(compared["loss_avoided_trade"].sum()),
            "loss_avoided_date_sum": float(date_table["loss_avoided_date"].sum()) if not date_table.empty else 0.0,
        }
    report = {
        "verdict": "BLOCKED",
        "totals": totals,
        "residuals": residual_summary(compared, date_table, annual_table),
        "caveat": "Midpoint results are diagnostic; this analysis does not establish whether midpoint fills are attainable.",
    }
    if any(key in report for key in FORBIDDEN_REPORT_KEYS):
        raise D2ComparisonError("report contains a forbidden field")
    gates = evaluate_gates(
        trades=compared,
        dates=date_table,
        annual=annual_table,
        worst=worst,
        issues=issues,
        report=report,
        require_official_coverage=require_official_coverage,
        handoff_problems=handoff_problems,
    )
    verdict = readiness_verdict(gates)
    report["verdict"] = verdict
    return ComparisonResult(
        verdict=verdict,
        gates=gates,
        trades=compared,
        dates=date_table,
        annual=annual_table,
        worst=worst,
        frequency=frequency,
        concentration=concentration,
        drawdown=drawdown,
        totals=totals,
        report=report,
        issues=issues,
    )


def _money(value: Any) -> str:
    if value is None or not _finite(value):
        return "undefined"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _residual_lines(residuals: dict[str, Any]) -> str:
    if not residuals:
        return "No residual summary."
    parts = []
    for level in ("trade_max_abs", "date_max_abs", "annual_max_abs"):
        values = residuals.get(level) or {}
        finite = {name: value for name, value in values.items() if value is not None}
        if not finite:
            parts.append(f"{level}: none")
            continue
        worst_name = max(finite, key=finite.get)
        parts.append(f"{level} largest absolute residual {worst_name}={_residual_text(finite[worst_name])}")
    identity = residuals.get("development_identity") or {}
    parts.append(
        "development "
        f"residual_identity_cross={_residual_text(identity.get('residual_identity_cross'))} "
        f"residual_identity_mid={_residual_text(identity.get('residual_identity_mid'))}"
    )
    return ". ".join(parts) + "."
    if value is None or not _finite(value):
        return "undefined"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def render_report_md(result: ComparisonResult) -> str:
    lines = [
        "# Sprint 009 D2 protection comparison",
        "",
        f"**Verdict:** `{result.verdict}`",
        "",
        "Fees = 0. Execution concession is not deducted twice. Midpoint results are diagnostic; this analysis does not establish whether midpoint fills are attainable. These are fixed-quantity expiry outcomes on the iron-fly-selected population. They are not intraperiod margin or liquidation results, and they do not establish unseen-tail safety. Gross payout is not loss avoided. D2 does not authorize uncovered trading.",
        "",
        "## Gates",
        "",
    ]
    for gate in result.gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"- {mark} `{gate.gate_id}`: {gate.detail}")
    if result.verdict != "READY":
        lines.extend(["", "## Residuals", ""])
        lines.append(_residual_lines(result.report.get("residuals", {})))
        lines.extend(["", "No interpretation. A failed gate is a named blocker, not a partial economic result.", ""])
        return "\n".join(lines)
    totals = result.totals
    freq = result.frequency
    trade_counts = freq.get("trade_counts", {})
    date_counts = freq.get("date_counts", {})
    lines.extend(
        [
            "",
            "## D2-A — When did wings pay?",
            "",
            f"Cross purchase cost is {_money(totals.get('cross_purchase_cost'))}. Gross expiry payout is {_money(totals.get('gross_payout'))}. Net wing contribution at cross is {_money(totals.get('wing_cross'))}.",
            f"Trades with positive payout: {trade_counts.get('payout_positive')} of {trade_counts.get('n_trades')}. Trades whose payout exceeded purchase cost: {trade_counts.get('payout_exceeded_cost')} of {trade_counts.get('n_trades')}. Those two counts are not the same event.",
            f"Date sums, including the zero-short date as not positive: payout positive on {date_counts.get('payout_positive')} of {date_counts.get('n_dates')} dates; net contribution positive on {date_counts.get('net_contribution_positive')} of {date_counts.get('n_dates')}.",
            "All-trade distributions include zeros and negatives. Conditional summaries use only trades with positive payout and are not the book.",
            f"Midpoint diagnostic, same quantities: purchase cost {_money(totals.get('midpoint_purchase_cost'))}, net wing contribution {_money(totals.get('wing_mid'))}.",
            "",
            "## D2-B — Worst observed losses",
            "",
            f"Historical P&L gained by removing wings is {_money(totals.get('pnl_gained_by_removing_wings'))}, equal to minus net wing contribution. That is not a decision to drop wings.",
            f"Trade-level loss avoided sums to {_money(totals.get('loss_avoided_trade_sum'))}. Date-level loss avoided, after netting trades within each date, sums to {_money(totals.get('loss_avoided_date_sum'))}. Those totals are not added together. Negative loss avoided means the wings increased the realized loss. Gross payout is not loss avoided.",
            "Concentration shares use gross losing dollars of the matching book and unit, not net P&L.",
            "",
            "## D2-C — Cumulative profit, drawdown, and years",
            "",
            f"Body-only cross ends at {_money(result.drawdown.get('body_cross', {}).get('ending_cumulative'))} with maximum dollar drawdown {_money(result.drawdown.get('body_cross', {}).get('max_drawdown'))}.",
            f"Iron-fly cross ends at {_money(result.drawdown.get('fly_cross', {}).get('ending_cumulative'))} with maximum dollar drawdown {_money(result.drawdown.get('fly_cross', {}).get('max_drawdown'))}.",
            "Drawdowns are non-positive and use an initial peak of zero. Annual P&L is that year's sum. Year-end cumulative is the endpoint of the continuous development series. This is not a compounded account path.",
            "",
            "## Residuals",
            "",
            _residual_lines(result.report.get("residuals", {})),
            "",
        ]
    )
    if not result.annual.empty:
        for _, row in result.annual.iterrows():
            lines.append(
                f"- {int(row['year'])}: body {_money(row['p_body_cross'])}, fly {_money(row['p_fly_cross'])}, "
                f"net wing {_money(row['net_cross'])}, year-end body cumulative {_money(row['ending_cumulative_body_cross'])}, "
                f"year-end fly cumulative {_money(row['ending_cumulative_fly_cross'])}."
            )
    lines.append("")
    return "\n".join(lines)
