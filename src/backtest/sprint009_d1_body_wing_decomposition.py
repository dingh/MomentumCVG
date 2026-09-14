"""Sprint 009 D1 — development body/wing margin decomposition.

Reads the accepted D0 matched panel. Does not call SurfaceRunner or rerun selection.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.sprint007_artifact_validation import expected_mid_fill_price
from src.backtest.sprint009_d0_body_wing_readiness import (
    signed_entry_cash,
    within_dollars,
)

ACCEPTED_D0_DIR = Path("C:/MomentumCVG_env/runs/sprint009_d0_20260913T215246Z")
SUPERSEDED_D0_DIR = Path("C:/MomentumCVG_env/runs/sprint009_d0_20260913T212939Z")
DEVELOPMENT_START = date(2020, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
ZERO_SHORT_DATE = date(2020, 3, 13)
EXPECTED_DEVELOPMENT_TRADES = 2087
EXPECTED_DEVELOPMENT_DATES = 209
EXPECTED_POSITIVE_DATES = 208
QUOTE_DIAGNOSTIC_TOL = 1e-6
WING_PAYOFF_FLOOR = -1e-6
ANNUAL_YEARS = (2020, 2021, 2022, 2023)

BODY_LEGS = (("body_put", -1), ("body_call", -1))
WING_LEGS = (("put_wing", 1), ("call_wing", 1))
ALL_LEGS = BODY_LEGS + WING_LEGS

DOLLAR_FIELDS = (
    "b_mid",
    "h_body",
    "w_mid",
    "h_wing",
    "w_pay",
    "p_body_cross",
    "p_fly_cross",
    "c_body",
    "pnl_cross_official",
    "pnl_body_cross",
    "pnl_wing_cross",
    "pnl_legs_sum",
    "pnl_mid_at_cross_q",
)
RESIDUAL_FIELDS = (
    "residual_body_vs_d0",
    "residual_wing_vs_d0",
    "residual_identity_vs_official",
    "residual_identity_vs_legs",
    "residual_mid_vs_d0",
)
FORBIDDEN_REPORT_KEYS = (
    "development_minus_later_pnl",
    "later_period_pnl",
    "filter_result",
    "protection_summary",
    "primary_window_anchor_as_development",
)


class D1DecompositionError(Exception):
    """Raised when the helper cannot write a blocked report."""


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class DecompositionResult:
    verdict: str
    gates: list[GateResult] = field(default_factory=list)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    dollars: dict[str, Any] = field(default_factory=dict)
    ratios: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def readiness_verdict(gates: list[GateResult]) -> str:
    if not gates or any(not gate.passed for gate in gates):
        return "BLOCKED"
    return "READY"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _in_development(day: date) -> bool:
    return DEVELOPMENT_START <= day <= DEVELOPMENT_END


def _key(row: pd.Series) -> tuple[date, str, str]:
    return (_as_date(row["trade_date"]), str(row["ticker"]), str(row["direction"]))


def _issue(key: tuple[date, str, str], field_name: str, reason: str) -> str:
    return f"{key}: {field_name}: {reason}"


def _ratio(numerator: float, denominator: float) -> float | None:
    if not _finite(numerator) or not _finite(denominator) or not denominator > 0:
        return None
    return float(numerator) / float(denominator)


def _ratio_reason(denominator: float) -> str:
    if not _finite(denominator):
        return "non-finite body midpoint credit"
    if not denominator > 0:
        return "non-positive body midpoint credit"
    return ""


def _date_ratio_reason(denominator: float, n_trades: int) -> str:
    if n_trades == 0 and _finite(denominator) and denominator == 0:
        return "zero body midpoint credit"
    return _ratio_reason(denominator)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _leg_quote(
    row: pd.Series,
    prefix: str,
    expected_unit: int,
    key: tuple[date, str, str],
    issues: list[str],
) -> dict[str, float] | None:
    unit_raw = row.get(f"{prefix}_unit_quantity")
    bid = row.get(f"{prefix}_bid")
    ask = row.get(f"{prefix}_ask")
    stored_mid = row.get(f"{prefix}_mid")
    payoff = row.get(f"{prefix}_expiry_payoff_per_unit")
    problems: list[str] = []
    if not _finite(unit_raw) or int(unit_raw) != expected_unit:
        problems.append(f"unit_quantity is not {expected_unit}")
    for name, value in (("bid", bid), ("ask", ask), ("mid", stored_mid), ("expiry_payoff_per_unit", payoff)):
        if not _finite(value):
            problems.append(f"{name} missing or non-finite")
    if _finite(bid) and _finite(ask) and float(ask) < float(bid):
        problems.append("crossed quote")
    if prefix in {name for name, _unit in WING_LEGS} and _finite(payoff) and float(payoff) < WING_PAYOFF_FLOOR:
        problems.append("expiry_payoff_per_unit disagrees with unsigned wing intrinsic")
    if problems:
        issues.append(_issue(key, prefix, "; ".join(problems)))
        return None
    unit = int(unit_raw)
    mid_fill = expected_mid_fill_price(bid, ask, unit)
    return {
        "unit": float(unit),
        "bid": float(bid),
        "ask": float(ask),
        "stored_mid": float(stored_mid),
        "mid_fill": float(mid_fill),
        "payoff": float(payoff),
        "abs_mid_gap": abs(float(stored_mid) - float(mid_fill)),
    }


def _blank_trade(row: pd.Series, reason: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trade_date": _as_date(row["trade_date"]),
        "ticker": str(row["ticker"]),
        "direction": "short",
        "window_label": "development",
        "Q": float(row["Q"]) if _finite(row.get("Q")) else None,
        "quantity_cross_signed": float(row["quantity_cross_signed"]) if _finite(row.get("quantity_cross_signed")) else None,
        "entry_spot": float(row["entry_spot"]) if _finite(row.get("entry_spot")) else None,
        "input_ok": False,
        "input_reason": reason,
        "ratio_reason": "non-finite body midpoint credit",
        "body_concession_ratio": None,
        "wing_premium_ratio": None,
        "wing_concession_ratio": None,
    }
    for prefix, _unit in ALL_LEGS:
        stored = row.get(f"{prefix}_mid")
        record[f"{prefix}_mid"] = float(stored) if _finite(stored) else None
    for name in DOLLAR_FIELDS:
        if name not in record:
            record[name] = float(row[name]) if name.startswith("pnl_") and _finite(row.get(name)) else None
    for name in RESIDUAL_FIELDS:
        record[name] = None
    return record


def decompose_trade(row: pd.Series) -> tuple[dict[str, Any], list[str], list[float]]:
    """Five-term decomposition of one development short iron fly. Does not drop the row."""
    key = _key(row)
    issues: list[str] = []
    if str(row.get("direction", "")) != "short":
        issues.append(_issue(key, "direction", "is not short"))
    if not bool(row.get("pairing_ok")):
        issues.append(_issue(key, "pairing_ok", "is not true"))
    quantity = row.get("quantity_cross_signed")
    q_value = row.get("Q")
    if not _finite(quantity) or not float(quantity) < 0:
        issues.append(_issue(key, "quantity_cross_signed", "sign is not negative"))
    if not _finite(q_value) or not float(q_value) > 0:
        issues.append(_issue(key, "Q", "magnitude is not positive"))
    for name in ("pnl_cross_official", "pnl_body_cross", "pnl_wing_cross", "pnl_legs_sum", "pnl_mid_at_cross_q"):
        if not _finite(row.get(name)):
            issues.append(_issue(key, name, "missing or non-finite"))
    quotes: dict[str, dict[str, float]] = {}
    for prefix, unit in ALL_LEGS:
        parsed = _leg_quote(row, prefix, unit, key, issues)
        if parsed is not None:
            quotes[prefix] = parsed
    if issues or len(quotes) != 4 or not _finite(q_value):
        return _blank_trade(row, "; ".join(issues)), issues, []

    q_mag = float(q_value)
    gaps = [quotes[prefix]["abs_mid_gap"] for prefix, _unit in ALL_LEGS]
    b_mid = 0.0
    h_body = 0.0
    c_body = 0.0
    for prefix, unit in BODY_LEGS:
        quote = quotes[prefix]
        mid_cash = signed_entry_cash(quote["mid_fill"], unit)
        b_mid += q_mag * (quote["payoff"] - mid_cash)
        h_body += q_mag * (quote["mid_fill"] - quote["bid"])
        c_body += q_mag * quote["mid_fill"]
    w_mid = 0.0
    h_wing = 0.0
    w_pay = 0.0
    for prefix, _unit in WING_LEGS:
        quote = quotes[prefix]
        w_mid += q_mag * quote["mid_fill"]
        h_wing += q_mag * (quote["ask"] - quote["mid_fill"])
        w_pay += q_mag * quote["payoff"]
    p_body = b_mid - h_body
    p_fly = b_mid - h_body - w_mid - h_wing + w_pay
    official = float(row["pnl_cross_official"])
    wing_derived = w_pay - w_mid - h_wing
    mid_derived = b_mid + w_pay - w_mid
    reason = _ratio_reason(c_body)
    record = {
        "trade_date": key[0],
        "ticker": key[1],
        "direction": "short",
        "window_label": "development",
        "Q": q_mag,
        "quantity_cross_signed": float(quantity),
        "entry_spot": float(row["entry_spot"]) if _finite(row.get("entry_spot")) else None,
        "pnl_cross_official": official,
        "pnl_body_cross": float(row["pnl_body_cross"]),
        "pnl_wing_cross": float(row["pnl_wing_cross"]),
        "pnl_legs_sum": float(row["pnl_legs_sum"]),
        "pnl_mid_at_cross_q": float(row["pnl_mid_at_cross_q"]),
        "b_mid": b_mid,
        "h_body": h_body,
        "w_mid": w_mid,
        "h_wing": h_wing,
        "w_pay": w_pay,
        "p_body_cross": p_body,
        "p_fly_cross": p_fly,
        "c_body": c_body,
        "residual_body_vs_d0": p_body - float(row["pnl_body_cross"]),
        "residual_wing_vs_d0": wing_derived - float(row["pnl_wing_cross"]),
        "residual_identity_vs_official": p_fly - official,
        "residual_identity_vs_legs": p_fly - float(row["pnl_legs_sum"]),
        "residual_mid_vs_d0": mid_derived - float(row["pnl_mid_at_cross_q"]),
        "body_concession_ratio": _ratio(h_body, c_body),
        "wing_premium_ratio": _ratio(w_mid, c_body),
        "wing_concession_ratio": _ratio(h_wing, c_body),
        "ratio_reason": reason,
        "input_ok": True,
        "input_reason": "",
    }
    for prefix, _unit in ALL_LEGS:
        record[f"{prefix}_mid"] = quotes[prefix]["stored_mid"]
    return record, [], gaps


def _scope_problems(matched: pd.DataFrame, calendar: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if not matched.empty:
        for _, row in matched.iterrows():
            day = _as_date(row["trade_date"])
            label = str(row.get("window_label", ""))
            if label == "development" and not _in_development(day):
                problems.append(f"{_key(row)}: window_label development outside 2020-01-01 through 2023-12-31")
            elif _in_development(day) and label != "development":
                problems.append(f"{_key(row)}: trade_date in development window but window_label is {label}")
    if not calendar.empty:
        for _, row in calendar.iterrows():
            day = _as_date(row["trade_date"])
            label = str(row.get("window_label", ""))
            if label == "development" and not _in_development(day):
                problems.append(f"{day}: calendar window_label development outside development window")
            elif _in_development(day) and label != "development":
                problems.append(f"{day}: calendar date in development window but window_label is {label}")
    return problems


def _development_frames(
    matched: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matched.empty:
        trades = matched
    else:
        days = matched["trade_date"].map(_as_date)
        trades = matched.loc[(matched["window_label"].astype(str) == "development") & days.map(_in_development)].copy()
    if calendar.empty:
        dates = calendar
    else:
        days = calendar["trade_date"].map(_as_date)
        dates = calendar.loc[(calendar["window_label"].astype(str) == "development") & days.map(_in_development)].copy()
    return trades, dates


def _sum_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns or not frame[column].notna().all():
        return None
    return float(frame[column].sum())


def _level_residuals(frame: pd.DataFrame) -> dict[str, float | None]:
    official = _sum_or_none(frame, "pnl_cross_official")
    body = _sum_or_none(frame, "p_body_cross")
    saved_body = _sum_or_none(frame, "pnl_body_cross")
    wing = None
    if _sum_or_none(frame, "w_pay") is not None:
        wing = (
            float(frame["w_pay"].sum())
            - float(frame["w_mid"].sum())
            - float(frame["h_wing"].sum())
        )
    saved_wing = _sum_or_none(frame, "pnl_wing_cross")
    fly = _sum_or_none(frame, "p_fly_cross")
    legs = _sum_or_none(frame, "pnl_legs_sum")
    mid = None
    if _sum_or_none(frame, "b_mid") is not None:
        mid = float(frame["b_mid"].sum()) + float(frame["w_pay"].sum()) - float(frame["w_mid"].sum())
    saved_mid = _sum_or_none(frame, "pnl_mid_at_cross_q")
    return {
        "residual_body_vs_d0": None if body is None or saved_body is None else body - saved_body,
        "residual_wing_vs_d0": None if wing is None or saved_wing is None else wing - saved_wing,
        "residual_identity_vs_official": None if fly is None or official is None else fly - official,
        "residual_identity_vs_legs": None if fly is None or legs is None else fly - legs,
        "residual_mid_vs_d0": None if mid is None or saved_mid is None else mid - saved_mid,
        "official": official if official is not None else 0.0,
    }


def _residuals_ok(residuals: dict[str, float | None], official: float) -> list[str]:
    failed: list[str] = []
    for name in RESIDUAL_FIELDS:
        value = residuals.get(name)
        if value is None or not within_dollars(value, 0.0, official):
            failed.append(name)
    return failed


def _build_dates(trades: pd.DataFrame, calendar: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = { _as_date(day): frame for day, frame in trades.groupby("trade_date") } if not trades.empty else {}
    calendar_days = [_as_date(value) for value in calendar["trade_date"]] if not calendar.empty else []
    seen = set(calendar_days)
    extra = sorted(set(grouped) - seen)
    if extra:
        issues.append(f"short trades missing from development calendar: {extra[:5]}")
    by_day = {}
    if not calendar.empty:
        for _, row in calendar.iterrows():
            by_day[_as_date(row["trade_date"])] = row
    for day in sorted(seen | set(grouped)):
        status = by_day.get(day)
        frame = grouped.get(day, pd.DataFrame())
        n_trades = int(len(frame))
        klass = str(status["short_book_class"]) if status is not None else "blocked"
        if status is None:
            issues.append(f"{day}: date missing from date_status calendar")
            klass = "blocked"
        elif klass == "blocked":
            issues.append(f"{day}: development calendar date is blocked")
        elif klass == "verified_zero_short" and n_trades != 0:
            issues.append(f"{day}: verified zero-short date has included trades")
        elif klass == "verified_positive_short" and n_trades == 0:
            issues.append(f"{day}: verified positive date has no trades")
        dollars = {name: 0.0 for name in ("b_mid", "h_body", "w_mid", "h_wing", "w_pay", "p_body_cross", "p_fly_cross", "c_body")}
        residuals = {name: 0.0 for name in RESIDUAL_FIELDS}
        if n_trades:
            level = _level_residuals(frame)
            for name in dollars:
                total = _sum_or_none(frame, name)
                dollars[name] = total
            for name in RESIDUAL_FIELDS:
                residuals[name] = level[name]
        c_body = dollars["c_body"]
        reason = _date_ratio_reason(c_body if c_body is not None else float("nan"), n_trades)
        rows.append(
            {
                "trade_date": day,
                "short_book_class": klass,
                "n_trades": n_trades,
                **dollars,
                **residuals,
                "body_concession_ratio": _ratio(dollars["h_body"] or float("nan"), c_body or float("nan")),
                "wing_premium_ratio": _ratio(dollars["w_mid"] or float("nan"), c_body or float("nan")),
                "wing_concession_ratio": _ratio(dollars["h_wing"] or float("nan"), c_body or float("nan")),
                "ratio_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _build_annual(trades: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if dates.empty:
        return pd.DataFrame(rows)
    dated = dates.copy()
    dated["year"] = dated["trade_date"].map(lambda value: _as_date(value).year)
    for year in ANNUAL_YEARS:
        date_frame = dated.loc[dated["year"] == year]
        if date_frame.empty:
            continue
        year_days = {_as_date(value) for value in date_frame["trade_date"]}
        year_trades = (
            trades.loc[trades["trade_date"].map(_as_date).isin(year_days)]
            if not trades.empty
            else trades
        )
        level = _level_residuals(year_trades) if not year_trades.empty else {name: 0.0 for name in (*RESIDUAL_FIELDS, "official")}
        c_body = _sum_or_none(year_trades, "c_body") if not year_trades.empty else 0.0
        row: dict[str, Any] = {
            "year": year,
            "n_dates": int(len(date_frame)),
            "n_zero_short_dates": int((date_frame["short_book_class"] == "verified_zero_short").sum()),
            "n_trades": int(date_frame["n_trades"].sum()),
        }
        for name in ("b_mid", "h_body", "w_mid", "h_wing", "w_pay", "p_body_cross", "p_fly_cross", "c_body"):
            row[name] = _sum_or_none(year_trades, name) if not year_trades.empty else 0.0
        row.update({name: level[name] for name in RESIDUAL_FIELDS})
        row["body_concession_ratio"] = _ratio(
            row["h_body"] if row["h_body"] is not None else float("nan"),
            c_body if c_body is not None else float("nan"),
        )
        row["wing_premium_ratio"] = _ratio(
            row["w_mid"] if row["w_mid"] is not None else float("nan"),
            c_body if c_body is not None else float("nan"),
        )
        row["wing_concession_ratio"] = _ratio(
            row["h_wing"] if row["h_wing"] is not None else float("nan"),
            c_body if c_body is not None else float("nan"),
        )
        row["ratio_reason"] = _date_ratio_reason(
            c_body if c_body is not None else float("nan"),
            int(date_frame["n_trades"].sum()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _null_ratio_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades.empty or "ratio_reason" not in trades.columns:
        return {}
    reasons = trades.loc[trades["ratio_reason"].astype(str) != "", "ratio_reason"]
    return {str(key): int(value) for key, value in reasons.value_counts().items()}


def _aggregate_payloads(trades: pd.DataFrame, dates: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    level = _level_residuals(trades) if not trades.empty else {name: 0.0 for name in RESIDUAL_FIELDS}
    if trades.empty:
        level["official"] = 0.0
    dollars = {
        "n_trades": int(len(trades)),
        "n_dates": int(len(dates)),
        "development_official_pnl": _sum_or_none(trades, "pnl_cross_official") if not trades.empty else 0.0,
        "b_mid": _sum_or_none(trades, "b_mid") if not trades.empty else 0.0,
        "h_body": _sum_or_none(trades, "h_body") if not trades.empty else 0.0,
        "w_mid": _sum_or_none(trades, "w_mid") if not trades.empty else 0.0,
        "h_wing": _sum_or_none(trades, "h_wing") if not trades.empty else 0.0,
        "w_pay": _sum_or_none(trades, "w_pay") if not trades.empty else 0.0,
        "p_body_cross": _sum_or_none(trades, "p_body_cross") if not trades.empty else 0.0,
        "p_fly_cross": _sum_or_none(trades, "p_fly_cross") if not trades.empty else 0.0,
        "c_body": _sum_or_none(trades, "c_body") if not trades.empty else 0.0,
        **{name: level.get(name) for name in RESIDUAL_FIELDS},
        "null_trade_ratio_counts": _null_ratio_counts(trades),
    }
    c_body = dollars["c_body"] if dollars["c_body"] is not None else float("nan")
    w_mid = dollars["w_mid"] if dollars["w_mid"] is not None else float("nan")
    h_wing = dollars["h_wing"] if dollars["h_wing"] is not None else float("nan")
    ratios = {
        "body_concession_over_body_midpoint_credit": {
            "numerator": "sum(h_body)",
            "denominator": "sum(c_body)",
            "value": _ratio(dollars["h_body"] if dollars["h_body"] is not None else float("nan"), c_body),
        },
        "wing_midpoint_premium_over_body_midpoint_credit": {
            "numerator": "sum(w_mid)",
            "denominator": "sum(c_body)",
            "value": _ratio(dollars["w_mid"] if dollars["w_mid"] is not None else float("nan"), c_body),
        },
        "wing_concession_over_body_midpoint_credit": {
            "numerator": "sum(h_wing)",
            "denominator": "sum(c_body)",
            "value": _ratio(dollars["h_wing"] if dollars["h_wing"] is not None else float("nan"), c_body),
        },
        "descriptive_wing_spread_percentage": {
            "numerator": "sum(h_wing)",
            "denominator": "sum(w_mid)",
            "value": _ratio(h_wing, w_mid),
            "label": "descriptive spread percentage, not a headline damage figure",
        },
        "null_trade_ratio_counts": dollars["null_trade_ratio_counts"],
        "aggregate_ratio_reason": _ratio_reason(c_body),
    }
    return dollars, ratios


def official_coverage_problems(trades: pd.DataFrame, dates: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if len(trades) != EXPECTED_DEVELOPMENT_TRADES:
        problems.append(f"development trades {len(trades)} != {EXPECTED_DEVELOPMENT_TRADES}")
    if len(dates) != EXPECTED_DEVELOPMENT_DATES:
        problems.append(f"development dates {len(dates)} != {EXPECTED_DEVELOPMENT_DATES}")
    if dates.empty:
        problems.append("development calendar is empty")
        return problems
    positive = int((dates["short_book_class"] == "verified_positive_short").sum())
    zero = dates.loc[dates["short_book_class"] == "verified_zero_short", "trade_date"].map(_as_date).tolist()
    if positive != EXPECTED_POSITIVE_DATES:
        problems.append(f"verified_positive_short dates {positive} != {EXPECTED_POSITIVE_DATES}")
    if zero != [ZERO_SHORT_DATE]:
        problems.append(f"verified zero-short dates {zero} != {[ZERO_SHORT_DATE]}")
    if not trades.empty and "input_ok" in trades.columns and not bool(trades["input_ok"].all()):
        problems.append("pairing or input failures remain in the development book")
    years = { _as_date(value).year for value in dates["trade_date"] }
    missing_years = [year for year in ANNUAL_YEARS if year not in years]
    if missing_years:
        problems.append(f"development calendar missing years {missing_years}")
    return problems


def evaluate_gates(
    *,
    trades: pd.DataFrame,
    dates: pd.DataFrame,
    annual: pd.DataFrame,
    issues: list[str],
    dollars: dict[str, Any],
    report: dict[str, Any],
    require_official_coverage: bool,
) -> list[GateResult]:
    gates: list[GateResult] = []
    coverage = official_coverage_problems(trades, dates) if require_official_coverage else []
    gates.append(
        GateResult(
            "provenance",
            not coverage,
            "development coverage matches the accepted D0 evidence"
            if not coverage
            else "; ".join(coverage[:6]),
        )
    )
    gates.append(
        GateResult(
            "inputs",
            not issues and (trades.empty or bool(trades["input_ok"].all())),
            "development inputs are finite and paired"
            if not issues and (trades.empty or bool(trades["input_ok"].all()))
            else "; ".join((issues or ["input_ok is false"])[:6]),
        )
    )
    official = float(dollars.get("development_official_pnl") or 0.0)
    for gate_id, fields in (
        ("body_cross", ("residual_body_vs_d0",)),
        ("wing_cross", ("residual_wing_vs_d0",)),
        ("identity", ("residual_identity_vs_official", "residual_identity_vs_legs")),
        ("midpoint", ("residual_mid_vs_d0",)),
    ):
        failed: list[str] = []
        if trades.empty or not bool(trades["input_ok"].all()):
            failed.append("incomplete development book")
        else:
            for _, row in trades.iterrows():
                reference = float(row["pnl_cross_official"])
                for name in fields:
                    value = row[name]
                    if value is None or not within_dollars(float(value), 0.0, reference):
                        failed.append(f"{row['trade_date']} {row['ticker']} {name}")
                        break
                if len(failed) >= 5:
                    break
            for _, row in dates.iterrows():
                reference = float(row.get("pnl_cross_official", 0.0) or 0.0) if "pnl_cross_official" in row else 0.0
                # date residuals already use date official via _level_residuals; tolerance uses date official sum of trades
                day_trades = trades.loc[trades["trade_date"].map(_as_date) == _as_date(row["trade_date"])]
                reference = float(day_trades["pnl_cross_official"].sum()) if not day_trades.empty else 0.0
                for name in fields:
                    value = row[name]
                    if value is None or not within_dollars(float(value), 0.0, reference):
                        failed.append(f"{row['trade_date']} {name}")
                        break
            for _, row in annual.iterrows():
                year_days = {
                    _as_date(value)
                    for value in dates.loc[dates["trade_date"].map(lambda item: _as_date(item).year) == int(row["year"]), "trade_date"]
                }
                year_trades = trades.loc[trades["trade_date"].map(_as_date).isin(year_days)]
                reference = float(year_trades["pnl_cross_official"].sum()) if not year_trades.empty else 0.0
                for name in fields:
                    value = row[name]
                    if value is None or not within_dollars(float(value), 0.0, reference):
                        failed.append(f"{int(row['year'])} {name}")
                        break
            for name in fields:
                value = dollars.get(name)
                if value is None or not within_dollars(float(value), 0.0, official):
                    failed.append(f"development {name}")
        gates.append(
            GateResult(
                gate_id,
                not failed,
                "reconciled to saved D0 dollars" if not failed else "; ".join(failed[:6]),
            )
        )
    zero = dates.loc[dates["short_book_class"] == "verified_zero_short"] if not dates.empty else dates
    calendar_ok = not dates.empty or not require_official_coverage
    calendar_problems: list[str] = []
    if require_official_coverage:
        if dates.empty or not (dates["trade_date"].map(_as_date) == ZERO_SHORT_DATE).any():
            calendar_problems.append("missing 2020-03-13")
        else:
            row = dates.loc[dates["trade_date"].map(_as_date) == ZERO_SHORT_DATE].iloc[0]
            if int(row["n_trades"]) != 0 or any(row[name] not in (0, 0.0) for name in ("b_mid", "h_body", "w_mid", "h_wing", "w_pay", "p_fly_cross")):
                calendar_problems.append("2020-03-13 is not a zero-dollar zero-short date")
    dropped = [item for item in issues if "missing from development calendar" in item or "date missing" in item]
    calendar_problems.extend(dropped)
    gates.append(
        GateResult(
            "calendar",
            calendar_ok and not calendar_problems and len(zero) == (1 if require_official_coverage else len(zero)),
            "development calendar retained, including verified zero-short dates"
            if not calendar_problems
            else "; ".join(calendar_problems[:6]),
        )
    )
    forbidden = [key for key in FORBIDDEN_REPORT_KEYS if key in report]
    gates.append(
        GateResult(
            "scope",
            not forbidden,
            "no later-period economic field" if not forbidden else f"forbidden keys {forbidden}",
        )
    )
    return gates


def decompose_development(
    matched: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    require_official_coverage: bool = False,
) -> DecompositionResult:
    issues = _scope_problems(matched, calendar)
    dev_trades, dev_calendar = _development_frames(matched, calendar)
    records: list[dict[str, Any]] = []
    gaps: list[float] = []
    for _, row in dev_trades.iterrows():
        record, row_issues, row_gaps = decompose_trade(row)
        records.append(record)
        issues.extend(row_issues)
        gaps.extend(row_gaps)
    trades = pd.DataFrame(records)
    dates = _build_dates(trades, dev_calendar, issues)
    annual = _build_annual(trades, dates)
    dollars, ratios = _aggregate_payloads(trades, dates)
    discrepant = [gap for gap in gaps if gap > QUOTE_DIAGNOSTIC_TOL]
    diagnostic = {
        "stored_vs_arithmetic_mid_count": len(discrepant),
        "stored_vs_arithmetic_mid_max_abs": max(gaps) if gaps else 0.0,
        "compared_legs": len(gaps),
        "threshold": QUOTE_DIAGNOSTIC_TOL,
        "blocks_run": False,
    }
    report = {
        "stored_vs_arithmetic_mid": diagnostic,
        "development_official_pnl": dollars.get("development_official_pnl"),
        "null_trade_ratio_counts": dollars.get("null_trade_ratio_counts"),
    }
    if any(key in report for key in FORBIDDEN_REPORT_KEYS):
        raise D1DecompositionError("report contains a forbidden field")
    gates = evaluate_gates(
        trades=trades,
        dates=dates,
        annual=annual,
        issues=issues,
        dollars=dollars,
        report=report,
        require_official_coverage=require_official_coverage,
    )
    verdict = readiness_verdict(gates)
    report["verdict"] = verdict
    return DecompositionResult(
        verdict=verdict,
        gates=gates,
        trades=trades,
        dates=dates,
        annual=annual,
        dollars=dollars,
        ratios=ratios,
        report=report,
        issues=issues,
    )


def _money(value: Any) -> str:
    if value is None or not _finite(value):
        return "undefined"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _ratio_text(payload: dict[str, Any]) -> str:
    value = payload.get("value")
    if value is None:
        return f"null ({payload.get('numerator')} / {payload.get('denominator')})"
    return f"{float(value):.4f} ({payload.get('numerator')} / {payload.get('denominator')})"


def _sign_clause(label: str, value: Any, official: float) -> str:
    if value is None or not _finite(value):
        return f"{label} is undefined"
    amount = float(value)
    if within_dollars(amount, 0.0, official):
        return f"{label} is within the development tolerance of zero ({_money(amount)})"
    if amount > 0:
        return f"{label} is positive ({_money(amount)})"
    return f"{label} is negative ({_money(amount)})"


def render_report_md(result: DecompositionResult) -> str:
    lines = [
        "# Sprint 009 D1 development decomposition",
        "",
        f"**Verdict:** `{result.verdict}`",
        "",
        "Fees = 0. Execution concession is not deducted twice. Midpoint fills are not claimed attainable. The primary-window −$146,279.85 is not this development result. Wing payout frequency and filter rules are not answered here.",
        "",
        "## Gates",
        "",
    ]
    for gate in result.gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"- {mark} `{gate.gate_id}`: {gate.detail}")
    diagnostic = result.report.get("stored_vs_arithmetic_mid", {})
    lines.extend(
        [
            "",
            "## Stored versus arithmetic midpoint",
            "",
            f"Discrepant legs: {diagnostic.get('stored_vs_arithmetic_mid_count', 0)}. Maximum absolute difference: {diagnostic.get('stored_vs_arithmetic_mid_max_abs', 0)}. This diagnostic does not block the run.",
            "",
        ]
    )
    if result.verdict != "READY":
        lines.append("No interpretation. A failed gate is a named blocker, not a partial economic result.")
        lines.append("")
        return "\n".join(lines)
    official = float(result.dollars.get("development_official_pnl") or 0.0)
    lines.extend(
        [
            "## D1-A — Body margin before and after execution cost",
            "",
            f"Development body midpoint P&L is {_money(result.dollars.get('b_mid'))}. Body execution concession is {_money(result.dollars.get('h_body'))}. Body cross P&L is {_money(result.dollars.get('p_body_cross'))}.",
            _sign_clause("Body midpoint margin", result.dollars.get("b_mid"), official) + ". " + _sign_clause("Body cross margin after execution concession", result.dollars.get("p_body_cross"), official) + ".",
            f"Body execution concession / body midpoint credit = {_ratio_text(result.ratios['body_concession_over_body_midpoint_credit'])}. This is a cost-burden ratio, not a return on capital, and not a trading decision.",
            "",
            "## D1-B — Wing midpoint premium versus execution concession",
            "",
            f"Wing midpoint premium is {_money(result.dollars.get('w_mid'))}. Wing execution concession is {_money(result.dollars.get('h_wing'))}. Gross wing expiry payout is {_money(result.dollars.get('w_pay'))} and is not net protection value.",
            f"Wing midpoint premium / body midpoint credit = {_ratio_text(result.ratios['wing_midpoint_premium_over_body_midpoint_credit'])}.",
            f"Wing execution concession / body midpoint credit = {_ratio_text(result.ratios['wing_concession_over_body_midpoint_credit'])}.",
            f"Descriptive wing spread percentage, beside the dollar concession and not a headline: {_ratio_text(result.ratios['descriptive_wing_spread_percentage'])}.",
            "",
            "## D1-C — Does the identity explain the development result?",
            "",
            f"Development official cross P&L is {_money(official)}. Identity residual versus that sum is {result.dollars.get('residual_identity_vs_official')}.",
            "The development cross result equals body midpoint P&L, minus body execution concession, minus wing midpoint premium, minus wing execution concession, plus gross wing expiry payout. That identity is a reconciliation, not a decision to drop wings or freeze a filter.",
            "",
            "Annual official cross P&L, descriptive only:",
            "",
        ]
    )
    if not result.annual.empty:
        for _, row in result.annual.iterrows():
            lines.append(
                f"- {int(row['year'])}: official cross {_money(row.get('p_fly_cross'))} on {int(row['n_trades'])} trades and {int(row['n_dates'])} dates."
            )
    lines.append("")
    return "\n".join(lines)
