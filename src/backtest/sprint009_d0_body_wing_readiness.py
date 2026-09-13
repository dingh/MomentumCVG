"""Sprint 009 D0 — matched short iron-fly readiness.

Read-only reconstruction of the official cross short book. Does not call
SurfaceRunner, run_d0_validation, or load_fill_primary_tables.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_EXECUTION_REPO_SHA,
    OFFICIAL_RUN_DIR,
    expected_cross_fill_price,
    expected_mid_fill_price,
    expected_run_output_path,
    sha256_file,
    verify_receipt_integrity,
)
from src.backtest.surface_decision_report import PRIMARY_END, PRIMARY_START

ACCEPTED_PRIMARY_SHORT_COUNT = 3322
ACCEPTED_PRIMARY_SHORT_PNL = -146_279.85
DEVELOPMENT_END = date(2023, 12, 31)
STRIKE_TOL = 1e-6
CASH_TOL = 1e-6
QUANTITY_TOL = 1e-6

TRADE_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "included_in_portfolio",
    "instrument_type",
    "expiry_date",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "quantity",
    "capital_at_risk_dollars",
    "pnl_total",
    "fill_label",
    "long_put_strike",
    "long_call_strike",
)
LEG_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "expiry_date",
    "option_type",
    "strike",
    "leg_index",
    "unit_quantity",
    "portfolio_quantity",
    "bid",
    "ask",
    "mid",
    "fill_price",
    "entry_cash_per_unit",
    "expiry_payoff_per_unit",
    "pnl_per_unit",
    "pnl_total_leg",
    "exit_spot",
    "included_in_portfolio",
    "fill_label",
)
FUNNEL_COLUMNS = (
    "trade_date",
    "n_included",
    "n_included_long",
    "n_included_short",
    "date_status",
    "date_reason",
)
DATE_STATUS_COLUMNS = ("trade_date", "status", "reason")
MID_TRADE_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "included_in_portfolio",
    "instrument_type",
    "quantity",
)
MID_LEG_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "expiry_date",
    "option_type",
    "strike",
    "leg_index",
    "unit_quantity",
    "bid",
    "ask",
    "mid",
    "exit_spot",
    "expiry_payoff_per_unit",
    "included_in_portfolio",
    "fill_label",
)

LEG_SPECS = (
    (0, "put_wing", "put", 1, "below"),
    (1, "body_put", "put", -1, "equal"),
    (2, "body_call", "call", -1, "equal"),
    (3, "call_wing", "call", 1, "above"),
)
LEG_FIELDS = (
    "option_type",
    "strike",
    "unit_quantity",
    "bid",
    "ask",
    "mid",
    "fill_price_cross",
    "entry_cash_per_unit",
    "expiry_payoff_per_unit",
    "exit_spot",
    "pnl_total_leg",
)
MATCHED_BASE_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "window_label",
    "instrument_type",
    "expiry_date",
    "body_strike",
    "Q",
    "quantity_cross_signed",
    "quantity_mid_abs",
    "entry_spot",
    "capital_at_risk_dollars",
    "pnl_cross_official",
    "pnl_body_cross",
    "pnl_wing_cross",
    "pnl_legs_sum",
    "pnl_mid_at_cross_q",
    "residual_body_wing_vs_legs",
    "residual_legs_vs_official",
    "pairing_ok",
    "pairing_reason",
)
MATCHED_COLUMNS = MATCHED_BASE_COLUMNS + tuple(
    f"{role}_{field_name}"
    for _idx, role, _otype, _qty, _side in LEG_SPECS
    for field_name in LEG_FIELDS
)
CALENDAR_COLUMNS = (
    "trade_date",
    "window_label",
    "short_book_class",
    "date_status",
    "date_reason",
    "funnel_date_status",
    "funnel_n_included_short",
    "n_included_short_trades",
    "blocker_reason",
)
FORBIDDEN_REPORT_KEYS = (
    "development_minus_later_pnl",
    "filter_result",
    "protection_summary",
    "development_vs_later",
)


class D0ReadinessError(Exception):
    """Raised when the helper cannot even write a blocked report."""


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    detail: str


@dataclass
class D0ReadinessResult:
    verdict: str
    gates: list[GateResult] = field(default_factory=list)
    matched: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict[str, Any] = field(default_factory=dict)
    inventory: dict[str, Any] = field(default_factory=dict)


def dollar_tolerance(official: float) -> float:
    magnitude = abs(float(official)) if math.isfinite(float(official)) else 0.0
    return max(0.01, 1e-9 * magnitude)


def within_dollars(left: float, right: float, official: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return abs(left - right) <= dollar_tolerance(official)


def window_label(trade_date: date) -> str:
    if trade_date < PRIMARY_START:
        return "pre_study"
    if trade_date <= DEVELOPMENT_END:
        return "development"
    if trade_date <= PRIMARY_END:
        return "later_period"
    return "unexpected"


def unsigned_intrinsic(option_type: str, strike: float, spot: float) -> float:
    kind = str(option_type).lower()
    if kind == "call":
        return max(float(spot) - float(strike), 0.0)
    if kind == "put":
        return max(float(strike) - float(spot), 0.0)
    raise ValueError(f"option_type must be call or put, got {option_type!r}")


def signed_entry_cash(fill_price: float, unit_quantity: int) -> float:
    magnitude = abs(int(unit_quantity))
    if unit_quantity > 0:
        return float(fill_price) * magnitude
    if unit_quantity < 0:
        return -float(fill_price) * magnitude
    raise ValueError("unit_quantity must be non-zero")


def reconstructed_leg_economics(
    *,
    option_type: str,
    strike: float,
    unit_quantity: int,
    bid: float,
    ask: float,
    exit_spot: float,
    quantity_magnitude: float,
) -> dict[str, float]:
    """Independent cash, settlement, and dollar P&L. Does not read logged P&L."""
    fill_price = expected_cross_fill_price(bid, ask, unit_quantity)
    mid_fill = expected_mid_fill_price(bid, ask, unit_quantity)
    entry_cash = signed_entry_cash(fill_price, unit_quantity)
    mid_cash = signed_entry_cash(mid_fill, unit_quantity)
    payoff = unsigned_intrinsic(option_type, strike, exit_spot) * int(unit_quantity)
    pnl_per_unit = payoff - entry_cash
    pnl_total = float(quantity_magnitude) * pnl_per_unit
    mid_pnl_total = float(quantity_magnitude) * (payoff - mid_cash)
    return {
        "fill_price": fill_price,
        "entry_cash_per_unit": entry_cash,
        "expiry_payoff_per_unit": payoff,
        "pnl_per_unit": pnl_per_unit,
        "pnl_total_leg": pnl_total,
        "pnl_mid_total_leg": mid_pnl_total,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].map(_as_date)
    if "expiry_date" in out.columns:
        out["expiry_date"] = out["expiry_date"].map(
            lambda value: _as_date(value) if pd.notna(value) else value
        )
    return out


def _trade_key(row: pd.Series) -> tuple[date, str, str]:
    return (_as_date(row["trade_date"]), str(row["ticker"]), str(row["direction"]))


def _leg_identity(row: pd.Series) -> tuple[Any, ...]:
    return (
        _as_date(row["expiry_date"]),
        str(row["option_type"]).lower(),
        round(float(row["strike"]), 6),
        int(row["leg_index"]),
    )


def _load_parquet(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    actual = pq.ParquetFile(path).schema_arrow.names
    missing = [name for name in columns if name not in actual]
    if missing:
        raise D0ReadinessError(f"{path.name} missing columns: {missing}")
    frame = pq.read_table(path, columns=list(columns)).to_pandas()
    return _prepare_dates(frame)


def _included_short_iron_flies(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.iloc[0:0].copy()
    mask = (
        (trades["included_in_portfolio"] == True)  # noqa: E712
        & (trades["direction"].astype(str) == "short")
        & (trades["instrument_type"].astype(str) == "iron_fly")
    )
    if "fill_label" in trades.columns:
        mask = mask & (trades["fill_label"].astype(str) == "cross")
    return trades.loc[mask].copy()


def _included_short_iron_flies_mid(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.iloc[0:0].copy()
    mask = (
        (trades["included_in_portfolio"] == True)  # noqa: E712
        & (trades["direction"].astype(str) == "short")
        & (trades["instrument_type"].astype(str) == "iron_fly")
    )
    return trades.loc[mask].copy()


def _index_legs(legs: pd.DataFrame) -> dict[tuple[date, str, str], pd.DataFrame]:
    """Keep every leg for the trade key. Extra rows are blockers, not drops."""
    grouped: dict[tuple[date, str, str], list[pd.Series]] = {}
    if legs.empty:
        return {}
    for _, row in legs.iterrows():
        grouped.setdefault(_trade_key(row), []).append(row)
    return {key: pd.DataFrame(rows) for key, rows in grouped.items()}


def _float_close(left: Any, right: Any, tol: float) -> bool:
    if not _finite(left) or not _finite(right):
        return False
    return abs(float(left) - float(right)) <= tol


def _field_issue(key: tuple[date, str, str], field: str, reason: str, *, leg_index: int | None = None) -> str:
    where = field if leg_index is None else f"{field} leg {leg_index}"
    return f"{key}: {where}: {reason}"


def _require_finite(
    key: tuple[date, str, str],
    field: str,
    value: Any,
    issues: list[str],
    *,
    leg_index: int | None = None,
) -> float | None:
    if _finite(value):
        return float(value)
    issues.append(_field_issue(key, field, "missing or non-finite", leg_index=leg_index))
    return None


def _short_quantity_problems(key: tuple[date, str, str], quantity: Any) -> list[str]:
    """Accepted short quantity is negative. Magnitude must be positive. Do not repair the sign."""
    if not _finite(quantity):
        return [_field_issue(key, "quantity", "missing or non-finite")]
    signed = float(quantity)
    problems: list[str] = []
    if not signed < 0:
        problems.append(_field_issue(key, "quantity", "sign is not negative"))
    if abs(signed) <= QUANTITY_TOL:
        problems.append(_field_issue(key, "quantity", "magnitude is not positive"))
    return problems


def _leg_identity_list(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    if frame.empty:
        return []
    return [_leg_identity(row) for _, row in frame.iterrows()]


def _fill_leg_cardinality_problems(frame: pd.DataFrame, label: str) -> list[str]:
    """Uniqueness and cardinality before any set or dict of leg keys is built."""
    identities = _leg_identity_list(frame)
    problems: list[str] = []
    if len(identities) != len(set(identities)):
        duplicates = sorted({item for item in identities if identities.count(item) > 1})
        problems.append(f"{label} duplicate leg keys {duplicates}")
    if len(frame) != 4:
        problems.append(f"{label} expected 4 legs, found {len(frame)}")
    return problems


def _nonnegative_integer_count(value: Any) -> tuple[int | None, str | None]:
    """Accept a finite nonnegative integer. Do not truncate a fractional count."""
    if value is None or pd.isna(value):
        return None, "null n_included_short"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "n_included_short is not a finite integer"
    if not math.isfinite(number):
        return None, "n_included_short is not a finite integer"
    if number < 0:
        return None, "n_included_short is negative"
    if not number.is_integer():
        return None, "fractional n_included_short"
    return int(number), None


def pair_selected_shorts(
    cross_trades: pd.DataFrame,
    cross_legs: pd.DataFrame,
    mid_trades: pd.DataFrame,
    mid_legs: pd.DataFrame,
) -> dict[str, Any]:
    """Set-difference pairing. Does not drop a cross row missing from mid."""
    cross_book = _included_short_iron_flies(cross_trades)
    mid_book = _included_short_iron_flies_mid(mid_trades)
    cross_keys = [_trade_key(row) for _, row in cross_book.iterrows()]
    mid_keys = [_trade_key(row) for _, row in mid_book.iterrows()]
    cross_set = set(cross_keys)
    mid_set = set(mid_keys)
    missing_from_mid = sorted(cross_set - mid_set)
    unmatched_mid = sorted(mid_set - cross_set)

    cross_leg_index = _index_legs(cross_legs)
    mid_leg_index = _index_legs(mid_legs)
    mid_qty = {
        _trade_key(row): abs(float(row["quantity"])) if _finite(row["quantity"]) else None
        for _, row in mid_book.iterrows()
    }
    pair_reasons: dict[tuple[date, str, str], str] = {}
    for key in sorted(cross_set & mid_set):
        reasons: list[str] = []
        cross_leg_frame = cross_leg_index.get(key, pd.DataFrame())
        mid_leg_frame = mid_leg_index.get(key, pd.DataFrame())
        reasons.extend(_fill_leg_cardinality_problems(cross_leg_frame, "cross"))
        reasons.extend(_fill_leg_cardinality_problems(mid_leg_frame, "midpoint"))
        if reasons:
            pair_reasons[key] = "; ".join(reasons)
            continue
        cross_ids = {_leg_identity(row) for _, row in cross_leg_frame.iterrows()}
        mid_ids = {_leg_identity(row) for _, row in mid_leg_frame.iterrows()}
        if cross_ids != mid_ids:
            reasons.append(
                "leg keys differ missing_in_mid="
                f"{sorted(cross_ids - mid_ids)} extra_in_mid={sorted(mid_ids - cross_ids)}"
            )
        else:
            mid_by_id = {_leg_identity(row): row for _, row in mid_leg_frame.iterrows()}
            for _, row in cross_leg_frame.iterrows():
                other = mid_by_id[_leg_identity(row)]
                for column in ("bid", "ask", "mid", "exit_spot", "expiry_payoff_per_unit"):
                    if not _float_close(row[column], other[column], CASH_TOL):
                        reasons.append(f"{column} mismatch leg_index={int(row['leg_index'])}")
                if int(row["unit_quantity"]) != int(other["unit_quantity"]):
                    reasons.append(f"unit_quantity mismatch leg_index={int(row['leg_index'])}")
        if reasons:
            pair_reasons[key] = "; ".join(reasons)
    for key in missing_from_mid:
        pair_reasons[key] = "missing from midpoint included short iron flies"
    return {
        "missing_from_mid": missing_from_mid,
        "unmatched_mid": unmatched_mid,
        "pair_reasons": pair_reasons,
        "mid_quantity_abs": mid_qty,
        "duplicate_cross_keys": sorted({key for key in cross_keys if cross_keys.count(key) > 1}),
        "duplicate_mid_keys": sorted({key for key in mid_keys if mid_keys.count(key) > 1}),
    }


def _role_for_index(leg_index: int) -> str | None:
    for idx, role, _otype, _qty, _side in LEG_SPECS:
        if idx == leg_index:
            return role
    return None


def _structure_problems(trade: pd.Series, legs: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if legs.empty or len(legs) != 4:
        return [f"expected 4 legs, found {0 if legs.empty else len(legs)}"]
    if legs.duplicated(subset=["leg_index", "option_type", "strike", "expiry_date"]).any():
        problems.append("duplicate leg key")
    if set(int(v) for v in legs["leg_index"]) != {0, 1, 2, 3}:
        problems.append(f"leg_index set {sorted(set(int(v) for v in legs['leg_index']))}")
    expiry = trade["expiry_date"]
    body = float(trade["body_strike"])
    by_index = {int(row["leg_index"]): row for _, row in legs.iterrows()}
    for idx, _role, option_type, unit_quantity, side in LEG_SPECS:
        row = by_index.get(idx)
        if row is None:
            problems.append(f"missing leg_index {idx}")
            continue
        if str(row["option_type"]).lower() != option_type:
            problems.append(f"leg {idx} option_type {row['option_type']}")
        if int(row["unit_quantity"]) != unit_quantity:
            problems.append(f"leg {idx} unit_quantity {row['unit_quantity']}")
        if _as_date(row["expiry_date"]) != _as_date(expiry):
            problems.append(f"leg {idx} expiry mismatch")
        strike = float(row["strike"])
        if side == "equal" and abs(strike - body) > STRIKE_TOL:
            problems.append(f"leg {idx} strike is not body_strike")
        if side == "below" and not strike < body - STRIKE_TOL:
            problems.append(f"leg {idx} put wing is not strictly below body")
        if side == "above" and not strike > body + STRIKE_TOL:
            problems.append(f"leg {idx} call wing is not strictly above body")
    put_body = by_index.get(1)
    call_body = by_index.get(2)
    if put_body is not None and call_body is not None:
        if abs(float(put_body["strike"]) - float(call_body["strike"])) > STRIKE_TOL:
            problems.append("body strikes differ")
    if _finite(trade.get("long_put_strike")) and 0 in by_index:
        if abs(float(trade["long_put_strike"]) - float(by_index[0]["strike"])) > STRIKE_TOL:
            problems.append("long_put_strike diagnostic disagrees with leg")
    if _finite(trade.get("long_call_strike")) and 3 in by_index:
        if abs(float(trade["long_call_strike"]) - float(by_index[3]["strike"])) > STRIKE_TOL:
            problems.append("long_call_strike diagnostic disagrees with leg")
    return problems


def build_matched_rows(
    cross_trades: pd.DataFrame,
    cross_legs: pd.DataFrame,
    mid_trades: pd.DataFrame,
    mid_legs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    pairing = pair_selected_shorts(cross_trades, cross_legs, mid_trades, mid_legs)
    pairing["calendar_incomplete_dates"] = set()
    book = _included_short_iron_flies(cross_trades)
    leg_index = _index_legs(cross_legs)
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if pairing["duplicate_cross_keys"]:
        issues.append(f"duplicate cross keys: {pairing['duplicate_cross_keys'][:5]}")
    if pairing["duplicate_mid_keys"]:
        issues.append(f"duplicate mid keys: {pairing['duplicate_mid_keys'][:5]}")

    for _, trade in book.iterrows():
        key = _trade_key(trade)
        legs = leg_index.get(key, pd.DataFrame())
        quantity_problems = _short_quantity_problems(key, trade["quantity"])
        q_signed = float(trade["quantity"]) if _finite(trade["quantity"]) else float("nan")
        q_mag = abs(q_signed) if _finite(q_signed) and abs(q_signed) > QUANTITY_TOL else float("nan")
        structure = _structure_problems(trade, legs)
        if structure:
            issues.append(f"{key}: " + "; ".join(structure))
        issues.extend(quantity_problems)
        entry_spot = _require_finite(key, "entry_spot", trade["entry_spot"], issues)
        capital = _require_finite(key, "capital_at_risk_dollars", trade["capital_at_risk_dollars"], issues)
        trade_exit = _require_finite(key, "exit_spot", trade["exit_spot"], issues)
        record: dict[str, Any] = {name: None for name in MATCHED_COLUMNS}
        record.update(
            {
                "trade_date": key[0],
                "ticker": key[1],
                "direction": "short",
                "window_label": window_label(key[0]),
                "instrument_type": "iron_fly",
                "expiry_date": _as_date(trade["expiry_date"]) if pd.notna(trade["expiry_date"]) else None,
                "body_strike": float(trade["body_strike"]) if _finite(trade["body_strike"]) else None,
                "Q": q_mag if _finite(q_mag) else None,
                "quantity_cross_signed": q_signed if _finite(q_signed) else None,
                "quantity_mid_abs": pairing["mid_quantity_abs"].get(key),
                "entry_spot": entry_spot,
                "capital_at_risk_dollars": capital,
                "pnl_cross_official": float(trade["pnl_total"]) if _finite(trade["pnl_total"]) else None,
                "pairing_ok": key not in pairing["pair_reasons"] and key not in pairing["duplicate_cross_keys"],
                "pairing_reason": pairing["pair_reasons"].get(key, ""),
            }
        )
        if key in pairing["duplicate_cross_keys"]:
            record["pairing_ok"] = False
            extra = "duplicate cross trade key"
            record["pairing_reason"] = (
                extra if not record["pairing_reason"] else record["pairing_reason"] + "; " + extra
            )
        recon_body = 0.0
        recon_wing = 0.0
        recon_sum = 0.0
        mid_sum = 0.0
        have_all = len(legs) == 4 and _finite(q_mag) and not quantity_problems
        quote_ok = True
        row_inconsistent = bool(structure) or bool(quantity_problems) or entry_spot is None or capital is None or trade_exit is None
        by_index = {} if legs.empty else {int(row["leg_index"]): row for _, row in legs.iterrows()}
        for idx, role, _otype, _qty, _side in LEG_SPECS:
            row = by_index.get(idx)
            if row is None:
                have_all = False
                continue
            prefix = role
            record[f"{prefix}_option_type"] = str(row["option_type"]).lower()
            record[f"{prefix}_strike"] = float(row["strike"])
            record[f"{prefix}_unit_quantity"] = int(row["unit_quantity"])
            record[f"{prefix}_bid"] = float(row["bid"]) if _finite(row["bid"]) else None
            record[f"{prefix}_ask"] = float(row["ask"]) if _finite(row["ask"]) else None
            record[f"{prefix}_mid"] = float(row["mid"]) if _finite(row["mid"]) else None
            record[f"{prefix}_exit_spot"] = float(row["exit_spot"]) if _finite(row["exit_spot"]) else None
            logged_leg_pnl = _require_finite(key, "pnl_total_leg", row["pnl_total_leg"], issues, leg_index=idx)
            logged_portfolio = _require_finite(
                key, "portfolio_quantity", row["portfolio_quantity"], issues, leg_index=idx
            )
            if logged_leg_pnl is None or logged_portfolio is None:
                row_inconsistent = True
            leg_exit = _require_finite(key, "exit_spot", row["exit_spot"], issues, leg_index=idx)
            if leg_exit is None:
                row_inconsistent = True
            elif trade_exit is not None and not _float_close(leg_exit, trade_exit, CASH_TOL):
                row_inconsistent = True
                issues.append(
                    _field_issue(
                        key,
                        "exit_spot",
                        "disagrees with trade settlement spot",
                        leg_index=idx,
                    )
                )
            bid, ask = row["bid"], row["ask"]
            if not _finite(bid) or not _finite(ask) or float(ask) < float(bid):
                quote_ok = False
                row_inconsistent = True
                issues.append(f"{key}: invalid quote leg {idx}")
                continue
            if not _finite(q_mag) or not _finite(row["exit_spot"]):
                quote_ok = False
                row_inconsistent = True
                continue
            rebuilt = reconstructed_leg_economics(
                option_type=str(row["option_type"]),
                strike=float(row["strike"]),
                unit_quantity=int(row["unit_quantity"]),
                bid=float(bid),
                ask=float(ask),
                exit_spot=float(row["exit_spot"]),
                quantity_magnitude=float(q_mag),
            )
            record[f"{prefix}_fill_price_cross"] = rebuilt["fill_price"]
            record[f"{prefix}_entry_cash_per_unit"] = rebuilt["entry_cash_per_unit"]
            record[f"{prefix}_expiry_payoff_per_unit"] = rebuilt["expiry_payoff_per_unit"]
            record[f"{prefix}_pnl_total_leg"] = rebuilt["pnl_total_leg"]
            recon_sum += rebuilt["pnl_total_leg"]
            if idx in (1, 2):
                recon_body += rebuilt["pnl_total_leg"]
            if idx in (0, 3):
                recon_wing += rebuilt["pnl_total_leg"]
            mid_sum += rebuilt["pnl_mid_total_leg"]
            if not _float_close(row["fill_price"], rebuilt["fill_price"], CASH_TOL):
                row_inconsistent = True
                issues.append(f"{key}: fill_price mismatch leg {idx}")
            if not _float_close(row["entry_cash_per_unit"], rebuilt["entry_cash_per_unit"], CASH_TOL):
                row_inconsistent = True
                issues.append(f"{key}: entry cash mismatch leg {idx}")
            if not _float_close(row["expiry_payoff_per_unit"], rebuilt["expiry_payoff_per_unit"], CASH_TOL):
                row_inconsistent = True
                issues.append(f"{key}: settlement mismatch leg {idx}")
            if not _float_close(row["pnl_per_unit"], rebuilt["pnl_per_unit"], CASH_TOL):
                row_inconsistent = True
                issues.append(f"{key}: pnl_per_unit mismatch leg {idx}")
            if logged_leg_pnl is None:
                row_inconsistent = True
            elif not _float_close(
                logged_leg_pnl, rebuilt["pnl_total_leg"], dollar_tolerance(logged_leg_pnl)
            ):
                row_inconsistent = True
                issues.append(
                    _field_issue(
                        key,
                        "pnl_total_leg",
                        "logged value disagrees with independent reconstruction",
                        leg_index=idx,
                    )
                )
            if logged_portfolio is None or not _finite(q_mag):
                row_inconsistent = True
            else:
                expected_port = float(q_mag) * int(row["unit_quantity"])
                if abs(logged_portfolio - expected_port) > max(QUANTITY_TOL, 1e-9 * abs(expected_port)):
                    row_inconsistent = True
                    issues.append(
                        _field_issue(
                            key,
                            "portfolio_quantity",
                            "is not Q times unit_quantity",
                            leg_index=idx,
                        )
                    )
        official = record["pnl_cross_official"]
        if have_all and quote_ok and official is not None:
            record["pnl_body_cross"] = recon_body
            record["pnl_wing_cross"] = recon_wing
            record["pnl_legs_sum"] = recon_sum
            record["pnl_mid_at_cross_q"] = mid_sum
            record["residual_body_wing_vs_legs"] = recon_body + recon_wing - recon_sum
            record["residual_legs_vs_official"] = recon_sum - official
            if not within_dollars(recon_body + recon_wing, recon_sum, official):
                row_inconsistent = True
                issues.append(f"{key}: body+wing does not equal four-leg sum")
            if not within_dollars(recon_sum, official, official):
                row_inconsistent = True
                issues.append(f"{key}: four-leg sum does not equal official P&L")
        else:
            row_inconsistent = True
            issues.append(f"{key}: reconstruction incomplete")
        if row_inconsistent:
            pairing["calendar_incomplete_dates"].add(key[0])
        rows.append(record)
    matched = pd.DataFrame(rows, columns=list(MATCHED_COLUMNS))
    return matched, pairing, issues


def classify_short_calendar(
    date_status: pd.DataFrame,
    funnel: pd.DataFrame,
    matched: pd.DataFrame,
    incomplete_dates: set[date] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    status = _prepare_dates(date_status)
    funnel_frame = _prepare_dates(funnel)
    status_dates = [_as_date(value) for value in status["trade_date"]]
    funnel_dates = [_as_date(value) for value in funnel_frame["trade_date"]] if not funnel_frame.empty else []
    if len(status_dates) != len(set(status_dates)):
        issues.append("duplicate date_status rows")
    if len(funnel_dates) != len(set(funnel_dates)):
        issues.append("duplicate funnel rows")
    status_only = set(status_dates)
    funnel_only = set(funnel_dates) - status_only
    if funnel_only:
        issues.append(f"funnel dates missing from date_status: {sorted(funnel_only)[:5]}")
    trade_dates = set()
    counts: dict[date, int] = {}
    incomplete: set[date] = set(incomplete_dates or ())
    if not matched.empty:
        for _, row in matched.iterrows():
            day = _as_date(row["trade_date"])
            trade_dates.add(day)
            counts[day] = counts.get(day, 0) + 1
            if row["pnl_legs_sum"] is None:
                incomplete.add(day)
        orphan_trades = trade_dates - status_only
        if orphan_trades:
            issues.append(f"short trades missing from date_status: {sorted(orphan_trades)[:5]}")
    funnel_by_date = {
        _as_date(row["trade_date"]): row for _, row in funnel_frame.iterrows()
    }
    rows: list[dict[str, Any]] = []
    all_dates = sorted(status_only | funnel_only | trade_dates)
    for day in all_dates:
        status_row = status[status["trade_date"].map(_as_date) == day]
        funnel_row = funnel_by_date.get(day)
        whole = str(status_row.iloc[0]["status"]) if not status_row.empty else None
        reason = str(status_row.iloc[0]["reason"]) if not status_row.empty else None
        funnel_status = str(funnel_row["date_status"]) if funnel_row is not None else None
        n_short = funnel_row["n_included_short"] if funnel_row is not None else None
        n_trades = counts.get(day, 0)
        blocker = ""
        count_value: int | None = None
        count_error: str | None = None
        if funnel_row is not None:
            count_value, count_error = _nonnegative_integer_count(n_short)
        if day not in status_only:
            short_class = "blocked"
            blocker = "date missing from date_status"
        elif whole == "failed":
            short_class = "blocked"
            blocker = "failed date_status"
        elif funnel_row is None:
            short_class = "blocked"
            blocker = "missing funnel row"
        elif count_error is not None:
            short_class = "blocked"
            blocker = count_error
        elif funnel_status is not None and whole is not None and funnel_status != whole:
            short_class = "blocked"
            blocker = "funnel date_status disagrees with date_status"
        elif whole == "valid_no_trade" and ((count_value or 0) > 0 or n_trades > 0):
            short_class = "blocked"
            blocker = "valid_no_trade date contains included positions"
        elif count_value == 0 and n_trades == 0 and whole in {"traded", "valid_no_trade"}:
            short_class = "verified_zero_short"
        elif (
            count_value is not None
            and count_value > 0
            and count_value == n_trades
            and day not in incomplete
            and whole == "traded"
        ):
            short_class = "verified_positive_short"
        elif (
            count_value is not None
            and count_value > 0
            and count_value == n_trades
            and day not in incomplete
            and whole != "traded"
        ):
            short_class = "blocked"
            blocker = "positive short book status is not traded"
        else:
            short_class = "blocked"
            blocker = "short count, legs, or settlement inconsistent"
        if window_label(day) == "unexpected" and short_class != "blocked":
            short_class = "blocked"
            blocker = "date after primary end"
        if short_class == "blocked" and blocker:
            issues.append(f"{day}: {blocker}")
        rows.append(
            {
                "trade_date": day,
                "window_label": window_label(day),
                "short_book_class": short_class,
                "date_status": whole,
                "date_reason": reason,
                "funnel_date_status": funnel_status,
                "funnel_n_included_short": count_value,
                "n_included_short_trades": n_trades,
                "blocker_reason": blocker,
            }
        )
    calendar = pd.DataFrame(rows, columns=list(CALENDAR_COLUMNS))
    return calendar, issues


def _primary_mask(frame: pd.DataFrame) -> pd.Series:
    days = frame["trade_date"].map(_as_date)
    return (days >= PRIMARY_START) & (days <= PRIMARY_END)


def readiness_verdict(gates: list[GateResult]) -> str:
    """READY only when every supplied gate passed. An empty list is not READY."""
    if not gates or any(not gate.passed for gate in gates):
        return "BLOCKED"
    return "READY"


def evaluate_panel_gates(
    *,
    reconstruction_issues: list[str],
    pairing: dict[str, Any],
    matched: pd.DataFrame,
    calendar: pd.DataFrame,
    calendar_issues: list[str],
) -> list[GateResult]:
    """Reconstruction, pairing, reconciliation, and calendar. Same objects the official run uses."""
    return [
        gate
        for gate in evaluate_gates(
            receipt_ok=True,
            receipt_detail="panel",
            reconstruction_issues=reconstruction_issues,
            pairing=pairing,
            matched=matched,
            calendar=calendar,
            calendar_issues=calendar_issues,
            decision_short=None,
        )
        if gate.gate_id in {"reconstruction", "pairing", "reconciliation", "calendar"}
    ]


def assess_panel(
    cross_trades: pd.DataFrame,
    cross_legs: pd.DataFrame,
    mid_trades: pd.DataFrame,
    mid_legs: pd.DataFrame,
    date_status: pd.DataFrame,
    funnel: pd.DataFrame,
) -> D0ReadinessResult:
    """Propagate panel gates to READY or BLOCKED. Does not replace the official primary anchor."""
    matched, pairing, reconstruction_issues = build_matched_rows(
        cross_trades, cross_legs, mid_trades, mid_legs
    )
    calendar, calendar_issues = classify_short_calendar(
        date_status,
        funnel,
        matched,
        incomplete_dates=pairing.get("calendar_incomplete_dates"),
    )
    gates = evaluate_panel_gates(
        reconstruction_issues=reconstruction_issues,
        pairing=pairing,
        matched=matched,
        calendar=calendar,
        calendar_issues=calendar_issues,
    )
    verdict = readiness_verdict(gates)
    return D0ReadinessResult(
        verdict=verdict,
        gates=gates,
        matched=matched,
        calendar=calendar,
        report={"verdict": verdict, "pairing_unmatched_mid": pairing.get("unmatched_mid", [])},
    )


def _report_has_forbidden(report: dict[str, Any]) -> bool:
    return any(key in report for key in FORBIDDEN_REPORT_KEYS)


def evaluate_gates(
    *,
    receipt_ok: bool,
    receipt_detail: str,
    reconstruction_issues: list[str],
    pairing: dict[str, Any],
    matched: pd.DataFrame,
    calendar: pd.DataFrame,
    calendar_issues: list[str],
    decision_short: dict[str, Any] | None,
) -> list[GateResult]:
    gates: list[GateResult] = []
    gates.append(GateResult("receipt", receipt_ok, receipt_detail))
    gates.append(
        GateResult(
            "reconstruction",
            not reconstruction_issues,
            "independent cash, settlement, and structure checks passed"
            if not reconstruction_issues
            else "; ".join(reconstruction_issues[:8]),
        )
    )
    pairing_problems = list(pairing.get("missing_from_mid") or []) + list(pairing.get("unmatched_mid") or [])
    pairing_problems += list(pairing.get("pair_reasons") or {})
    gates.append(
        GateResult(
            "pairing",
            not pairing.get("missing_from_mid")
            and not pairing.get("unmatched_mid")
            and not pairing.get("pair_reasons")
            and not pairing.get("duplicate_cross_keys")
            and not pairing.get("duplicate_mid_keys"),
            "selected short keys, legs, quotes, and settlements pair"
            if not pairing_problems
            else (
                f"missing_mid={len(pairing.get('missing_from_mid') or [])} "
                f"unmatched_mid={len(pairing.get('unmatched_mid') or [])} "
                f"pair_failures={len(pairing.get('pair_reasons') or {})}"
            ),
        )
    )
    residual_fail = 0
    if not matched.empty:
        for _, row in matched.iterrows():
            official = row["pnl_cross_official"]
            if official is None or row["residual_legs_vs_official"] is None:
                residual_fail += 1
                continue
            if not within_dollars(0.0, float(row["residual_body_wing_vs_legs"]), float(official)):
                residual_fail += 1
            elif not within_dollars(0.0, float(row["residual_legs_vs_official"]), float(official)):
                residual_fail += 1
    gates.append(
        GateResult(
            "reconciliation",
            residual_fail == 0 and not matched.empty,
            f"trade residuals failed={residual_fail}",
        )
    )
    primary = matched.loc[_primary_mask(matched)] if not matched.empty else matched
    count = int(len(primary))
    pnl = float(primary["pnl_cross_official"].sum()) if count else float("nan")
    residual_sum = (
        float(primary["residual_legs_vs_official"].sum())
        if count and primary["residual_legs_vs_official"].notna().all()
        else float("nan")
    )
    reconstructed = (
        float(primary["pnl_legs_sum"].sum())
        if count and primary["pnl_legs_sum"].notna().all()
        else float("nan")
    )
    count_ok = count == ACCEPTED_PRIMARY_SHORT_COUNT
    pnl_ok = within_dollars(pnl, ACCEPTED_PRIMARY_SHORT_PNL, ACCEPTED_PRIMARY_SHORT_PNL)
    reconstructed_ok = within_dollars(reconstructed, ACCEPTED_PRIMARY_SHORT_PNL, ACCEPTED_PRIMARY_SHORT_PNL)
    aggregate_ok = within_dollars(residual_sum, 0.0, ACCEPTED_PRIMARY_SHORT_PNL)
    decision_ok = False
    decision_detail = "decision report short block missing"
    if decision_short is not None:
        decision_count = int(decision_short.get("n_traded_rows", -1))
        decision_pnl = float(decision_short.get("pnl_total", float("nan")))
        decision_ok = (
            decision_count == ACCEPTED_PRIMARY_SHORT_COUNT
            and within_dollars(decision_pnl, ACCEPTED_PRIMARY_SHORT_PNL, ACCEPTED_PRIMARY_SHORT_PNL)
        )
        decision_detail = f"decision n={decision_count} pnl={decision_pnl}"
    gates.append(
        GateResult(
            "primary_anchor",
            count_ok and pnl_ok and reconstructed_ok and aggregate_ok and decision_ok,
            f"primary iron flies n={count} official_pnl={pnl} reconstructed_pnl={reconstructed} residual_sum={residual_sum}; closeout n={ACCEPTED_PRIMARY_SHORT_COUNT} pnl={ACCEPTED_PRIMARY_SHORT_PNL}; {decision_detail}",
        )
    )
    blocked_dates = 0 if calendar.empty else int((calendar["short_book_class"] == "blocked").sum())
    gates.append(
        GateResult(
            "calendar",
            blocked_dates == 0 and not calendar_issues,
            f"blocked_dates={blocked_dates} issues={len(calendar_issues)}",
        )
    )
    return gates


def _progress(stage: str) -> None:
    print(f"stage {stage}", flush=True)


def run_d0_readiness(run_dir: Path | None = None) -> D0ReadinessResult:
    official = Path(run_dir) if run_dir is not None else OFFICIAL_RUN_DIR
    _progress("inventory")
    receipt_path = official / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_gate = verify_receipt_integrity(receipt, official)
    inventory = {
        "official_run_dir": str(official),
        "execution_sha": OFFICIAL_EXECUTION_REPO_SHA,
        "receipt_sha256": sha256_file(receipt_path),
        "files": [],
    }
    required = {
        "trade_log_cross": (expected_run_output_path(official, "trade_log", "cross"), TRADE_COLUMNS),
        "leg_log_cross": (expected_run_output_path(official, "leg_log", "cross"), LEG_COLUMNS),
        "trade_log_mid": (expected_run_output_path(official, "trade_log", "mid"), MID_TRADE_COLUMNS),
        "leg_log_mid": (expected_run_output_path(official, "leg_log", "mid"), MID_LEG_COLUMNS),
        "funnel_summary_cross": (expected_run_output_path(official, "funnel_summary", "cross"), FUNNEL_COLUMNS),
        "date_status_cross": (expected_run_output_path(official, "date_status", "cross"), DATE_STATUS_COLUMNS),
    }
    column_failures: list[str] = []
    for role, (path, columns) in required.items():
        names = pq.ParquetFile(path).schema_arrow.names if path.exists() else []
        missing = [name for name in columns if name not in names]
        if missing:
            column_failures.append(f"{role} missing {missing}")
        inventory["files"].append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
                "columns": names,
            }
        )
    if column_failures or not receipt_gate.passed:
        report = {
            "verdict": "BLOCKED",
            "column_failures": column_failures,
            "pairing_unmatched_mid": [],
            "note": "input gap; baseline was not rerun",
        }
        return D0ReadinessResult(
            verdict="BLOCKED",
            gates=[receipt_gate, GateResult("columns", not column_failures, "; ".join(column_failures) or "columns present")],
            report=report,
            inventory=inventory,
        )

    cross_trades = _load_parquet(required["trade_log_cross"][0], TRADE_COLUMNS)
    cross_legs = _load_parquet(required["leg_log_cross"][0], LEG_COLUMNS)
    mid_trades = _load_parquet(required["trade_log_mid"][0], MID_TRADE_COLUMNS)
    mid_legs = _load_parquet(required["leg_log_mid"][0], MID_LEG_COLUMNS)
    funnel = _load_parquet(required["funnel_summary_cross"][0], FUNNEL_COLUMNS)
    date_status = _load_parquet(required["date_status_cross"][0], DATE_STATUS_COLUMNS)

    _progress("pairing")
    _progress("reconstruction")
    matched, pairing, reconstruction_issues = build_matched_rows(
        cross_trades, cross_legs, mid_trades, mid_legs
    )
    _progress("calendar")
    calendar, calendar_issues = classify_short_calendar(
        date_status,
        funnel,
        matched,
        incomplete_dates=pairing.get("calendar_incomplete_dates"),
    )
    decision = json.loads((official / "decision_report.json").read_text(encoding="utf-8"))
    decision_short = (
        decision.get("by_fill", {})
        .get("cross", {})
        .get("primary", {})
        .get("long_short", {})
        .get("short")
    )
    gates = evaluate_gates(
        receipt_ok=receipt_gate.passed,
        receipt_detail=receipt_gate.detail,
        reconstruction_issues=reconstruction_issues,
        pairing=pairing,
        matched=matched,
        calendar=calendar,
        calendar_issues=calendar_issues,
        decision_short=decision_short,
    )
    gates.insert(1, GateResult("columns", True, "required columns present"))
    verdict = readiness_verdict(gates)
    class_counts = (
        {str(key): int(value) for key, value in calendar["short_book_class"].value_counts().items()}
        if not calendar.empty
        else {}
    )
    window_counts = (
        {str(key): int(value) for key, value in matched["window_label"].value_counts().items()}
        if not matched.empty
        else {}
    )
    primary = matched.loc[_primary_mask(matched)] if not matched.empty else matched
    report = {
        "verdict": verdict,
        "pairing_unmatched_mid": [
            {"trade_date": item[0].isoformat(), "ticker": item[1], "direction": item[2]}
            for item in pairing["unmatched_mid"]
        ],
        "missing_from_mid": [
            {"trade_date": item[0].isoformat(), "ticker": item[1], "direction": item[2]}
            for item in pairing["missing_from_mid"]
        ],
        "class_counts": class_counts,
        "window_counts": window_counts,
        "n_matched_rows": int(len(matched)),
        "reconstruction_issue_count": len(reconstruction_issues),
        "calendar_issue_count": len(calendar_issues),
        "primary_anchor": {
            "window": "2020-01-01 through 2026-07-10",
            "accepted_count": ACCEPTED_PRIMARY_SHORT_COUNT,
            "accepted_pnl": ACCEPTED_PRIMARY_SHORT_PNL,
            "observed_count": int(len(primary)),
            "observed_official_pnl": float(primary["pnl_cross_official"].sum()) if len(primary) else None,
            "observed_reconstructed_pnl": (
                float(primary["pnl_legs_sum"].sum())
                if len(primary) and primary["pnl_legs_sum"].notna().all()
                else None
            ),
            "residual_legs_vs_official_sum": (
                float(primary["residual_legs_vs_official"].sum())
                if len(primary) and primary["residual_legs_vs_official"].notna().all()
                else None
            ),
        },
    }
    if _report_has_forbidden(report):
        raise D0ReadinessError("report contains a forbidden comparative field")
    return D0ReadinessResult(
        verdict=verdict,
        gates=gates,
        matched=matched,
        calendar=calendar,
        report=report,
        inventory=inventory,
    )


def render_report_md(result: D0ReadinessResult) -> str:
    lines = [
        "# Sprint 009 D0 readiness report",
        "",
        f"**Verdict:** `{result.verdict}`",
        "",
        "Later-period rows may be stored. This report has no comparative economic, filter, or protection result.",
        "",
        "## Gates",
        "",
    ]
    for gate in result.gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"- {mark} `{gate.gate_id}`: {gate.detail}")
    lines.extend(["", "## Calendar", ""])
    if result.calendar.empty:
        lines.append("No calendar written.")
    else:
        counts = result.calendar.groupby(["window_label", "short_book_class"]).size()
        for key, count in counts.items():
            lines.append(f"- {key[0]} / {key[1]}: {int(count)}")
    lines.append("")
    lines.append(f"Unmatched midpoint keys: {len(result.report.get('pairing_unmatched_mid', []))}")
    lines.append("")
    return "\n".join(lines)


def export_d0_evidence(result: D0ReadinessResult, evidence_dir: Path, *, code_sha: str | None) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if OFFICIAL_RUN_DIR.resolve() == evidence_dir.resolve():
        raise D0ReadinessError("refusing to write into the official run directory")
    (evidence_dir / "input_inventory.json").write_text(
        json.dumps({**result.inventory, "code_sha": code_sha}, indent=2, default=str),
        encoding="utf-8",
    )
    result.matched.to_parquet(evidence_dir / "matched_short_iron_flies.parquet", index=False)
    result.calendar.to_parquet(evidence_dir / "short_calendar.parquet", index=False)
    payload = {
        **result.report,
        "code_sha": code_sha,
        "gates": [
            {"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail}
            for gate in result.gates
        ],
    }
    (evidence_dir / "d0_report.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    (evidence_dir / "d0_report.md").write_text(render_report_md(result), encoding="utf-8")
    return evidence_dir
