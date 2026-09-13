"""Synthetic Sprint 009 D0 checks. Does not read official artifacts."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtest.sprint009_d0_body_wing_readiness import (
    FORBIDDEN_REPORT_KEYS,
    MATCHED_COLUMNS,
    GateResult,
    assess_panel,
    build_matched_rows,
    classify_short_calendar,
    readiness_verdict,
    reconstructed_leg_economics,
    signed_entry_cash,
    unsigned_intrinsic,
    window_label,
)

DAY = date(2024, 1, 5)
DEV = date(2021, 6, 4)


def _legs(quantity: float = -2.0, *, ask_bump: float = 0.0, payoff_bump: float = 0.0, swap_wings: bool = False, exit_spot: float = 95.0) -> pd.DataFrame:
    specs = [
        (0, "put", 1, 90.0, 0.40, 0.60),
        (1, "put", -1, 100.0, 2.00, 2.20),
        (2, "call", -1, 100.0, 2.10, 2.30),
        (3, "call", 1, 112.0, 0.30, 0.50),
    ]
    if swap_wings:
        specs[0] = (0, "call", 1, 90.0, 0.40, 0.60)
    rows = []
    for index, option_type, unit, strike, bid, ask in specs:
        ask = ask + ask_bump
        fill = ask if unit > 0 else bid
        entry = fill * abs(unit) if unit > 0 else -fill * abs(unit)
        intrinsic = max(strike - exit_spot, 0.0) if option_type == "put" else max(exit_spot - strike, 0.0)
        payoff = intrinsic * unit + payoff_bump
        pnl_per = payoff - entry
        rows.append(
            {
                "trade_date": DAY,
                "ticker": "AAA",
                "direction": "short",
                "expiry_date": date(2024, 1, 12),
                "option_type": option_type,
                "strike": strike,
                "leg_index": index,
                "unit_quantity": unit,
                "portfolio_quantity": abs(quantity) * unit,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "fill_price": fill,
                "entry_cash_per_unit": entry,
                "expiry_payoff_per_unit": payoff,
                "pnl_per_unit": pnl_per,
                "pnl_total_leg": abs(quantity) * pnl_per,
                "exit_spot": exit_spot,
                "included_in_portfolio": True,
                "fill_label": "cross",
            }
        )
    return pd.DataFrame(rows)


def _trade(quantity: float = -2.0, pnl: float | None = None, legs: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = legs if legs is not None else _legs(quantity)
    official = float(frame["pnl_total_leg"].sum()) if pnl is None else pnl
    return pd.DataFrame(
        [
            {
                "trade_date": DAY,
                "ticker": "AAA",
                "direction": "short",
                "included_in_portfolio": True,
                "instrument_type": "iron_fly",
                "expiry_date": date(2024, 1, 12),
                "entry_spot": 101.0,
                "exit_spot": 95.0,
                "body_strike": 100.0,
                "quantity": quantity,
                "capital_at_risk_dollars": 250.0,
                "pnl_total": official,
                "fill_label": "cross",
                "long_put_strike": 90.0,
                "long_call_strike": 112.0,
            }
        ]
    )


def _mid_trade(quantity: float = -4.0) -> pd.DataFrame:
    frame = _trade(quantity)
    frame["fill_label"] = "mid"
    return frame


def _mid_legs(cross_legs: pd.DataFrame) -> pd.DataFrame:
    frame = cross_legs.copy()
    frame["fill_label"] = "mid"
    return frame


def test_cash_signs_and_independent_settlement() -> None:
    sold = signed_entry_cash(2.0, -1)
    bought = signed_entry_cash(0.5, 1)
    assert sold == -2.0
    assert bought == 0.5
    assert unsigned_intrinsic("put", 90.0, 100.0) == 0.0
    assert unsigned_intrinsic("put", 110.0, 100.0) == 10.0
    rebuilt = reconstructed_leg_economics(
        option_type="put",
        strike=100.0,
        unit_quantity=-1,
        bid=2.0,
        ask=2.2,
        exit_spot=100.0,
        quantity_magnitude=2.0,
    )
    assert rebuilt["entry_cash_per_unit"] < 0
    assert rebuilt["fill_price"] == 2.0
    assert rebuilt["expiry_payoff_per_unit"] == 0.0
    assert rebuilt["pnl_total_leg"] == 2.0 * (0.0 - rebuilt["entry_cash_per_unit"])


def test_quantity_is_share_equivalent_not_times_100() -> None:
    legs = _legs(-2.0)
    wrong = abs(-2.0) * 100 * -1
    assert float(legs.loc[legs["leg_index"] == 1, "portfolio_quantity"].iloc[0]) == -2.0
    assert wrong != -2.0
    matched, _, issues = build_matched_rows(_trade(-2.0, legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert matched.iloc[0]["Q"] == 2.0
    assert not any("portfolio_quantity" in item for item in issues)


def test_swapped_wing_keeps_row() -> None:
    legs = _legs(swap_wings=True)
    matched, _, issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("option_type" in item for item in issues)


def test_reconciliation_failure_is_a_gate_input() -> None:
    legs = _legs()
    trade = _trade(pnl=999.0, legs=legs)
    matched, _, issues = build_matched_rows(trade, legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert abs(matched.iloc[0]["residual_legs_vs_official"]) > 0.01
    assert any("official P&L" in item for item in issues)


def test_missing_and_extra_mid_keys_are_not_inner_joined_away() -> None:
    legs = _legs()
    extra = _mid_trade()
    extra["ticker"] = "BBB"
    extra_legs = _mid_legs(legs)
    extra_legs["ticker"] = "BBB"
    matched, pairing, _issues = build_matched_rows(
        _trade(legs=legs),
        legs,
        pd.concat([extra], ignore_index=True),
        extra_legs,
    )
    assert len(matched) == 1
    assert matched.iloc[0]["ticker"] == "AAA"
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert pairing["unmatched_mid"][0][1] == "BBB"
    assert matched.iloc[0]["quantity_mid_abs"] is None


def test_quote_mismatch_keeps_cross_row() -> None:
    legs = _legs()
    mid_legs = _mid_legs(legs)
    mid_legs.loc[mid_legs["leg_index"] == 0, "bid"] = 9.0
    matched, pairing, _issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), mid_legs)
    assert len(matched) == 1
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert "AAA" not in {key[1] for key in pairing["unmatched_mid"]}


def test_long_only_and_missing_funnel() -> None:
    matched = pd.DataFrame(columns=list(MATCHED_COLUMNS))
    status = pd.DataFrame(
        [
            {"trade_date": DEV, "status": "traded", "reason": "ok"},
            {"trade_date": DAY, "status": "traded", "reason": "ok"},
        ]
    )
    funnel = pd.DataFrame(
        [
            {
                "trade_date": DEV,
                "n_included": 1,
                "n_included_long": 1,
                "n_included_short": 0,
                "date_status": "traded",
                "date_reason": "ok",
            }
        ]
    )
    calendar, issues = classify_short_calendar(status, funnel, matched)
    by_day = {row.trade_date: row for row in calendar.itertuples()}
    assert by_day[DEV].short_book_class == "verified_zero_short"
    assert by_day[DAY].short_book_class == "blocked"
    assert "null" not in by_day[DAY].blocker_reason
    assert "missing funnel" in by_day[DAY].blocker_reason
    assert any(DAY.isoformat() in item or str(DAY) in item for item in issues)


def test_null_short_count_is_blocked_not_cash() -> None:
    status = pd.DataFrame([{"trade_date": DEV, "status": "traded", "reason": "ok"}])
    funnel = pd.DataFrame(
        [
            {
                "trade_date": DEV,
                "n_included": 0,
                "n_included_long": 0,
                "n_included_short": float("nan"),
                "date_status": "traded",
                "date_reason": "ok",
            }
        ]
    )
    calendar, _issues = classify_short_calendar(status, funnel, pd.DataFrame(columns=list(MATCHED_COLUMNS)))
    assert calendar.iloc[0]["short_book_class"] == "blocked"
    assert calendar.iloc[0]["blocker_reason"] == "null n_included_short"


def test_times_100_quantity_fails_and_row_is_kept() -> None:
    legs = _legs(-2.0)
    legs["portfolio_quantity"] = legs["portfolio_quantity"] * 100
    matched, _, issues = build_matched_rows(_trade(-2.0, legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("portfolio_quantity" in item for item in issues)


def test_non_body_strike_and_duplicate_leg_keep_row() -> None:
    legs = _legs()
    trade = _trade(legs=legs)
    trade["body_strike"] = 99.0
    matched, _, issues = build_matched_rows(trade, legs, _mid_trade(), _mid_legs(legs))
    assert len(matched) == 1
    assert any("not body_strike" in item for item in issues)
    dup = pd.concat([legs, legs.iloc[[0]]], ignore_index=True)
    matched_dup, _, dup_issues = build_matched_rows(_trade(legs=legs), dup, _mid_trade(), _mid_legs(legs))
    assert len(matched_dup) == 1
    assert any("duplicate leg key" in item or "expected 4 legs" in item for item in dup_issues)


def test_paired_settlement_mismatch_keeps_cross_row() -> None:
    legs = _legs()
    mid_legs = _mid_legs(legs)
    mid_legs.loc[mid_legs["leg_index"] == 1, "expiry_payoff_per_unit"] = 99.0
    matched, pairing, _issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), mid_legs)
    assert len(matched) == 1
    assert matched.iloc[0]["ticker"] == "AAA"
    assert bool(matched.iloc[0]["pairing_ok"]) is False
    assert "expiry_payoff" in matched.iloc[0]["pairing_reason"]
    assert pairing["unmatched_mid"] == []


def test_saved_row_contract_and_later_period_storage() -> None:
    legs = _legs()
    matched, _, issues = build_matched_rows(_trade(legs=legs), legs, _mid_trade(), _mid_legs(legs))
    assert not issues
    row = matched.iloc[0]
    for column in MATCHED_COLUMNS:
        assert column in matched.columns
    assert row["window_label"] == "later_period"
    assert window_label(DEV) == "development"
    assert row["Q"] == 2.0
    assert row["entry_spot"] == 101.0
    assert row["capital_at_risk_dollars"] == 250.0
    assert row["put_wing_entry_cash_per_unit"] > 0
    assert row["body_put_entry_cash_per_unit"] < 0
    assert row["pnl_mid_at_cross_q"] is not None
    assert row["residual_legs_vs_official"] == 0.0 or abs(row["residual_legs_vs_official"]) < 1e-6
    assert not set(FORBIDDEN_REPORT_KEYS) & set(matched.columns)


VNT = date(2020, 3, 13)


def _status_funnel(*, n_short: float = 1, status: str = "traded") -> tuple[pd.DataFrame, pd.DataFrame]:
    status_rows = [
        {"trade_date": DAY, "status": status, "reason": "ok"},
        {"trade_date": DEV, "status": "traded", "reason": "long only"},
        {"trade_date": VNT, "status": "valid_no_trade", "reason": "no book"},
    ]
    funnel_rows = [
        {
            "trade_date": DAY,
            "n_included": n_short,
            "n_included_long": 0,
            "n_included_short": n_short,
            "date_status": status,
            "date_reason": "ok",
        },
        {
            "trade_date": DEV,
            "n_included": 1,
            "n_included_long": 1,
            "n_included_short": 0,
            "date_status": "traded",
            "date_reason": "long only",
        },
        {
            "trade_date": VNT,
            "n_included": 0,
            "n_included_long": 0,
            "n_included_short": 0,
            "date_status": "valid_no_trade",
            "date_reason": "no book",
        },
    ]
    return pd.DataFrame(status_rows), pd.DataFrame(funnel_rows)


def _panel(trade: pd.DataFrame | None = None, legs: pd.DataFrame | None = None, mid_legs: pd.DataFrame | None = None, **funnel_kwargs):
    frame = legs if legs is not None else _legs()
    book = trade if trade is not None else _trade(legs=frame)
    status, funnel = _status_funnel(**funnel_kwargs)
    return assess_panel(book, frame, _mid_trade(), mid_legs if mid_legs is not None else _mid_legs(frame), status, funnel)


def _gate(result, gate_id: str) -> GateResult:
    return next(gate for gate in result.gates if gate.gate_id == gate_id)


def _cannot_be_ready(result) -> None:
    identity = [
        GateResult("receipt", True, "ok"),
        GateResult("columns", True, "ok"),
        GateResult("primary_anchor", True, "identity held only to test propagation"),
    ]
    assert result.verdict == "BLOCKED"
    assert readiness_verdict(identity + result.gates) == "BLOCKED"


def test_valid_control_panel_is_ready() -> None:
    result = _panel()
    assert result.verdict == "READY"
    assert all(gate.passed for gate in result.gates)
    by_day = {row.trade_date: row.short_book_class for row in result.calendar.itertuples()}
    assert by_day[DAY] == "verified_positive_short"
    assert by_day[DEV] == "verified_zero_short"
    assert by_day[VNT] == "verified_zero_short"
    identity = [
        GateResult("receipt", True, "ok"),
        GateResult("columns", True, "ok"),
        GateResult("primary_anchor", True, "identity held only to test propagation"),
    ]
    assert readiness_verdict(identity + result.gates) == "READY"


def test_missing_required_fields_fail_reconstruction_and_block_ready() -> None:
    cases = {
        "entry_spot": ("trade", "entry_spot", "missing or non-finite"),
        "capital_at_risk_dollars": ("trade", "capital_at_risk_dollars", "missing or non-finite"),
        "portfolio_quantity": ("leg", "portfolio_quantity", "missing or non-finite"),
        "pnl_total_leg": ("leg", "pnl_total_leg", "missing or non-finite"),
    }
    for field, (where, name, reason) in cases.items():
        trade = _trade()
        legs = _legs()
        if where == "trade":
            trade[field] = float("nan")
        else:
            legs.loc[legs["leg_index"] == 1, field] = float("nan")
        result = _panel(trade=trade, legs=legs)
        assert len(result.matched) == 1
        assert any(name in item and reason in item for item in _gate(result, "reconstruction").detail.split("; "))
        assert not _gate(result, "reconstruction").passed
        if field == "pnl_total_leg":
            assert result.matched.iloc[0]["body_put_pnl_total_leg"] is not None
        _cannot_be_ready(result)


def test_bad_short_quantity_and_settlement_spot_block_ready() -> None:
    positive = _panel(trade=_trade(quantity=2.0))
    assert any("quantity" in item and "sign is not negative" in item for item in _gate(positive, "reconstruction").detail.split("; "))
    assert not _gate(positive, "reconstruction").passed
    _cannot_be_ready(positive)

    zero = _trade(quantity=0.0)
    zero_result = _panel(trade=zero)
    assert any("quantity" in item and "magnitude is not positive" in item for item in _gate(zero_result, "reconstruction").detail.split("; "))
    _cannot_be_ready(zero_result)

    legs = _legs(exit_spot=90.0)
    legs["expiry_payoff_per_unit"] = [
        (max(strike - 90.0, 0.0) if option_type == "put" else max(90.0 - strike, 0.0)) * unit
        for option_type, strike, unit in zip(legs["option_type"], legs["strike"], legs["unit_quantity"])
    ]
    legs["pnl_per_unit"] = legs["expiry_payoff_per_unit"] - legs["entry_cash_per_unit"]
    legs["pnl_total_leg"] = 2.0 * legs["pnl_per_unit"]
    trade = _trade(pnl=float(legs["pnl_total_leg"].sum()), legs=legs)
    spot = _panel(trade=trade, legs=legs)
    assert any("exit_spot" in item and "disagrees with trade settlement spot" in item for item in _gate(spot, "reconstruction").detail.split("; "))
    assert not _gate(spot, "reconstruction").passed
    _cannot_be_ready(spot)


def test_duplicate_midpoint_legs_fail_pairing_and_keep_cross_row() -> None:
    legs = _legs()
    mid = pd.concat([_mid_legs(legs), _mid_legs(legs).iloc[[1]]], ignore_index=True)
    result = _panel(legs=legs, mid_legs=mid)
    assert len(result.matched) == 1
    assert result.matched.iloc[0]["ticker"] == "AAA"
    assert bool(result.matched.iloc[0]["pairing_ok"]) is False
    assert "midpoint duplicate leg keys" in result.matched.iloc[0]["pairing_reason"]
    assert not _gate(result, "pairing").passed
    _cannot_be_ready(result)


def test_fractional_count_and_incompatible_status_block_calendar() -> None:
    fractional = _panel(n_short=1.7)
    stored_count = fractional.calendar.loc[fractional.calendar["trade_date"] == DAY, "funnel_n_included_short"].iloc[0]
    assert pd.isna(stored_count)
    assert fractional.calendar.loc[fractional.calendar["trade_date"] == DAY, "blocker_reason"].iloc[0] == "fractional n_included_short"
    assert not _gate(fractional, "calendar").passed
    _cannot_be_ready(fractional)

    occupied = _panel(status="valid_no_trade")
    assert occupied.calendar.loc[occupied.calendar["trade_date"] == DAY, "blocker_reason"].iloc[0] == "valid_no_trade date contains included positions"
    assert not _gate(occupied, "calendar").passed
    _cannot_be_ready(occupied)

    status, funnel = _status_funnel()
    status.loc[status["trade_date"] == DAY, "status"] = "unknown"
    funnel.loc[funnel["trade_date"] == DAY, "date_status"] = "unknown"
    unknown = assess_panel(_trade(), _legs(), _mid_trade(), _mid_legs(_legs()), status, funnel)
    assert unknown.calendar.loc[unknown.calendar["trade_date"] == DAY, "short_book_class"].iloc[0] == "blocked"
    assert "not traded" in unknown.calendar.loc[unknown.calendar["trade_date"] == DAY, "blocker_reason"].iloc[0]
    assert not _gate(unknown, "calendar").passed
    _cannot_be_ready(unknown)
