"""Hand-calculated Sprint 009 D2 checks. Does not read official artifacts."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.backtest.sprint009_d2_protection_comparison import (
    ACCEPTED_D1_CODE_SHA,
    ACCEPTED_D1_DIR,
    ACCEPTED_D1_HASHES,
    ACCEPTED_D1_VERDICT,
    GateResult,
    compare_development,
    d1_handoff_problems,
    drawdown_path,
    loss_avoided,
    readiness_verdict,
)

ZERO = date(2020, 3, 13)
LATER = date(2024, 1, 5)


def _row(
    day: date,
    ticker: str,
    *,
    body: float,
    fly: float,
    w_mid: float,
    h_wing: float,
    w_pay: float,
    b_mid: float,
    q: float = 2.0,
    input_ok: object = True,
) -> dict:
    net = w_pay - w_mid - h_wing
    return {
        "trade_date": day,
        "ticker": ticker,
        "direction": "short",
        "window_label": "development",
        "Q": q,
        "quantity_cross_signed": -q,
        "input_ok": input_ok,
        "b_mid": b_mid,
        "h_body": 0.4,
        "w_mid": w_mid,
        "h_wing": h_wing,
        "w_pay": w_pay,
        "p_body_cross": body,
        "p_fly_cross": fly,
        "pnl_body_cross": body,
        "pnl_cross_official": fly,
        "pnl_legs_sum": fly,
        "pnl_wing_cross": net,
        "pnl_mid_at_cross_q": b_mid + w_pay - w_mid,
    }


def _calendar(days: list[date], zero: date | None = None) -> pd.DataFrame:
    rows = [{"trade_date": day, "window_label": "development", "short_book_class": "verified_positive_short"} for day in days]
    if zero is not None:
        rows.append({"trade_date": zero, "window_label": "development", "short_book_class": "verified_zero_short"})
    return pd.DataFrame(rows)


def _annual_from(trades: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    dated = dates.copy()
    dated["year"] = dated["trade_date"].map(lambda value: value.year if isinstance(value, date) else pd.Timestamp(value).year)
    rows = []
    for year, frame in dated.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "n_dates": int(len(frame)),
                "n_zero_short_dates": int((frame["short_book_class"] == "verified_zero_short").sum()),
                "n_trades": int(frame["n_trades"].sum()) if "n_trades" in frame else int(len(trades)),
                "p_body_cross": float(frame["p_body_cross"].sum()) if "p_body_cross" in frame else float(trades["p_body_cross"].sum()),
                "p_fly_cross": float(frame["p_fly_cross"].sum()) if "p_fly_cross" in frame else float(trades["p_fly_cross"].sum()),
                "w_pay": float(frame["w_pay"].sum()) if "w_pay" in frame else float(trades["w_pay"].sum()),
                "w_mid": float(frame["w_mid"].sum()) if "w_mid" in frame else float(trades["w_mid"].sum()),
                "h_wing": float(frame["h_wing"].sum()) if "h_wing" in frame else float(trades["h_wing"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _saved_dates(trades: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    grouped = trades.groupby("trade_date")
    rows = []
    for _, status in calendar.iterrows():
        day = status["trade_date"]
        frame = grouped.get_group(day) if day in grouped.groups else trades.iloc[0:0]
        rows.append(
            {
                "trade_date": day,
                "window_label": "development",
                "short_book_class": status["short_book_class"],
                "n_trades": int(len(frame)),
                "b_mid": float(frame["b_mid"].sum()) if len(frame) else 0.0,
                "h_body": float(frame["h_body"].sum()) if len(frame) else 0.0,
                "w_mid": float(frame["w_mid"].sum()) if len(frame) else 0.0,
                "h_wing": float(frame["h_wing"].sum()) if len(frame) else 0.0,
                "w_pay": float(frame["w_pay"].sum()) if len(frame) else 0.0,
                "p_body_cross": float(frame["p_body_cross"].sum()) if len(frame) else 0.0,
                "p_fly_cross": float(frame["p_fly_cross"].sum()) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _panel(trades: list[dict], calendar: pd.DataFrame, *, scale_saved: bool = False):
    frame = pd.DataFrame(trades)
    dates = _saved_dates(frame, calendar)
    if scale_saved:
        frame = frame.copy()
        for name in ("w_mid", "h_wing", "w_pay", "p_body_cross", "p_fly_cross", "b_mid"):
            frame[name] = frame[name] * 100.0
    annual = _annual_from(frame if not scale_saved else pd.DataFrame(trades), dates)
    if scale_saved:
        annual = _annual_from(frame, dates)
    return compare_development(frame, dates, annual, require_official_coverage=False)


def _consistent(day: date, ticker: str, *, body: float, w_mid: float, h_wing: float, w_pay: float) -> dict:
    fly = body + (w_pay - w_mid - h_wing)
    return _row(day, ticker, body=body, fly=fly, w_mid=w_mid, h_wing=h_wing, w_pay=w_pay, b_mid=3.0)


def _book() -> tuple[list[dict], pd.DataFrame]:
    start = date(2021, 1, 4)
    days = [start + timedelta(days=index * 7) for index in range(12)]
    trades = [
        _consistent(days[0], "MMM", body=-10.0, w_mid=10.0, h_wing=10.0, w_pay=0.0),
        _consistent(days[1], "PPP", body=4.0, w_mid=2.0, h_wing=2.0, w_pay=2.0),
        _consistent(days[2], "LLL", body=-8.0, w_mid=1.0, h_wing=1.0, w_pay=0.0),
    ]
    for index, day in enumerate(days[3:], start=3):
        trades.append(_consistent(day, "AAA", body=-1.0 - index / 10.0, w_mid=1.0, h_wing=1.0, w_pay=0.0))
    trades.append(
        {
            **_consistent(LATER, "ZZZ", body=-1.0, w_mid=1.0, h_wing=1.0, w_pay=0.0),
            "window_label": "later_period",
        }
    )
    return trades, _calendar(days, ZERO)


def test_loss_avoided_keeps_a_negative_and_is_not_payout() -> None:
    assert loss_avoided(-10.0, -30.0) == -20.0
    assert loss_avoided(-10.0, -30.0) != 5.0
    assert loss_avoided(4.0, 4.0) == 0.0


def test_drawdown_starts_at_zero_and_all_positive_is_flat() -> None:
    cumulative, drawdowns, ending, deepest = drawdown_path([-5.0, 3.0])
    assert cumulative == [-5.0, -2.0]
    assert drawdowns[0] == -5.0
    assert deepest == -5.0
    assert ending == -2.0
    _, _, _, flat = drawdown_path([1.0, 2.0, 3.0])
    assert flat == 0.0


def test_valid_control_passes_runner_verdict() -> None:
    trades, calendar = _book()
    result = _panel(trades, calendar)
    assert result.verdict == "READY"
    assert readiness_verdict(result.gates) == "READY"
    assert set(result.trades["Q"]) == {2.0}
    assert "ZZZ" not in set(result.trades["ticker"])
    assert LATER not in set(result.dates["trade_date"])
    zero = result.dates.loc[result.dates["trade_date"].map(lambda value: value == ZERO)].iloc[0]
    assert zero["n_trades"] == 0
    assert zero["p_fly_cross"] == 0.0
    assert zero["cumulative_body_cross"] == 0.0
    paid = result.trades.loc[result.trades["w_pay"] > 1e-6]
    assert len(paid) == 1
    assert paid.iloc[0]["net_cross"] < 0
    assert result.frequency["trade_counts"]["payout_positive"] == 1
    assert result.frequency["trade_counts"]["net_contribution_positive"] == 0
    body_worst = result.worst.loc[result.worst["list_id"] == "body_trades"].sort_values("rank")
    assert body_worst.iloc[0]["ticker"] == "MMM"
    assert body_worst.iloc[0]["p_body_cross"] == -10.0
    assert body_worst.iloc[0]["p_fly_cross"] == -30.0
    assert body_worst.iloc[0]["loss_avoided"] == -20.0
    assert body_worst.iloc[0]["w_pay"] == 0.0
    assert result.drawdown["body_cross"]["max_drawdown"] < 0


def test_quantity_scale_and_non_boolean_input_are_blocked() -> None:
    trades, calendar = _book()
    scaled = _panel(trades, calendar, scale_saved=True)
    assert scaled.verdict == "BLOCKED"
    assert not next(gate for gate in scaled.gates if gate.gate_id == "reconciliation").passed
    assert readiness_verdict([GateResult("provenance", True, "held")] + scaled.gates) == "BLOCKED"

    broken = [dict(trades[0], input_ok="true")]
    calendar = _calendar([trades[0]["trade_date"]], ZERO)
    # pad to keep the named input failure independent of needing ten rows
    result = compare_development(pd.DataFrame(broken), _saved_dates(pd.DataFrame(broken), calendar), None, require_official_coverage=False)
    assert result.verdict == "BLOCKED"
    assert any("input_ok" in item for item in result.issues)
    assert readiness_verdict(result.gates) == "BLOCKED"


def test_zero_loss_denominator_stays_null_and_trades_remain() -> None:
    days = [date(2022, 6, 6) + timedelta(days=7 * index) for index in range(10)]
    trades = [
        _row(day, f"T{index}", body=1.0 + index, fly=1.0 + index, w_mid=0.0, h_wing=0.0, w_pay=0.0, b_mid=1.0)
        for index, day in enumerate(days)
    ]
    result = _panel(trades, _calendar(days))
    assert result.verdict == "READY"
    assert len(result.trades) == 10
    assert result.concentration["body_trades"]["worst10_share"] is None
    assert result.concentration["body_trades"]["reason"] == "no gross losing dollars"
    assert result.drawdown["body_cross"]["max_drawdown"] == 0.0
    assert result.frequency["conditional_positive_payout"]["reason"] == "no positive wing payout"


def test_d1_handoff_mismatch_blocks_acceptance() -> None:
    receipt = {"code_sha": ACCEPTED_D1_CODE_SHA, "verdict": ACCEPTED_D1_VERDICT}
    assert d1_handoff_problems(receipt, ACCEPTED_D1_DIR, ACCEPTED_D1_HASHES) == []
    bad = d1_handoff_problems(
        {"code_sha": "not-the-sha", "verdict": "BLOCKED"},
        ACCEPTED_D1_DIR,
        {name: "deadbeef" for name in ACCEPTED_D1_HASHES},
    )
    assert any("code SHA" in item for item in bad)
    assert any("verdict" in item for item in bad)
    assert any("hash" in item for item in bad)
