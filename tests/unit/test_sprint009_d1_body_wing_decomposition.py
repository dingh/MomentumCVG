"""Hand-calculated Sprint 009 D1 checks. Does not read official artifacts."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.sprint007_artifact_validation import expected_mid_fill_price
from src.backtest.sprint009_d0_body_wing_readiness import signed_entry_cash
from src.backtest.sprint009_d1_body_wing_decomposition import (
    ACCEPTED_D0_CODE_SHA,
    ACCEPTED_D0_DIR,
    ACCEPTED_D0_VERDICT,
    GateResult,
    d0_receipt_problems,
    decompose_development,
    official_acceptance_verdict,
    readiness_verdict,
)

DAY = date(2021, 6, 4)
ZERO = date(2020, 3, 13)
LATER = date(2024, 1, 5)


def _saved_from_components(*, b_mid: float, h_body: float, w_mid: float, h_wing: float, w_pay: float) -> dict[str, float]:
    p_body = b_mid - h_body
    p_fly = b_mid - h_body - w_mid - h_wing + w_pay
    return {
        "pnl_body_cross": p_body,
        "pnl_wing_cross": w_pay - w_mid - h_wing,
        "pnl_cross_official": p_fly,
        "pnl_legs_sum": p_fly,
        "pnl_mid_at_cross_q": b_mid + w_pay - w_mid,
    }


def _trade(
    *,
    day: date = DAY,
    ticker: str = "AAA",
    window: str = "development",
    q: float = 2.0,
    body_put_payoff: float = 0.0,
    put_wing_payoff: float = 0.0,
    stored_body_put_mid: float | None = None,
    zero_body_quotes: bool = False,
    zero_body_spread: bool = False,
    official_pnl: float | None = None,
) -> dict:
    specs = {
        "body_put": (2.0, 2.2, -1, body_put_payoff),
        "body_call": (2.1, 2.3, -1, 0.0),
        "put_wing": (0.4, 0.6, 1, put_wing_payoff),
        "call_wing": (0.3, 0.5, 1, 0.0),
    }
    if zero_body_quotes:
        specs["body_put"] = (0.0, 0.0, -1, 0.0)
        specs["body_call"] = (0.0, 0.0, -1, 0.0)
    if zero_body_spread:
        specs["body_put"] = (2.0, 2.0, -1, body_put_payoff)
        specs["body_call"] = (2.1, 2.1, -1, 0.0)
    row: dict = {
        "trade_date": day,
        "ticker": ticker,
        "direction": "short",
        "window_label": window,
        "pairing_ok": True,
        "Q": q,
        "quantity_cross_signed": -q,
        "entry_spot": 101.0,
    }
    b_mid = 0.0
    h_body = 0.0
    w_mid = 0.0
    h_wing = 0.0
    w_pay = 0.0
    for prefix, (bid, ask, unit, payoff) in specs.items():
        mid_fill = expected_mid_fill_price(bid, ask, unit)
        stored = stored_body_put_mid if prefix == "body_put" and stored_body_put_mid is not None else mid_fill
        row[f"{prefix}_bid"] = bid
        row[f"{prefix}_ask"] = ask
        row[f"{prefix}_mid"] = stored
        row[f"{prefix}_unit_quantity"] = unit
        row[f"{prefix}_expiry_payoff_per_unit"] = payoff
        if prefix.startswith("body"):
            mid_cash = signed_entry_cash(mid_fill, unit)
            b_mid += q * (payoff - mid_cash)
            h_body += q * (mid_fill - bid)
        else:
            w_mid += q * mid_fill
            h_wing += q * (ask - mid_fill)
            w_pay += q * payoff
    saved = _saved_from_components(b_mid=b_mid, h_body=h_body, w_mid=w_mid, h_wing=h_wing, w_pay=w_pay)
    if official_pnl is not None:
        saved["pnl_cross_official"] = official_pnl
    row.update(saved)
    return row


def _calendar(*rows: tuple[date, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"trade_date": day, "window_label": "development" if day <= date(2023, 12, 31) else "later_period", "short_book_class": klass} for day, klass in rows]
    )


def _panel(trades: list[dict], calendar: pd.DataFrame):
    return decompose_development(pd.DataFrame(trades), calendar, require_official_coverage=False)


def _gate(result, gate_id: str) -> GateResult:
    return next(gate for gate in result.gates if gate.gate_id == gate_id)


def _cannot_be_ready(result) -> None:
    identity = [GateResult("provenance", True, "held only to test propagation")]
    assert result.verdict == "BLOCKED"
    assert readiness_verdict(identity + result.gates) == "BLOCKED"


def test_appendix_signs_and_identity() -> None:
    sold = signed_entry_cash(expected_mid_fill_price(2.0, 2.2, -1), -1)
    bought = signed_entry_cash(expected_mid_fill_price(0.4, 0.6, 1), 1)
    assert sold < 0
    assert bought > 0
    result = _panel([_trade()], _calendar((DAY, "verified_positive_short")))
    row = result.trades.iloc[0]
    assert row["h_body"] == pytest.approx(0.40)
    assert row["h_wing"] == pytest.approx(0.40)
    assert row["b_mid"] == pytest.approx(8.60)
    assert row["w_mid"] == pytest.approx(1.80)
    assert row["p_body_cross"] == pytest.approx(8.20)
    assert row["p_body_cross"] != row["b_mid"] - 2 * row["h_body"]
    assert row["p_fly_cross"] == pytest.approx(6.00)
    assert result.verdict == "READY"
    assert readiness_verdict(result.gates) == "READY"


def test_quantity_times_100_fails_identity() -> None:
    scaled = _trade(q=200.0)
    scaled["pnl_cross_official"] = 6.00
    scaled["pnl_legs_sum"] = 6.00
    scaled["pnl_body_cross"] = 8.20
    scaled["pnl_wing_cross"] = -2.20
    scaled["pnl_mid_at_cross_q"] = 6.80
    result = _panel([scaled], _calendar((DAY, "verified_positive_short")))
    assert result.trades.iloc[0]["h_body"] == pytest.approx(40.0)
    assert not _gate(result, "identity").passed
    _cannot_be_ready(result)


def test_signed_wing_payoff_is_not_rescaled() -> None:
    result = _panel(
        [_trade(put_wing_payoff=5.0, body_put_payoff=-4.0)],
        _calendar((DAY, "verified_positive_short")),
    )
    row = result.trades.iloc[0]
    assert row["w_pay"] == pytest.approx(10.0)
    assert row["b_mid"] != 8.60
    assert result.verdict == "READY"


def test_undefined_ratio_keeps_the_trade() -> None:
    result = _panel([_trade(zero_body_quotes=True)], _calendar((DAY, "verified_positive_short")))
    row = result.trades.iloc[0]
    assert len(result.trades) == 1
    assert row["c_body"] == pytest.approx(0.0)
    assert row["h_wing"] == pytest.approx(0.40)
    assert pd.isna(row["wing_concession_ratio"]) or row["wing_concession_ratio"] is None
    assert row["ratio_reason"] == "non-positive body midpoint credit"
    assert result.dollars["h_wing"] == pytest.approx(0.40)
    assert result.verdict == "READY"


def test_zero_short_date_and_later_period_isolation() -> None:
    trades = [
        _trade(),
        _trade(day=LATER, window="later_period", ticker="BBB"),
    ]
    calendar = pd.DataFrame(
        [
            {"trade_date": DAY, "window_label": "development", "short_book_class": "verified_positive_short"},
            {"trade_date": ZERO, "window_label": "development", "short_book_class": "verified_zero_short"},
            {"trade_date": LATER, "window_label": "later_period", "short_book_class": "verified_positive_short"},
        ]
    )
    result = _panel(trades, calendar)
    assert set(result.trades["ticker"]) == {"AAA"}
    zero = result.dates.loc[result.dates["trade_date"].map(lambda value: value == ZERO)].iloc[0]
    assert zero["n_trades"] == 0
    assert zero["p_fly_cross"] == 0.0
    assert zero["ratio_reason"] == "zero body midpoint credit"
    assert LATER not in set(result.dates["trade_date"])
    assert result.verdict == "READY"
    assert result.dollars["p_fly_cross"] == pytest.approx(6.00)


def test_stored_mid_difference_passes_and_pnl_mismatch_fails() -> None:
    shifted = _trade(stored_body_put_mid=2.11)
    passed = _panel([shifted], _calendar((DAY, "verified_positive_short")))
    assert passed.verdict == "READY"
    assert passed.trades.iloc[0]["body_put_mid"] == 2.11
    assert passed.trades.iloc[0]["b_mid"] == pytest.approx(8.60)
    assert passed.report["stored_vs_arithmetic_mid"]["stored_vs_arithmetic_mid_count"] == 1
    assert passed.report["stored_vs_arithmetic_mid"]["stored_vs_arithmetic_mid_max_abs"] == abs(2.11 - 2.10)
    mismatched = _trade(official_pnl=999.0)
    failed = _panel([mismatched], _calendar((DAY, "verified_positive_short")))
    assert not _gate(failed, "identity").passed
    _cannot_be_ready(failed)


def test_zero_numerator_date_ratio_stays_defined() -> None:
    result = _panel(
        [_trade(zero_body_spread=True)],
        _calendar((DAY, "verified_positive_short"), (ZERO, "verified_zero_short")),
    )
    day = result.dates.loc[result.dates["trade_date"].map(lambda value: value == DAY)].iloc[0]
    zero = result.dates.loc[result.dates["trade_date"].map(lambda value: value == ZERO)].iloc[0]
    assert day["h_body"] == pytest.approx(0.0)
    assert day["c_body"] > 0
    assert day["body_concession_ratio"] == pytest.approx(0.0)
    assert day["wing_concession_ratio"] == pytest.approx(day["h_wing"] / day["c_body"])
    assert day["ratio_reason"] == ""
    assert zero["ratio_reason"] == "zero body midpoint credit"
    assert pd.isna(zero["body_concession_ratio"]) or zero["body_concession_ratio"] is None
    assert result.ratios["null_date_ratio_counts"] == {"zero body midpoint credit": 1}
    assert result.verdict == "READY"
    assert readiness_verdict(result.gates) == "READY"


def test_non_boolean_pairing_and_fractional_unit_are_blocked() -> None:
    stringy = _trade()
    stringy["pairing_ok"] = "true"
    string_result = _panel([stringy], _calendar((DAY, "verified_positive_short")))
    assert not _gate(string_result, "inputs").passed
    assert string_result.verdict == "BLOCKED"
    assert readiness_verdict(string_result.gates) == "BLOCKED"

    missing = _trade()
    missing["pairing_ok"] = None
    missing_result = _panel([missing], _calendar((DAY, "verified_positive_short")))
    assert not _gate(missing_result, "inputs").passed
    _cannot_be_ready(missing_result)

    fractional = _trade()
    fractional["body_put_unit_quantity"] = -1.5
    fractional_result = _panel([fractional], _calendar((DAY, "verified_positive_short")))
    assert any("unit_quantity is not exactly -1" in item for item in fractional_result.issues)
    assert not _gate(fractional_result, "inputs").passed
    _cannot_be_ready(fractional_result)


def test_official_d0_receipt_mismatch_blocks_acceptance() -> None:
    receipt = {
        "code_sha": ACCEPTED_D0_CODE_SHA,
        "verdict": ACCEPTED_D0_VERDICT,
        "evidence_dir": str(ACCEPTED_D0_DIR),
    }
    assert d0_receipt_problems(receipt, ACCEPTED_D0_DIR) == []
    assert official_acceptance_verdict(receipt, ACCEPTED_D0_DIR, "READY") == ("READY", [])
    bad = {**receipt, "code_sha": "not-the-accepted-sha", "verdict": "BLOCKED"}
    verdict, problems = official_acceptance_verdict(bad, ACCEPTED_D0_DIR, "READY")
    assert verdict == "BLOCKED"
    assert any("code SHA" in item for item in problems)
    assert any("verdict" in item for item in problems)
