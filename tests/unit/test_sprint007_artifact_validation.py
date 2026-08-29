"""Sprint 007 D0 artifact validation tests (synthetic fixtures only)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.sprint007_artifact_validation import (
    ACCEPTED_PRIMARY_INCLUDED_KEYS,
    _compare_paired_legs,
    _duplicate_trade_keys,
    _included_trade_keys,
    verify_included_key_parity,
    verify_unique_trade_keys,
)


def _trade_row(
    *,
    trade_date: date,
    ticker: str,
    direction: str,
    included: bool = True,
    exit_spot: float = 100.0,
) -> dict:
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": included,
        "pnl_total": 1.0,
        "quantity": 10.0,
        "capital_at_risk_dollars": 100.0,
        "return_on_max_loss": 0.01,
        "entry_cost_per_share": 1.0,
        "net_credit_per_share": 0.5,
        "max_loss_per_share": 2.0,
        "spread_cost_ratio": 0.1,
        "leg_spread_to_credit_ratio": 0.2,
        "exit_spot": exit_spot,
        "instrument_type": "ironfly" if direction == "short" else "straddle",
        "fill_label": "mid",
    }


def _leg_row(
    *,
    trade_date: date,
    ticker: str,
    direction: str,
    leg_index: int,
    fill_price: float,
    fill_label: str,
) -> dict:
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "expiry_date": date(2020, 1, 10),
        "option_type": "call",
        "strike": 100.0,
        "leg_index": leg_index,
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "fill_price": fill_price,
        "exit_spot": 100.0,
        "expiry_payoff_per_unit": 0.0,
        "included_in_portfolio": True,
        "fill_label": fill_label,
    }


def test_included_trade_keys_and_duplicate_detection() -> None:
    frame = pd.DataFrame(
        [
            _trade_row(trade_date=date(2020, 1, 3), ticker="AAA", direction="long"),
            _trade_row(trade_date=date(2020, 1, 3), ticker="AAA", direction="long"),
            _trade_row(trade_date=date(2020, 1, 3), ticker="BBB", direction="short"),
        ]
    )
    assert len(_included_trade_keys(frame)) == 2
    assert len(_duplicate_trade_keys(frame)) == 1


def test_compare_paired_legs_detects_quote_mismatch() -> None:
    trade_keys = {(date(2020, 1, 3), "AAA", "long")}
    mid_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="mid",
    )])
    cross_legs = pd.DataFrame([{
        **_leg_row(
            trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
            leg_index=0, fill_price=1.2, fill_label="cross",
        ),
        "ask": 1.5,
    }])
    result = _compare_paired_legs(mid_legs, cross_legs, trade_keys)
    assert result["quote_mismatches"] == 1
    assert result["fill_price_same_count"] == 0


def test_compare_paired_legs_passes_when_quotes_match_and_fill_differs() -> None:
    trade_keys = {(date(2020, 1, 3), "AAA", "long")}
    mid_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="mid",
    )])
    cross_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="cross",
    )])
    result = _compare_paired_legs(mid_legs, cross_legs, trade_keys)
    assert result["quote_mismatches"] == 0
    assert result["settlement_mismatches"] == 0
    assert result["fill_price_same_count"] == 0


def _write_minimal_run_dir(tmp_path: Path, *, include_duplicate: bool = False) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trade_date = date(2020, 1, 3)
    rows = [
        _trade_row(trade_date=trade_date, ticker="AAA", direction="long"),
        _trade_row(trade_date=trade_date, ticker="BBB", direction="short"),
    ]
    if include_duplicate:
        rows.append(_trade_row(trade_date=trade_date, ticker="AAA", direction="long"))
    trade = pd.DataFrame(rows)
    for label in ("mid", "cross"):
        out = trade.copy()
        out["fill_label"] = label
        out.to_parquet(run_dir / f"trade_log_sprint006_baseline_v1_{label}.parquet", index=False)

    status = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "status": ["traded"],
            "reason": [""],
        }
    )
    summary = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "cycle_pnl_total": [0.0],
            "cycle_return_on_capital_at_risk": [0.0],
            "short_cycle_pnl_total": [0.0],
            "long_cycle_pnl_total": [0.0],
        }
    )
    for label in ("mid", "cross"):
        status.to_parquet(run_dir / f"date_status_sprint006_baseline_v1_{label}.parquet", index=False)
        summary.to_parquet(run_dir / f"date_summary_sprint006_baseline_v1_{label}.parquet", index=False)
    return run_dir


def test_verify_unique_trade_keys_fails_on_duplicate(tmp_path: Path) -> None:
    run_dir = _write_minimal_run_dir(tmp_path, include_duplicate=True)
    result = verify_unique_trade_keys(run_dir)
    assert not result.passed


@pytest.mark.parametrize(
    "expected_count,should_pass",
    [(2, True), (ACCEPTED_PRIMARY_INCLUDED_KEYS, False)],
)
def test_verify_included_key_parity_respects_expected_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_count: int,
    should_pass: bool,
) -> None:
    monkeypatch.setattr(
        "src.backtest.sprint007_artifact_validation.ACCEPTED_PRIMARY_INCLUDED_KEYS",
        expected_count,
    )
    run_dir = _write_minimal_run_dir(tmp_path)
    result = verify_included_key_parity(run_dir)
    assert result.passed is should_pass
