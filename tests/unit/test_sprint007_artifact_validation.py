"""Sprint 007 D0 artifact validation tests (synthetic fixtures only)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.sprint007_artifact_validation import (
    ACCEPTED_PRIMARY_INCLUDED_KEYS,
    RUN_OUTPUT_NAMES,
    _compare_paired_legs,
    _duplicate_trade_keys,
    _included_trade_keys,
    collect_artifact_inventory,
    expected_cross_fill_price,
    leg_fill_convention_ok,
    verify_calendar_completeness_for_fill,
    verify_included_key_parity,
    verify_unique_trade_keys,
    write_execution_receipt,
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
    unit_quantity: int = 1,
    portfolio_quantity: float = 10.0,
) -> dict:
    bid, ask, mid = 1.0, 1.2, 1.1
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "expiry_date": date(2020, 1, 10),
        "option_type": "call",
        "strike": 100.0,
        "leg_index": leg_index,
        "unit_quantity": unit_quantity,
        "portfolio_quantity": portfolio_quantity,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "fill_price": fill_price,
        "entry_cash_per_unit": fill_price * abs(unit_quantity),
        "pnl_per_unit": 0.0,
        "pnl_total_leg": 0.0,
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


def test_expected_cross_fill_price_buy_and_sell() -> None:
    assert expected_cross_fill_price(1.0, 1.2, 1) == 1.2
    assert expected_cross_fill_price(1.0, 1.2, -1) == 1.0


def test_leg_fill_convention_mid_and_cross() -> None:
    mid_row = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="mid", unit_quantity=1,
    ))
    cross_buy = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="cross", unit_quantity=1,
    ))
    cross_sell = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="BBB", direction="short",
        leg_index=0, fill_price=1.0, fill_label="cross", unit_quantity=-1,
    ))
    assert leg_fill_convention_ok(mid_row, fill_label="mid")
    assert leg_fill_convention_ok(cross_buy, fill_label="cross")
    assert leg_fill_convention_ok(cross_sell, fill_label="cross")
    assert not leg_fill_convention_ok(cross_buy, fill_label="mid")


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


def test_compare_paired_legs_detects_unit_quantity_mismatch() -> None:
    trade_keys = {(date(2020, 1, 3), "AAA", "long")}
    mid_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="mid", unit_quantity=1,
        portfolio_quantity=10.0,
    )])
    cross_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="cross", unit_quantity=2,
        portfolio_quantity=20.0,
    )])
    result = _compare_paired_legs(mid_legs, cross_legs, trade_keys)
    assert result["unit_quantity_mismatches"] == 1


def test_compare_paired_legs_allows_different_portfolio_quantity() -> None:
    trade_keys = {(date(2020, 1, 3), "AAA", "long")}
    mid_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="mid", portfolio_quantity=10.0,
    )])
    cross_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="cross", portfolio_quantity=99.0,
    )])
    result = _compare_paired_legs(mid_legs, cross_legs, trade_keys)
    assert result["unit_quantity_mismatches"] == 0
    assert result["cross_fill_convention_violations"] == 0
    assert result["mid_fill_convention_violations"] == 0


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


def test_collect_artifact_inventory_counts_all_sprint006_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "official"
    run_dir.mkdir()
    receipt_path = run_dir / "run_receipt.json"
    receipt = {"runs": [], "decision_report": {}}
    outputs: dict[str, dict[str, str]] = {}
    for fill in ("mid", "cross"):
        run_outputs: dict[str, dict[str, str]] = {}
        for name in RUN_OUTPUT_NAMES:
            path = run_dir / f"{name}_sprint006_baseline_v1_{fill}.parquet"
            pd.DataFrame({"trade_date": [date(2020, 1, 3)]}).to_parquet(path, index=False)
            run_outputs[name] = {"path": str(path), "sha256": "abc"}
        receipt["runs"].append({"fill_label": fill, "outputs": run_outputs})
        outputs.update(run_outputs)
    for name in ("decision_report_json", "decision_report_md"):
        path = run_dir / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        receipt["decision_report"][name] = {"path": str(path), "sha256": "abc"}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    inventory = collect_artifact_inventory(receipt, run_dir)
    assert len(inventory) == 1 + (len(RUN_OUTPUT_NAMES) * 2) + 2


def test_write_execution_receipt_preserves_distinct_repo_identities(tmp_path: Path) -> None:
    notebook = tmp_path / "nb.ipynb"
    html = tmp_path / "nb.html"
    notebook.write_text("{}", encoding="utf-8")
    html.write_text("<html></html>", encoding="utf-8")
    path = write_execution_receipt(
        evidence_dir=tmp_path,
        executed_notebook=notebook,
        html_export=html,
        d0_code_commit_sha="d0sha123",
        sprint006_execution_repo_sha="sprint006sha",
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["d0_code_commit_sha"] == "d0sha123"
    assert receipt["sprint006_execution_repo_sha"] == "sprint006sha"
    assert "repo_sha" not in receipt
