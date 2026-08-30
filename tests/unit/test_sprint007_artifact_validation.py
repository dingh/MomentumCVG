"""Sprint 007 D0 artifact validation tests (synthetic fixtures only)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.sprint007_artifact_validation import (
    ACCEPTED_PRIMARY_INCLUDED_KEYS,
    DECISION_REPORT_ROLES,
    RUN_OUTPUT_NAMES,
    _compare_paired_legs,
    _duplicate_trade_keys,
    _included_trade_keys,
    collect_artifact_inventory,
    expected_cross_fill_price,
    expected_mid_fill_price,
    expected_run_output_path,
    leg_fill_convention_ok,
    sha256_file,
    verify_calendar_completeness_for_fill,
    verify_included_key_parity,
    verify_receipt_integrity,
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
    fill_label: str = "mid",
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
        "fill_label": fill_label,
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
    bid: float = 1.0,
    ask: float = 1.2,
    mid: float | None = None,
) -> dict:
    if mid is None:
        mid = (bid + ask) / 2.0
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


def _write_receipt_run_outputs(run_dir: Path, fill: str) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for name in RUN_OUTPUT_NAMES:
        path = expected_run_output_path(run_dir, name, fill)
        if name == "run_summary":
            path.write_text("{}", encoding="utf-8")
        else:
            pd.DataFrame({"trade_date": [date(2020, 1, 3)]}).to_parquet(path, index=False)
        outputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    return outputs


def _write_valid_receipt(tmp_path: Path, *, extra_mid_run: bool = False, wrong_role: bool = False) -> tuple[Path, dict]:
    run_dir = tmp_path / "official"
    run_dir.mkdir()
    receipt: dict = {
        "repo_sha": "e205b9acc5d0400aa38169de721acb7fb8268f29",
        "result_complete": True,
        "has_unresolved_failures": False,
        "contract": {"sha256": "4012b4a472448004e1a1b14e8814f506911ea0e263e35157b4e13e27ed51a54c"},
        "runs": [],
        "decision_report": {},
    }
    for fill in ("mid", "cross"):
        receipt["runs"].append({"fill_label": fill, "outputs": _write_receipt_run_outputs(run_dir, fill)})
    if extra_mid_run:
        receipt["runs"].append({"fill_label": "mid", "outputs": _write_receipt_run_outputs(run_dir, "mid")})
    if wrong_role:
        receipt["runs"][0]["outputs"]["trade_log"]["path"] = str(run_dir / "wrong_trade_log.parquet")
    for role, filename in (
        ("decision_report_json", "decision_report.json"),
        ("decision_report_md", "decision_report.md"),
    ):
        path = run_dir / filename
        path.write_text("{}", encoding="utf-8")
        receipt["decision_report"][role] = {"path": str(path), "sha256": sha256_file(path)}
    receipt_path = run_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return run_dir, receipt


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


def test_expected_mid_fill_uses_fill_assumption_formula_not_vendor_mid() -> None:
    assert expected_mid_fill_price(1.0, 1.4, 1) == pytest.approx(1.2)
    assert expected_mid_fill_price(1.0, 1.4, -1) == pytest.approx(1.2)


def test_leg_fill_convention_ignores_vendor_mid_when_fill_matches_formula() -> None:
    row = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="mid",
        bid=1.0, ask=1.4, mid=1.25,
    ))
    assert leg_fill_convention_ok(row, fill_label="mid")


def test_leg_fill_convention_fails_when_fill_price_matches_vendor_mid_only() -> None:
    row = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.25, fill_label="mid",
        bid=1.0, ask=1.4, mid=1.25,
    ))
    assert not leg_fill_convention_ok(row, fill_label="mid")


def test_leg_fill_convention_requires_matching_row_fill_label() -> None:
    row = pd.Series(_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="cross",
    ))
    assert not leg_fill_convention_ok(row, fill_label="mid")


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


def test_compare_paired_legs_detects_incorrect_fill_label() -> None:
    trade_keys = {(date(2020, 1, 3), "AAA", "long")}
    mid_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.1, fill_label="cross",
    )])
    cross_legs = pd.DataFrame([_leg_row(
        trade_date=date(2020, 1, 3), ticker="AAA", direction="long",
        leg_index=0, fill_price=1.2, fill_label="cross",
    )])
    result = _compare_paired_legs(mid_legs, cross_legs, trade_keys)
    assert result["mid_fill_label_mismatches"] == 1


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


def test_verify_receipt_integrity_passes_valid_inventory(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path)
    result = verify_receipt_integrity(receipt, run_dir)
    assert result.passed


def test_verify_receipt_integrity_fails_on_missing_cross_run(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path)
    receipt["runs"] = [run for run in receipt["runs"] if run["fill_label"] != "cross"]
    result = verify_receipt_integrity(receipt, run_dir)
    assert not result.passed
    assert "cross run" in result.detail


def test_verify_receipt_integrity_fails_on_missing_output_role(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path)
    del receipt["runs"][0]["outputs"]["trade_log"]
    result = verify_receipt_integrity(receipt, run_dir)
    assert not result.passed
    assert "output roles mismatch" in result.detail


def test_verify_receipt_integrity_fails_on_duplicate_mid_run(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path, extra_mid_run=True)
    result = verify_receipt_integrity(receipt, run_dir)
    assert not result.passed
    assert "mid run" in result.detail


def test_verify_receipt_integrity_fails_on_wrong_artifact_path(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path, wrong_role=True)
    result = verify_receipt_integrity(receipt, run_dir)
    assert not result.passed
    assert "path mismatch" in result.detail


def test_collect_artifact_inventory_counts_all_sprint006_outputs(tmp_path: Path) -> None:
    run_dir, receipt = _write_valid_receipt(tmp_path)
    inventory = collect_artifact_inventory(receipt, run_dir)
    assert len(inventory) == 1 + (len(RUN_OUTPUT_NAMES) * 2) + len(DECISION_REPORT_ROLES)


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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trade_date = date(2020, 1, 3)
    trade = pd.DataFrame([
        _trade_row(trade_date=trade_date, ticker="AAA", direction="long"),
        _trade_row(trade_date=trade_date, ticker="BBB", direction="short"),
    ])
    for label in ("mid", "cross"):
        out = trade.copy()
        out["fill_label"] = label
        out.to_parquet(run_dir / f"trade_log_sprint006_baseline_v1_{label}.parquet", index=False)
        pd.DataFrame({"trade_date": [trade_date], "status": ["traded"], "reason": [""]}).to_parquet(
            run_dir / f"date_status_sprint006_baseline_v1_{label}.parquet", index=False
        )
        pd.DataFrame({
            "trade_date": [trade_date],
            "cycle_pnl_total": [0.0],
            "cycle_return_on_capital_at_risk": [0.0],
            "short_cycle_pnl_total": [0.0],
            "long_cycle_pnl_total": [0.0],
        }).to_parquet(run_dir / f"date_summary_sprint006_baseline_v1_{label}.parquet", index=False)
    result = verify_included_key_parity(run_dir)
    assert result.passed is should_pass
