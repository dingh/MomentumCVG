"""Unit tests for option surface A1/A2 contract checks (Sprint 004 C6.2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.features.option_surface_analyzer import (
    DOCUMENTED_SURFACE_FAILURE_TAGS,
    _metadata_failure_row,
    _metadata_success_row,
)
from src.features.option_surface_contract import (
    SURFACE_META_REQUIRED_COLUMNS,
    SURFACE_QUOTES_REQUIRED_COLUMNS,
    check_a1_a2_join,
    check_failure_vocabulary,
    check_meta_grain,
    check_quote_grain,
    check_required_columns,
    check_settlement_fields,
    check_surface_valid_invariant,
    check_weekly_date_alignment,
    compute_overall_verdict,
    run_contract_checks,
)


def _meta_row(**overrides) -> dict:
    flags = {
        "has_body_call": overrides.pop("has_body_call", True),
        "has_body_put": overrides.pop("has_body_put", True),
        "n_surface_quotes": overrides.pop("n_surface_quotes", 4),
    }
    base = _metadata_success_row(
        ticker=overrides.pop("ticker", "AAPL"),
        entry_date=overrides.pop("entry_date", date(2024, 1, 5)),
        expiry_date=overrides.pop("expiry_date", date(2024, 1, 12)),
        dte_target=overrides.pop("dte_target", 7),
        frequency=overrides.pop("frequency", "weekly"),
        entry_spot=overrides.pop("entry_spot", 100.0),
        exit_spot=overrides.pop("exit_spot", 101.0),
        body_strike=overrides.pop("body_strike", 100.0),
        spot_move_pct=overrides.pop("spot_move_pct", 0.01),
        realized_volatility=overrides.pop("realized_volatility", 0.2),
        has_body_call=flags["has_body_call"],
        has_body_put=flags["has_body_put"],
        n_surface_quotes=flags["n_surface_quotes"],
        processing_time=overrides.pop("processing_time", 0.1),
        failure_reason=overrides.pop("failure_reason", None),
    )
    base.update(overrides)
    return base


def _quote_row(**overrides) -> dict:
    base = {
        "ticker": "AAPL",
        "entry_date": date(2024, 1, 5),
        "expiry_date": date(2024, 1, 12),
        "entry_spot": 100.0,
        "body_strike": 100.0,
        "side": "call",
        "is_body": True,
        "is_otm": False,
        "strike": 100.0,
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "spread_pct": 0.18,
        "iv": 0.2,
        "delta": 0.5,
        "abs_delta": 0.5,
        "gamma": 0.01,
        "vega": 0.1,
        "theta": -0.01,
        "volume": 100,
        "open_interest": 1000,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides,expected_valid",
    [
        ({}, True),
        ({"has_body_call": False}, False),
        ({"has_body_put": False}, False),
        ({"n_surface_quotes": 0}, False),
        (
            {
                "has_body_call": False,
                "has_body_put": False,
                "n_surface_quotes": 0,
                "failure_reason": "target_weekly_body_not_quotable",
            },
            False,
        ),
    ],
)
def test_surface_valid_invariant_consistent_rows_pass(overrides, expected_valid):
    row = _meta_row(**overrides)
    meta = pd.DataFrame([row])
    result = check_surface_valid_invariant(meta)
    assert row["surface_valid"] is expected_valid
    assert result.status == "PASS"


def test_surface_valid_invariant_mismatch_fails():
    row = _meta_row(has_body_call=False)
    row["surface_valid"] = True
    meta = pd.DataFrame([row])
    result = check_surface_valid_invariant(meta)
    assert result.status == "FAIL"
    assert result.metrics["violation_count"] == 1


def test_surface_valid_invariant_hard_failure_row():
    row = _metadata_failure_row(
        ticker="AAPL",
        entry_date=date(2024, 1, 5),
        dte_target=7,
        frequency="weekly",
        failure_reason="no_spot_price",
        processing_time=0.1,
    )
    meta = pd.DataFrame([row])
    result = check_surface_valid_invariant(meta)
    assert result.status == "PASS"
    assert row["surface_valid"] is False


def test_missing_required_meta_column_fails():
    meta = pd.DataFrame([{k: v for k, v in _meta_row().items() if k != "body_strike"}])
    quotes = pd.DataFrame([_quote_row()])
    result = check_required_columns(meta, quotes)
    assert result.status == "FAIL"
    assert "body_strike" in result.metrics["missing_meta_columns"]


def test_missing_required_quotes_column_fails():
    meta = pd.DataFrame([_meta_row()])
    quotes = pd.DataFrame([{k: v for k, v in _quote_row().items() if k != "mid"}])
    result = check_required_columns(meta, quotes)
    assert result.status == "FAIL"
    assert "mid" in result.metrics["missing_quotes_columns"]


def test_unknown_failure_reason_warns_not_crashes():
    meta = pd.DataFrame(
        [
            _meta_row(
                surface_valid=False,
                has_body_call=False,
                has_body_put=False,
                n_surface_quotes=0,
                failure_reason="mystery_tag",
            )
        ]
    )
    result = check_failure_vocabulary(meta)
    assert result.status == "WARN"
    assert "mystery_tag" in result.warnings[0]


def test_null_failure_reason_on_invalid_row_warns():
    meta = pd.DataFrame(
        [
            _meta_row(
                surface_valid=False,
                has_body_call=False,
                has_body_put=False,
                n_surface_quotes=0,
                failure_reason=None,
            )
        ]
    )
    result = check_failure_vocabulary(meta)
    assert result.status == "WARN"


def test_documented_failure_tags_match_producer_constant():
    for tag in (
        "no_spot_price",
        "no_expiry_found",
        "no_target_weekly_expiry",
        "no_expiries_on_entry_chain",
        "target_weekly_expiry_not_listed",
        "no_options_at_entry",
        "no_strikes_in_chain",
        "no_spot_at_expiry",
        "target_weekly_body_not_quotable",
    ):
        assert tag in DOCUMENTED_SURFACE_FAILURE_TAGS


def test_valid_meta_without_quotes_fails_join_check():
    meta = pd.DataFrame([_meta_row()])
    quotes = pd.DataFrame(columns=list(SURFACE_QUOTES_REQUIRED_COLUMNS))
    result = check_a1_a2_join(meta, quotes)
    assert result.status == "FAIL"
    assert result.metrics["valid_meta_without_quotes_count"] == 1


def test_orphan_quote_row_fails_join_check():
    meta = pd.DataFrame([_meta_row()])
    quotes = pd.DataFrame(
        [
            _quote_row(),
            _quote_row(ticker="MSFT"),
        ]
    )
    result = check_a1_a2_join(meta, quotes)
    assert result.status == "FAIL"
    assert result.metrics["orphan_quote_count"] == 1


def test_invalid_meta_with_quotes_is_warn_not_fail():
    meta = pd.DataFrame(
        [
            _meta_row(
                surface_valid=False,
                has_body_call=False,
                has_body_put=True,
                n_surface_quotes=2,
                failure_reason="target_weekly_body_not_quotable",
            )
        ]
    )
    quotes = pd.DataFrame([_quote_row(), _quote_row(side="put", delta=-0.5, abs_delta=0.5)])
    result = check_a1_a2_join(meta, quotes)
    assert result.status == "WARN"
    assert result.metrics["invalid_meta_with_quotes_count"] == 1


def test_duplicate_quote_grain_fails():
    quotes = pd.DataFrame([_quote_row(), _quote_row()])
    result = check_quote_grain(quotes)
    assert result.status == "FAIL"
    assert result.metrics["duplicate_key_count"] == 1


def test_clean_metadata_grain_passes():
    meta = pd.DataFrame([_meta_row(), _meta_row(ticker="MSFT", entry_date=date(2024, 1, 12))])
    result = check_meta_grain(meta)
    assert result.status == "PASS"
    assert result.metrics["duplicate_key_count"] == 0


def test_duplicate_metadata_grain_fails():
    meta = pd.DataFrame([_meta_row(), _meta_row()])
    result = check_meta_grain(meta)
    assert result.status == "FAIL"
    assert result.metrics["duplicate_key_count"] == 1


def test_duplicate_metadata_grain_different_expiry_still_fails():
    meta = pd.DataFrame(
        [
            _meta_row(expiry_date=date(2024, 1, 12)),
            _meta_row(expiry_date=date(2024, 1, 19)),
        ]
    )
    result = check_meta_grain(meta)
    assert result.status == "FAIL"
    assert result.metrics["duplicate_key_count"] == 1


def test_missing_metadata_grain_column_fails():
    meta = pd.DataFrame([{k: v for k, v in _meta_row().items() if k != "entry_date"}])
    result = check_meta_grain(meta)
    assert result.status == "FAIL"
    assert "entry_date" in result.failures[0]


def test_overall_verdict_fails_when_meta_grain_fails():
    meta = pd.DataFrame([_meta_row(), _meta_row()])
    quotes = pd.DataFrame([_quote_row()])
    results = run_contract_checks(
        meta,
        quotes,
        frequency="monthly",
    )
    assert check_meta_grain(meta).status == "FAIL"
    assert compute_overall_verdict(results) == "FAIL"


def test_valid_row_dte_actual_mismatch_fails():
    meta = pd.DataFrame([_meta_row(dte_actual=99)])
    result = check_settlement_fields(meta)
    assert result.status == "FAIL"
    assert result.metrics["dte_mismatch_count"] == 1


def test_weekly_entry_date_outside_schedule_warns(tmp_path: Path):
    friday = date(2024, 1, 5)
    path = tmp_path / "2024" / f"ORATS_SMV_Strikes_{friday.strftime('%Y%m%d')}.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")

    meta = pd.DataFrame([_meta_row(entry_date=date(2024, 1, 4))])
    result = check_weekly_date_alignment(
        meta,
        frequency="weekly",
        data_root=tmp_path,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert result.status == "WARN"
    assert result.metrics["misaligned_entry_count"] == 1


def test_compute_overall_verdict_priority():
    pass_result = type("R", (), {"status": "PASS"})()
    warn_result = type("R", (), {"status": "WARN"})()
    fail_result = type("R", (), {"status": "FAIL"})()
    assert compute_overall_verdict([pass_result]) == "PASS"
    assert compute_overall_verdict([pass_result, warn_result]) == "WARN"
    assert compute_overall_verdict([warn_result, fail_result]) == "FAIL"


def test_required_column_sets_cover_contract_minimum():
    assert "surface_valid" in SURFACE_META_REQUIRED_COLUMNS
    assert "failure_reason" in SURFACE_META_REQUIRED_COLUMNS
    assert {"bid", "ask", "mid", "side", "strike"} <= SURFACE_QUOTES_REQUIRED_COLUMNS
