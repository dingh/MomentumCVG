"""Unit tests for option surface assembly-readiness (Sprint 004 C6.3)."""

from __future__ import annotations

from datetime import date

import pytest

from src.features.option_surface_analyzer import _metadata_success_row
from src.features.option_surface_readiness import (
    DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
    QuoteRecord,
    compute_readiness_verdict,
    compute_surface_readiness,
    count_ironcondor_candidates,
    count_symmetric_ironfly_pairs,
    expected_is_otm,
    is_otm_call_wing,
    is_quotable,
    quote_from_mapping,
    run_readiness_audit,
)


def _meta(**overrides) -> dict:
    base = _metadata_success_row(
        ticker=overrides.pop("ticker", "AAPL"),
        entry_date=overrides.pop("entry_date", date(2024, 1, 5)),
        expiry_date=overrides.pop("expiry_date", date(2024, 1, 12)),
        dte_target=7,
        frequency="weekly",
        entry_spot=100.0,
        exit_spot=101.0,
        body_strike=overrides.pop("body_strike", 100.0),
        spot_move_pct=0.01,
        realized_volatility=0.2,
        has_body_call=overrides.pop("has_body_call", True),
        has_body_put=overrides.pop("has_body_put", True),
        n_surface_quotes=overrides.pop("n_surface_quotes", 4),
        processing_time=0.1,
        failure_reason=overrides.pop("failure_reason", None),
    )
    base.update(overrides)
    return base


def _quote(**overrides) -> dict:
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


def _full_surface_quotes(body_strike: float = 100.0) -> list[dict]:
    return [
        _quote(side="call", is_body=True, is_otm=False, strike=body_strike),
        _quote(side="put", is_body=True, is_otm=False, strike=body_strike),
        _quote(side="call", is_body=False, is_otm=True, strike=body_strike + 5, abs_delta=0.25),
        _quote(side="put", is_body=False, is_otm=True, strike=body_strike - 5, abs_delta=0.25),
    ]


class TestBodyStraddleReadiness:
    def test_valid_body_pair_ready(self) -> None:
        row = compute_surface_readiness(_meta(), _full_surface_quotes())
        assert row.body_pair_ready is True
        assert row.straddle_ready is True

    def test_missing_body_call(self) -> None:
        quotes = [q for q in _full_surface_quotes() if q["side"] != "call" or not q["is_body"]]
        row = compute_surface_readiness(_meta(has_body_call=False), quotes)
        assert row.body_pair_ready is False
        assert "missing_body_call" in row.readiness_failure_reasons

    def test_missing_body_put(self) -> None:
        quotes = [q for q in _full_surface_quotes() if q["side"] != "put" or not q["is_body"]]
        row = compute_surface_readiness(_meta(has_body_put=False), quotes)
        assert row.body_pair_ready is False
        assert "missing_body_put" in row.readiness_failure_reasons

    def test_non_positive_body_quote_not_quotable(self) -> None:
        quotes = _full_surface_quotes()
        quotes[0]["bid"] = 0.0
        row = compute_surface_readiness(_meta(), quotes)
        assert row.body_pair_ready is False

    def test_surface_valid_true_missing_body_pair_is_fail(self) -> None:
        quotes = [q for q in _full_surface_quotes() if not q["is_body"]]
        row = compute_surface_readiness(_meta(surface_valid=True), quotes)
        assert "surface_valid_body_contradiction" in row.consistency_failures
        verdict = compute_readiness_verdict([row], contract_passed=True)
        assert verdict.status == "FAIL"

    def test_a1_body_flags_disagree_with_a2(self) -> None:
        quotes = _full_surface_quotes()
        quotes = [q for q in quotes if q["side"] != "call" or not q["is_body"]]
        row = compute_surface_readiness(_meta(has_body_call=True), quotes)
        assert "body_flag_mismatch" in row.consistency_failures

    def test_duplicate_body_call_rows_fail(self) -> None:
        quotes = _full_surface_quotes()
        quotes.append(dict(quotes[0]))
        row = compute_surface_readiness(_meta(), quotes)
        assert "duplicate_body_call" in row.consistency_failures

    def test_body_strike_mismatch_fail(self) -> None:
        quotes = _full_surface_quotes()
        quotes[0]["strike"] = 101.0
        row = compute_surface_readiness(_meta(body_strike=100.0), quotes)
        assert "body_strike_mismatch" in row.consistency_failures


class TestOtmWingReadiness:
    def test_otm_wing_pair_available(self) -> None:
        row = compute_surface_readiness(_meta(), _full_surface_quotes())
        assert row.otm_wing_pair_available is True

    def test_only_call_wing(self) -> None:
        quotes = [q for q in _full_surface_quotes() if q["side"] != "put" or q["is_body"]]
        row = compute_surface_readiness(_meta(), quotes)
        assert row.otm_wing_pair_available is False
        assert "no_otm_put_wing" in row.readiness_failure_reasons

    def test_only_put_wing(self) -> None:
        quotes = [q for q in _full_surface_quotes() if q["side"] != "call" or q["is_body"]]
        row = compute_surface_readiness(_meta(), quotes)
        assert row.otm_wing_pair_available is False
        assert "no_otm_call_wing" in row.readiness_failure_reasons

    def test_zero_bid_wing_not_quotable(self) -> None:
        quotes = _full_surface_quotes()
        quotes[2]["bid"] = 0.0
        row = compute_surface_readiness(_meta(), quotes)
        assert row.otm_call_wing_available is False

    def test_incorrect_is_otm_classification_detected(self) -> None:
        quotes = _full_surface_quotes()
        quotes[2]["is_otm"] = False
        row = compute_surface_readiness(_meta(), quotes)
        assert "classification_mismatch" in row.consistency_failures


class TestIronFlyReadiness:
    def test_exact_symmetric_pair_ready(self) -> None:
        row = compute_surface_readiness(_meta(), _full_surface_quotes())
        assert row.ironfly_candidate_ready is True
        assert row.ironfly_candidate_pair_count == 1

    def test_asymmetric_wings_only_not_ready(self) -> None:
        quotes = _full_surface_quotes()
        quotes[3]["strike"] = 97.0  # 3 vs 5 from body
        row = compute_surface_readiness(_meta(), quotes, ironfly_symmetry_tolerance=0.0)
        assert row.ironfly_candidate_ready is False
        assert "no_symmetric_ironfly_pair" in row.readiness_failure_reasons

    def test_multiple_symmetric_pairs_count(self) -> None:
        quotes = _full_surface_quotes()
        quotes.extend(
            [
                _quote(side="call", is_body=False, is_otm=True, strike=110.0, abs_delta=0.15),
                _quote(side="put", is_body=False, is_otm=True, strike=90.0, abs_delta=0.15),
            ]
        )
        row = compute_surface_readiness(_meta(), quotes)
        assert row.ironfly_candidate_pair_count == 2
        assert row.ironfly_candidate_ready is True

    def test_body_pair_missing_ironfly_not_ready(self) -> None:
        quotes = [q for q in _full_surface_quotes() if not q["is_body"]]
        row = compute_surface_readiness(_meta(has_body_call=False, has_body_put=False), quotes)
        assert row.ironfly_candidate_ready is False


class TestIronCondorReadiness:
    def test_minimum_valid_leg_configuration(self) -> None:
        row = compute_surface_readiness(_meta(), _full_surface_quotes())
        assert row.ironcondor_candidate_ready is True
        assert row.ironcondor_candidate_count >= 1

    def test_insufficient_put_side_legs(self) -> None:
        quotes = [
            _quote(side="call", is_body=True, is_otm=False, strike=100.0),
            _quote(side="put", is_body=True, is_otm=False, strike=100.0),
            _quote(side="call", is_body=False, is_otm=True, strike=105.0, abs_delta=0.25),
            _quote(side="call", is_body=False, is_otm=True, strike=110.0, abs_delta=0.15),
        ]
        row = compute_surface_readiness(_meta(), quotes)
        assert row.ironcondor_candidate_ready is False
        assert "insufficient_condor_legs" in row.readiness_failure_reasons

    def test_insufficient_call_side_legs(self) -> None:
        quotes = [
            _quote(side="call", is_body=True, is_otm=False, strike=100.0),
            _quote(side="put", is_body=True, is_otm=False, strike=100.0),
            _quote(side="put", is_body=False, is_otm=True, strike=95.0, abs_delta=0.25),
            _quote(side="put", is_body=False, is_otm=True, strike=90.0, abs_delta=0.15),
        ]
        row = compute_surface_readiness(_meta(), quotes)
        assert row.ironcondor_candidate_ready is False

    def test_single_strike_per_side_no_condor_vertical(self) -> None:
        """Body-only quotable strikes cannot form inner/outer vertical spreads."""
        quotes = [
            _quote(side="call", is_body=True, is_otm=False, strike=100.0),
            _quote(side="put", is_body=True, is_otm=False, strike=100.0),
        ]
        row = compute_surface_readiness(_meta(), quotes)
        assert row.ironcondor_candidate_ready is False
        assert row.ironcondor_candidate_count == 0


class TestIntegrationVerdict:
    def test_clean_synthetic_produces_pass(self) -> None:
        row = compute_surface_readiness(_meta(), _full_surface_quotes())
        verdict = compute_readiness_verdict([row], contract_passed=True)
        assert verdict.status == "PASS"

    def test_lack_of_wings_warn_not_fail(self) -> None:
        quotes = [
            _quote(side="call", is_body=True, is_otm=False, strike=100.0),
            _quote(side="put", is_body=True, is_otm=False, strike=100.0),
        ]
        row = compute_surface_readiness(_meta(), quotes)
        verdict = compute_readiness_verdict([row], contract_passed=True)
        assert verdict.status in ("PASS", "WARN")
        assert verdict.status != "FAIL"

    def test_body_inconsistency_produces_fail(self) -> None:
        quotes = [
            _quote(side="put", is_body=True, is_otm=False, strike=100.0),
        ]
        row = compute_surface_readiness(_meta(has_body_call=True), quotes)
        verdict = compute_readiness_verdict([row], contract_passed=True)
        assert verdict.status == "FAIL"

    def test_report_metrics_include_conditional_rates(self) -> None:
        rows = [
            compute_surface_readiness(_meta(), _full_surface_quotes()),
            compute_surface_readiness(
                _meta(ticker="MSFT", surface_valid=False, has_body_call=False, has_body_put=False),
                [],
            ),
        ]
        verdict = compute_readiness_verdict(rows, contract_passed=True)
        assert "straddle_ready_among_surface_valid_rate" in verdict.metrics
        assert "ironfly_candidate_ready_among_surface_valid_rate" in verdict.metrics

    def test_contract_failure_blocks_readiness(self) -> None:
        verdict = run_readiness_audit([], {}, contract_passed=False)
        assert verdict.blocked is True
        assert verdict.status == "FAIL"


@pytest.mark.parametrize(
    "bid,ask,mid,expected",
    [
        (1.0, 1.2, 1.1, True),
        (0.0, 1.2, 1.1, False),
        (-1.0, 1.0, 0.5, False),
    ],
)
def test_is_quotable(bid: float, ask: float, mid: float, expected: bool) -> None:
    assert is_quotable(bid, ask, mid) is expected


def test_symmetric_pair_tolerance() -> None:
    call = quote_from_mapping(_quote(side="call", strike=105.0, is_body=False, is_otm=True))
    put = quote_from_mapping(_quote(side="put", strike=97.0, is_body=False, is_otm=True))
    assert count_symmetric_ironfly_pairs([call], [put], 100.0, symmetry_tolerance=0.0) == 0
    assert count_symmetric_ironfly_pairs([call], [put], 100.0, symmetry_tolerance=3.0) == 1


def test_expected_is_otm_call_put() -> None:
    assert expected_is_otm("call", 105.0, 100.0) is True
    assert expected_is_otm("put", 95.0, 100.0) is True
    assert expected_is_otm("call", 100.0, 100.0) is False


def test_is_otm_call_wing_requires_all_conditions() -> None:
    q = QuoteRecord(
        side="call",
        strike=105.0,
        body_strike=100.0,
        bid=1.0,
        ask=1.2,
        mid=1.1,
        is_body=False,
        is_otm=True,
    )
    assert is_otm_call_wing(q) is True
    q_bad = QuoteRecord(
        side="call",
        strike=105.0,
        body_strike=100.0,
        bid=0.0,
        ask=1.2,
        mid=1.1,
        is_body=False,
        is_otm=True,
    )
    assert is_otm_call_wing(q_bad) is False
