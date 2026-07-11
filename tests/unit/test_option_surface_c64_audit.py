"""Unit tests for C6.4 surface audit helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.features.option_surface_c64_audit import (
    C64ContractAdjustment,
    WeeklyExpiryEvidence,
    adjust_c64_contract_verdict,
    aggregate_c64_verdict,
    build_metrics_normalized_view,
    classify_weekly_expiry_row,
    compute_weekly_expiry_evidence,
    duplicate_verdict,
    legacy_lineage_verdict,
    load_bounded_artifacts,
    triage_duplicates,
    triage_meta_duplicates,
    triage_quote_duplicates,
    weekly_expiry_verdict,
)
from src.features.option_surface_contract import (
    ContractCheckResult,
    check_meta_grain,
    check_quote_grain,
    compute_overall_verdict,
)


def _meta_grain_fail_result() -> ContractCheckResult:
    return check_meta_grain(
        pd.DataFrame(
            [
                {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
                {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            ]
        )
    )


def _quote_grain_fail_result() -> ContractCheckResult:
    return check_quote_grain(
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "entry_date": date(2024, 1, 5),
                    "expiry_date": date(2024, 1, 12),
                    "strike": 100.0,
                    "side": "call",
                },
                {
                    "ticker": "AAPL",
                    "entry_date": date(2024, 1, 5),
                    "expiry_date": date(2024, 1, 12),
                    "strike": 100.0,
                    "side": "call",
                },
            ]
        )
    )


def _settlement_fail_result() -> ContractCheckResult:
    return ContractCheckResult(
        name="settlement_readiness",
        status="FAIL",
        failures=["settlement corruption"],
    )


def test_identical_duplicate_classification() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    triage = triage_meta_duplicates(df)
    assert triage.duplicate_key_count == 1
    assert triage.identical_key_count == 1
    assert triage.groups[0].classification == "IDENTICAL_DUPLICATE"


def test_conflicting_duplicate_classification() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": False},
        ]
    )
    triage = triage_meta_duplicates(df)
    assert triage.conflicting_key_count == 1
    assert "surface_valid" in triage.groups[0].differing_columns


def test_legacy_identical_a1_duplicate_adjusts_contract_to_warn() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    results = [_meta_grain_fail_result()]
    raw = compute_overall_verdict(results)
    assert raw == "FAIL"
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=True)
    assert adj.raw_verdict == "FAIL"
    assert adj.adjusted_verdict == "WARN"
    assert adj.adjustment_notes
    final = aggregate_c64_verdict(adj.adjusted_verdict, "WARN", "PASS", "WARN")
    assert final == "WARN"


def test_legacy_identical_a2_duplicate_adjusts_contract_to_warn() -> None:
    quote_df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "entry_date": date(2024, 1, 5),
                "expiry_date": date(2024, 1, 12),
                "strike": 100.0,
                "side": "call",
            },
            {
                "ticker": "AAPL",
                "entry_date": date(2024, 1, 5),
                "expiry_date": date(2024, 1, 12),
                "strike": 100.0,
                "side": "call",
            },
        ]
    )
    meta_triage = triage_meta_duplicates(pd.DataFrame(columns=["ticker", "entry_date"]))
    quote_triage = triage_quote_duplicates(quote_df)
    results = [_quote_grain_fail_result()]
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=True)
    assert adj.adjusted_verdict == "WARN"


def test_legacy_conflicting_duplicate_stays_fail() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": False},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    results = [_meta_grain_fail_result()]
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=True)
    assert adj.adjusted_verdict == "FAIL"
    dup_status, dup_blocking, _ = duplicate_verdict(meta_triage, quote_triage, legacy_mode=True)
    assert dup_status == "FAIL"
    assert dup_blocking


def test_fresh_identical_duplicate_stays_fail() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "x": 1},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "x": 1},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    results = [_meta_grain_fail_result()]
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=False)
    assert adj.adjusted_verdict == "FAIL"
    dup_status, dup_blocking, _ = duplicate_verdict(meta_triage, quote_triage, legacy_mode=False)
    assert dup_status == "FAIL"


def test_fresh_conflicting_duplicate_stays_fail() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": False},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    adj = adjust_c64_contract_verdict(
        [_meta_grain_fail_result()], meta_triage, quote_triage, legacy_mode=False
    )
    assert adj.adjusted_verdict == "FAIL"


def test_legacy_identical_plus_settlement_failure_stays_fail() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    results = [_meta_grain_fail_result(), _settlement_fail_result()]
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=True)
    assert adj.adjusted_verdict == "FAIL"


def test_legacy_identical_plus_join_failure_stays_fail() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    )
    join_fail = ContractCheckResult(
        name="a1_a2_join_integrity",
        status="FAIL",
        failures=["orphan quote rows"],
    )
    results = [_meta_grain_fail_result(), join_fail]
    adj = adjust_c64_contract_verdict(results, meta_triage, quote_triage, legacy_mode=True)
    assert adj.adjusted_verdict == "FAIL"


def test_exact_target_match_classification() -> None:
    assert (
        classify_weekly_expiry_row(
            expiry=date(2024, 1, 12),
            target=date(2024, 1, 12),
            reason_str=None,
        )
        == "exact_target_match"
    )


def test_silent_mismatch_classification() -> None:
    assert (
        classify_weekly_expiry_row(
            expiry=date(2024, 1, 19),
            target=date(2024, 1, 12),
            reason_str=None,
        )
        == "silent_mismatch"
    )


@pytest.mark.parametrize(
    "reason",
    [
        "no_target_weekly_expiry",
        "target_weekly_expiry_not_listed",
        "no_expiries_on_entry_chain",
    ],
)
def test_documented_target_failures_not_silent(reason: str) -> None:
    assert (
        classify_weekly_expiry_row(expiry=None, target=date(2024, 1, 12), reason_str=reason)
        == "documented_target_failure"
    )


@pytest.mark.parametrize(
    "reason",
    ["no_spot_price", "no_options_at_entry"],
)
def test_other_producer_failures_not_silent(reason: str) -> None:
    assert (
        classify_weekly_expiry_row(expiry=None, target=date(2024, 1, 12), reason_str=reason)
        == "other_producer_failure"
    )


def test_missing_expiry_without_failure_classification() -> None:
    assert (
        classify_weekly_expiry_row(expiry=None, target=date(2024, 1, 12), reason_str=None)
        == "missing_expiry_without_failure"
    )


def test_weekly_expiry_evidence_counts_no_spot_as_other_not_silent(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "adjusted"
    for day in (date(2024, 1, 5), date(2024, 1, 12)):
        path = data_root / str(day.year) / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")

    meta_df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "entry_date": date(2024, 1, 5),
                "expiry_date": None,
                "failure_reason": "no_spot_price",
            }
        ]
    )
    evidence = compute_weekly_expiry_evidence(
        meta_df,
        data_root=data_root,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert evidence.silent_mismatch_count == 0
    assert evidence.other_producer_failure_count == 1


def test_missing_expiry_without_failure_fails_fresh_verdict() -> None:
    evidence = WeeklyExpiryEvidence(
        eligible_row_count=1,
        exact_target_match_count=0,
        silent_mismatch_count=0,
        documented_target_failure_count=0,
        other_producer_failure_count=0,
        missing_expiry_without_failure_count=1,
        no_target_weekly_expiry_count=0,
        target_weekly_expiry_not_listed_count=0,
        no_expiries_on_entry_chain_count=0,
        target_weekly_body_not_quotable_count=0,
    )
    status, blocking, _ = weekly_expiry_verdict(evidence, legacy_mode=False)
    assert status == "FAIL"
    assert blocking


def test_unknown_legacy_lineage_warns() -> None:
    status, _, warnings = legacy_lineage_verdict(
        legacy_mode=True,
        historical_producer_commit=None,
    )
    assert status == "WARN"
    assert warnings


def test_metrics_normalized_view_drops_identical_meta_rows() -> None:
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    quotes_df = pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"])
    meta_triage = triage_meta_duplicates(meta_df)
    quote_triage = triage_quote_duplicates(quotes_df)
    meta_norm, quotes_norm, info = build_metrics_normalized_view(
        meta_df, quotes_df, meta_triage, quote_triage
    )
    assert info.applied is True
    assert info.raw_meta_row_count == 2
    assert info.normalized_meta_row_count == 1
    assert len(meta_norm) == 1
    assert len(quotes_norm) == 0


def test_bounded_load_uses_filter_then_fallback(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.parquet"
    quotes_path = tmp_path / "quotes.parquet"
    meta_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "surface_valid": True},
            {"ticker": "MSFT", "entry_date": date(2024, 1, 5), "surface_valid": True},
        ]
    )
    quotes_df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "entry_date": date(2024, 1, 5),
                "expiry_date": date(2024, 1, 12),
                "strike": 100.0,
                "side": "call",
            },
            {
                "ticker": "MSFT",
                "entry_date": date(2024, 1, 5),
                "expiry_date": date(2024, 1, 12),
                "strike": 200.0,
                "side": "call",
            },
        ]
    )
    meta_df.to_parquet(meta_path, index=False)
    quotes_df.to_parquet(quotes_path, index=False)

    loaded_meta, loaded_quotes = load_bounded_artifacts(
        meta_path,
        quotes_path,
        sample_tickers=["AAPL"],
    )
    assert len(loaded_meta) == 1
    assert loaded_meta.iloc[0]["ticker"] == "AAPL"
    assert len(loaded_quotes) == 1

    real_read = pd.read_parquet

    def _read_with_filter_fallback(path, filters=None):
        if filters is not None:
            raise ValueError("bad filter")
        return real_read(path)

    with patch("src.features.option_surface_c64_audit.pd.read_parquet", side_effect=_read_with_filter_fallback):
        fallback_meta, fallback_quotes = load_bounded_artifacts(
            meta_path,
            quotes_path,
            sample_tickers=["AAPL"],
        )
    assert len(fallback_meta) == 1
    assert len(fallback_quotes) == 1
