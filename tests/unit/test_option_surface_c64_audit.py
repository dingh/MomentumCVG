"""Unit tests for C6.4 surface audit helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.option_surface_c64_audit import (
    duplicate_verdict,
    triage_duplicates,
    triage_meta_duplicates,
    weekly_expiry_verdict,
    WeeklyExpiryEvidence,
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
    assert triage.conflicting_key_count == 0
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
    assert triage.groups[0].classification == "CONFLICTING_DUPLICATE"
    assert "surface_valid" in triage.groups[0].differing_columns


def test_duplicate_verdict_legacy_vs_fresh() -> None:
    meta = triage_meta_duplicates(
        pd.DataFrame(
            [
                {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "x": 1},
                {"ticker": "AAPL", "entry_date": date(2024, 1, 5), "x": 1},
            ]
        )
    )
    quote = triage_duplicates(
        pd.DataFrame(columns=["ticker", "entry_date", "expiry_date", "strike", "side"]),
        ("ticker", "entry_date", "expiry_date", "strike", "side"),
        grain_label="A2",
    )
    legacy_status, _, legacy_warns = duplicate_verdict(meta, quote, legacy_mode=True)
    fresh_status, fresh_blocking, _ = duplicate_verdict(meta, quote, legacy_mode=False)
    assert legacy_status == "WARN"
    assert legacy_warns
    assert fresh_status == "FAIL"
    assert fresh_blocking


def test_weekly_expiry_verdict_legacy_warn_fresh_fail() -> None:
    evidence = WeeklyExpiryEvidence(
        eligible_row_count=10,
        exact_target_match_count=8,
        silent_mismatch_count=2,
        missing_target_failure_count=0,
        target_not_listed_failure_count=0,
        no_expiries_on_entry_chain_count=0,
    )
    legacy_status, _, legacy_warns = weekly_expiry_verdict(evidence, legacy_mode=True)
    fresh_status, fresh_blocking, _ = weekly_expiry_verdict(evidence, legacy_mode=False)
    assert legacy_status == "WARN"
    assert legacy_warns
    assert fresh_status == "FAIL"
    assert fresh_blocking
