"""Unit tests for src/data/security_types.py (persistent ORATS security-type
dictionary: date-specific classification policy, bounded valid-empty
fallback, cache behaviour, atomic update, version enforcement)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import security_types as st
from src.data.security_types import (
    SecurityTypesError,
    classification_digest,
    classify_ticker_with_fallback,
    company_equity_tickers,
    ensure_security_types,
    load_security_types,
    snapshot_classification,
    validate_security_types,
)

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc)
CLASSIFIED_AT = "2026-07-18T12:00:00Z"

D1 = date(2024, 1, 3)
D2 = date(2024, 1, 4)
D3 = date(2024, 1, 5)  # newest

EMPTY_OBS = pd.DataFrame(columns=["ticker", "tradeDate", "assetType"])


def _obs(ticker: str, trade_date: date, asset_type: object, n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "tradeDate": trade_date.isoformat(),
                "assetType": asset_type,
            }
        ]
        * n
    )


class FetchSpy:
    """Scripted date-specific responses keyed by (ticker, iso_date)."""

    def __init__(self, responses: dict[tuple[str, str], object]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def __call__(self, ticker: str, trade_date: date) -> pd.DataFrame:
        key = (ticker, trade_date.isoformat())
        self.calls.append(key)
        result = self.responses[key]
        if isinstance(result, BaseException):
            raise result
        return result


def _now_fn():
    return NOW


def _classify(ticker, dates, spy):
    return classify_ticker_with_fallback(
        ticker, dates, spy, classified_at_utc=CLASSIFIED_AT
    )


# ── classification policy (date-specific observation) ─────────────────────────


class TestClassificationPolicy:
    @pytest.mark.parametrize("asset_type", [0, 1, 2, 3])
    def test_types_0_to_3_are_company_equity(self, asset_type):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, asset_type)})
        row = _classify("AAA", [D3], spy)
        assert row["classification"] == "company_equity"
        assert row["observed_asset_types"] == str(asset_type)

    @pytest.mark.parametrize("asset_type", [4, 5, 6, 7, 8, 9])
    def test_any_type_4_to_9_is_non_company(self, asset_type):
        spy = FetchSpy({("SPY", D3.isoformat()): _obs("SPY", D3, asset_type)})
        row = _classify("SPY", [D3], spy)
        assert row["classification"] == "non_company_equity"

    def test_latest_observed_date_is_attempted_first(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, 0)})
        row = _classify("AAA", [D1, D3, D2], spy)  # unordered input
        assert spy.calls == [("AAA", D3.isoformat())]
        assert row["source_date_min"] == D3.isoformat()
        assert row["source_date_max"] == D3.isoformat()

    def test_valid_empty_falls_back_to_earlier_observed_dates_only(self):
        spy = FetchSpy(
            {
                ("DLST", D3.isoformat()): EMPTY_OBS,
                ("DLST", D2.isoformat()): EMPTY_OBS,
                ("DLST", D1.isoformat()): _obs("DLST", D1, 0),
            }
        )
        row = _classify("DLST", [D2, D1, D3], spy)
        # Newest-first attempts; only dates the ticker actually appeared on.
        assert spy.calls == [
            ("DLST", D3.isoformat()),
            ("DLST", D2.isoformat()),
            ("DLST", D1.isoformat()),
        ]
        assert row["classification"] == "company_equity"

    def test_successful_fallback_stores_returned_date(self):
        spy = FetchSpy(
            {
                ("DLST", D3.isoformat()): EMPTY_OBS,
                ("DLST", D2.isoformat()): _obs("DLST", D2, 7),
            }
        )
        row = _classify("DLST", [D3, D2, D1], spy)
        assert row["source_date_min"] == D2.isoformat()
        assert row["source_date_max"] == D2.isoformat()
        assert row["classification"] == "non_company_equity"

    def test_exhausting_bounded_attempts_fails(self):
        days = [date(2024, 1, d) for d in range(2, 10)]  # 8 observed dates
        spy = FetchSpy({("GONE", d.isoformat()): EMPTY_OBS for d in days})
        with pytest.raises(SecurityTypesError, match="cannot classify"):
            _classify("GONE", days, spy)
        # Bounded: latest observed date + 4 fallbacks, newest first.
        assert len(spy.calls) == st.MAX_OBSERVED_DATE_ATTEMPTS
        expected = [
            ("GONE", d.isoformat()) for d in sorted(days, reverse=True)[:5]
        ]
        assert spy.calls == expected

    def test_http_error_fails_without_fallback(self):
        spy = FetchSpy(
            {
                ("AAA", D3.isoformat()): RuntimeError("HTTP 500"),
                ("AAA", D2.isoformat()): _obs("AAA", D2, 0),
            }
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            _classify("AAA", [D3, D2], spy)
        assert spy.calls == [("AAA", D3.isoformat())]

    def test_mismatched_ticker_fails(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("BBB", D3, 0)})
        with pytest.raises(SecurityTypesError, match="unexpected"):
            _classify("AAA", [D3], spy)

    def test_mismatched_trade_date_fails(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D1, 0)})
        with pytest.raises(SecurityTypesError, match="do not match the"):
            _classify("AAA", [D3], spy)

    @pytest.mark.parametrize("bad_type", [None, float("nan"), "equity"])
    def test_malformed_asset_type_fails(self, bad_type):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, bad_type)})
        with pytest.raises(SecurityTypesError, match="missing/malformed assetType"):
            _classify("AAA", [D3], spy)

    @pytest.mark.parametrize("bad_type", [-1, 10, 42])
    def test_out_of_domain_asset_type_fails(self, bad_type):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, bad_type)})
        with pytest.raises(SecurityTypesError, match="outside 0-9"):
            _classify("AAA", [D3], spy)

    def test_non_integer_asset_type_fails(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, 2.5)})
        with pytest.raises(SecurityTypesError, match="non-integer"):
            _classify("AAA", [D3], spy)

    def test_conflicting_rows_on_requested_date_fail(self):
        conflicting = pd.concat(
            [_obs("AAA", D3, 0), _obs("AAA", D3, 5)], ignore_index=True
        )
        spy = FetchSpy({("AAA", D3.isoformat()): conflicting})
        with pytest.raises(SecurityTypesError, match="conflicting duplicate"):
            _classify("AAA", [D3], spy)

    def test_identical_duplicate_rows_are_deduped(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, 0, n=2)})
        row = _classify("AAA", [D3], spy)
        assert row["classification"] == "company_equity"

    def test_ticker_normalization(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("aaa", D3, 0)})
        row = _classify(" aaa ", [D3], spy)
        assert row["ticker"] == "AAA"

    def test_empty_observed_dates_fail(self):
        with pytest.raises(SecurityTypesError, match="empty"):
            _classify("AAA", [], FetchSpy({}))

    def test_none_response_fails(self):
        class NoneFetcher:
            def __call__(self, ticker, trade_date):
                return None

        with pytest.raises(SecurityTypesError, match="returned None"):
            _classify("AAA", [D3], NoneFetcher())

    def test_version_and_metadata(self):
        spy = FetchSpy({("AAA", D3.isoformat()): _obs("AAA", D3, 1)})
        row = _classify("AAA", [D3], spy)
        assert row["source"] == "orats_core"
        assert row["classified_at_utc"] == CLASSIFIED_AT
        assert row["classification_version"] == st.CLASSIFICATION_VERSION == 2


# ── cache behaviour ───────────────────────────────────────────────────────────


class TestEnsureSecurityTypes:
    def _dict_path(self, tmp_path: Path) -> Path:
        return tmp_path / "orats_security_types.parquet"

    def _seed(self, tmp_path: Path, spec: dict[str, int]) -> Path:
        """Seed the dictionary with {ticker: asset_type} observed at D3."""
        path = self._dict_path(tmp_path)
        spy = FetchSpy(
            {(t, D3.isoformat()): _obs(t, D3, at) for t, at in spec.items()}
        )
        ensure_security_types(
            {t: [D3] for t in spec},
            path,
            fetch_observation_fn=spy,
            now_fn=_now_fn,
        )
        return path

    def test_absent_dictionary_classifies_all_candidates_and_writes(self, tmp_path):
        path = self._dict_path(tmp_path)
        spy = FetchSpy(
            {
                ("AAA", D3.isoformat()): _obs("AAA", D3, 0),
                ("SPY", D3.isoformat()): _obs("SPY", D3, 5),
            }
        )
        result = ensure_security_types(
            {"AAA": [D3], "SPY": [D3]},
            path,
            fetch_observation_fn=spy,
            now_fn=_now_fn,
        )
        assert sorted(spy.calls) == [
            ("AAA", D3.isoformat()),
            ("SPY", D3.isoformat()),
        ]
        assert path.is_file()
        on_disk = load_security_types(path)
        pd.testing.assert_frame_equal(result, on_disk)
        assert company_equity_tickers(result) == {"AAA"}

    def test_full_coverage_makes_zero_api_calls_and_no_rewrite(self, tmp_path):
        path = self._seed(tmp_path, {"AAA": 0})
        before = path.read_bytes()

        spy = FetchSpy({})  # any call would KeyError
        result = ensure_security_types(
            {"AAA": [D3, D2]}, path, fetch_observation_fn=spy, now_fn=_now_fn
        )
        assert spy.calls == []
        assert path.read_bytes() == before
        assert set(result["ticker"]) == {"AAA"}

    def test_classifies_only_missing_tickers(self, tmp_path):
        path = self._seed(tmp_path, {"AAA": 0})
        spy = FetchSpy({("BBB", D3.isoformat()): _obs("BBB", D3, 1)})
        result = ensure_security_types(
            {"AAA": [D3], "BBB": [D3]},
            path,
            fetch_observation_fn=spy,
            now_fn=_now_fn,
        )
        assert spy.calls == [("BBB", D3.isoformat())]
        assert set(result["ticker"]) == {"AAA", "BBB"}

    def test_merge_never_alters_existing_rows(self, tmp_path):
        # Seed AAA as non-company; a hypothetical re-fetch would say company.
        path = self._seed(tmp_path, {"AAA": 6})
        existing = load_security_types(path)

        spy = FetchSpy(
            {
                ("AAA", D3.isoformat()): _obs("AAA", D3, 0),  # must never be used
                ("BBB", D3.isoformat()): _obs("BBB", D3, 1),
            }
        )
        merged = ensure_security_types(
            {"AAA": [D3], "BBB": [D3]},
            path,
            fetch_observation_fn=spy,
            now_fn=_now_fn,
        )
        assert spy.calls == [("BBB", D3.isoformat())]
        merged_aaa = merged[merged["ticker"] == "AAA"].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            merged_aaa, existing[existing["ticker"] == "AAA"].reset_index(drop=True)
        )
        assert merged_aaa.iloc[0]["classification"] == "non_company_equity"

    def test_fetch_failure_leaves_dictionary_byte_for_byte_unchanged(self, tmp_path):
        path = self._seed(tmp_path, {"AAA": 0})
        before = path.read_bytes()

        spy = FetchSpy({("BBB", D3.isoformat()): RuntimeError("API down")})
        with pytest.raises(RuntimeError, match="API down"):
            ensure_security_types(
                {"AAA": [D3], "BBB": [D3]},
                path,
                fetch_observation_fn=spy,
                now_fn=_now_fn,
            )
        assert path.read_bytes() == before
        assert not list(tmp_path.glob("*.tmp-*"))

    def test_exhausted_fallback_leaves_dictionary_unchanged(self, tmp_path):
        path = self._seed(tmp_path, {"AAA": 0})
        before = path.read_bytes()

        spy = FetchSpy(
            {
                ("GONE", D3.isoformat()): EMPTY_OBS,
                ("GONE", D2.isoformat()): EMPTY_OBS,
            }
        )
        with pytest.raises(SecurityTypesError, match="cannot classify"):
            ensure_security_types(
                {"AAA": [D3], "GONE": [D3, D2]},
                path,
                fetch_observation_fn=spy,
                now_fn=_now_fn,
            )
        assert path.read_bytes() == before

    def test_partial_batch_failure_publishes_nothing_when_absent(self, tmp_path):
        path = self._dict_path(tmp_path)
        spy = FetchSpy(
            {
                ("AAA", D3.isoformat()): _obs("AAA", D3, 0),
                ("BBB", D3.isoformat()): _obs("CCC", D3, 0),  # mismatched ticker
            }
        )
        with pytest.raises(SecurityTypesError, match="unexpected"):
            ensure_security_types(
                {"AAA": [D3], "BBB": [D3]},
                path,
                fetch_observation_fn=spy,
                now_fn=_now_fn,
            )
        assert not path.exists()

    def test_empty_candidate_mapping_fails(self, tmp_path):
        with pytest.raises(SecurityTypesError, match="empty"):
            ensure_security_types(
                {},
                self._dict_path(tmp_path),
                fetch_observation_fn=FetchSpy({}),
                now_fn=_now_fn,
            )

    def test_candidate_with_no_observed_dates_fails(self, tmp_path):
        with pytest.raises(SecurityTypesError, match="empty"):
            ensure_security_types(
                {"AAA": []},
                self._dict_path(tmp_path),
                fetch_observation_fn=FetchSpy({}),
                now_fn=_now_fn,
            )

    def test_version_1_dictionary_is_rejected(self, tmp_path):
        path = self._dict_path(tmp_path)
        v1 = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "classification": "company_equity",
                    "observed_asset_types": "2,3",
                    "source": "orats_core",
                    "source_date_min": "2007-01-03",
                    "source_date_max": "2026-07-17",
                    "classified_at_utc": CLASSIFIED_AT,
                    "classification_version": 1,
                }
            ]
        )
        v1.to_parquet(path, index=False)
        before = path.read_bytes()
        with pytest.raises(SecurityTypesError, match="rebuild"):
            ensure_security_types(
                {"AAA": [D3]},
                path,
                fetch_observation_fn=FetchSpy({}),
                now_fn=_now_fn,
            )
        assert path.read_bytes() == before  # never deleted or rewritten

    def test_corrupt_existing_dictionary_fails_closed(self, tmp_path):
        path = self._dict_path(tmp_path)
        pd.DataFrame({"ticker": ["AAA"], "wrong": [1]}).to_parquet(path, index=False)
        with pytest.raises(SecurityTypesError, match="missing columns"):
            ensure_security_types(
                {"AAA": [D3]},
                path,
                fetch_observation_fn=FetchSpy({}),
                now_fn=_now_fn,
            )


# ── dictionary schema validation ──────────────────────────────────────────────


class TestValidateSecurityTypes:
    def _row(self, **overrides) -> dict:
        row = {
            "ticker": "AAA",
            "classification": "company_equity",
            "observed_asset_types": "0",
            "source": "orats_core",
            "source_date_min": "2024-01-05",
            "source_date_max": "2024-01-05",
            "classified_at_utc": CLASSIFIED_AT,
            "classification_version": 2,
        }
        row.update(overrides)
        return row

    def test_valid_frame_passes(self):
        validate_security_types(pd.DataFrame([self._row()]))

    def test_version_1_row_fails_with_rebuild_instruction(self):
        df = pd.DataFrame([self._row(classification_version=1)])
        with pytest.raises(SecurityTypesError, match="rebuild"):
            validate_security_types(df)

    def test_multi_type_observed_asset_types_fails(self):
        df = pd.DataFrame([self._row(observed_asset_types="0,3")])
        with pytest.raises(SecurityTypesError, match="exactly one"):
            validate_security_types(df)

    def test_source_date_min_max_must_match(self):
        df = pd.DataFrame([self._row(source_date_min="2024-01-04")])
        with pytest.raises(SecurityTypesError, match="must equal"):
            validate_security_types(df)

    def test_duplicate_ticker_fails(self):
        df = pd.DataFrame([self._row(), self._row()])
        with pytest.raises(SecurityTypesError, match="duplicate ticker"):
            validate_security_types(df)

    def test_invalid_classification_fails(self):
        df = pd.DataFrame([self._row(classification="etf")])
        with pytest.raises(SecurityTypesError, match="invalid classification"):
            validate_security_types(df)

    def test_classification_inconsistent_with_observed_type_fails(self):
        df = pd.DataFrame(
            [self._row(classification="company_equity", observed_asset_types="5")]
        )
        with pytest.raises(SecurityTypesError, match="inconsistent"):
            validate_security_types(df)

    def test_non_normalized_ticker_fails(self):
        df = pd.DataFrame([self._row(ticker=" aaa")])
        with pytest.raises(SecurityTypesError, match="non-normalized"):
            validate_security_types(df)

    def test_malformed_observed_types_fails(self):
        df = pd.DataFrame([self._row(observed_asset_types="x")])
        with pytest.raises(SecurityTypesError, match="malformed"):
            validate_security_types(df)

    def test_invalid_source_fails(self):
        df = pd.DataFrame([self._row(source="vendor_x")])
        with pytest.raises(SecurityTypesError, match="invalid source"):
            validate_security_types(df)


# ── snapshot-local subset and digest ──────────────────────────────────────────


class TestSnapshotClassification:
    def _dictionary(self) -> pd.DataFrame:
        rows = [
            {
                "ticker": t,
                "classification": cls,
                "observed_asset_types": types,
                "source": "orats_core",
                "source_date_min": "2024-01-05",
                "source_date_max": "2024-01-05",
                "classified_at_utc": CLASSIFIED_AT,
                "classification_version": 2,
            }
            for t, cls, types in [
                ("AAA", "company_equity", "0"),
                ("BBB", "company_equity", "1"),
                ("SPY", "non_company_equity", "5"),
            ]
        ]
        return pd.DataFrame(rows)

    def test_subset_contains_exactly_candidates(self):
        subset = snapshot_classification(self._dictionary(), ["SPY", "AAA"])
        assert list(subset["ticker"]) == ["AAA", "SPY"]

    def test_uncovered_candidate_fails(self):
        with pytest.raises(SecurityTypesError, match="missing from"):
            snapshot_classification(self._dictionary(), ["AAA", "ZZZ"])

    def test_digest_deterministic_and_timestamp_free(self):
        subset_a = snapshot_classification(self._dictionary(), ["AAA", "SPY"])
        other = self._dictionary()
        other["classified_at_utc"] = "1999-01-01T00:00:00Z"
        subset_b = snapshot_classification(other, ["SPY", "AAA"])
        assert classification_digest(subset_a) == classification_digest(subset_b)

    def test_digest_changes_with_classification_content(self):
        subset = snapshot_classification(self._dictionary(), ["AAA", "SPY"])
        changed = subset.copy()
        changed.loc[changed["ticker"] == "AAA", "classification"] = (
            "non_company_equity"
        )
        assert classification_digest(subset) != classification_digest(changed)
