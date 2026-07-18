"""Unit tests for src/data/security_types.py (persistent ORATS security-type
dictionary: classification policy, cache behaviour, atomic update)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import security_types as st
from src.data.security_types import (
    SecurityTypesError,
    classification_digest,
    classify_asset_type_history,
    company_equity_tickers,
    ensure_security_types,
    load_security_types,
    snapshot_classification,
    validate_security_types,
)

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc)
CLASSIFIED_AT = "2026-07-18T12:00:00Z"


def _history(ticker: str, rows: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "tradeDate": trade_date, "assetType": asset_type}
            for trade_date, asset_type in rows
        ]
    )


class FetchSpy:
    """Scripted per-ticker Core history; records every fetch call."""

    def __init__(self, histories: dict[str, object]):
        self.histories = histories
        self.calls: list[str] = []

    def __call__(self, ticker: str) -> pd.DataFrame:
        self.calls.append(ticker)
        result = self.histories[ticker]
        if isinstance(result, BaseException):
            raise result
        return result


def _now_fn():
    return NOW


# ── classification policy ─────────────────────────────────────────────────────


class TestClassificationPolicy:
    @pytest.mark.parametrize("asset_type", [0, 1, 2, 3])
    def test_types_0_to_3_are_company_equity(self, asset_type):
        row = classify_asset_type_history(
            "AAA",
            _history("AAA", [("2024-01-02", asset_type)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["classification"] == "company_equity"
        assert row["observed_asset_types"] == str(asset_type)

    @pytest.mark.parametrize("asset_type", [4, 5, 6, 7, 8, 9])
    def test_any_type_4_to_9_is_non_company(self, asset_type):
        row = classify_asset_type_history(
            "SPY",
            _history("SPY", [("2024-01-02", asset_type)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["classification"] == "non_company_equity"

    def test_mixed_history_resolves_non_company_globally(self):
        row = classify_asset_type_history(
            "MIX",
            _history("MIX", [("2024-01-02", 2), ("2024-01-03", 7), ("2024-01-04", 0)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["classification"] == "non_company_equity"
        assert row["observed_asset_types"] == "0,2,7"

    def test_source_date_bounds_and_metadata(self):
        row = classify_asset_type_history(
            "AAA",
            _history("AAA", [("2024-03-01", 0), ("2020-01-02", 1)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["source"] == "orats_core"
        assert row["source_date_min"] == "2020-01-02"
        assert row["source_date_max"] == "2024-03-01"
        assert row["classified_at_utc"] == CLASSIFIED_AT
        assert row["classification_version"] == st.CLASSIFICATION_VERSION

    def test_missing_history_fails(self):
        with pytest.raises(SecurityTypesError, match="No historical"):
            classify_asset_type_history(
                "AAA", pd.DataFrame(), classified_at_utc=CLASSIFIED_AT
            )

    @pytest.mark.parametrize("bad_type", [None, float("nan"), "equity"])
    def test_malformed_asset_type_fails(self, bad_type):
        with pytest.raises(SecurityTypesError, match="missing/malformed assetType"):
            classify_asset_type_history(
                "AAA",
                _history("AAA", [("2024-01-02", bad_type)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    @pytest.mark.parametrize("bad_type", [-1, 10, 42])
    def test_out_of_domain_asset_type_fails(self, bad_type):
        with pytest.raises(SecurityTypesError, match="outside 0-9"):
            classify_asset_type_history(
                "AAA",
                _history("AAA", [("2024-01-02", bad_type)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    def test_non_integer_asset_type_fails(self):
        with pytest.raises(SecurityTypesError, match="non-integer"):
            classify_asset_type_history(
                "AAA",
                _history("AAA", [("2024-01-02", 2.5)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    def test_unexpected_ticker_fails(self):
        with pytest.raises(SecurityTypesError, match="unexpected"):
            classify_asset_type_history(
                "AAA",
                _history("BBB", [("2024-01-02", 0)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    def test_unparseable_trade_date_fails(self):
        with pytest.raises(SecurityTypesError, match="unparseable tradeDate"):
            classify_asset_type_history(
                "AAA",
                _history("AAA", [("not-a-date", 0)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    def test_conflicting_duplicate_trade_date_fails(self):
        with pytest.raises(SecurityTypesError, match="conflicting duplicate"):
            classify_asset_type_history(
                "AAA",
                _history("AAA", [("2024-01-02", 0), ("2024-01-02", 5)]),
                classified_at_utc=CLASSIFIED_AT,
            )

    def test_identical_duplicate_rows_are_deduped(self):
        row = classify_asset_type_history(
            "AAA",
            _history("AAA", [("2024-01-02", 0), ("2024-01-02", 0)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["classification"] == "company_equity"

    def test_ticker_normalization(self):
        row = classify_asset_type_history(
            " aaa ",
            _history("aaa", [("2024-01-02", 0)]),
            classified_at_utc=CLASSIFIED_AT,
        )
        assert row["ticker"] == "AAA"


# ── cache behaviour ───────────────────────────────────────────────────────────


class TestEnsureSecurityTypes:
    def _dict_path(self, tmp_path: Path) -> Path:
        return tmp_path / "orats_security_types.parquet"

    def test_absent_dictionary_fetches_all_candidates_and_writes(self, tmp_path):
        path = self._dict_path(tmp_path)
        fetch = FetchSpy(
            {
                "AAA": _history("AAA", [("2024-01-02", 0)]),
                "SPY": _history("SPY", [("2024-01-02", 5)]),
            }
        )
        result = ensure_security_types(
            ["AAA", "SPY"], path, fetch_history_fn=fetch, now_fn=_now_fn
        )
        assert sorted(fetch.calls) == ["AAA", "SPY"]
        assert path.is_file()
        on_disk = load_security_types(path)
        pd.testing.assert_frame_equal(result, on_disk)
        assert set(result["ticker"]) == {"AAA", "SPY"}
        assert company_equity_tickers(result) == {"AAA"}

    def test_full_coverage_makes_zero_api_calls_and_no_rewrite(self, tmp_path):
        path = self._dict_path(tmp_path)
        seed = FetchSpy({"AAA": _history("AAA", [("2024-01-02", 0)])})
        ensure_security_types(["AAA"], path, fetch_history_fn=seed, now_fn=_now_fn)
        before = path.read_bytes()

        fetch = FetchSpy({})  # any fetch would KeyError
        result = ensure_security_types(
            ["AAA"], path, fetch_history_fn=fetch, now_fn=_now_fn
        )
        assert fetch.calls == []
        assert path.read_bytes() == before
        assert set(result["ticker"]) == {"AAA"}

    def test_fetches_only_missing_tickers(self, tmp_path):
        path = self._dict_path(tmp_path)
        seed = FetchSpy({"AAA": _history("AAA", [("2024-01-02", 0)])})
        ensure_security_types(["AAA"], path, fetch_history_fn=seed, now_fn=_now_fn)

        fetch = FetchSpy({"BBB": _history("BBB", [("2024-01-02", 1)])})
        result = ensure_security_types(
            ["AAA", "BBB"], path, fetch_history_fn=fetch, now_fn=_now_fn
        )
        assert fetch.calls == ["BBB"]
        assert set(result["ticker"]) == {"AAA", "BBB"}

    def test_merge_never_alters_existing_rows(self, tmp_path):
        path = self._dict_path(tmp_path)
        # Seed AAA as non-company; a hypothetical re-fetch would say company.
        seed = FetchSpy({"AAA": _history("AAA", [("2024-01-02", 6)])})
        ensure_security_types(["AAA"], path, fetch_history_fn=seed, now_fn=_now_fn)
        existing = load_security_types(path)

        fetch = FetchSpy(
            {
                "AAA": _history("AAA", [("2024-01-02", 0)]),  # must never be used
                "BBB": _history("BBB", [("2024-01-02", 1)]),
            }
        )
        merged = ensure_security_types(
            ["AAA", "BBB"], path, fetch_history_fn=fetch, now_fn=_now_fn
        )
        assert fetch.calls == ["BBB"]
        merged_aaa = merged[merged["ticker"] == "AAA"].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            merged_aaa, existing[existing["ticker"] == "AAA"].reset_index(drop=True)
        )
        assert merged_aaa.iloc[0]["classification"] == "non_company_equity"

    def test_fetch_failure_leaves_dictionary_byte_for_byte_unchanged(self, tmp_path):
        path = self._dict_path(tmp_path)
        seed = FetchSpy({"AAA": _history("AAA", [("2024-01-02", 0)])})
        ensure_security_types(["AAA"], path, fetch_history_fn=seed, now_fn=_now_fn)
        before = path.read_bytes()

        fetch = FetchSpy({"BBB": RuntimeError("API down")})
        with pytest.raises(RuntimeError, match="API down"):
            ensure_security_types(
                ["AAA", "BBB"], path, fetch_history_fn=fetch, now_fn=_now_fn
            )
        assert path.read_bytes() == before
        assert not list(tmp_path.glob("*.tmp-*"))

    def test_validation_failure_leaves_dictionary_unchanged(self, tmp_path):
        path = self._dict_path(tmp_path)
        seed = FetchSpy({"AAA": _history("AAA", [("2024-01-02", 0)])})
        ensure_security_types(["AAA"], path, fetch_history_fn=seed, now_fn=_now_fn)
        before = path.read_bytes()

        fetch = FetchSpy({"BBB": _history("BBB", [("2024-01-02", 12)])})
        with pytest.raises(SecurityTypesError, match="outside 0-9"):
            ensure_security_types(
                ["AAA", "BBB"], path, fetch_history_fn=fetch, now_fn=_now_fn
            )
        assert path.read_bytes() == before

    def test_partial_batch_failure_publishes_nothing_when_absent(self, tmp_path):
        path = self._dict_path(tmp_path)
        fetch = FetchSpy(
            {
                "AAA": _history("AAA", [("2024-01-02", 0)]),
                "BBB": pd.DataFrame(),  # missing history fails the update
            }
        )
        with pytest.raises(SecurityTypesError, match="No historical"):
            ensure_security_types(
                ["AAA", "BBB"], path, fetch_history_fn=fetch, now_fn=_now_fn
            )
        assert not path.exists()

    def test_empty_candidate_set_fails(self, tmp_path):
        with pytest.raises(SecurityTypesError, match="empty"):
            ensure_security_types(
                [], self._dict_path(tmp_path), fetch_history_fn=FetchSpy({}),
                now_fn=_now_fn,
            )

    def test_corrupt_existing_dictionary_fails_closed(self, tmp_path):
        path = self._dict_path(tmp_path)
        pd.DataFrame({"ticker": ["AAA"], "wrong": [1]}).to_parquet(path, index=False)
        with pytest.raises(SecurityTypesError, match="missing columns"):
            ensure_security_types(
                ["AAA"], path, fetch_history_fn=FetchSpy({}), now_fn=_now_fn
            )


# ── dictionary schema validation ──────────────────────────────────────────────


class TestValidateSecurityTypes:
    def _row(self, **overrides) -> dict:
        row = {
            "ticker": "AAA",
            "classification": "company_equity",
            "observed_asset_types": "0",
            "source": "orats_core",
            "source_date_min": "2024-01-02",
            "source_date_max": "2024-01-03",
            "classified_at_utc": CLASSIFIED_AT,
            "classification_version": 1,
        }
        row.update(overrides)
        return row

    def test_valid_frame_passes(self):
        validate_security_types(pd.DataFrame([self._row()]))

    def test_duplicate_ticker_fails(self):
        df = pd.DataFrame([self._row(), self._row()])
        with pytest.raises(SecurityTypesError, match="duplicate ticker"):
            validate_security_types(df)

    def test_invalid_classification_fails(self):
        df = pd.DataFrame([self._row(classification="etf")])
        with pytest.raises(SecurityTypesError, match="invalid classification"):
            validate_security_types(df)

    def test_classification_inconsistent_with_observed_types_fails(self):
        df = pd.DataFrame(
            [self._row(classification="company_equity", observed_asset_types="0,5")]
        )
        with pytest.raises(SecurityTypesError, match="inconsistent"):
            validate_security_types(df)

    def test_non_normalized_ticker_fails(self):
        df = pd.DataFrame([self._row(ticker=" aaa")])
        with pytest.raises(SecurityTypesError, match="non-normalized"):
            validate_security_types(df)

    def test_malformed_observed_types_fails(self):
        df = pd.DataFrame([self._row(observed_asset_types="0,x")])
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
                "source_date_min": "2024-01-02",
                "source_date_max": "2024-01-03",
                "classified_at_utc": CLASSIFIED_AT,
                "classification_version": 1,
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
