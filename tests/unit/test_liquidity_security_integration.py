"""Integration tests: date-specific security-type stage inside the C4
liquidity build (scripts/build_liquidity_panel.py).

Non-company tickers (ORATS Core assetType 4-9 at the accepted observation)
must be removed from candidate daily data before weekly / panel / rank
construction; every Core request must be date-specific; existing dictionary
tickers make zero API requests; and every final liquidity artifact must be
company-equity only. The HTTP boundary is always mocked via an injected
fetch_observation_fn.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = PROJECT_ROOT / "scripts" / "build_liquidity_panel.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_liquidity_panel_sec", CLI_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_liquidity_panel_sec"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


blp = _load_module()

from src.data.security_types import (  # noqa: E402
    SecurityTypesError,
    classification_digest,
    ensure_security_types,
    load_security_types,
)

EQUITY_TICKERS = ["AL1", "AL2", "AL3", "AL4", "AL5"]
ETF_TICKER = "SPY"
INDEX_TICKER = "VIXY"  # non-company at its accepted observation
DELISTED_TICKER = "DLST"  # valid-empty on newest dates; row on the oldest
ALL_TICKERS = EQUITY_TICKERS + [ETF_TICKER, INDEX_TICKER, DELISTED_TICKER]
COMPANY_TICKERS = EQUITY_TICKERS + [DELISTED_TICKER]

TRADING_DAYS = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
LATEST_DAY = TRADING_DAYS[-1]
EXPIRY = date(2024, 1, 12)

EMPTY_OBS = pd.DataFrame(columns=["ticker", "tradeDate", "assetType"])


def _orats_row(ticker: str, trade_date: date, dollar_scale: float = 1.0) -> dict:
    return {
        "ticker": ticker,
        "expirDate": EXPIRY,
        "strike": 100.0,
        "stkPx": 100.0,
        "cBidPx": 1.0 * dollar_scale,
        "cAskPx": 1.1 * dollar_scale,
        "pBidPx": 1.0 * dollar_scale,
        "pAskPx": 1.1 * dollar_scale,
        "cVolu": 10.0,
        "pVolu": 10.0,
    }


def _load_day_fn(trade_date: date) -> pd.DataFrame:
    if trade_date not in TRADING_DAYS:
        return pd.DataFrame()
    rows = [
        _orats_row(t, trade_date, dollar_scale=float(i + 1))
        for i, t in enumerate(ALL_TICKERS)
    ]
    return pd.DataFrame(rows)


def _obs(ticker: str, trade_date: date, asset_type: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "tradeDate": trade_date.isoformat(),
                "assetType": asset_type,
            }
        ]
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


def _core_responses() -> dict[tuple[str, str], object]:
    """Date-specific mocked Core responses for the fixture universe.

    Every active ticker resolves at its latest observed date. The synthetic
    delisted ticker returns valid-empty on the two newest observed dates and
    a company-equity row on the oldest.
    """
    responses: dict[tuple[str, str], object] = {
        (t, LATEST_DAY.isoformat()): _obs(t, LATEST_DAY, 0)
        for t in EQUITY_TICKERS
    }
    responses[(ETF_TICKER, LATEST_DAY.isoformat())] = _obs(
        ETF_TICKER, LATEST_DAY, 5
    )
    responses[(INDEX_TICKER, LATEST_DAY.isoformat())] = _obs(
        INDEX_TICKER, LATEST_DAY, 7
    )
    responses[(DELISTED_TICKER, TRADING_DAYS[2].isoformat())] = EMPTY_OBS
    responses[(DELISTED_TICKER, TRADING_DAYS[1].isoformat())] = EMPTY_OBS
    responses[(DELISTED_TICKER, TRADING_DAYS[0].isoformat())] = _obs(
        DELISTED_TICKER, TRADING_DAYS[0], 0
    )
    return responses


def _run_build(tmp_path: Path, fetch) -> blp.BuildResult:
    return blp.build_panel(
        tmp_path / "raw",
        tmp_path / "cache",
        LATEST_DAY,
        LATEST_DAY,
        mode="backfill",
        lookback_weeks=12,
        min_valid_quote_weeks=1,
        load_day_fn=_load_day_fn,
        show_progress=False,
        security_types_path=tmp_path / "reference" / "orats_security_types.parquet",
        fetch_observation_fn=fetch,
    )


def _write_all_artifacts(tmp_path: Path, result: blp.BuildResult) -> Path:
    """Mirror main(): stamp params, build liquid tickers, write atomically."""
    cache = tmp_path / "cache"
    result.panel = blp.stamp_panel_universe_params(
        result.panel, dvol_top_pct=0.4, spread_bot_pct=1.0
    )
    universe = blp.build_liquid_tickers(result.panel, 0.4, 1.0)
    blp.write_artifacts(cache, result, liquid_tickers=universe)
    return cache


# Monkeypatch discovery: build_panel discovers trading dates from the raw
# root; the fixture data is injected via load_day_fn, so discovery is stubbed.
@pytest.fixture(autouse=True)
def _stub_discovery(monkeypatch):
    monkeypatch.setattr(
        blp, "discover_orats_trading_dates", lambda root, start, end: TRADING_DAYS
    )


class TestCandidateTickerDates:
    def test_dates_normalized_distinct_newest_first_regardless_of_order(self):
        rows = [
            {"trade_date": d, "ticker": t}
            for t in ("bbb", "AAA")
            for d in (TRADING_DAYS[1], TRADING_DAYS[0], TRADING_DAYS[2])
        ]
        # Duplicate rows and reversed/shuffled orderings must not matter.
        forward = pd.DataFrame(rows + rows[:2])
        backward = pd.DataFrame(list(reversed(rows)) + rows[-2:])

        expected = {
            "AAA": [TRADING_DAYS[2], TRADING_DAYS[1], TRADING_DAYS[0]],
            "BBB": [TRADING_DAYS[2], TRADING_DAYS[1], TRADING_DAYS[0]],
        }
        assert blp.candidate_ticker_dates_from_daily(forward) == expected
        assert blp.candidate_ticker_dates_from_daily(backward) == expected


class TestCompanyEquityFilterInBuild:
    def test_non_company_removed_before_weekly_panel_and_ranks(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        result = _run_build(tmp_path, fetch)

        # Removed from daily before weekly/panel construction.
        assert set(result.daily["ticker"]) == set(COMPANY_TICKERS)
        assert set(result.weekly["ticker"]) == set(COMPANY_TICKERS)
        assert set(result.panel["ticker"]) == set(COMPANY_TICKERS)
        for banned in (ETF_TICKER, INDEX_TICKER):
            assert banned not in set(result.weekly["ticker"])
            assert banned not in set(result.panel["ticker"])

    def test_non_company_absent_from_every_final_artifact(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        result = _run_build(tmp_path, fetch)
        cache = _write_all_artifacts(tmp_path, result)

        daily = pd.read_parquet(cache / blp.DAILY_FILENAME)
        weekly = pd.read_parquet(cache / blp.WEEKLY_FILENAME)
        panel = pd.read_parquet(cache / blp.PANEL_FILENAME)
        liquid = pd.read_csv(cache / blp.LIQUID_TICKERS_FILENAME)

        for banned in (ETF_TICKER, INDEX_TICKER):
            assert banned not in set(daily["ticker"])
            assert banned not in set(weekly["ticker"])
            assert banned not in set(panel["ticker"])
            assert banned not in set(liquid["Ticker"])
        # Historical superset (liquid_tickers.csv) still ranks company names.
        assert not liquid.empty
        assert set(liquid["Ticker"]) <= set(COMPANY_TICKERS)

    def test_every_core_request_is_date_specific(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        _run_build(tmp_path, fetch)

        assert fetch.calls, "expected at least one Core request"
        for ticker, iso_date in fetch.calls:
            assert iso_date in {d.isoformat() for d in TRADING_DAYS}
        # Active tickers were resolved at their latest observed date.
        for t in [*EQUITY_TICKERS, ETF_TICKER, INDEX_TICKER]:
            assert (t, LATEST_DAY.isoformat()) in fetch.calls

    def test_delisted_ticker_falls_back_and_stores_returned_date(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        result = _run_build(tmp_path, fetch)

        # Newest-first fallback over dates the ticker actually appeared on.
        dlst_calls = [c for c in fetch.calls if c[0] == DELISTED_TICKER]
        assert dlst_calls == [
            (DELISTED_TICKER, TRADING_DAYS[2].isoformat()),
            (DELISTED_TICKER, TRADING_DAYS[1].isoformat()),
            (DELISTED_TICKER, TRADING_DAYS[0].isoformat()),
        ]

        dictionary = load_security_types(
            tmp_path / "reference" / "orats_security_types.parquet"
        )
        row = dictionary[dictionary["ticker"] == DELISTED_TICKER].iloc[0]
        assert row["classification"] == "company_equity"
        assert row["source_date_min"] == TRADING_DAYS[0].isoformat()
        assert row["source_date_max"] == TRADING_DAYS[0].isoformat()
        # And it participates in liquidity like any other company equity.
        assert DELISTED_TICKER in set(result.daily["ticker"])

    def test_dictionary_covers_all_candidates_after_build(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        _run_build(tmp_path, fetch)

        called_tickers = {t for t, _ in fetch.calls}
        assert called_tickers == set(ALL_TICKERS)
        dictionary = load_security_types(
            tmp_path / "reference" / "orats_security_types.parquet"
        )
        assert set(dictionary["ticker"]) == set(ALL_TICKERS)

    def test_covering_dictionary_makes_zero_api_calls(self, tmp_path):
        security_path = tmp_path / "reference" / "orats_security_types.parquet"
        seed = FetchSpy(_core_responses())
        ensure_security_types(
            {
                **{t: [LATEST_DAY] for t in ALL_TICKERS if t != DELISTED_TICKER},
                DELISTED_TICKER: list(reversed(TRADING_DAYS)),
            },
            security_path,
            fetch_observation_fn=seed,
        )
        before = security_path.read_bytes()

        fetch = FetchSpy({})  # any call would KeyError
        result = _run_build(tmp_path, fetch)
        assert fetch.calls == []
        assert security_path.read_bytes() == before
        assert set(result.daily["ticker"]) == set(COMPANY_TICKERS)

    def test_classification_failure_fails_build_and_preserves_dictionary(
        self, tmp_path
    ):
        security_path = tmp_path / "reference" / "orats_security_types.parquet"
        ensure_security_types(
            {t: [LATEST_DAY] for t in EQUITY_TICKERS},
            security_path,
            fetch_observation_fn=FetchSpy(_core_responses()),
        )
        before = security_path.read_bytes()

        responses = _core_responses()
        responses[(ETF_TICKER, LATEST_DAY.isoformat())] = RuntimeError(
            "Core API failure"
        )
        with pytest.raises(RuntimeError, match="Core API failure"):
            _run_build(tmp_path, FetchSpy(responses))
        assert security_path.read_bytes() == before

    def test_malformed_core_record_fails_liquidity_build(self, tmp_path):
        responses = _core_responses()
        responses[(ETF_TICKER, LATEST_DAY.isoformat())] = _obs(
            ETF_TICKER, LATEST_DAY, 11
        )
        with pytest.raises(SecurityTypesError, match="outside 0-9"):
            _run_build(tmp_path, FetchSpy(responses))


class TestSnapshotLocalClassificationArtifact:
    def test_written_matching_candidate_set_and_deterministic(self, tmp_path):
        fetch = FetchSpy(_core_responses())
        result = _run_build(tmp_path, fetch)
        cache = _write_all_artifacts(tmp_path, result)

        artifact = cache / blp.SECURITY_CLASSIFICATION_FILENAME
        assert artifact.is_file()
        written = pd.read_parquet(artifact)
        # Exactly the candidate tickers used by this build, incl. non-company.
        assert set(written["ticker"]) == set(ALL_TICKERS)
        assert set(
            written.loc[
                written["classification"] == "non_company_equity", "ticker"
            ]
        ) == {ETF_TICKER, INDEX_TICKER}

        # Deterministic: a second identical build produces the same
        # classification content digest (timestamps excluded by design).
        other_root = tmp_path / "again"
        result2 = _run_build(other_root, FetchSpy(_core_responses()))
        cache2 = _write_all_artifacts(other_root, result2)
        written2 = pd.read_parquet(cache2 / blp.SECURITY_CLASSIFICATION_FILENAME)
        assert classification_digest(written) == classification_digest(written2)


class TestIncrementalEquityFilter:
    def test_incremental_filters_prior_and_new_rows(self, tmp_path):
        """Prior artifacts containing a non-company ticker are cleansed and the
        new week is filtered before weekly/panel append, using date-specific
        classification at each ticker's latest observed date."""
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        wm = date(2024, 1, 5)
        new_days = [date(2024, 1, d) for d in range(8, 13)]
        latest_new_day = new_days[-1]
        expiry = date(2024, 1, 19)

        def daily_row(trade_date: date, ticker: str) -> dict:
            return {
                "trade_date": trade_date,
                "ticker": ticker,
                "daily_atm_straddle_dollar_vol": 1000.0,
                "daily_atm_spread_pct": 0.05,
                "daily_has_valid_quote": True,
                "n_candidate_expiries": 1,
                "n_expiries_total": 1,
                "no_expiry_in_band": False,
                "liquidity_source": blp.LIQUIDITY_SOURCE,
            }

        daily = pd.DataFrame(
            [daily_row(wm, "AL1"), daily_row(wm, ETF_TICKER)]
        )
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": wm,
                    "ticker": t,
                    "weekly_atm_straddle_dollar_vol": 1000.0,
                    "weekly_atm_spread_pct": 0.05,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                }
                for t in ("AL1", ETF_TICKER)
            ]
        )
        panel = blp.aggregate_rolling_weekly_panel(
            weekly, [wm], [wm], lookback_weeks=12, min_valid_quote_weeks=1
        )
        panel = blp.stamp_panel_universe_params(
            panel, dvol_top_pct=0.4, spread_bot_pct=1.0
        )
        daily.to_parquet(cache / blp.DAILY_FILENAME, index=False)
        weekly.to_parquet(cache / blp.WEEKLY_FILENAME, index=False)
        panel.to_parquet(cache / blp.PANEL_FILENAME, index=False)

        def load_day(trade_date: date) -> pd.DataFrame:
            if trade_date not in new_days:
                return pd.DataFrame()
            rows = [
                {
                    "ticker": t,
                    "expirDate": expiry,
                    "strike": 100.0,
                    "stkPx": 100.0,
                    "cBidPx": 1.0,
                    "cAskPx": 1.1,
                    "pBidPx": 1.0,
                    "pAskPx": 1.1,
                    "cVolu": 10.0,
                    "pVolu": 10.0,
                }
                for t in ("AL1", ETF_TICKER)
            ]
            return pd.DataFrame(rows)

        state = blp.validate_incremental_artifacts(
            cache,
            lookback_weeks=12,
            min_valid_quote_weeks=1,
            dte_min=5,
            dte_max=60,
            dvol_top_pct=0.4,
            spread_bot_pct=1.0,
        )
        # Merged daily makes the latest observed date the newest new-week day.
        responses = {
            ("AL1", latest_new_day.isoformat()): _obs("AL1", latest_new_day, 0),
            (ETF_TICKER, latest_new_day.isoformat()): _obs(
                ETF_TICKER, latest_new_day, 5
            ),
        }
        fetch = FetchSpy(responses)
        classify_fn = blp.make_core_classifier(
            tmp_path / "reference" / "orats_security_types.parquet",
            fetch_observation_fn=fetch,
        )
        result = blp.run_incremental(
            tmp_path / "raw",
            cache,
            state,
            load_day,
            [wm, *new_days],
            classify_fn=classify_fn,
        )

        assert sorted(fetch.calls) == [
            ("AL1", latest_new_day.isoformat()),
            (ETF_TICKER, latest_new_day.isoformat()),
        ]
        assert set(result.daily["ticker"]) == {"AL1"}
        assert set(result.weekly["ticker"]) == {"AL1"}
        assert set(result.panel["ticker"]) == {"AL1"}
        assert set(result.classification["ticker"]) == {"AL1", ETF_TICKER}
