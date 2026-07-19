"""Focused split-fetch correctness tests; all HTTP behavior is mocked."""

from __future__ import annotations

import logging

import pytest
import requests

import src.data.corporate_actions as corporate_actions
from src.data.corporate_actions import (
    CorporateActionsFetchError,
    OratsCorporateActionsFetcher,
)


def _track_processed(monkeypatch, fetcher):
    processed: set[str] = set()
    monkeypatch.setattr(fetcher, "_load_checkpoint", lambda _path: ([], processed))
    monkeypatch.setattr(fetcher, "_save_parquet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(corporate_actions, "sleep", lambda _seconds: None)
    return processed


def test_valid_empty_split_response_succeeds_and_marks_ticker_processed(
    monkeypatch, tmp_path
):
    fetcher = OratsCorporateActionsFetcher(token="SECRET", rate_limit=0)
    processed = _track_processed(monkeypatch, fetcher)
    monkeypatch.setattr(fetcher, "_get", lambda _url, _params: {"data": []})

    result = fetcher.fetch_all_splits(
        ["EMPTY"], checkpoint_path=tmp_path / "checkpoint.parquet"
    )

    assert result.empty
    assert list(result.columns) == ["ticker", "split_date", "divisor"]
    assert processed == {"EMPTY"}


def test_request_failure_raises_and_does_not_mark_ticker_processed(
    monkeypatch, tmp_path
):
    fetcher = OratsCorporateActionsFetcher(
        token="SECRET", rate_limit=0, max_retries=1
    )
    processed = _track_processed(monkeypatch, fetcher)
    monkeypatch.setattr(corporate_actions.time, "sleep", lambda _seconds: None)

    def fail_request(*_args, **_kwargs):
        raise requests.ConnectionError("mocked connection failure")

    monkeypatch.setattr(fetcher.session, "get", fail_request)

    with pytest.raises(CorporateActionsFetchError, match="failed after 1 attempts"):
        fetcher.fetch_all_splits(
            ["FAILED"], checkpoint_path=tmp_path / "checkpoint.parquet"
        )

    assert "FAILED" not in processed


@pytest.mark.parametrize("payload", [{}, {"data": {}}])
def test_missing_or_malformed_split_data_raises_and_does_not_mark_processed(
    monkeypatch, tmp_path, payload
):
    fetcher = OratsCorporateActionsFetcher(token="SECRET", rate_limit=0)
    processed = _track_processed(monkeypatch, fetcher)
    monkeypatch.setattr(fetcher, "_get", lambda _url, _params: payload)

    with pytest.raises(CorporateActionsFetchError, match="expected list field 'data'"):
        fetcher.fetch_all_splits(
            ["MALFORMED"], checkpoint_path=tmp_path / "checkpoint.parquet"
        )

    assert "MALFORMED" not in processed


def test_token_never_appears_in_request_failure_logs(monkeypatch, caplog):
    token = "TOP_SECRET_ORATS_TOKEN"
    fetcher = OratsCorporateActionsFetcher(token=token, max_retries=1)

    class Response:
        status_code = 401
        headers = {}

    monkeypatch.setattr(fetcher.session, "get", lambda *_args, **_kwargs: Response())

    with caplog.at_level(logging.WARNING), pytest.raises(CorporateActionsFetchError):
        fetcher.fetch_splits_for_ticker("AAA")

    assert token not in caplog.text
    assert "HTTP 401" in caplog.text
    assert "ticker=AAA" in caplog.text
    assert "/splits" in caplog.text


def test_keyboard_interrupt_saves_checkpoint_then_propagates(monkeypatch, tmp_path):
    fetcher = OratsCorporateActionsFetcher(token="SECRET", rate_limit=0)
    processed = _track_processed(monkeypatch, fetcher)
    saved = []
    monkeypatch.setattr(
        fetcher,
        "_save_parquet",
        lambda df, path, dedup_cols: saved.append((df.copy(), path, dedup_cols)),
    )

    def interrupt(_ticker):
        raise KeyboardInterrupt

    monkeypatch.setattr(fetcher, "fetch_splits_for_ticker", interrupt)
    checkpoint = tmp_path / "checkpoint.parquet"

    with pytest.raises(KeyboardInterrupt):
        fetcher.fetch_all_splits(["AAA"], checkpoint_path=checkpoint)

    assert processed == set()
    assert len(saved) == 1
    assert saved[0][1] == checkpoint
