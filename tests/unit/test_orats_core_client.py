"""Unit tests for src/data/orats_core_client.py (date-specific Core client).

The HTTP boundary is always mocked — no live ORATS request is ever made.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import pytest
import requests

from src.data import orats_core_client as occ
from src.data.orats_core_client import OratsCoreClient, OratsCoreError

TOKEN = "SECRET_TOKEN_abc123"
TRADE_DATE = date(2024, 1, 2)


# ── fakes ─────────────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Scripted session: pops one response (or raises one exception) per get."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(occ.time, "sleep", lambda seconds: None)


@pytest.fixture
def _token_env(monkeypatch):
    monkeypatch.setenv("ORATS_API_TOKEN", TOKEN)


def _client(script) -> tuple[OratsCoreClient, FakeSession]:
    session = FakeSession(script)
    return OratsCoreClient(rate_limit=0.0, max_retries=3, session=session), session


def _payload(rows):
    return {"data": rows}


def _row(asset_type=0, ticker="AAPL", trade_date="2024-01-02"):
    return {"ticker": ticker, "tradeDate": trade_date, "assetType": asset_type}


# ── date-specific request contract ────────────────────────────────────────────


def test_request_always_includes_trade_date(_token_env):
    client, session = _client([FakeResponse(200, _payload([_row()]))])
    df = client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert len(df) == 1
    params = session.calls[0]["params"]
    assert params["ticker"] == "AAPL"
    assert params["tradeDate"] == "2024-01-02"
    assert params["fields"] == occ.CORE_FIELDS


def test_every_retry_attempt_includes_trade_date(_token_env):
    client, session = _client(
        [
            FakeResponse(429, headers={"Retry-After": "0"}),
            requests.ConnectionError("transient"),
            FakeResponse(200, _payload([_row(asset_type=1)])),
        ]
    )
    df = client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert len(df) == 1
    assert len(session.calls) == 3
    for call in session.calls:
        assert call["params"]["tradeDate"] == "2024-01-02"


def test_no_unrestricted_full_history_path(_token_env):
    """The former full-history method must not exist on the client."""
    client, _ = _client([])
    assert not hasattr(client, "fetch_asset_type_history")


def test_datetime_trade_date_is_normalized_to_date(_token_env):
    client, session = _client([FakeResponse(200, _payload([]))])
    client.fetch_asset_type_at_date("AAPL", datetime(2024, 1, 2, 15, 30))
    assert session.calls[0]["params"]["tradeDate"] == "2024-01-02"


def test_non_date_trade_date_fails_before_any_request(_token_env):
    client, session = _client([])
    with pytest.raises(OratsCoreError, match="datetime.date"):
        client.fetch_asset_type_at_date("AAPL", "2024-01-02")  # type: ignore[arg-type]
    assert session.calls == []


# ── success and valid-empty ───────────────────────────────────────────────────


def test_success_returns_observation_frame(_token_env):
    client, _ = _client([FakeResponse(200, _payload([_row(asset_type=3)]))])
    df = client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert list(df.columns) == ["ticker", "tradeDate", "assetType"]
    assert df.iloc[0]["assetType"] == 3


def test_empty_data_is_successful_empty_not_error(_token_env):
    client, _ = _client([FakeResponse(200, _payload([]))])
    df = client.fetch_asset_type_at_date("NOPE", TRADE_DATE)
    assert df.empty
    assert list(df.columns) == ["ticker", "tradeDate", "assetType"]


# ── failures are never silent-empty ───────────────────────────────────────────


def test_missing_data_key_is_failure(_token_env):
    client, _ = _client([FakeResponse(200, {"rows": []})])
    with pytest.raises(OratsCoreError, match="missing the 'data' key"):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)


def test_unparseable_body_is_failure(_token_env):
    client, _ = _client([FakeResponse(200, ValueError("not json"))])
    with pytest.raises(OratsCoreError, match="unparseable body"):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)


def test_non_retryable_http_error_fails_immediately(_token_env):
    client, session = _client([FakeResponse(404, text="not found")])
    with pytest.raises(OratsCoreError, match="HTTP 404"):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert len(session.calls) == 1


def test_exhausted_retries_fail(_token_env):
    client, session = _client([FakeResponse(500)] * 3)
    with pytest.raises(OratsCoreError, match="failed after 3 attempts"):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert len(session.calls) == 3


# ── interrupt propagation ─────────────────────────────────────────────────────


def test_keyboard_interrupt_propagates(_token_env):
    client, _ = _client([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)


# ── credentials ───────────────────────────────────────────────────────────────


def test_token_read_only_from_env(monkeypatch):
    monkeypatch.delenv("ORATS_API_TOKEN", raising=False)
    client, session = _client([FakeResponse(200, _payload([]))])
    with pytest.raises(OratsCoreError, match="ORATS_API_TOKEN"):
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert session.calls == []


def test_token_never_appears_in_logs_or_errors(_token_env, caplog):
    """Token must not leak via warnings, errors, or exception text on any path."""
    caplog.set_level(logging.DEBUG)

    client, _ = _client(
        [
            FakeResponse(500, text=f"boom {TOKEN}"),
            requests.ConnectionError(f"cannot reach https://x?token={TOKEN}"),
            FakeResponse(403, text=f"denied for token {TOKEN}"),
        ]
    )
    with pytest.raises(OratsCoreError) as excinfo:
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)

    assert TOKEN not in caplog.text
    assert TOKEN not in str(excinfo.value)


def test_transport_failure_error_text_is_redacted(_token_env, caplog):
    caplog.set_level(logging.DEBUG)
    client, _ = _client([requests.ConnectionError(f"token={TOKEN}")] * 3)
    with pytest.raises(OratsCoreError) as excinfo:
        client.fetch_asset_type_at_date("AAPL", TRADE_DATE)
    assert TOKEN not in caplog.text
    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)
