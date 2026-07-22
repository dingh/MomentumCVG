"""Minimal ORATS Core historical client (security-type classification input).

Fetches Core ``assetType`` records for one ``(ticker, tradeDate)`` pair at a
time from the ORATS datav2 historical Core endpoint. Consumed by
``src/data/security_types.py`` to classify tickers as company equities vs
non-company securities (ETF / index / VIX-style).

Every request is date-specific: ``tradeDate`` is always sent. There is no
production path that requests unrestricted full history for a ticker.

Credentials
-----------
The API token is read exclusively from the ``ORATS_API_TOKEN`` environment
variable at request time. There is no CLI token argument. The token, request
parameters, authorization material, and complete credential-bearing URLs are
never logged; exception/body text is redacted before logging as defence in
depth.

Outcome contract
----------------
* HTTP 200 with an empty ``data`` list is a *successful empty* response and
  returns an empty DataFrame (columns ``ticker``, ``tradeDate``,
  ``assetType``).
* HTTP 404 (ORATS ``Not Found`` for that ticker/date) is treated the same as
  a successful empty observation: return an empty DataFrame so callers can
  fall back to earlier observed dates. It is not a transport/auth failure.
* Other HTTP errors, malformed / unparseable bodies, and exhausted retries
  raise :class:`OratsCoreError`. Those failures are never returned as empty.
* ``KeyboardInterrupt`` always propagates.

Retry behaviour is narrow and mirrors ``OratsCorporateActionsFetcher``:
429/5xx and transport errors are retried with capped exponential backoff up
to ``max_retries``; every other status fails immediately.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CORE_HIST_URL = "https://api.orats.io/datav2/hist/cores"
CORE_FIELDS = "ticker,tradeDate,assetType"
ASSET_TYPE_HISTORY_COLUMNS = ("ticker", "tradeDate", "assetType")

_RETRYABLE_STATUSES = (429, 500, 502, 503, 504)
_TOKEN_ENV_VAR = "ORATS_API_TOKEN"


class OratsCoreError(RuntimeError):
    """HTTP / API / parse failure fetching ORATS Core history."""


def _redact(text: Any, token: str) -> str:
    """Return ``text`` as a string with any token occurrence replaced."""
    return str(text).replace(token, "***") if token else str(text)


class OratsCoreClient:
    """Date-specific historical Core ``assetType`` fetcher.

    Parameters
    ----------
    rate_limit:
        Seconds to sleep before each request after the first (default 0.7 s).
    max_retries:
        Maximum attempts per request on 429 / 5xx / transport errors.
    timeout:
        Per-request HTTP timeout in seconds.
    session:
        Injectable ``requests.Session``-like object (tests mock the HTTP
        boundary here; no live request is made in tests).
    """

    def __init__(
        self,
        *,
        rate_limit: float = 0.7,
        max_retries: int = 5,
        timeout: int = 30,
        session: Any | None = None,
    ) -> None:
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self._request_made = False

    @staticmethod
    def _resolve_token() -> str:
        """Read the API token from ``ORATS_API_TOKEN`` only (never logged)."""
        token = os.environ.get(_TOKEN_ENV_VAR)
        if not token:
            raise OratsCoreError(
                f"{_TOKEN_ENV_VAR} environment variable is not set; it is the "
                "only supported credential source for the ORATS Core client."
            )
        return token

    def _parse_success_body(self, response: Any, ticker: str, token: str) -> pd.DataFrame:
        """Parse a 200 body; distinguish valid-empty from parse failure."""
        try:
            payload = response.json()
        except Exception as exc:
            raise OratsCoreError(
                f"ORATS Core returned an unparseable body for ticker {ticker}: "
                f"{_redact(exc, token)}"
            ) from exc

        if not isinstance(payload, dict) or "data" not in payload:
            raise OratsCoreError(
                f"ORATS Core response for ticker {ticker} is missing the "
                "'data' key; cannot distinguish empty history from failure."
            )
        data = payload["data"]
        if not isinstance(data, list):
            raise OratsCoreError(
                f"ORATS Core 'data' for ticker {ticker} is not a list."
            )

        rows: list[dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                raise OratsCoreError(
                    f"ORATS Core 'data' entry for ticker {ticker} is not an object."
                )
            rows.append(
                {
                    "ticker": entry.get("ticker"),
                    "tradeDate": entry.get("tradeDate"),
                    "assetType": entry.get("assetType"),
                }
            )
        if not rows:
            return pd.DataFrame(columns=list(ASSET_TYPE_HISTORY_COLUMNS))
        return pd.DataFrame(rows, columns=list(ASSET_TYPE_HISTORY_COLUMNS))

    def fetch_asset_type_at_date(self, ticker: str, trade_date: date) -> pd.DataFrame:
        """Fetch Core ``assetType`` rows for one ticker on one trade date.

        ``tradeDate`` is always included in the request; unrestricted
        full-history requests are not supported. Returns a DataFrame with
        columns ``ticker``, ``tradeDate``, ``assetType``. An empty DataFrame
        is a *successful* empty response (no Core record on that date);
        failures raise :class:`OratsCoreError`.
        """
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        if not isinstance(trade_date, date):
            raise OratsCoreError(
                f"trade_date must be a datetime.date; got {trade_date!r}"
            )
        token = self._resolve_token()
        params = {
            "token": token,
            "ticker": ticker,
            "tradeDate": trade_date.isoformat(),
            "fields": CORE_FIELDS,
        }

        if self._request_made and self.rate_limit > 0:
            time.sleep(self.rate_limit)
        self._request_made = True

        request_label = f"ticker {ticker} tradeDate {trade_date.isoformat()}"
        last_error = "no attempt made"
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    CORE_HIST_URL, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = f"transport error: {_redact(exc, token)}"
                backoff = min(60.0, (2**attempt) * 1.0)
                logger.warning(
                    "ORATS Core request error for %s — backing off %.1fs "
                    "(attempt %d/%d): %s",
                    request_label, backoff, attempt + 1, self.max_retries,
                    _redact(exc, token),
                )
                time.sleep(backoff)
                continue

            if response.status_code == 200:
                return self._parse_success_body(response, ticker, token)

            # Ticker/date absent in Core: same outcome as HTTP 200 + empty data
            # so classification can fall back to earlier observed dates.
            if response.status_code == 404:
                logger.info(
                    "ORATS Core HTTP 404 for %s — treating as empty observation",
                    request_label,
                )
                return pd.DataFrame(columns=list(ASSET_TYPE_HISTORY_COLUMNS))

            if response.status_code in _RETRYABLE_STATUSES:
                backoff = min(60.0, (2**attempt) * 1.0)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    "ORATS Core HTTP %s for %s — backing off %.1fs "
                    "(attempt %d/%d)",
                    response.status_code, request_label, backoff,
                    attempt + 1, self.max_retries,
                )
                time.sleep(backoff)
                continue

            # Non-retryable HTTP error: fail immediately. Never log the URL,
            # params, or raw body (redacted excerpt only in the exception).
            raise OratsCoreError(
                f"ORATS Core HTTP {response.status_code} for {request_label}: "
                f"{_redact(getattr(response, 'text', '')[:200], token)}"
            )

        raise OratsCoreError(
            f"ORATS Core request failed after {self.max_retries} attempts "
            f"for {request_label}: {last_error}"
        )
