"""
Shared pytest fixtures for unit tests.

This module provides reusable test fixtures including:
- Sample option chains loaded from CSV files
- Real VZ option data for testing
- Helper functions for creating test data
"""

import sys
from pathlib import Path
from datetime import date
from decimal import Decimal
from typing import List

import pytest
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import OptionQuote

# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Helper Functions
# =============================================================================

def load_option_chain_from_csv(csv_filename: str) -> List[OptionQuote]:
    """
    Load option chain from CSV fixture file.
    
    Args:
        csv_filename: Name of CSV file in tests/fixtures/
        
    Returns:
        List of OptionQuote objects
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    csv_path = FIXTURES_DIR / csv_filename
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    chain = []
    for _, row in df.iterrows():
        chain.append(OptionQuote(
            ticker=row['ticker'],
            trade_date=date.fromisoformat(row['trade_date']),
            expiry_date=date.fromisoformat(row['expiry_date']),
            strike=Decimal(str(row['strike'])),
            option_type=row['option_type'],
            bid=Decimal(str(row['bid'])),
            ask=Decimal(str(row['ask'])),
            mid=Decimal(str(row['mid'])),
            iv=float(row['iv']),
            delta=float(row['delta']),
            gamma=float(row['gamma']),
            vega=float(row['vega']),
            theta=float(row['theta']),
            volume=int(row['volume']),
            open_interest=int(row['open_interest'])
        ))
    
    return chain


# =============================================================================
# Option Chain Fixtures
# =============================================================================

@pytest.fixture
def sample_option_chain_atm() -> List[OptionQuote]:
    """
    Well-formed option chain with strikes 43.5/44.0/44.5/45.0.
    
    - All strikes have both call and put
    - Single expiry: 2024-12-06
    - All premiums positive
    - Trade date: 2024-11-29
    - Spot price: ~44.50 (ATM strike = 44.5)
    
    Use for happy path testing.
    """
    return load_option_chain_from_csv("sample_option_chain_atm.csv")


@pytest.fixture
def sample_option_chain_unsorted() -> List[OptionQuote]:
    """
    Same as sample_option_chain_atm but strikes in random order.
    
    Tests that builder handles unsorted input correctly.
    """
    return load_option_chain_from_csv("sample_option_chain_unsorted.csv")


@pytest.fixture
def sample_option_chain_multiple_expiries() -> List[OptionQuote]:
    """
    Option chain with mixed expiry dates: 2024-12-06 AND 2024-12-13.
    
    Use for testing error handling (multiple expiries not allowed).
    """
    return load_option_chain_from_csv("sample_option_chain_multiple_expiries.csv")


@pytest.fixture
def sample_option_chain_missing_call() -> List[OptionQuote]:
    """
    Option chain MISSING call at strike 44.5.
    
    Has puts at all strikes, but calls are incomplete.
    Use for testing error handling.
    """
    return load_option_chain_from_csv("sample_option_chain_missing_call.csv")


@pytest.fixture
def sample_option_chain_missing_put() -> List[OptionQuote]:
    """
    Option chain MISSING put at strike 44.5.
    
    Has calls at all strikes, but puts are incomplete.
    Use for testing error handling.
    """
    return load_option_chain_from_csv("sample_option_chain_missing_put.csv")


@pytest.fixture
def sample_option_chain_invalid_call_premium() -> List[OptionQuote]:
    """
    Option chain with call at strike 44.5 having mid = 0.00.
    
    Use for testing premium validation.
    """
    return load_option_chain_from_csv("sample_option_chain_invalid_call_premium.csv")


@pytest.fixture
def sample_option_chain_invalid_put_premium() -> List[OptionQuote]:
    """
    Option chain with put at strike 44.5 having mid = 0.00.
    
    Use for testing premium validation.
    """
    return load_option_chain_from_csv("sample_option_chain_invalid_put_premium.csv")


# =============================================================================
# IronButterflyBuilder Fixtures (stubs — implement before running IBF tests)
# =============================================================================

@pytest.fixture
def sample_ibf_chain_atm() -> List[OptionQuote]:
    """
    Minimal clean chain suitable for a valid iron butterfly construction.

    Source: AAPL, trade_date=2026-02-13, expiry_date=2026-02-20 (7 DTE)
    Use with ibf_ticker / ibf_trade_date / ibf_expiry_date fixtures.

    Strikes (3 × 2 options = 6 rows):
      245.0  call + put  (long put wing  — OTM put, |delta|≈0.169)
      255.0  call + put  (body — ATM, call delta≈0.546, put delta≈-0.454)
      265.0  call + put  (long call wing — OTM call, delta≈0.171)

    Economics (wing width = 10.0):
      net_credit = (4.65 + 3.65) − (0.87 + 1.19) = 6.24
      yield_on_capital = 6.24 / 10.0 = 62.4%  ✓ > 5% threshold

    Spread quality: all legs < 3% spread_pct  ✓ < 25% threshold
    """
    return load_option_chain_from_csv("sample_ibf_chain_atm.csv")


@pytest.fixture
def sample_ibf_chain_multi_width() -> List[OptionQuote]:
    """
    Chain with TWO valid symmetric wing pairs around the same body at 255.0,
    each pair at a different width and therefore different long-wing |delta|.

    Source: AAPL, trade_date=2026-02-13, expiry_date=2026-02-20 (7 DTE)
    Use with ibf_ticker / ibf_trade_date / ibf_expiry_date fixtures.

    Body (short legs):
      255.0  call + put  (delta ≈ +0.546 / −0.454)

    Pair A — width 2.5  (higher delta wings):
      257.5  call  delta ≈ +0.450
      252.5  put   delta ≈ −0.365
      avg |delta| ≈ 0.408

    Pair B — width 5.0  (lower delta wings):
      260.0  call  delta ≈ +0.350
      250.0  put   delta ≈ −0.286
      avg |delta| ≈ 0.318

    Design intent:
      wing_delta=0.408  → builder selects Pair A (width=2.5)
      wing_delta=0.318  → builder selects Pair B (width=5.0)
      (any wing_delta between 0.363 and 0.408 selects Pair A;
       any wing_delta below 0.363 selects Pair B)

    Both pairs: net_credit > 0, all spreads < 25%.
    """
    return load_option_chain_from_csv("sample_ibf_chain_multi_width.csv")


@pytest.fixture
def sample_ibf_chain_no_mirror() -> List[OptionQuote]:
    """
    Chain where OTM call wings exist but NO mirrored put strikes exist,
    so no symmetric wing pair can be formed.

    Required content
    ----------------
      - 44.5 call  (body)
      - 44.5 put   (body)
      - 45.0 call  (OTM call candidate only — NO 44.0 put present)

    Purpose: verifies that _select_wing_pair() raises ValueError when
    symmetry cannot be satisfied even though OTM calls are available.

    Store as:  tests/fixtures/sample_ibf_chain_no_mirror.csv
    Format: same columns as sample_option_chain_atm.csv
    """
    pass


# =============================================================================
# IronButterflyBuilder — AAPL-specific date/ticker fixtures
# (IBF fixtures use AAPL 2026-02-13/2026-02-20, not the VZ fixtures above)
# =============================================================================

@pytest.fixture
def ibf_ticker() -> str:
    """Ticker for IBF fixtures: AAPL"""
    return "AAPL"


@pytest.fixture
def ibf_trade_date() -> date:
    """Trade date for IBF fixtures: 2026-02-13"""
    return date(2026, 2, 13)


@pytest.fixture
def ibf_expiry_date() -> date:
    """Expiry date for IBF fixtures: 2026-02-20 (7 DTE)"""
    return date(2026, 2, 20)


@pytest.fixture
def ibf_spot_price() -> Decimal:
    """Spot price for IBF fixtures: 255.81 (AAPL spot; ATM body strike = 255.0)"""
    return Decimal("255.81")


# =============================================================================
# Shared Constants
# =============================================================================

@pytest.fixture
def trade_date() -> date:
    """Standard trade date for VZ fixtures: 2024-11-29"""
    return date(2024, 11, 29)


@pytest.fixture
def expiry_date() -> date:
    """Standard expiry date for VZ fixtures: 2024-12-06"""
    return date(2024, 12, 6)


@pytest.fixture
def ticker() -> str:
    """Standard ticker for fixtures: VZ"""
    return "VZ"


# =============================================================================
# Real VZ Option Data (from test_models.py)
# =============================================================================

@pytest.fixture
def vz_put():
    """Real VZ put option from 2024-11-29."""
    return OptionQuote(
        ticker='VZ',
        trade_date=date(2024, 11, 29),
        expiry_date=date(2024, 12, 6),
        strike=Decimal('44.50'),
        option_type='put',
        bid=Decimal('0.43'),
        ask=Decimal('0.45'),
        mid=Decimal('0.44'),
        iv=0.160773,
        delta=-0.513032,
        gamma=0.400476,
        vega=0.024816,
        theta=-0.029108,
        volume=491,
        open_interest=785
    )


@pytest.fixture
def vz_call():
    """Real VZ call option from 2024-11-29."""
    return OptionQuote(
        ticker='VZ',
        trade_date=date(2024, 11, 29),
        expiry_date=date(2024, 12, 6),
        strike=Decimal('44.50'),
        option_type='call',
        bid=Decimal('0.39'),
        ask=Decimal('0.41'),
        mid=Decimal('0.40'),
        iv=0.160773,
        delta=0.486968,
        gamma=0.400476,
        vega=0.024816,
        theta=-0.029108,
        volume=1029,
        open_interest=1248
    )
