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
