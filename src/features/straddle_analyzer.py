"""
Straddle analysis and historical data generation.

This module provides tools for analyzing straddle performance and
building historical databases for momentum strategies.
"""

import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.orats_provider import ORATSDataProvider
from ..strategy.builders import StraddleBuilder
from ..core.models import Signal, OptionQuote, StrategyType


logger = logging.getLogger(__name__)


class StraddleHistoryBuilder:
    """Build historical straddle database for momentum strategy."""
    
    def __init__(
        self,
        data_root: str,
        dte_target: int = 7,
        max_spread_pct: float = 0.50,
        min_volume: int = 10,
        min_oi: int = 0
    ):
        """
        Initialize builder.
        
        Args:
            data_root: Path to ORATS data directory
            dte_target: Target days to expiry (7 for weekly)
            max_spread_pct: Maximum bid-ask spread as % of mid (50%)
            min_volume: Minimum volume required
            min_oi: Minimum open interest required
        """
        self.data_root = data_root
        self.dte_target = dte_target
        self.max_spread_pct = max_spread_pct
        self.min_volume = min_volume
        self.min_oi = min_oi
        
        # Initialize components (will be created per-worker for parallel processing)
        self.provider = None
        self.builder = None
    
    def _init_worker_components(self):
        """Initialize provider and builder (called in each worker process)."""
        if self.provider is None:
            self.provider = ORATSDataProvider(data_root=self.data_root,
                                              min_volume=self.min_volume,
                                              min_open_interest=self.min_oi,
                                              max_spread_pct=self.max_spread_pct)
            self.builder = StraddleBuilder()
            logger.info(f"Worker initialized with data_root={self.data_root}")
    
    def check_tradeability(
        self, 
        call: OptionQuote, 
        put: OptionQuote
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Check if straddle is tradeable based on liquidity metrics.
        
        Args:
            call: Call option quote
            put: Put option quote
        
        Returns:
            Tuple of (is_tradeable, failure_reason, metrics_dict)
        """
        metrics = {
            'call_spread_pct': 0.0,
            'put_spread_pct': 0.0,
            'avg_spread_pct': 0.0,
            'call_volume': int(call.volume or 0),
            'put_volume': int(put.volume or 0),
            'call_open_interest': int(call.open_interest or 0),
            'put_open_interest': int(put.open_interest or 0),
        }
        
        # Check premiums
        if call.mid <= 0 or put.mid <= 0:
            return False, 'zero_premium', metrics
        
        # Calculate spreads
        call_spread_pct =  float((call.ask - call.bid) / call.mid if call.mid > 0 else 999.0)
        put_spread_pct = float((put.ask - put.bid) / put.mid if put.mid > 0 else 999.0)
        
        metrics['call_spread_pct'] = call_spread_pct
        metrics['put_spread_pct'] = put_spread_pct
        metrics['avg_spread_pct'] = (call_spread_pct + put_spread_pct) / 2
        
        # Check call spread
        if call_spread_pct > self.max_spread_pct:
            return False, f'wide_call_spread_{call_spread_pct:.1%}', metrics
        
        # Check put spread
        if put_spread_pct > self.max_spread_pct:
            return False, f'wide_put_spread_{put_spread_pct:.1%}', metrics
        
        call_vol = int(call.volume or 0)
        put_vol  = int(put.volume or 0)

        if call_vol < self.min_volume or put_vol < self.min_volume:
            return False, f'low_volume_c{call_vol}_p{put_vol}', metrics
        
        # Check volume
        #if call.volume < self.min_volume or put.volume < self.min_volume:
        call_oi = int(call.open_interest or 0)
        put_oi  = int(put.open_interest or 0)

        if call_oi < self.min_oi or put_oi < self.min_oi:
            return False, f'low_oi_c{call_oi}_p{put_oi}', metrics        
        
        return True, 'ok', metrics
    
    def calculate_realized_volatility(
        self, 
        entry_spot: float, 
        exit_spot: float, 
        days_held: int
    ) -> Optional[float]:
        """
        Calculate annualized realized volatility.
        
        Formula: RV = |log(S_exit / S_entry)| * sqrt(252 / days_held)
        
        Args:
            entry_spot: Spot price at entry
            exit_spot: Spot price at exit
            days_held: Number of days held
        
        Returns:
            Annualized realized volatility, or None if invalid inputs
        """
        if entry_spot <= 0 or exit_spot <= 0 or days_held <= 0:
            return None
        
        log_return = abs(np.log(exit_spot / entry_spot))
        annualization_factor = np.sqrt(252 / days_held)
        realized_vol = log_return * annualization_factor
        
        return float(realized_vol)
    
    def _find_best_expiry(
        self,
        ticker: str,
        trade_date: date,
        target_dte: int,
        tolerance_days: int = 4
    ) -> Optional[date]:
        """
        Find expiry date that aligns with rebalancing schedule.
        
        For monthly: targets first Friday of next month that's actually IN next month
        For weekly: targets next Friday (~7 DTE)
        
        Args:
            ticker: Stock ticker
            trade_date: Trade date
            target_dte: Target days to expiry (7 for weekly, 30 for monthly)
            tolerance_days: Maximum deviation from target (±7 days for monthly, ±4 for weekly)
        
        Returns:
            Best expiry date, or None if none found within tolerance
        """
        from datetime import timedelta
        
        try:
            # Get available expiries from provider
            expiries = self.provider.get_available_expiries(ticker, trade_date)
            
            if not expiries:
                return None
            
            # For monthly rebalancing, target first Friday of next month
            if target_dte >= 28:  # Monthly (28-30 days)
                # Find first Friday of next month
                next_month = trade_date.month + 1 if trade_date.month < 12 else 1
                next_year = trade_date.year if trade_date.month < 12 else trade_date.year + 1
                
                # Find all Fridays/Thursdays in the next month that have expiries
                target_month_expiries = [
                    exp for exp in expiries
                    if exp.year == next_year 
                    and exp.month == next_month 
                    and exp.weekday() in [3, 4]  # Thursday or Friday
                    and exp >= trade_date
                ]
                
                if not target_month_expiries:
                    # No Friday/Thursday expiries in target month
                    logger.warning(f"{ticker} on {trade_date}: No Fri/Thu expiry in {next_year}-{next_month:02d}")
                    
                    # Fallback: find any expiry close to ~30 days
                    expiry_diffs = [
                        (exp, abs((exp - trade_date).days - target_dte))
                        for exp in expiries
                        if exp >= trade_date
                    ]
                    
                    if expiry_diffs:
                        best_expiry = min(expiry_diffs, key=lambda x: x[1])
                        if best_expiry[1] <= tolerance_days:
                            return best_expiry[0]
                    
                    return None
                
                # Sort by date and take the first one (earliest Friday/Thursday in the month)
                target_month_expiries.sort()
                best_expiry = target_month_expiries[0]
                
                # Verify it's within reasonable DTE range (20-45 days)
                dte = (best_expiry - trade_date).days
                if dte < 20 or dte > 45:
                    logger.warning(f"{ticker} on {trade_date}: First Fri of {next_year}-{next_month:02d} is {dte} DTE (unusual)")
                
                return best_expiry
            
            else:  # Weekly (7 days)
                # Original logic: find expiry closest to target DTE
                expiry_dtes = [(exp, (exp - trade_date).days) for exp in expiries]
                
                # Filter to positive DTEs within tolerance (±4 days for weekly)
                valid_expiries = [
                    (exp, dte) for exp, dte in expiry_dtes
                    if dte > 0 and abs(dte - target_dte) <= 4
                ]
                
                if not valid_expiries:
                    return None
                
                # Filter to only expiries that fall on Fridays (weekday == 4)
                friday_expiries = [(exp, dte) for exp, dte in valid_expiries if exp.weekday() == 4]
                
                if friday_expiries:
                    # Prefer Fridays - return closest to target DTE
                    best_expiry = min(friday_expiries, key=lambda x: abs(x[1] - target_dte))
                    return best_expiry[0]
                
                # If no Fridays, try Thursday
                thursday_expiries = [(exp, dte) for exp, dte in valid_expiries if exp.weekday() == 3]
                if thursday_expiries:
                    best_expiry = min(thursday_expiries, key=lambda x: abs(x[1] - target_dte))
                    return best_expiry[0]
                
                # Fallback to any day
                logger.warning(f"{ticker} on {trade_date}: No Fri/Thu expiry in range, using any day")
                best_expiry = min(valid_expiries, key=lambda x: abs(x[1] - target_dte))
                return best_expiry[0]
            
        except Exception as e:
            logger.error(f"Error finding expiry for {ticker} on {trade_date}: {str(e)}")
            return None
    
    def process_single_straddle(
        self,
        ticker: str,
        entry_date: date,
    ) -> Dict:
        """
        Process single straddle: build at entry, calculate P&L at expiry.
        
        Args:
            ticker: Stock ticker
            entry_date: Trade entry date
        
        Returns:
            Dictionary with all straddle metrics
        """
        start_time = datetime.now()
        
        # Initialize components if needed
        self._init_worker_components()
        
        result = {
            'ticker': ticker,
            'entry_date': entry_date,
            'dte_category': 'weekly' if self.dte_target <= 10 else 'monthly',
            'dte_target': self.dte_target,
            'dte_actual': None,
            'expiry_date': None,
            'entry_spot': None,
            'strike': None,
            'entry_cost': None,
            'entry_iv': None,
            'entry_delta': None,
            'entry_gamma': None,
            'entry_vega': None,
            'entry_theta': None,
            'exit_spot': None,
            'exit_value': None,
            'exit_type': 'expired',
            'pnl': None,
            'return_pct': None,
            'annualized_return': None,
            'spot_move_pct': None,
            'realized_volatility': None,
            'iv_rv_spread': None,
            'expected_move': None,
            'move_vs_expected': None,
            'is_tradeable': False,
            'failure_reason': None,
            'call_spread_pct': None,
            'put_spread_pct': None,
            'avg_spread_pct': None,
            'call_volume': None,
            'put_volume': None,
            'call_open_interest': None,
            'put_open_interest': None,
            'days_held': None,
            'processing_time': None,
        }
        
        try:
            # Get spot price at entry - returns None if not found
            entry_spot = self.provider.get_spot_price(ticker, entry_date)
            if entry_spot is None:
                result['failure_reason'] = 'no_spot_price'
                result['processing_time'] = (datetime.now() - start_time).total_seconds()
                return result
            
            result['entry_spot'] = float(entry_spot)
            
            # Find expiry closest to target DTE
            expiry_date = self._find_best_expiry(ticker, entry_date, self.dte_target)
            if expiry_date is None:
                result['failure_reason'] = 'no_expiry_found'
                return result
            
            # Get option chain for that expiry
            entry_chain = self.provider.get_option_chain(
                ticker=ticker,
                trade_date=entry_date,
                expiry_date=expiry_date
            )
            
            if not entry_chain:
                result['failure_reason'] = 'no_options_at_entry'
                return result
                        
            try:
                strategy = self.builder.build_strategy(
                    ticker=ticker,
                    trade_date=entry_date,
                    expiry_date=expiry_date,
                    option_chain=entry_chain,
                    spot_price=entry_spot
                )
            except ValueError as e:
                result['failure_reason'] = f'build_failed_{str(e)[:50]}'
                return result
            
            # Extract straddle info
            call_leg = next((leg for leg in strategy.legs if leg.option.option_type == 'call'), None)
            put_leg = next((leg for leg in strategy.legs if leg.option.option_type == 'put'), None)
            
            if not call_leg or not put_leg:
                result['failure_reason'] = 'missing_legs'
                return result
            
            # Record entry metrics
            result['strike'] = float(call_leg.option.strike)
            result['entry_cost'] = float(strategy.net_premium)
            result['entry_iv'] = float((call_leg.option.iv + put_leg.option.iv) / 2)
            result['entry_delta'] = float(strategy.net_delta)
            result['entry_gamma'] = float(strategy.net_gamma)
            result['entry_vega'] = float(strategy.net_vega)
            result['entry_theta'] = float(strategy.net_theta)
            result['expiry_date'] = call_leg.option.expiry_date
            result['dte_actual'] = (call_leg.option.expiry_date - entry_date).days
            
            # Check tradeability
            is_tradeable, failure_reason, liquidity_metrics = self.check_tradeability(
                call_leg.option, put_leg.option
            )
            result['is_tradeable'] = is_tradeable
            result['failure_reason'] = failure_reason
            result.update(liquidity_metrics)
            
            # Get spot price at expiry
            exit_spot = self.provider.get_spot_price(ticker, call_leg.option.expiry_date)
            if exit_spot is None:
                result['failure_reason'] = 'no_spot_price_at_expiry'
                return result
            
            result['exit_spot'] = float(exit_spot)
            
            # Calculate P&L
            # For long straddle: P&L = (intrinsic_value_at_expiry - entry_cost)
            call_intrinsic = call_leg.calculate_intrinsic_value(exit_spot)
            put_intrinsic = put_leg.calculate_intrinsic_value(exit_spot)
            exit_value = call_intrinsic + put_intrinsic
            
            result['exit_value'] = float(exit_value)
            result['pnl'] = float(exit_value) - result['entry_cost']
            result['return_pct'] = (result['pnl'] / result['entry_cost']) * 100 if result['entry_cost'] > 0 else None
            
            # Calculate annualized return
            days_held = result['dte_actual']
            result['days_held'] = days_held
            if days_held > 0 and result['return_pct'] is not None:
                result['annualized_return'] = result['return_pct'] * (365 / days_held)
            
            # Calculate spot move
            result['spot_move_pct'] = ((result['exit_spot'] - result['entry_spot']) / result['entry_spot']) * 100
            
            # Calculate realized volatility
            result['realized_volatility'] = self.calculate_realized_volatility(
                result['entry_spot'], result['exit_spot'], days_held
            )
            
            # Calculate IV vs RV spread
            if result['realized_volatility'] is not None:
                result['iv_rv_spread'] = result['entry_iv'] - result['realized_volatility']
            
            # Calculate expected move (from IV)
            result['expected_move'] = result['entry_iv'] * np.sqrt(days_held / 252) * result['entry_spot']
            
            # Calculate move vs expected
            actual_move = abs(result['exit_spot'] - result['entry_spot'])
            if result['expected_move'] and result['expected_move'] > 0:
                result['move_vs_expected'] = actual_move / result['expected_move']
            
        except ValueError as e:
            # Expected errors (missing data, build failures) - don't log traceback
            error_msg = str(e)
            if 'No data found' in error_msg or 'not found' in error_msg.lower():
                result['failure_reason'] = 'data_missing'
            else:
                result['failure_reason'] = f'value_error_{error_msg[:50]}'
        
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.error(
                f"Unexpected error processing {ticker} on {entry_date}: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            result['failure_reason'] = f'error_{type(e).__name__}_{str(e)[:100]}'
        
        # Record processing time
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
