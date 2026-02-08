"""Backtest engine: orchestrates backtest execution

The BacktestEngine is the main orchestrator for running backtests of option trading strategies.
It coordinates between data providers, strategy signal generation, portfolio optimization,
and trade execution to simulate historical performance.

Key responsibilities:
- Generate trading dates based on available feature data
- Close expired positions using intrinsic value settlement
- Generate trading signals from the strategy
- Build option strategies from signals
- Optimize portfolio allocation
- Execute trades via the executor
- Track performance metrics

Example usage:
    # Setup components
    data_provider = ORATSDataProvider(data_root='c:/ORATS/data/ORATS_Adjusted')
    features = pd.read_parquet('cache/features_all.parquet')
    strategy = MomentumCVGStrategy(max_lag=60, min_lag=8, cvg_filter_pct=0.50)
    optimizer = EqualWeightOptimizer(max_positions=10)
    executor = BacktestExecutor(data_provider=data_provider)
    
    # Create engine
    engine = BacktestEngine(
        features=features,
        strategy=strategy,
        optimizer=optimizer,
        executor=executor,
        data_provider=data_provider
    )
    
    # Configure and run backtest
    config = BacktestConfig(
        initial_capital=100000,
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        target_dte=30,
        max_positions=10
    )
    
    results = engine.run(config)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal

import pandas as pd
import numpy as np

from src.core.models import Position, Signal
from src.strategy.base import IStrategy
from src.strategy.builders import StraddleBuilder
from src.portfolio.base import IPortfolioOptimizer
from src.execution.backtest_executor import IExecutor
from src.data.orats_provider import IDataProvider

logger = logging.getLogger(__name__)


@dataclass
class BacktestExecutionConfig:
    """Execution configuration for a backtest run (internal engine config)
    
    Note: This is the internal config used by BacktestEngine.run().
    For full backtest configuration (strategy, data, universe, etc.),
    use src.backtest.config.BacktestConfig instead.
    
    Attributes:
        initial_capital: Starting capital in dollars
        start_date: First trading date
        end_date: Last trading date
        target_dte: Target days to expiration for new positions
        max_positions: Maximum number of concurrent positions
        capital_mode: Capital allocation mode
            - 'compound': Reinvest P&L (available_capital grows/shrinks)
            - 'fixed': Use initial_capital for every rebalance (isolates strategy performance)
        position_constraints: Additional constraints passed to optimizer
        hooks: Optional callbacks for Phase 2 extensibility
            - pre_rebalance: Called before each rebalance, can skip date
            - post_rebalance: Called after positions opened
            - pre_exit: Called before closing positions, can prevent close
            - post_exit: Called after positions closed
    """
    initial_capital: Decimal
    start_date: date
    end_date: date
    target_dte: int = 7
    max_positions: int = 10
    capital_mode: str = 'compound'
    position_constraints: Dict[str, Any] = field(default_factory=dict)
    hooks: Dict[str, Callable] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.initial_capital, Decimal):
            self.initial_capital = Decimal(str(self.initial_capital))
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.target_dte < 1:
            raise ValueError("target_dte must be at least 1")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.capital_mode not in ('compound', 'fixed'):
            raise ValueError("capital_mode must be 'compound' or 'fixed'")


class BacktestEngine:
    """Main orchestrator for option strategy backtesting
    
    The engine coordinates the full backtest workflow:
    1. Generate trading dates from feature data (data-driven)
    2. For each date:
       a. Close expired positions
       b. Generate signals from strategy
       c. Build option strategies from signals
       d. Optimize portfolio allocation
       e. Execute new positions
    3. Calculate performance metrics
    
    Attributes:
        features: Pre-computed features DataFrame (must have 'date' and 'ticker' columns)
        strategy: Strategy that generates signals
        optimizer: Portfolio optimizer for position sizing
        executor: Trade executor for entry/exit simulation
        data_provider: Provides option chain data
        open_positions: Currently open positions
        closed_positions: Positions that have been closed
        capital_history: Daily capital snapshots for performance calculation
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        strategy: IStrategy,
        optimizer: IPortfolioOptimizer,
        executor: IExecutor,
        data_provider: IDataProvider
    ):
        """Initialize backtest engine
        
        Args:
            features: Pre-computed features DataFrame with 'date' and 'ticker' columns
            strategy: Strategy for signal generation
            optimizer: Portfolio optimizer for position sizing
            executor: Trade executor for simulation
            data_provider: Data provider for option chains
            
        Raises:
            ValueError: If features missing required columns
        """
        # Validate features DataFrame
        required_cols = {'date', 'ticker'}
        missing = required_cols - set(features.columns)
        if missing:
            raise ValueError(f"Features DataFrame missing required columns: {missing}")
        
        self.features = features
        self.strategy = strategy
        self.optimizer = optimizer
        self.executor = executor
        self.data_provider = data_provider
        
        # State tracking
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.capital_history: List[Dict] = []
        
        logger.info(
            f"Initialized BacktestEngine with {len(features)} feature rows, "
            f"strategy={strategy.__class__.__name__}, "
            f"optimizer={optimizer.__class__.__name__}, "
            f"executor={executor.execution_mode}"
        )
    
    def run(self, config: BacktestExecutionConfig) -> Dict[str, Any]:
        """Run backtest with given configuration
        
        Main orchestration loop:
        1. Generate trading dates from features (data-driven, no calendar math)
        2. For each date:
           - Pre-rebalance hook (optional, Phase 2)
           - Close expired positions
           - Generate signals
           - Build option strategies
           - Optimize portfolio
           - Execute entries
           - Post-rebalance hook (optional, Phase 2)
        3. Close all remaining positions at end date
        4. Calculate performance metrics
        
        Args:
            config: Backtest configuration
            
        Returns:
            Dictionary with performance metrics:
                - total_return: Overall return percentage
                - sharpe_ratio: Annualized Sharpe ratio
                - max_drawdown: Maximum peak-to-trough drawdown
                - num_trades: Total number of trades
                - win_rate: Percentage of profitable trades
                - avg_win: Average profit on winning trades
                - avg_loss: Average loss on losing trades
                - equity_curve: List of daily capital values
                - closed_positions: List of all closed Position objects
        """
        logger.info(
            f"Starting backtest: {config.start_date} to {config.end_date}, "
            f"initial_capital=${config.initial_capital:,.2f}, "
            f"capital_mode={config.capital_mode}"
        )
        
        # Reset state
        self.open_positions = []
        self.closed_positions = []
        self.capital_history = []
        
        # Generate trading dates from features (data-driven)
        trade_dates = self._generate_trade_dates(config.start_date, config.end_date)
        logger.info(f"Generated {len(trade_dates)} trading dates from features")
        
        # Track capital (actual P&L vs available for allocation)
        actual_capital = config.initial_capital
        
        # Main backtest loop
        for i, trade_date in enumerate(trade_dates):
            logger.info(f"[{i+1}/{len(trade_dates)}] Processing {trade_date}")
            
            # Pre-rebalance hook (Phase 2: regime filters, skip conditions, etc.)
            if 'pre_rebalance' in config.hooks:
                state = {
                    'date': trade_date,
                    'capital': actual_capital,
                    'open_positions': self.open_positions.copy(),
                    'closed_positions': self.closed_positions.copy()
                }
                should_skip = config.hooks['pre_rebalance'](state)
                if should_skip:
                    logger.info(f"  Skipping rebalance due to pre_rebalance hook")
                    continue
            
            # Step 1: Close expired positions
            positions_to_close = self._get_expired_positions(trade_date)
            if positions_to_close:
                logger.info(f"  Closing {len(positions_to_close)} expired positions")
                
                # Pre-exit hook (Phase 2: custom exit logic)
                if 'pre_exit' in config.hooks:
                    positions_to_close = config.hooks['pre_exit'](
                        trade_date, positions_to_close
                    )
                
                closed = self._close_positions(positions_to_close, trade_date)
                
                # Add exit_value (settlement proceeds), not pnl
                # Entry cost was already subtracted when position opened
                actual_capital += sum(p.exit_value for p in closed)
                
                # Calculate total P&L for logging
                total_pnl = sum(p.pnl for p in closed)
                logger.info(
                    f"  Realized P&L: ${total_pnl:,.2f}, "
                    f"Actual capital: ${actual_capital:,.2f}"
                )
                
                # Post-exit hook (Phase 2: logging, notifications, etc.)
                if 'post_exit' in config.hooks:
                    config.hooks['post_exit'](trade_date, closed)
            
            # Step 2: Generate signals from strategy
            logger.info(f"  Generating signals for {trade_date}")
            # Filter features for current trade date
            features_for_date = self.features[self.features['date'] == pd.Timestamp(trade_date)]            
            signals = self.strategy.generate_signals(features_for_date, trade_date)
            logger.info(f"  Generated {len(signals)} signals")
            
            if not signals:
                # Record capital snapshot even if no trades
                self.capital_history.append({
                    'date': trade_date,
                    'capital': actual_capital,
                    'num_positions': len(self.open_positions)
                })
                continue
            
            # Step 3: Build option strategies from signals
            logger.info(f"  Building option strategies")
            option_strategies = self._build_strategies(
                signals, trade_date, config.target_dte
            )
            logger.info(f"  Built {len(option_strategies)} option strategies")
            
            if not option_strategies:
                self.capital_history.append({
                    'date': trade_date,
                    'capital': actual_capital,
                    'num_positions': len(self.open_positions)
                })
                continue
            
            # Step 4: Optimize portfolio allocation
            logger.info(f"  Optimizing portfolio")
            
            # Determine available capital for allocation
            if config.capital_mode == 'fixed':
                available_capital = config.initial_capital
                logger.debug(f"  Fixed mode: available_capital=${available_capital:,.2f}")
            else:  # 'compound'
                available_capital = actual_capital
                logger.debug(f"  Compound mode: available_capital=${available_capital:,.2f}")
            
            # Unpack option_strategies into separate structures for optimizer
            signals_list = [sig for sig, _ in option_strategies]
            strategies_dict = {sig.ticker: strat for sig, strat in option_strategies}
            
            new_positions = self.optimizer.optimize(
                signals=signals_list,
                option_strategies=strategies_dict,
                current_positions=self.open_positions,
                available_capital=available_capital,
                current_date=trade_date,
                constraints={'max_positions': config.max_positions, **config.position_constraints}
            )
            logger.info(f"  Optimizer allocated {len(new_positions)} new positions")
            
            # Step 5: Add positions and update capital
            # Note: Optimizer already created Position objects with entry_cost set
            # No need for executor.execute_entry() in Phase 1 (perfect fills assumed)
            total_entry_cost = Decimal('0')
            for position in new_positions:
                self.open_positions.append(position)
                total_entry_cost += position.entry_cost
                logger.debug(
                    f"    Opened {position.ticker} "
                    f"{position.quantity:.3f}x {position.strategy.strategy_type}, "
                    f"cost=${position.entry_cost:,.2f}"
                )
            
            # Update actual capital (always tracks true cash flow)
            actual_capital -= total_entry_cost
            
            logger.info(
                f"  Opened {len(new_positions)} positions "
                f"(cost=${total_entry_cost:,.2f}), "
                f"actual capital: ${actual_capital:,.2f}"
            )
            
            # Post-rebalance hook (Phase 2: delta hedging, custom adjustments)
            if 'post_rebalance' in config.hooks:
                config.hooks['post_rebalance'](trade_date, new_positions)
            
            # Record capital snapshot
            self.capital_history.append({
                'date': trade_date,
                'capital': actual_capital,
                'num_positions': len(self.open_positions)
            })
        
        # Step 6: Close all remaining positions at end date
        if self.open_positions:
            logger.info(f"Closing {len(self.open_positions)} remaining positions at {config.end_date}")
            closed = self._close_positions(self.open_positions.copy(), config.end_date)
            
            # Add exit_value, not pnl (entry_cost already subtracted at entry)
            actual_capital += sum(p.exit_value for p in closed)
            logger.info(f"Final capital: ${actual_capital:,.2f}")
            
            # Final capital snapshot
            self.capital_history.append({
                'date': config.end_date,
                'capital': actual_capital,
                'num_positions': 0
            })
        
        # Step 7: Calculate performance metrics
        logger.info("Calculating performance metrics")
        performance = self._generate_performance_report(config.initial_capital)
        
        logger.info(
            f"Backtest complete: {len(self.closed_positions)} trades, "
            f"total_return={performance['total_return']:.2%}, "
            f"sharpe={performance['sharpe_ratio']:.2f}"
        )
        
        return performance
    
    def _generate_trade_dates(self, start_date: date, end_date: date) -> List[date]:
        """Generate trading dates from features DataFrame
        
        Extract unique dates from features that fall within [start_date, end_date].
        This is data-driven (not calendar-based), ensuring we only trade when
        features are available and avoiding manual holiday/weekend handling.
        
        Args:
            start_date: First trading date
            end_date: Last trading date
            
        Returns:
            Sorted list of trading dates
        """
        # Filter features by date range
        date_mask = (
            (self.features['date'] >= pd.Timestamp(start_date)) &
            (self.features['date'] <= pd.Timestamp(end_date))
        )
        
        # Extract unique dates
        trade_dates = self.features.loc[date_mask, 'date'].unique()
        
        # Convert Timestamps to date objects and sort
        return sorted([ts.date() for ts in trade_dates])
    
    def _get_expired_positions(self, current_date: date) -> List[Position]:
        """Get positions that have expired (max_expiry <= current_date)
        
        For multi-leg strategies with different expiries (e.g., calendar spreads),
        a position is considered expired when ALL legs have expired (max_expiry).
        
        Args:
            current_date: Current trading date
            
        Returns:
            List of expired positions
        """
        return [
            pos for pos in self.open_positions
            if pos.strategy.max_expiry <= current_date
        ]
    
    def _close_positions(
        self, positions: List[Position], exit_date: date
    ) -> List[Position]:
        """Close positions and update tracking
        
        Args:
            positions: Positions to close
            exit_date: Date of exit
            
        Returns:
            List of closed positions with exit_date, exit_value, pnl populated
        """
        closed = []
        for position in positions:
            try:
                # Get spot price at exit date
                spot_price = self.data_provider.get_spot_price(
                    ticker=position.ticker,
                    trade_date=exit_date
                )
                
                if spot_price is None:
                    logger.warning(
                        f"  No spot price for {position.ticker} on {exit_date}, "
                        f"skipping position close"
                    )
                    continue
                
                # Execute exit with spot price
                closed_position = self.executor.execute_exit(
                    position=position,
                    spot_price=spot_price,
                    exit_date=exit_date
                )
                
                self.closed_positions.append(closed_position)
                self.open_positions.remove(position)
                closed.append(closed_position)
                
                logger.debug(
                    f"  Closed {closed_position.ticker} "
                    f"{closed_position.quantity}x {closed_position.strategy.strategy_type}, "
                    f"spot=${spot_price:,.2f}, P&L=${closed_position.pnl:,.2f}"
                )
            except Exception as e:
                logger.error(f"  Failed to close position {position.ticker}: {e}")
                continue
        
        return closed
    
    def _build_strategies(
        self,
        signals: List[Signal],
        trade_date: date,
        target_dte: int
    ) -> List[tuple[Signal, Any]]:  # Any = OptionStrategy
        """Build option strategies from signals
        
        For each signal:
        1. Find appropriate expiry date (target_dte)
        2. Load option chain
        3. Get spot price
        4. Build strategy using StraddleBuilder
        
        Args:
            signals: List of trading signals
            trade_date: Current trading date
            target_dte: Target days to expiration
            
        Returns:
            List of (signal, option_strategy) tuples for optimizer
        """
        results = []
        builder = StraddleBuilder()
        expiry_date = None
        for signal in signals:
            try:
                # Find expiry date using actual ORATS data (NOT calendar math)
                if expiry_date is None:
                    expiry_date = self._find_expiry_date(
                        ticker=signal.ticker,
                        trade_date=trade_date,
                        target_dte=target_dte
                    )
                
                if expiry_date is None:
                    logger.warning(
                        f"  No suitable expiry for {signal.ticker} "
                        f"on {trade_date} (target={target_dte} DTE)"
                    )
                    continue                
                
                # Load option chain for found expiry
                chain = self.data_provider.get_option_chain(
                    ticker=signal.ticker,
                    trade_date=trade_date,
                    expiry_date=expiry_date
                )
                
                if not chain:
                    logger.warning(
                        f"  No option chain for {signal.ticker} "
                        f"{trade_date} -> {expiry_date}"
                    )
                    continue
                
                # Get spot price
                spot_price = self.data_provider.get_spot_price(
                    ticker=signal.ticker,
                    trade_date=trade_date
                )
                
                if spot_price is None:
                    logger.warning(
                        f"  No spot price for {signal.ticker} on {trade_date}"
                    )
                    continue
                
                # Build strategy with updated signature
                option_strategy = builder.build_strategy(
                    ticker=signal.ticker,
                    trade_date=trade_date,
                    expiry_date=expiry_date,
                    option_chain=chain,
                    spot_price=spot_price
                )

                if option_strategy:
                    results.append((signal, option_strategy))
                    logger.debug(
                        f"  Built {signal.ticker} {signal.strategy_type} "
                        f"{trade_date} -> {expiry_date} "
                        f"({(expiry_date - trade_date).days} DTE)"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"  Failed to build strategy for {signal.ticker}: {e}"
                )
                continue
        
        return results
    
    def _find_expiry_date(self, ticker: str, trade_date: date, target_dte: int) -> Optional[date]:
        """Find expiry date based on target DTE using actual ORATS data
        
        Queries data provider for available expiries and selects the best match:
        - For monthly (target_dte >= 28): First Friday/Thursday of next month
        - For weekly (target_dte < 28): Closest expiry to target, preferring Friday
        
        This ensures returned expiry has actual option data available.
        
        Args:
            ticker: Stock ticker
            trade_date: Current trading date
            target_dte: Target days to expiration
            
        Returns:
            Best expiry date, or None if no suitable expiry found
        """
        try:
            # Get available expiries from data provider
            expiries = self.data_provider.get_available_expiries(ticker, trade_date)
            
            if not expiries:
                logger.warning(f"No expiries available for {ticker} on {trade_date}")
                return None
            
            # Monthly rebalancing (target DTE >= 28 days)
            if target_dte >= 28:
                # Target first Friday/Thursday of next month
                next_month = trade_date.month + 1 if trade_date.month < 12 else 1
                next_year = trade_date.year if trade_date.month < 12 else trade_date.year + 1
                
                # Filter to Fri/Thu in target month
                target_month_expiries = [
                    exp for exp in expiries
                    if exp.year == next_year
                    and exp.month == next_month
                    and exp.weekday() in [3, 4]  # Thursday or Friday
                    and exp > trade_date  # Must be in future
                ]
                
                if target_month_expiries:
                    # Return earliest (first Fri/Thu of month)
                    target_month_expiries.sort()
                    best_expiry = target_month_expiries[0]
                    
                    # Sanity check DTE (should be 20-45 days for monthly)
                    dte = (best_expiry - trade_date).days
                    if dte < 20 or dte > 45:
                        logger.warning(
                            f"{ticker} on {trade_date}: Monthly expiry {best_expiry} "
                            f"is {dte} DTE (unusual, expected 20-45)"
                        )
                    
                    return best_expiry
                
                # Fallback: no Fri/Thu in next month, find any expiry ~30 days out
                logger.warning(
                    f"{ticker} on {trade_date}: No Fri/Thu expiry in "
                    f"{next_year}-{next_month:02d}, using fallback"
                )
                expiry_diffs = [
                    (exp, abs((exp - trade_date).days - target_dte))
                    for exp in expiries
                    if exp > trade_date
                ]
                
                if expiry_diffs:
                    best_expiry = min(expiry_diffs, key=lambda x: x[1])
                    if best_expiry[1] <= 10:  # Within ±10 days tolerance
                        return best_expiry[0]
                
                logger.warning(
                    f"{ticker} on {trade_date}: No suitable monthly expiry found "
                    f"(target={target_dte} DTE)"
                )
                return None
            
            # Weekly rebalancing (target DTE < 28 days)
            else:
                # Calculate DTE for all future expiries
                expiry_dtes = [
                    (exp, (exp - trade_date).days)
                    for exp in expiries
                    if exp > trade_date
                ]
                
                # Filter to expiries within tolerance (±4 days for weekly)
                tolerance = 4
                valid_expiries = [
                    (exp, dte) for exp, dte in expiry_dtes
                    if 0 < dte <= (target_dte + tolerance)
                ]
                
                if not valid_expiries:
                    logger.warning(
                        f"{ticker} on {trade_date}: No expiry within "
                        f"{target_dte}±{tolerance} days"
                    )
                    return None
                
                # Prefer Friday expiries
                friday_expiries = [
                    (exp, dte) for exp, dte in valid_expiries
                    if exp.weekday() == 4
                ]
                
                if friday_expiries:
                    # Return Friday closest to target DTE
                    best_expiry = min(
                        friday_expiries,
                        key=lambda x: abs(x[1] - target_dte)
                    )
                    return best_expiry[0]
                
                # No Fridays, try Thursday
                thursday_expiries = [
                    (exp, dte) for exp, dte in valid_expiries
                    if exp.weekday() == 3
                ]
                
                if thursday_expiries:
                    logger.debug(
                        f"{ticker} on {trade_date}: No Friday expiry, using Thursday"
                    )
                    best_expiry = min(
                        thursday_expiries,
                        key=lambda x: abs(x[1] - target_dte)
                    )
                    return best_expiry[0]
                
                # Last resort: any day within tolerance
                logger.debug(
                    f"{ticker} on {trade_date}: No Fri/Thu expiry, using any day"
                )
                best_expiry = min(
                    valid_expiries,
                    key=lambda x: abs(x[1] - target_dte)
                )
                return best_expiry[0]
        
        except Exception as e:
            logger.error(
                f"Error finding expiry for {ticker} on {trade_date}: {e}",
                exc_info=True
            )
            return None
    
    def _generate_performance_report(
        self, initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Calculate performance metrics from closed positions and capital history
        
        Args:
            initial_capital: Starting capital
            
        Returns:
            Dictionary with performance metrics:
                - total_return: Overall return percentage
                - sharpe_ratio: Annualized Sharpe ratio (assumes daily rebalancing)
                - max_drawdown: Maximum peak-to-trough drawdown
                - num_trades: Total number of trades
                - win_rate: Percentage of profitable trades
                - avg_win: Average profit on winning trades
                - avg_loss: Average loss on losing trades
                - equity_curve: DataFrame with daily capital values
                - closed_positions: List of all closed Position objects
        """
        if not self.capital_history:
            logger.warning("No capital history to generate report")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'equity_curve': pd.DataFrame(),
                'closed_positions': []
            }
        
        # Build equity curve DataFrame
        equity_df = pd.DataFrame(self.capital_history)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.sort_values('date')
        
        # Convert Decimal to float for pandas numerical operations
        equity_df['capital'] = equity_df['capital'].apply(float)
        initial_capital_float = float(initial_capital)
        
        # Calculate returns
        final_capital = equity_df['capital'].iloc[-1]
        total_return = (final_capital - initial_capital_float) / initial_capital_float
        
        # Calculate daily returns for Sharpe ratio
        equity_df['returns'] = equity_df['capital'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade statistics
        num_trades = len(self.closed_positions)
        if num_trades > 0:
            winning_trades = [p for p in self.closed_positions if p.pnl > 0]
            losing_trades = [p for p in self.closed_positions if p.pnl < 0]
            
            win_rate = len(winning_trades) / num_trades
            avg_win = (
                sum(float(p.pnl) for p in winning_trades) / len(winning_trades)
                if winning_trades else 0.0
            )
            avg_loss = (
                sum(float(p.pnl) for p in losing_trades) / len(losing_trades)
                if losing_trades else 0.0
            )
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': equity_df,
            'closed_positions': self.closed_positions
        }
