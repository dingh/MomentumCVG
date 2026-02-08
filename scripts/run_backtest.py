"""
Run backtest from JSON configuration file.

This script loads a complete backtest configuration from JSON,
instantiates all components, runs the backtest, and saves results.

Usage:
    python scripts/run_backtest.py configs/baseline_sp500.json
    python scripts/run_backtest.py configs/all_tickers.json
    python scripts/run_backtest.py configs/momentum_30_4.json --verbose
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestEngine
import pandas as pd

logger = logging.getLogger(__name__)


def save_results(config: BacktestConfig, results: dict):
    """Save backtest results to files"""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trades CSV
    if config.output['save_trades'] and results['closed_positions']:
        trades = []
        for pos in results['closed_positions']:
            trades.append({
                'ticker': pos.ticker,
                'entry_date': pos.entry_date,
                'exit_date': pos.exit_date,
                'strategy_type': pos.strategy.strategy_type.value,
                'quantity': float(pos.quantity),
                'entry_cost': float(pos.entry_cost),
                'exit_value': float(pos.exit_value),
                'pnl': float(pos.pnl),
                'pnl_pct': pos.pnl_pct,
                'holding_period': pos.holding_period,
                'num_legs': len(pos.strategy.legs),
                'net_delta': pos.strategy.net_delta,
                'net_vega': pos.strategy.net_vega
            })
        
        trades_df = pd.DataFrame(trades)
        trades_path = output_dir / config.output['trades_filename']
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trades to {trades_path}")
    
    # Save equity curve CSV
    if config.output['save_equity_curve']:
        equity_curve = results['equity_curve']
        equity_path = output_dir / config.output['equity_curve_filename']
        equity_curve.to_csv(equity_path, index=False)
        logger.info(f"Saved equity curve to {equity_path}")
    
    # Save summary JSON
    if config.output['save_summary_json']:
        summary = {
            'config_name': config.config_name,
            'description': config.description,
            'execution': config.execution,
            'strategy': config.strategy,
            'universe': config.universe,
            'performance': {
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'num_trades': results['num_trades'],
                'win_rate': results['win_rate'],
                'avg_win': results['avg_win'],
                'avg_loss': results['avg_loss']
            },
            'run_timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_dir / config.output['summary_filename']
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_path}")


def print_summary(config: BacktestConfig, results: dict):
    """Print backtest summary to console"""
    print("\n" + "="*80)
    print(f"BACKTEST COMPLETE: {config.config_name}")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Universe: {config.universe['type']}")
    print(f"  Strategy: {config.strategy['type']} "
          f"(momentum {config.strategy['params']['max_lag']}/{config.strategy['params']['min_lag']})")
    print(f"  Period: {config.execution['start_date']} to {config.execution['end_date']}")
    print(f"  Target DTE: {config.execution['target_dte']}")
    print(f"  Max Positions: {config.execution['max_positions']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return:     {results['total_return']:>10.2%}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:>10.2%}")
    
    print(f"\nTrade Statistics:")
    print(f"  Number of Trades: {results['num_trades']:>10}")
    print(f"  Win Rate:         {results['win_rate']:>10.2%}")
    print(f"  Avg Win:          ${results['avg_win']:>10,.2f}")
    print(f"  Avg Loss:         ${results['avg_loss']:>10,.2f}")
    
    print(f"\nResults saved to: {config.output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run backtest from JSON configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python scripts/run_backtest.py configs/baseline_sp500.json
            python scripts/run_backtest.py configs/all_tickers.json --verbose
            python scripts/run_backtest.py configs/momentum_30_4.json
        """
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to JSON config file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = BacktestConfig.from_json(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override logging level if verbose
    if args.verbose:
        config.logging['level'] = 'DEBUG'
    
    # Setup logging
    config.setup_logging()
    
    # Setup output directory
    config.setup_output_dir()
    
    logger.info(f"Starting backtest: {config.config_name}")
    logger.info(f"Description: {config.description}")
    
    try:
        # Instantiate components
        components = config.instantiate_components()
        
        # Create engine
        engine = BacktestEngine(
            features=components['features'],
            strategy=components['strategy'],
            optimizer=components['optimizer'],
            executor=components['executor'],
            data_provider=components['data_provider']
        )
        
        # Run backtest
        results = engine.run(config.to_execution_config())
        
        # Save results
        save_results(config, results)
        
        # Print summary
        print_summary(config, results)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
