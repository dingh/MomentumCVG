"""
Build feature dataset from straddle history.

This script generates pre-computed features for backtesting by applying
multiple feature calculators (momentum, CVG, etc.) to straddle history data.

Windows are automatically generated using generate_momentum_windows() function
which creates all combinations of (max_lag, min_lag) within specified ranges.

Usage:
    # Basic usage with default ranges (short: 2-24, long: 6-60, step: 2)
    python scripts/build_features.py \
        --input cache/straddle_history_weekly_2018_2024.parquet \
        --output cache/straddle_features_weekly_2018_2024.parquet

    # Custom ranges for narrow testing
    python scripts/build_features.py \
        --input cache/straddle_history_weekly_2018_2024.parquet \
        --output cache/features_narrow.parquet \
        --short-min 4 --short-max 12 \
        --long-min 12 --long-max 30 \
        --step 4
        
    # Date range filtering
    python scripts/build_features.py \
        --input cache/straddle_history_weekly_2018_2024.parquet \
        --output cache/features_2023.parquet \
        --start-date 2023-01-01 \
        --end-date 2023-12-31
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.base import IFeatureCalculator, FeatureDataContext
from src.features.momentum_calculator import MomentumCalculator
from src.features.cvg_calculator import CVGCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_momentum_windows(
    short_range: Tuple[int, int] = (2, 24),
    long_range: Tuple[int, int] = (12, 60),
    step: int = 2
) -> List[Tuple[int, int]]:
    """
    Generate momentum window combinations.
    
    Args:
        short_range: (min, max) for short lag (default: 2 to 24)
        long_range: (min, max) for long lag (default: 6 to 60)
        step: Step size for generating windows (default: 2)
    
    Returns:
        List of (max_lag, min_lag) tuples
        
    Logic:
        - Short lag (min_lag): 2, 4, 6, ..., 24
        - Long lag (max_lag): 6, 8, 10, ..., 60
        - Only keep windows where max_lag > min_lag
        
    Example output:
        [(6, 2), (8, 2), (10, 2), ..., (60, 2),
         (8, 4), (10, 4), ..., (60, 4),
         ...
         (60, 24)]
    """
    windows = []
    
    short_lags = range(short_range[0], short_range[1] + 1, step)
    long_lags = range(long_range[0], long_range[1] + 1, step)
    
    for max_lag in long_lags:
        for min_lag in short_lags:
            if max_lag > min_lag:  # Must have valid window
                windows.append((max_lag, min_lag))
    
    logger.info(f"Generated {len(windows)} momentum windows:")
    logger.info(f"  Short lags (min_lag): {list(short_lags)}")
    logger.info(f"  Long lags (max_lag): {list(long_lags)}")
    logger.info(f"  First 5: {windows[:5]}")
    logger.info(f"  Last 5: {windows[-5:]}")
    
    return windows


def build_features(
    calculators: List[IFeatureCalculator],
    context: FeatureDataContext,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Build features using multiple calculators (follows IFeatureCalculator protocol).
    
    This is a general-purpose function that works with any calculator implementing
    the IFeatureCalculator protocol. Each calculator's features are computed
    independently and then merged on (ticker, date).
    
    Args:
        calculators: List of feature calculator instances (e.g., [MomentumCalculator(), CVGCalculator()])
        context: FeatureDataContext with required data sources (e.g., straddle_history)
        start_date: Start date for feature calculation (inclusive)
        end_date: End date for feature calculation (inclusive)
        
    Returns:
        DataFrame with columns: ticker, date, <all_features>
        All features from all calculators merged on (ticker, date)
        
    Raises:
        ValueError: If calculators list is empty or required data sources are missing
    """
    
    logger.info(f"Building features from {start_date.date()} to {end_date.date()}")
    logger.info(f"Using {len(calculators)} calculator(s)")
    
    if len(calculators) == 0:
        raise ValueError("No calculators provided!")
    
    # Validate context has all required data sources
    _validate_context(calculators, context)
    
    # Calculate features from each calculator
    all_features = []
    
    for i, calc in enumerate(calculators, 1):
        calc_name = calc.__class__.__name__
        logger.info(f"[{i}/{len(calculators)}] Calculating {calc_name} features...")
        logger.info(f"  Feature names: {calc.feature_names}")
        
        try:
            # Call calculate_bulk (uses protocol method)
            features = calc.calculate_bulk(
                context=context,
                start_date=start_date,
                end_date=end_date,
                tickers=None  # All tickers
            )
            
            logger.info(f"  ✓ Generated {len(features):,} records")
            all_features.append(features)
            
        except Exception as e:
            logger.error(f"  ✗ Error in {calc_name}: {e}")
            raise
    
    # Merge all features on (ticker, date)
    logger.info("Merging features from all calculators...")
    result = _merge_features(all_features)
    
    logger.info(f"✓ Final dataset: {len(result):,} records with {len(result.columns)} columns")
    
    return result


def _validate_context(
    calculators: List[IFeatureCalculator],
    context: FeatureDataContext
) -> None:
    """
    Validate that context has all required data sources.
    
    Args:
        calculators: List of calculator instances
        context: FeatureDataContext to validate
        
    Raises:
        ValueError: If any required data source is missing
    """
    available = set(context.available_sources)
    
    for calc in calculators:
        calc_name = calc.__class__.__name__
        required = set(calc.required_data_sources)
        missing = required - available
        
        if missing:
            raise ValueError(
                f"{calc_name} requires data sources: {missing}\n"
                f"Available in context: {available}"
            )
    
    logger.info(f"✓ Context validation passed")
    logger.info(f"  Available data sources: {sorted(available)}")


def _merge_features(feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge feature DataFrames on (ticker, date).
    
    Args:
        feature_dfs: List of DataFrames, each with columns [ticker, date, features...]
        
    Returns:
        Single DataFrame with all features merged on (ticker, date)
        
    Raises:
        ValueError: If feature_dfs is empty or merge fails
    """
    if len(feature_dfs) == 0:
        raise ValueError("No feature DataFrames to merge!")
    
    if len(feature_dfs) == 1:
        return feature_dfs[0]
    
    # Start with first DataFrame
    result = feature_dfs[0].copy()
    
    # Merge each subsequent DataFrame
    for i, df in enumerate(feature_dfs[1:], 2):
        logger.info(f"  Merging calculator {i}/{len(feature_dfs)}...")
        
        result = result.merge(
            df,
            on=['ticker', 'date'],
            how='outer',
            validate='1:1'  # Ensure no duplicates
        )
    
    return result


def create_calculators(
    args: argparse.Namespace
) -> List[IFeatureCalculator]:
    """
    Create calculator instances based on CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        List of calculator instances implementing IFeatureCalculator
        
    Raises:
        ValueError: If no calculators are enabled
    """
    calculators = []
    
    # Generate windows from ranges
    windows = generate_momentum_windows(
        short_range=(args.short_min, args.short_max),
        long_range=(args.long_min, args.long_max),
        step=args.step
    )
    
    # Add momentum calculator
    if not args.skip_momentum:
        calculators.append(
            MomentumCalculator(
                windows=windows,
                min_periods=args.min_periods_momentum
            )
        )
    
    # Add CVG calculator
    if not args.skip_cvg:
        calculators.append(
            CVGCalculator(
                windows=windows,
                min_periods=args.min_periods_cvg
            )
        )
    
    # Future calculators can be added here:
    # if args.include_volatility:
    #     calculators.append(VolatilitySurfaceCalculator(...))
    # if args.include_greeks:
    #     calculators.append(GreeksCalculator(...))
    
    if len(calculators) == 0:
        raise ValueError("No calculators specified! (all calculators skipped)")
    
    logger.info(f"Created {len(calculators)} calculator(s) with {len(windows)} window(s) each:")
    for calc in calculators:
        logger.info(f"  - {calc.__class__.__name__}: {len(calc.feature_names)} features")
    
    return calculators


def load_straddle_history(path: str) -> pd.DataFrame:
    """
    Load and validate straddle history from parquet.
    
    Args:
        path: Path to straddle history parquet file
        
    Returns:
        DataFrame with validated straddle history
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    logger.info(f"Loading straddle history from: {path}")
    
    df = pd.read_parquet(path)
    
    # Validate required columns
    # return_pct is needed by MomentumCalculator;
    # vol_gap (or realized_volatility + entry_iv) is needed by CVGCalculator.
    required = ['ticker', 'entry_date', 'return_pct']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # CVGCalculator requires vol_gap (or its component columns).
    # Check now so the error is clear and points here, not buried in the calculator.
    has_vol_gap = 'vol_gap' in df.columns
    has_components = ('realized_volatility' in df.columns and 'entry_iv' in df.columns)
    if not has_vol_gap and not has_components:
        raise ValueError(
            "CVGCalculator requires either a 'vol_gap' column or both "
            "'realized_volatility' and 'entry_iv' columns. "
            "Neither found in the input file. "
            "If you only want momentum features, re-run with --skip-cvg."
        )
    
    # Convert dates
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    
    # Log summary
    logger.info(f"✓ Loaded {len(df):,} straddle records")
    logger.info(f"  Date range: {df['entry_date'].min().date()} to {df['entry_date'].max().date()}")
    logger.info(f"  Tickers: {df['ticker'].nunique()}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df


def validate_and_summarize(
    df_features: pd.DataFrame,
    calculators: List[IFeatureCalculator]
) -> None:
    """
    Validate features and print summary statistics.
    
    Args:
        df_features: Generated features DataFrame
        calculators: List of calculators used (for feature name lookup)
    """
    logger.info("\n" + "="*80)
    logger.info("FEATURE SUMMARY")
    logger.info("="*80)
    
    # Basic stats
    logger.info(f"Total records: {len(df_features):,}")
    logger.info(f"Unique tickers: {df_features['ticker'].nunique()}")
    logger.info(f"Date range: {df_features['date'].min().date()} to {df_features['date'].max().date()}")
    logger.info(f"Total columns: {len(df_features.columns)}")
    
    # Per-calculator summary
    for calc in calculators:
        calc_name = calc.__class__.__name__
        logger.info(f"\n{calc_name} Features:")
        
        for feature in calc.feature_names:
            if feature in df_features.columns:
                completeness = (1 - df_features[feature].isna().mean()) * 100
                logger.info(f"  {feature:40s}: {completeness:5.1f}% complete")
    
    # CVG validation (if CVGCalculator was used)
    for calc in calculators:
        if calc.__class__.__name__ == 'CVGCalculator' and calc.feature_names:
            # Pick the first cvg_* feature as the representative column
            cvg_col = next((f for f in calc.feature_names if f.startswith('cvg_') and 'count' not in f), None)
            if cvg_col and cvg_col in df_features.columns:
                cvg = df_features[cvg_col].dropna()
                logger.info(f"\nCVG Statistics ({cvg_col}):")
                logger.info(f"  Count: {len(cvg):,}")
                logger.info(f"  Mean: {cvg.mean():.3f} (paper: ~1.24)")
                logger.info(f"  Std: {cvg.std():.3f}")
                logger.info(f"  Median: {cvg.median():.3f}")
                logger.info(f"  Min: {cvg.min():.3f}")
                logger.info(f"  Max: {cvg.max():.3f}")

                # Validate range: 0 ≤ CVG ≤ 2
                out_of_range = ((cvg < 0) | (cvg > 2)).sum()
                if out_of_range > 0:
                    logger.warning(f"  ⚠️  {out_of_range} CVG values out of range [0, 2]!")
            break

    # Momentum validation (if MomentumCalculator was used)
    for calc in calculators:
        if calc.__class__.__name__ == 'MomentumCalculator' and calc.feature_names:
            # Pick the first *_mean feature as the representative column
            mom_col = next((f for f in calc.feature_names if f.endswith('_mean')), None)
            if mom_col and mom_col in df_features.columns:
                mom = df_features[mom_col].dropna()
                logger.info(f"\nMomentum Statistics ({mom_col}):")
                logger.info(f"  Count: {len(mom):,}")
                logger.info(f"  Mean: {mom.mean():.4f}")
                logger.info(f"  Std: {mom.std():.4f}")
                logger.info(f"  Median: {mom.median():.4f}")
                logger.info(f"  Min: {mom.min():.4f}")
                logger.info(f"  Max: {mom.max():.4f}")
            break


def save_features(df_features: pd.DataFrame, output_path: str) -> None:
    """
    Save features to parquet with compression.
    
    Args:
        output_path: Path to save features parquet file
    """
    output_path_obj = Path(output_path)
    
    # Create output directory if needed
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving features to: {output_path}")
    
    df_features.to_parquet(
        output_path,
        index=False,
        compression='snappy'
    )
    
    file_size = output_path_obj.stat().st_size / 1024 / 1024
    logger.info(f"✅ Features saved successfully")
    logger.info(f"  File size: {file_size:.1f} MB")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Build feature dataset from straddle history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default ranges)
  python scripts/build_features.py \\
      --input cache/straddle_history_weekly_2018_2024.parquet \\
      --output cache/straddle_features_weekly_2018_2024.parquet
  
  # Custom date range
  python scripts/build_features.py \\
      --input cache/straddle_history_weekly_2018_2024.parquet \\
      --output cache/features_2023.parquet \\
      --start-date 2023-01-01 \\
      --end-date 2023-12-31
  
  # Custom window ranges (narrow range for testing)
  python scripts/build_features.py \\
      --input cache/straddle_history_weekly_2018_2024.parquet \\
      --output cache/features_narrow.parquet \\
      --short-min 4 --short-max 12 \\
      --long-min 12 --long-max 30 \\
      --step 4
  
  # Skip CVG (momentum only)
  python scripts/build_features.py \\
      --input cache/straddle_history_weekly_2018_2024.parquet \\
      --output cache/momentum_only.parquet \\
      --skip-cvg
"""
    )
    
    # Input/output
    parser.add_argument(
        '--input',
        required=True,
        help='Path to straddle history parquet file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output features parquet file'
    )
    
    # Date range
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), default: min date in data'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), default: max date in data'
    )
    
    # Window generation parameters
    parser.add_argument(
        '--short-min',
        type=int,
        default=2,
        help='Minimum short lag (min_lag) for window generation (default: 2)'
    )
    parser.add_argument(
        '--short-max',
        type=int,
        default=24,
        help='Maximum short lag (min_lag) for window generation (default: 24)'
    )
    parser.add_argument(
        '--long-min',
        type=int,
        default=6,
        help='Minimum long lag (max_lag) for window generation (default: 6)'
    )
    parser.add_argument(
        '--long-max',
        type=int,
        default=60,
        help='Maximum long lag (max_lag) for window generation (default: 60)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=2,
        help='Step size for window generation (default: 2)'
    )
    
    # Calculator selection
    parser.add_argument(
        '--skip-momentum',
        action='store_true',
        help='Skip momentum features (default: include)'
    )
    parser.add_argument(
        '--skip-cvg',
        action='store_true',
        help='Skip CVG features (default: include)'
    )
    
    # Min periods for calculators
    parser.add_argument(
        '--min-periods-momentum',
        type=int,
        default=3,
        help='Min observations for momentum features (default: 3)'
    )
    parser.add_argument(
        '--min-periods-cvg',
        type=int,
        default=5,
        help='Min observations for CVG features (default: 5)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    try:
        # Parse arguments
        args = parse_args()
        
        logger.info("="*80)
        logger.info("FEATURE GENERATION SCRIPT")
        logger.info("="*80)
        
        # Load straddle history
        straddle_history = load_straddle_history(args.input)
        
        # Parse date range
        if args.start_date:
            start_date = pd.to_datetime(args.start_date)
        else:
            start_date = straddle_history['entry_date'].min()
            logger.info(f"Using min date from data: {start_date.date()}")
        
        if args.end_date:
            end_date = pd.to_datetime(args.end_date)
        else:
            end_date = straddle_history['entry_date'].max()
            logger.info(f"Using max date from data: {end_date.date()}")
        
        # Validate date range
        if start_date > end_date:
            raise ValueError(f"start_date ({start_date.date()}) must be <= end_date ({end_date.date()})")
        
        # Create calculators (uses IFeatureCalculator protocol)
        # This will generate windows internally using generate_momentum_windows()
        calculators = create_calculators(args)
        
        # Create context (follows FeatureDataContext)
        context = FeatureDataContext(straddle_history=straddle_history)
        
        # Build features (general - works with any IFeatureCalculator!)
        logger.info("\n" + "="*80)
        logger.info("FEATURE CALCULATION")
        logger.info("="*80)
        
        features = build_features(
            calculators=calculators,
            context=context,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate & summarize
        validate_and_summarize(features, calculators)
        
        # Save
        save_features(features, args.output)
        
        logger.info("\n" + "="*80)
        logger.info("✅ FEATURE GENERATION COMPLETE!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())


