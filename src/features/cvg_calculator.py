"""
CVG (Continuous Volatility Gap) Feature Calculator

Calculates CVG features from historical volatility gaps (Realized Vol - Implied Vol).
Implements the IFeatureCalculator protocol.

This module computes CVG/DVG features from a panel of monthly (or weekly)
volatility gaps:

    vol_gap = realized_volatility - implied_volatility

The core definitions follow *Mind the Gap: Continuous Volatility Gaps and Option
Momentum* (Gan & Nguyen). In particular (their Eq. 1):

    DVG = sign(cgap) * (%neg - %pos)   if cgap > 0
        = sign(cgap) * (%pos - %neg)   if cgap < 0
        = 0                             if cgap == 0
    CVG = 1 - DVG

Where:
  - %pos/%neg are computed from monthly volatility gaps after subtracting the
    cross-sectional median in each month (to remove market-wide shifts).
  - cgap is the *cumulative* volatility gap over the formation window,
    **then** adjusted by subtracting the cross-sectional median of cgap.

Important implementation notes / fixes vs the previous version:

Fix #1 - Correct cgap adjustment (paper definition):
    The paper defines cgap as the cumulative RAW volatility gap over the window,
    and then subtracts the cross-sectional median of that cumulative quantity.
    The previous implementation summed per-month median-adjusted gaps, which is
    not generally equal. This file now implements the paper definition.

Fix #2 - Correct handling of sign(cgap) when cgap == 0:
    sign(0) = 0 -> DVG should be 0 -> CVG should be 1.
    The previous implementation fell into the "negative" branch and could
    produce nonzero DVG when cgap was exactly 0.

Fix #3 - BUGFIX in calculate_bulk(): groupwise shift:
    After groupby().rolling(), pandas returns a MultiIndex series.
    Calling .shift(min_lag) shifts the *entire* series, which can leak values
    across ticker boundaries. We now shift *within* each ticker via
    .groupby(level=0).shift(min_lag).

CVG measures trend continuity:
- High CVG (>1): Continuous trend (all gaps same sign as cumulative)
- CVG = 1: Perfectly balanced trend
- Low CVG (<1): Discontinuous/choppy trend (many reversals)

Both longs and shorts prefer HIGH CVG (continuous trends).

Features computed per window:
- cvg: Continuous Volatility Gap (primary signal for trend quality)
- dvg: Discontinuous Volatility Gap (cvg = 1 - dvg)
- cgap: Cumulative adjusted volatility gap (paper definition: raw cgap minus
        cross-sectional median, NOT sum of per-month-adjusted gaps)
- pct_pos: Percentage of positive per-month-adjusted gaps
- pct_neg: Percentage of negative per-month-adjusted gaps
- volgap_mean: Mean per-month-adjusted volatility gap
- cvg_count: Number of observations

Example:
    >>> from src.features.base import FeatureDataContext
    >>> from src.features.cvg_calculator import CVGCalculator
    >>> 
    >>> # Setup
    >>> context = FeatureDataContext(
    ...     straddle_history=pd.read_parquet('straddles.parquet')
    ... )
    >>> calculator = CVGCalculator(windows=[(12, 2)])
    >>> 
    >>> # Single date calculation (live trading)
    >>> features = calculator.calculate(context, date, tickers)
    >>> 
    >>> # Bulk calculation (backtesting)
    >>> all_features = calculator.calculate_bulk(context, dates)
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from .base import IFeatureCalculator, FeatureDataContext


class CVGCalculator:
    """
    Calculate CVG (Continuous Volatility Gap) features from historical vol gaps.
    
    Implements IFeatureCalculator protocol following the paper definition from
    Gan & Nguyen ("Mind the Gap"):
    
        DVG = sign(cgap) * (%neg - %pos)   [cgap > 0]
            = sign(cgap) * (%pos - %neg)   [cgap < 0]
            = 0                             [cgap == 0]  <- Fix #2
        CVG = 1 - DVG
    
    cgap is the cumulative RAW vol_gap over the window, cross-sectionally
    adjusted by subtracting the median cgap across all tickers on that date.
    (Fix #1: NOT the sum of per-date median-adjusted gaps.)
    %pos/%neg are derived from per-date median-adjusted vol_gap values.
    
    Attributes:
        windows: List of (max_lag, min_lag) tuples defining lookback periods.
            Example: [(12, 2)] means use data from t-12 to t-2 weeks.
        min_periods: Minimum observations required for valid calculation.
        vol_gap_col: Column name for raw volatility gap (RV - IV).
    
    Example:
        >>> calculator = CVGCalculator(
        ...     windows=[(12, 2), (8, 1)],
        ...     min_periods=2
        ... )
        >>> calculator.feature_names
        ['cvg_12_2', 'dvg_12_2', 'cgap_12_2', 'pct_pos_12_2',
         'pct_neg_12_2', 'volgap_mean_12_2', 'cvg_count_12_2',
         'cvg_8_1', 'dvg_8_1', ...]
    """
    
    def __init__(
        self,
        windows: List[Tuple[int, int]] = None,
        min_periods: int = 1,
        vol_gap_col: str = 'vol_gap'
    ):
        """
        Initialize CVG calculator.
        
        Args:
            windows: List of (max_lag, min_lag) tuples. Default [(12, 2)].
                Each tuple defines a lookback window in weeks:
                - max_lag: How far back to start (e.g., 12 weeks ago)
                - min_lag: How far back to end (e.g., 2 weeks ago)
                Using min_lag > 0 avoids look-ahead bias.
            
            min_periods: Minimum observations needed for valid calculation.
                Default 1 (need at least 1 to calculate DVG/CVG).
                
            vol_gap_col: Column name for volatility gap (RV - IV).
                Default 'vol_gap'. If missing, will try to compute from
                'realized_volatility' - 'entry_iv'.
                
        Example:
            >>> # Standard CVG: t-12 to t-2 weeks
            >>> calc = CVGCalculator(windows=[(12, 2)])
            >>> 
            >>> # Multiple windows for different timeframes
            >>> calc = CVGCalculator(windows=[(12, 2), (8, 1), (20, 4)])
        """
        self.windows = windows if windows is not None else [(12, 2)]
        self.min_periods = min_periods
        self.vol_gap_col = vol_gap_col
        
        # Validate windows
        for max_lag, min_lag in self.windows:
            if max_lag <= min_lag:
                raise ValueError(
                    f"max_lag ({max_lag}) must be > min_lag ({min_lag})"
                )
            if min_lag < 0:
                raise ValueError(f"min_lag ({min_lag}) must be >= 0")
    
    @property
    def feature_names(self) -> List[str]:
        """
        List of feature column names produced by this calculator.
        
        Returns:
            List of feature names in format: cvg_{max_lag}_{min_lag}, etc.
            
        Example:
            >>> calculator = CVGCalculator(windows=[(12, 2)])
            >>> calculator.feature_names
            ['cvg_12_2', 'dvg_12_2', 'cgap_12_2', 'pct_pos_12_2',
             'pct_neg_12_2', 'volgap_mean_12_2', 'volgap_std_12_2', 'cvg_count_12_2']
        """
        names = []
        for max_lag, min_lag in self.windows:
            suffix = f'{max_lag}_{min_lag}'
            names.extend([
                f'cvg_{suffix}',          # Continuous Volatility Gap (primary)
                f'dvg_{suffix}',          # Discontinuous Volatility Gap
                f'cgap_{suffix}',         # Cumulative adjusted gap
                f'pct_pos_{suffix}',      # % positive gaps
                f'pct_neg_{suffix}',      # % negative gaps
                f'volgap_mean_{suffix}',  # Mean adjusted gap
                #f'volgap_std_{suffix}',   # Std of adjusted gap
                f'cvg_count_{suffix}'     # Number of observations
            ])
        return names
    
    @property
    def required_data_sources(self) -> List[str]:
        """
        List of required data sources from FeatureDataContext.
        
        Returns:
            ['straddle_history'] - requires historical straddle data with vol gaps
        """
        return ['straddle_history']

    def _resolve_vol_gap_col(self, history: pd.DataFrame) -> str:
        """
        Resolve the vol_gap column name, computing vol_gap = RV - IV if needed.

        Modifies history in-place when computing from component columns.
        Raises ValueError if neither source is available.
        """
        if self.vol_gap_col in history.columns:
            return self.vol_gap_col
        if 'realized_volatility' in history.columns and 'entry_iv' in history.columns:
            history['vol_gap'] = history['realized_volatility'] - history['entry_iv']
            return 'vol_gap'
        raise ValueError(
            f"Need either '{self.vol_gap_col}' column or "
            "'realized_volatility' and 'entry_iv' columns"
        )

    def calculate(
        self,
        context: FeatureDataContext,
        date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Calculate CVG features for specified tickers at a single date.

        Two-stage pipeline (paper definition):

        Stage 1 — Per-date median adjustment (removes market-wide shifts):
            vol_gap_adjusted = vol_gap - median(vol_gap per date)
            Used for: %pos, %neg, volgap_mean

        Stage 2 — Cross-sectional cgap adjustment (Fix #1):
            raw_cgap = sum(RAW vol_gap) over window  [NOT sum of adjusted gaps]
            cgap = raw_cgap - median(raw_cgap across all tickers)
            Used for: DVG sign determination

        DVG sign rules (Fix #2):
            cgap > 0  -> DVG = %neg - %pos
            cgap < 0  -> DVG = %pos - %neg
            cgap == 0 -> DVG = 0, CVG = 1  (sign(0) = 0)

        Args:
            context: Data context with 'straddle_history' DataFrame.
            date: Target date; features use history up to and including this date.
            tickers: List of ticker symbols.

        Returns:
            DataFrame with one row per ticker and CVG feature columns.
        """
        tickers = [t.upper() for t in tickers]
        history = context.get('straddle_history').copy()

        if not pd.api.types.is_datetime64_any_dtype(history['entry_date']):
            history['entry_date'] = pd.to_datetime(history['entry_date'])

        vol_gap_col = self._resolve_vol_gap_col(history)

        # Filter to data up to and including target date
        history = history[history['entry_date'] <= date].copy()

        if len(history) == 0:
            return pd.DataFrame([
                {'ticker': ticker, 'date': date, **{fn: np.nan for fn in self.feature_names}}
                for ticker in tickers
            ])

        # Stage 1: Per-date cross-sectional median adjustment
        # Used for %pos / %neg / volgap_mean (not for cgap)
        history['vol_gap_adjusted'] = history.groupby('entry_date')[vol_gap_col].transform(
            lambda x: x - x.median()
        )

        history = history.sort_values(['ticker', 'entry_date'])

        # --- Pass 1: collect window data and raw_cgap for all tickers ---
        # raw_cgap uses RAW vol_gap (Fix #1 — NOT sum of adjusted gaps)
        ticker_info = {}
        for ticker in tickers:
            ticker_data = history[history['ticker'] == ticker].copy()
            ticker_info[ticker] = {'data': ticker_data, 'windows': {}}

            if len(ticker_data) == 0:
                continue

            target_rows = ticker_data[ticker_data['entry_date'] == date]
            if len(target_rows) == 0:
                continue

            target_position = ticker_data.index.get_loc(target_rows.index[0])

            for max_lag, min_lag in self.windows:
                suffix = f'{max_lag}_{min_lag}'
                start_idx = max(0, target_position - max_lag)
                end_idx = target_position - min_lag

                if end_idx < start_idx:
                    ticker_info[ticker]['windows'][suffix] = None  # collapsed
                    continue

                window_data = ticker_data.iloc[start_idx:end_idx + 1]
                raw_cgap = window_data[vol_gap_col].sum(min_count=1)  # NaN when all values NaN
                ticker_info[ticker]['windows'][suffix] = {
                    'window_data': window_data,
                    'raw_cgap': raw_cgap
                }

        # --- Stage 2: cross-sectional median of raw_cgap per window (Fix #1) ---
        cgap_medians = {}
        for max_lag, min_lag in self.windows:
            suffix = f'{max_lag}_{min_lag}'
            raw_cgaps = [
                info['windows'][suffix]['raw_cgap']
                for info in ticker_info.values()
                if (info['windows'].get(suffix) is not None and
                    not pd.isna(info['windows'][suffix]['raw_cgap']))
            ]
            cgap_medians[suffix] = float(np.median(raw_cgaps)) if raw_cgaps else np.nan

        # --- Pass 2: build result rows with adjusted cgap ---
        results = []
        for ticker in tickers:
            row = {'ticker': ticker, 'date': date}
            info = ticker_info[ticker]

            # No data or target date missing
            if len(info['data']) == 0 or not info['data']['entry_date'].eq(date).any():
                row.update({fn: np.nan for fn in self.feature_names})
                results.append(row)
                continue

            for max_lag, min_lag in self.windows:
                suffix = f'{max_lag}_{min_lag}'
                w = info['windows'].get(suffix)

                if w is None:  # collapsed window
                    row.update({
                        f'cvg_{suffix}': np.nan,
                        f'dvg_{suffix}': np.nan,
                        f'cgap_{suffix}': np.nan,
                        f'pct_pos_{suffix}': np.nan,
                        f'pct_neg_{suffix}': np.nan,
                        f'volgap_mean_{suffix}': np.nan,
                        f'cvg_count_{suffix}': 0
                    })
                    continue

                # Fix #1: cgap = raw_cgap minus cross-sectional median
                median = cgap_medians.get(suffix, np.nan)
                adjusted_cgap = w['raw_cgap'] - (median if not np.isnan(median) else 0.0)

                features = self._calculate_window_features(
                    w['window_data'], suffix, adjusted_cgap
                )
                row.update(features)

            results.append(row)

        return pd.DataFrame(results)
    
    def _calculate_window_features(
        self,
        window_data: pd.DataFrame,
        suffix: str,
        adjusted_cgap: float
    ) -> dict:
        """
        Calculate CVG features for a single window.

        Args:
            window_data: DataFrame slice with 'vol_gap_adjusted' column
                (per-date median-adjusted vol gaps — used for %pos/%neg/mean).
            suffix: Feature name suffix (e.g., '12_2').
            adjusted_cgap: Pre-computed cross-sectionally-adjusted cumulative gap
                (raw_cgap minus cross-sectional median). Implements Fix #1:
                paper definition of cgap, NOT sum of per-date-adjusted gaps.

        Returns:
            Dict of features. NaN if count < min_periods.

        DVG / CVG rules (Fix #2):
            cgap > 0  -> DVG = %neg - %pos
            cgap < 0  -> DVG = %pos - %neg
            cgap == 0 -> DVG = 0, CVG = 1  (previously fell into negative branch)
        """
        # %pos/%neg/mean from per-date-adjusted gaps (Stage 1)
        adjusted_gaps = window_data['vol_gap_adjusted'].dropna().values
        count = len(adjusted_gaps)

        features = {
            f'cvg_{suffix}': np.nan,
            f'dvg_{suffix}': np.nan,
            f'cgap_{suffix}': np.nan,
            f'pct_pos_{suffix}': np.nan,
            f'pct_neg_{suffix}': np.nan,
            f'volgap_mean_{suffix}': np.nan,
            f'cvg_count_{suffix}': count
        }

        if count < self.min_periods:
            return features

        pos_count = np.sum(adjusted_gaps >= 0)
        neg_count = np.sum(adjusted_gaps < 0)
        pct_pos = pos_count / count
        pct_neg = neg_count / count
        mean_gap = np.mean(adjusted_gaps)

        # DVG using sign(adjusted_cgap) — Fix #2: explicit cgap == 0 branch
        if adjusted_cgap > 0:
            dvg = pct_neg - pct_pos
        elif adjusted_cgap < 0:
            dvg = pct_pos - pct_neg
        else:
            dvg = 0.0  # sign(0) = 0 -> DVG = 0 -> CVG = 1

        cvg = 1.0 - dvg

        features.update({
            f'cvg_{suffix}': cvg,
            f'dvg_{suffix}': dvg,
            f'cgap_{suffix}': adjusted_cgap,
            f'pct_pos_{suffix}': pct_pos,
            f'pct_neg_{suffix}': pct_neg,
            f'volgap_mean_{suffix}': mean_gap,
        })

        return features
    
    def calculate_bulk(
        self,
        context: FeatureDataContext,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate CVG features for a date range efficiently (vectorized).

        Two-stage pipeline (paper definition):

        Stage 1 — Per-date median adjustment (applied globally before windowing):
            vol_gap_adjusted = vol_gap - median(vol_gap) per date
            Used for: %pos, %neg, volgap_mean (rolling on adjusted column)

        Stage 2 — cgap cross-sectional adjustment per date (Fix #1):
            cgap_raw = rolling sum of RAW vol_gap per ticker (NOT adjusted)
            cgap = cgap_raw - median(cgap_raw across tickers) per date

        Fix #2 — DVG when cgap == 0: explicitly yields DVG = 0, CVG = 1.

        Fix #3 — Groupwise shift: uses .groupby(level=0).shift(min_lag) after
            groupby().rolling() to prevent cross-ticker value leakage across
            the (ticker, position) MultiIndex boundary.

        Args:
            context: Data context with 'straddle_history' DataFrame.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            tickers: Optional ticker filter. None = all tickers.

        Returns:
            DataFrame with ticker, date, and CVG feature columns.
        """
        history = context.get('straddle_history').copy()

        if not pd.api.types.is_datetime64_any_dtype(history['entry_date']):
            history['entry_date'] = pd.to_datetime(history['entry_date'])

        if tickers is not None:
            tickers = [t.upper() for t in tickers]
            history = history[history['ticker'].isin(tickers)].copy()

        vol_gap_col = self._resolve_vol_gap_col(history)

        history = history.sort_values(['ticker', 'entry_date'])

        # Stage 1: Per-date cross-sectional median adjustment of raw vol_gap
        # Used for %pos / %neg / volgap_mean rolling calculations
        history['vol_gap_adjusted'] = history.groupby('entry_date')[vol_gap_col].transform(
            lambda x: x - x.median()
        )

        for max_lag, min_lag in self.windows:
            window_size = max_lag - min_lag + 1
            suffix = f'{max_lag}_{min_lag}'

            grouped_adjusted = history.groupby('ticker')['vol_gap_adjusted']
            grouped_raw = history.groupby('ticker')[vol_gap_col]

            # Count (NaN-aware, from adjusted column)
            history[f'cvg_count_{suffix}'] = (
                grouped_adjusted
                .rolling(window=window_size, min_periods=1)
                .count()
                .groupby(level=0).shift(min_lag)           # Fix #3
                .reset_index(level=0, drop=True)
            )

            # Mean adjusted gap
            history[f'volgap_mean_{suffix}'] = (
                grouped_adjusted
                .rolling(window=window_size, min_periods=1)
                .mean()
                .groupby(level=0).shift(min_lag)           # Fix #3
                .reset_index(level=0, drop=True)
            )

            # Count positive adjusted gaps
            history[f'_pos_count_{suffix}'] = (
                grouped_adjusted
                .rolling(window=window_size, min_periods=1)
                .apply(lambda x: (x >= 0).sum(), raw=True)
                .groupby(level=0).shift(min_lag)           # Fix #3
                .reset_index(level=0, drop=True)
            )

            # Count negative adjusted gaps
            history[f'_neg_count_{suffix}'] = (
                grouped_adjusted
                .rolling(window=window_size, min_periods=1)
                .apply(lambda x: (x < 0).sum(), raw=True)
                .groupby(level=0).shift(min_lag)           # Fix #3
                .reset_index(level=0, drop=True)
            )

            history[f'pct_pos_{suffix}'] = np.where(
                history[f'cvg_count_{suffix}'] > 0,
                history[f'_pos_count_{suffix}'] / history[f'cvg_count_{suffix}'],
                np.nan
            )
            history[f'pct_neg_{suffix}'] = np.where(
                history[f'cvg_count_{suffix}'] > 0,
                history[f'_neg_count_{suffix}'] / history[f'cvg_count_{suffix}'],
                np.nan
            )

            # Fix #1: cgap = rolling sum of RAW vol_gap, then cross-sectionally adjusted
            # Step A: per-ticker rolling sum of raw vol_gap
            history[f'_cgap_raw_{suffix}'] = (
                grouped_raw
                .rolling(window=window_size, min_periods=1)
                .sum()
                .groupby(level=0).shift(min_lag)           # Fix #3
                .reset_index(level=0, drop=True)
            )
            # Step B: cross-sectional median subtraction per date
            history[f'cgap_{suffix}'] = history.groupby('entry_date')[f'_cgap_raw_{suffix}'].transform(
                lambda x: x - x.median()
            )

            # Fix #2: DVG with explicit cgap == 0 branch -> DVG = 0 -> CVG = 1
            count_ok = history[f'cvg_count_{suffix}'] >= self.min_periods
            history[f'dvg_{suffix}'] = np.where(
                count_ok,
                np.where(
                    history[f'cgap_{suffix}'] > 0,
                    history[f'pct_neg_{suffix}'] - history[f'pct_pos_{suffix}'],
                    np.where(
                        history[f'cgap_{suffix}'] < 0,
                        history[f'pct_pos_{suffix}'] - history[f'pct_neg_{suffix}'],
                        0.0     # Fix #2: cgap == 0 -> DVG = 0
                    )
                ),
                np.nan
            )
            history[f'cvg_{suffix}'] = np.where(
                history[f'dvg_{suffix}'].notna(),
                1.0 - history[f'dvg_{suffix}'],
                np.nan
            )

            # Set all features to NaN where count < min_periods
            mask = history[f'cvg_count_{suffix}'] < self.min_periods
            for feat in [f'cvg_{suffix}', f'dvg_{suffix}', f'cgap_{suffix}',
                         f'pct_pos_{suffix}', f'pct_neg_{suffix}', f'volgap_mean_{suffix}']:
                history.loc[mask, feat] = np.nan

            # Drop intermediate columns
            history = history.drop(columns=[
                f'_pos_count_{suffix}',
                f'_neg_count_{suffix}',
                f'_cgap_raw_{suffix}'
            ])

        # Drop Stage 1 helper column
        history = history.drop(columns=['vol_gap_adjusted'])

        # Filter to target date range (inclusive)
        result = history[
            (history['entry_date'] >= start_date) &
            (history['entry_date'] <= end_date)
        ].copy()

        # Select and rename output columns
        output_cols = ['ticker', 'entry_date'] + self.feature_names
        result = result[output_cols].copy()
        result = result.rename(columns={'entry_date': 'date'})

        return result
