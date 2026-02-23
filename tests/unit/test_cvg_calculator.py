"""
Unit tests for CVGCalculator.

Tests cover:
- Initialization and configuration validation
- Cross-sectional median adjustment (paper definition, Gan & Nguyen)
- DVG/CVG sign rules for all three cgap branches (>, <, ==0)
- Fix #1: cgap = raw_cgap - cross-sectional median (NOT sum of adjusted gaps)
- Fix #2: sign(0) = 0 → DVG = 0 → CVG = 1 (old code fell into negative branch)
- Fix #3: groupwise shift in calculate_bulk() prevents cross-ticker leakage
- Boundary conditions (early dates, collapsed windows, NaN vol_gaps)
- Consistency between calculate() and calculate_bulk()
- Universe dependency: cgap cross-sectional median is relative to ticker set passed in

=============================================================================
Fixture files required
=============================================================================

tests/fixtures/sample_vol_gap_history.csv
    Columns : ticker (str), entry_date (YYYY-MM-DD), vol_gap (float, NaN allowed)
    Tickers : AAPL, TSLA, ADP, UBER
    Dates   : 52 weekly trading dates  2019-01-04 → 2019-12-27  (208 rows total)
              (real ORATS-derived data; vol_gap values are floating-point, not
              round integers — fixture-based tests compute expected values
              directly from the DataFrame rather than hard-coding them)

    Selected position index (0-based, per ticker — key dates only):
        0  = 2019-01-04   1  = 2019-01-11   2  = 2019-01-18   3  = 2019-01-25
        4  = 2019-02-01  10  = 2019-03-15  21  = 2019-05-31  26  = 2019-07-05
       27  = 2019-07-12  33  = 2019-08-23  35  = 2019-09-06  51  = 2019-12-27

    Known NaN patterns (incomplete trades / pre-IPO):
      - UBER: NaN on positions 0–20 (2019-01-04 through 2019-05-24).
              First non-NaN: position 21 (2019-05-31).
              UBER IPO was May 10 2019; first settlement ~4 weeks later.
      - ADP:  Scattered NaN values throughout (e.g. positions 1, 2, 6, 7,
              14, 20, 21, 30, 31, 34, …). Reflect real missing straddle data.

    PRIMARY_DATE design (position 35 = 2019-09-06):
      Window (8,2) at p=35 covers positions [27..33]  (Jul 12 – Aug 23).
      Counts within that window:
          AAPL = 7  (all non-NaN)   TSLA = 7  (all non-NaN)
          ADP  = 5  (2 NaN: Aug 2, Aug 9)   UBER = 7  (all non-NaN)
      All four tickers satisfy count ≥ min_periods=3 → non-NaN cgap.

    Note on Fix #1 / Fix #2 regression tests:
      These use inline DataFrames with controlled round numbers rather than
      this fixture, because the fixture values were not designed to produce
      specific rank-reversal or zero-cgap scenarios.

tests/fixtures/sample_vol_gap_history_rv_iv.csv
    Same rows as above but with columns:
        ticker, entry_date, return_pct, realized_volatility, entry_iv
    No 'vol_gap' column.
    Constraint: realized_volatility − entry_iv == vol_gap from the main fixture.
    The extra return_pct column is benign — CVGCalculator ignores it.
    Used by: test_vol_gap_resolved_from_components.

=============================================================================
Standard calculator configuration used throughout
=============================================================================

    CVGCalculator(windows=[(8, 2)], min_periods=3)

    window_size = 8 − 2 + 1 = 7    (7 rows per full window)
    shift       = 2                 (skip 2 most recent rows to avoid look-ahead)

    Row-based lookback at position p:
        start = max(0, p − 8)
        end   = p − 2
        rows  = [start .. end]  inclusive

    Key positions in the 52-date fixture:
        p=1  → end=−1   < start=0    → collapsed (count=0)      [2019-01-11]
        p=2  → rows [0..0] = 1 row                              [2019-01-18]
        p=10 → rows [2..8] = 7 rows  ← AAPL first full window   [2019-03-15]
        p=35 → rows [27..33] = 7 rows ← PRIMARY_DATE            [2019-09-06]
               all 4 tickers count ≥ 3 (UBER and ADP both have data here)

"""

import pytest
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.features.cvg_calculator import CVGCalculator
from src.features.base import FeatureDataContext


# ============================================================================
# Hand-computable reference values
# (Fill in the TODO constants once the fixture CSV is created.)
# ============================================================================

# Primary test date: position 35 (2019-09-06) in the 52-date fixture.
# Window (8,2) covers positions [27..33] = Jul 12 – Aug 23 2019.
# All 4 tickers have count ≥ min_periods=3 at this date:
#   AAPL=7, TSLA=7, UBER=7, ADP=5 (Aug 2 and Aug 9 are NaN for ADP).
PRIMARY_DATE = pd.Timestamp('2019-09-06')   # position 35

# Fix #2 test uses an inline DataFrame (controlled values), not the fixture,
# because the fixture was not designed to produce TSLA adjusted_cgap == 0.
# This constant is kept as a named reference for any future fixture-based test.
FIX2_DATE = pd.Timestamp('2019-09-06')      # same as PRIMARY_DATE; adjust if needed

# Expected values at PRIMARY_DATE — computed directly from the fixture DataFrame
# inside the tests (no hard-coded constants here, since the fixture uses real
# floating-point values that would require re-computing if the fixture changes).


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_vol_gap_history():
    """
    Load 4-ticker, 12-date vol_gap history from CSV.

    See module docstring for full fixture design specification.
    Columns: ticker, entry_date (datetime), vol_gap (float, NaN allowed).
    """
    fixtures_dir = Path(__file__).parent.parent / 'fixtures'
    df = pd.read_csv(fixtures_dir / 'sample_vol_gap_history.csv')
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    return df


@pytest.fixture
def sample_vol_gap_history_rv_iv():
    """
    Load vol_gap history expressed as realized_volatility and entry_iv components.

    Columns: ticker, entry_date, realized_volatility, entry_iv.
    No 'vol_gap' column — CVGCalculator must derive it.
    Constraint: realized_volatility − entry_iv == vol_gap from main fixture.
    """
    fixtures_dir = Path(__file__).parent.parent / 'fixtures'
    df = pd.read_csv(fixtures_dir / 'sample_vol_gap_history_rv_iv.csv')
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    return df


@pytest.fixture
def vol_gap_context(sample_vol_gap_history):
    """FeatureDataContext wrapping the vol_gap fixture."""
    return FeatureDataContext(straddle_history=sample_vol_gap_history)


@pytest.fixture
def rv_iv_context(sample_vol_gap_history_rv_iv):
    """FeatureDataContext wrapping the RV/IV component fixture (no vol_gap column)."""
    return FeatureDataContext(straddle_history=sample_vol_gap_history_rv_iv)


@pytest.fixture
def cvg_calculator():
    """
    Standard CVGCalculator: window (8, 2), min_periods=3.

    Full 7-row window available from position 10 onward in the 12-date fixture.
    """
    return CVGCalculator(windows=[(8, 2)], min_periods=3)


@pytest.fixture
def cvg_calculator_no_min():
    """CVGCalculator with min_periods=1 — allows single-observation windows."""
    return CVGCalculator(windows=[(8, 2)], min_periods=1)


@pytest.fixture
def cvg_calculator_multi_window():
    """
    CVGCalculator with two windows: (8, 2) and (6, 1).

    Feature count: 7 stats × 2 windows = 14 features.
    """
    return CVGCalculator(windows=[(8, 2), (6, 1)], min_periods=2)


# ============================================================================
# Test Class: Initialization & Configuration
# ============================================================================

class TestCVGCalculatorInit:
    """Test CVGCalculator initialization and configuration."""

    def test_init_default_parameters(self):
        """
        Test initialization with default parameters.

        Verifies:
        - Default window is [(12, 2)]
        - Default min_periods is 1
        - Default vol_gap_col is 'vol_gap'
        - Feature names generated correctly (7 stats × 1 window = 7 features)
        """
        calc = CVGCalculator()

        assert calc.windows == [(12, 2)]
        assert calc.min_periods == 1
        assert calc.vol_gap_col == 'vol_gap'

        names = calc.feature_names
        assert len(names) == 7
        # All names must be strings and non-empty
        assert all(isinstance(n, str) and n for n in names)

    def test_init_custom_windows(self):
        """
        Test initialization with multiple custom windows.

        Given: windows=[(8, 2), (6, 1), (12, 4)], min_periods=3
        Verifies:
        - Windows stored correctly
        - 21 feature names (7 stats × 3 windows)
        - All expected prefixes present: cvg_8_2, cvg_6_1, cvg_12_4
        - All 7 stats present per window:
            cvg, dvg, cgap, pct_pos, pct_neg, volgap_mean, cvg_count
        """
        calc = CVGCalculator(windows=[(8, 2), (6, 1), (12, 4)], min_periods=3)

        assert calc.windows == [(8, 2), (6, 1), (12, 4)]
        assert calc.min_periods == 3

        names = calc.feature_names
        assert len(names) == 21  # 7 stats × 3 windows

        stats = ['cvg', 'dvg', 'cgap', 'pct_pos', 'pct_neg', 'volgap_mean', 'cvg_count']
        for max_lag, min_lag in [(8, 2), (6, 1), (12, 4)]:
            suffix = f'{max_lag}_{min_lag}'
            for stat in stats:
                assert f'{stat}_{suffix}' in names, (
                    f"Expected '{stat}_{suffix}' in feature_names"
                )

    def test_init_invalid_window_max_equal_to_min(self):
        """
        Test that max_lag == min_lag raises ValueError.

        Window (4, 4) has zero width — no lookback possible.
        Expected: ValueError with message mentioning 'max_lag'.
        """
        with pytest.raises(ValueError, match='max_lag'):
            CVGCalculator(windows=[(4, 4)])

    def test_init_invalid_window_max_less_than_min(self):
        """
        Test that max_lag < min_lag raises ValueError (inverted window).

        Window (2, 5) is inverted.
        Expected: ValueError with message mentioning 'max_lag'.
        """
        with pytest.raises(ValueError, match='max_lag'):
            CVGCalculator(windows=[(2, 5)])

    def test_init_invalid_window_negative_min_lag(self):
        """
        Test that negative min_lag raises ValueError.

        Window (8, -1) would introduce look-ahead bias.
        Expected: ValueError with message mentioning 'min_lag'.
        """
        with pytest.raises(ValueError, match='min_lag'):
            CVGCalculator(windows=[(8, -1)])

    def test_feature_names_order_consistent(self):
        """
        Test feature_names returns names in the correct per-window order.

        Given: windows=[(8, 2), (6, 1)]
        Expected order within each window: cvg, dvg, cgap, pct_pos, pct_neg,
            volgap_mean, cvg_count
        Expected inter-window order: all (8,2) features before all (6,1) features.
        """
        calc = CVGCalculator(windows=[(8, 2), (6, 1)])

        expected = [
            'cvg_8_2', 'dvg_8_2', 'cgap_8_2',
            'pct_pos_8_2', 'pct_neg_8_2', 'volgap_mean_8_2', 'cvg_count_8_2',
            'cvg_6_1', 'dvg_6_1', 'cgap_6_1',
            'pct_pos_6_1', 'pct_neg_6_1', 'volgap_mean_6_1', 'cvg_count_6_1',
        ]
        assert calc.feature_names == expected

    def test_required_data_sources(self):
        """
        Test required_data_sources property.

        Expected: ['straddle_history']
        """
        calc = CVGCalculator()
        assert calc.required_data_sources == ['straddle_history']


# ============================================================================
# Test Class: Cross-Sectional Median Adjustment (paper definition)
# ============================================================================

class TestCrossMedianAdjustment:
    """
    Test the two-stage cross-sectional median adjustment that is the
    mathematical core distinguishing CVGCalculator from MomentumCalculator.

    Stage 1: vol_gap_adjusted = vol_gap - median(vol_gap per date)
             Removes market-wide vol-regime shifts.
             Used for: %pos, %neg, volgap_mean.

    Stage 2: cgap = sum(RAW vol_gap over window) - median(sum(RAW) across tickers)
             Cross-sectionally centers the cumulative gap.
             Used for: DVG sign determination.
             Fix #1: this is NOT sum(vol_gap_adjusted over window).
    """

    def test_stage1_per_date_median_subtracted_from_pct_signals(self):
        """
        Verify that Stage 1 (per-date median) is applied to %pos/%neg/volgap_mean.

        Inline setup — 3 tickers, 3 identical dates, known vol_gap values:
            date d1: AAPL=2, TSLA=6, ADP=10  → per-date median = 6
            Adjusted: AAPL=−4, TSLA=0, ADP=4

        With window (4, 0) and min_periods=1 at date d3 (first window date is d1):
        - AAPL: all adjusted gaps negative → pct_pos=0, pct_neg=1
        - ADP:  all adjusted gaps positive → pct_pos=1, pct_neg=0
        - TSLA: all adjusted gaps zero → pct_pos=1, pct_neg=0
          (zero counts as non-negative: x >= 0)

        Expected: assert these pct values match the above analysis.
        """
        # Inline data: 3 tickers × 3 dates, same vol_gap per date so the
        # per-date median shifts are identical on every row.
        d1 = pd.Timestamp('2020-01-03')
        d2 = pd.Timestamp('2020-01-10')
        d3 = pd.Timestamp('2020-01-17')

        rows = []
        for dt in [d1, d2, d3]:
            rows += [
                {'ticker': 'AAPL', 'entry_date': dt, 'vol_gap': 2.0},
                {'ticker': 'TSLA', 'entry_date': dt, 'vol_gap': 6.0},
                {'ticker': 'ADP',  'entry_date': dt, 'vol_gap': 10.0},
            ]
        history = pd.DataFrame(rows)
        ctx = FeatureDataContext(straddle_history=history)

        # window (4, 0): at d3, lookback covers [d1..d3] (3 rows per ticker)
        calc = CVGCalculator(windows=[(4, 0)], min_periods=1)
        result = calc.calculate(ctx, d3, ['AAPL', 'TSLA', 'ADP'])
        result = result.set_index('ticker')

        # AAPL: adjusted gaps are all −4 (negative) → pct_neg=1, pct_pos=0
        assert result.loc['AAPL', 'pct_pos_4_0'] == pytest.approx(0.0)
        assert result.loc['AAPL', 'pct_neg_4_0'] == pytest.approx(1.0)

        # ADP: adjusted gaps are all +4 (positive) → pct_pos=1, pct_neg=0
        assert result.loc['ADP', 'pct_pos_4_0'] == pytest.approx(1.0)
        assert result.loc['ADP', 'pct_neg_4_0'] == pytest.approx(0.0)

        # TSLA: adjusted gaps are all 0 (zero counts as non-negative) → pct_pos=1
        assert result.loc['TSLA', 'pct_pos_4_0'] == pytest.approx(1.0)
        assert result.loc['TSLA', 'pct_neg_4_0'] == pytest.approx(0.0)

        # volgap_mean should also reflect the Stage 1 adjusted values
        assert result.loc['AAPL', 'volgap_mean_4_0'] == pytest.approx(-4.0)
        assert result.loc['ADP',  'volgap_mean_4_0'] == pytest.approx(4.0)
        assert result.loc['TSLA', 'volgap_mean_4_0'] == pytest.approx(0.0)

    def test_fix1_cgap_differs_from_sum_of_adjusted_gaps(self):
        """
        Regression test for Fix #1: paper cgap ≠ sum(per-date-adjusted gaps).

        Inline setup designed to maximise divergence (rank reversal):

            date d1: AAPL=10, TSLA=2, ADP=4  → per-date median=4
                     adjusted: AAPL=6, TSLA=−2, ADP=0
            date d2: AAPL=2,  TSLA=10, ADP=4  → per-date median=4
                     adjusted: AAPL=−2, TSLA=6, ADP=0

        Old method (incorrect):
            cgap(AAPL) = sum(adjusted) = 6 + (−2) = 4  [positive]

        New method (paper):
            raw_sums:  AAPL=12, TSLA=12, ADP=8
            median(raw_sums) = 12
            cgap(AAPL) = 12 − 12 = 0  [zero → DVG=0, CVG=1 via Fix #2]

        The two methods produce different signs for AAPL's cgap, which would
        flip the DVG calculation entirely.

        Setup: build context with these 2 dates × 3 tickers.
        Use window (2, 0), min_periods=1, calculate() at date d2.

        Assert:
        - aapl_row['cgap_2_0'] == pytest.approx(0.0)                 [paper]
        - aapl_row['cgap_2_0'] != pytest.approx(4.0)                 [old, wrong]
        - aapl_row['cvg_2_0']  == pytest.approx(1.0)                 [DVG=0 → CVG=1]
        """
        d1 = pd.Timestamp('2020-01-03')
        d2 = pd.Timestamp('2020-01-10')

        history = pd.DataFrame([
            {'ticker': 'AAPL', 'entry_date': d1, 'vol_gap': 10.0},
            {'ticker': 'TSLA', 'entry_date': d1, 'vol_gap':  2.0},
            {'ticker': 'ADP',  'entry_date': d1, 'vol_gap':  4.0},
            {'ticker': 'AAPL', 'entry_date': d2, 'vol_gap':  2.0},
            {'ticker': 'TSLA', 'entry_date': d2, 'vol_gap': 10.0},
            {'ticker': 'ADP',  'entry_date': d2, 'vol_gap':  4.0},
        ])
        ctx = FeatureDataContext(straddle_history=history)

        # window (2, 0): at d2, lookback covers [d1, d2] = 2 rows per ticker
        # raw_sums: AAPL=12, TSLA=12, ADP=8 → median = 12
        # paper cgap(AAPL) = 12 − 12 = 0
        # old cgap(AAPL)   = sum(adjusted) = (10−4)+(2−4) = 4  [wrong]
        calc = CVGCalculator(windows=[(2, 0)], min_periods=1)
        result = calc.calculate(ctx, d2, ['AAPL', 'TSLA', 'ADP'])
        aapl = result[result['ticker'] == 'AAPL'].iloc[0]

        # Paper-correct cgap: 0.0
        assert aapl['cgap_2_0'] == pytest.approx(0.0), (
            f"Expected cgap=0.0 (paper), got {aapl['cgap_2_0']}. "
            "Non-zero result indicates old sum-of-adjusted approach is in use."
        )
        # Verify it is NOT the old wrong value
        assert aapl['cgap_2_0'] != pytest.approx(4.0)

        # cgap == 0 → DVG = 0 (Fix #2) → CVG = 1
        assert aapl['cvg_2_0'] == pytest.approx(1.0)
        assert aapl['dvg_2_0'] == pytest.approx(0.0)

    def test_cgap_cross_sectional_median_is_zero_across_tickers(self,
                                                                   cvg_calculator,
                                                                   vol_gap_context,
                                                                   sample_vol_gap_history):
        """
        Verify that after Stage 2 adjustment, median(cgap) across tickers ≈ 0.

        This is a mathematical consequence of subtracting the cross-sectional
        median: the median of (x - median(x)) is 0 (or very close for even N).

        Setup: call calculate_bulk() for all 4 tickers at PRIMARY_DATE.
        At that date all 4 tickers should have non-NaN cgap values.

        Assert: np.median(result.loc[result['date']==PRIMARY_DATE, 'cgap_8_2']) ≈ 0.0
        """
        all_tickers = sample_vol_gap_history['ticker'].unique().tolist()

        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=PRIMARY_DATE,
            end_date=PRIMARY_DATE,
            tickers=all_tickers,
        )

        at_primary = result[result['date'] == PRIMARY_DATE]['cgap_8_2'].dropna()

        # All 4 tickers must have valid cgap at PRIMARY_DATE by fixture design
        assert len(at_primary) == 4, (
            f"Expected 4 non-NaN cgap values at PRIMARY_DATE, got {len(at_primary)}"
        )

        # Median of (x - median(x)) must be 0 — exact for odd N, ~0 for even N
        assert np.median(at_primary.values) == pytest.approx(0.0, abs=1e-10)

    def test_vol_gap_resolved_from_components(self, rv_iv_context, vol_gap_context,
                                              cvg_calculator):
        """
        Verify that CVGCalculator accepts realized_volatility + entry_iv in
        place of a pre-computed vol_gap column, and produces identical results.

        Given:
        - rv_iv_context: context with realized_volatility, entry_iv (no vol_gap)
        - vol_gap_context: same data with vol_gap = RV − IV pre-computed

        Both contexts should produce numerically identical output from calculate()
        and calculate_bulk() at PRIMARY_DATE.

        Assert: every feature column matches to rel=1e-6.
        """
        tickers = ['AAPL', 'TSLA', 'ADP', 'UBER']
        feature_cols = cvg_calculator.feature_names

        # --- calculate() comparison ---
        result_vg = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, tickers)
        result_rv = cvg_calculator.calculate(rv_iv_context,  PRIMARY_DATE, tickers)

        result_vg = result_vg.set_index('ticker').sort_index()
        result_rv = result_rv.set_index('ticker').sort_index()

        for ticker in tickers:
            for col in feature_cols:
                vg_val = result_vg.loc[ticker, col]
                rv_val = result_rv.loc[ticker, col]
                if pd.isna(vg_val):
                    assert pd.isna(rv_val), (
                        f"calculate() {ticker}/{col}: vol_gap is NaN but rv_iv is {rv_val}"
                    )
                else:
                    assert vg_val == pytest.approx(rv_val, rel=1e-6), (
                        f"calculate() {ticker}/{col}: {vg_val} vs {rv_val}"
                    )

        # --- calculate_bulk() comparison ---
        bulk_vg = cvg_calculator.calculate_bulk(
            vol_gap_context, PRIMARY_DATE, PRIMARY_DATE, tickers=tickers
        )
        bulk_rv = cvg_calculator.calculate_bulk(
            rv_iv_context,  PRIMARY_DATE, PRIMARY_DATE, tickers=tickers
        )

        bulk_vg = bulk_vg.set_index('ticker').sort_index()
        bulk_rv = bulk_rv.set_index('ticker').sort_index()

        for ticker in tickers:
            for col in feature_cols:
                vg_val = bulk_vg.loc[ticker, col]
                rv_val = bulk_rv.loc[ticker, col]
                if pd.isna(vg_val):
                    assert pd.isna(rv_val), (
                        f"calculate_bulk() {ticker}/{col}: vol_gap NaN but rv_iv {rv_val}"
                    )
                else:
                    assert vg_val == pytest.approx(rv_val, rel=1e-6), (
                        f"calculate_bulk() {ticker}/{col}: {vg_val} vs {rv_val}"
                    )


# ============================================================================
# Test Class: Window Feature Calculation (_calculate_window_features)
# ============================================================================

class TestWindowFeatureCalculation:
    """
    Unit tests for _calculate_window_features(window_data, suffix, adjusted_cgap).

    All tests construct inline DataFrames with a 'vol_gap_adjusted' column
    (Stage 1 pre-adjustment applied manually) and supply adjusted_cgap directly.
    This isolates the DVG/CVG math from the adjustment pipeline.

    Key distinction from MomentumCalculator:
        adjusted_cgap is passed in as a parameter (pre-computed by caller).
        %pos/%neg are computed from 'vol_gap_adjusted' (Stage 1 values).
    """

    def test_positive_cgap_dvg_is_neg_minus_pos(self, cvg_calculator):
        """
        When adjusted_cgap > 0: DVG = pct_neg - pct_pos.

        Given:
        - adjusted_gaps = [3, −1, 2, −2, 4]  → pct_pos=3/5=0.6, pct_neg=2/5=0.4
        - adjusted_cgap = 5.0  (positive)

        Expected:
        - dvg = 0.4 − 0.6 = −0.2
        - cvg = 1 − (−0.2) = 1.2
        - pct_pos = 0.6, pct_neg = 0.4, count = 5
        """
        window_data = pd.DataFrame({'vol_gap_adjusted': [3.0, -1.0, 2.0, -2.0, 4.0]})
        result = cvg_calculator._calculate_window_features(window_data, '8_2', 5.0)

        assert result['pct_pos_8_2'] == pytest.approx(0.6)
        assert result['pct_neg_8_2'] == pytest.approx(0.4)
        assert result['dvg_8_2']     == pytest.approx(-0.2)   # pct_neg - pct_pos
        assert result['cvg_8_2']     == pytest.approx(1.2)    # 1 - (-0.2)
        assert result['cvg_count_8_2'] == 5

    def test_negative_cgap_dvg_is_pos_minus_neg(self, cvg_calculator):
        """
        When adjusted_cgap < 0: DVG = pct_pos - pct_neg.

        Given:
        - adjusted_gaps = [3, −1, 2, −2, 4]  → pct_pos=0.6, pct_neg=0.4
        - adjusted_cgap = −5.0  (negative)

        Expected:
        - dvg = 0.6 − 0.4 = 0.2
        - cvg = 1 − 0.2 = 0.8
        """
        window_data = pd.DataFrame({'vol_gap_adjusted': [3.0, -1.0, 2.0, -2.0, 4.0]})
        result = cvg_calculator._calculate_window_features(window_data, '8_2', -5.0)

        assert result['dvg_8_2'] == pytest.approx(0.2)    # pct_pos - pct_neg
        assert result['cvg_8_2'] == pytest.approx(0.8)    # 1 - 0.2

    def test_zero_cgap_dvg_is_zero_cvg_is_one(self, cvg_calculator):
        """
        Fix #2 regression test: when adjusted_cgap == 0, DVG must be 0 and CVG = 1.

        The old code had `if cgap > 0: ... else: pct_pos - pct_neg`, which
        would produce a non-zero DVG when cgap == 0 (whenever pct_pos ≠ pct_neg).

        Given:
        - adjusted_gaps = [3, −1, 2, −2, 4]  → pct_pos=0.6, pct_neg=0.4
        - adjusted_cgap = 0.0  (zero)

        Expected (Fix #2):
        - dvg = 0.0  exactly  (old code would give 0.6−0.4 = 0.2)
        - cvg = 1.0  exactly
        """
        window_data = pd.DataFrame({'vol_gap_adjusted': [3.0, -1.0, 2.0, -2.0, 4.0]})
        result = cvg_calculator._calculate_window_features(window_data, '8_2', 0.0)

        # pct_pos (0.6) ≠ pct_neg (0.4), so the old code would give dvg=0.2.
        # Fix #2 requires dvg=0.0 when cgap==0 regardless of pct distribution.
        assert result['dvg_8_2'] == pytest.approx(0.0), (
            f"Fix #2 violated: expected dvg=0.0 when cgap=0, got {result['dvg_8_2']}. "
            "Old code fell into the negative branch producing pct_pos - pct_neg = 0.2."
        )
        assert result['cvg_8_2'] == pytest.approx(1.0)

    def test_all_gaps_positive_maximum_continuity(self, cvg_calculator):
        """
        All adjusted gaps positive → pct_pos=1, pct_neg=0.
        With positive cgap → DVG = 0 − 1 = −1 → CVG = 2.0 (maximum possible).

        Given:
        - adjusted_gaps = [1, 2, 3, 4, 5]
        - adjusted_cgap = 15.0  (positive)

        Expected:
        - pct_pos = 1.0, pct_neg = 0.0
        - dvg = −1.0, cvg = 2.0
        """
        window_data = pd.DataFrame({'vol_gap_adjusted': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = cvg_calculator._calculate_window_features(window_data, '8_2', 15.0)

        assert result['pct_pos_8_2'] == pytest.approx(1.0)
        assert result['pct_neg_8_2'] == pytest.approx(0.0)
        assert result['dvg_8_2']     == pytest.approx(-1.0)  # pct_neg - pct_pos = 0 - 1
        assert result['cvg_8_2']     == pytest.approx(2.0)   # 1 - (-1) = 2 (maximum)
        assert result['volgap_mean_8_2'] == pytest.approx(3.0)  # mean([1,2,3,4,5])

    def test_insufficient_data_returns_nan(self, cvg_calculator):
        """
        When count < min_periods, all signal features are NaN; count is reported.

        Given: 2 adjusted gaps, min_periods=3

        Expected:
        - cvg_count = 2
        - cvg, dvg, cgap, pct_pos, pct_neg, volgap_mean all NaN
        """
        # cvg_calculator has min_periods=3; 2 rows is insufficient
        window_data = pd.DataFrame({'vol_gap_adjusted': [1.0, -2.0]})
        result = cvg_calculator._calculate_window_features(window_data, '8_2', 0.5)

        # Count is reported even when insufficient
        assert result['cvg_count_8_2'] == 2

        # All signal features must be NaN
        signal_cols = ['cvg_8_2', 'dvg_8_2', 'cgap_8_2',
                       'pct_pos_8_2', 'pct_neg_8_2', 'volgap_mean_8_2']
        for col in signal_cols:
            assert pd.isna(result[col]), f"Expected {col} to be NaN, got {result[col]}"

    def test_nan_gaps_excluded_from_pct_calculations(self, cvg_calculator):
        """
        NaN values in 'vol_gap_adjusted' are excluded from count and % calculations.

        Given:
        - adjusted_gaps column = [3.0, NaN, −1.0, NaN, 2.0]
        - adjusted_cgap = 4.0

        Expected:
        - count = 3  (NaN rows excluded)
        - pct_pos = 2/3 ≈ 0.667  (3 and 2 are positive)
        - pct_neg = 1/3 ≈ 0.333  (−1 is negative)
        """
        window_data = pd.DataFrame({
            'vol_gap_adjusted': [3.0, np.nan, -1.0, np.nan, 2.0]
        })
        # adjusted_cgap=4.0 > 0 → dvg = pct_neg - pct_pos = 1/3 - 2/3 = -1/3
        result = cvg_calculator._calculate_window_features(window_data, '8_2', 4.0)

        assert result['cvg_count_8_2'] == 3
        assert result['pct_pos_8_2']   == pytest.approx(2 / 3)
        assert result['pct_neg_8_2']   == pytest.approx(1 / 3)
        # volgap_mean over the 3 non-NaN values: (3 + -1 + 2) / 3 = 4/3
        assert result['volgap_mean_8_2'] == pytest.approx(4.0 / 3.0)
        # cgap is the passed-in adjusted value, not re-derived here
        assert result['cgap_8_2'] == pytest.approx(4.0)


# ============================================================================
# Test Class: Single Date Calculation (calculate method)
# ============================================================================

class TestCalculateSingleDate:
    """Test calculate() method for single date feature generation."""

    def test_calculate_basic_single_ticker(self, cvg_calculator, vol_gap_context,
                                           sample_vol_gap_history):
        """
        Smoke test: calculate() returns correct schema and exact feature values
        for AAPL at PRIMARY_DATE (full window available).

        Given:
        - AAPL at PRIMARY_DATE (position 35 = 2019-09-06), window (8,2)
          → rows [27..33] = 7 rows, all AAPL vol_gap non-NaN

        Stage 1 (per-date median) adjusted AAPL values at each position:
            p27 (2019-07-12): -0.06666 - (-0.07346) = +0.00680  ≥ 0
            p28 (2019-07-19): -0.03157 - (-0.04564) = +0.01407  ≥ 0
            p29 (2019-07-26): -0.05073 - (-0.04087) = -0.00986  < 0
            p30 (2019-08-02): +0.07037 - (-0.10895) = +0.17932  ≥ 0
            p31 (2019-08-09): +0.13706 - (+0.23311) = -0.09605  < 0
            p32 (2019-08-16): +0.09754 - (+0.01840) = +0.07914  ≥ 0
            p33 (2019-08-23): -0.22216 - (-0.24904) = +0.02688  ≥ 0
            → 5 non-negative, 2 negative
            → pct_pos = 5/7, pct_neg = 2/7
            → volgap_mean = sum / 7 ≈ 0.028615377504689090

        Note on cgap/dvg/cvg:
            This test passes only ['AAPL'], so Stage 2's cross-sectional median
            = median([raw_cgap(AAPL)]) = raw_cgap(AAPL) itself → cgap = 0.
            Fix #2 then gives dvg = 0.0, cvg = 1.0.
            The fixture-meaningful cgap test is test_calculate_fix1_cgap_exact_value
            which uses all 4 tickers.

        Expected:
        - 1 row returned with correct schema
        - cvg_count == 7
        - pct_pos == 5/7, pct_neg == 2/7  (exact from fixture)
        - volgap_mean ≈ 0.028615377504689090
        - cgap == 0.0  (single-ticker: cgap always 0 by definition)
        - dvg == 0.0, cvg == 1.0  (Fix #2: cgap == 0 → dvg = 0)
        """
        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, ['AAPL'])

        assert len(result) == 1

        expected_cols = {'ticker', 'date'} | set(cvg_calculator.feature_names)
        assert set(result.columns) == expected_cols

        row = result.iloc[0]
        assert row['ticker'] == 'AAPL'
        assert row['date'] == PRIMARY_DATE

        # All 7 signal features must be non-NaN (full 7-row window available)
        for fn in cvg_calculator.feature_names:
            assert not pd.isna(row[fn]), f"Expected {fn} to be non-NaN for AAPL at PRIMARY_DATE"

        # Window count: 7 rows, all AAPL non-NaN
        assert row['cvg_count_8_2'] == 7

        # Exact pct values derived from Stage 1 (per-date median) of fixture data:
        # signs over window: +,+,-,+,-,+,+ → 5 non-negative, 2 negative
        assert row['pct_pos_8_2'] == pytest.approx(5 / 7)
        assert row['pct_neg_8_2'] == pytest.approx(2 / 7)
        assert row['pct_pos_8_2'] + row['pct_neg_8_2'] == pytest.approx(1.0)

        # Exact volgap_mean: sum of 7 adjusted AAPL values / 7
        assert row['volgap_mean_8_2'] == pytest.approx(0.02861537750468909, rel=1e-6)

        # Single-ticker cgap is always 0: Stage 2 median == raw_cgap(AAPL) → cgap = 0
        # Fix #2 then forces dvg = 0.0, cvg = 1.0
        assert row['cgap_8_2'] == pytest.approx(0.0)
        assert row['dvg_8_2']  == pytest.approx(0.0)
        assert row['cvg_8_2']  == pytest.approx(1.0)

        # Fundamental identity as a final sanity check
        assert row['cvg_8_2'] == pytest.approx(1.0 - row['dvg_8_2'], abs=1e-10)

    def test_calculate_multiple_tickers(self, cvg_calculator, vol_gap_context):
        """
        calculate() returns one row per requested ticker, all satisfying CVG identity.

        Given: ['AAPL', 'TSLA', 'ADP'] at PRIMARY_DATE

        Expected:
        - 3 rows, one per ticker
        - All rows satisfy cvg == 1.0 − dvg (within abs=1e-10)
        - UBER not in result (not requested)
        """
        tickers = ['AAPL', 'TSLA', 'ADP']
        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, tickers)

        assert len(result) == 3
        assert set(result['ticker'].tolist()) == set(tickers)
        assert 'UBER' not in result['ticker'].values

        # All 3 rows must satisfy cvg == 1 - dvg where dvg is non-NaN
        for _, row in result.iterrows():
            if not pd.isna(row['dvg_8_2']):
                assert row['cvg_8_2'] == pytest.approx(1.0 - row['dvg_8_2'], abs=1e-10), (
                    f"{row['ticker']}: cvg={row['cvg_8_2']}, dvg={row['dvg_8_2']}"
                )

    def test_calculate_fix1_cgap_exact_value(self, cvg_calculator, vol_gap_context,
                                              sample_vol_gap_history):
        """
        Verify cgap matches the paper definition (Fix #1) with exact arithmetic.

        Using AAPL at PRIMARY_DATE (position 35 = 2019-09-06):
            raw_cgap(AAPL) = sum(vol_gap[AAPL, positions 27..33])
            cross_median   = median(raw_cgap for all 4 tickers at that window)
            expected_cgap  = raw_cgap(AAPL) - cross_median

        Note: ADP has 2 NaN in the window (positions 30 & 31). Its raw_cgap is
        the sum of its 5 non-NaN values; NaN positions do not contribute to the
        sum. Verify the expected cross_median uses the same NaN-safe sum.

        Compute expected_cgap directly from the fixture DataFrame (no black box),
        then assert it matches the calculator output to rel=1e-5.

        This test fails if the old approach (sum of adjusted) is used, because
        sum(adjusted) ≠ raw_cgap − cross_median when rank-reversal dates are
        in the window (by fixture design).
        """
        all_tickers = ['AAPL', 'TSLA', 'ADP', 'UBER']

        # Compute expected raw_cgap for each ticker: sum of vol_gap at positions 27-33.
        # (At PRIMARY_DATE=pos 35, window (8,2) → start=27, end=33 inclusive → iloc[27:34].)
        # This mirrors exactly what calculate() does: filter ≤ date then iloc slice.
        raw_cgaps = {}
        for t in all_tickers:
            t_data = (
                sample_vol_gap_history[sample_vol_gap_history['ticker'] == t]
                .sort_values('entry_date')
                .reset_index(drop=True)
            )
            # iloc[27:34] = positions 27..33 inclusive (same window as the calculator)
            window = t_data.iloc[27:34]
            raw_cgaps[t] = window['vol_gap'].sum()  # pandas .sum() skips NaN

        cross_median = float(np.median(list(raw_cgaps.values())))
        expected_cgap_aapl = raw_cgaps['AAPL'] - cross_median

        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, all_tickers)
        result = result.set_index('ticker')

        assert result.loc['AAPL', 'cgap_8_2'] == pytest.approx(expected_cgap_aapl, rel=1e-5), (
            f"AAPL cgap mismatch. Expected (paper) {expected_cgap_aapl:.6f}, "
            f"got {result.loc['AAPL', 'cgap_8_2']:.6f}. "
            "Non-match indicates the old sum-of-adjusted approach."
        )

    def test_calculate_fix2_zero_adjusted_cgap(self, cvg_calculator):
        """
        Fix #2 regression via inline DataFrame: ticker whose raw_cgap equals
        the cross-sectional median gets adjusted_cgap == 0 → DVG=0, CVG=1.

        Inline setup (2 dates, 3 tickers, window (2, 0)):
            d1: AAPL=10, TSLA=6, ADP=2   → raw_cgaps at d2 = {AAPL:15, TSLA:9, ADP:3}
            d2: AAPL=5,  TSLA=3, ADP=1
            median(raw_cgaps) = 9
            TSLA adjusted_cgap = 9 − 9 = 0.0

        Expected for TSLA:
        - cgap_2_0 == pytest.approx(0.0)
        - dvg_2_0  == pytest.approx(0.0)   ← Fix #2: NOT pct_pos − pct_neg
        - cvg_2_0  == pytest.approx(1.0)

        Note: adjust the inline values so that TSLA raw_cgap == median exactly.
        """
        d1 = pd.Timestamp('2020-01-03')
        d2 = pd.Timestamp('2020-01-10')

        # raw_cgaps: AAPL=10+5=15, TSLA=6+3=9, ADP=2+1=3
        # np.median([15, 9, 3]) = 9 → TSLA adjusted_cgap = 9 - 9 = 0.0
        history = pd.DataFrame([
            {'ticker': 'AAPL', 'entry_date': d1, 'vol_gap': 10.0},
            {'ticker': 'TSLA', 'entry_date': d1, 'vol_gap':  6.0},
            {'ticker': 'ADP',  'entry_date': d1, 'vol_gap':  2.0},
            {'ticker': 'AAPL', 'entry_date': d2, 'vol_gap':  5.0},
            {'ticker': 'TSLA', 'entry_date': d2, 'vol_gap':  3.0},
            {'ticker': 'ADP',  'entry_date': d2, 'vol_gap':  1.0},
        ])
        ctx = FeatureDataContext(straddle_history=history)
        calc = CVGCalculator(windows=[(2, 0)], min_periods=1)

        result = calc.calculate(ctx, d2, ['AAPL', 'TSLA', 'ADP'])
        tsla = result[result['ticker'] == 'TSLA'].iloc[0]

        # TSLA raw_cgap (9) == cross_median (9) → adjusted_cgap = 0.0
        assert tsla['cgap_2_0'] == pytest.approx(0.0)

        # Fix #2: cgap == 0 → DVG = 0, CVG = 1 (old code would use pct_pos - pct_neg)
        assert tsla['dvg_2_0'] == pytest.approx(0.0), (
            f"Fix #2 violated: expected dvg=0.0 when cgap=0, got {tsla['dvg_2_0']}"
        )
        assert tsla['cvg_2_0'] == pytest.approx(1.0)

    def test_calculate_collapsed_window(self, cvg_calculator, vol_gap_context):
        """
        Test early position where end_idx < start_idx (window collapses).

        At position 1 (2019-01-11) with window (8, 2):
            start = max(0, 1−8) = 0
            end   = 1 − 2 = −1   → −1 < 0 → collapsed

        Expected:
        - cvg_count == 0
        - cvg, dvg, cgap, pct_pos, pct_neg, volgap_mean all NaN
        """
        collapsed_date = pd.Timestamp('2019-01-11')  # position 1
        result = cvg_calculator.calculate(vol_gap_context, collapsed_date, ['AAPL'])

        assert len(result) == 1
        row = result.iloc[0]

        # Collapsed window: end_idx = 1-2 = -1 < start_idx=0 → count=0
        assert row['cvg_count_8_2'] == 0

        signal_cols = ['cvg_8_2', 'dvg_8_2', 'cgap_8_2',
                       'pct_pos_8_2', 'pct_neg_8_2', 'volgap_mean_8_2']
        for col in signal_cols:
            assert pd.isna(row[col]), f"Expected {col} NaN for collapsed window, got {row[col]}"

    def test_calculate_ticker_not_in_history(self, cvg_calculator, vol_gap_context):
        """
        Ticker with no history returns a row of all NaN features.

        Given: ticker 'XYZ' not in sample_vol_gap_history

        Expected:
        - 1 row returned (no error)
        - All 7 feature columns NaN
        """
        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, ['XYZ'])

        assert len(result) == 1
        row = result.iloc[0]
        assert row['ticker'] == 'XYZ'

        for fn in cvg_calculator.feature_names:
            assert pd.isna(row[fn]), f"Expected {fn} NaN for unknown ticker, got {row[fn]}"

    def test_calculate_empty_ticker_list(self, cvg_calculator, vol_gap_context):
        """
        Empty ticker list returns zero-row DataFrame.

        Expected:
        - len(result) == 0
        """
        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, [])
        assert len(result) == 0


# ============================================================================
# Test Class: Bulk Calculation (calculate_bulk method)
# ============================================================================

class TestCalculateBulk:
    """Test calculate_bulk() for correctness, schema, and fix regressions."""

    def test_calculate_bulk_output_schema(self, cvg_calculator, vol_gap_context):
        """
        Verify output column list is exactly ['ticker', 'date'] + feature_names.

        Given: AAPL over any 3-date range within the fixture

        Expected:
        - Column order: ['ticker', 'date', 'cvg_8_2', 'dvg_8_2', 'cgap_8_2',
                          'pct_pos_8_2', 'pct_neg_8_2', 'volgap_mean_8_2', 'cvg_count_8_2']
        - 'entry_date' NOT in columns
        """
        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=pd.Timestamp('2019-02-22'),  # position 7
            end_date=pd.Timestamp('2019-03-08'),    # position 9 (3 dates)
            tickers=['AAPL'],
        )

        expected_cols = ['ticker', 'date'] + cvg_calculator.feature_names
        assert list(result.columns) == expected_cols, (
            f"Column mismatch.\nExpected: {expected_cols}\nGot:      {list(result.columns)}"
        )
        assert 'entry_date' not in result.columns

    def test_calculate_bulk_date_filtering(self, cvg_calculator, vol_gap_context):
        """
        Only dates within [start_date, end_date] inclusive are returned.

        Given: 5-date range covering positions 7-11

        Expected:
        - Exactly 5 rows per ticker
        - result['date'].min() == start_date
        - result['date'].max() == end_date
        - No dates outside the range
        """
        # Positions 7-11 = 2019-02-22 to 2019-03-22 (5 consecutive weekly dates)
        start_date = pd.Timestamp('2019-02-22')  # position 7
        end_date   = pd.Timestamp('2019-03-22')  # position 11

        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=start_date,
            end_date=end_date,
            tickers=['AAPL', 'TSLA'],
        )

        # Exactly 5 dates × 2 tickers = 10 rows
        assert len(result[result['ticker'] == 'AAPL']) == 5
        assert len(result[result['ticker'] == 'TSLA']) == 5

        # Inclusive bounds
        assert result['date'].min() == start_date
        assert result['date'].max() == end_date

        # No dates outside [start_date, end_date]
        assert (result['date'] >= start_date).all()
        assert (result['date'] <= end_date).all()

    def test_calculate_bulk_empty_date_range(self, cvg_calculator, vol_gap_context):
        """
        Future date range with no data returns empty DataFrame with correct schema.

        Given: 2025-01-01 to 2025-12-31

        Expected:
        - len(result) == 0
        - Columns == ['ticker', 'date'] + feature_names  (schema preserved)
        """
        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=pd.Timestamp('2025-01-01'),
            end_date=pd.Timestamp('2025-12-31'),
            tickers=['AAPL'],
        )

        assert len(result) == 0

        # Schema must be preserved even for empty result
        expected_cols = ['ticker', 'date'] + cvg_calculator.feature_names
        assert list(result.columns) == expected_cols

    def test_calculate_bulk_cvg_identity_all_rows(self, cvg_calculator, vol_gap_context,
                                                   sample_vol_gap_history):
        """
        For every non-NaN row, the identity cvg == 1.0 − dvg must hold.

        Run calculate_bulk() for all 4 tickers across all 12 dates.
        Filter to rows where dvg is not NaN.

        Assert for every such row:
            abs(row['cvg_8_2'] - (1.0 - row['dvg_8_2'])) < 1e-10
        """
        all_tickers = sample_vol_gap_history['ticker'].unique().tolist()
        all_dates   = sorted(sample_vol_gap_history['entry_date'].unique())

        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=all_dates[0],
            end_date=all_dates[-1],
            tickers=all_tickers,
        )

        valid = result[result['dvg_8_2'].notna()]
        assert len(valid) > 0, "Expected at least some non-NaN dvg rows in full fixture"

        for _, row in valid.iterrows():
            diff = abs(row['cvg_8_2'] - (1.0 - row['dvg_8_2']))
            assert diff < 1e-10, (
                f"CVG identity violated: ticker={row['ticker']}, date={row['date']}, "
                f"cvg={row['cvg_8_2']:.10f}, dvg={row['dvg_8_2']:.10f}, "
                f"1-dvg={1.0 - row['dvg_8_2']:.10f}, diff={diff:.2e}"
            )

    def test_calculate_bulk_fix3_no_cross_ticker_leakage(self, cvg_calculator,
                                                          vol_gap_context):
        """
        Fix #3 regression: shifted rolling values must not bleed across tickers.

        Before Fix #3, ADP's first shifted rows received AAPL's last rolling values
        (same bug as in MomentumCalculator.calculate_bulk).

        Approach: at positions 0 and 1, every ticker's window is too early to have
        min_lag=2 rows of lookback → features should be NaN (count < min_periods).
        Before Fix #3, some tickers' early rows would have spurious non-NaN values
        from the previous ticker's tail.

        Given: calculate_bulk() for ['AAPL', 'ADP'] over the first 3 dates.

        Expected:
        - ADP at position 0 and 1: all signal features NaN (no valid window)
        - Values match calculate() called individually for ADP at those dates
        """
        tickers    = ['AAPL', 'ADP']
        date_p0    = pd.Timestamp('2019-01-04')   # position 0 — collapsed window
        date_p1    = pd.Timestamp('2019-01-11')   # position 1 — collapsed window
        date_p2    = pd.Timestamp('2019-01-18')   # position 2 — count=1 < min_periods=3
        signal_cols = ['cvg_8_2', 'dvg_8_2', 'cgap_8_2',
                       'pct_pos_8_2', 'pct_neg_8_2', 'volgap_mean_8_2']

        bulk = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=date_p0,
            end_date=date_p2,
            tickers=tickers,
        )

        # At positions 0 and 1, both tickers have collapsed windows (end_idx < 0).
        # Before Fix #3, ADP's first 2 rows would have received AAPL's shifted
        # rolling values (spurious non-NaN). Fix #3 ensures they are NaN.
        for early_date in [date_p0, date_p1]:
            for ticker in tickers:
                row = bulk[(bulk['ticker'] == ticker) & (bulk['date'] == early_date)].iloc[0]
                for col in signal_cols:
                    assert pd.isna(row[col]), (
                        f"Fix #3 violation: {ticker} at {early_date} has "
                        f"non-NaN {col}={row[col]:.6f}. "
                        "Indicates cross-ticker leakage from previous ticker's "
                        "shifted rolling values."
                    )

        # Cross-validate: bulk ADP must match individual calculate() at all 3 dates.
        # Use the same 2-ticker set so Stage 2 cross-sectional median is identical.
        for check_date in [date_p0, date_p1, date_p2]:
            single = cvg_calculator.calculate(vol_gap_context, check_date, tickers)
            adp_single = single[single['ticker'] == 'ADP'].iloc[0]
            adp_bulk   = bulk[(bulk['ticker'] == 'ADP') & (bulk['date'] == check_date)].iloc[0]

            for col in signal_cols:
                sv = adp_single[col]
                bv = adp_bulk[col]
                if pd.isna(sv):
                    assert pd.isna(bv), (
                        f"ADP at {check_date} {col}: calculate()=NaN but "
                        f"calculate_bulk()={bv:.6f}. Spurious bulk value (Fix #3 bug)."
                    )
                else:
                    assert sv == pytest.approx(bv, rel=1e-6), (
                        f"ADP at {check_date} {col}: calculate()={sv} != "
                        f"calculate_bulk()={bv}"
                    )

    def test_calculate_bulk_cgap_median_zero_per_date(self, cvg_calculator,
                                                        vol_gap_context,
                                                        sample_vol_gap_history):
        """
        At any date where all 4 tickers have non-NaN cgap, the cross-sectional
        median of cgap values should be ≈ 0 (consequence of Stage 2 adjustment).

        Given: calculate_bulk() for all 4 tickers, filter to PRIMARY_DATE.
        All 4 tickers should have valid cgap at that date (full window available).

        Assert: np.median(cgap_values_at_primary_date) ≈ 0.0  (abs < 1e-10)
        """
        all_tickers = sample_vol_gap_history['ticker'].unique().tolist()

        result = cvg_calculator.calculate_bulk(
            vol_gap_context,
            start_date=PRIMARY_DATE,
            end_date=PRIMARY_DATE,
            tickers=all_tickers,
        )

        at_primary = result[result['date'] == PRIMARY_DATE]['cgap_8_2'].dropna()

        # All 4 tickers must have valid cgap at PRIMARY_DATE by fixture design
        assert len(at_primary) == 4, (
            f"Expected 4 non-NaN cgap values at PRIMARY_DATE, got {len(at_primary)}"
        )

        # Mathematical consequence of Stage 2: median(x - median(x)) == 0
        assert np.median(at_primary.values) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Test Class: Consistency Between Methods
# ============================================================================

class TestConsistency:
    """
    Cross-validate calculate() vs calculate_bulk() output.

    IMPORTANT difference from MomentumCalculator:
        The cgap cross-sectional median in calculate() is computed ONLY over
        the tickers passed in. Therefore, passing a subset of tickers will
        give different cgap values than passing all tickers.

        Both methods must receive the SAME ticker set to be consistent.
        The test test_cgap_cross_sectional_depends_on_ticker_universe (below)
        explicitly documents and verifies this non-isolation property.

    Known divergence on *_count for collapsed windows (same as MomentumCalculator):
        calculate()      → count = 0   (explicit)
        calculate_bulk() → count = NaN (shift produces NaN at group start)
    Both mean "no valid window" and are treated as equivalent in _assert_features_match.
    """

    @staticmethod
    def _assert_features_match(single_row, bulk_row, feature_names, label):
        """
        Compare one row from calculate() against one row from calculate_bulk().

        Signal columns (cvg, dvg, cgap, pct_pos, pct_neg, volgap_mean):
            Must match to rel=1e-6, or both NaN.

        Count column (cvg_count):
            Treat (single=0, bulk=NaN) as equivalent — both mean collapsed window.
            Any other mismatch is a failure.
        """
        signal_cols = [fn for fn in feature_names if not fn.startswith('cvg_count')]
        count_cols  = [fn for fn in feature_names if fn.startswith('cvg_count')]

        for col in signal_cols:
            sv = single_row[col]
            bv = bulk_row[col]
            if pd.isna(sv):
                assert pd.isna(bv), (
                    f"{label} {col}: calculate()=NaN but calculate_bulk()={bv}"
                )
            else:
                assert sv == pytest.approx(bv, rel=1e-6), (
                    f"{label} {col}: calculate()={sv:.8f} != calculate_bulk()={bv:.8f}"
                )

        for col in count_cols:
            sv = single_row[col]
            bv = bulk_row[col]
            # Collapsed window: calculate() returns count=0, calculate_bulk() returns NaN
            # (due to groupby shift at group boundary). Both mean "no valid window".
            if sv == 0 and pd.isna(bv):
                continue
            if pd.isna(sv):
                assert pd.isna(bv), (
                    f"{label} {col}: calculate()=NaN but calculate_bulk()={bv}"
                )
            else:
                assert sv == pytest.approx(bv), (
                    f"{label} {col}: calculate()={sv} != calculate_bulk()={bv}"
                )

    def test_calculate_vs_bulk_single_date_all_tickers(
        self, cvg_calculator, vol_gap_context, sample_vol_gap_history
    ):
        """
        Cross-validate calculate() and calculate_bulk() for all 4 tickers at
        PRIMARY_DATE.

        Both methods called with the same 4-ticker universe so that the
        cross-sectional cgap median is identical.

        Expected: every (ticker, feature) pair matches to rel=1e-6.
        """
        all_tickers   = sample_vol_gap_history['ticker'].unique().tolist()
        feature_names = cvg_calculator.feature_names

        single = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, all_tickers)
        bulk   = cvg_calculator.calculate_bulk(
            vol_gap_context, PRIMARY_DATE, PRIMARY_DATE, tickers=all_tickers
        )

        single_idx = single.set_index('ticker').sort_index()
        bulk_idx   = bulk.set_index('ticker').sort_index()

        for ticker in sorted(all_tickers):
            TestConsistency._assert_features_match(
                single_idx.loc[ticker],
                bulk_idx.loc[ticker],
                feature_names,
                label=f"{ticker}@PRIMARY_DATE",
            )

    def test_calculate_vs_bulk_full_fixture(
        self, cvg_calculator, vol_gap_context, sample_vol_gap_history
    ):
        """
        Exhaustive cross-validation: calculate() vs calculate_bulk() across
        all 12 dates × all 4 tickers.

        This is the strongest regression check and will catch:
        - Any remaining Fix #3 shift contamination
        - Any Stage 1 / Stage 2 pipeline inconsistency
        - Collapsed-window handling differences

        Build ground truth row-by-row via calculate() (all 4 tickers each time).
        Compare against one calculate_bulk() call.

        Expected: all (ticker, date, feature) triples match via _assert_features_match.
        """
        all_tickers   = sample_vol_gap_history['ticker'].unique().tolist()
        all_dates     = sorted(pd.to_datetime(sample_vol_gap_history['entry_date'].unique()))
        feature_names = cvg_calculator.feature_names

        # Build the bulk result once over the entire fixture span
        bulk = cvg_calculator.calculate_bulk(
            vol_gap_context, all_dates[0], all_dates[-1], tickers=all_tickers
        )
        # Index by (ticker, date) for O(1) lookup per pair
        bulk_idx = bulk.set_index(['ticker', 'date'])

        # Ground-truth row-by-row via calculate(), same 4-ticker universe each time
        # so the Stage 2 cross-sectional cgap median is identical in both paths.
        for dt in all_dates:
            single     = cvg_calculator.calculate(vol_gap_context, dt, all_tickers)
            single_idx = single.set_index('ticker')

            for ticker in all_tickers:
                single_row = single_idx.loc[ticker]
                bulk_row   = bulk_idx.loc[(ticker, dt)]

                TestConsistency._assert_features_match(
                    single_row,
                    bulk_row,
                    feature_names,
                    label=f"{ticker}@{dt.date()}",
                )

    def test_cgap_cross_sectional_depends_on_ticker_universe(
        self, cvg_calculator, vol_gap_context
    ):
        """
        Document and verify that cgap values change when the ticker universe changes.

        This is EXPECTED BEHAVIOR: cgap is defined relative to the cross-sectional
        median of whichever tickers are in the calculation.

        Given PRIMARY_DATE:
        - result_2 = calculate(date, ['AAPL', 'TSLA'])          → 2-ticker median
        - result_4 = calculate(date, ['AAPL', 'TSLA', 'ADP', 'UBER']) → 4-ticker median

        Expected:
        - AAPL cgap in result_2 != AAPL cgap in result_4  (different medians)
        - This confirms calculate_bulk(tickers=['AAPL','TSLA']) is NOT the same
          as calculate_bulk(all) filtered to ['AAPL','TSLA'] — unlike momentum.
        """
        result_2 = cvg_calculator.calculate(
            vol_gap_context, PRIMARY_DATE, ['AAPL', 'TSLA']
        ).set_index('ticker')

        result_4 = cvg_calculator.calculate(
            vol_gap_context, PRIMARY_DATE, ['AAPL', 'TSLA', 'ADP', 'UBER']
        ).set_index('ticker')

        cgap_2ticker = result_2.loc['AAPL', 'cgap_8_2']
        cgap_4ticker = result_4.loc['AAPL', 'cgap_8_2']

        # Both must be non-NaN (full window available at PRIMARY_DATE)
        assert not pd.isna(cgap_2ticker), "AAPL cgap is NaN in 2-ticker result"
        assert not pd.isna(cgap_4ticker), "AAPL cgap is NaN in 4-ticker result"

        # The cross-sectional median changes with the universe → cgap values differ.
        # 2-ticker median: median([raw_cgap(AAPL), raw_cgap(TSLA)])
        # 4-ticker median: median([raw_cgap(AAPL), raw_cgap(TSLA), raw_cgap(ADP), raw_cgap(UBER)])
        # These are different medians → different cgap(AAPL) values.
        assert abs(cgap_2ticker - cgap_4ticker) > 1e-8, (
            f"Expected AAPL cgap to differ between 2-ticker and 4-ticker universes. "
            f"cgap_2={cgap_2ticker:.8f}, cgap_4={cgap_4ticker:.8f}. "
            "If equal, the cross-sectional median is insensitive to universe — "
            "which would indicate ADP and UBER raw_cgaps lie symmetrically "
            "around the AAPL/TSLA median (extremely unlikely for real data)."
        )


# ============================================================================
# Test Class: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_ticker_list(self, cvg_calculator, vol_gap_context):
        """
        calculate() with tickers=[] returns zero-row DataFrame.

        Expected: len(result) == 0
        """
        result = cvg_calculator.calculate(vol_gap_context, PRIMARY_DATE, [])
        assert len(result) == 0

    def test_date_before_all_history(self, cvg_calculator, vol_gap_context):
        """
        Request date before first fixture date filters history to empty.

        Given: calculate() at 2018-01-01 for AAPL

        Expected:
        - 1 row returned
        - All 7 feature columns NaN (empty history path)
        """
        pre_history_date = pd.Timestamp('2018-01-01')
        result = cvg_calculator.calculate(vol_gap_context, pre_history_date, ['AAPL'])

        assert len(result) == 1
        row = result.iloc[0]
        assert row['ticker'] == 'AAPL'

        for fn in cvg_calculator.feature_names:
            assert pd.isna(row[fn]), (
                f"Expected {fn} to be NaN for date before all history, got {row[fn]}"
            )

    def test_single_observation_min_periods_one(self, cvg_calculator_no_min,
                                                 vol_gap_context):
        """
        When min_periods=1, a window with exactly 1 valid row produces features.

        Given: AAPL at position 2 (2019-01-18) with window (8, 2):
            start = max(0, 2−8) = 0
            end   = 2 − 2 = 0   → rows [0..0] = exactly 1 row
            count = 1 ≥ min_periods=1 → valid

        Expected:
        - cvg_count == 1
        - cvg and dvg are not NaN
        - cvg == pytest.approx(1.0 − dvg)
        """
        # position 2 = 2019-01-18; window (8,2): start=max(0,2-8)=0, end=2-2=0
        # → iloc[0..0] = 1 row; min_periods=1 → valid
        date_p2 = pd.Timestamp('2019-01-18')
        result = cvg_calculator_no_min.calculate(vol_gap_context, date_p2, ['AAPL'])

        assert len(result) == 1
        row = result.iloc[0]

        assert row['cvg_count_8_2'] == 1
        assert not pd.isna(row['cvg_8_2']), "Expected cvg to be non-NaN with min_periods=1"
        assert not pd.isna(row['dvg_8_2']), "Expected dvg to be non-NaN with min_periods=1"
        assert row['cvg_8_2'] == pytest.approx(1.0 - row['dvg_8_2'], abs=1e-10)

    def test_all_nan_vol_gaps_returns_zero_count(self, cvg_calculator):
        """
        A ticker with all NaN vol_gap values produces count=0 and all NaN features.

        Inline setup: single ticker 'XYZ', 10 dates, all vol_gap = NaN.
        Use window (4, 1), min_periods=1, calculate at last date.

        Expected:
        - cvg_count == 0
        - cvg, dvg, cgap, pct_pos, pct_neg, volgap_mean all NaN
        """
        dates = pd.date_range('2020-01-03', periods=10, freq='W-FRI')
        history = pd.DataFrame({
            'ticker':     ['XYZ'] * 10,
            'entry_date': dates,
            'vol_gap':    [np.nan] * 10,
        })
        ctx = FeatureDataContext(straddle_history=history)
        calc = CVGCalculator(windows=[(4, 1)], min_periods=1)

        result = calc.calculate(ctx, dates[-1], ['XYZ'])

        assert len(result) == 1
        row = result.iloc[0]

        assert row['cvg_count_4_1'] == 0

        signal_cols = ['cvg_4_1', 'dvg_4_1', 'cgap_4_1',
                       'pct_pos_4_1', 'pct_neg_4_1', 'volgap_mean_4_1']
        for col in signal_cols:
            assert pd.isna(row[col]), (
                f"Expected {col} NaN for all-NaN vol_gap, got {row[col]}"
            )

    def test_vol_gap_missing_raises_value_error(self, cvg_calculator):
        """
        Context with neither 'vol_gap' nor ('realized_volatility' + 'entry_iv')
        raises ValueError with a descriptive message.

        Inline setup: context with columns ['ticker', 'entry_date', 'return_pct']
        (wrong schema — no vol_gap and no component columns).

        Expected: pytest.raises(ValueError) containing 'vol_gap'
        """
        bad_history = pd.DataFrame([
            {'ticker': 'AAPL', 'entry_date': pd.Timestamp('2020-01-03'), 'return_pct': 0.05},
            {'ticker': 'AAPL', 'entry_date': pd.Timestamp('2020-01-10'), 'return_pct': -0.02},
        ])
        ctx = FeatureDataContext(straddle_history=bad_history)

        with pytest.raises(ValueError, match='vol_gap'):
            cvg_calculator.calculate(ctx, pd.Timestamp('2020-01-10'), ['AAPL'])
