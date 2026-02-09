# Unit Testing Tracker - Options Backtesting System

**Start Date:** _____________  
**Target Completion:** _____________  
**Overall Coverage Goal:** 85%+

---

## Progress Overview

| Layer | Status | Tests Written | Coverage | Time Spent | Notes |
|-------|--------|---------------|----------|------------|-------|
| **Layer 1: Core Models** | ✅ Complete | 45/45 | 100% | __h / 15h | All model classes tested |
| **Layer 2: Builders** | ⬜ Not Started | 0/8 | 0% | 0h / 6h | |
| **Layer 3: Optimizer** | ⬜ Not Started | 0/12 | 0% | 0h / 12h | |
| **Layer 4: Executor** | ⬜ Not Started | 0/6 | 0% | 0h / 4h | |
| **Layer 5: Strategy** | ⬜ Not Started | 0/10 | 0% | 0h / 8h | |
| **Setup & Infrastructure** | ✅ Complete | - | - | __h / 5h | pytest.ini + conftest.py |
| **TOTAL** | **54%** | **45/84** | **~54%** | **__h / 50h** | |

**Status Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Complete | ⚠️ Blocked

---

## Layer 1: Core Models (15h estimated)
**File:** `tests/unit/test_models.py`  
**Target:** [src/core/models.py](../../src/core/models.py)

### OptionQuote (2h)
- [x] `test_dte_calculation` - Standard, same-day, future
- [x] `test_spread_calculation` - Normal, zero, wide spread
- [x] `test_spread_pct_normal` - Percentage calculation
- [x] `test_spread_pct_zero_mid` - Edge case: zero division protection
- [x] `test_immutability` - Frozen dataclass validation

**Coverage Target:** 100% | **Actual:** 100%

### OptionLeg (4h)
- [x] `test_long_leg_direction` - is_long = True for qty > 0
- [x] `test_short_leg_direction` - is_short = True for qty < 0
- [x] `test_long_leg_premium` - Positive cost (pays premium)
- [x] `test_short_leg_premium` - Negative cost (receives premium)
- [x] `test_call_intrinsic_itm` - Call ITM payoff
- [x] `test_call_intrinsic_atm` - Call ATM (zero)
- [x] `test_call_intrinsic_otm` - Call OTM (zero)
- [x] `test_put_intrinsic_itm` - Put ITM payoff
- [x] `test_put_intrinsic_atm` - Put ATM (zero)
- [x] `test_put_intrinsic_otm` - Put OTM (zero)
- [x] `test_greek_exposures` - Delta, vega, gamma, theta
- [x] `test_intrinsic_value_unsigned_regardless_of_quantity` - Unsigned intrinsic for both long/short

**Coverage Target:** 100% | **Actual:** 100%

### OptionStrategy (4h)
- [x] `test_straddle_net_premium` - Sum of leg premiums
- [x] `test_straddle_is_debit_spread` - Classification
- [x] `test_straddle_net_delta_near_zero` - ATM delta ~0
- [x] `test_straddle_payoff_up_move` - Spot moves up (call wins)
- [x] `test_straddle_payoff_down_move` - Spot moves down (put wins)
- [x] `test_straddle_payoff_no_move` - ATM at expiry (both worthless)
- [x] `test_calendar_spread_multiple_expiries` - Multi-leg with different dates
- [x] `test_greek_aggregation` - Sum of all legs
- [x] `test_synthetic_long_net_premium` - Long call + short put premium
- [x] `test_synthetic_long_is_credit_spread` - Credit classification
- [x] `test_synthetic_long_payoff_up_move` - Linear profit (tests mixed legs)
- [x] `test_synthetic_long_payoff_down_move` - Linear loss (tests mixed legs)
- [x] `test_synthetic_long_payoff_no_move` - Zero payoff at ATM
- [x] `test_synthetic_long_greek_aggregation` - Mixed long/short greek aggregation

**Coverage Target:** 95% | **Actual:** 100% ✅

### Signal (1h)
- [x] `test_signal_validation_valid` - Conviction 0.0, 0.5, 1.0
- [x] `test_signal_validation_invalid` - Conviction -0.1, 1.5
- [x] `test_signal_immutability` - Frozen dataclass

**Coverage Target:** 100% | **Actual:** 100% ✅

### Position (4h)
- [x] `test_open_position_state` - is_open = True, pnl = None
- [x] `test_closed_position_state` - is_open = False, pnl calculated
- [x] `test_winning_debit_position_pnl` - Long trade profit
- [x] `test_losing_debit_position_pnl` - Long trade loss
- [x] `test_winning_credit_position_pnl` - Short trade profit
- [x] `test_losing_credit_position_pnl` - Short trade loss
- [x] `test_pnl_pct_calculation` - Percentage return
- [x] `test_pnl_pct_zero_entry_cost` - Edge case: division by zero
- [x] `test_holding_period_calculation` - Days between entry/exit
- [x] `test_greek_exposures_with_quantity` - Strategy greeks × position qty
- [x] `test_strategy_type_property` - Convenience property

**Coverage Target:** 95% | **Actual:** 100% ✅

---

## Layer 2: Builders (6h estimated)
**File:** `tests/unit/test_builders.py`  
**Target:** [src/strategy/builders.py](../../src/strategy/builders.py)

### StraddleBuilder (6h)
- [ ] `test_find_atm_strike_exact_match` - Spot = 150, strike 150 available
- [ ] `test_find_atm_strike_closest` - Spot = 150.25, picks 150 over 155
- [ ] `test_find_atm_strike_tie_lower_wins` - Spot = 150.50, picks 150 over 151
- [ ] `test_build_strategy_happy_path` - Valid chain, returns 2-leg straddle
- [ ] `test_build_strategy_empty_chain` - Raises ValueError
- [ ] `test_build_strategy_multiple_expiries` - Raises ValueError
- [ ] `test_build_strategy_expiry_mismatch` - Raises ValueError
- [ ] `test_build_strategy_no_atm_options` - Raises ValueError

**Coverage Target:** 90% | **Actual:** ___%

---

## Layer 3: Optimizer (12h estimated)
**File:** `tests/unit/test_optimizer.py`  
**Target:** [src/portfolio/optimizer.py](../../src/portfolio/optimizer.py)

### EqualWeightOptimizer (12h)
- [ ] `test_create_position_long_signal` - Positive quantity, positive entry_cost
- [ ] `test_create_position_short_signal` - Negative quantity, negative entry_cost
- [ ] `test_create_position_fractional_quantity` - qty = 2.73 contracts
- [ ] `test_create_position_zero_premium` - Returns None
- [ ] `test_optimize_balanced_long_short` - 10 longs + 10 shorts = 20 positions
- [ ] `test_optimize_notional_allocation` - $10k per side, correct per-position notional
- [ ] `test_optimize_unbalanced_signals` - 15 longs, 5 shorts
- [ ] `test_optimize_empty_signals` - Returns empty list
- [ ] `test_optimize_no_strategies_available` - Returns empty list
- [ ] `test_optimize_duplicate_tickers` - Raises ValueError
- [ ] `test_optimize_low_capital_warning` - Logs warning but proceeds
- [ ] `test_net_cash_change_near_zero` - Balanced long/short ≈ $0 net

**Coverage Target:** 85% | **Actual:** ___%

---

## Layer 4: Executor (4h estimated)
**File:** `tests/unit/test_executor.py`  
**Target:** [src/execution/backtest_executor.py](../../src/execution/backtest_executor.py)

### BacktestExecutor (4h)
- [ ] `test_execute_entry_noop` - Phase 1: returns position unchanged
- [ ] `test_execute_exit_long_straddle_up_move` - Spot moves up, call ITM
- [ ] `test_execute_exit_long_straddle_down_move` - Spot moves down, put ITM
- [ ] `test_execute_exit_short_straddle` - Negative exit_value
- [ ] `test_execute_exit_fractional_quantity` - qty = 2.5 contracts
- [ ] `test_execute_exit_all_otm` - Zero intrinsic value

**Coverage Target:** 90% | **Actual:** ___%

---

## Layer 5: Strategy (8h estimated)
**File:** `tests/unit/test_strategy.py`  
**Target:** [src/strategy/momentum_cvg.py](../../src/strategy/momentum_cvg.py)

### MomentumCVGStrategy (8h)
- [ ] `test_validation_invalid_lag_range` - max_lag ≤ min_lag raises ValueError
- [ ] `test_validation_invalid_momentum_pct` - Out of (0,1) raises ValueError
- [ ] `test_validation_invalid_cvg_pct` - Out of (0,1] raises ValueError
- [ ] `test_generate_signals_empty_features` - Returns empty list
- [ ] `test_generate_signals_missing_columns` - Raises ValueError
- [ ] `test_generate_signals_data_quality_filter` - Removes low-count tickers
- [ ] `test_generate_signals_momentum_filtering` - Selects correct quantiles
- [ ] `test_generate_signals_cvg_filtering` - Reduces candidates
- [ ] `test_generate_signals_no_duplicates` - All unique tickers
- [ ] `test_generate_signals_conviction_scoring` - Correct percentile calculation

**Coverage Target:** 85% | **Actual:** ___%

---

## Setup & Infrastructure (5h estimated)

### Environment Setup (1h)
- [x] Install pytest, pytest-cov, pytest-mock
- [x] Create `tests/unit/` directory structure
- [x] Create `tests/conftest.py` with shared fixtures
- [x] Create `pytest.ini` configuration

### Fixtures Library (2h)
- [ ] `sample_call` - ATM call option fixture
- [ ] `sample_put` - ATM put option fixture
- [ ] `atm_straddle` - Long straddle strategy fixture
- [ ] `short_straddle` - Short straddle strategy fixture
- [ ] `sample_option_chain` - List[OptionQuote] for builders
- [ ] `sample_signals` - List[Signal] for optimizer
- [ ] `sample_features_df` - DataFrame for strategy tests

### Documentation (2h)
- [ ] Create `tests/README.md` - Testing guide
- [ ] Document TDD workflow
- [ ] Document parametrize patterns
- [ ] Document fixture usage
- [ ] Add pytest command cheatsheet

---

## Known Bugs to Test & Fix

### Critical Bugs (Must Fix During Testing)
- [ ] **Bug #1: Capital Tracking** - [engine.py:267](../../src/backtest/engine.py) - Adds exit_value instead of pnl
  - Test: `test_capital_tracking_short_position`
  - Fix: `actual_capital += sum(p.pnl for p in closed)`

- [ ] **Bug #2: Intrinsic Value Sign** - [backtest_executor.py:211](../../src/execution/backtest_executor.py) - Potential double-counting
  - Test: `test_short_straddle_intrinsic_calculation`
  - Fix: Verify leg.quantity usage in intrinsic calculation

- [ ] **Bug #3: Date/Timestamp Conversion** - [engine.py:280](../../src/backtest/engine.py) - Mixing pd.Timestamp and date
  - Test: `test_features_date_filtering_consistency`
  - Fix: Standardize date types throughout

- [ ] **Bug #4: Duplicate Signal Detection** - [optimizer.py:128](../../src/portfolio/optimizer.py) - Root cause unclear
  - Test: `test_strategy_generates_no_duplicates`
  - Investigate: Can strategy generate duplicate tickers?

---

## Testing Workflow Checklist

### Daily Workflow
- [ ] Run tests: `pytest tests/unit/test_models.py -v`
- [ ] Check coverage: `pytest --cov=src.core.models --cov-report=term-missing`
- [ ] Update tracker: Mark completed tests, record time spent
- [ ] Commit changes: `git commit -m "test: add OptionLeg premium tests"`

### TDD Cycle (Per Test)
1. [ ] Write test with expected behavior
2. [ ] Run pytest - see FAILED (red)
3. [ ] Implement/fix code
4. [ ] Run pytest - see PASSED (green)
5. [ ] Refactor if needed
6. [ ] Commit test + code together

### Weekly Review
- [ ] Generate HTML coverage report: `pytest --cov=src --cov-report=html`
- [ ] Review `htmlcov/index.html` for gaps
- [ ] Identify untested branches
- [ ] Update time estimates for remaining work

---

## Notes & Discoveries

### Bug Findings
_Record any new bugs discovered during testing_

- 

### Testing Patterns Learned
_Document useful pytest patterns as you learn them_

- 

### Blocked Items
_Issues preventing progress_

- 

### Questions for Research
_Technical questions that need investigation_

- 

---

## Coverage Metrics

### Current Status
- **Overall Coverage:** ___%
- **Last Updated:** _____________
- **Tests Passing:** ___/___
- **Tests Failing:** ___

### Coverage by Module
```
src/core/models.py         : ____%
src/strategy/builders.py   : ____%
src/portfolio/optimizer.py : ____%
src/execution/backtest_executor.py : ____%
src/strategy/momentum_cvg.py : ____%
```

### Pytest Commands Reference
```bash
# Run all tests
pytest tests/

# Run specific file
pytest tests/unit/test_models.py -v

# Run specific test
pytest tests/unit/test_models.py::TestOptionLeg::test_long_leg_premium -v

# Pattern matching
pytest -k "intrinsic" -v

# Stop at first failure
pytest -x

# Show print statements
pytest -v -s

# Coverage report
pytest --cov=src tests/ --cov-report=term-missing

# HTML coverage report
pytest --cov=src tests/ --cov-report=html

# Run with coverage threshold
pytest --cov=src tests/ --cov-fail-under=85
```

---

**Next Actions:**
1. Set start date and target completion date
2. Install testing dependencies: `pip install -r requirements-dev.txt`
3. Begin with Layer 1: Create `tests/unit/test_models.py`
4. Write first test: `test_option_quote_dte_calculation`
5. Follow TDD cycle for each subsequent test
