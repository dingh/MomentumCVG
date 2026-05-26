# IronButterflyBuilder — Unit Test Plan

## Gap Analysis

### Already implemented (real test bodies)
- `TestIronButterflyBuilderInit` — ✅ complete
- `TestIronButterflyBuilderHappyPath` — ✅ complete
- `TestStraddleBuilder*` — ✅ complete

### Stub-only (need implementation)
| Class | Tests | Key gap |
|---|---|---|
| `TestIronButterflyBuilderDeltaSelection` | 2 | need bodies, rely on `sample_ibf_chain_multi_width` |
| `TestIronButterflyBuilderChainValidation` | 7 | all `pass` — straightforward ports from straddle equivalents |
| `TestIronButterflyBuilderWingValidation` | 7 | all `pass` — need inline chain construction |
| `TestIronButterflyBuilderHelpers` | 6 | all `pass` — isolated unit tests |

### Missing entirely
| What | Why needed |
|---|---|
| `TestEnumerateCandidates` | Core new logic — bucketing algorithm has no coverage at all |
| Fixtures in `conftest.py` | `sample_ibf_chain_no_mirror`, `ibf_*` fixtures referenced but may be incomplete |

---

## Plan

### Step 1 — Audit `conftest.py` fixtures

Verify these IBF fixtures exist and have correct values matching the hard-coded test assertions:
- `sample_ibf_chain_atm` — body=255.0, wings ±10, net_credit=6.24
- `sample_ibf_chain_multi_width` — body=255.0, Pair A ±2.5 (δ≈0.408), Pair B ±5.0 (δ≈0.318)
- `sample_ibf_chain_no_mirror` — body + OTM calls only, no mirror puts
- `ibf_ticker`, `ibf_trade_date`, `ibf_expiry_date`, `ibf_spot_price`

If any are missing, create them before writing tests.

---

### Step 2 — Add `TestEnumerateCandidates` (highest priority — new logic)

New test class covering `enumerate_candidates()` directly:

| Test | Chain setup | Assertion |
|---|---|---|
| `test_returns_empty_list_when_no_otm_strikes` | body only, no OTM | `== []` |
| `test_returns_empty_list_when_no_mirror_put` | OTM calls, no mirror puts | `== []` |
| `test_returns_empty_list_when_net_credit_zero` | wing cost == body premium | `== []` |
| `test_returns_empty_list_when_spread_too_wide` | one leg has spread_pct > max | `== []` |
| `test_returns_empty_list_when_yield_below_threshold` | net_credit/width < min_yield | `== []` |
| `test_single_valid_pair_returns_one_candidate` | one symmetric pair | `len == 1` |
| `test_candidate_fields_correct` | one pair, known values | assert all 14 fields |
| `test_bucketing_two_candidates_same_bucket_closer_wins` | δ=0.134 and δ=0.152 | only 0.152 returned |
| `test_bucketing_two_candidates_different_buckets` | δ=0.12 and δ=0.22 | both returned (0.10 and 0.20 buckets) |
| `test_bucketing_returns_at_most_one_per_target` | 5 pairs all near δ=0.15 | `len <= 1` for that bucket |
| `test_custom_wing_delta_targets_respected` | any valid chain | override targets, verify bucketing changes |
| `test_sorted_ascending_by_wing_width` | 3 valid pairs | widths ascending |
| `test_default_targets_are_10_15_20_30` | any valid chain | call with no targets, verify ≤4 returned, all claim correct buckets |
| `test_net_greeks_computed_correctly` | one pair, known greeks | assert `net_delta`, `net_gamma`, `net_vega`, `net_theta` |

---

### Step 3 — Complete `TestIronButterflyBuilderDeltaSelection`

Both tests need bodies — straightforward given `sample_ibf_chain_multi_width`:

```
test_selects_narrow_wings_when_wing_delta_targets_high:
    builder = IronButterflyBuilder(wing_delta=0.408, ...)
    → assert legs[3].option.strike == Decimal('257.5')  # call wing
    → assert legs[0].option.strike == Decimal('252.5')  # put wing

test_selects_wide_wings_when_wing_delta_targets_low:
    builder = IronButterflyBuilder(wing_delta=0.318, ...)
    → assert legs[3].option.strike == Decimal('260.0')
    → assert legs[0].option.strike == Decimal('250.0')
```

---

### Step 4 — Complete `TestIronButterflyBuilderChainValidation`

Direct ports of the straddle equivalents — same pattern, different error messages:

| Test | Inline chain | Expected message |
|---|---|---|
| `test_empty_chain_raises` | `[]` | `"Empty option chain"` |
| `test_multiple_expiries_raises` | reuse `sample_option_chain_multiple_expiries` | `"multiple expiries"` |
| `test_expiry_mismatch_raises` | valid chain + wrong expiry arg | `"expiry mismatch"` |
| `test_missing_body_call_raises` | puts only at body | `"No call option found at body strike"` |
| `test_missing_body_put_raises` | calls only at body | `"No put option found at body strike"` |
| `test_zero_mid_body_call_raises` | body call mid=0 | `"Invalid short call mid"` |
| `test_zero_mid_body_put_raises` | body put mid=0 | `"Invalid short put mid"` |

Each uses a **minimal inline chain** (2–4 `OptionQuote` objects constructed directly in the test) — no CSV fixtures needed.

---

### Step 5 — Complete `TestIronButterflyBuilderWingValidation`

Each test builds a minimal 6-option inline chain (body call+put, one wing call+put with one bad property):

| Test | Bad property | Expected message |
|---|---|---|
| `test_no_symmetric_wing_pair_raises` | reuse `sample_ibf_chain_no_mirror` | `"No valid symmetric wing pairs"` |
| `test_wide_spread_sc_raises` | `bid=0.01, ask=5.00` on short call | `"No valid symmetric wing pairs"` |
| `test_wide_spread_sp_raises` | `bid=0.01, ask=5.00` on short put | `"No valid symmetric wing pairs"` |
| `test_wide_spread_lc_raises` | `bid=0.01, ask=5.00` on long call | `"No valid symmetric wing pairs"` |
| `test_wide_spread_lp_raises` | `bid=0.01, ask=5.00` on long put | `"No valid symmetric wing pairs"` |
| `test_negative_net_credit_raises` | wing mids > body mids | `"No valid symmetric wing pairs"` |
| `test_yield_below_threshold_raises` | tiny net_credit vs large width | `"No valid symmetric wing pairs"` |

> **Note:** The spread/yield/credit checks are *silent filters* inside `enumerate_candidates` — they don't raise individually. The `ValueError` comes from `build_strategy_at_body` when `candidates` is empty. The test docstrings' expected messages should be updated to `"No valid symmetric wing pairs"` to match actual behavior.

---

### Step 6 — Complete `TestIronButterflyBuilderHelpers`

All isolated, no fixtures needed:

| Test | Setup | Assert |
|---|---|---|
| `test_compute_yield_on_capital_correct_value` | `net_credit=4.20, width=10.00` | `== 0.42` |
| `test_compute_yield_on_capital_zero_width_raises` | `width=0` | `ValueError("wing_width must be positive")` |
| `test_find_atm_strike_exact_match` | `sample_ibf_chain_atm`, spot=255.0 | `== Decimal('255.0')` |
| `test_find_atm_strike_between_strikes` | `sample_ibf_chain_atm`, spot=257.0 | nearest strike |
| `test_get_option_at_strike_found` | `sample_ibf_chain_atm`, strike=255.0 | not None, correct type |
| `test_get_option_at_strike_not_found` | `sample_ibf_chain_atm`, strike=999.0 | `is None` |

---

## Implementation Order

```
1. conftest.py                             — verify/add IBF fixtures
2. TestEnumerateCandidates                 — new class, 14 tests (highest priority)
3. TestIronButterflyBuilderDeltaSelection  — 2 tests
4. TestIronButterflyBuilderChainValidation — 7 tests
5. TestIronButterflyBuilderWingValidation  — 7 tests (update docstring error messages)
6. TestIronButterflyBuilderHelpers         — 6 tests
```
