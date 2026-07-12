"""Unit tests for src/data/pit_universe_audit.py (Sprint 004 C7.2).

Synthetic fixtures only — no production data is read. Covers the PIT-universe
trust foundation: strict prior-snapshot resolution, artifact validation, the
independent reference universe, production/reference comparison, deterministic
membership hashing, independent rolling recomputation, future invariance,
superset coverage, and missing/new ticker classification.

Test map (design memo §12 / task Part 14): T1–T33.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.data import pit_universe_audit as audit
from src.data.pit_universe_audit import (
    ArtifactValidationError,
    check_artifact_envelope,
    check_build_param_homogeneity,
    check_full_history_superset_coverage,
    check_future_invariance,
    check_panel_grain,
    check_required_columns,
    check_rolling_provenance,
    check_superset_coverage,
    classify_snapshot_membership,
    compare_universe_to_reference,
    compute_reference_universe,
    membership_hash,
    normalize_date_column,
    normalize_date_value,
    recompute_rolling_snapshot,
)

SNAP1 = pd.Timestamp("2024-01-05")
SNAP2 = pd.Timestamp("2024-01-12")


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _ref_panel(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df


def _two_snapshot_panel() -> pd.DataFrame:
    return _ref_panel(
        [
            dict(month_date=SNAP1, ticker="A", atm_straddle_dollar_vol=4_000_000, atm_spread_pct=0.010, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="B", atm_straddle_dollar_vol=3_000_000, atm_spread_pct=0.012, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="C", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=0.014, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="D", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.016, has_valid_atm_pair=False),
            dict(month_date=SNAP2, ticker="A", atm_straddle_dollar_vol=5_000_000, atm_spread_pct=0.009, has_valid_atm_pair=True),
            dict(month_date=SNAP2, ticker="B", atm_straddle_dollar_vol=4_500_000, atm_spread_pct=0.011, has_valid_atm_pair=True),
        ]
    )


def _full_panel(rows, **build_params) -> pd.DataFrame:
    """Panel with full provenance + build-param columns stamped homogeneously."""
    defaults = dict(
        lookback_weeks=3,
        min_valid_quote_weeks=2,
        dte_min=5,
        dte_max=60,
        dvol_top_pct=0.20,
        spread_bot_pct=1.0,
        liquidity_source="raw_option_bid_x_volume_sum_dte_5_60",
    )
    defaults.update(build_params)
    df = pd.DataFrame(rows)
    for k, v in defaults.items():
        df[k] = v
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df


# ---------------------------------------------------------------------------
# T1 — strict global snapshot selection
# ---------------------------------------------------------------------------

def test_t1_strict_global_snapshot_selection():
    panel = _two_snapshot_panel()
    # Trade 2024-01-10: only 2024-01-05 is strictly before → snap1.
    res = compute_reference_universe(date(2024, 1, 10), panel, 1.0, 1.0)
    assert res.resolved_snapshot_date == SNAP1
    assert set(res.selected["ticker"]) == {"A", "B", "C"}  # D invalid pair
    # Trade after snap2 resolves snap2.
    res2 = compute_reference_universe(date(2024, 1, 15), panel, 1.0, 1.0)
    assert res2.resolved_snapshot_date == SNAP2
    assert set(res2.selected["ticker"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# T2 — same-day snapshot prohibited
# ---------------------------------------------------------------------------

def test_t2_same_day_snapshot_prohibited():
    panel = _two_snapshot_panel()
    # trade_date == snap2 date → strict `<` falls back to snap1.
    res = compute_reference_universe(SNAP2, panel, 1.0, 1.0)
    assert res.resolved_snapshot_date == SNAP1
    assert set(res.selected["ticker"]) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# T3 — before / on-first snapshot empty
# ---------------------------------------------------------------------------

def test_t3_before_and_on_first_snapshot_empty():
    panel = _two_snapshot_panel()
    on_first = compute_reference_universe(SNAP1, panel, 1.0, 1.0)
    assert on_first.resolved_snapshot_date is None
    assert on_first.empty_reason == "before_first_snapshot"
    assert on_first.selected.empty

    before = compute_reference_universe(date(2023, 12, 1), panel, 1.0, 1.0)
    assert before.resolved_snapshot_date is None
    assert before.empty_reason == "before_first_snapshot"


# ---------------------------------------------------------------------------
# T4 — duplicate grain FAIL
# ---------------------------------------------------------------------------

def test_t4_duplicate_grain_fail():
    panel = _two_snapshot_panel()
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    res = check_panel_grain(dup)
    assert res.status == "FAIL"
    assert res.details["duplicate_row_count"] >= 2

    clean = check_panel_grain(panel)
    assert clean.status == "PASS"


# ---------------------------------------------------------------------------
# T5 — missing required columns FAIL
# ---------------------------------------------------------------------------

def test_t5_missing_required_columns_fail():
    panel = _two_snapshot_panel().drop(columns=["atm_spread_pct"])
    res = check_required_columns(panel, required=audit.REFERENCE_REQUIRED_COLUMNS)
    assert res.status == "FAIL"
    assert "atm_spread_pct" in res.details["missing"]

    ok = check_required_columns(_two_snapshot_panel(), required=audit.REFERENCE_REQUIRED_COLUMNS)
    assert ok.status == "PASS"


# ---------------------------------------------------------------------------
# T6 — parseable heterogeneous dates normalize
# ---------------------------------------------------------------------------

def test_t6_heterogeneous_dates_normalize():
    frame = pd.DataFrame(
        {
            "month_date": [
                date(2024, 1, 5),
                datetime(2024, 1, 12, 15, 30, 0),
                "2024-01-19",
                pd.Timestamp("2024-01-26"),
            ]
        }
    )
    out = normalize_date_column(frame, "month_date")
    assert list(out) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
        pd.Timestamp("2024-01-19"),
        pd.Timestamp("2024-01-26"),
    ]
    # No time component survives (date precision).
    assert all(ts == ts.normalize() for ts in out)


# ---------------------------------------------------------------------------
# T7 — unparseable dates FAIL
# ---------------------------------------------------------------------------

def test_t7_unparseable_dates_fail():
    frame = pd.DataFrame({"month_date": ["2024-01-05", "not-a-date"]})
    with pytest.raises(ArtifactValidationError):
        normalize_date_column(frame, "month_date")
    # Null required date also fails.
    with pytest.raises(ArtifactValidationError):
        normalize_date_value(None, label="month_date")


# ---------------------------------------------------------------------------
# T8 — timezone-aware dates FAIL explicitly (no silent calendar-day shift)
# ---------------------------------------------------------------------------

def test_t8_timezone_aware_dates_fail():
    # A tz-aware instant near midnight would shift day under UTC conversion.
    aware = pd.Timestamp("2024-01-05 23:30:00", tz="America/New_York")
    with pytest.raises(ArtifactValidationError):
        normalize_date_value(aware, label="month_date")

    frame = pd.DataFrame({"month_date": pd.to_datetime(["2024-01-05"]).tz_localize("UTC")})
    with pytest.raises(ArtifactValidationError):
        normalize_date_column(frame, "month_date")

    # Confirm the naive counterpart normalizes without shifting the calendar day.
    naive = normalize_date_value(datetime(2024, 1, 5, 23, 30, 0), label="month_date")
    assert naive == pd.Timestamp("2024-01-05")


# ---------------------------------------------------------------------------
# T9 — invalid ATM pair exclusion
# ---------------------------------------------------------------------------

def test_t9_invalid_atm_pair_exclusion():
    panel = _two_snapshot_panel()
    res = compute_reference_universe(date(2024, 1, 10), panel, 1.0, 1.0)
    assert "D" not in set(res.selected["ticker"])
    assert res.exclusions["invalid_atm_pair"] == 1


# ---------------------------------------------------------------------------
# T10 — non-finite volume/spread exclusion
# ---------------------------------------------------------------------------

def test_t10_nonfinite_volume_spread_exclusion():
    panel = _ref_panel(
        [
            dict(month_date=SNAP1, ticker="A", atm_straddle_dollar_vol=4_000_000, atm_spread_pct=0.010, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="E", atm_straddle_dollar_vol=float("nan"), atm_spread_pct=0.012, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="F", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=float("nan"), has_valid_atm_pair=True),
        ]
    )
    res = compute_reference_universe(date(2024, 1, 10), panel, 1.0, 1.0)
    assert set(res.selected["ticker"]) == {"A"}
    assert res.exclusions["missing_or_nonfinite_dvol"] == 1
    assert res.exclusions["missing_or_nonfinite_spread"] == 1


# ---------------------------------------------------------------------------
# T11 — rank directions
# ---------------------------------------------------------------------------

def test_t11_rank_directions():
    panel = _two_snapshot_panel()
    res = compute_reference_universe(date(2024, 1, 10), panel, 1.0, 1.0)
    sel = res.selected.set_index("ticker")
    # Highest dollar volume → dvol rank ~1.0 (A); lowest (C) → 1/3.
    assert sel.loc["A", "dvol_rank_pct"] == pytest.approx(1.0)
    assert sel.loc["C", "dvol_rank_pct"] == pytest.approx(1 / 3)
    # Tightest spread (A, 0.010) → spread rank ~1.0.
    assert sel.loc["A", "spread_rank_pct"] == pytest.approx(1.0)
    assert sel.loc["C", "spread_rank_pct"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# T12 — AND filtering (dvol and spread orders differ so AND matters)
# ---------------------------------------------------------------------------

def test_t12_and_filtering():
    panel = _ref_panel(
        [
            dict(month_date=SNAP1, ticker="X", atm_straddle_dollar_vol=3_000_000, atm_spread_pct=0.03, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="Y", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="Z", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=0.02, has_valid_atm_pair=True),
        ]
    )
    # dvol_top=0.5 → {X,Z}; spread_bottom=0.5 → {Y,Z}; AND → {Z}.
    res = compute_reference_universe(date(2024, 1, 10), panel, 0.5, 0.5)
    assert set(res.selected["ticker"]) == {"Z"}


# ---------------------------------------------------------------------------
# T13 — boundary ties with method="average"
# ---------------------------------------------------------------------------

def test_t13_boundary_ties_average():
    panel = _ref_panel(
        [
            dict(month_date=SNAP1, ticker="A", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.02, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="B", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.02, has_valid_atm_pair=True),
            dict(month_date=SNAP1, ticker="C", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True),
        ]
    )
    res = compute_reference_universe(date(2024, 1, 10), panel, 1.0, 1.0)
    sel = res.selected.set_index("ticker")
    # A and B tie: ranks 1,2 → average 1.5 → pct 0.5 each; C → 1.0.
    assert sel.loc["A", "dvol_rank_pct"] == pytest.approx(0.5)
    assert sel.loc["B", "dvol_rank_pct"] == pytest.approx(0.5)
    assert sel.loc["C", "dvol_rank_pct"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T14 — shuffled-row deterministic membership / hash
# ---------------------------------------------------------------------------

def test_t14_shuffled_row_deterministic_hash():
    panel = _two_snapshot_panel()
    trade = date(2024, 1, 10)
    ref = compute_reference_universe(trade, panel, 1.0, 1.0)
    h1 = membership_hash(ref.trade_date, ref.resolved_snapshot_date, 1.0, 1.0, ref.selected)

    shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ref2 = compute_reference_universe(trade, shuffled, 1.0, 1.0)
    h2 = membership_hash(ref2.trade_date, ref2.resolved_snapshot_date, 1.0, 1.0, ref2.selected)

    # Repeated run on the same data is also stable.
    h3 = membership_hash(ref.trade_date, ref.resolved_snapshot_date, 1.0, 1.0, ref.selected)
    assert h1 == h2 == h3
    assert len(h1) == 16


# ---------------------------------------------------------------------------
# T15 — production/reference mismatch detection
# ---------------------------------------------------------------------------

def test_t15_production_reference_mismatch_detection():
    panel = _two_snapshot_panel()
    trade = date(2024, 1, 10)

    # Real S1 matches the independent reference.
    ok = compare_universe_to_reference(trade, panel, 1.0, 1.0)
    assert ok.match
    assert ok.status == "PASS"

    # A buggy production path that injects a phantom ticker is detected.
    def _buggy(td, pnl, cfg):
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "PHANTOM"],
                "dvol_rank_pct": [1.0, 2 / 3, 1 / 3, 0.99],
                "spread_rank_pct": [1.0, 2 / 3, 1 / 3, 0.99],
            }
        )

    bad = compare_universe_to_reference(trade, panel, 1.0, 1.0, step1_fn=_buggy)
    assert not bad.match
    assert bad.status == "FAIL"
    assert "PHANTOM" in bad.production_only


# ---------------------------------------------------------------------------
# T16 — mixed build parameters FAIL
# ---------------------------------------------------------------------------

def test_t16_mixed_build_parameters_fail():
    rows = [
        dict(month_date=SNAP1, ticker="A", atm_straddle_dollar_vol=1.0, atm_spread_pct=0.01, has_valid_atm_pair=True,
             valid_quote_weeks=3, zero_volume_weeks=0, window_start_date=SNAP1, window_end_date=SNAP1, window_shortfall=0),
        dict(month_date=SNAP2, ticker="B", atm_straddle_dollar_vol=1.0, atm_spread_pct=0.01, has_valid_atm_pair=True,
             valid_quote_weeks=3, zero_volume_weeks=0, window_start_date=SNAP2, window_end_date=SNAP2, window_shortfall=0),
    ]
    panel = _full_panel(rows)
    panel.loc[1, "lookback_weeks"] = 26  # heterogeneous build param
    res = check_build_param_homogeneity(panel)
    assert res.status == "FAIL"
    assert "lookback_weeks" in res.details

    clean = check_build_param_homogeneity(_full_panel(rows))
    assert clean.status == "PASS"


# ---------------------------------------------------------------------------
# T17 — invalid requested parameter range FAIL
# ---------------------------------------------------------------------------

def test_t17_invalid_requested_parameter_range_fail():
    assert check_artifact_envelope(0.0, 1.0, 0.20, 1.0).status == "FAIL"
    assert check_artifact_envelope(1.5, 1.0, 0.20, 1.0).status == "FAIL"
    assert check_artifact_envelope(0.10, 0.0, 0.20, 1.0).status == "FAIL"


# ---------------------------------------------------------------------------
# T18 — unsupported dvol envelope FAIL
# ---------------------------------------------------------------------------

def test_t18_unsupported_dvol_envelope_fail():
    res = check_artifact_envelope(0.50, 1.0, 0.20, 1.0)
    assert res.status == "FAIL"
    assert res.supported is False


# ---------------------------------------------------------------------------
# T19 — supported narrower dvol/spread configuration PASS
# ---------------------------------------------------------------------------

def test_t19_supported_narrower_configuration_pass():
    res = check_artifact_envelope(0.10, 0.50, 0.20, 1.0)
    assert res.status == "PASS"
    assert res.supported is True
    # Canonical baseline is also supported.
    assert check_artifact_envelope(0.20, 1.0, 0.20, 1.0).supported is True


# ---------------------------------------------------------------------------
# T20 — sample superset coverage failure
# ---------------------------------------------------------------------------

def test_t20_sample_superset_coverage_failure():
    fail = check_superset_coverage(["A", "B", "Z"], ["A", "B"])
    assert fail.status == "FAIL"
    assert "Z" in fail.missing_from_superset

    ok = check_superset_coverage(["A", "B"], pd.DataFrame({"Ticker": ["A", "B", "C"]}))
    assert ok.status == "PASS"


# ---------------------------------------------------------------------------
# Full-history coverage panel builder (5 tickers × 2 snapshots)
# ---------------------------------------------------------------------------

def _coverage_panel() -> pd.DataFrame:
    rows = []
    for snap in (SNAP1, SNAP2):
        for i, t in enumerate(["T1", "T2", "T3", "T4", "T5"]):
            rows.append(
                dict(
                    month_date=snap,
                    ticker=f"{t}_{snap.day}",
                    atm_straddle_dollar_vol=(i + 1) * 1_000_000,
                    atm_spread_pct=0.01,
                    has_valid_atm_pair=True,
                )
            )
    return _full_panel(rows, dvol_top_pct=0.20, spread_bot_pct=1.0)


# ---------------------------------------------------------------------------
# T21 — full-history coverage PASS on synthetic panel
# ---------------------------------------------------------------------------

def test_t21_full_history_coverage_pass():
    panel = _coverage_panel()
    # dvol_top=0.20 keeps rank_pct >= 0.8 → top two tickers per snapshot (T4_*, T5_*).
    liquid = ["T4_5", "T4_12", "T5_5", "T5_12"]
    res = check_full_history_superset_coverage(panel, liquid)
    assert res.status == "PASS"
    assert res.missing_ticker_count == 0
    assert res.snapshots_checked == 2
    assert res.canonical_params == (0.20, 1.0)


# ---------------------------------------------------------------------------
# T22 — full-history coverage catches missing ticker
# ---------------------------------------------------------------------------

def test_t22_full_history_coverage_catches_missing():
    panel = _coverage_panel()
    liquid = ["T4_5", "T4_12", "T5_5"]  # missing terminal-snapshot selection T5_12
    res = check_full_history_superset_coverage(panel, liquid)
    assert res.status == "FAIL"
    assert res.missing_ticker_count >= 1
    assert "T5_12" in res.sample_missing_tickers


# ---------------------------------------------------------------------------
# Rolling recompute fixtures
# ---------------------------------------------------------------------------

W1 = pd.Timestamp("2024-01-05")
W2 = pd.Timestamp("2024-01-12")
W3 = pd.Timestamp("2024-01-19")
W4 = pd.Timestamp("2024-01-26")


def _weekly(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    return df


def _wrow(week, ticker, vol, spread, valid):
    return dict(
        week_end_date=week,
        ticker=ticker,
        weekly_atm_straddle_dollar_vol=vol,
        weekly_atm_spread_pct=spread,
        weekly_has_valid_quote=valid,
    )


# ---------------------------------------------------------------------------
# T23 — independent rolling recompute matches panel
# ---------------------------------------------------------------------------

def test_t23_rolling_recompute_matches_panel():
    weekly = _weekly(
        [
            _wrow(W1, "AAA", 100.0, 0.02, True),
            _wrow(W2, "AAA", 200.0, 0.04, True),
            _wrow(W3, "AAA", 300.0, 0.06, True),
        ]
    )
    # Hand-computed: vol=(100+200+300)/3=200; spread=mean(.02,.04,.06)=0.04.
    panel = pd.DataFrame(
        [
            dict(
                month_date=W3, ticker="AAA",
                atm_straddle_dollar_vol=200.0, atm_spread_pct=0.04,
                valid_quote_weeks=3, zero_volume_weeks=0, has_valid_atm_pair=True,
                window_start_date=W1, window_end_date=W3, window_shortfall=0,
            )
        ]
    )
    panel["month_date"] = pd.to_datetime(panel["month_date"])
    res = check_rolling_provenance(
        W3, ["AAA"], weekly, panel, lookback_weeks=3, min_valid_quote_weeks=2
    )
    assert res.status == "PASS"
    assert res.recomputed_matches_panel
    assert res.field_mismatches == ()


# ---------------------------------------------------------------------------
# T24 — missing ticker-week zero-fill and fixed denominator
# ---------------------------------------------------------------------------

def test_t24_missing_week_zero_fill_fixed_denominator():
    # A filler ticker establishes the global weekly calendar (W1, W2, W3);
    # BBB is present only in W2, so W1 and W3 are missing ticker-weeks.
    weekly = _weekly(
        [
            _wrow(W1, "ZZZ", 10.0, 0.03, True),
            _wrow(W2, "ZZZ", 10.0, 0.03, True),
            _wrow(W3, "ZZZ", 10.0, 0.03, True),
            _wrow(W2, "BBB", 300.0, 0.04, True),
        ]
    )
    rec = recompute_rolling_snapshot(
        W3, ["BBB"], weekly, lookback_weeks=3, min_valid_quote_weeks=2
    )
    row = rec.loc["BBB"]
    # Denominator stays 3 despite two missing weeks → 300/3 = 100.
    assert row["atm_straddle_dollar_vol"] == pytest.approx(100.0)
    assert int(row["zero_volume_weeks"]) == 2
    assert int(row["valid_quote_weeks"]) == 1
    assert bool(row["has_valid_atm_pair"]) is False


# ---------------------------------------------------------------------------
# T25 — valid spread mean only across valid-quote weeks
# ---------------------------------------------------------------------------

def test_t25_spread_mean_only_valid_weeks():
    weekly = _weekly(
        [
            _wrow(W1, "CCC", 100.0, 0.02, True),
            _wrow(W2, "CCC", 200.0, 0.99, False),  # invalid quote → spread excluded
            _wrow(W3, "CCC", 300.0, 0.06, True),
        ]
    )
    rec = recompute_rolling_snapshot(
        W3, ["CCC"], weekly, lookback_weeks=3, min_valid_quote_weeks=2
    )
    row = rec.loc["CCC"]
    assert row["atm_spread_pct"] == pytest.approx(0.04)  # mean(0.02, 0.06)
    assert int(row["valid_quote_weeks"]) == 2


# ---------------------------------------------------------------------------
# T26 — both-NaN expected spread compares equal
# ---------------------------------------------------------------------------

def test_t26_both_nan_spread_equal():
    assert audit._floats_equal(float("nan"), float("nan")) is True
    weekly = _weekly(
        [
            _wrow(W1, "DDD", 100.0, float("nan"), False),
            _wrow(W2, "DDD", 200.0, float("nan"), False),
            _wrow(W3, "DDD", 300.0, float("nan"), False),
        ]
    )
    panel = pd.DataFrame(
        [
            dict(
                month_date=W3, ticker="DDD",
                atm_straddle_dollar_vol=200.0, atm_spread_pct=float("nan"),
                valid_quote_weeks=0, zero_volume_weeks=0, has_valid_atm_pair=False,
                window_start_date=W1, window_end_date=W3, window_shortfall=0,
            )
        ]
    )
    panel["month_date"] = pd.to_datetime(panel["month_date"])
    res = check_rolling_provenance(
        W3, ["DDD"], weekly, panel, lookback_weeks=3, min_valid_quote_weeks=2
    )
    assert res.status == "PASS"


# ---------------------------------------------------------------------------
# T27 — one-sided NaN mismatch FAIL
# ---------------------------------------------------------------------------

def test_t27_one_sided_nan_mismatch_fail():
    assert audit._floats_equal(float("nan"), 0.05) is False
    weekly = _weekly(
        [
            _wrow(W1, "EEE", 100.0, float("nan"), False),
            _wrow(W2, "EEE", 200.0, float("nan"), False),
            _wrow(W3, "EEE", 300.0, float("nan"), False),
        ]
    )
    # Stored spread is a number but recompute yields NaN → mismatch.
    panel = pd.DataFrame(
        [
            dict(
                month_date=W3, ticker="EEE",
                atm_straddle_dollar_vol=200.0, atm_spread_pct=0.05,
                valid_quote_weeks=0, zero_volume_weeks=0, has_valid_atm_pair=False,
                window_start_date=W1, window_end_date=W3, window_shortfall=0,
            )
        ]
    )
    panel["month_date"] = pd.to_datetime(panel["month_date"])
    res = check_rolling_provenance(
        W3, ["EEE"], weekly, panel, lookback_weeks=3, min_valid_quote_weeks=2
    )
    assert res.status == "FAIL"
    assert any(m[1] == "atm_spread_pct" for m in res.field_mismatches)


# ---------------------------------------------------------------------------
# T28 — future weekly rows do not alter snapshot
# ---------------------------------------------------------------------------

def test_t28_future_weekly_rows_do_not_alter_snapshot():
    weekly = _weekly(
        [
            _wrow(W1, "AAA", 100.0, 0.02, True),
            _wrow(W2, "AAA", 200.0, 0.04, True),
            _wrow(W3, "AAA", 300.0, 0.06, True),
            _wrow(W4, "AAA", 999.0, 0.99, True),  # future week beyond S=W3
        ]
    )
    assert check_future_invariance(W3, ["AAA"], weekly, 3, 2) is True
    rec = recompute_rolling_snapshot(W3, ["AAA"], weekly, 3, 2)
    # Unaffected by W4.
    assert rec.loc["AAA", "atm_straddle_dollar_vol"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# T29 — deliberately future-contaminated stored row is detected
# ---------------------------------------------------------------------------

def test_t29_future_contaminated_row_detected():
    weekly = _weekly(
        [
            _wrow(W1, "AAA", 100.0, 0.02, True),
            _wrow(W2, "AAA", 200.0, 0.04, True),
            _wrow(W3, "AAA", 300.0, 0.06, True),
            _wrow(W4, "AAA", 400.0, 0.08, True),
        ]
    )
    # Contaminated stored row used window [W2,W3,W4] → (200+300+400)/3 = 300.
    contaminated = pd.DataFrame(
        [
            dict(
                month_date=W3, ticker="AAA",
                atm_straddle_dollar_vol=300.0, atm_spread_pct=0.06,
                valid_quote_weeks=3, zero_volume_weeks=0, has_valid_atm_pair=True,
                window_start_date=W2, window_end_date=W3, window_shortfall=0,
            )
        ]
    )
    contaminated["month_date"] = pd.to_datetime(contaminated["month_date"])
    res = check_rolling_provenance(W3, ["AAA"], weekly, contaminated, lookback_weeks=3, min_valid_quote_weeks=2)
    assert res.status == "FAIL"
    # The correct recompute (200) disagrees with the contaminated stored 300.
    assert any(m[1] == "atm_straddle_dollar_vol" for m in res.field_mismatches)


# ---------------------------------------------------------------------------
# T30 — early-history window_shortfall behavior
# ---------------------------------------------------------------------------

def test_t30_early_history_window_shortfall():
    weekly = _weekly(
        [
            _wrow(W1, "AAA", 100.0, 0.02, True),
            _wrow(W2, "AAA", 200.0, 0.04, True),
        ]
    )
    rec = recompute_rolling_snapshot(W2, ["AAA"], weekly, lookback_weeks=3, min_valid_quote_weeks=2)
    row = rec.loc["AAA"]
    assert int(row["window_shortfall"]) == 1  # only 2 of 3 weeks available
    # Denominator still lookback_weeks=3 → (100+200)/3.
    assert row["atm_straddle_dollar_vol"] == pytest.approx(300.0 / 3)


# ---------------------------------------------------------------------------
# T31 — missing / new ticker classification
# ---------------------------------------------------------------------------

def test_t31_missing_new_ticker_classification():
    snap = pd.DataFrame(
        [
            dict(ticker="SEL", atm_straddle_dollar_vol=5_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True, valid_quote_weeks=12),
            dict(ticker="MID1", atm_straddle_dollar_vol=3_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True, valid_quote_weeks=12),
            dict(ticker="MID2", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True, valid_quote_weeks=12),
            dict(ticker="LOWV", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.01, has_valid_atm_pair=True, valid_quote_weeks=12),
            dict(ticker="BADPAIR", atm_straddle_dollar_vol=4_000_000, atm_spread_pct=0.01, has_valid_atm_pair=False, valid_quote_weeks=1),
            dict(ticker="NANV", atm_straddle_dollar_vol=float("nan"), atm_spread_pct=0.01, has_valid_atm_pair=True, valid_quote_weeks=12),
        ]
    )
    # 4 eligible tickers; dvol_top=0.5 keeps rank_pct >= 0.5 → SEL, MID1, MID2.
    # LOWV (lowest dvol → rank 0.25) falls below the dvol threshold.
    cls = classify_snapshot_membership(
        snap, dvol_top_pct=0.5, spread_bottom_pct=1.0,
        all_panel_tickers=["SEL", "MID1", "MID2", "LOWV", "BADPAIR", "NANV", "GHOST"],
        min_valid_quote_weeks=3,
    )
    assert "SEL" in cls.selected
    assert "LOWV" in cls.below_dvol_threshold
    assert "BADPAIR" in cls.invalid_atm_pair
    assert "NANV" in cls.missing_or_nonfinite_dvol
    assert "GHOST" in cls.missing_from_snapshot
    assert "BADPAIR" in cls.new_or_insufficient_history  # valid_quote_weeks < 3


# ---------------------------------------------------------------------------
# T32 — terminal snapshot included in full-history coverage
# ---------------------------------------------------------------------------

def test_t32_terminal_snapshot_included():
    panel = _coverage_panel()
    # Include every non-terminal (SNAP1) selection, but omit the terminal ones.
    liquid = ["T4_5", "T5_5"]
    res = check_full_history_superset_coverage(panel, liquid)
    # Terminal snapshot IS checked → its selection is flagged missing.
    assert "T5_12" in res.sample_missing_tickers
    assert res.snapshots_checked == 2


# ---------------------------------------------------------------------------
# T33 — membership hash changes with params / snapshot / ranks
# ---------------------------------------------------------------------------

def test_t33_membership_hash_changes():
    members = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "dvol_rank_pct": [1.0, 0.5],
            "spread_rank_pct": [1.0, 0.5],
        }
    )
    base = membership_hash("2024-01-15", "2024-01-12", 0.20, 1.0, members)

    # Changing dvol_top_pct changes the hash.
    assert membership_hash("2024-01-15", "2024-01-12", 0.10, 1.0, members) != base
    # Changing spread_bottom_pct changes the hash.
    assert membership_hash("2024-01-15", "2024-01-12", 0.20, 0.5, members) != base
    # Changing the resolved snapshot changes the hash.
    assert membership_hash("2024-01-15", "2024-01-05", 0.20, 1.0, members) != base
    # Changing a rank value changes the hash.
    m2 = members.copy()
    m2.loc[1, "dvol_rank_pct"] = 0.4
    assert membership_hash("2024-01-15", "2024-01-12", 0.20, 1.0, m2) != base
    # Changing membership changes the hash.
    m3 = pd.concat(
        [members, pd.DataFrame({"ticker": ["C"], "dvol_rank_pct": [0.3], "spread_rank_pct": [0.3]})],
        ignore_index=True,
    )
    assert membership_hash("2024-01-15", "2024-01-12", 0.20, 1.0, m3) != base
