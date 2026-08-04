"""
Sprint 005 D1 — feature_backfill_v1 contract tests
(G1–G6, G8–G10, G12–G18 unit portion, G16–G17).

G1–G2 freeze the versioned window/config contract.
G3–G6 lock weekly Momentum inclusive-lag, null-slot, and partial-history semantics.
G8–G10 / G12–G15 lock CVG cross-sectional cumulative, sign-branch, sparse,
and independent-count semantics (G18 unit range asserts included).
G16–G17 prove synthetic PIT expiry-before-feature-date for min_lag=2 and baseline 42:8.

G7/G11 (zero-neutral) are intentionally deferred. Configuration- and synthetic-panel
only: no D2 artifact, no Windows data paths.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.base import FeatureDataContext
from src.features.cvg_calculator import CVGCalculator
from src.features.momentum_calculator import MomentumCalculator

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "feature_backfill_v1.json"
BUILD_FEATURES_PATH = REPO_ROOT / "scripts" / "build_features.py"


def _load_spec() -> dict:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_generate_momentum_windows():
    spec = importlib.util.spec_from_file_location(
        "build_features_for_contract", BUILD_FEATURES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.generate_momentum_windows


def _oracle_windows(
    *,
    min_lag_start: int,
    min_lag_end: int,
    max_lag_start: int,
    max_lag_end: int,
    step: int,
) -> list[tuple[int, int]]:
    return [
        (max_lag, min_lag)
        for max_lag in range(max_lag_start, max_lag_end + 1, step)
        for min_lag in range(min_lag_start, min_lag_end + 1, step)
        if max_lag > min_lag
    ]


def _windows_from_spec(spec: dict) -> list[tuple[int, int]]:
    """Expand the frozen grid with every bound supplied explicitly from the spec."""
    windows_cfg = spec["windows"]
    generate = _load_generate_momentum_windows()
    return generate(
        short_range=(windows_cfg["min_lag_start"], windows_cfg["min_lag_end"]),
        long_range=(windows_cfg["max_lag_start"], windows_cfg["max_lag_end"]),
        step=windows_cfg["step"],
    )


def _render_output_columns(templates: list[str], max_lag: int, min_lag: int) -> list[str]:
    rendered = []
    for template in templates:
        if "{max}" in template or "{min}" in template:
            rendered.append(
                template.replace("{max}", str(max_lag)).replace("{min}", str(min_lag))
            )
        else:
            rendered.append(template)
    return rendered


def _momentum_min_periods(spec: dict) -> int:
    return int(spec["momentum"]["min_periods"])


def _cvg_min_periods(spec: dict) -> int:
    return int(spec["cvg"]["min_periods"])


# ---------------------------------------------------------------------------
# G1 — Exact 281-window expansion
# ---------------------------------------------------------------------------


def test_g1_grid_expands_to_281_unique_windows_in_deterministic_order():
    spec = _load_spec()
    windows_cfg = spec["windows"]

    assert windows_cfg["min_lag_start"] == 2
    assert windows_cfg["min_lag_end"] == 24
    assert windows_cfg["max_lag_start"] == 6
    assert windows_cfg["max_lag_end"] == 60
    assert windows_cfg["step"] == 2
    assert windows_cfg["require_max_gt_min"] is True
    assert windows_cfg["order"] == "max_lag_outer_min_lag_inner"
    assert windows_cfg["expected_count"] == 281

    windows = _windows_from_spec(spec)
    oracle = _oracle_windows(
        min_lag_start=windows_cfg["min_lag_start"],
        min_lag_end=windows_cfg["min_lag_end"],
        max_lag_start=windows_cfg["max_lag_start"],
        max_lag_end=windows_cfg["max_lag_end"],
        step=windows_cfg["step"],
    )

    assert len(windows) == 281
    assert len(set(windows)) == 281
    assert windows == oracle
    assert windows[0] == (6, 2)
    assert windows[:5] == [(6, 2), (6, 4), (8, 2), (8, 4), (8, 6)]
    assert windows[-5:] == [(60, 16), (60, 18), (60, 20), (60, 22), (60, 24)]
    assert windows[-1] == (60, 24)
    assert all(max_lag > min_lag for max_lag, min_lag in windows)
    assert (42, 8) in windows


# ---------------------------------------------------------------------------
# G2 — Configuration authority
# ---------------------------------------------------------------------------


def test_g2_spec_is_sole_authority_for_windows_min_periods_baseline_and_columns():
    spec = _load_spec()
    windows = _windows_from_spec(spec)

    # Baseline 42:8 is present and explicitly designated.
    baseline = spec["baseline_window"]
    assert baseline == {"max_lag": 42, "min_lag": 8}
    assert (baseline["max_lag"], baseline["min_lag"]) in windows

    # Both minimum-period values are explicitly 1.
    assert spec["momentum"]["min_periods"] == 1
    assert spec["cvg"]["min_periods"] == 1

    mom = MomentumCalculator(
        windows=windows,
        min_periods=spec["momentum"]["min_periods"],
    )
    cvg = CVGCalculator(
        windows=windows,
        min_periods=spec["cvg"]["min_periods"],
    )
    assert mom.min_periods == 1
    assert cvg.min_periods == 1
    assert mom.min_periods == spec["momentum"]["min_periods"]
    assert cvg.min_periods == spec["cvg"]["min_periods"]

    # Required output-column templates render correctly, including for 42:8.
    templates = spec["output_columns_per_window"]
    assert "mom_{max}_{min}_mean" in templates
    assert "mom_{max}_{min}_count" in templates
    assert "cvg_{max}_{min}" in templates
    assert "cvg_count_{max}_{min}" in templates

    rendered_baseline = _render_output_columns(
        templates, baseline["max_lag"], baseline["min_lag"]
    )
    assert rendered_baseline == [
        "ticker",
        "date",
        "mom_42_8_mean",
        "mom_42_8_count",
        "cvg_42_8",
        "cvg_count_42_8",
    ]

    # Semantics-only: lineage / machine-specific fields must not be spec keys.
    forbidden_keys = {
        "snapshot_id",
        "repo_sha",
        "file_sha256",
        "a1_key_digest",
        "build_id",
        "observation_path",
        "derived_path",
        "cache_path",
    }

    def _collect_keys(obj, found: set[str]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                found.add(key)
                _collect_keys(value, found)
        elif isinstance(obj, list):
            for item in obj:
                _collect_keys(item, found)

    present_keys: set[str] = set()
    _collect_keys(spec, present_keys)
    assert forbidden_keys.isdisjoint(present_keys)


def test_g2_grid_generation_uses_explicit_spec_bounds_not_function_defaults():
    """All bounds come from the versioned spec; helper defaults are not consulted."""
    spec = _load_spec()
    windows_cfg = spec["windows"]
    generate = _load_generate_momentum_windows()

    # Every bound is supplied explicitly from the frozen contract.
    from_spec = generate(
        short_range=(windows_cfg["min_lag_start"], windows_cfg["min_lag_end"]),
        long_range=(windows_cfg["max_lag_start"], windows_cfg["max_lag_end"]),
        step=windows_cfg["step"],
    )
    assert len(from_spec) == windows_cfg["expected_count"] == 281
    assert len(set(from_spec)) == 281
    assert from_spec[0] == (6, 2)
    assert from_spec[-1] == (60, 24)
    assert (42, 8) in from_spec
    assert all(max_lag > min_lag for max_lag, min_lag in from_spec)


# ---------------------------------------------------------------------------
# G3 / G5 / G6 — Inclusive endpoints, retained null week, partial history
# ---------------------------------------------------------------------------


def test_g3_g5_g6_momentum_inclusive_null_slot_and_partial_history():
    """
    Compact (4, 2) literal window — not part of the production 281-window grid.

    positions:  0    1    2     3    4    5
    returns:   10   20   NaN   30   40   50
    """
    spec = _load_spec()
    min_periods = _momentum_min_periods(spec)
    assert min_periods == 1

    dates = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    assert list(dates) == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-17"),
        pd.Timestamp("2020-01-24"),
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-07"),
    ]
    returns = [10.0, 20.0, np.nan, 30.0, 40.0, 50.0]
    history = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "entry_date": dates,
            "return_pct": returns,
        }
    )
    # G5: null week remains a scheduled row (not dropped).
    assert len(history) == 6
    assert pd.isna(history.loc[2, "return_pct"])

    calc = MomentumCalculator(windows=[(4, 2)], min_periods=min_periods)
    context = FeatureDataContext(straddle_history=history)
    result = calc.calculate_bulk(
        context,
        start_date=dates[0],
        end_date=dates[-1],
        tickers=["AAA"],
    )

    # G3 / G5 at position 5: inclusive window uses positions 1–3 → [20, NaN, 30].
    feature_date = dates[5]
    row = result.loc[result["date"] == feature_date].iloc[0]
    assert dates[1] == pd.Timestamp("2020-01-10")
    assert dates[3] == pd.Timestamp("2020-01-24")
    contributing = returns[1:4]
    assert len(contributing) == 3
    assert contributing[0] == 20.0
    assert pd.isna(contributing[1])
    assert contributing[2] == 30.0
    assert row["mom_4_2_mean"] == 25.0
    assert row["mom_4_2_count"] == 2

    # G6 at position 2: partial history uses position 0 only → [10].
    partial_date = dates[2]
    partial = result.loc[result["date"] == partial_date].iloc[0]
    assert returns[0] == 10.0
    assert partial["mom_4_2_mean"] == 10.0
    assert partial["mom_4_2_count"] == 1
    assert not pd.isna(partial["mom_4_2_mean"])


# ---------------------------------------------------------------------------
# G16 — Synthetic PIT at min_lag = 2
# ---------------------------------------------------------------------------


def test_g16_synthetic_pit_at_min_lag_2():
    spec = _load_spec()
    min_periods = _momentum_min_periods(spec)
    assert min_periods == 1

    dates = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    history = pd.DataFrame(
        {
            "ticker": ["T"] * 6,
            "entry_date": dates,
            "return_pct": [1.0] * 6,
            "expiry_date": dates + pd.Timedelta(days=7),
        }
    )

    calc = MomentumCalculator(windows=[(4, 2)], min_periods=min_periods)
    context = FeatureDataContext(straddle_history=history)
    result = calc.calculate_bulk(
        context,
        start_date=dates[0],
        end_date=dates[-1],
        tickers=["T"],
    )

    feature_date = dates[5]
    row = result.loc[result["date"] == feature_date].iloc[0]
    contributing_positions = list(range(1, 4))  # inclusive 1..3
    assert contributing_positions == [1, 2, 3]
    assert row["mom_4_2_mean"] == 1.0
    assert row["mom_4_2_count"] == 3

    contributing = history.iloc[contributing_positions]
    assert (contributing["expiry_date"] < feature_date).all()


# ---------------------------------------------------------------------------
# G4 / G17 — Baseline 42:8 inclusive endpoints + synthetic PIT
# ---------------------------------------------------------------------------


def test_g4_g17_baseline_42_8_inclusive_endpoints_and_pit():
    spec = _load_spec()
    min_periods = _momentum_min_periods(spec)
    assert min_periods == 1
    baseline = spec["baseline_window"]
    assert (baseline["max_lag"], baseline["min_lag"]) == (42, 8)

    dates = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    history = pd.DataFrame(
        {
            "ticker": ["BBB"] * 50,
            "entry_date": dates,
            "return_pct": [1.0] * 50,
            "expiry_date": dates + pd.Timedelta(days=7),
        }
    )

    calc = MomentumCalculator(
        windows=[(baseline["max_lag"], baseline["min_lag"])],
        min_periods=min_periods,
    )
    context = FeatureDataContext(straddle_history=history)
    result = calc.calculate_bulk(
        context,
        start_date=dates[0],
        end_date=dates[-1],
        tickers=["BBB"],
    )

    feature_pos = 42
    feature_date = dates[feature_pos]
    row = result.loc[result["date"] == feature_date].iloc[0]

    # Inclusive contributing positions for (42, 8): start = 42-42 = 0, end = 42-8 = 34.
    contributing_positions = list(range(0, 35))
    assert contributing_positions == list(range(0, 35))
    assert len(contributing_positions) == 35
    assert row["mom_42_8_mean"] == 1.0
    assert row["mom_42_8_count"] == 35

    contributing = history.iloc[contributing_positions]
    assert (contributing["expiry_date"] < feature_date).all()


# ---------------------------------------------------------------------------
# G8 — Raw cumulative gap + feature-date second median
# ---------------------------------------------------------------------------


def test_g8_raw_cumulative_gap_and_second_median():
    """Test-only window (2, 0) — not part of the production 281-window grid."""
    spec = _load_spec()
    min_periods = _cvg_min_periods(spec)
    assert min_periods == 1

    d1 = pd.Timestamp("2020-01-03")
    d2 = pd.Timestamp("2020-01-10")
    history = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_date": d1, "vol_gap": 10.0},
            {"ticker": "TSLA", "entry_date": d1, "vol_gap": 2.0},
            {"ticker": "ADP", "entry_date": d1, "vol_gap": 4.0},
            {"ticker": "AAPL", "entry_date": d2, "vol_gap": 2.0},
            {"ticker": "TSLA", "entry_date": d2, "vol_gap": 10.0},
            {"ticker": "ADP", "entry_date": d2, "vol_gap": 4.0},
        ]
    )
    # Hand calc: raw sums AAPL=12, TSLA=12, ADP=8 → second median = 12.
    assert 10.0 + 2.0 == 12.0
    assert 2.0 + 10.0 == 12.0
    assert 4.0 + 4.0 == 8.0
    assert float(np.median([12.0, 12.0, 8.0])) == 12.0

    calc = CVGCalculator(windows=[(2, 0)], min_periods=min_periods)
    result = calc.calculate_bulk(
        FeatureDataContext(straddle_history=history),
        start_date=d1,
        end_date=d2,
    )
    aapl = result.loc[(result["ticker"] == "AAPL") & (result["date"] == d2)].iloc[0]
    assert aapl["cgap_2_0"] == pytest.approx(0.0)
    assert aapl["cvg_2_0"] == pytest.approx(1.0)
    assert aapl["cgap_2_0"] != pytest.approx(4.0)


# ---------------------------------------------------------------------------
# G9 / G10 / G12 / G14 / G18 — shared three-ticker panel
# ---------------------------------------------------------------------------


def test_g9_g10_g12_g14_g18_shared_panel_sign_branches_universe_and_range():
    """Test-only window (3, 0) — not part of the production 281-window grid."""
    spec = _load_spec()
    min_periods = _cvg_min_periods(spec)
    assert min_periods == 1

    d1 = pd.Timestamp("2020-01-03")
    d2 = pd.Timestamp("2020-01-10")
    d3 = pd.Timestamp("2020-01-17")
    history = pd.DataFrame(
        [
            {"ticker": "A", "entry_date": d1, "vol_gap": 10.0},
            {"ticker": "B", "entry_date": d1, "vol_gap": 6.0},
            {"ticker": "C", "entry_date": d1, "vol_gap": 2.0},
            {"ticker": "A", "entry_date": d2, "vol_gap": 10.0},
            {"ticker": "B", "entry_date": d2, "vol_gap": 6.0},
            {"ticker": "C", "entry_date": d2, "vol_gap": 2.0},
            {"ticker": "A", "entry_date": d3, "vol_gap": 2.0},
            {"ticker": "B", "entry_date": d3, "vol_gap": 6.0},
            {"ticker": "C", "entry_date": d3, "vol_gap": 10.0},
        ]
    )

    calc = CVGCalculator(windows=[(3, 0)], min_periods=min_periods)
    full = calc.calculate_bulk(
        FeatureDataContext(straddle_history=history),
        start_date=d1,
        end_date=d3,
    )
    a = full.loc[(full["ticker"] == "A") & (full["date"] == d3)].iloc[0]
    b = full.loc[(full["ticker"] == "B") & (full["date"] == d3)].iloc[0]
    c = full.loc[(full["ticker"] == "C") & (full["date"] == d3)].iloc[0]

    # G9 — positive final cgap (ticker A).
    assert a["cvg_count_3_0"] == 3
    assert a["pct_pos_3_0"] == pytest.approx(2 / 3)
    assert a["pct_neg_3_0"] == pytest.approx(1 / 3)
    assert a["cgap_3_0"] == pytest.approx(4.0)
    assert a["dvg_3_0"] == pytest.approx(-1 / 3)
    assert a["cvg_3_0"] == pytest.approx(4 / 3)

    # G10 — negative final cgap (ticker C).
    assert c["cvg_count_3_0"] == 3
    assert c["pct_pos_3_0"] == pytest.approx(1 / 3)
    assert c["pct_neg_3_0"] == pytest.approx(2 / 3)
    assert c["cgap_3_0"] == pytest.approx(-4.0)
    assert c["dvg_3_0"] == pytest.approx(-1 / 3)
    assert c["cvg_3_0"] == pytest.approx(4 / 3)

    # G12 — exact cgap == 0 → DVG = 0 → CVG = 1 (ticker B).
    # Do not assert B pct_pos/pct_neg here (zero-neutral belongs to Commit 4).
    assert b["cgap_3_0"] == pytest.approx(0.0)
    assert b["dvg_3_0"] == pytest.approx(0.0)
    assert b["cvg_3_0"] == pytest.approx(1.0)

    # G14 — prefiltering before CVG changes other names' cgap.
    subset = calc.calculate_bulk(
        FeatureDataContext(straddle_history=history),
        start_date=d1,
        end_date=d3,
        tickers=["A", "B"],
    )
    subset_a = subset.loc[(subset["ticker"] == "A") & (subset["date"] == d3)].iloc[0]
    assert a["cgap_3_0"] == pytest.approx(4.0)
    assert subset_a["cgap_3_0"] == pytest.approx(2.0)
    assert a["cgap_3_0"] != subset_a["cgap_3_0"]
    assert a["cvg_3_0"] == pytest.approx(4 / 3)
    assert subset_a["cvg_3_0"] == pytest.approx(4 / 3)

    # G18 unit portion — finite CVG values exercised here stay in [0, 2].
    for value in (a["cvg_3_0"], b["cvg_3_0"], c["cvg_3_0"], subset_a["cvg_3_0"]):
        assert 0.0 <= value <= 2.0


# ---------------------------------------------------------------------------
# G13 — Sparse history participates in the second median
# ---------------------------------------------------------------------------


def test_g13_sparse_history_participates_in_second_median():
    """Test-only window (2, 0) — not part of the production 281-window grid."""
    spec = _load_spec()
    min_periods = _cvg_min_periods(spec)
    assert min_periods == 1

    d1 = pd.Timestamp("2020-01-03")
    d2 = pd.Timestamp("2020-01-10")
    history = pd.DataFrame(
        [
            {"ticker": "A", "entry_date": d1, "vol_gap": 10.0},
            {"ticker": "B", "entry_date": d1, "vol_gap": 4.0},
            {"ticker": "S", "entry_date": d1, "vol_gap": np.nan},
            {"ticker": "A", "entry_date": d2, "vol_gap": 2.0},
            {"ticker": "B", "entry_date": d2, "vol_gap": 4.0},
            {"ticker": "S", "entry_date": d2, "vol_gap": 6.0},
        ]
    )
    # Hand calc: raw A=12, B=8, S=6 → median 8 → A cgap=4 (not counterfactual 2).
    assert float(np.median([12.0, 8.0, 6.0])) == 8.0
    assert 12.0 - 8.0 == 4.0
    assert float(np.median([12.0, 8.0])) == 10.0
    assert 12.0 - 10.0 == 2.0

    calc = CVGCalculator(windows=[(2, 0)], min_periods=min_periods)
    result = calc.calculate_bulk(
        FeatureDataContext(straddle_history=history),
        start_date=d1,
        end_date=d2,
    )
    a = result.loc[(result["ticker"] == "A") & (result["date"] == d2)].iloc[0]
    s = result.loc[(result["ticker"] == "S") & (result["date"] == d2)].iloc[0]

    assert a["cgap_2_0"] == pytest.approx(4.0)
    assert a["cvg_count_2_0"] == 2
    assert a["pct_pos_2_0"] == pytest.approx(0.5)
    assert a["pct_neg_2_0"] == pytest.approx(0.5)
    assert a["dvg_2_0"] == pytest.approx(0.0)
    assert a["cvg_2_0"] == pytest.approx(1.0)
    assert a["cgap_2_0"] != pytest.approx(2.0)

    assert s["cgap_2_0"] == pytest.approx(-2.0)
    assert s["cvg_count_2_0"] == 1
    assert s["pct_pos_2_0"] == pytest.approx(1.0)
    assert s["pct_neg_2_0"] == pytest.approx(0.0)
    assert s["dvg_2_0"] == pytest.approx(1.0)
    assert s["cvg_2_0"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# G15 — Independent Momentum and CVG availability
# ---------------------------------------------------------------------------


def test_g15_independent_momentum_and_cvg_availability():
    """Test-only window (4, 0) — not part of the production 281-window grid."""
    spec = _load_spec()
    mom_min = _momentum_min_periods(spec)
    cvg_min = _cvg_min_periods(spec)
    assert mom_min == 1
    assert cvg_min == 1

    dates = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    assert list(dates) == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-17"),
        pd.Timestamp("2020-01-24"),
        pd.Timestamp("2020-01-31"),
    ]
    history = pd.DataFrame(
        {
            "ticker": ["X"] * 5 + ["Y"] * 5,
            "entry_date": list(dates) + list(dates),
            "return_pct": [10.0, np.nan, 20.0, np.nan, np.nan]
            + [1.0, 1.0, 1.0, 1.0, 1.0],
            "vol_gap": [np.nan, 0.10, 0.20, 0.30, 0.40]
            + [0.05, 0.05, 0.05, 0.05, 0.05],
        }
    )
    feature_date = dates[-1]
    context = FeatureDataContext(straddle_history=history)

    mom = MomentumCalculator(windows=[(4, 0)], min_periods=mom_min).calculate_bulk(
        context, start_date=dates[0], end_date=feature_date
    )
    cvg = CVGCalculator(windows=[(4, 0)], min_periods=cvg_min).calculate_bulk(
        context, start_date=dates[0], end_date=feature_date
    )
    x_mom = mom.loc[(mom["ticker"] == "X") & (mom["date"] == feature_date)].iloc[0]
    x_cvg = cvg.loc[(cvg["ticker"] == "X") & (cvg["date"] == feature_date)].iloc[0]

    assert x_mom["mom_4_0_count"] == 2
    assert x_mom["mom_4_0_mean"] == pytest.approx(15.0)
    assert x_cvg["cvg_count_4_0"] == 4
    assert x_cvg["cgap_4_0"] == pytest.approx(0.375)
    assert x_cvg["pct_pos_4_0"] == pytest.approx(1.0)
    assert x_cvg["pct_neg_4_0"] == pytest.approx(0.0)
    assert x_cvg["dvg_4_0"] == pytest.approx(-1.0)
    assert x_cvg["cvg_4_0"] == pytest.approx(2.0)
