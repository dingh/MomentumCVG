"""
Contract: S5 Phase 1 — step5_select_and_size SELECTION.

Scope (Sprint 003 Deliverable 2 / Phase 2): per-side cap + rank extracted from
SurfaceRunner._select_size_and_settle into a pure function on S4 output.
Sizing (`quantity`) and simulate (S7 settle) are later phases and are NOT
asserted here.

See docs/surface_engine_data_contract.md § S5 and
docs/surface_engine_portfolio_metrics_design.md § S5 Phase 1.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.pipeline import (
    EXCLUSION_EARNINGS,
    EXCLUSION_INVALID_MAX_LOSS,
    EXCLUSION_MAX_NAMES_CAP,
    EXCLUSION_NO_STRUCTURE,
    step5_select_and_size,
)
from tests.contract.conftest import make_contract_config


# ---------------------------------------------------------------------------
# Synthetic S4-output builder (step5 is a pure function on S4 output, so we
# hand-build the post-S3+S4 frame for fully auditable cap / rank cases).
# ---------------------------------------------------------------------------

def _s4_row(
    ticker: str,
    direction: str,
    signal_rank_pct: float,
    *,
    structure_ok: bool = True,
    had_earnings_nearby: bool = False,
    max_loss_per_share: float = 2.0,
    instrument_type: str = "short_ironfly",
) -> dict:
    return dict(
        ticker=ticker,
        direction=direction,
        signal_score=signal_rank_pct,
        signal_rank_pct=signal_rank_pct,
        cvg_score=1.0,
        cvg_rank_pct=1.0,
        structure_ok=structure_ok,
        had_earnings_nearby=had_earnings_nearby,
        instrument_type=instrument_type,
        max_loss_per_share=max_loss_per_share,
    )


def _s4_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema / annotation
# ---------------------------------------------------------------------------

def test_adds_inclusion_columns_on_every_row():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    assert "included_in_portfolio" in out.columns
    assert "exclusion_reason" in out.columns
    assert len(out) == len(structures)
    # Every row is annotated (no NaN inclusion flags).
    assert out["included_in_portfolio"].notna().all()


def test_preserves_input_columns():
    structures = _s4_frame([_s4_row("L1", "long", 0.95)])
    out = step5_select_and_size(None, structures, make_contract_config())
    for col in structures.columns:
        assert col in out.columns


def test_pure_function_does_not_mutate_input():
    structures = _s4_frame([_s4_row("L1", "long", 0.95), _s4_row("S1", "short", 0.05)])
    snapshot = structures.copy(deep=True)
    step5_select_and_size(None, structures, make_contract_config())
    pd.testing.assert_frame_equal(structures, snapshot)


def test_empty_structures_returns_empty_with_columns():
    out = step5_select_and_size(None, pd.DataFrame(), make_contract_config())
    assert out.empty
    assert "included_in_portfolio" in out.columns
    assert "exclusion_reason" in out.columns


# ---------------------------------------------------------------------------
# Eligibility (reads S3/S4 flags; does not re-run upstream filters)
# ---------------------------------------------------------------------------

def test_eligible_rows_included():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    for tkr in ("L1", "S1"):
        row = out[out["ticker"] == tkr].iloc[0]
        assert bool(row["included_in_portfolio"]) is True
        assert row["exclusion_reason"] is None


def test_no_tradeable_structure_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    bad = out[out["ticker"] == "BAD"].iloc[0]
    assert bool(bad["included_in_portfolio"]) is False
    assert bad["exclusion_reason"] == EXCLUSION_NO_STRUCTURE


def test_earnings_exclusion():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("S1", "short", 0.05, had_earnings_nearby=True),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    short = out[out["ticker"] == "S1"].iloc[0]
    assert bool(short["included_in_portfolio"]) is False
    assert short["exclusion_reason"] == EXCLUSION_EARNINGS


def test_no_structure_takes_priority_over_earnings():
    # A row that is both structure_ok==False AND earnings-flagged → no_tradeable_structure.
    structures = _s4_frame(
        [
            _s4_row(
                "BAD", "short", 0.05,
                structure_ok=False, had_earnings_nearby=True,
                max_loss_per_share=float("nan"),
            ),
        ]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    bad = out.iloc[0]
    assert bad["exclusion_reason"] == EXCLUSION_NO_STRUCTURE


# ---------------------------------------------------------------------------
# Per-side cap + rank (decision 003: independent long / short pools)
# ---------------------------------------------------------------------------

def test_per_side_cap_honored_independently():
    # 3 longs + 3 shorts, cap = 2 per side ⇒ 4 included (cap is per side, not global).
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_mid", "long", 0.80),
            _s4_row("L_lo", "long", 0.70),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_mid", "short", 0.20),
            _s4_row("S_hi", "short", 0.30),
        ]
    )
    cfg = make_contract_config(max_names_per_side=2)
    out = step5_select_and_size(None, structures, cfg)

    included = set(out.loc[out["included_in_portfolio"], "ticker"])
    assert included == {"L_hi", "L_mid", "S_lo", "S_mid"}
    assert int(out["included_in_portfolio"].sum()) == 4


def test_overflow_marked_max_names_cap():
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_mid", "long", 0.80),
            _s4_row("L_lo", "long", 0.70),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_mid", "short", 0.20),
            _s4_row("S_hi", "short", 0.30),
        ]
    )
    cfg = make_contract_config(max_names_per_side=2)
    out = step5_select_and_size(None, structures, cfg)

    # Long overflow = lowest rank; short overflow = highest rank.
    assert out.loc[out["ticker"] == "L_lo", "exclusion_reason"].iloc[0] == EXCLUSION_MAX_NAMES_CAP
    assert out.loc[out["ticker"] == "S_hi", "exclusion_reason"].iloc[0] == EXCLUSION_MAX_NAMES_CAP


def test_rank_direction_long_descending_short_ascending():
    # cap = 1 ⇒ keep best signal per side: long = highest rank, short = lowest rank.
    structures = _s4_frame(
        [
            _s4_row("L_hi", "long", 0.90),
            _s4_row("L_lo", "long", 0.60),
            _s4_row("S_lo", "short", 0.10),
            _s4_row("S_hi", "short", 0.40),
        ]
    )
    cfg = make_contract_config(max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)

    included = set(out.loc[out["included_in_portfolio"], "ticker"])
    assert included == {"L_hi", "S_lo"}


def test_excluded_rows_not_counted_against_cap():
    # An earnings-excluded long should not consume a long slot; the next eligible
    # long is still included under cap = 1.
    structures = _s4_frame(
        [
            _s4_row("L_top_earn", "long", 0.95, had_earnings_nearby=True),
            _s4_row("L_next", "long", 0.85),
        ]
    )
    cfg = make_contract_config(max_names_per_side=1)
    out = step5_select_and_size(None, structures, cfg)

    assert out.loc[out["ticker"] == "L_top_earn", "exclusion_reason"].iloc[0] == EXCLUSION_EARNINGS
    assert bool(out.loc[out["ticker"] == "L_next", "included_in_portfolio"].iloc[0]) is True


# ---------------------------------------------------------------------------
# invalid_max_loss (sizing-eligibility reject at the selection boundary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0.0, -1.0, None, float("nan")])
def test_invalid_max_loss_excludes_selected_row(bad_value):
    structures = _s4_frame([_s4_row("S1", "short", 0.05, max_loss_per_share=bad_value)])
    out = step5_select_and_size(None, structures, make_contract_config())
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is False
    assert row["exclusion_reason"] == EXCLUSION_INVALID_MAX_LOSS


def test_long_straddle_positive_premium_not_invalid():
    # Long straddle carries max_loss_per_share = premium paid (> 0) ⇒ stays included.
    structures = _s4_frame(
        [_s4_row("L1", "long", 0.95, instrument_type="long_straddle", max_loss_per_share=4.0)]
    )
    out = step5_select_and_size(None, structures, make_contract_config())
    row = out.iloc[0]
    assert bool(row["included_in_portfolio"]) is True
    assert row["exclusion_reason"] is None


# ---------------------------------------------------------------------------
# Diagnostics toggle
# ---------------------------------------------------------------------------

def test_include_diagnostics_false_drops_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    cfg = make_contract_config(include_diagnostics=False)
    out = step5_select_and_size(None, structures, cfg)
    assert set(out["ticker"]) == {"L1"}
    assert out["included_in_portfolio"].all()


def test_include_diagnostics_true_keeps_excluded():
    structures = _s4_frame(
        [
            _s4_row("L1", "long", 0.95),
            _s4_row("BAD", "short", 0.05, structure_ok=False, max_loss_per_share=float("nan")),
        ]
    )
    cfg = make_contract_config(include_diagnostics=True)
    out = step5_select_and_size(None, structures, cfg)
    assert set(out["ticker"]) == {"L1", "BAD"}
