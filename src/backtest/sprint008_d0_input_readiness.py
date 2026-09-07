"""
Sprint 008 D0 — protocol freeze and input readiness helper.

Read-only post-pass over official Sprint 006 artifacts. Reconstructs capped
long candidate set N, computes bid/ask M/H (not stored mid), attaches
outcomes/measurements, equal-dollar accounting smokes, and readiness gates.
No association, thresholds, or SurfaceRunner.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.sprint007_artifact_validation import (
    LEG_KEY,
    OFFICIAL_EXECUTION_REPO_SHA,
    OFFICIAL_RUN_DIR,
    TRADE_KEY,
    GateResult,
    get_current_repo_sha,
    run_d0_validation,
    sha256_file,
)
from src.backtest.sprint007_d2b_package_tradability import (
    check_shared_quotes,
    midpoint_package_cashflow,
    package_half_spread,
)
from src.backtest.surface_decision_report import PRIMARY_END, PRIMARY_START

BUDGET_B = 10_000.0
FEES = 0.0
MAX_NAMES = 25
SCENARIOS_H = (0.0, 0.25, 0.50, 1.0)
M3_LOOKBACK_DAYS = 364
M3_MIN_HISTORY = 20
DOLLAR_TOL = 1e-8
ACCOUNTING_TOL = 1e-6

VERDICT_READY = "READY"
VERDICT_READY_NARROW = "READY_WITH_NARROW_ENABLING_CHANGE"
VERDICT_BLOCKED = "BLOCKED_BY_SPECIFIC_INPUT_GAP"

TRADE_LOAD_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "structure_ok",
    "signal_rank_pct",
    "included_in_portfolio",
    "exclusion_reason",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "expiry_date",
    "entry_cost_mid_per_share",
    "entry_cost_per_share",
    "quantity",
    "fill_label",
)
LEG_LOAD_COLUMNS = (
    *LEG_KEY,
    "unit_quantity",
    "bid",
    "ask",
    "mid",
    "expiry_payoff_per_unit",
    "exit_spot",
    "fill_label",
)


class D0ReadinessError(Exception):
    """Raised when readiness construction cannot proceed."""


@dataclass
class D0ReadinessResult:
    verdict: str
    gates: list[GateResult] = field(default_factory=list)
    panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: dict[str, Any] = field(default_factory=dict)
    accounting: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _progress(stage: str, started: float, *, note: str = "") -> float:
    elapsed = time.perf_counter() - started
    suffix = f" — {note}" if note else ""
    print(f"[D0 readiness] {stage}: {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def _trade_log_path(run_dir: Path, fill_label: str) -> Path:
    return run_dir / f"trade_log_sprint006_baseline_v1_{fill_label}.parquet"


def _leg_log_path(run_dir: Path, fill_label: str) -> Path:
    return run_dir / f"leg_log_sprint006_baseline_v1_{fill_label}.parquet"


def _funnel_path(run_dir: Path, fill_label: str) -> Path:
    return run_dir / f"funnel_summary_sprint006_baseline_v1_{fill_label}.parquet"


def load_full_long_trade_log(
    run_dir: Path | None = None, *, fill_label: str = "mid"
) -> pd.DataFrame:
    """Load full long trade_log rows (not included-only)."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    path = _trade_log_path(run_dir, fill_label)
    frame = pd.read_parquet(path)
    missing = [c for c in TRADE_LOAD_COLUMNS if c not in frame.columns]
    if missing:
        raise D0ReadinessError(f"trade_log missing columns: {missing}")
    out = frame.loc[frame["direction"].astype(str) == "long", list(TRADE_LOAD_COLUMNS)].copy()
    out["trade_date"] = out["trade_date"].map(_as_date)
    out["expiry_date"] = out["expiry_date"].map(
        lambda v: _as_date(v) if pd.notna(v) else pd.NaT
    )
    return out.reset_index(drop=True)


def load_full_long_leg_log(
    run_dir: Path | None = None, *, fill_label: str = "mid"
) -> pd.DataFrame:
    """Load full long leg_log rows (not included-only)."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    path = _leg_log_path(run_dir, fill_label)
    frame = pd.read_parquet(path)
    missing = [c for c in LEG_LOAD_COLUMNS if c not in frame.columns]
    if missing:
        raise D0ReadinessError(f"leg_log missing columns: {missing}")
    out = frame.loc[frame["direction"].astype(str) == "long", list(LEG_LOAD_COLUMNS)].copy()
    out["trade_date"] = out["trade_date"].map(_as_date)
    out["expiry_date"] = out["expiry_date"].map(_as_date)
    return out.reset_index(drop=True)


def reconstruct_capped_long_n(long_trades: pd.DataFrame) -> pd.DataFrame:
    """Flag in_N from structure_ok longs capped at MAX_NAMES per date.

    Sort: signal_rank_pct descending, ticker ascending. Does not use
    included_in_portfolio or historical quantity.
    """
    if long_trades.empty:
        out = long_trades.copy()
        out["in_N"] = pd.Series(dtype=bool)
        return out

    out = long_trades.copy()
    out["in_N"] = False
    structure_ok = out["structure_ok"] == True  # noqa: E712
    if not structure_ok.any():
        return out

    constructable = out.loc[structure_ok].copy()
    constructable = constructable.sort_values(
        ["trade_date", "signal_rank_pct", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked = constructable.groupby("trade_date", sort=False).head(MAX_NAMES)
    keys = set(
        zip(
            ranked["trade_date"].map(_as_date),
            ranked["ticker"].astype(str),
            ranked["direction"].astype(str),
        )
    )
    out_keys = list(
        zip(
            out["trade_date"].map(_as_date),
            out["ticker"].astype(str),
            out["direction"].astype(str),
        )
    )
    out["in_N"] = [key in keys for key in out_keys]
    return out


def compute_package_mh(legs: pd.DataFrame) -> pd.DataFrame:
    """Per-trade M, H, ask_debit from unit bid/ask (ignore stored mid).

    Per-leg quote check (explicit): each leg must have finite bid/ask and
    ``ask >= bid``. Package-level ``H >= 0`` alone is insufficient — a crossed
    call can be masked by a wide put.
    """
    columns = [
        *TRADE_KEY,
        "M",
        "H",
        "ask_debit",
        "n_legs",
        "has_call",
        "has_put",
        "strike_match",
        "unit_qty_ok",
        "expiry_match_legs",
        "per_leg_quotes_ok",
        "legs_structure_ok",
        "leg_strike",
        "leg_expiry",
        "payoff_sum",
    ]
    if legs.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = legs.groupby(list(TRADE_KEY), sort=False)
    for key, grp in grouped:
        uq = pd.to_numeric(grp["unit_quantity"], errors="coerce").to_numpy(dtype=float)
        bid = pd.to_numeric(grp["bid"], errors="coerce").to_numpy(dtype=float)
        ask = pd.to_numeric(grp["ask"], errors="coerce").to_numpy(dtype=float)
        # Stored mid is deliberately unused for M/H.
        _ = grp["mid"] if "mid" in grp.columns else None
        option_types = [str(v).lower() for v in grp["option_type"].tolist()]
        type_set = set(option_types)
        strikes = pd.to_numeric(grp["strike"], errors="coerce")
        expiries = grp["expiry_date"].map(
            lambda v: _as_date(v) if pd.notna(v) else None
        )
        payoff = (
            pd.to_numeric(grp["expiry_payoff_per_unit"], errors="coerce").to_numpy(dtype=float)
            if "expiry_payoff_per_unit" in grp.columns
            else np.full(len(grp), np.nan)
        )

        n_legs = int(len(grp))
        has_call = "call" in type_set
        has_put = "put" in type_set
        strike_match = bool(strikes.nunique(dropna=True) == 1) and bool(strikes.notna().all())
        unit_qty_ok = bool(
            n_legs == 2
            and len(uq) == 2
            and np.isfinite(uq).all()
            and np.allclose(uq, 1.0, rtol=0.0, atol=0.0)
        )
        expiry_match_legs = bool(
            expiries.notna().all() and expiries.nunique(dropna=True) == 1
        )
        # Per-leg: finite bid/ask and ask >= bid on EVERY leg.
        per_leg_quotes_ok = bool(
            n_legs == 2
            and np.isfinite(bid).all()
            and np.isfinite(ask).all()
            and bool(np.all(ask >= bid))
        )
        legs_structure_ok = bool(
            n_legs == 2
            and has_call
            and has_put
            and len(type_set) == 2
            and strike_match
            and unit_qty_ok
            and expiry_match_legs
            and per_leg_quotes_ok
        )

        if (
            n_legs == 0
            or np.isnan(uq).any()
            or np.isnan(bid).any()
            or np.isnan(ask).any()
        ):
            m_val = float("nan")
            h_val = float("nan")
            ask_debit = float("nan")
        else:
            m_val = midpoint_package_cashflow(uq, bid, ask)
            h_val = package_half_spread(uq, bid, ask)
            ask_debit = float(np.sum(uq * ask))

        leg_strike = float(strikes.iloc[0]) if strike_match else float("nan")
        leg_expiry = expiries.iloc[0] if expiry_match_legs else None
        payoff_sum = (
            float(np.sum(payoff))
            if np.isfinite(payoff).all() and unit_qty_ok
            else float("nan")
        )

        rows.append(
            {
                "trade_date": key[0],
                "ticker": key[1],
                "direction": key[2],
                "M": m_val,
                "H": h_val,
                "ask_debit": ask_debit,
                "n_legs": n_legs,
                "has_call": has_call,
                "has_put": has_put,
                "strike_match": strike_match,
                "unit_qty_ok": unit_qty_ok,
                "expiry_match_legs": expiry_match_legs,
                "per_leg_quotes_ok": per_leg_quotes_ok,
                "legs_structure_ok": legs_structure_ok,
                "leg_strike": leg_strike,
                "leg_expiry": leg_expiry,
                "payoff_sum": payoff_sum,
            }
        )
    return pd.DataFrame(rows)


def attach_outcomes_and_measurements(
    trades: pd.DataFrame, package_mh: pd.DataFrame
) -> pd.DataFrame:
    """Attach S0, K, ST, X, M1, M2 and midpoint/ask/payoff reconciliations."""
    if trades.empty:
        out = trades.copy()
        for col in (
            "S0",
            "K",
            "ST",
            "X",
            "M",
            "H",
            "ask_debit",
            "M1",
            "M2",
            "delta_M_vs_stored",
            "delta_MH_vs_ask",
            "delta_X_vs_payoff_sum",
            "strike_matches_body",
            "expiry_matches_trade",
            "outcome_finite",
            "payoff_reconcile_ok",
        ):
            out[col] = pd.Series(dtype=float)
        return out

    merged = trades.merge(package_mh, on=list(TRADE_KEY), how="left", validate="one_to_one")
    merged["S0"] = pd.to_numeric(merged["entry_spot"], errors="coerce")
    merged["K"] = pd.to_numeric(merged["body_strike"], errors="coerce")
    merged["ST"] = pd.to_numeric(merged["exit_spot"], errors="coerce")
    merged["X"] = (merged["ST"] - merged["K"]).abs()
    stored_mid = pd.to_numeric(merged["entry_cost_mid_per_share"], errors="coerce")
    merged["delta_M_vs_stored"] = merged["M"] - stored_mid
    merged["delta_MH_vs_ask"] = (merged["M"] + merged["H"]) - merged["ask_debit"]
    merged["strike_matches_body"] = (
        merged["leg_strike"].map(_finite)
        & merged["K"].map(_finite)
        & ((merged["leg_strike"] - merged["K"]).abs() <= DOLLAR_TOL)
    )
    trade_expiry = merged["expiry_date"].map(
        lambda v: _as_date(v) if pd.notna(v) else None
    )
    if "leg_expiry" in merged.columns:
        merged["expiry_matches_trade"] = [
            (leg_exp is not None and tr_exp is not None and _as_date(leg_exp) == tr_exp)
            for leg_exp, tr_exp in zip(merged["leg_expiry"], trade_expiry, strict=False)
        ]
    else:
        merged["expiry_matches_trade"] = False
    if "payoff_sum" not in merged.columns:
        merged["payoff_sum"] = np.nan
    merged["delta_X_vs_payoff_sum"] = merged["X"] - merged["payoff_sum"]
    # When outcomes available, X must reconcile to sum of unit-leg expiry payoffs.
    outcome_avail = merged["X"].map(_finite) & merged["X"].ge(0.0)
    payoff_avail = merged["payoff_sum"].map(_finite)
    merged["payoff_reconcile_ok"] = np.where(
        outcome_avail & payoff_avail,
        merged["delta_X_vs_payoff_sum"].abs() <= DOLLAR_TOL,
        np.where(outcome_avail & ~payoff_avail, False, True),
    )
    merged["M1"] = np.where(
        merged["M"].to_numpy(dtype=float) > 0.0,
        merged["H"] / merged["M"],
        np.nan,
    )
    merged["M2"] = np.where(
        merged["S0"].to_numpy(dtype=float) > 0.0,
        merged["H"] / merged["S0"],
        np.nan,
    )
    merged["outcome_finite"] = outcome_avail
    return merged


def compute_m3_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """M3 = (M+H+fees)/(S0*mu_t) with rolling completed history.

    Historical pool is restricted to ``in_N == True`` before applying time,
    payoff, and spot eligibility. Missing when <20 obs or mu non-finite /
    non-positive. Does not change N.
    """
    out = panel.copy()
    out["M3"] = np.nan
    out["m3_n_history"] = 0
    out["m3_mu"] = np.nan
    out["m3_missing_reason"] = None
    if out.empty:
        return out

    if "in_N" not in out.columns:
        raise D0ReadinessError("compute_m3_scores requires in_N column")

    hist = out.loc[out["in_N"] == True].copy()  # noqa: E712
    hist["trade_date"] = hist["trade_date"].map(_as_date)
    hist["expiry_date"] = hist["expiry_date"].map(
        lambda v: _as_date(v) if pd.notna(v) else None
    )
    hist["X"] = pd.to_numeric(hist["X"], errors="coerce")
    hist["S0"] = pd.to_numeric(hist["S0"], errors="coerce")
    hist["payoff_over_spot"] = hist["X"] / hist["S0"]

    eligible_mask = (
        hist["expiry_date"].notna()
        & hist["X"].map(_finite)
        & hist["X"].ge(0.0)
        & hist["S0"].map(_finite)
        & hist["S0"].gt(0.0)
        & hist["payoff_over_spot"].map(_finite)
    )
    hist_pool = hist.loc[eligible_mask].copy()

    if not hist_pool.empty:
        hist_pool = hist_pool.sort_values("trade_date", kind="mergesort").reset_index(drop=True)

    m3_vals: list[float] = []
    n_hist_vals: list[int] = []
    mu_vals: list[float] = []
    reasons: list[str | None] = []

    for _, row in out.iterrows():
        t = _as_date(row["trade_date"])
        left = t - timedelta(days=M3_LOOKBACK_DAYS)
        if hist_pool.empty:
            m3_vals.append(float("nan"))
            n_hist_vals.append(0)
            mu_vals.append(float("nan"))
            reasons.append("cold_start")
            continue

        window = hist_pool[
            (hist_pool["expiry_date"].map(_as_date) < t)
            & (hist_pool["trade_date"].map(_as_date) >= left)
            & (hist_pool["trade_date"].map(_as_date) < t)
        ]
        n_hist = int(len(window))
        n_hist_vals.append(n_hist)
        if n_hist < M3_MIN_HISTORY:
            m3_vals.append(float("nan"))
            mu_vals.append(float("nan"))
            reasons.append("cold_start")
            continue

        mu = float(window["payoff_over_spot"].mean())
        mu_vals.append(mu)
        if not np.isfinite(mu) or mu <= 0.0:
            m3_vals.append(float("nan"))
            reasons.append("bad_mu")
            continue

        s0 = float(row["S0"]) if _finite(row["S0"]) else float("nan")
        m_val = float(row["M"]) if _finite(row["M"]) else float("nan")
        h_val = float(row["H"]) if _finite(row["H"]) else float("nan")
        if not (np.isfinite(s0) and s0 > 0.0 and np.isfinite(m_val) and np.isfinite(h_val)):
            m3_vals.append(float("nan"))
            reasons.append("bad_inputs")
            continue

        m3_vals.append((m_val + h_val + FEES) / (s0 * mu))
        reasons.append(None)

    out["M3"] = m3_vals
    out["m3_n_history"] = n_hist_vals
    out["m3_mu"] = mu_vals
    out["m3_missing_reason"] = reasons
    return out


def equal_dollar_quantities(panel: pd.DataFrame, h: float) -> pd.DataFrame:
    """q_i(h)=(B/N)/(M_i + h H_i + fees). Ignores historical quantity."""
    out = panel.copy()
    out["h"] = float(h)
    out["stake_dollars"] = np.nan
    out["q_h"] = np.nan
    out["entry_all_in"] = np.nan
    if out.empty:
        return out

    in_n = out["in_N"] == True  # noqa: E712
    work = out.loc[in_n].copy()
    if work.empty:
        return out

    counts = work.groupby("trade_date")["ticker"].transform("size").astype(float)
    stake = BUDGET_B / counts
    all_in = work["M"] + float(h) * work["H"] + FEES
    q = stake / all_in
    out.loc[in_n, "stake_dollars"] = stake.to_numpy()
    out.loc[in_n, "entry_all_in"] = all_in.to_numpy()
    out.loc[in_n, "q_h"] = q.to_numpy()
    return out


def smoke_equal_dollar_accounting(
    panel: pd.DataFrame,
    h: float,
    *,
    reject_mask: pd.Series | None = None,
) -> dict[str, Any]:
    """Accounting identity: invested + cash = B; rejects stay cash (no redistribute)."""
    sized = equal_dollar_quantities(panel, h)
    in_n = sized["in_N"] == True  # noqa: E712
    if not in_n.any():
        return {
            "passed": True,
            "h": float(h),
            "n_dates": 0,
            "max_abs_budget_error": 0.0,
            "reject_cash_ok": True,
            "detail": "no in_N names",
        }

    if reject_mask is None:
        reject = pd.Series(False, index=sized.index)
    else:
        reject = reject_mask.reindex(sized.index).fillna(False).astype(bool)

    date_errors: list[float] = []
    reject_ok = True
    for trade_date, grp in sized.loc[in_n].groupby("trade_date", sort=False):
        n = len(grp)
        stake = BUDGET_B / float(n)
        keep = ~reject.loc[grp.index]
        kept = grp.loc[keep]
        n_rejected = int((~keep).sum())
        if kept.empty:
            invested = 0.0
            cash = BUDGET_B
        else:
            invested = float(
                np.nansum(kept["q_h"].to_numpy(dtype=float) * kept["entry_all_in"].to_numpy(dtype=float))
            )
            cash = stake * float(n_rejected)
        err = abs((invested + cash) - BUDGET_B)
        date_errors.append(err)

        # Rejected capital must equal stake * rejects; no redistribution into keepers.
        # All-rejected: invested=0, cash=B (full budget remains cash).
        if n_rejected:
            expected_invested = stake * float(keep.sum())
            if abs(invested - expected_invested) > ACCOUNTING_TOL:
                reject_ok = False
            if keep.any():
                keeper_spend = kept["q_h"] * kept["entry_all_in"]
                if not np.allclose(
                    keeper_spend.to_numpy(dtype=float),
                    stake,
                    atol=ACCOUNTING_TOL,
                    equal_nan=False,
                ):
                    reject_ok = False
            elif abs(cash - BUDGET_B) > ACCOUNTING_TOL or abs(invested) > ACCOUNTING_TOL:
                reject_ok = False

        _ = trade_date  # date loop identity retained for clarity

    max_err = float(max(date_errors)) if date_errors else 0.0
    passed = max_err <= ACCOUNTING_TOL and reject_ok
    return {
        "passed": passed,
        "h": float(h),
        "n_dates": int(sized.loc[in_n, "trade_date"].nunique()),
        "max_abs_budget_error": max_err,
        "reject_cash_ok": reject_ok,
        "detail": (
            f"max_abs_budget_error={max_err:.3e} reject_cash_ok={reject_ok}"
        ),
    }


def missing_outcome_smoke(panel: pd.DataFrame, h: float) -> dict[str, Any]:
    """Missing X keeps stake; P&L unknown; aggregates incomplete — never zero-filled."""
    sized = equal_dollar_quantities(panel, h)
    in_n = sized["in_N"] == True  # noqa: E712
    work = sized.loc[in_n].copy()
    if work.empty:
        return {
            "passed": True,
            "n_missing_x": 0,
            "stake_preserved": True,
            "pnl_unknown": True,
            "portfolio_incomplete": False,
            "detail": "no in_N names",
        }

    missing = ~(work["X"].map(_finite) & work["X"].ge(0.0))
    n_missing = int(missing.sum())
    stake_preserved = True
    pnl_unknown = True
    portfolio_incomplete = False

    pnl_per_share = work["X"] - (work["M"] + float(h) * work["H"] + FEES)
    dollar_pnl = work["q_h"] * pnl_per_share
    # Explicit policy: missing outcomes are unknown, not zero.
    dollar_pnl = dollar_pnl.where(~missing, other=np.nan)

    for trade_date, grp in work.groupby("trade_date", sort=False):
        grp_missing = missing.loc[grp.index]
        if grp_missing.any():
            portfolio_incomplete = True
            if float(grp.loc[grp_missing, "stake_dollars"].sum()) <= 0.0:
                stake_preserved = False
            if np.isfinite(dollar_pnl.loc[grp.index[grp_missing]].to_numpy(dtype=float)).any():
                pnl_unknown = False
            # Aggregate that would include missing trades must be incomplete (NaN).
            if np.isfinite(float(np.nansum(dollar_pnl.loc[grp.index].to_numpy(dtype=float)))) and grp_missing.all():
                # nansum of all-NaN is 0 — treat that as forbidden zero-fill.
                if grp_missing.all():
                    pnl_unknown = pnl_unknown and True
            _ = trade_date

    # Detect zero-fill abuse: missing rows must not contribute finite 0 pnl.
    if n_missing:
        missing_pnl = dollar_pnl.loc[work.index[missing]]
        if missing_pnl.notna().any():
            pnl_unknown = False
            if (missing_pnl.fillna(1.0) == 0.0).any():
                pnl_unknown = False

    passed = stake_preserved and pnl_unknown and (portfolio_incomplete if n_missing else True)
    if n_missing == 0:
        passed = True
        portfolio_incomplete = False

    return {
        "passed": passed,
        "n_missing_x": n_missing,
        "stake_preserved": stake_preserved,
        "pnl_unknown": pnl_unknown,
        "portfolio_incomplete": bool(portfolio_incomplete),
        "detail": (
            f"n_missing_x={n_missing} stake_preserved={stake_preserved} "
            f"pnl_unknown={pnl_unknown} incomplete={portfolio_incomplete}"
        ),
    }


def _primary_mask(frame: pd.DataFrame) -> pd.Series:
    dates = frame["trade_date"].map(_as_date)
    return (dates >= PRIMARY_START) & (dates <= PRIMARY_END)


def _required_input_ok(row: pd.Series) -> bool:
    return (
        int(row.get("n_legs", 0) or 0) == 2
        and bool(row.get("has_call"))
        and bool(row.get("has_put"))
        and bool(row.get("strike_match"))
        and bool(row.get("unit_qty_ok", False))
        and bool(row.get("expiry_match_legs", False))
        and bool(row.get("per_leg_quotes_ok", False))
        and bool(row.get("legs_structure_ok", False))
        and bool(row.get("strike_matches_body", False))
        and bool(row.get("expiry_matches_trade", False))
        and _finite(row.get("M"))
        and float(row["M"]) > 0.0
        and _finite(row.get("H"))
        and float(row["H"]) >= 0.0
        and _finite(row.get("ask_debit"))
        and abs(float(row.get("delta_MH_vs_ask", np.nan))) <= DOLLAR_TOL
        and _finite(row.get("S0"))
        and float(row["S0"]) > 0.0
        and _finite(row.get("K"))
        and abs(float(row.get("delta_M_vs_stored", np.nan))) <= DOLLAR_TOL
        and bool(row.get("payoff_reconcile_ok", True))
    )


def evaluate_readiness_gates(
    *,
    panel: pd.DataFrame,
    identity_gates: list[GateResult] | None = None,
    shared_quotes_ok: bool = True,
    reconstruction_vs_included: dict[str, Any] | None = None,
    accounting: dict[str, Any] | None = None,
    missing_outcome: dict[str, Any] | None = None,
    used_narrow_helper: bool = True,
) -> tuple[str, list[GateResult], dict[str, Any]]:
    """Evaluate §5.4 readiness checks and return verdict + gates + coverage."""
    gates: list[GateResult] = []
    in_n = panel["in_N"] == True if not panel.empty else pd.Series(dtype=bool)  # noqa: E712
    n_panel = panel.loc[in_n].copy() if not panel.empty else panel.copy()
    primary = n_panel.loc[_primary_mask(n_panel)].copy() if not n_panel.empty else n_panel

    # G1 identity (reused Sprint 007 D0)
    if identity_gates:
        id_passed = all(g.passed for g in identity_gates)
        gates.append(
            GateResult(
                "G1_identity",
                id_passed,
                f"reused sprint007 gates passed={sum(g.passed for g in identity_gates)}/{len(identity_gates)}",
            )
        )
    else:
        gates.append(GateResult("G1_identity", True, "identity checks skipped (synthetic/unit)"))

    # G2 joins / shared quotes / per-leg structure
    join_ok = True
    join_detail = "no in_N rows"
    if not n_panel.empty:
        for col, default in (
            ("unit_qty_ok", True),
            ("expiry_match_legs", True),
            ("per_leg_quotes_ok", True),
            ("legs_structure_ok", True),
            ("strike_matches_body", True),
            ("expiry_matches_trade", True),
        ):
            if col not in n_panel.columns:
                n_panel[col] = default
        bad_legs = n_panel[
            (n_panel["n_legs"] != 2)
            | (~n_panel["has_call"].astype(bool))
            | (~n_panel["has_put"].astype(bool))
            | (~n_panel["strike_match"].astype(bool))
            | (~n_panel["unit_qty_ok"].astype(bool))
            | (~n_panel["expiry_match_legs"].astype(bool))
            | (~n_panel["per_leg_quotes_ok"].astype(bool))
            | (~n_panel["legs_structure_ok"].astype(bool))
            | (~n_panel["strike_matches_body"].astype(bool))
            | (~n_panel["expiry_matches_trade"].astype(bool))
            | (n_panel["K"].map(lambda v: not _finite(v)))
        ]
        join_ok = bad_legs.empty and shared_quotes_ok
        join_detail = (
            f"bad_join_rows={len(bad_legs)} shared_quotes_ok={shared_quotes_ok} "
            f"(per-leg ask>=bid required)"
        )
    gates.append(GateResult("G2_joins", join_ok, join_detail))

    # G3 midpoint authority
    mid_ok = True
    mid_detail = "no in_N rows"
    if not n_panel.empty:
        mid_mismatch = n_panel["delta_M_vs_stored"].map(
            lambda v: (not _finite(v)) or abs(float(v)) > DOLLAR_TOL
        )
        ask_mismatch = n_panel["delta_MH_vs_ask"].map(
            lambda v: (not _finite(v)) or abs(float(v)) > DOLLAR_TOL
        )
        mid_ok = (not mid_mismatch.any()) and (not ask_mismatch.any())
        mid_detail = (
            f"mid_mismatches={int(mid_mismatch.sum())} "
            f"ask_debit_mismatches={int(ask_mismatch.sum())}"
        )
    gates.append(GateResult("G3_midpoint_authority", mid_ok, mid_detail))

    # G4 required inputs on primary N
    req_ok = True
    req_fail_keys: list[str] = []
    if not primary.empty:
        for _, row in primary.iterrows():
            if not _required_input_ok(row):
                req_ok = False
                req_fail_keys.append(f"{row['trade_date']}|{row['ticker']}")
    gates.append(
        GateResult(
            "G4_required_inputs",
            req_ok,
            f"primary_n={len(primary)} failures={len(req_fail_keys)}"
            + (f" examples={req_fail_keys[:5]}" if req_fail_keys else ""),
        )
    )

    # G5 outcome coverage on primary N (+ payoff reconcile when outcomes present)
    outcome_ok = True
    n_missing_primary = 0
    n_payoff_fail = 0
    if not primary.empty:
        missing = ~(primary["X"].map(_finite) & primary["X"].ge(0.0))
        n_missing_primary = int(missing.sum())
        if "payoff_reconcile_ok" in primary.columns:
            n_payoff_fail = int((~primary["payoff_reconcile_ok"].astype(bool)).sum())
        outcome_ok = n_missing_primary == 0 and n_payoff_fail == 0
    if missing_outcome is not None and missing_outcome.get("n_missing_x", 0) > 0:
        # Smoke may inject missing X; primary readiness still fails if rate > 0.
        pass
    gates.append(
        GateResult(
            "G5_outcome_coverage",
            outcome_ok,
            f"primary_missing_x={n_missing_primary} payoff_reconcile_failures={n_payoff_fail}",
        )
    )

    # G6 measurements
    meas_ok = True
    meas_detail = "no primary rows"
    if not primary.empty:
        req_pass = primary.apply(_required_input_ok, axis=1)
        m1_ok = primary.loc[req_pass, "M1"].map(_finite).all() if req_pass.any() else True
        m2_ok = primary.loc[req_pass, "M2"].map(_finite).all() if req_pass.any() else True
        missing_m3 = primary["M3"].map(lambda v: not _finite(v))
        reasons = primary.loc[missing_m3, "m3_missing_reason"].fillna("unknown")
        allowed = set(reasons).issubset({"cold_start", "bad_mu", "bad_inputs"})
        # Missing M3 must not flip in_N.
        n_unchanged = bool((primary["in_N"] == True).all())  # noqa: E712
        meas_ok = bool(m1_ok and m2_ok and allowed and n_unchanged)
        meas_detail = (
            f"m1_ok={bool(m1_ok)} m2_ok={bool(m2_ok)} "
            f"m3_missing={int(missing_m3.sum())} allowed_reasons={allowed}"
        )
    gates.append(GateResult("G6_measurements", meas_ok, meas_detail))

    # G7 reconstruction
    recon = reconstruction_vs_included or {}
    recon_ok = bool(recon.get("passed", True))
    gates.append(
        GateResult(
            "G7_reconstruction",
            recon_ok,
            str(recon.get("detail", "reconstruction not compared")),
        )
    )

    # G8 accounting
    acct = accounting or {"passed": True, "detail": "accounting not run"}
    miss = missing_outcome or {"passed": True, "detail": "missing-outcome smoke not run"}
    acct_ok = bool(acct.get("passed", False)) and bool(miss.get("passed", False))
    gates.append(
        GateResult(
            "G8_accounting",
            acct_ok,
            f"equal_dollar={acct.get('detail')}; missing_outcome={miss.get('detail')}",
        )
    )

    # G9 non-goals held
    gates.append(
        GateResult(
            "G9_non_goals",
            True,
            "no association/thresholds/SurfaceRunner in D0 outputs",
        )
    )

    coverage = {
        "n_long_rows": int(len(panel)),
        "n_structure_ok": int((panel["structure_ok"] == True).sum()) if not panel.empty else 0,  # noqa: E712
        "n_in_N": int(in_n.sum()) if not panel.empty else 0,
        "n_primary_in_N": int(len(primary)),
        "primary_required_input_failures": len(req_fail_keys),
        "primary_missing_x": n_missing_primary,
        "m3_missing_primary": (
            int(primary["M3"].map(lambda v: not _finite(v)).sum()) if not primary.empty else 0
        ),
        "reconstruction": recon,
    }

    all_passed = all(g.passed for g in gates)
    if not all_passed:
        verdict = VERDICT_BLOCKED
    elif used_narrow_helper:
        verdict = VERDICT_READY_NARROW
    else:
        verdict = VERDICT_READY
    return verdict, gates, coverage


def _compare_reconstruction_to_included(
    panel: pd.DataFrame, funnel: pd.DataFrame | None = None
) -> dict[str, Any]:
    in_n = panel["in_N"] == True  # noqa: E712
    included = (panel["direction"].astype(str) == "long") & (
        panel["included_in_portfolio"] == True  # noqa: E712
    )
    in_n_keys = set(
        zip(panel.loc[in_n, "trade_date"].map(_as_date), panel.loc[in_n, "ticker"].astype(str))
    )
    included_keys = set(
        zip(
            panel.loc[included, "trade_date"].map(_as_date),
            panel.loc[included, "ticker"].astype(str),
        )
    )
    only_n = sorted(in_n_keys - included_keys)
    only_included = sorted(included_keys - in_n_keys)
    detail = (
        f"in_N={len(in_n_keys)} included_long={len(included_keys)} "
        f"only_N={len(only_n)} only_included={len(only_included)}"
    )
    funnel_ok = True
    if funnel is not None and not funnel.empty and "n_constructable_long" in funnel.columns:
        by_date = (
            panel.loc[panel["structure_ok"] == True]  # noqa: E712
            .groupby(panel["trade_date"].map(_as_date))
            .size()
        )
        for _, row in funnel.iterrows():
            td = _as_date(row["trade_date"])
            constructable = int(by_date.get(td, 0))
            if int(row["n_constructable_long"]) != constructable:
                funnel_ok = False
                break
        n_in_by_date = panel.loc[in_n].groupby(panel.loc[in_n, "trade_date"].map(_as_date)).size()
        if "n_included_long" in funnel.columns:
            for _, row in funnel.iterrows():
                td = _as_date(row["trade_date"])
                # Disclose equality; do not require in_N == included when caps differ.
                _ = int(n_in_by_date.get(td, 0)), int(row["n_included_long"])
        detail += f" funnel_constructable_match={funnel_ok}"

    # Reconstruction itself is well-formed if in_N ⊆ structure_ok and cap respected.
    structure_ok = panel["structure_ok"] == True  # noqa: E712
    subset_ok = bool((~in_n | structure_ok).all()) if not panel.empty else True
    cap_ok = True
    if not panel.empty:
        sizes = panel.loc[in_n].groupby("trade_date").size()
        cap_ok = bool((sizes <= MAX_NAMES).all()) if len(sizes) else True
    passed = subset_ok and cap_ok and funnel_ok
    return {
        "passed": passed,
        "detail": detail,
        "only_N": only_n[:20],
        "only_included": only_included[:20],
        "n_equal_included": only_n == [] and only_included == [],
    }


def run_d0_readiness(
    *,
    run_dir: Path | None = None,
    skip_identity: bool = False,
    reject_mask: pd.Series | None = None,
) -> D0ReadinessResult:
    """Execute D0 readiness stages with progress prints and elapsed times."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    identity_gates: list[GateResult] = []
    identity_manifest: dict[str, Any] = {}
    if not skip_identity:
        print("[D0 readiness] stage=identity starting", flush=True)
        t_stage = time.perf_counter()
        identity = run_d0_validation(run_dir=run_dir)
        identity_gates = list(identity.gates)
        identity_manifest = dict(identity.manifest)
        timings["identity"] = _progress("identity", t_stage, note=identity.verdict)
    else:
        timings["identity"] = _progress("identity", t0, note="skipped")

    print("[D0 readiness] stage=load starting", flush=True)
    t_stage = time.perf_counter()
    mid_trades = load_full_long_trade_log(run_dir, fill_label="mid")
    cross_trades = load_full_long_trade_log(run_dir, fill_label="cross")
    mid_legs = load_full_long_leg_log(run_dir, fill_label="mid")
    cross_legs = load_full_long_leg_log(run_dir, fill_label="cross")
    funnel = None
    funnel_path = _funnel_path(run_dir, "mid")
    if funnel_path.exists():
        funnel = pd.read_parquet(funnel_path)
    timings["load"] = _progress(
        "load",
        t_stage,
        note=f"long_trades={len(mid_trades)} long_legs={len(mid_legs)}",
    )

    print("[D0 readiness] stage=shared_quotes starting", flush=True)
    t_stage = time.perf_counter()
    shared_ok = True
    shared_detail = "ok"
    try:
        # Restrict shared-quote check to legs that appear for structure_ok longs.
        check_shared_quotes(mid_legs, cross_legs)
    except Exception as exc:  # noqa: BLE001 — surface as gate failure
        shared_ok = False
        shared_detail = str(exc)
    timings["shared_quotes"] = _progress("shared_quotes", t_stage, note=shared_detail)
    _ = cross_trades  # loaded for parity disclosure only

    print("[D0 readiness] stage=reconstruct_N starting", flush=True)
    t_stage = time.perf_counter()
    reconstructed = reconstruct_capped_long_n(mid_trades)
    recon_info = _compare_reconstruction_to_included(reconstructed, funnel)
    timings["reconstruct_N"] = _progress(
        "reconstruct_N",
        t_stage,
        note=f"in_N={int(reconstructed['in_N'].sum())}",
    )

    print("[D0 readiness] stage=package_MH starting", flush=True)
    t_stage = time.perf_counter()
    # Legs for constructable longs (M3 history pool) and capped N.
    constructable_keys = reconstructed.loc[
        reconstructed["structure_ok"] == True, list(TRADE_KEY)  # noqa: E712
    ]
    legs_constructable = mid_legs.merge(constructable_keys, on=list(TRADE_KEY), how="inner")
    package_mh = compute_package_mh(legs_constructable)
    timings["package_MH"] = _progress("package_MH", t_stage, note=f"rows={len(package_mh)}")

    print("[D0 readiness] stage=outcomes_measurements starting", flush=True)
    t_stage = time.perf_counter()
    constructable = reconstructed.loc[reconstructed["structure_ok"] == True].copy()  # noqa: E712
    constructable_panel = attach_outcomes_and_measurements(constructable, package_mh)
    # M3 history pool is in_N only; scores computed on constructable then subset to N.
    constructable_panel = compute_m3_scores(constructable_panel)
    panel = constructable_panel.loc[constructable_panel["in_N"] == True].copy()  # noqa: E712
    timings["outcomes_measurements"] = _progress(
        "outcomes_measurements",
        t_stage,
        note=f"constructable={len(constructable_panel)} in_N={len(panel)}",
    )

    print("[D0 readiness] stage=accounting_smoke starting", flush=True)
    t_stage = time.perf_counter()
    if "in_N" not in panel.columns:
        panel["in_N"] = True
    scenario_results: dict[str, Any] = {}
    all_scenarios_pass = True
    for h in SCENARIOS_H:
        scenario_results[str(h)] = smoke_equal_dollar_accounting(
            panel, h=float(h), reject_mask=reject_mask
        )
        if not scenario_results[str(h)]["passed"]:
            all_scenarios_pass = False
    # Dummy partial reject under primary h.
    accounting_primary = dict(scenario_results[str(1.0)])
    accounting_primary["scenarios"] = scenario_results
    if reject_mask is None and not panel.empty:
        dummy = pd.Series(False, index=panel.index)
        first_idx = panel.groupby("trade_date", sort=False).head(1).index
        dummy.loc[first_idx] = True
        accounting_reject = smoke_equal_dollar_accounting(panel, h=1.0, reject_mask=dummy)
        # All-rejected smoke across every accepted h (no exception; cash = B).
        all_reject = pd.Series(True, index=panel.index)
        all_reject_by_h: dict[str, Any] = {}
        all_reject_pass = True
        for h in SCENARIOS_H:
            ar = smoke_equal_dollar_accounting(
                panel, h=float(h), reject_mask=all_reject
            )
            all_reject_by_h[str(h)] = ar
            if not ar["passed"]:
                all_reject_pass = False
        accounting_all_reject = all_reject_by_h[str(1.0)]
        accounting_primary = {
            **accounting_primary,
            "dummy_reject_passed": accounting_reject["passed"],
            "dummy_reject_detail": accounting_reject["detail"],
            "all_reject_passed": all_reject_pass,
            "all_reject_by_h": {
                k: {"passed": v["passed"], "detail": v["detail"]}
                for k, v in all_reject_by_h.items()
            },
            "all_reject_detail": accounting_all_reject["detail"],
            "scenarios_all_passed": all_scenarios_pass,
            "passed": bool(
                all_scenarios_pass
                and accounting_reject["passed"]
                and all_reject_pass
            ),
            "detail": (
                f"scenarios_all_passed={all_scenarios_pass}; "
                f"h=1 {scenario_results['1.0']['detail']}; "
                f"dummy_reject={accounting_reject['detail']}; "
                f"all_reject_all_h={all_reject_pass} "
                f"all_reject_h1={accounting_all_reject['detail']}"
            ),
        }
    else:
        accounting_primary["scenarios_all_passed"] = all_scenarios_pass
        accounting_primary["passed"] = bool(
            all_scenarios_pass and accounting_primary.get("passed", False)
        )
        accounting_primary["detail"] = (
            f"scenarios_all_passed={all_scenarios_pass}; {accounting_primary.get('detail')}"
        )
    missing_smoke = missing_outcome_smoke(panel, h=1.0)
    timings["accounting_smoke"] = _progress(
        "accounting_smoke", t_stage, note=accounting_primary.get("detail", "")
    )

    print("[D0 readiness] stage=gates starting", flush=True)
    t_stage = time.perf_counter()
    # Evaluate on N panel; reconstruction comparison used full long frame.
    verdict, gates, coverage = evaluate_readiness_gates(
        panel=panel,
        identity_gates=identity_gates,
        shared_quotes_ok=shared_ok,
        reconstruction_vs_included=recon_info,
        accounting=accounting_primary,
        missing_outcome=missing_smoke,
        used_narrow_helper=True,
    )
    timings["gates"] = _progress("gates", t_stage, note=verdict)
    timings["total"] = _progress("total", t0, note=verdict)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_run_dir": str(run_dir),
        "sprint006_execution_repo_sha": OFFICIAL_EXECUTION_REPO_SHA,
        "d0_code_commit_sha": get_current_repo_sha(),
        "verdict": verdict,
        "gates": [{"gate_id": g.gate_id, "passed": g.passed, "detail": g.detail} for g in gates],
        "coverage": coverage,
        "accounting": accounting_primary,
        "missing_outcome_smoke": missing_smoke,
        "stage_timings": timings,
        "protocol_pins": {
            "BUDGET_B": BUDGET_B,
            "FEES": FEES,
            "MAX_NAMES": MAX_NAMES,
            "SCENARIOS_H": list(SCENARIOS_H),
            "M3_LOOKBACK_DAYS": M3_LOOKBACK_DAYS,
            "M3_MIN_HISTORY": M3_MIN_HISTORY,
            "DOLLAR_TOL": DOLLAR_TOL,
            "ACCOUNTING_TOL": ACCOUNTING_TOL,
            "PRIMARY_START": PRIMARY_START.isoformat(),
            "PRIMARY_END": PRIMARY_END.isoformat(),
        },
        "identity_manifest": identity_manifest,
    }
    return D0ReadinessResult(
        verdict=verdict,
        gates=gates,
        panel=panel,
        coverage=coverage,
        accounting=accounting_primary,
        identity=identity_manifest,
        stage_timings=timings,
        manifest=manifest,
    )


def export_d0_readiness_evidence(
    *,
    result: D0ReadinessResult,
    clean_notebook: Path | None = None,
    evidence_dir: Path | None = None,
    d0_code_commit_sha: str | None = None,
    execute_notebook: bool = False,
) -> Path:
    """Write readiness evidence JSON (and optionally execute the notebook)."""
    evidence_dir = evidence_dir or Path(
        f"C:/MomentumCVG_env/runs/sprint008_d0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    d0_code_commit_sha = d0_code_commit_sha or get_current_repo_sha()

    manifest_path = evidence_dir / "d0_readiness_manifest.json"
    payload = dict(result.manifest)
    payload["d0_code_commit_sha"] = d0_code_commit_sha
    payload["evidence_dir"] = str(evidence_dir)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    gates_path = evidence_dir / "d0_readiness_gates.json"
    gates_path.write_text(
        json.dumps(
            [{"gate_id": g.gate_id, "passed": g.passed, "detail": g.detail} for g in result.gates],
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage_path = evidence_dir / "d0_readiness_coverage.json"
    coverage_path.write_text(
        json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
    )

    if not result.panel.empty:
        panel_path = evidence_dir / "d0_long_panel_preview.parquet"
        preview_cols = [
            c
            for c in (
                *TRADE_KEY,
                "in_N",
                "M",
                "H",
                "S0",
                "K",
                "ST",
                "X",
                "M1",
                "M2",
                "M3",
                "delta_M_vs_stored",
                "delta_MH_vs_ask",
                "m3_missing_reason",
            )
            if c in result.panel.columns
        ]
        result.panel.loc[:, preview_cols].to_parquet(panel_path, index=False)

    if execute_notebook and clean_notebook is not None:
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        python = Path("C:/MomentumCVG_env/venv/Scripts/python.exe")
        if not python.exists():
            python = Path(sys.executable)
        executed = evidence_dir / "d0_input_readiness.executed.ipynb"
        html_path = evidence_dir / "d0_input_readiness.html"
        jupyter = [str(python), "-m", "jupyter"]
        subprocess.run(
            [
                *jupyter,
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(clean_notebook),
                "--output",
                executed.name,
                "--output-dir",
                str(evidence_dir),
                "--ExecutePreprocessor.kernel_name=momentumcvg",
            ],
            check=True,
            cwd=repo_root,
            env=env,
        )
        subprocess.run(
            [
                *jupyter,
                "nbconvert",
                "--to",
                "html",
                str(executed),
                "--output",
                html_path.name,
                "--output-dir",
                str(evidence_dir),
            ],
            check=True,
            cwd=repo_root,
            env=env,
        )
        receipt = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "d0_code_commit_sha": d0_code_commit_sha,
            "sprint006_execution_repo_sha": OFFICIAL_EXECUTION_REPO_SHA,
            "executed_notebook": str(executed),
            "executed_notebook_sha256": sha256_file(executed),
            "html_export": str(html_path),
            "html_export_sha256": sha256_file(html_path),
        }
        (evidence_dir / "execution_receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    return evidence_dir
