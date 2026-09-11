"""
Sprint 008 D1 cost diagnosis and fixed U-exclusion follow-up.

Explains M1/M2 L vs U gaps via cost dispersion and r=g-a decomposition,
then compares the equal-dollar baseline to excluding the within-date U group.
Preserves historical D1 STOP_NO_THRESHOLDS. No broader threshold search.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_EXECUTION_REPO_SHA,
    OFFICIAL_RUN_DIR,
    get_current_repo_sha,
)
from src.backtest.sprint008_d0_input_readiness import (
    ACCOUNTING_TOL,
    BUDGET_B,
    CROSSED_QUOTE_POLICY_VERSION,
    DOLLAR_TOL,
    FEES,
    _required_input_ok,
)
from src.backtest.sprint008_d1_measurement_validation import (
    DEV_A_END,
    DEV_A_START,
    DEV_B_END,
    DEV_B_START,
    DEV_END,
    DEV_START,
    PRIMARY_H,
    assert_no_evaluation_rows,
    attach_scenario_economics,
    build_d1_base_panel,
    filter_development_panel,
    is_development_date,
    is_evaluation_date,
    reconcile_economics,
)
from src.backtest.sprint008_d1_within_date_followup import (
    FOLLOWUP_MEASUREMENTS,
    GROUP_FRACTION_DENOM,
    HAC_MAXLAGS,
    MIN_SCORED,
    FollowupValidationError,
    bonferroni_adjust_p,
    newey_west_intercept_inference,
    scored_candidates_for_date,
    select_within_date_groups,
)

# Prior within-date follow-up point estimates (reconciliation targets).
PRIOR_MEAN_D = {"M1": 0.05238934614809024, "M2": 0.0755143594293551}
PRIOR_N_ELIGIBLE = 209
PRIOR_EVIDENCE = "C:/MomentumCVG_env/runs/sprint008_d1_within_date_20260908T195615Z/"
ORIGINAL_D1_GATE = "STOP_NO_THRESHOLDS"

FAMILY_SIZE = 4  # M1/M2 uplift + M1/M2 win-rate L-U
ALPHA = 0.05
ORDINARY_CI_LEVEL = 0.95
ADJUSTED_CI_LEVEL = 1.0 - ALPHA / FAMILY_SIZE  # 0.9875
RECONCILE_D_TOL = 1e-9
PRIOR_D_TOL = 1e-8


class CostDiagnosisError(FollowupValidationError):
    """Hard failure for cost-diagnosis construction or accounting."""


@dataclass
class CostDiagnosisResult:
    trade_level: pd.DataFrame = field(default_factory=pd.DataFrame)
    date_decomp: pd.DataFrame = field(default_factory=pd.DataFrame)
    date_portfolio: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_dispersion: dict[str, Any] = field(default_factory=dict)
    decomposition_summary: dict[str, Any] = field(default_factory=dict)
    exclusion_summary: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    interpretation: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    panel_dev: pd.DataFrame = field(default_factory=pd.DataFrame)


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _progress(stage: str, started: float, *, note: str = "") -> float:
    elapsed = time.perf_counter() - started
    suffix = f" — {note}" if note else ""
    print(f"[D1 cost diagnosis] {stage}: {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def enforce_d0_required_inputs(panel: pd.DataFrame) -> dict[str, Any]:
    """Fail hard if any development in_N row fails D0 `_required_input_ok`."""
    if panel.empty:
        return {"n_checked": 0, "n_failed": 0}
    work = panel.loc[panel["in_N"] == True].copy()  # noqa: E712
    failures: list[str] = []
    for idx, row in work.iterrows():
        if not _required_input_ok(row):
            failures.append(f"{_as_date(row['trade_date'])}:{row.get('ticker')}")
    if failures:
        preview = ", ".join(failures[:20])
        raise CostDiagnosisError(
            f"D0 required-input failures on analysis path: n={len(failures)} "
            f"(preview: {preview})"
        )
    return {"n_checked": int(len(work)), "n_failed": 0}


def require_all_executed_outcomes(panel_econ: pd.DataFrame) -> None:
    """Missing outcome on any executed baseline trade (incl. middle) is fatal."""
    mask = (
        (panel_econ["in_N"] == True)  # noqa: E712
        & (panel_econ["analysis_eligible"] == True)  # noqa: E712
        & (pd.to_numeric(panel_econ.get("q_h", 0), errors="coerce") > 0)
    )
    executed = panel_econ.loc[mask]
    if executed.empty:
        return
    bad = ~(executed["assoc_valid"].astype(bool) & executed["r"].map(_finite))
    if bool(bad.any()):
        keys = [
            f"{_as_date(r.trade_date)}:{r.ticker}"
            for r in executed.loc[bad].itertuples(index=False)
        ]
        raise CostDiagnosisError(
            f"Missing required outcomes for executed baseline trades: {keys[:30]}"
        )


def select_groups_with_middle(day: pd.DataFrame, measurement: str) -> dict[str, Any]:
    """Extend prior L/U selection with middle = remaining scored rows."""
    base = select_within_date_groups(day, measurement)
    if not base["eligible"]:
        return base
    work = day.copy()
    work["_m"] = pd.to_numeric(work[measurement], errors="coerce")
    work = work.loc[work["_m"].map(_finite)].copy()
    work["_tk"] = work["ticker"].astype(str)
    # Preserve original index for parent-panel labeling (match select_within_date_groups).
    work = work.sort_values(["_m", "_tk"], ascending=True, kind="mergesort")
    k = int(base["k"])
    n = int(len(work))
    middle = work.iloc[k : n - k].copy() if n > 2 * k else work.iloc[0:0].copy()
    out = dict(base)
    out["middle"] = middle
    out["group_fraction"] = float(k / n) if n else 0.0
    return out


def attach_decomposition_columns(panel_econ: pd.DataFrame) -> pd.DataFrame:
    """Attach g=(X-M)/C, a=H/C with identity r=g-a for association-valid rows."""
    out = panel_econ.copy()
    m = pd.to_numeric(out["M"], errors="coerce")
    h = pd.to_numeric(out["H"], errors="coerce")
    x = pd.to_numeric(out["X"], errors="coerce")
    c = pd.to_numeric(out["C"], errors="coerce")
    valid = out["assoc_valid"].astype(bool) & c.map(_finite) & (c > 0)
    g = np.where(valid.to_numpy(), (x - m) / c, np.nan)
    a = np.where(valid.to_numpy(), h / c, np.nan)
    out["g"] = g
    out["a"] = a  # spread drag on invested capital H/C
    out["m_over_s0"] = np.where(
        pd.to_numeric(out["S0"], errors="coerce").map(_finite)
        & (pd.to_numeric(out["S0"], errors="coerce") > 0),
        m / pd.to_numeric(out["S0"], errors="coerce"),
        np.nan,
    )
    # Dollar P&L on equal stake: p = (B/N) * r = stake_dollars * r
    stake = pd.to_numeric(out["stake_dollars"], errors="coerce")
    r = pd.to_numeric(out["r"], errors="coerce")
    out["p"] = np.where(valid.to_numpy(), stake * r, np.nan)
    # Identity check
    if valid.any():
        err = (pd.Series(g) - pd.Series(a) - out["r"]).abs()
        max_err = float(err.loc[valid].max())
        if max_err > ACCOUNTING_TOL:
            raise CostDiagnosisError(f"r=g-a identity failed: max_err={max_err}")
    return out


def assign_within_date_labels(
    panel_econ: pd.DataFrame,
    measurement: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label each scored trade L/middle/U; return labeled frame, paired, excluded."""
    work = panel_econ.copy()
    work["trade_date"] = work["trade_date"].map(_as_date)
    label_col = f"group_{measurement}"
    work[label_col] = pd.NA
    paired_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for trade_date, day_all in work.groupby("trade_date", sort=True):
        td = _as_date(trade_date)
        if not is_development_date(td):
            continue
        scored = scored_candidates_for_date(day_all, measurement)
        selection = select_groups_with_middle(scored, measurement)
        if not selection["eligible"]:
            excluded_rows.append(
                {
                    "measurement": measurement,
                    "trade_date": td,
                    "reason": selection["reason"],
                    "n_scored": selection["n_scored"],
                    "k": selection["k"],
                }
            )
            continue

        low = selection["low"]
        high = selection["high"]
        middle = selection["middle"]
        # Outcomes already required globally; still verify selected sets.
        for name, grp in (("low", low), ("high", high), ("middle", middle)):
            if grp.empty and name == "middle":
                continue
            if name != "middle" and grp.empty:
                raise CostDiagnosisError(f"{measurement} {td}: empty {name}")
            bad = ~(grp["assoc_valid"].astype(bool) & grp["r"].map(_finite))
            if bool(bad.any()):
                raise CostDiagnosisError(
                    f"{measurement} {td} {name}: missing outcomes "
                    f"{grp.loc[bad, 'ticker'].tolist()}"
                )

        idx_low = low.index
        idx_high = high.index
        idx_mid = middle.index
        work.loc[idx_low, label_col] = "L"
        work.loc[idx_high, label_col] = "U"
        work.loc[idx_mid, label_col] = "middle"

        def _mean(col: str, frame: pd.DataFrame) -> float:
            return float(np.mean(frame[col].to_numpy(dtype=float))) if len(frame) else float("nan")

        l_r, u_r = _mean("r", low), _mean("r", high)
        l_g, u_g = _mean("g", low), _mean("g", high)
        l_a, u_a = _mean("a", low), _mean("a", high)
        d_net = l_r - u_r
        d_gross = l_g - u_g
        spread_saving = u_a - l_a
        if abs(d_net - (d_gross + spread_saving)) > RECONCILE_D_TOL:
            raise CostDiagnosisError(
                f"{measurement} {td}: d_net != d_gross+spread_saving "
                f"({d_net} vs {d_gross + spread_saving})"
            )

        l_win = float(np.mean(low["p"].to_numpy(dtype=float) > 0.0))
        u_win = float(np.mean(high["p"].to_numpy(dtype=float) > 0.0))
        paired_rows.append(
            {
                "measurement": measurement,
                "trade_date": td,
                "n_scored": int(selection["n_scored"]),
                "k": int(selection["k"]),
                "group_fraction": float(selection["group_fraction"]),
                "n_middle": int(len(middle)),
                "low_cutoff_tie": bool(selection["low_cutoff_tie"]),
                "high_cutoff_tie": bool(selection["high_cutoff_tie"]),
                "mean_r_L": l_r,
                "mean_r_U": u_r,
                "mean_r_middle": _mean("r", middle) if len(middle) else float("nan"),
                "d_net": d_net,
                "d_gross": d_gross,
                "spread_saving": spread_saving,
                "mean_a_L": l_a,
                "mean_a_U": u_a,
                "mean_a_middle": _mean("a", middle) if len(middle) else float("nan"),
                "mean_g_L": l_g,
                "mean_g_U": u_g,
                "mean_g_middle": _mean("g", middle) if len(middle) else float("nan"),
                "mean_H_over_C_L": l_a,
                "mean_H_over_C_U": u_a,
                "mean_M_over_S0_L": _mean("m_over_s0", low),
                "mean_M_over_S0_U": _mean("m_over_s0", high),
                "winrate_net_L": l_win,
                "winrate_net_U": u_win,
                "winrate_net_diff_LU": l_win - u_win,
                "winrate_gross_L": float(np.mean(low["g"].to_numpy(dtype=float) > 0.0)),
                "winrate_gross_U": float(np.mean(high["g"].to_numpy(dtype=float) > 0.0)),
            }
        )

    paired = pd.DataFrame(paired_rows)
    excluded = pd.DataFrame(excluded_rows)
    if excluded.empty:
        excluded = pd.DataFrame(
            columns=["measurement", "trade_date", "reason", "n_scored", "k"]
        )
    if not paired.empty:
        paired = paired.sort_values("trade_date").reset_index(drop=True)
    return work, paired, excluded


def summarize_cost_dispersion(trade_level: pd.DataFrame, measurement: str) -> dict[str, Any]:
    label = f"group_{measurement}"
    scored = trade_level.loc[trade_level[label].notna()].copy()
    if scored.empty:
        return {"measurement": measurement, "empty": True}

    def _grp(name: str) -> pd.DataFrame:
        return scored.loc[scored[label] == name]

    out: dict[str, Any] = {"measurement": measurement, "weighting": "pooled_trades"}
    for name in ("L", "middle", "U"):
        g = _grp(name)
        out[name] = {
            "n": int(len(g)),
            "mean_score": float(pd.to_numeric(g[measurement], errors="coerce").mean())
            if len(g)
            else None,
            "mean_H_over_C": float(g["a"].mean()) if len(g) else None,
            "mean_M_over_S0": float(g["m_over_s0"].mean()) if len(g) else None,
            "mean_r": float(g["r"].mean()) if len(g) else None,
            "mean_g": float(g["g"].mean()) if len(g) else None,
            "std_r": float(g["r"].std(ddof=1)) if len(g) > 1 else None,
        }
    # Date-weighted mean of within-date U-L spread-drag difference
    # (computed from date_decomp elsewhere); placeholder filled by caller.
    out["note"] = (
        "ATM sample establishes dispersion among selected ATM trades; "
        "it does not establish that ATM selection caused that dispersion."
    )
    return out


def summarize_decomposition(paired: pd.DataFrame, measurement: str) -> dict[str, Any]:
    sub = paired.loc[paired["measurement"] == measurement]
    if sub.empty:
        return {"measurement": measurement, "n_dates": 0}
    d_net = float(sub["d_net"].mean())
    d_gross = float(sub["d_gross"].mean())
    spread_saving = float(sub["spread_saving"].mean())
    return {
        "measurement": measurement,
        "weighting": "equal_weight_eligible_dates",
        "n_dates": int(len(sub)),
        "mean_d_net": d_net,
        "mean_d_gross": d_gross,
        "mean_spread_saving": spread_saving,
        "identity_check_ok": abs(d_net - (d_gross + spread_saving)) < 1e-10,
        "mean_a_L": float(sub["mean_a_L"].mean()),
        "mean_a_U": float(sub["mean_a_U"].mean()),
        "mean_a_U_minus_L": float((sub["mean_a_U"] - sub["mean_a_L"]).mean()),
        "mean_g_L": float(sub["mean_g_L"].mean()),
        "mean_g_U": float(sub["mean_g_U"].mean()),
        "mean_r_L": float(sub["mean_r_L"].mean()),
        "mean_r_U": float(sub["mean_r_U"].mean()),
        "mean_r_middle": float(sub["mean_r_middle"].mean()),
        "half_2020_2021": {
            "n": int(((sub["trade_date"] >= DEV_A_START) & (sub["trade_date"] <= DEV_A_END)).sum()),
            "mean_d_net": float(
                sub.loc[
                    (sub["trade_date"] >= DEV_A_START) & (sub["trade_date"] <= DEV_A_END),
                    "d_net",
                ].mean()
            ),
        },
        "half_2022_2023": {
            "n": int(((sub["trade_date"] >= DEV_B_START) & (sub["trade_date"] <= DEV_B_END)).sum()),
            "mean_d_net": float(
                sub.loc[
                    (sub["trade_date"] >= DEV_B_START) & (sub["trade_date"] <= DEV_B_END),
                    "d_net",
                ].mean()
            ),
        },
    }


def pooled_win_diagnostics(trade_level: pd.DataFrame, measurement: str) -> dict[str, Any]:
    label = f"group_{measurement}"
    scored = trade_level.loc[
        trade_level[label].notna() & trade_level["assoc_valid"].astype(bool)
    ].copy()
    if scored.empty:
        return {}
    gross_win = scored["g"] > 0
    net_win = scored["p"] > 0
    gross_win_net_non = gross_win & ~net_win
    out: dict[str, Any] = {
        "weighting": "pooled_scored_trades",
        "n": int(len(scored)),
        "gross_win_rate": float(gross_win.mean()),
        "net_win_rate": float(net_win.mean()),
        "gross_winner_became_net_nonwinner_rate": float(gross_win_net_non.mean()),
        "mean_winning_r": float(scored.loc[net_win, "r"].mean()) if bool(net_win.any()) else None,
        "mean_losing_r": float(scored.loc[~net_win, "r"].mean()) if bool((~net_win).any()) else None,
        "median_r": float(scored["r"].median()),
        "by_group": {},
    }
    for name in ("L", "middle", "U"):
        g = scored.loc[scored[label] == name]
        if g.empty:
            continue
        nw = g["p"] > 0
        out["by_group"][name] = {
            "n": int(len(g)),
            "net_win_rate": float(nw.mean()),
            "gross_win_rate": float((g["g"] > 0).mean()),
            "mean_r": float(g["r"].mean()),
            "median_r": float(g["r"].median()),
        }
    # Unfiltered baseline (all executed)
    base = trade_level.loc[trade_level["assoc_valid"] == True]  # noqa: E712
    out["unfiltered_baseline"] = {
        "n": int(len(base)),
        "n_dates": int(base["trade_date"].nunique()),
        "mean_r": float(base["r"].mean()) if len(base) else None,
        "mean_g": float(base["g"].mean()) if len(base) else None,
        "mean_a": float(base["a"].mean()) if len(base) else None,
        "net_win_rate": float((base["p"] > 0).mean()) if len(base) else None,
        "gross_win_rate": float((base["g"] > 0).mean()) if len(base) else None,
        "total_p": float(base["p"].sum()) if len(base) else None,
    }
    return out


def build_portfolio_comparison(
    trade_level: pd.DataFrame,
    paired: pd.DataFrame,
    measurement: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Baseline vs exclude-U on full development calendar; one row per entry date."""
    label = f"group_{measurement}"
    work = trade_level.copy()
    work["trade_date"] = work["trade_date"].map(_as_date)
    eligible_dates = set(
        paired.loc[paired["measurement"] == measurement, "trade_date"].map(_as_date)
    )
    rows: list[dict[str, Any]] = []

    for trade_date, day in work.groupby("trade_date", sort=True):
        td = _as_date(trade_date)
        if not is_development_date(td):
            continue
        executed = day.loc[
            (day["in_N"] == True)  # noqa: E712
            & (day["analysis_eligible"] == True)  # noqa: E712
            & (day["assoc_valid"] == True)  # noqa: E712
        ].copy()
        # Cash holds (crossed) contribute 0
        n_in_n = int((day["in_N"] == True).sum())  # noqa: E712
        stake = BUDGET_B / n_in_n if n_in_n else 0.0

        p_all = executed["p"].to_numpy(dtype=float)
        r_base = float(np.nansum(p_all) / BUDGET_B)

        exclude_u = td in eligible_dates
        if exclude_u:
            u_mask = executed[label] == "U"
            retained = executed.loc[~u_mask]
            u_trades = executed.loc[u_mask]
        else:
            retained = executed
            u_trades = executed.iloc[0:0]

        p_ret = retained["p"].to_numpy(dtype=float)
        p_u = u_trades["p"].to_numpy(dtype=float) if len(u_trades) else np.array([])
        r_filt = float(np.nansum(p_ret) / BUDGET_B)
        uplift = r_filt - r_base
        # Identity: uplift = -sum_U(p)/B
        if len(p_u):
            expected = float(-np.nansum(p_u) / BUDGET_B)
            if abs(uplift - expected) > ACCOUNTING_TOL:
                raise CostDiagnosisError(
                    f"{measurement} {td}: uplift identity failed ({uplift} vs {expected})"
                )

        invested_base = float(len(executed) * stake)
        invested_filt = float(len(retained) * stake)
        rows.append(
            {
                "measurement": measurement,
                "trade_date": td,
                "exclude_u_applied": bool(exclude_u),
                "n_in_N": n_in_n,
                "n_executed": int(len(executed)),
                "n_retained": int(len(retained)),
                "n_excluded_u": int(len(u_trades)),
                "R_baseline": r_base,
                "R_filtered": r_filt,
                "uplift": uplift,
                "pnl_baseline": float(np.nansum(p_all)),
                "pnl_filtered": float(np.nansum(p_ret)),
                "pnl_u": float(np.nansum(p_u)) if len(p_u) else 0.0,
                "invested_frac_baseline": invested_base / BUDGET_B,
                "invested_frac_filtered": invested_filt / BUDGET_B,
                "cash_frac_filtered": 1.0 - invested_filt / BUDGET_B,
            }
        )

    port = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)

    # Retention metrics over full development (pooled dollars)
    executed_all = work.loc[
        (work["in_N"] == True)  # noqa: E712
        & (work["analysis_eligible"] == True)  # noqa: E712
        & (work["assoc_valid"] == True)  # noqa: E712
    ].copy()
    # Mark excluded U only on eligible dates
    is_u_excl = []
    for row in executed_all.itertuples(index=False):
        td = _as_date(row.trade_date)
        lab = getattr(row, label, None)
        is_u_excl.append((td in eligible_dates) and (lab == "U") is True)
    executed_all = executed_all.copy()
    executed_all["_excl"] = np.asarray(is_u_excl, dtype=bool)
    retained_all = executed_all.loc[~executed_all["_excl"].to_numpy()]
    excluded_all = executed_all.loc[executed_all["_excl"].to_numpy()]

    base_p = executed_all["p"].to_numpy(dtype=float)
    ret_p = retained_all["p"].to_numpy(dtype=float)
    excl_p = excluded_all["p"].to_numpy(dtype=float)

    losses_avoided = float(-np.sum(excl_p[excl_p < 0])) if len(excl_p) else 0.0
    winning_sacrificed = float(np.sum(excl_p[excl_p > 0])) if len(excl_p) else 0.0
    pnl_improvement = float(np.sum(ret_p) - np.sum(base_p))
    # improvement = -sum(excl_p) = losses_avoided - winning_sacrificed
    expected_imp = losses_avoided - winning_sacrificed
    if abs(pnl_improvement - expected_imp) > 1e-6:
        # numerically: pnl_improvement should equal -sum(excl_p)
        if abs(pnl_improvement - float(-np.sum(excl_p))) > ACCOUNTING_TOL:
            raise CostDiagnosisError(
                f"{measurement}: P&L improvement identity failed "
                f"({pnl_improvement} vs {-np.sum(excl_p)})"
            )

    win_base = base_p > 0
    win_ret = ret_p > 0
    sum_win_base = float(np.sum(base_p[win_base])) if bool(win_base.any()) else 0.0
    sum_win_ret = float(np.sum(ret_p[win_ret])) if bool(win_ret.any()) else 0.0
    if sum_win_base > 0:
        win_profit_retention = sum_win_ret / sum_win_base
    else:
        win_profit_retention = None  # NA

    n_win_base = int(win_base.sum())
    n_win_ret = int(win_ret.sum())
    winner_count_retention = (
        float(n_win_ret / n_win_base) if n_win_base > 0 else None
    )

    # Top 5 / 10 winning trades by baseline p
    winners = executed_all.loc[executed_all["p"] > 0].sort_values("p", ascending=False)
    top_ret = {}
    for top_n in (5, 10):
        top = winners.head(top_n)
        if top.empty or float(top["p"].sum()) <= 0:
            top_ret[f"top{top_n}_profit_retention"] = None
            top_ret[f"top{top_n}_count_retained"] = 0
            top_ret[f"top{top_n}_n"] = int(len(top))
        else:
            retained_mask = ~top["_excl"].to_numpy()
            top_ret[f"top{top_n}_profit_retention"] = float(
                top.loc[retained_mask, "p"].sum() / top["p"].sum()
            )
            top_ret[f"top{top_n}_count_retained"] = int(retained_mask.sum())
            top_ret[f"top{top_n}_n"] = int(len(top))

    # Cumulative fixed-budget P&L and peak-to-trough
    cum_base = port["pnl_baseline"].cumsum()
    cum_filt = port["pnl_filtered"].cumsum()
    def _max_dd(cum: pd.Series) -> float:
        peak = cum.cummax()
        dd = cum - peak
        return float(dd.min()) if len(dd) else 0.0

    summary = {
        "measurement": measurement,
        "n_dates": int(len(port)),
        "n_dates_exclusion_applied": int(port["exclude_u_applied"].sum()),
        "mean_R_baseline": float(port["R_baseline"].mean()),
        "mean_R_filtered": float(port["R_filtered"].mean()),
        "mean_uplift": float(port["uplift"].mean()),
        "total_pnl_baseline": float(port["pnl_baseline"].sum()),
        "total_pnl_filtered": float(port["pnl_filtered"].sum()),
        "total_pnl_improvement": pnl_improvement,
        "losses_avoided": losses_avoided,
        "winning_profits_sacrificed": winning_sacrificed,
        "identity_improvement_ok": abs(pnl_improvement - float(-np.sum(excl_p)))
        <= ACCOUNTING_TOL,
        "mean_invested_frac_baseline": float(port["invested_frac_baseline"].mean()),
        "mean_invested_frac_filtered": float(port["invested_frac_filtered"].mean()),
        "mean_cash_frac_filtered": float(port["cash_frac_filtered"].mean()),
        "trade_coverage_retained_frac": float(len(retained_all) / max(len(executed_all), 1)),
        "n_executed_trades": int(len(executed_all)),
        "n_retained_trades": int(len(retained_all)),
        "n_excluded_u_trades": int(len(excluded_all)),
        "winning_profit_retention": win_profit_retention,
        "winner_count_retention": winner_count_retention,
        "n_winners_baseline": n_win_base,
        "n_winners_retained": n_win_ret,
        **top_ret,
        "cum_pnl_baseline_final": float(cum_base.iloc[-1]) if len(cum_base) else 0.0,
        "cum_pnl_filtered_final": float(cum_filt.iloc[-1]) if len(cum_filt) else 0.0,
        "peak_to_trough_baseline": _max_dd(cum_base),
        "peak_to_trough_filtered": _max_dd(cum_filt),
        "note_drawdown": (
            "Peak-to-trough on cumulative fixed-budget date P&L; "
            "not compounded equity or intraholding drawdown."
        ),
    }
    port["cum_pnl_baseline"] = cum_base
    port["cum_pnl_filtered"] = cum_filt
    return port, summary


def _hac_bundle(series: np.ndarray, name: str) -> dict[str, Any]:
    hac = newey_west_intercept_inference(
        np.asarray(series, dtype=float),
        maxlags=HAC_MAXLAGS,
        ordinary_level=ORDINARY_CI_LEVEL,
        adjusted_level=ADJUSTED_CI_LEVEL,
    )
    p_adj = bonferroni_adjust_p(hac["p_raw"], family_size=FAMILY_SIZE)
    return {
        "name": name,
        "n": int(len(series)),
        "mean": float(np.mean(series)),
        "hac": hac,
        "p_adjusted": p_adj,
        "family_size": FAMILY_SIZE,
        "adjusted_ci_level": ADJUSTED_CI_LEVEL,
        "stat_sig_adj": bool(p_adj < ALPHA),
    }


def run_inference(
    paired: pd.DataFrame,
    portfolios: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    out: dict[str, Any] = {"family_size": FAMILY_SIZE, "contrasts": {}}
    for m in FOLLOWUP_MEASUREMENTS:
        port = portfolios[m]
        uplift = port["uplift"].to_numpy(dtype=float)
        out["contrasts"][f"{m}_mean_uplift"] = _hac_bundle(uplift, f"{m}_mean_uplift")
        # Calendar gaps on portfolio dates
        dates = port["trade_date"].map(_as_date).tolist()
        gaps = [int((dates[i] - dates[i - 1]).days) for i in range(1, len(dates))]
        out["contrasts"][f"{m}_mean_uplift"]["calendar_gap_days"] = {
            "min": min(gaps) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "max": max(gaps) if gaps else None,
            "interpretation": (
                "HAC lags index successive development entry dates "
                "(full calendar for uplift)."
            ),
        }

        sub = paired.loc[paired["measurement"] == m]
        wr = sub["winrate_net_diff_LU"].to_numpy(dtype=float)
        out["contrasts"][f"{m}_winrate_LU"] = _hac_bundle(wr, f"{m}_winrate_LU")
        edates = sub["trade_date"].map(_as_date).tolist()
        egaps = [int((edates[i] - edates[i - 1]).days) for i in range(1, len(edates))]
        out["contrasts"][f"{m}_winrate_LU"]["calendar_gap_days"] = {
            "min": min(egaps) if egaps else None,
            "median": float(np.median(egaps)) if egaps else None,
            "max": max(egaps) if egaps else None,
            "interpretation": (
                "HAC lags index successive eligible split dates for L/U win-rate."
            ),
        }

        # Reproduce prior mean d_net inference (unchanged settings; exploratory)
        d_net = sub["d_net"].to_numpy(dtype=float)
        prior_style = newey_west_intercept_inference(
            d_net,
            maxlags=HAC_MAXLAGS,
            ordinary_level=0.95,
            adjusted_level=0.975,  # prior family size 2
        )
        out["reproduced_prior_mean_d_net"] = out.get("reproduced_prior_mean_d_net", {})
        out["reproduced_prior_mean_d_net"][m] = {
            "mean": float(np.mean(d_net)),
            "hac_family_size_2_style": prior_style,
            "p_adjusted_family_2": bonferroni_adjust_p(prior_style["p_raw"], family_size=2),
            "note": "Reproduction of prior within-date follow-up inference; not a new gate.",
        }
    return out


def build_interpretation(
    *,
    decomp: dict[str, Any],
    dispersion: dict[str, Any],
    exclusion: dict[str, Any],
    inference: dict[str, Any],
    win_diag: dict[str, Any],
) -> dict[str, Any]:
    """Answer decision questions and pick at most one next experiment."""
    # Use M1 as primary narrative; cite both.
    base = win_diag.get("M1", {}).get("unfiltered_baseline", {})
    gross_pos = (base.get("mean_g") is not None) and (base["mean_g"] > 0)

    answers = {
        "baseline_positive_gross_midpoint": {
            "answer": bool(gross_pos),
            "detail": {
                m: win_diag.get(m, {}).get("unfiltered_baseline")
                for m in FOLLOWUP_MEASUREMENTS
            },
        },
        "spread_variation_meaningful": {},
        "savings_offset_by_gross": {},
        "exclusion_improves_total_profit": {},
        "consistency_vs_concentration": {},
        "unresolved": [],
    }

    for m in FOLLOWUP_MEASUREMENTS:
        d = decomp[m]
        saving = d.get("mean_spread_saving")
        d_gross = d.get("mean_d_gross")
        d_net = d.get("mean_d_net")
        a_gap = d.get("mean_a_U_minus_L")
        # Compare saving to |d_gross| and to baseline return variability
        std_r = dispersion[m].get("L", {}).get("std_r")
        answers["spread_variation_meaningful"][m] = {
            "mean_U_minus_L_H_over_C": a_gap,
            "mean_spread_saving": saving,
            "vs_abs_mean_d_gross": None
            if d_gross is None
            else (abs(saving) / max(abs(d_gross), 1e-12)),
            "comment": (
                "Spread-drag gap is the mechanical upper bound on net L−U from costs alone."
            ),
        }
        answers["savings_offset_by_gross"][m] = {
            "mean_d_gross": d_gross,
            "mean_spread_saving": saving,
            "mean_d_net": d_net,
            "gross_offsets_savings": bool(
                d_gross is not None and saving is not None and d_gross < 0 and abs(d_gross) > 0.5 * abs(saving)
            ),
        }
        excl = exclusion[m]
        uplift_inf = inference["contrasts"][f"{m}_mean_uplift"]
        answers["exclusion_improves_total_profit"][m] = {
            "total_pnl_improvement": excl.get("total_pnl_improvement"),
            "mean_uplift": excl.get("mean_uplift"),
            "winning_profit_retention": excl.get("winning_profit_retention"),
            "top5_profit_retention": excl.get("top5_profit_retention"),
            "top10_profit_retention": excl.get("top10_profit_retention"),
            "hac_mean_uplift_sig_adj": uplift_inf.get("stat_sig_adj"),
            "hac_p_adj": uplift_inf.get("p_adjusted"),
        }

    # Concentration: share of uplift from top weeks
    conc = {}
    for m in FOLLOWUP_MEASUREMENTS:
        # Use exclusion summary totals vs weekly variability already in port via inference n
        conc[m] = {
            "note": "See date_portfolio uplift distribution and top-winner retention in exports.",
            "winner_profit_retention": exclusion[m].get("winning_profit_retention"),
            "peak_to_trough_change": {
                "baseline": exclusion[m].get("peak_to_trough_baseline"),
                "filtered": exclusion[m].get("peak_to_trough_filtered"),
            },
        }
    answers["consistency_vs_concentration"] = conc
    answers["unresolved"] = [
        "Fees remain unmodeled.",
        "Quote-based full-cross results do not establish achievable fills.",
        "Post-hoc relative to D1 / prior within-date study; not independent confirmation.",
        "Broader threshold search and evaluation-period validation remain unauthorized.",
    ]

    # Recommend one next action from the menu based on patterns
    # Heuristic (documented): look at whether savings are large vs gross offset,
    # and whether exclusion improves total P&L with acceptable retention and sig.
    m1_save = decomp["M1"].get("mean_spread_saving") or 0.0
    m1_gross = decomp["M1"].get("mean_d_gross") or 0.0
    m1_imp = exclusion["M1"].get("total_pnl_improvement") or 0.0
    m1_ret = exclusion["M1"].get("winning_profit_retention")
    m1_sig = inference["contrasts"]["M1_mean_uplift"].get("stat_sig_adj", False)
    m2_sig = inference["contrasts"]["M2_mean_uplift"].get("stat_sig_adj", False)

    if abs(m1_save) < 0.02 and abs(decomp["M2"].get("mean_spread_saving") or 0.0) < 0.02:
        recommendation = {
            "choice": "limited_cost_dispersion",
            "action": (
                "Limited cost dispersion: explain the limited opportunity for "
                "additional spread filtering."
            ),
        }
    elif m1_gross < 0 and abs(m1_gross) >= 0.5 * abs(m1_save):
        recommendation = {
            "choice": "savings_offset_by_weaker_gross",
            "action": (
                "Savings offset by weaker gross payoff: propose a bounded investigation "
                "of payoff relative to premium; acknowledge that existing M3 was already tested."
            ),
        }
    elif (m1_imp > 0 or (exclusion["M2"].get("total_pnl_improvement") or 0) > 0) and (
        m1_sig or m2_sig
    ) and (m1_ret is not None and m1_ret >= 0.7):
        recommendation = {
            "choice": "favorable_exclusion_freeze_for_forward_validation",
            "action": (
                "Favorable exclusion economics with acceptable winner retention: "
                "propose freezing a rule for separately authorized chronological/forward validation."
            ),
        }
    else:
        recommendation = {
            "choice": "wide_uncertainty_or_inconclusive",
            "action": (
                "Wide uncertainty or concentrated/inconclusive results: identify what "
                "additional independent evidence is needed, or recommend closing this "
                "direction as inconclusive."
            ),
        }

    return {
        "answers": answers,
        "recommendation": recommendation,
        "disclaimer": (
            "Fees remain unmodeled; quote-based results do not establish achievable "
            "fills or dependable income."
        ),
    }


def _working_tree_status() -> str:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        return "dirty" if dirty else "clean"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        "src/backtest/sprint008_d1_cost_diagnosis.py",
        "src/backtest/sprint008_d1_within_date_followup.py",
        "src/backtest/sprint008_d1_measurement_validation.py",
        "src/backtest/sprint008_d0_input_readiness.py",
    ]
    hashes = {}
    for rel in paths:
        p = root / rel
        if p.exists():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    dirty = _working_tree_status()
    diff_sha = None
    if dirty == "dirty":
        try:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD", "--", *paths],
                cwd=root,
            )
            diff_sha = hashlib.sha256(diff).hexdigest()[:16] if diff else "no_diff_in_paths"
        except (OSError, subprocess.CalledProcessError):
            diff_sha = "unavailable"
    return {
        "code_sha": get_current_repo_sha(),
        "working_tree": dirty,
        "file_sha256_16": hashes,
        "tracked_diff_sha256_16": diff_sha,
    }


def _package_versions() -> dict[str, str]:
    import sys

    import numpy
    import pandas
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_cost_diagnosis(*, run_dir: Path | None = None) -> CostDiagnosisResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    panel = build_d1_base_panel(run_dir=run_dir or OFFICIAL_RUN_DIR)
    timings["build_panel"] = _progress("build_panel", t0, note=f"rows={len(panel)}")

    t1 = time.perf_counter()
    panel_dev = filter_development_panel(panel)
    assert_no_evaluation_rows(panel_dev, context="panel_dev")
    if panel_dev["trade_date"].map(is_evaluation_date).any():
        raise CostDiagnosisError("Evaluation rows leaked into development panel")
    d0_check = enforce_d0_required_inputs(panel_dev)
    timings["d0_checks"] = _progress("d0_checks", t1, note=str(d0_check))

    t2 = time.perf_counter()
    panel_econ = attach_scenario_economics(panel_dev, PRIMARY_H)
    recon = reconcile_economics(panel_econ)
    if not recon["passed"]:
        raise CostDiagnosisError(f"Economics reconciliation failed: {recon}")
    panel_econ = attach_decomposition_columns(panel_econ)
    require_all_executed_outcomes(panel_econ)
    timings["economics"] = _progress("economics", t2, note=f"recon={recon}")

    trade_level = panel_econ.copy()
    paired_all = []
    excluded_all = []
    decomp_summaries = {}
    dispersion = {}
    win_diag = {}
    reconciliation = {"prior_evidence": PRIOR_EVIDENCE, "measurements": {}}

    for m in FOLLOWUP_MEASUREMENTS:
        tm = time.perf_counter()
        trade_level, paired, excluded = assign_within_date_labels(trade_level, m)
        paired_all.append(paired)
        excluded_all.append(excluded)
        mean_d = float(paired["d_net"].mean()) if not paired.empty else float("nan")
        n_elig = int(len(paired))
        delta = abs(mean_d - PRIOR_MEAN_D[m]) if n_elig else float("nan")
        ok = n_elig == PRIOR_N_ELIGIBLE and delta <= PRIOR_D_TOL
        reconciliation["measurements"][m] = {
            "n_eligible": n_elig,
            "mean_d_net": mean_d,
            "prior_mean_d_net": PRIOR_MEAN_D[m],
            "abs_delta": delta,
            "n_match_prior": n_elig == PRIOR_N_ELIGIBLE,
            "mean_match_prior": bool(ok),
            "n_excluded": int(len(excluded)),
            "exclusion_reasons": (
                excluded["reason"].value_counts().astype(int).to_dict()
                if not excluded.empty
                else {}
            ),
        }
        if not ok:
            print(
                f"[D1 cost diagnosis] RECONCILE WARNING {m}: "
                f"mean_d={mean_d} prior={PRIOR_MEAN_D[m]} n={n_elig}",
                flush=True,
            )
        decomp_summaries[m] = summarize_decomposition(paired, m)
        dispersion[m] = summarize_cost_dispersion(trade_level, m)
        dispersion[m]["date_weighted_mean_a_U_minus_L"] = decomp_summaries[m].get(
            "mean_a_U_minus_L"
        )
        win_diag[m] = pooled_win_diagnostics(trade_level, m)
        decomp_summaries[m]["win_diagnostics"] = win_diag[m]
        timings[f"groups_{m}"] = _progress(
            f"groups_{m}", tm, note=f"eligible={n_elig} mean_d_net={mean_d:.6f}"
        )

    paired_df = pd.concat(paired_all, ignore_index=True)
    excluded_df = pd.concat(excluded_all, ignore_index=True)

    portfolios = {}
    exclusion_summaries = {}
    port_frames = []
    for m in FOLLOWUP_MEASUREMENTS:
        tm = time.perf_counter()
        port, summary = build_portfolio_comparison(trade_level, paired_df, m)
        portfolios[m] = port
        exclusion_summaries[m] = summary
        port_frames.append(port)
        timings[f"exclusion_{m}"] = _progress(
            f"exclusion_{m}",
            tm,
            note=f"mean_uplift={summary['mean_uplift']:.6f} "
            f"pnl_imp={summary['total_pnl_improvement']:.2f}",
        )

    port_df = pd.concat(port_frames, ignore_index=True)

    t_inf = time.perf_counter()
    inference = run_inference(paired_df, portfolios)
    timings["inference"] = _progress("inference", t_inf)

    interpretation = build_interpretation(
        decomp=decomp_summaries,
        dispersion=dispersion,
        exclusion=exclusion_summaries,
        inference=inference,
        win_diag=win_diag,
    )

    provenance = _source_provenance()
    report = {
        "protocol": "docs/tmp/sprint008_d1_cost_diagnosis_protocol.md",
        "status": "exploratory_cost_diagnosis_awaiting_review",
        "preserves_original_d1_gate": ORIGINAL_D1_GATE,
        "post_hoc_disclosure": (
            "Designed after seeing D1 and within-date follow-up results; "
            "exploratory, not independent confirmation."
        ),
        "bounded_amendment": (
            "Exactly two fixed U-exclusions (M1, M2); no broader D2 threshold search."
        ),
        "pins": {
            "dev_window": [str(DEV_START), str(DEV_END)],
            "h": PRIMARY_H,
            "fees": FEES,
            "budget_b": BUDGET_B,
            "family_size": FAMILY_SIZE,
            "adjusted_ci_level": ADJUSTED_CI_LEVEL,
            "hac_maxlags": HAC_MAXLAGS,
            "crossed_quote_policy": CROSSED_QUOTE_POLICY_VERSION,
            "official_artifacts": str(OFFICIAL_RUN_DIR),
            "official_execution_sha": OFFICIAL_EXECUTION_REPO_SHA,
            "dollar_tol": DOLLAR_TOL,
        },
        "reconciliation": reconciliation,
        "cost_dispersion": dispersion,
        "decomposition": decomp_summaries,
        "exclusion": exclusion_summaries,
        "inference": inference,
        "interpretation": interpretation,
        "environment": _package_versions(),
        "provenance": provenance,
        "stage_timings": timings,
        "total_seconds": float(time.perf_counter() - t0),
        "excluded_dates": excluded_df.to_dict(orient="records"),
    }
    timings["total"] = _progress("total", t0)

    return CostDiagnosisResult(
        trade_level=trade_level,
        date_decomp=paired_df,
        date_portfolio=port_df,
        cost_dispersion=dispersion,
        decomposition_summary=decomp_summaries,
        exclusion_summary=exclusion_summaries,
        inference=inference,
        reconciliation=reconciliation,
        interpretation=interpretation,
        report=report,
        stage_timings=timings,
        panel_dev=panel_dev,
    )


def _plot_decomposition(decomp: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = []
    nets, grosses, savings = [], [], []
    for m in FOLLOWUP_MEASUREMENTS:
        labels.append(m)
        nets.append(decomp[m]["mean_d_net"])
        grosses.append(decomp[m]["mean_d_gross"])
        savings.append(decomp[m]["mean_spread_saving"])
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, nets, w, label="d_net")
    ax.bar(x, grosses, w, label="d_gross")
    ax.bar(x + w, savings, w, label="spread_saving")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean date-level difference")
    ax.set_title("L−U decomposition (equal-weight eligible dates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cumulative(port_df: pd.DataFrame, measurement: str, path: Path) -> None:
    sub = port_df.loc[port_df["measurement"] == measurement].sort_values("trade_date")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub["trade_date"], sub["cum_pnl_baseline"], label="baseline")
    ax.plot(sub["trade_date"], sub["cum_pnl_filtered"], label="exclude U")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Cumulative fixed-budget $ P&L — {measurement}")
    ax.set_ylabel("Cumulative $")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_drag_by_group(trade_level: pd.DataFrame, measurement: str, path: Path) -> None:
    label = f"group_{measurement}"
    scored = trade_level.loc[trade_level[label].notna()].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        scored.loc[scored[label] == g, "a"].dropna().to_numpy(dtype=float)
        for g in ("L", "middle", "U")
    ]
    ax.boxplot(data, tick_labels=["L", "middle", "U"], showfliers=False)
    ax.set_ylabel("H / C (spread drag on capital)")
    ax.set_title(f"Within-date group spread drag — {measurement}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def export_cost_diagnosis_evidence(
    *,
    result: CostDiagnosisResult,
    evidence_dir: Path,
    command: str,
) -> Path:
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    # Trade-level export (key columns)
    cols = [
        c
        for c in [
            "trade_date",
            "ticker",
            "in_N",
            "analysis_eligible",
            "M",
            "H",
            "S0",
            "X",
            "C",
            "q_h",
            "stake_dollars",
            "M1",
            "M2",
            "r",
            "g",
            "a",
            "p",
            "m_over_s0",
            "group_M1",
            "group_M2",
            "assoc_valid",
        ]
        if c in result.trade_level.columns
    ]
    result.trade_level[cols].to_parquet(evidence_dir / "trade_level.parquet", index=False)
    result.trade_level[cols].to_csv(evidence_dir / "trade_level.csv", index=False)
    result.date_decomp.to_parquet(evidence_dir / "date_decomposition.parquet", index=False)
    result.date_decomp.to_csv(evidence_dir / "date_decomposition.csv", index=False)
    result.date_portfolio.to_parquet(evidence_dir / "date_portfolio.parquet", index=False)
    result.date_portfolio.to_csv(evidence_dir / "date_portfolio.csv", index=False)

    report = dict(result.report)
    report["command"] = command
    report["evidence_dir"] = str(evidence_dir)
    report["exported_utc"] = datetime.now(timezone.utc).isoformat()
    (evidence_dir / "cost_diagnosis_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # Plots
    _plot_decomposition(result.decomposition_summary, evidence_dir / "decomp_bars.png")
    for m in FOLLOWUP_MEASUREMENTS:
        _plot_cumulative(result.date_portfolio, m, evidence_dir / f"cum_pnl_{m}.png")
        _plot_drag_by_group(result.trade_level, m, evidence_dir / f"drag_boxplot_{m}.png")

    # Markdown report
    lines = [
        "# Sprint 008 D1 cost diagnosis — report",
        "",
        f"- Evidence: `{evidence_dir}`",
        f"- Provenance: `{json.dumps(report.get('provenance'), default=str)}`",
        f"- Command: `{command}`",
        f"- Runtime s: `{report.get('total_seconds')}`",
        f"- Preserves D1 gate: `{ORIGINAL_D1_GATE}`",
        f"- Post-hoc: {report.get('post_hoc_disclosure')}",
        "",
        "## Reconciliation to prior within-date follow-up",
        "",
        "```json",
        json.dumps(result.reconciliation, indent=2, default=str),
        "```",
        "",
        "## Decomposition (date-weighted)",
        "",
        "```json",
        json.dumps(result.decomposition_summary, indent=2, default=str),
        "```",
        "",
        "## Exclusion summaries",
        "",
        "```json",
        json.dumps(result.exclusion_summary, indent=2, default=str),
        "```",
        "",
        "## Inference (family size 4)",
        "",
        "```json",
        json.dumps(result.inference, indent=2, default=str),
        "```",
        "",
        "## Interpretation",
        "",
        "```json",
        json.dumps(result.interpretation, indent=2, default=str),
        "```",
        "",
        "## Disclaimer",
        "",
        result.interpretation.get("disclaimer", ""),
        "",
    ]
    (evidence_dir / "cost_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")

    (evidence_dir / "execution_receipt.json").write_text(
        json.dumps(
            {
                "command": command,
                "provenance": report.get("provenance"),
                "environment": report.get("environment"),
                "total_seconds": report.get("total_seconds"),
                "stage_timings": result.stage_timings,
                "preserves_original_d1_gate": ORIGINAL_D1_GATE,
                "official_run_dir": str(OFFICIAL_RUN_DIR),
                "official_execution_sha": OFFICIAL_EXECUTION_REPO_SHA,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return evidence_dir
