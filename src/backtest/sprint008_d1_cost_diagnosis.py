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
# Reviewed evidence whose core economics this correction must reconcile.
# That run's drawdowns are superseded (peak omitted the initial zero).
PRIOR_COST_DIAGNOSIS_EVIDENCE = (
    "C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260911T162501Z/"
)
CORE_PNL_TOL = 1e-4


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


def fixed_budget_max_drawdown(pnl: np.ndarray) -> float:
    """Peak-to-trough of cumulative fixed-budget dollar P&L.

    The running peak is the maximum of zero and cumulative P&L observed so
    far (the path starts at $0). This is not compounded account equity and
    not intraholding-period risk.
    """
    arr = np.asarray(pnl, dtype=float)
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = 0.0
    worst = 0.0
    for level in cum:
        peak = max(peak, float(level))
        worst = min(worst, float(level) - peak)
    return float(worst)


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


def _zero_cash_date_row(measurement: str, trade_date: date) -> dict[str, Any]:
    """Verified N=0 date: original budget stays cash. Not a missing-data fill."""
    return {
        "measurement": measurement,
        "trade_date": trade_date,
        "exclude_u_applied": False,
        "n_in_N": 0,
        "n_executed": 0,
        "n_retained": 0,
        "n_excluded_u": 0,
        "R_baseline": 0.0,
        "R_filtered": 0.0,
        "R_gross": 0.0,
        "R_drag": 0.0,
        "uplift": 0.0,
        "pnl_baseline": 0.0,
        "pnl_filtered": 0.0,
        "pnl_gross": 0.0,
        "pnl_drag": 0.0,
        "pnl_u": 0.0,
        "uplift_dollars": 0.0,
        "invested_frac_baseline": 0.0,
        "invested_frac_filtered": 0.0,
        "cash_frac_filtered": 1.0,
    }


def _exclusion_date_row(
    day: pd.DataFrame,
    trade_date: date,
    measurement: str,
    eligible_dates: set[date],
) -> dict[str, Any]:
    """One entry date of baseline vs exclude-U. Accounting unchanged from D1."""
    label = f"group_{measurement}"
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

    exclude_u = trade_date in eligible_dates
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
                f"{measurement} {trade_date}: uplift identity failed ({uplift} vs {expected})"
            )

    invested_base = float(len(executed) * stake)
    invested_filt = float(len(retained) * stake)
    # Date-level gross and spread contributions on the same stake and B.
    # p_g = (B/N)*g, p_a = (B/N)*a; cash names contribute 0.
    if len(executed):
        stake_x = pd.to_numeric(executed["stake_dollars"], errors="coerce")
        pnl_gross = float((stake_x * pd.to_numeric(executed["g"], errors="coerce")).sum())
        pnl_drag = float((stake_x * pd.to_numeric(executed["a"], errors="coerce")).sum())
    else:
        pnl_gross = 0.0
        pnl_drag = 0.0
    r_gross = pnl_gross / BUDGET_B
    r_drag = pnl_drag / BUDGET_B
    if abs((r_gross - r_drag) - r_base) > ACCOUNTING_TOL:
        raise CostDiagnosisError(
            f"{measurement} {trade_date}: date gross-drag identity failed "
            f"({r_gross} - {r_drag} vs {r_base})"
        )
    return {
        "measurement": measurement,
        "trade_date": trade_date,
        "exclude_u_applied": bool(exclude_u),
        "n_in_N": n_in_n,
        "n_executed": int(len(executed)),
        "n_retained": int(len(retained)),
        "n_excluded_u": int(len(u_trades)),
        "R_baseline": r_base,
        "R_filtered": r_filt,
        "R_gross": r_gross,
        "R_drag": r_drag,
        "uplift": uplift,
        "pnl_baseline": float(np.nansum(p_all)),
        "pnl_filtered": float(np.nansum(p_ret)),
        "pnl_gross": pnl_gross,
        "pnl_drag": pnl_drag,
        "pnl_u": float(np.nansum(p_u)) if len(p_u) else 0.0,
        "uplift_dollars": float(-np.nansum(p_u)) if len(p_u) else 0.0,
        "invested_frac_baseline": invested_base / BUDGET_B,
        "invested_frac_filtered": invested_filt / BUDGET_B,
        "cash_frac_filtered": 1.0 - invested_filt / BUDGET_B,
    }


def build_portfolio_comparison(
    trade_level: pd.DataFrame,
    paired: pd.DataFrame,
    measurement: str,
    *,
    window_start: date = DEV_START,
    window_end: date = DEV_END,
    entry_calendar: pd.DataFrame | None = None,
    reporting_periods: dict[str, tuple[date, date]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Baseline vs exclude-U; one row per entry date in the requested window.

    Defaults reproduce the D1 development loop and half-period reconciliation.
    Pass ``entry_calendar`` to retain verified zero-long dates that have no
    trade rows. Those dates are not inferred from a missing panel row.
    """
    work = trade_level.copy()
    work["trade_date"] = work["trade_date"].map(_as_date)
    eligible_dates = set(
        paired.loc[paired["measurement"] == measurement, "trade_date"].map(_as_date)
    ) if not paired.empty and "measurement" in paired.columns else set()
    rows: list[dict[str, Any]] = []
    use_calendar = entry_calendar is not None

    if not use_calendar:
        for trade_date, day in work.groupby("trade_date", sort=True):
            td = _as_date(trade_date)
            if td < window_start or td > window_end:
                continue
            rows.append(_exclusion_date_row(day, td, measurement, eligible_dates))
    else:
        cal = entry_calendar.copy()
        required = {"trade_date", "n_in_N", "long_book_class"}
        missing = required - set(cal.columns)
        if missing:
            raise CostDiagnosisError(
                f"entry_calendar missing columns: {sorted(missing)}"
            )
        cal["trade_date"] = cal["trade_date"].map(_as_date)
        if cal["trade_date"].duplicated().any():
            raise CostDiagnosisError("entry_calendar has duplicate trade_date")
        allowed = {"verified_zero_long", "has_long_candidates"}
        classes = set(cal["long_book_class"].astype(str))
        if not classes.issubset(allowed):
            raise CostDiagnosisError(
                f"entry_calendar has unknown long_book_class: {sorted(classes - allowed)}"
            )
        cal = cal.sort_values("trade_date").reset_index(drop=True)
        cal_in_window = cal.loc[
            (cal["trade_date"] >= window_start) & (cal["trade_date"] <= window_end)
        ]
        cal_dates = set(cal_in_window["trade_date"].tolist())
        panel_in_window = {
            _as_date(d)
            for d in work["trade_date"].unique()
            if window_start <= _as_date(d) <= window_end
        }
        extra = sorted(panel_in_window - cal_dates)
        if extra:
            raise CostDiagnosisError(
                f"{measurement}: in-window panel dates missing from entry_calendar: "
                f"{extra[:10]}"
            )
        day_map = {td: day for td, day in work.groupby("trade_date", sort=True)}
        empty_day = work.iloc[0:0]
        for rec in cal_in_window.itertuples(index=False):
            td = _as_date(rec.trade_date)
            cls = str(rec.long_book_class)
            n_cal = int(rec.n_in_N)
            day = day_map.get(td, empty_day)
            if cls == "verified_zero_long":
                if n_cal != 0:
                    raise CostDiagnosisError(
                        f"{measurement} {td}: verified_zero_long requires n_in_N=0"
                    )
                if not day.empty:
                    in_n = day["in_N"] == True if "in_N" in day.columns else False  # noqa: E712
                    structure_ok = (
                        day["structure_ok"] == True  # noqa: E712
                        if "structure_ok" in day.columns
                        else False
                    )
                    if bool(np.any(in_n) or np.any(structure_ok)):
                        raise CostDiagnosisError(
                            f"{measurement} {td}: verified_zero_long has structure_ok "
                            "or in_N panel rows"
                        )
                rows.append(_zero_cash_date_row(measurement, td))
                continue
            if n_cal <= 0:
                raise CostDiagnosisError(
                    f"{measurement} {td}: has_long_candidates requires n_in_N>0"
                )
            if day.empty:
                raise CostDiagnosisError(
                    f"{measurement} {td}: has_long_candidates but no panel rows"
                )
            n_panel = int((day["in_N"] == True).sum())  # noqa: E712
            if n_panel != n_cal:
                raise CostDiagnosisError(
                    f"{measurement} {td}: panel in_N count {n_panel} != calendar {n_cal}"
                )
            rows.append(_exclusion_date_row(day, td, measurement, eligible_dates))

    wdates = work["trade_date"].map(_as_date)
    work = work.loc[(wdates >= window_start) & (wdates <= window_end)].copy()
    port = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    label = f"group_{measurement}"

    # Retention metrics over the requested window (pooled dollars)
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

    # Cumulative fixed-budget P&L. Peak includes the initial $0 starting point.
    cum_base = port["pnl_baseline"].cumsum()
    cum_filt = port["pnl_filtered"].cumsum()
    dd_base = fixed_budget_max_drawdown(port["pnl_baseline"].to_numpy(dtype=float))
    dd_filt = fixed_budget_max_drawdown(port["pnl_filtered"].to_numpy(dtype=float))

    summary = {
        "measurement": measurement,
        "n_dates": int(len(port)),
        "n_dates_exclusion_applied": int(port["exclude_u_applied"].sum()),
        "mean_R_baseline": float(port["R_baseline"].mean()),
        "mean_R_filtered": float(port["R_filtered"].mean()),
        "mean_R_gross": float(port["R_gross"].mean()),
        "mean_R_drag": float(port["R_drag"].mean()),
        "mean_uplift": float(port["uplift"].mean()),
        "total_pnl_baseline": float(port["pnl_baseline"].sum()),
        "total_pnl_filtered": float(port["pnl_filtered"].sum()),
        "total_pnl_gross": float(port["pnl_gross"].sum()),
        "total_pnl_drag": float(port["pnl_drag"].sum()),
        "total_pnl_improvement": pnl_improvement,
        "losses_avoided": losses_avoided,
        "winning_profits_sacrificed": winning_sacrificed,
        "identity_improvement_ok": abs(pnl_improvement - float(-np.sum(excl_p)))
        <= ACCOUNTING_TOL,
        "date_gross_minus_drag_ok": bool(
            ((port["R_gross"] - port["R_drag"] - port["R_baseline"]).abs().max() <= ACCOUNTING_TOL)
        ),
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
        "peak_to_trough_baseline": dd_base,
        "peak_to_trough_filtered": dd_filt,
        "note_drawdown": (
            "Peak-to-trough of cumulative fixed-budget dollar P&L. "
            "Running peak is max(0, cumulative P&L so far). "
            "Not compounded account equity and not intraholding-period risk. "
            + (
                "Complete requested window under the fixed-budget convention."
                if reporting_periods is not None
                else "Complete development period under the fixed-budget convention."
            )
        ),
        "weekly_uplift": weekly_uplift_distribution(port),
    }
    full_map = {
        "pnl_baseline": summary["total_pnl_baseline"],
        "pnl_filtered": summary["total_pnl_filtered"],
        "losses_avoided": summary["losses_avoided"],
        "winning_profits_sacrificed": summary["winning_profits_sacrificed"],
    }
    if reporting_periods is None:
        summary["half_periods"] = {
            "2020-2021": exclusion_window_metrics(
                port, executed_all, DEV_A_START, DEV_A_END
            ),
            "2022-2023": exclusion_window_metrics(
                port, executed_all, DEV_B_START, DEV_B_END
            ),
        }
        h1 = summary["half_periods"]["2020-2021"]
        h2 = summary["half_periods"]["2022-2023"]
        half_ok = True
        for key, full in full_map.items():
            if abs((h1[key] + h2[key]) - full) > 0.05:
                half_ok = False
                raise CostDiagnosisError(
                    f"{measurement}: half-period {key} does not reconcile "
                    f"({h1[key]} + {h2[key]} vs {full})"
                )
        summary["half_period_reconcile_ok"] = half_ok
    else:
        period_metrics: dict[str, Any] = {}
        for name, bounds in reporting_periods.items():
            start, end = bounds
            period_metrics[name] = exclusion_window_metrics(
                port, executed_all, start, end
            )
        period_ok = True
        for key, full in full_map.items():
            parts = float(sum(period_metrics[name][key] for name in period_metrics))
            if abs(parts - full) > 0.05:
                period_ok = False
                raise CostDiagnosisError(
                    f"{measurement}: reporting-period {key} does not reconcile "
                    f"({parts} vs {full})"
                )
        summary["reporting_periods"] = period_metrics
        summary["period_reconcile_ok"] = period_ok
    port["cum_pnl_baseline"] = cum_base
    port["cum_pnl_filtered"] = cum_filt
    return port, summary


def exclusion_window_metrics(
    port: pd.DataFrame,
    executed_all: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, Any]:
    """Descriptive fixed-exclusion stats on a date window. No significance test."""
    dates = port["trade_date"].map(_as_date)
    sub = port.loc[(dates >= start) & (dates <= end)]
    edates = executed_all["trade_date"].map(_as_date)
    exec_w = executed_all.loc[(edates >= start) & (edates <= end)]
    excl_w = exec_w.loc[exec_w["_excl"].to_numpy()]
    ret_w = exec_w.loc[~exec_w["_excl"].to_numpy()]
    excl_p = excl_w["p"].to_numpy(dtype=float) if len(excl_w) else np.array([])
    base_p = exec_w["p"].to_numpy(dtype=float) if len(exec_w) else np.array([])
    ret_p = ret_w["p"].to_numpy(dtype=float) if len(ret_w) else np.array([])
    losses_avoided = float(-np.sum(excl_p[excl_p < 0])) if len(excl_p) else 0.0
    winning_sacrificed = float(np.sum(excl_p[excl_p > 0])) if len(excl_p) else 0.0
    sum_win_base = float(np.sum(base_p[base_p > 0])) if len(base_p) else 0.0
    sum_win_ret = float(np.sum(ret_p[ret_p > 0])) if len(ret_p) else 0.0
    return {
        "start": str(start),
        "end": str(end),
        "n_dates": int(len(sub)),
        "pnl_baseline": float(sub["pnl_baseline"].sum()) if len(sub) else 0.0,
        "pnl_filtered": float(sub["pnl_filtered"].sum()) if len(sub) else 0.0,
        "mean_uplift": float(sub["uplift"].mean()) if len(sub) else None,
        "losses_avoided": losses_avoided,
        "winning_profits_sacrificed": winning_sacrificed,
        "winning_profit_retention": (
            sum_win_ret / sum_win_base if sum_win_base > 0 else None
        ),
        "weighting": "descriptive_split_no_new_test",
    }


def weekly_uplift_distribution(port: pd.DataFrame) -> dict[str, Any]:
    """Distribution of weekly uplift on the complete development calendar."""
    if port.empty:
        return {"n_dates": 0}
    uplift = port["uplift"].to_numpy(dtype=float)
    dollars = port["uplift_dollars"].to_numpy(dtype=float)
    zero_tol = 1e-8  # dollars
    n = int(len(port))
    n_pos = int(np.sum(dollars > zero_tol))
    n_neg = int(np.sum(dollars < -zero_tol))
    n_zero = n - n_pos - n_neg
    pos_total = float(dollars[dollars > zero_tol].sum()) if n_pos else 0.0
    neg_abs_total = float((-dollars[dollars < -zero_tol]).sum()) if n_neg else 0.0

    pos = port.loc[port["uplift_dollars"] > zero_tol]
    neg = port.loc[port["uplift_dollars"] < -zero_tol]
    top_pos = pos.nlargest(5, "uplift_dollars") if len(pos) else pos
    top_neg = neg.nsmallest(5, "uplift_dollars") if len(neg) else neg

    def _weeks(frame: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for row in frame.itertuples(index=False):
            out.append(
                {
                    "trade_date": str(_as_date(row.trade_date)),
                    "uplift": float(row.uplift),
                    "dollar_contribution": float(row.uplift_dollars),
                }
            )
        return out

    top_pos_sum = float(top_pos["uplift_dollars"].sum()) if len(top_pos) else 0.0
    top_neg_abs = float((-top_neg["uplift_dollars"]).sum()) if len(top_neg) else 0.0
    return {
        "n_dates": n,
        "mean": float(np.mean(uplift)),
        "median": float(np.median(uplift)),
        "std": float(np.std(uplift, ddof=1)) if n > 1 else None,
        "frac_positive": n_pos / n,
        "frac_zero": n_zero / n,
        "frac_negative": n_neg / n,
        "n_positive": n_pos,
        "n_zero": n_zero,
        "n_negative": n_neg,
        "total_positive_dollar_contributions": pos_total,
        "total_absolute_negative_dollar_contributions": neg_abs_total,
        "five_largest_positive_weeks": _weeks(top_pos),
        "five_largest_negative_weeks": _weeks(top_neg),
        "top5_positive_share_of_positive_contributions": (
            top_pos_sum / pos_total if pos_total > 0 else None
        ),
        "top5_negative_share_of_absolute_negative_contributions": (
            top_neg_abs / neg_abs_total if neg_abs_total > 0 else None
        ),
        "note": (
            "Concentration is versus total positive weekly dollar contributions "
            "and total absolute negative weekly dollar contributions separately. "
            "Not versus the small net improvement. "
            "Winner-profit retention is retention of baseline winners, not "
            "concentration of the filter's incremental improvement."
        ),
    }


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
    weighting: dict[str, Any],
) -> dict[str, Any]:
    """Record observed facts. Do not issue an automatic strategy decision."""
    base = win_diag.get("M1", {}).get("unfiltered_baseline", {})
    facts: dict[str, Any] = {
        "cost_savings_are_intended_mechanism": (
            "Lower execution cost can improve net economics without predicting "
            "a better gross payoff. A cost contribution is not evidence against usefulness."
        ),
        "pooled_baseline": base,
        "weighting": weighting,
        "cost_contribution": {},
        "gross_payoff_offset": {},
        "whole_book_improvement": {},
        "uncertainty_and_winner_sacrifice": {},
        "limits_on_inference": [
            "An insignificant gross-return difference does not establish equivalence.",
            "An insignificant uplift does not establish no benefit.",
            "An interval that includes zero does not establish a zero effect.",
            "No automatic next-experiment recommendation is issued.",
            "Final strategy decision awaits review.",
        ],
        "unresolved": [
            "Fees remain unmodeled.",
            "Quote-based full-cross results do not establish achievable fills or dependable income.",
            "Post-hoc relative to D1 and the within-date follow-up; not independent confirmation.",
            "Evaluation-period outcomes remain closed.",
            "Broader threshold search remains unauthorized. Historical D1 STOP_NO_THRESHOLDS is preserved.",
        ],
    }
    for m in FOLLOWUP_MEASUREMENTS:
        d = decomp[m]
        excl = exclusion[m]
        uplift_inf = inference["contrasts"][f"{m}_mean_uplift"]
        facts["cost_contribution"][m] = {
            "date_weighted_mean_spread_saving_pp": None
            if d.get("mean_spread_saving") is None
            else 100.0 * float(d["mean_spread_saving"]),
            "date_weighted_mean_a_U_minus_L_pp": None
            if d.get("mean_a_U_minus_L") is None
            else 100.0 * float(d["mean_a_U_minus_L"]),
            "comment": (
                "This is the observed within-date cost contribution of L versus U "
                "on the full-cross capital basis (percentage points)."
            ),
        }
        facts["gross_payoff_offset"][m] = {
            "date_weighted_mean_d_gross_pp": None
            if d.get("mean_d_gross") is None
            else 100.0 * float(d["mean_d_gross"]),
            "date_weighted_mean_d_net_pp": None
            if d.get("mean_d_net") is None
            else 100.0 * float(d["mean_d_net"]),
            "comment": (
                "Gross L−U is an observational payoff difference, not a test of "
                "equivalence. Identity: d_net = d_gross + spread_saving."
            ),
        }
        facts["whole_book_improvement"][m] = {
            "total_pnl_baseline": excl.get("total_pnl_baseline"),
            "total_pnl_filtered": excl.get("total_pnl_filtered"),
            "total_pnl_improvement": excl.get("total_pnl_improvement"),
            "mean_uplift_pp": None
            if excl.get("mean_uplift") is None
            else 100.0 * float(excl["mean_uplift"]),
            "half_periods": excl.get("half_periods"),
            "comment": (
                "Historical whole-book improvement on original B with rejected "
                "stakes left in cash. Point improvement is not a significance claim."
            ),
        }
        facts["uncertainty_and_winner_sacrifice"][m] = {
            "hac_mean_uplift": {
                "n": uplift_inf.get("n"),
                "mean": uplift_inf.get("mean"),
                "se_hac": uplift_inf.get("hac", {}).get("se_hac"),
                "p_raw": uplift_inf.get("hac", {}).get("p_raw"),
                "p_adjusted": uplift_inf.get("p_adjusted"),
                "ci_ordinary_95": uplift_inf.get("hac", {}).get("ci_ordinary"),
                "ci_adjusted_98_75": uplift_inf.get("hac", {}).get("ci_adjusted"),
                "interval_includes_zero": True,
            },
            "winning_profit_retention": excl.get("winning_profit_retention"),
            "top5_profit_retention": excl.get("top5_profit_retention"),
            "top10_profit_retention": excl.get("top10_profit_retention"),
            "weekly_concentration": {
                "top5_positive_share_of_positive_contributions": excl.get(
                    "weekly_uplift", {}
                ).get("top5_positive_share_of_positive_contributions"),
                "top5_negative_share_of_absolute_negative_contributions": excl.get(
                    "weekly_uplift", {}
                ).get("top5_negative_share_of_absolute_negative_contributions"),
            },
            "comment": (
                "Winning-profit retention measures how much baseline winning "
                "profit is kept. It is not a statement about concentration of "
                "the filter's incremental improvement. Weekly concentration is "
                "reported against total positive and total absolute negative "
                "weekly contributions separately."
            ),
        }
        ci = uplift_inf.get("hac", {}).get("ci_ordinary") or [None, None]
        facts["uncertainty_and_winner_sacrifice"][m]["hac_mean_uplift"][
            "interval_includes_zero"
        ] = bool(ci[0] is not None and ci[0] <= 0 <= ci[1])
    facts["decision_status"] = "awaiting_review"
    facts["recommendation"] = None
    return {
        "facts": facts,
        "decision_status": "awaiting_review",
        "recommendation": None,
        "disclaimer": (
            "Fees remain unmodeled; quote-based results do not establish achievable "
            "fills or dependable income. No automatic close of this research direction."
        ),
    }


def _weighting_bridge(
    win_diag: dict[str, Any],
    baseline_port: pd.DataFrame,
) -> dict[str, Any]:
    """Separate pooled trade means from date-weighted returns on original B."""
    pooled = win_diag.get("M1", {}).get("unfiltered_baseline", {})
    mean_r_pooled = pooled.get("mean_r")
    mean_g_pooled = pooled.get("mean_g")
    mean_a_pooled = pooled.get("mean_a")
    mean_r_date = float(baseline_port["R_baseline"].mean()) if len(baseline_port) else None
    mean_g_date = float(baseline_port["R_gross"].mean()) if len(baseline_port) else None
    mean_a_date = float(baseline_port["R_drag"].mean()) if len(baseline_port) else None
    total_pnl = float(baseline_port["pnl_baseline"].sum()) if len(baseline_port) else 0.0
    n_dates = int(len(baseline_port))
    identity_ok = bool(
        ((baseline_port["R_gross"] - baseline_port["R_drag"] - baseline_port["R_baseline"]).abs().max()
         <= ACCOUNTING_TOL)
        if len(baseline_port)
        else True
    )
    return {
        "pooled_trade_weighted": {
            "definition": (
                "Equal weight per executed trade. mean(g), mean(a), mean(r) "
                "over analysis-eligible executed trades. Not a return on B."
            ),
            "mean_g": mean_g_pooled,
            "mean_a": mean_a_pooled,
            "mean_r": mean_r_pooled,
            "n_trades": pooled.get("n"),
        },
        "date_weighted_on_B": {
            "definition": (
                "Equal weight per development entry date. "
                "R_t = sum_i p_i / B including cash (rejected stakes contribute 0). "
                "G_t = sum_i (B/N) g_i / B; A_t = sum_i (B/N) a_i / B; "
                "verified G_t - A_t = R_t."
            ),
            "mean_G": mean_g_date,
            "mean_A": mean_a_date,
            "mean_R": mean_r_date,
            "n_dates": n_dates,
            "total_pnl_dollars": total_pnl,
            "gross_minus_drag_equals_net": identity_ok,
            "coverage": "complete development calendar under fixed-budget convention",
        },
        "why_pooled_mean_can_differ_from_dollar_pnl": (
            "Total dollar P&L equals B times the sum of date-level returns on B, "
            "so its sign follows the date-equal-weighted mean of R_t. "
            "The pooled trade mean weights each trade equally, so dates with larger N "
            "receive more weight. When N varies, a slightly negative pooled trade mean "
            "can coexist with positive total dollar P&L."
        ),
    }


def reconcile_prior_core_pnl(exclusion: dict[str, Any]) -> dict[str, Any]:
    """Compare core P&L to the reviewed 2026-09-11 evidence. Drawdown may differ."""
    prior_path = Path(PRIOR_COST_DIAGNOSIS_EVIDENCE) / "cost_diagnosis_report.json"
    out: dict[str, Any] = {
        "prior_evidence": PRIOR_COST_DIAGNOSIS_EVIDENCE,
        "prior_available": prior_path.exists(),
        "drawdown_expected_to_change": (
            "Prior peak omitted the initial $0. Corrected peak is max(0, cumulative P&L so far)."
        ),
        "measurements": {},
    }
    if not prior_path.exists():
        out["note"] = "Prior evidence JSON not found; core reconciliation skipped."
        return out
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_excl = prior.get("exclusion", {})
    for m in FOLLOWUP_MEASUREMENTS:
        cur = exclusion[m]
        old = prior_excl.get(m, {})
        checks = {
            "total_pnl_baseline": (
                cur.get("total_pnl_baseline"),
                old.get("total_pnl_baseline"),
            ),
            "total_pnl_filtered": (
                cur.get("total_pnl_filtered"),
                old.get("total_pnl_filtered"),
            ),
            "total_pnl_improvement": (
                cur.get("total_pnl_improvement"),
                old.get("total_pnl_improvement"),
            ),
            "n_excluded_u_trades": (
                cur.get("n_excluded_u_trades"),
                old.get("n_excluded_u_trades"),
            ),
            "winning_profit_retention": (
                cur.get("winning_profit_retention"),
                old.get("winning_profit_retention"),
            ),
        }
        diffs = {}
        ok = True
        for key, (now, then) in checks.items():
            if then is None or now is None:
                ok = False
                diffs[key] = {"now": now, "prior": then, "match": False}
                continue
            match = abs(float(now) - float(then)) <= CORE_PNL_TOL
            ok = ok and match
            diffs[key] = {"now": now, "prior": then, "match": match}
        dd_now = cur.get("peak_to_trough_baseline")
        dd_old = old.get("peak_to_trough_baseline")
        out["measurements"][m] = {
            "core_pnl_match": ok,
            "checks": diffs,
            "prior_peak_to_trough_baseline": dd_old,
            "corrected_peak_to_trough_baseline": dd_now,
            "prior_peak_to_trough_filtered": old.get("peak_to_trough_filtered"),
            "corrected_peak_to_trough_filtered": cur.get("peak_to_trough_filtered"),
        }
    return out


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
    weighting = _weighting_bridge(win_diag, portfolios["M1"])
    core_reconcile = reconcile_prior_core_pnl(exclusion_summaries)

    t_inf = time.perf_counter()
    inference = run_inference(paired_df, portfolios)
    timings["inference"] = _progress("inference", t_inf)

    interpretation = build_interpretation(
        decomp=decomp_summaries,
        dispersion=dispersion,
        exclusion=exclusion_summaries,
        inference=inference,
        win_diag=win_diag,
        weighting=weighting,
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
        "core_pnl_reconciliation": core_reconcile,
        "weighting": weighting,
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
    ax.plot(
        [sub["trade_date"].iloc[0], *sub["trade_date"]],
        [0.0, *sub["cum_pnl_baseline"]],
        label="baseline",
    )
    ax.plot(
        [sub["trade_date"].iloc[0], *sub["trade_date"]],
        [0.0, *sub["cum_pnl_filtered"]],
        label="exclude U",
    )
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


def _pp(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f} pp"


def _usd(value: Any) -> str:
    if value is None:
        return "NA"
    return f"${float(value):,.2f}"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def render_readable_report(
    result: CostDiagnosisResult,
    *,
    command: str,
    evidence_dir: Path,
) -> str:
    """Numerical evidence reviewable without opening JSON."""
    report = result.report
    lines = [
        "# Sprint 008 D1 cost diagnosis — corrected evidence report",
        "",
        f"- Evidence directory: `{evidence_dir}`",
        f"- Command: `{command}`",
        f"- Runtime (s): {_num(report.get('total_seconds'), 2)}",
        f"- Code SHA: `{report.get('provenance', {}).get('code_sha')}`",
        f"- Working tree: `{report.get('provenance', {}).get('working_tree')}`",
        f"- Environment: `{json.dumps(report.get('environment'), default=str)}`",
        f"- Historical D1 gate preserved: `{ORIGINAL_D1_GATE}`",
        f"- Decision status: awaiting review (no automatic recommendation)",
        f"- Post-hoc disclosure: {report.get('post_hoc_disclosure')}",
        "",
        "Return differences below are in **percentage points** (pp) unless labeled as dollars.",
        "Cumulative totals cover the **complete development period** under the fixed-budget convention ($B per entry date; cash return 0).",
        "",
        "## Weighting",
        "",
        result.report.get("weighting", {}).get("why_pooled_mean_can_differ_from_dollar_pnl", ""),
        "",
    ]
    w = report.get("weighting", {})
    pooled = w.get("pooled_trade_weighted", {})
    dated = w.get("date_weighted_on_B", {})
    lines.extend(
        [
            "| Quantity | Pooled trade-weighted | Date-weighted return on original B (includes cash) |",
            "|---|---:|---:|",
            f"| Gross | {_pp(pooled.get('mean_g'))} | {_pp(dated.get('mean_G'))} |",
            f"| Spread drag | {_pp(pooled.get('mean_a'))} | {_pp(dated.get('mean_A'))} |",
            f"| Net | {_pp(pooled.get('mean_r'))} | {_pp(dated.get('mean_R'))} |",
            f"| Sample | {pooled.get('n_trades')} trades | {dated.get('n_dates')} dates |",
            "",
            f"Date-level identity \(G_t - A_t = R_t\): **{dated.get('gross_minus_drag_equals_net')}**.",
            f"Total baseline dollar P&L: {_usd(dated.get('total_pnl_dollars'))}.",
            "",
            "## Reconciliation",
            "",
            "Within-date L−U net means vs prior follow-up:",
            "",
        ]
    )
    for m, rec in result.reconciliation.get("measurements", {}).items():
        lines.append(
            f"- {m}: eligible {rec.get('n_eligible')}, mean d_net {_num(rec.get('mean_d_net'), 6)}, "
            f"prior {_num(rec.get('prior_mean_d_net'), 6)}, match={rec.get('mean_match_prior')}"
        )
    lines.extend(["", "Core P&L vs reviewed cost-diagnosis evidence (drawdown expected to change):", ""])
    core = report.get("core_pnl_reconciliation", {})
    lines.append(f"- Prior evidence: `{core.get('prior_evidence')}`")
    lines.append(f"- {core.get('drawdown_expected_to_change')}")
    for m, rec in core.get("measurements", {}).items():
        lines.append(f"- {m} core P&L match: **{rec.get('core_pnl_match')}**")
        lines.append(
            f"  - Corrected baseline drawdown {_usd(rec.get('corrected_peak_to_trough_baseline'))} "
            f"(prior {_usd(rec.get('prior_peak_to_trough_baseline'))})"
        )
        lines.append(
            f"  - Corrected filtered drawdown {_usd(rec.get('corrected_peak_to_trough_filtered'))} "
            f"(prior {_usd(rec.get('prior_peak_to_trough_filtered'))})"
        )
    lines.extend(
        [
            "",
            "Drawdown definition: running peak = maximum of 0 and cumulative P&L so far. "
            "This is a drawdown of cumulative fixed-budget dollar P&L, not compounded equity "
            "and not intraholding-period risk.",
            "",
            "## Decomposition (date-weighted L minus U)",
            "",
            "Identity: d_net = d_gross + spread_saving. Cost savings are an intended mechanism, "
            "not evidence against usefulness.",
            "",
            "| | M1 | M2 |",
            "|---|---:|---:|",
        ]
    )
    d1 = result.decomposition_summary["M1"]
    d2 = result.decomposition_summary["M2"]
    for label, key in (
        ("d_net", "mean_d_net"),
        ("d_gross", "mean_d_gross"),
        ("spread_saving", "mean_spread_saving"),
    ):
        lines.append(f"| {label} | {_pp(d1.get(key))} | {_pp(d2.get(key))} |")
    lines.extend(
        [
            "",
            "## Fixed U exclusion — full development",
            "",
            "| | M1 | M2 |",
            "|---|---:|---:|",
        ]
    )
    e1 = result.exclusion_summary["M1"]
    e2 = result.exclusion_summary["M2"]
    rows = [
        ("Dates", "n_dates", "n"),
        ("Baseline $ P&L", "total_pnl_baseline", "usd"),
        ("Filtered $ P&L", "total_pnl_filtered", "usd"),
        ("Improvement $", "total_pnl_improvement", "usd"),
        ("Mean weekly uplift", "mean_uplift", "pp"),
        ("Losses avoided $", "losses_avoided", "usd"),
        ("Winning profits sacrificed $", "winning_profits_sacrificed", "usd"),
        ("Winning-profit retention", "winning_profit_retention", "pct"),
        ("Top-5 winner-profit retention", "top5_profit_retention", "pct"),
        ("Top-10 winner-profit retention", "top10_profit_retention", "pct"),
        ("Baseline drawdown $", "peak_to_trough_baseline", "usd"),
        ("Filtered drawdown $", "peak_to_trough_filtered", "usd"),
        ("Half-period $ reconcile", "half_period_reconcile_ok", "raw"),
    ]
    for label, key, kind in rows:
        def _fmt(val: Any, kind: str = kind) -> str:
            if kind == "usd":
                return _usd(val)
            if kind == "pp":
                return _pp(val)
            if kind == "pct":
                return _pct(val)
            return str(val)

        lines.append(f"| {label} | {_fmt(e1.get(key))} | {_fmt(e2.get(key))} |")
    lines.extend(
        [
            "",
            "Winning-profit retention is retention of **baseline winners**. "
            "It is not concentration of the filter's incremental improvement.",
            "",
            "## Fixed U exclusion — descriptive halves (no new tests)",
            "",
        ]
    )
    for m in FOLLOWUP_MEASUREMENTS:
        lines.append(f"### {m}")
        lines.append("")
        lines.append(
            "| Half | Dates | Baseline $ | Filtered $ | Mean weekly uplift | Losses avoided $ | Winning profits sacrificed $ | Winning-profit retention |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        halves = result.exclusion_summary[m].get("half_periods", {})
        for name in ("2020-2021", "2022-2023"):
            h = halves.get(name, {})
            lines.append(
                f"| {name} | {h.get('n_dates')} | {_usd(h.get('pnl_baseline'))} | "
                f"{_usd(h.get('pnl_filtered'))} | {_pp(h.get('mean_uplift'))} | "
                f"{_usd(h.get('losses_avoided'))} | {_usd(h.get('winning_profits_sacrificed'))} | "
                f"{_pct(h.get('winning_profit_retention'))} |"
            )
        lines.append("")
        wk = result.exclusion_summary[m].get("weekly_uplift", {})
        lines.extend(
            [
                f"Weekly uplift ({m}), complete calendar (n={wk.get('n_dates')}):",
                f"- Mean {_pp(wk.get('mean'))}; median {_pp(wk.get('median'))}; std {_pp(wk.get('std'))}.",
                f"- Fractions positive / zero / negative: {_pct(wk.get('frac_positive'))} / "
                f"{_pct(wk.get('frac_zero'))} / {_pct(wk.get('frac_negative'))}.",
                f"- Five largest positive weeks' share of **total positive** dollar contributions: "
                f"{_pct(wk.get('top5_positive_share_of_positive_contributions'))}.",
                f"- Five largest negative weeks' share of **total absolute negative** dollar contributions: "
                f"{_pct(wk.get('top5_negative_share_of_absolute_negative_contributions'))}.",
                "",
                "Largest positive weeks:",
                "",
            ]
        )
        for row in wk.get("five_largest_positive_weeks", []):
            lines.append(
                f"- {row.get('trade_date')}: {_usd(row.get('dollar_contribution'))} "
                f"({_pp(row.get('uplift'))})"
            )
        lines.extend(["", "Largest negative weeks:", ""])
        for row in wk.get("five_largest_negative_weeks", []):
            lines.append(
                f"- {row.get('trade_date')}: {_usd(row.get('dollar_contribution'))} "
                f"({_pp(row.get('uplift'))})"
            )
        lines.append("")
    lines.extend(
        [
            "## Frozen inference (family size 4)",
            "",
            "HAC: maxlags=3, Bartlett kernel, small-sample correction, Student-t with T−1 df. "
            "Adjusted p = min(1, 4 × raw p). Adjusted interval is 98.75%.",
            "",
            "| Contrast | n | Point estimate | HAC SE | Ordinary 95% CI | Raw p | Adjusted p | Adjusted 98.75% CI |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for name, contrast in result.inference.get("contrasts", {}).items():
        hac = contrast.get("hac", {})
        unit = "pp" if "uplift" in name or "winrate" in name else ""
        point = _pp(contrast.get("mean")) if unit else _num(contrast.get("mean"))
        se = _pp(hac.get("se_hac")) if unit else _num(hac.get("se_hac"))
        ci95 = hac.get("ci_ordinary") or [None, None]
        ciadj = hac.get("ci_adjusted") or [None, None]
        lines.append(
            f"| {name} | {contrast.get('n')} | {point} | {se} | "
            f"[{_pp(ci95[0])}, {_pp(ci95[1])}] | {_num(hac.get('p_raw'), 4)} | "
            f"{_num(contrast.get('p_adjusted'), 4)} | "
            f"[{_pp(ciadj[0])}, {_pp(ciadj[1])}] |"
        )
    lines.extend(
        [
            "",
            "An interval that includes zero does not establish no effect. "
            "An insignificant gross difference does not establish equivalence.",
            "",
            "## Interpretation (facts only; decision awaiting review)",
            "",
            result.interpretation.get("facts", {}).get(
                "cost_savings_are_intended_mechanism", ""
            ),
            "",
        ]
    )
    for note in result.interpretation.get("facts", {}).get("limits_on_inference", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Limitations", ""])
    for note in result.interpretation.get("facts", {}).get("unresolved", []):
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            result.interpretation.get("disclaimer", ""),
            "",
        ]
    )
    return "\n".join(lines)


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

    # Markdown report (readable without external JSON)
    (evidence_dir / "cost_diagnosis_report.md").write_text(
        render_readable_report(result, command=command, evidence_dir=evidence_dir),
        encoding="utf-8",
    )

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
