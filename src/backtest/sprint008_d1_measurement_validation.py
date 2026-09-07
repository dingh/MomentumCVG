"""
Sprint 008 D1 — measurement validation and gate decision.

Development-only association of M1/M2/M3 with equal-dollar net return under
frozen quintiles and Bonferroni block-bootstrap. Reuses D0 panel construction.
No thresholds, no evaluation-period analysis, no SurfaceRunner.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_EXECUTION_REPO_SHA,
    OFFICIAL_RUN_DIR,
    TRADE_KEY,
    get_current_repo_sha,
)
from src.backtest.sprint008_d0_input_readiness import (
    ACCOUNTING_TOL,
    BUDGET_B,
    CROSSED_QUOTE_POLICY_VERSION,
    DOLLAR_TOL,
    FEES,
    MAX_NAMES,
    SCENARIOS_H,
    attach_outcomes_and_measurements,
    compute_m3_scores,
    compute_package_mh,
    equal_dollar_quantities,
    load_full_long_leg_log,
    load_full_long_trade_log,
    reconstruct_capped_long_n,
)

DEV_START = date(2020, 1, 1)
DEV_END = date(2023, 12, 31)
DEV_A_START = date(2020, 1, 1)
DEV_A_END = date(2021, 12, 31)
DEV_B_START = date(2022, 1, 1)
DEV_B_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)

BLOCK_LEN = 4
N_BOOT = 10_000
SEED = 20260907
FAMILY_SIZE = 3
ALPHA = 0.05
BONFERRONI_LO = ALPHA / (2 * FAMILY_SIZE)  # 0.05/6
BONFERRONI_HI = 1.0 - ALPHA / (2 * FAMILY_SIZE)  # 1 - 0.05/6
ORDINARY_LO = 0.025
ORDINARY_HI = 0.975

MEASUREMENTS = ("M1", "M2", "M3")
PRIMARY_H = 1.0
SENSITIVITY_H = (0.0, 0.25, 0.50)
N_QUINTILES = 5
MIN_VALID_BOOT_DELTA = 1000
ECON_BAR = 0.05
WRONG_SIGN_SOFT = 0.02
MIN_Q_N_FLOOR = 100
MIN_Q_N_FRAC = 0.05
MIN_Q_DATES = 20
MIN_GROUP_FOR_DELTA = 2

LABEL_SUPPORTED = "supported"
LABEL_UNSUPPORTED = "unsupported"
LABEL_INCONCLUSIVE = "inconclusive"

GATE_AUTHORIZE = "AUTHORIZE_D2"
GATE_STOP = "STOP_NO_THRESHOLDS"


class D1ValidationError(Exception):
    """Raised when D1 construction or reconciliation cannot proceed."""


@dataclass
class D1ValidationResult:
    labels: dict[str, str] = field(default_factory=dict)
    predicates: dict[str, dict[str, bool]] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    delta_tables: dict[str, Any] = field(default_factory=dict)
    quintile_tables: pd.DataFrame = field(default_factory=pd.DataFrame)
    within_date: dict[str, Any] = field(default_factory=dict)
    stability_halves: dict[str, Any] = field(default_factory=dict)
    spearman_diagnostic: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    winner_attribution: dict[str, Any] = field(default_factory=dict)
    panel_dev: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysis_panels: dict[str, pd.DataFrame] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)


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
    print(f"[D1 validation] {stage}: {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def is_development_date(value: Any) -> bool:
    d = _as_date(value)
    return DEV_START <= d <= DEV_END


def is_evaluation_date(value: Any) -> bool:
    return _as_date(value) >= EVAL_START


def bonferroni_quantiles(*, alpha: float = ALPHA, family_size: int = FAMILY_SIZE) -> tuple[float, float]:
    lo = alpha / (2 * family_size)
    hi = 1.0 - alpha / (2 * family_size)
    return float(lo), float(hi)


def build_d1_base_panel(
    *,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """Load official artifacts and reconstruct capped in_N panel with M1–M3.

    M3 history may use pre-development completed trades; association filtering
    happens later. Raises on unexpected D0-style reconciliation failures.
    """
    run_dir = run_dir or OFFICIAL_RUN_DIR
    mid_trades = load_full_long_trade_log(run_dir, fill_label="mid")
    mid_legs = load_full_long_leg_log(run_dir, fill_label="mid")
    reconstructed = reconstruct_capped_long_n(mid_trades)

    constructable_keys = reconstructed.loc[
        reconstructed["structure_ok"] == True, list(TRADE_KEY)  # noqa: E712
    ]
    legs_constructable = mid_legs.merge(constructable_keys, on=list(TRADE_KEY), how="inner")
    package_mh = compute_package_mh(legs_constructable)
    constructable = reconstructed.loc[reconstructed["structure_ok"] == True].copy()  # noqa: E712
    constructable_panel = attach_outcomes_and_measurements(constructable, package_mh)
    constructable_panel = compute_m3_scores(constructable_panel)
    panel = constructable_panel.loc[constructable_panel["in_N"] == True].copy()  # noqa: E712

    # Unexpected midpoint / payoff reconciliation failures → hard error.
    if not panel.empty:
        mid_bad = panel["delta_M_vs_stored"].map(_finite) & (
            panel["delta_M_vs_stored"].abs() > DOLLAR_TOL
        )
        if bool(mid_bad.any()):
            raise D1ValidationError(
                f"Unexpected midpoint authority failures: n={int(mid_bad.sum())}"
            )
        outcome_ok = panel["outcome_finite"] == True  # noqa: E712
        pay_bad = outcome_ok & (panel["payoff_reconcile_ok"] == False)  # noqa: E712
        if bool(pay_bad.any()):
            raise D1ValidationError(
                f"Unexpected payoff reconciliation failures: n={int(pay_bad.sum())}"
            )
    return panel.reset_index(drop=True)


def filter_development_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep development window only; drop evaluation and pre-dev association rows."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    out["trade_date"] = out["trade_date"].map(_as_date)
    if out["trade_date"].map(is_evaluation_date).any():
        # Firewall: analysis copies never retain eval rows.
        out = out.loc[~out["trade_date"].map(is_evaluation_date)].copy()
    mask = out["trade_date"].map(is_development_date)
    return out.loc[mask].reset_index(drop=True)


def assert_no_evaluation_rows(frame: pd.DataFrame, *, context: str) -> None:
    if frame.empty or "trade_date" not in frame.columns:
        return
    n_eval = int(frame["trade_date"].map(is_evaluation_date).sum())
    if n_eval:
        raise D1ValidationError(f"{context}: found {n_eval} evaluation-period rows")


def attach_scenario_economics(panel: pd.DataFrame, h: float) -> pd.DataFrame:
    """Attach scenario-specific q, C, r, g, d and dollar identities.

    Quantities are computed for this ``h`` only via ``equal_dollar_quantities``;
    never reuse another scenario's q.
    """
    sized = equal_dollar_quantities(panel, float(h))
    out = sized.copy()
    h = float(h)
    m = pd.to_numeric(out["M"], errors="coerce")
    hh = pd.to_numeric(out["H"], errors="coerce")
    x = pd.to_numeric(out["X"], errors="coerce")
    q = pd.to_numeric(out["q_h"], errors="coerce")
    c = m + h * hh + FEES
    out["C"] = c
    out["h_scenario"] = h

    eligible = out.get("analysis_eligible", pd.Series(True, index=out.index)).astype(bool)
    valid = (
        eligible
        & c.map(_finite)
        & (c > 0.0)
        & x.map(_finite)
        & (x >= 0.0)
        & q.map(_finite)
        & (q > 0.0)
    )
    out["assoc_valid"] = valid

    r = np.where(valid.to_numpy(), (x - c) / c, np.nan)
    g = np.where(valid.to_numpy(), (x - m) / c, np.nan)
    d = np.where(valid.to_numpy(), (h * hh + FEES) / c, np.nan)
    out["r"] = r
    out["g"] = g
    out["d"] = d
    out["dollar_net"] = np.where(valid.to_numpy(), q * (x - c), np.nan)
    out["dollar_gross"] = np.where(valid.to_numpy(), q * (x - m), np.nan)
    out["dollar_drag"] = np.where(valid.to_numpy(), q * (h * hh + FEES), np.nan)
    out["stake_invested"] = np.where(valid.to_numpy(), q * c, np.nan)
    return out


def reconcile_economics(panel: pd.DataFrame, *, tol: float = ACCOUNTING_TOL) -> dict[str, Any]:
    """Verify gross−drag = net and q*C = stake for association-valid rows."""
    work = panel.loc[panel.get("assoc_valid", False) == True].copy()  # noqa: E712
    if work.empty:
        return {"passed": True, "n": 0, "max_abs_net_err": 0.0, "max_abs_stake_err": 0.0}

    net_err = (work["dollar_gross"] - work["dollar_drag"] - work["dollar_net"]).abs()
    stake_target = pd.to_numeric(work["stake_dollars"], errors="coerce")
    stake_err = (work["stake_invested"] - stake_target).abs()
    # Identity r = g - d
    ret_err = (work["g"] - work["d"] - work["r"]).abs()
    max_net = float(np.nanmax(net_err.to_numpy(dtype=float))) if len(net_err) else 0.0
    max_stake = float(np.nanmax(stake_err.to_numpy(dtype=float))) if len(stake_err) else 0.0
    max_ret = float(np.nanmax(ret_err.to_numpy(dtype=float))) if len(ret_err) else 0.0
    passed = max_net <= tol and max_stake <= tol and max_ret <= tol
    return {
        "passed": passed,
        "n": int(len(work)),
        "max_abs_net_err": max_net,
        "max_abs_stake_err": max_stake,
        "max_abs_ret_err": max_ret,
    }


def analysis_set_for_measurement(
    panel_econ: pd.DataFrame,
    measurement: str,
) -> pd.DataFrame:
    """Development association set for one measurement (finite m and valid r)."""
    if panel_econ.empty:
        return panel_econ.copy()
    assert_no_evaluation_rows(panel_econ, context=f"analysis_set[{measurement}]")
    m = pd.to_numeric(panel_econ[measurement], errors="coerce")
    mask = (
        (panel_econ["in_N"] == True)  # noqa: E712
        & (panel_econ["analysis_eligible"] == True)  # noqa: E712
        & (panel_econ["assoc_valid"] == True)  # noqa: E712
        & m.map(_finite)
        & panel_econ["r"].map(_finite)
    )
    return panel_econ.loc[mask].copy().reset_index(drop=True)


def assign_frozen_quintiles(anal: pd.DataFrame, measurement: str) -> pd.Series:
    """Equal-count Q1..Q5 by sorting (m, trade_date, ticker); freeze memberships."""
    if anal.empty:
        return pd.Series(dtype=object)
    work = anal.copy()
    work["_m"] = pd.to_numeric(work[measurement], errors="coerce")
    work["_td"] = work["trade_date"].map(_as_date)
    work["_tk"] = work["ticker"].astype(str)
    work = work.sort_values(["_m", "_td", "_tk"], ascending=True, kind="mergesort")
    n = len(work)
    edges = np.linspace(0, n, N_QUINTILES + 1).astype(int)
    labels = np.empty(n, dtype=object)
    for q in range(N_QUINTILES):
        labels[edges[q] : edges[q + 1]] = f"Q{q + 1}"
    out = pd.Series(labels, index=work.index, name="quintile")
    # Reindex to original anal order
    return out.reindex(anal.index)


def compute_delta_from_groups(
    frame: pd.DataFrame,
    *,
    quintile_col: str = "quintile",
    return_col: str = "r",
) -> float:
    """Primary Δ = mean(r_Q1) - mean(r_Q5). Undefined → NaN."""
    if frame.empty or quintile_col not in frame.columns:
        return float("nan")
    q1 = frame.loc[frame[quintile_col] == "Q1", return_col]
    q5 = frame.loc[frame[quintile_col] == "Q5", return_col]
    if len(q1) < MIN_GROUP_FOR_DELTA or len(q5) < MIN_GROUP_FOR_DELTA:
        return float("nan")
    m1 = float(np.nanmean(q1.to_numpy(dtype=float)))
    m5 = float(np.nanmean(q5.to_numpy(dtype=float)))
    if not (np.isfinite(m1) and np.isfinite(m5)):
        return float("nan")
    return m1 - m5


def quintile_group_table(anal: pd.DataFrame, measurement: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in [f"Q{i}" for i in range(1, N_QUINTILES + 1)]:
        grp = anal.loc[anal["quintile"] == q]
        rows.append(
            {
                "measurement": measurement,
                "quintile": q,
                "n": int(len(grp)),
                "n_dates": int(grp["trade_date"].map(_as_date).nunique()) if len(grp) else 0,
                "mean_r": float(np.nanmean(grp["r"])) if len(grp) else float("nan"),
                "mean_g": float(np.nanmean(grp["g"])) if len(grp) else float("nan"),
                "mean_d": float(np.nanmean(grp["d"])) if len(grp) else float("nan"),
                "sum_dollar_net": float(np.nansum(grp["dollar_net"])) if len(grp) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def winner_attribution(anal: pd.DataFrame, measurement: str, tops: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
    if anal.empty:
        return {"measurement": measurement, "tops": {}}
    ranked = anal.sort_values("dollar_net", ascending=False, kind="mergesort")
    out: dict[str, Any] = {"measurement": measurement, "tops": {}}
    for k in tops:
        head = ranked.head(k)
        total = float(np.nansum(head["dollar_net"]))
        by_q = (
            head.groupby("quintile", sort=False)["dollar_net"]
            .agg(["count", "sum"])
            .reset_index()
            if len(head)
            else pd.DataFrame(columns=["quintile", "count", "sum"])
        )
        out["tops"][str(k)] = {
            "n": int(len(head)),
            "dollar_sum": total,
            "by_quintile": [
                {
                    "quintile": str(r["quintile"]),
                    "count": int(r["count"]),
                    "dollar_sum": float(r["sum"]),
                    "dollar_share": float(r["sum"] / total) if total else float("nan"),
                }
                for _, r in by_q.iterrows()
            ],
        }
    return out


def evaluate_p_cov(anal: pd.DataFrame) -> bool:
    if anal.empty:
        return False
    n_anal = len(anal)
    floor = max(MIN_Q_N_FLOOR, MIN_Q_N_FRAC * n_anal)
    for q in [f"Q{i}" for i in range(1, N_QUINTILES + 1)]:
        grp = anal.loc[anal["quintile"] == q]
        if len(grp) < floor:
            return False
    for q in ("Q1", "Q5"):
        grp = anal.loc[anal["quintile"] == q]
        n_dates = int(grp["trade_date"].map(_as_date).nunique())
        if n_dates < MIN_Q_DATES:
            return False
    return True


def evaluate_p_gross(anal: pd.DataFrame, measurement: str) -> bool:
    """Gross-edge criterion for M1/M2; M3 treated as N/A pass."""
    if measurement == "M3":
        return True
    if anal.empty:
        return False
    q1 = anal.loc[anal["quintile"] == "Q1"]
    if q1.empty:
        return False
    g_all = float(np.nanmean(anal["g"].to_numpy(dtype=float)))
    g_q1 = float(np.nanmean(q1["g"].to_numpy(dtype=float)))
    if not np.isfinite(g_all) or not np.isfinite(g_q1):
        return False
    if g_all > 0.0:
        mean_ok = g_q1 >= 0.50 * g_all
    else:
        mean_ok = g_q1 >= g_all

    pos_all = anal.loc[anal["g"] > 0.0]
    pos_q1 = q1.loc[q1["g"] > 0.0]
    gplus_all = float(np.nansum(pos_all["dollar_gross"])) if len(pos_all) else 0.0
    gplus_q1 = float(np.nansum(pos_q1["dollar_gross"])) if len(pos_q1) else 0.0
    if gplus_all <= 0.0:
        return False
    share_ok = (gplus_q1 / gplus_all) >= 0.15
    return bool(mean_ok and share_ok)


def half_period_deltas(anal: pd.DataFrame) -> dict[str, float]:
    def _sub(start: date, end: date) -> float:
        mask = anal["trade_date"].map(
            lambda v: start <= _as_date(v) <= end
        )
        return compute_delta_from_groups(anal.loc[mask])

    return {
        "dev_a": _sub(DEV_A_START, DEV_A_END),
        "dev_b": _sub(DEV_B_START, DEV_B_END),
    }


def within_date_spearman_summary(
    anal: pd.DataFrame,
    measurement: str,
) -> dict[str, Any]:
    """Within-date Spearman of (m, r); exclude constant m or r; empty → P-wd False."""
    excluded_constant = 0
    excluded_small = 0
    rhos: list[float] = []
    if anal.empty:
        return {
            "measurement": measurement,
            "n_dates_considered": 0,
            "n_valid": 0,
            "n_excluded_constant": 0,
            "n_excluded_small": 0,
            "median_rho": float("nan"),
            "frac_rho_lt_0": float("nan"),
            "p_wd": False,
            "rhos": [],
        }

    for _, grp in anal.groupby(anal["trade_date"].map(_as_date), sort=True):
        if len(grp) < 2:
            excluded_small += 1
            continue
        m = pd.to_numeric(grp[measurement], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(grp["r"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(m) & np.isfinite(r)
        if int(ok.sum()) < 2:
            excluded_small += 1
            continue
        m = m[ok]
        r = r[ok]
        if np.unique(m).size < 2 or np.unique(r).size < 2:
            excluded_constant += 1
            continue
        rho, _ = spearmanr(m, r)
        if not np.isfinite(rho):
            excluded_constant += 1
            continue
        rhos.append(float(rho))

    n_valid = len(rhos)
    if n_valid == 0:
        p_wd = False
        median_rho = float("nan")
        frac_lt = float("nan")
    else:
        median_rho = float(np.median(rhos))
        frac_lt = float(np.mean(np.asarray(rhos) < 0.0))
        p_wd = bool(median_rho <= 0.0 or frac_lt >= 0.50)

    return {
        "measurement": measurement,
        "n_dates_considered": int(anal["trade_date"].map(_as_date).nunique()),
        "n_valid": n_valid,
        "n_excluded_constant": excluded_constant,
        "n_excluded_small": excluded_small,
        "median_rho": median_rho,
        "frac_rho_lt_0": frac_lt,
        "p_wd": p_wd,
        "rhos": rhos,
    }


def pooled_spearman(anal: pd.DataFrame, measurement: str) -> dict[str, Any]:
    if anal.empty or len(anal) < 2:
        return {"measurement": measurement, "rho": float("nan"), "n": int(len(anal))}
    m = pd.to_numeric(anal[measurement], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(anal["r"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(m) & np.isfinite(r)
    if int(ok.sum()) < 2 or np.unique(m[ok]).size < 2 or np.unique(r[ok]).size < 2:
        return {"measurement": measurement, "rho": float("nan"), "n": int(ok.sum())}
    rho, pval = spearmanr(m[ok], r[ok])
    return {
        "measurement": measurement,
        "rho": float(rho) if np.isfinite(rho) else float("nan"),
        "pvalue": float(pval) if np.isfinite(pval) else float("nan"),
        "n": int(ok.sum()),
    }


def precompute_date_quintile_stats(
    anal: pd.DataFrame,
    *,
    return_col: str = "r",
) -> tuple[list[date], dict[date, dict[str, tuple[float, int]]]]:
    """Per (date, quintile) sum_r and count for efficient bootstrap aggregation."""
    dates = sorted({_as_date(d) for d in anal["trade_date"].tolist()})
    stats: dict[date, dict[str, tuple[float, int]]] = {}
    for d in dates:
        stats[d] = {f"Q{i}": (0.0, 0) for i in range(1, N_QUINTILES + 1)}
    for td, grp in anal.groupby(anal["trade_date"].map(_as_date), sort=False):
        d = _as_date(td)
        for q, sub in grp.groupby("quintile", sort=False):
            s = float(np.nansum(sub[return_col].to_numpy(dtype=float)))
            c = int(sub[return_col].map(_finite).sum())
            stats[d][str(q)] = (s, c)
    return dates, stats


def delta_from_date_multiplicity(
    date_occurrences: list[date],
    stats: dict[date, dict[str, tuple[float, int]]],
) -> float:
    """Accumulate sum_r/count with date multiplicity (no dedupe)."""
    sum_q1 = 0.0
    cnt_q1 = 0
    sum_q5 = 0.0
    cnt_q5 = 0
    for d in date_occurrences:
        s1, c1 = stats[d]["Q1"]
        s5, c5 = stats[d]["Q5"]
        sum_q1 += s1
        cnt_q1 += c1
        sum_q5 += s5
        cnt_q5 += c5
    if cnt_q1 < MIN_GROUP_FOR_DELTA or cnt_q5 < MIN_GROUP_FOR_DELTA:
        return float("nan")
    return (sum_q1 / cnt_q1) - (sum_q5 / cnt_q5)


def explicit_path_delta_by_row_duplication(
    anal: pd.DataFrame,
    date_occurrences: list[date],
    *,
    return_col: str = "r",
) -> float:
    """Reference Δ via explicit row duplication of each date occurrence."""
    pieces: list[pd.DataFrame] = []
    by_date = {d: g for d, g in anal.groupby(anal["trade_date"].map(_as_date), sort=False)}
    for d in date_occurrences:
        if d not in by_date:
            continue
        pieces.append(by_date[d])
    if not pieces:
        return float("nan")
    path = pd.concat(pieces, ignore_index=True)
    return compute_delta_from_groups(path, return_col=return_col)


def sample_block_date_path(
    dates: list[date],
    rng: np.random.Generator,
    *,
    block_len: int = BLOCK_LEN,
) -> list[date]:
    """Overlapping moving-block bootstrap path of length T (multiplicity preserved)."""
    t = len(dates)
    if t == 0:
        return []
    if t < block_len:
        # Degenerate: sample individual dates with replacement.
        idx = rng.integers(0, t, size=t)
        return [dates[int(i)] for i in idx]
    n_starts = t - block_len + 1
    collected: list[date] = []
    while len(collected) < t:
        start = int(rng.integers(0, n_starts))
        collected.extend(dates[start : start + block_len])
    return collected[:t]


def run_block_bootstrap_deltas(
    anal: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    block_len: int = BLOCK_LEN,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Block bootstrap of Δ with precomputed (date, quintile) sums."""
    dates, stats = precompute_date_quintile_stats(anal)
    t = len(dates)
    rng = np.random.default_rng(seed)
    valid_deltas: list[float] = []
    n_invalid = 0
    invalid_reasons = {"empty_or_small_group": 0, "nonfinite": 0}

    for rep in range(1, n_boot + 1):
        path = sample_block_date_path(dates, rng, block_len=block_len)
        delta = delta_from_date_multiplicity(path, stats)
        if not np.isfinite(delta):
            n_invalid += 1
            # Distinguish empty vs other nonfinite
            sum_q1 = sum(stats[d]["Q1"][1] for d in path)
            sum_q5 = sum(stats[d]["Q5"][1] for d in path)
            if sum_q1 < MIN_GROUP_FOR_DELTA or sum_q5 < MIN_GROUP_FOR_DELTA:
                invalid_reasons["empty_or_small_group"] += 1
            else:
                invalid_reasons["nonfinite"] += 1
        else:
            valid_deltas.append(float(delta))
        if progress_every and rep % progress_every == 0:
            print(
                f"[D1 validation] bootstrap {rep}/{n_boot} "
                f"(valid={len(valid_deltas)} invalid={n_invalid})",
                flush=True,
            )

    arr = np.asarray(valid_deltas, dtype=float)
    n_valid = int(arr.size)
    interval_defined = n_valid >= MIN_VALID_BOOT_DELTA

    def _pct(q: float) -> float:
        if n_valid == 0:
            return float("nan")
        return float(np.quantile(arr, q, method="linear"))

    bonf_lo_q, bonf_hi_q = bonferroni_quantiles()
    result = {
        "n_reps": int(n_boot),
        "n_dates_T": int(t),
        "block_len": int(block_len),
        "seed": int(seed),
        "n_valid_delta": n_valid,
        "n_invalid_delta": int(n_invalid),
        "invalid_reasons": invalid_reasons,
        "interval_defined": bool(interval_defined),
        "bonferroni_lo_quantile": bonf_lo_q,
        "bonferroni_hi_quantile": bonf_hi_q,
        "adj_lower": _pct(bonf_lo_q) if interval_defined else float("nan"),
        "adj_upper": _pct(bonf_hi_q) if interval_defined else float("nan"),
        "ordinary_lower": _pct(ORDINARY_LO) if n_valid else float("nan"),
        "ordinary_upper": _pct(ORDINARY_HI) if n_valid else float("nan"),
        "valid_deltas": valid_deltas,
    }
    return result


def evaluate_predicates(
    *,
    anal: pd.DataFrame,
    measurement: str,
    point_delta: float,
    boot: dict[str, Any],
    within: dict[str, Any],
    halves: dict[str, float],
) -> dict[str, bool]:
    p_cov = evaluate_p_cov(anal)
    p_delta_def = bool(np.isfinite(point_delta)) and not anal.loc[anal["quintile"] == "Q1"].empty and not anal.loc[anal["quintile"] == "Q5"].empty
    p_econ = bool(np.isfinite(point_delta) and point_delta >= ECON_BAR)
    adj_lower = boot.get("adj_lower", float("nan"))
    interval_defined = bool(boot.get("interval_defined", False))
    p_stat = bool(interval_defined and np.isfinite(adj_lower) and adj_lower > 0.0)
    p_sign = bool(np.isfinite(point_delta) and point_delta > 0.0)
    da = halves.get("dev_a", float("nan"))
    db = halves.get("dev_b", float("nan"))
    p_half = bool(np.isfinite(da) and np.isfinite(db) and da > 0.0 and db > 0.0)
    p_wd = bool(within.get("p_wd", False))
    p_gross = evaluate_p_gross(anal, measurement)
    adj_ok_lower = bool(interval_defined and np.isfinite(adj_lower) and adj_lower > 0.0)
    p_wrong = bool(
        (np.isfinite(point_delta) and point_delta <= 0.0)
        or (
            np.isfinite(point_delta)
            and point_delta < WRONG_SIGN_SOFT
            and not adj_ok_lower
        )
    )
    return {
        "P-cov": p_cov,
        "P-delta-def": p_delta_def,
        "P-econ": p_econ,
        "P-stat": p_stat,
        "P-sign": p_sign,
        "P-half": p_half,
        "P-wd": p_wd,
        "P-gross": p_gross,
        "P-wrong": p_wrong,
        "interval_defined": interval_defined,
    }


def classify_measurement(
    predicates: dict[str, bool],
    *,
    point_delta: float,
    adj_upper: float,
) -> tuple[str, int]:
    """Ordered decision table; returns (label, matching_row_number). Row 9 skipped."""
    p = predicates
    if (not p["P-cov"]) or (not p["P-delta-def"]) or (not p["interval_defined"]):
        return LABEL_INCONCLUSIVE, 1
    if p["P-wrong"]:
        return LABEL_UNSUPPORTED, 2
    if (
        p["P-econ"]
        and p["P-stat"]
        and p["P-sign"]
        and p["P-half"]
        and p["P-wd"]
        and p["P-gross"]
    ):
        return LABEL_SUPPORTED, 3
    if (
        p["P-econ"]
        and p["P-stat"]
        and p["P-sign"]
        and p["P-half"]
        and p["P-gross"]
        and (not p["P-wd"])
    ):
        return LABEL_INCONCLUSIVE, 4
    if (
        p["P-econ"]
        and p["P-stat"]
        and p["P-sign"]
        and p["P-wd"]
        and p["P-gross"]
        and (not p["P-half"])
    ):
        return LABEL_INCONCLUSIVE, 5
    if (
        p["P-econ"]
        and p["P-stat"]
        and p["P-sign"]
        and p["P-half"]
        and p["P-wd"]
        and (not p["P-gross"])
    ):
        return LABEL_INCONCLUSIVE, 6
    if (
        p["P-sign"]
        and p["P-stat"]
        and (not p["P-econ"])
        and np.isfinite(point_delta)
        and point_delta >= WRONG_SIGN_SOFT
    ):
        return LABEL_INCONCLUSIVE, 7
    if p["P-econ"] and p["P-sign"] and (not p["P-stat"]):
        return LABEL_INCONCLUSIVE, 8
    # Row 10
    if (np.isfinite(point_delta) and point_delta <= 0.0) or (
        np.isfinite(adj_upper) and adj_upper < 0.0
    ):
        return LABEL_UNSUPPORTED, 10
    return LABEL_INCONCLUSIVE, 10


def sprint_level_gate(
    labels: dict[str, str],
    *,
    deltas: dict[str, float],
    within: dict[str, dict[str, Any]],
    anal_by_m: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    supported = [m for m, lab in labels.items() if lab == LABEL_SUPPORTED]
    if not supported:
        return {
            "decision": GATE_STOP,
            "authorize_d2": False,
            "supported_measurements": [],
            "ranked_candidates": [],
            "detail": "Zero supported measurements — D2 is a short stop record only",
        }

    def _rank_key(m: str) -> tuple[float, float, float]:
        delta = float(deltas.get(m, float("nan")))
        frac = float(within.get(m, {}).get("frac_rho_lt_0", float("nan")))
        if not np.isfinite(frac):
            frac = -1.0
        anal = anal_by_m.get(m, pd.DataFrame())
        gshare = float("nan")
        if m != "M3" and not anal.empty:
            q1 = anal.loc[anal["quintile"] == "Q1"]
            pos_all = anal.loc[anal["g"] > 0.0]
            pos_q1 = q1.loc[q1["g"] > 0.0]
            gplus_all = float(np.nansum(pos_all["dollar_gross"])) if len(pos_all) else 0.0
            gplus_q1 = float(np.nansum(pos_q1["dollar_gross"])) if len(pos_q1) else 0.0
            if gplus_all > 0:
                gshare = gplus_q1 / gplus_all
        if not np.isfinite(gshare):
            gshare = -1.0
        # Higher delta, higher within-date pass margin, higher G+ share
        return (
            delta if np.isfinite(delta) else -np.inf,
            frac,
            gshare,
        )

    ranked = sorted(supported, key=_rank_key, reverse=True)
    return {
        "decision": GATE_AUTHORIZE,
        "authorize_d2": True,
        "supported_measurements": supported,
        "ranked_candidates": ranked,
        "detail": f"Authorize D2 for {ranked}",
    }


def _sensitivity_for_measurement(
    panel_dev: pd.DataFrame,
    measurement: str,
    frozen_keys: pd.DataFrame,
) -> dict[str, Any]:
    """Recompute scenario-specific economics; reuse frozen quintile memberships."""
    out: dict[str, Any] = {}
    for h in SENSITIVITY_H:
        econ = attach_scenario_economics(panel_dev, h)
        anal = analysis_set_for_measurement(econ, measurement)
        # Attach frozen quintiles by trade key
        merged = anal.merge(
            frozen_keys,
            on=["trade_date", "ticker"],
            how="inner",
            validate="one_to_one",
        )
        assert_no_evaluation_rows(merged, context=f"sensitivity[{measurement},h={h}]")
        out[str(h)] = {
            "n": int(len(merged)),
            "delta": compute_delta_from_groups(merged),
            "mean_r_Q1": float(
                np.nanmean(merged.loc[merged["quintile"] == "Q1", "r"])
            )
            if (merged["quintile"] == "Q1").any()
            else float("nan"),
            "mean_r_Q5": float(
                np.nanmean(merged.loc[merged["quintile"] == "Q5", "r"])
            )
            if (merged["quintile"] == "Q5").any()
            else float("nan"),
        }
    return out


def run_d1_validation(
    *,
    run_dir: Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    block_len: int = BLOCK_LEN,
    panel: pd.DataFrame | None = None,
    progress_every: int = 1000,
) -> D1ValidationResult:
    """Execute D1 measurement validation on development data only."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    print("[D1 validation] stage=build_panel starting", flush=True)
    t_stage = time.perf_counter()
    if panel is None:
        base = build_d1_base_panel(run_dir=run_dir)
    else:
        base = panel.copy()
        if "in_N" in base.columns:
            base = base.loc[base["in_N"] == True].copy()  # noqa: E712
    panel_dev = filter_development_panel(base)
    assert_no_evaluation_rows(panel_dev, context="panel_dev")
    timings["build_panel"] = _progress(
        "build_panel",
        t_stage,
        note=f"in_N_dev={len(panel_dev)} crossed={int(panel_dev.get('crossed_quote_excluded', pd.Series(dtype=bool)).fillna(False).sum()) if len(panel_dev) else 0}",
    )

    print("[D1 validation] stage=primary_economics starting", flush=True)
    t_stage = time.perf_counter()
    econ_h1 = attach_scenario_economics(panel_dev, PRIMARY_H)
    recon = reconcile_economics(econ_h1)
    if not recon["passed"]:
        raise D1ValidationError(f"Economics reconciliation failed under h=1: {recon}")
    # Disclose N vs association exclusions
    n_in_n = int(len(econ_h1))
    n_crossed = int(econ_h1["crossed_quote_excluded"].fillna(False).sum()) if "crossed_quote_excluded" in econ_h1.columns else 0
    timings["primary_economics"] = _progress(
        "primary_economics", t_stage, note=f"recon_ok n={recon['n']}"
    )

    labels: dict[str, str] = {}
    predicates: dict[str, dict[str, bool]] = {}
    delta_tables: dict[str, Any] = {}
    within_date: dict[str, Any] = {}
    stability_halves: dict[str, Any] = {}
    spearman_diagnostic: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    winner_attr: dict[str, Any] = {}
    analysis_panels: dict[str, pd.DataFrame] = {}
    quintile_frames: list[pd.DataFrame] = []
    point_deltas: dict[str, float] = {}
    classify_rows: dict[str, int] = {}

    for measurement in MEASUREMENTS:
        print(f"[D1 validation] stage=measurement_{measurement} starting", flush=True)
        t_m = time.perf_counter()
        anal = analysis_set_for_measurement(econ_h1, measurement)
        assert_no_evaluation_rows(anal, context=f"anal[{measurement}]")
        anal = anal.copy()
        anal["quintile"] = assign_frozen_quintiles(anal, measurement)
        analysis_panels[measurement] = anal
        point_delta = compute_delta_from_groups(anal)
        point_deltas[measurement] = point_delta
        qtable = quintile_group_table(anal, measurement)
        quintile_frames.append(qtable)
        winner_attr[measurement] = winner_attribution(anal, measurement)
        halves = half_period_deltas(anal)
        stability_halves[measurement] = halves
        within = within_date_spearman_summary(anal, measurement)
        within_date[measurement] = {k: v for k, v in within.items() if k != "rhos"}
        within_date[measurement]["n_rhos"] = len(within.get("rhos", []))
        spearman_diagnostic[measurement] = pooled_spearman(anal, measurement)

        print(
            f"[D1 validation] bootstrap_{measurement} starting "
            f"(n_boot={n_boot} seed={seed})",
            flush=True,
        )
        t_boot = time.perf_counter()
        boot = run_block_bootstrap_deltas(
            anal,
            n_boot=n_boot,
            seed=seed,
            block_len=block_len,
            progress_every=progress_every,
        )
        boot_public = {k: v for k, v in boot.items() if k != "valid_deltas"}
        boot_public["point_delta"] = point_delta
        delta_tables[measurement] = boot_public
        timings[f"bootstrap_{measurement}"] = _progress(
            f"bootstrap_{measurement}",
            t_boot,
            note=f"valid={boot['n_valid_delta']} adj_lo={boot.get('adj_lower')}",
        )

        preds = evaluate_predicates(
            anal=anal,
            measurement=measurement,
            point_delta=point_delta,
            boot=boot,
            within=within,
            halves=halves,
        )
        label, row_no = classify_measurement(
            preds,
            point_delta=point_delta,
            adj_upper=float(boot.get("adj_upper", float("nan"))),
        )
        predicates[measurement] = preds
        labels[measurement] = label
        classify_rows[measurement] = row_no

        frozen_keys = anal[["trade_date", "ticker", "quintile"]].copy()
        frozen_keys["trade_date"] = frozen_keys["trade_date"].map(_as_date)
        sensitivity[measurement] = _sensitivity_for_measurement(
            panel_dev, measurement, frozen_keys
        )
        timings[f"measurement_{measurement}"] = _progress(
            f"measurement_{measurement}",
            t_m,
            note=f"label={label} row={row_no} delta={point_delta:.6g}",
        )

    gate = sprint_level_gate(
        labels,
        deltas=point_deltas,
        within=within_date,
        anal_by_m=analysis_panels,
    )
    quintile_tables = (
        pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame()
    )

    coverage = {
        "n_in_N_dev": n_in_n,
        "n_crossed_quote_excluded": n_crossed,
        "n_analysis_by_measurement": {m: int(len(analysis_panels[m])) for m in MEASUREMENTS},
        "dev_start": DEV_START.isoformat(),
        "dev_end": DEV_END.isoformat(),
        "eval_start_excluded": EVAL_START.isoformat(),
        "n_in_N_original_preserved": n_in_n,  # crossed stay in N
    }
    timings["total"] = _progress("total", t0, note=gate["decision"])

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_run_dir": str(run_dir),
        "sprint006_execution_repo_sha": OFFICIAL_EXECUTION_REPO_SHA,
        "d1_code_commit_sha": get_current_repo_sha(),
        "labels": labels,
        "classify_rows": classify_rows,
        "gate": gate,
        "protocol_pins": {
            "BUDGET_B": BUDGET_B,
            "FEES": FEES,
            "MAX_NAMES": MAX_NAMES,
            "SCENARIOS_H": list(SCENARIOS_H),
            "PRIMARY_H": PRIMARY_H,
            "SENSITIVITY_H": list(SENSITIVITY_H),
            "MEASUREMENTS": list(MEASUREMENTS),
            "DEV_START": DEV_START.isoformat(),
            "DEV_END": DEV_END.isoformat(),
            "DEV_A": [DEV_A_START.isoformat(), DEV_A_END.isoformat()],
            "DEV_B": [DEV_B_START.isoformat(), DEV_B_END.isoformat()],
            "EVAL_START": EVAL_START.isoformat(),
            "BLOCK_LEN": BLOCK_LEN,
            "N_BOOT": int(n_boot),
            "SEED": int(seed),
            "FAMILY_SIZE": FAMILY_SIZE,
            "ALPHA": ALPHA,
            "BONFERRONI_LO": BONFERRONI_LO,
            "BONFERRONI_HI": BONFERRONI_HI,
            "CROSSED_QUOTE_POLICY_VERSION": CROSSED_QUOTE_POLICY_VERSION,
            "ECON_BAR": ECON_BAR,
        },
        "coverage": coverage,
        "stage_timings": timings,
        "economics_reconciliation": recon,
    }
    return D1ValidationResult(
        labels=labels,
        predicates=predicates,
        gate=gate,
        delta_tables=delta_tables,
        quintile_tables=quintile_tables,
        within_date=within_date,
        stability_halves=stability_halves,
        spearman_diagnostic=spearman_diagnostic,
        sensitivity=sensitivity,
        winner_attribution=winner_attr,
        panel_dev=panel_dev,
        analysis_panels=analysis_panels,
        stage_timings=timings,
        manifest=manifest,
        coverage=coverage,
    )


def export_d1_evidence(
    *,
    result: D1ValidationResult,
    evidence_dir: Path,
    command: str | None = None,
) -> Path:
    """Write D1 evidence artifacts listed in the design."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Firewall: never write evaluation-period association rows.
    assert_no_evaluation_rows(result.panel_dev, context="export.panel_dev")
    for m, anal in result.analysis_panels.items():
        assert_no_evaluation_rows(anal, context=f"export.anal[{m}]")

    manifest = dict(result.manifest)
    if command:
        manifest["command"] = command
    manifest["evidence_dir"] = str(evidence_dir)
    (evidence_dir / "d1_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    labels_payload = {
        "labels": result.labels,
        "predicates": result.predicates,
        "classify_rows": result.manifest.get("classify_rows", {}),
        "spearman_does_not_override": True,
    }
    (evidence_dir / "d1_measurement_labels.json").write_text(
        json.dumps(labels_payload, indent=2, default=str), encoding="utf-8"
    )
    (evidence_dir / "d1_gate.json").write_text(
        json.dumps(result.gate, indent=2, default=str), encoding="utf-8"
    )
    (evidence_dir / "d1_delta_bootstrap.json").write_text(
        json.dumps(result.delta_tables, indent=2, default=str), encoding="utf-8"
    )
    (evidence_dir / "d1_spearman_diagnostic.json").write_text(
        json.dumps(result.spearman_diagnostic, indent=2, default=str), encoding="utf-8"
    )
    if result.quintile_tables is not None and not result.quintile_tables.empty:
        result.quintile_tables.to_parquet(evidence_dir / "d1_quintile_tables.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(evidence_dir / "d1_quintile_tables.parquet", index=False)
    (evidence_dir / "d1_within_date_summary.json").write_text(
        json.dumps(result.within_date, indent=2, default=str), encoding="utf-8"
    )
    (evidence_dir / "d1_stability_halves.json").write_text(
        json.dumps(result.stability_halves, indent=2, default=str), encoding="utf-8"
    )
    # Extra disclosures kept alongside design list
    (evidence_dir / "d1_sensitivity.json").write_text(
        json.dumps(result.sensitivity, indent=2, default=str), encoding="utf-8"
    )
    (evidence_dir / "d1_winner_attribution.json").write_text(
        json.dumps(result.winner_attribution, indent=2, default=str), encoding="utf-8"
    )

    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": get_current_repo_sha(),
        "evidence_dir": str(evidence_dir),
        "gate": result.gate,
        "labels": result.labels,
        "command": command,
        "official_run_dir": result.manifest.get("official_run_dir"),
        "sprint006_execution_repo_sha": OFFICIAL_EXECUTION_REPO_SHA,
    }
    (evidence_dir / "execution_receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str), encoding="utf-8"
    )
    return evidence_dir
