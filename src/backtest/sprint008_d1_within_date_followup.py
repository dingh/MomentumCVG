"""
Sprint 008 D1 follow-up — within-date lowest vs highest 20% (M1, M2).

Exploratory paired date-level contrast after original pooled D1.
Does not amend D1 labels or STOP_NO_THRESHOLDS. No thresholds / D2 / eval period.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_EXECUTION_REPO_SHA,
    OFFICIAL_RUN_DIR,
    get_current_repo_sha,
)
from src.backtest.sprint008_d0_input_readiness import (
    BUDGET_B,
    CROSSED_QUOTE_POLICY_VERSION,
    FEES,
)
from src.backtest.sprint008_d1_measurement_validation import (
    DEV_A_END,
    DEV_A_START,
    DEV_B_END,
    DEV_B_START,
    DEV_END,
    DEV_START,
    PRIMARY_H,
    D1ValidationError,
    assert_no_evaluation_rows,
    attach_scenario_economics,
    build_d1_base_panel,
    filter_development_panel,
    is_development_date,
    is_evaluation_date,
    reconcile_economics,
)

FOLLOWUP_MEASUREMENTS = ("M1", "M2")
MIN_SCORED = 5
GROUP_FRACTION_DENOM = 5
HAC_MAXLAGS = 3
HAC_KERNEL = "bartlett"
HAC_USE_CORRECTION = True
FAMILY_SIZE = 2
ALPHA = 0.05
ECON_BAR = 0.05  # reference only; not a gate
ORDINARY_CI_LEVEL = 0.95
ADJUSTED_CI_LEVEL = 0.975  # 1 - ALPHA/FAMILY_SIZE

# Preserved original D1 point findings (development, h=1) for report comparison.
ORIGINAL_D1_DELTA = {
    "M1": 0.0632,
    "M2": 0.0886,
    "M3": -0.0352,
}
ORIGINAL_D1_LABELS = {
    "M1": "inconclusive",
    "M2": "inconclusive",
    "M3": "unsupported",
}
ORIGINAL_D1_GATE = "STOP_NO_THRESHOLDS"
ORIGINAL_D1_EVIDENCE = "C:/MomentumCVG_env/runs/sprint008_d1_20260907T223037Z/"
ORIGINAL_D1_SHA = "72629a0d29f56771d1ff4a4ee3fe9cb227d593e4"


class FollowupValidationError(D1ValidationError):
    """Hard failure for follow-up construction / missing selected outcomes."""


@dataclass
class WithinDateFollowupResult:
    paired_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    measurement_summaries: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    panel_dev: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_original: dict[str, Any] = field(default_factory=dict)


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
    print(f"[D1 within-date follow-up] {stage}: {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def bartlett_weights(maxlags: int) -> np.ndarray:
    """Bartlett kernel weights for lags 1..maxlags: 1 - j/(maxlags+1)."""
    j = np.arange(1, int(maxlags) + 1, dtype=float)
    return 1.0 - j / (float(maxlags) + 1.0)


def newey_west_intercept_inference(
    d: np.ndarray,
    *,
    maxlags: int = HAC_MAXLAGS,
    use_correction: bool = HAC_USE_CORRECTION,
    ordinary_level: float = ORDINARY_CI_LEVEL,
    adjusted_level: float = ADJUSTED_CI_LEVEL,
) -> dict[str, Any]:
    """Intercept-only Newey–West/HAC inference for mean(d).

    Residuals are demeaned. Gamma_j uses divisor T. Bartlett weights.
    Small-sample correction multiplies the HAC covariance by T/(T-1) (k=1).
    Inference uses Student-t with df = T-1.
    """
    arr = np.asarray(d, dtype=float)
    if arr.ndim != 1:
        raise ValueError("d must be 1-d")
    t_obs = int(arr.size)
    if t_obs < 2:
        raise FollowupValidationError(f"Need >=2 eligible dates for HAC; got {t_obs}")
    if maxlags < 0:
        raise ValueError("maxlags must be >= 0")
    if maxlags >= t_obs:
        raise FollowupValidationError(
            f"HAC maxlags={maxlags} requires T > maxlags; got T={t_obs}"
        )

    mean = float(np.mean(arr))
    e = arr - mean
    gamma0 = float(np.dot(e, e) / t_obs)
    s = gamma0
    weights = bartlett_weights(maxlags)
    gammas = [gamma0]
    for j, w in enumerate(weights, start=1):
        gamma_j = float(np.dot(e[j:], e[:-j]) / t_obs)
        gammas.append(gamma_j)
        s += 2.0 * float(w) * gamma_j

    # Var(mean) = S / T; optional small-sample factor T/(T-k), k=1.
    var_mean = s / t_obs
    if use_correction:
        var_mean *= t_obs / (t_obs - 1)
    se = float(np.sqrt(max(var_mean, 0.0)))
    df = t_obs - 1
    if se == 0.0:
        t_stat = float("inf") if mean != 0.0 else 0.0
        p_raw = 0.0 if mean != 0.0 else 1.0
    else:
        t_stat = mean / se
        p_raw = float(2.0 * stats.t.sf(abs(t_stat), df=df))

    def _ci(level: float) -> tuple[float, float]:
        alpha = 1.0 - level
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        half = crit * se
        return mean - half, mean + half

    lo95, hi95 = _ci(ordinary_level)
    lo_adj, hi_adj = _ci(adjusted_level)
    return {
        "mean": mean,
        "se_hac": se,
        "t_stat": float(t_stat),
        "df": int(df),
        "p_raw": float(p_raw),
        "ci_ordinary_level": float(ordinary_level),
        "ci_ordinary": [float(lo95), float(hi95)],
        "ci_adjusted_level": float(adjusted_level),
        "ci_adjusted": [float(lo_adj), float(hi_adj)],
        "maxlags": int(maxlags),
        "kernel": HAC_KERNEL,
        "use_correction": bool(use_correction),
        "bartlett_weights": [float(w) for w in weights],
        "gamma": [float(g) for g in gammas],
        "S_hac": float(s),
    }


def bonferroni_adjust_p(p_raw: float, *, family_size: int = FAMILY_SIZE) -> float:
    return float(min(1.0, family_size * float(p_raw)))


def paired_ttest_diagnostic(l_vals: np.ndarray, u_vals: np.ndarray) -> dict[str, Any]:
    """Ordinary paired t-test of L_t vs U_t (diagnostic only)."""
    l_arr = np.asarray(l_vals, dtype=float)
    u_arr = np.asarray(u_vals, dtype=float)
    if l_arr.size != u_arr.size or l_arr.size < 2:
        raise FollowupValidationError("Paired t-test requires >=2 matched dates")
    res = stats.ttest_rel(l_arr, u_arr, nan_policy="raise")
    d = l_arr - u_arr
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n": int(l_arr.size),
        "mean_d": float(np.mean(d)),
    }


def select_within_date_groups(
    day: pd.DataFrame,
    measurement: str,
) -> dict[str, Any]:
    """Form L/U groups from scored candidates without using future returns.

    ``day`` must already be restricted to one trade_date and scored-eligible rows
    (in_N, analysis_eligible, finite score). Returns exclusion reason or groups.
    """
    work = day.copy()
    work["_m"] = pd.to_numeric(work[measurement], errors="coerce")
    work = work.loc[work["_m"].map(_finite)].copy()
    n = int(len(work))
    if n < MIN_SCORED:
        return {
            "eligible": False,
            "reason": "fewer_than_five_scored",
            "n_scored": n,
            "k": 0,
        }
    scores = work["_m"].to_numpy(dtype=float)
    if float(np.nanmax(scores) - np.nanmin(scores)) <= 0.0:
        return {
            "eligible": False,
            "reason": "no_score_variation",
            "n_scored": n,
            "k": 0,
        }

    work["_tk"] = work["ticker"].astype(str)
    # Keep original index so callers can label the parent panel (no reset_index).
    work = work.sort_values(["_m", "_tk"], ascending=True, kind="mergesort")
    k = int(n // GROUP_FRACTION_DENOM)
    if k < 1:
        return {
            "eligible": False,
            "reason": "k_zero",
            "n_scored": n,
            "k": 0,
        }

    low = work.iloc[:k].copy()
    high = work.iloc[-k:].copy()
    # Cutoff-tie disclosure: boundary score shared with adjacent non-selected row.
    low_cut_tie = bool(k < n and float(work.iloc[k - 1]["_m"]) == float(work.iloc[k]["_m"]))
    high_cut_tie = bool(
        k < n and float(work.iloc[n - k - 1]["_m"]) == float(work.iloc[n - k]["_m"])
    )
    return {
        "eligible": True,
        "reason": None,
        "n_scored": n,
        "k": k,
        "low": low,
        "high": high,
        "low_cutoff_tie": low_cut_tie,
        "high_cutoff_tie": high_cut_tie,
        "low_score_max": float(low["_m"].max()),
        "high_score_min": float(high["_m"].min()),
    }


def _require_selected_outcomes(selected: pd.DataFrame, *, context: str) -> None:
    """Missing required outcome on a selected trade is a hard input failure."""
    if selected.empty:
        raise FollowupValidationError(f"{context}: empty selected group")
    if "assoc_valid" not in selected.columns:
        raise FollowupValidationError(f"{context}: missing assoc_valid")
    bad = ~(selected["assoc_valid"].astype(bool) & selected["r"].map(_finite))
    if bool(bad.any()):
        tickers = selected.loc[bad, "ticker"].astype(str).tolist()
        raise FollowupValidationError(
            f"{context}: selected trade(s) missing required outcome: {tickers}"
        )


def scored_candidates_for_date(
    day_all: pd.DataFrame,
    measurement: str,
) -> pd.DataFrame:
    """Entry-eligible scored candidates; membership ignores future returns."""
    m = pd.to_numeric(day_all[measurement], errors="coerce")
    mask = (
        (day_all["in_N"] == True)  # noqa: E712
        & (day_all["analysis_eligible"] == True)  # noqa: E712
        & m.map(_finite)
    )
    return day_all.loc[mask].copy()


def build_paired_observations(
    panel_econ: pd.DataFrame,
    measurement: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build date-level L_t, U_t, d_t and exclusion log for one measurement."""
    assert_no_evaluation_rows(panel_econ, context=f"followup[{measurement}]")
    if panel_econ.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = panel_econ.copy()
    work["trade_date"] = work["trade_date"].map(_as_date)
    paired_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for trade_date, day_all in work.groupby("trade_date", sort=True):
        td = _as_date(trade_date)
        if not is_development_date(td):
            continue
        scored = scored_candidates_for_date(day_all, measurement)
        selection = select_within_date_groups(scored, measurement)
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
        _require_selected_outcomes(low, context=f"{measurement} {td} low")
        _require_selected_outcomes(high, context=f"{measurement} {td} high")

        l_t = float(np.mean(low["r"].to_numpy(dtype=float)))
        u_t = float(np.mean(high["r"].to_numpy(dtype=float)))
        d_t = l_t - u_t
        paired_rows.append(
            {
                "measurement": measurement,
                "trade_date": td,
                "n_scored": int(selection["n_scored"]),
                "k": int(selection["k"]),
                "L_t": l_t,
                "U_t": u_t,
                "d_t": d_t,
                "low_cutoff_tie": bool(selection["low_cutoff_tie"]),
                "high_cutoff_tie": bool(selection["high_cutoff_tie"]),
                "low_score_max": float(selection["low_score_max"]),
                "high_score_min": float(selection["high_score_min"]),
                "low_tickers": ",".join(low["ticker"].astype(str).tolist()),
                "high_tickers": ",".join(high["ticker"].astype(str).tolist()),
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
        dates = paired["trade_date"].map(_as_date).tolist()
        gaps: list[float | None] = [None]
        for i in range(1, len(dates)):
            gaps.append(float((dates[i] - dates[i - 1]).days))
        paired["calendar_gap_days"] = gaps
    else:
        paired = pd.DataFrame(
            columns=[
                "measurement",
                "trade_date",
                "n_scored",
                "k",
                "L_t",
                "U_t",
                "d_t",
                "low_cutoff_tie",
                "high_cutoff_tie",
                "low_score_max",
                "high_score_min",
                "low_tickers",
                "high_tickers",
                "calendar_gap_days",
            ]
        )
    return paired, excluded


def summarize_measurement(
    paired: pd.DataFrame,
    excluded: pd.DataFrame,
    measurement: str,
) -> dict[str, Any]:
    """Point estimates, diagnostics, HAC primary inference for one measurement."""
    if paired.empty or "measurement" not in paired.columns:
        sub = pd.DataFrame()
    else:
        sub = paired.loc[paired["measurement"] == measurement].copy()
    if excluded.empty or "measurement" not in excluded.columns:
        ex = pd.DataFrame(columns=["measurement", "trade_date", "reason", "n_scored", "k"])
    else:
        ex = excluded.loc[excluded["measurement"] == measurement].copy()
    reason_counts = (
        ex["reason"].value_counts(dropna=False).astype(int).to_dict() if not ex.empty else {}
    )

    if sub.empty:
        return {
            "measurement": measurement,
            "n_eligible_dates": 0,
            "n_excluded_dates": int(len(ex)),
            "exclusion_reasons": reason_counts,
            "error": "no_eligible_dates",
        }

    l_vals = sub["L_t"].to_numpy(dtype=float)
    u_vals = sub["U_t"].to_numpy(dtype=float)
    d_vals = sub["d_t"].to_numpy(dtype=float)
    paired_t = paired_ttest_diagnostic(l_vals, u_vals)
    hac = newey_west_intercept_inference(d_vals)
    p_adj = bonferroni_adjust_p(hac["p_raw"], family_size=FAMILY_SIZE)

    def _half_mean(start: date, end: date) -> dict[str, Any]:
        mask = sub["trade_date"].map(lambda d: start <= _as_date(d) <= end)
        part = sub.loc[mask]
        if part.empty:
            return {"n_dates": 0, "mean_d": None, "mean_L": None, "mean_U": None}
        return {
            "n_dates": int(len(part)),
            "mean_d": float(part["d_t"].mean()),
            "mean_L": float(part["L_t"].mean()),
            "mean_U": float(part["U_t"].mean()),
        }

    gap_series = pd.to_numeric(sub["calendar_gap_days"], errors="coerce")
    gaps = [float(g) for g in gap_series.tolist() if pd.notna(g)]
    mean_d = float(np.mean(d_vals))
    return {
        "measurement": measurement,
        "n_eligible_dates": int(len(sub)),
        "n_excluded_dates": int(len(ex)),
        "exclusion_reasons": {str(k): int(v) for k, v in reason_counts.items()},
        "mean_L": float(np.mean(l_vals)),
        "mean_U": float(np.mean(u_vals)),
        "mean_d": mean_d,
        "median_k": float(np.median(sub["k"].to_numpy(dtype=float))),
        "mean_k": float(np.mean(sub["k"].to_numpy(dtype=float))),
        "n_low_cutoff_ties": int(sub["low_cutoff_tie"].sum()),
        "n_high_cutoff_ties": int(sub["high_cutoff_tie"].sum()),
        "calendar_gap_days": {
            "n": int(len(gaps)),
            "min": int(min(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "mean": float(np.mean(gaps)) if gaps else None,
            "max": int(max(gaps)) if gaps else None,
            "interpretation": (
                "HAC lags index successive eligible entry-date observations, "
                "not calendar days; calendar_gap_days discloses spacing between "
                "those eligible dates."
            ),
        },
        "paired_ttest": paired_t,
        "hac": hac,
        "p_adjusted_bonferroni": p_adj,
        "family_size": FAMILY_SIZE,
        "statistically_significant_adj": bool(p_adj < ALPHA),
        "economic_magnitude_vs_5pp": {
            "benchmark": ECON_BAR,
            "mean_d": mean_d,
            "meets_or_exceeds_benchmark": bool(mean_d >= ECON_BAR),
            "note": "Economic reference only; not a classification gate.",
        },
        "sign_note": (
            "Positive mean_d favors lower scores within date; "
            "significant negative mean_d is unfavorable."
        ),
        "half_2020_2021": _half_mean(DEV_A_START, DEV_A_END),
        "half_2022_2023": _half_mean(DEV_B_START, DEV_B_END),
        "original_d1_comparison": {
            "original_delta_q1_minus_q5": ORIGINAL_D1_DELTA.get(measurement),
            "original_label": ORIGINAL_D1_LABELS.get(measurement),
            "followup_mean_d": mean_d,
            "note": (
                "Original D1 is pooled trade-weighted Q1−Q5; follow-up is "
                "equal-weighted mean of within-date L−U. Membership and "
                "weighting both differ."
            ),
        },
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
    }


def run_within_date_followup(
    *,
    run_dir: Path | None = None,
) -> WithinDateFollowupResult:
    """Execute development-only within-date L vs U follow-up for M1 and M2."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    panel = build_d1_base_panel(run_dir=run_dir or OFFICIAL_RUN_DIR)
    timings["build_panel"] = _progress("build_panel", t0, note=f"rows={len(panel)}")

    t1 = time.perf_counter()
    panel_dev = filter_development_panel(panel)
    assert_no_evaluation_rows(panel_dev, context="panel_dev")
    if panel_dev["trade_date"].map(is_evaluation_date).any():
        raise FollowupValidationError("Evaluation rows leaked into development panel")

    # Preserve original N before measurement filters / grouping.
    n_by_date = (
        panel_dev.loc[panel_dev["in_N"] == True]  # noqa: E712
        .groupby(panel_dev.loc[panel_dev["in_N"] == True, "trade_date"].map(_as_date))  # noqa: E712
        .size()
        .astype(int)
    )
    n_original = {
        "n_in_N_rows": int((panel_dev["in_N"] == True).sum()),  # noqa: E712
        "n_dates": int(n_by_date.shape[0]),
        "n_by_date_hash": int(pd.util.hash_pandas_object(n_by_date, index=True).sum()),
        "budget_b": float(BUDGET_B),
        "fees": float(FEES),
        "h": float(PRIMARY_H),
        "crossed_quote_policy": CROSSED_QUOTE_POLICY_VERSION,
    }
    timings["filter_dev"] = _progress("filter_dev", t1, note=f"dev_rows={len(panel_dev)}")

    t2 = time.perf_counter()
    panel_econ = attach_scenario_economics(panel_dev, PRIMARY_H)
    recon = reconcile_economics(panel_econ)
    if not recon["passed"]:
        raise FollowupValidationError(f"Economics reconciliation failed: {recon}")
    # Stake identity: q*C == B/N using original N (via equal_dollar_quantities).
    valid = panel_econ["assoc_valid"] == True  # noqa: E712
    if valid.any():
        stake = pd.to_numeric(panel_econ.loc[valid, "stake_dollars"], errors="coerce")
        invested = pd.to_numeric(panel_econ.loc[valid, "stake_invested"], errors="coerce")
        if float((invested - stake).abs().max()) > 1e-8:
            raise FollowupValidationError("Stake identity q*C != B/N for association rows")
    timings["economics"] = _progress("economics", t2, note=f"recon={recon}")

    paired_all: list[pd.DataFrame] = []
    excluded_all: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}

    for measurement in FOLLOWUP_MEASUREMENTS:
        tm = time.perf_counter()
        paired, excluded = build_paired_observations(panel_econ, measurement)
        # Guard: grouping must not depend on returns — verify scores alone determine sets
        # via reconstruction check on ticker lists vs score sort (already by construction).
        assert_no_evaluation_rows(paired, context=f"paired[{measurement}]")
        paired_all.append(paired)
        excluded_all.append(excluded)
        summaries[measurement] = summarize_measurement(paired, excluded, measurement)
        timings[f"analyze_{measurement}"] = _progress(
            f"analyze_{measurement}",
            tm,
            note=(
                f"eligible={summaries[measurement].get('n_eligible_dates')} "
                f"mean_d={summaries[measurement].get('mean_d')}"
            ),
        )

    paired_df = (
        pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
    )
    excluded_df = (
        pd.concat(excluded_all, ignore_index=True) if excluded_all else pd.DataFrame()
    )

    support_notes = []
    for m, s in summaries.items():
        if s.get("n_eligible_dates", 0) == 0:
            support_notes.append(f"{m}: no eligible dates")
            continue
        sig = s.get("statistically_significant_adj", False)
        mean_d = s.get("mean_d")
        if sig and mean_d is not None and mean_d > 0:
            support_notes.append(
                f"{m}: adjusted-significant positive within-date mean_d "
                f"({mean_d:.4f}); exploratory only — does not authorize D2"
            )
        elif sig and mean_d is not None and mean_d < 0:
            support_notes.append(
                f"{m}: adjusted-significant NEGATIVE within-date mean_d "
                f"({mean_d:.4f}) — unfavorable"
            )
        else:
            support_notes.append(
                f"{m}: not adjusted-significant (mean_d={mean_d}); "
                f"does not overturn D1 STOP_NO_THRESHOLDS"
            )

    report = {
        "protocol": "docs/tmp/sprint008_d1_within_date_followup_protocol.md",
        "status": "exploratory_followup_awaiting_review",
        "preserves_original_d1_gate": ORIGINAL_D1_GATE,
        "original_d1_evidence": ORIGINAL_D1_EVIDENCE,
        "original_d1_sha": ORIGINAL_D1_SHA,
        "original_d1_labels": ORIGINAL_D1_LABELS,
        "post_hoc_disclosure": (
            "This within-date analysis was proposed after viewing the original "
            "pooled-quintile D1 results."
        ),
        "question": (
            "Among candidates on the same entry date, do lower M1/M2 scores "
            "identify higher subsequent net percentage returns?"
        ),
        "pins": {
            "measurements": list(FOLLOWUP_MEASUREMENTS),
            "dev_window": [str(DEV_START), str(DEV_END)],
            "h": PRIMARY_H,
            "fees": FEES,
            "budget_b": BUDGET_B,
            "min_scored": MIN_SCORED,
            "k_rule": "floor(n_scored/5)",
            "hac_maxlags": HAC_MAXLAGS,
            "hac_kernel": HAC_KERNEL,
            "hac_use_correction": HAC_USE_CORRECTION,
            "family_size": FAMILY_SIZE,
            "alpha": ALPHA,
            "ordinary_ci_level": ORDINARY_CI_LEVEL,
            "adjusted_ci_level": ADJUSTED_CI_LEVEL,
            "econ_bar_reference": ECON_BAR,
            "crossed_quote_policy": CROSSED_QUOTE_POLICY_VERSION,
            "official_artifacts": str(OFFICIAL_RUN_DIR),
            "official_execution_sha": OFFICIAL_EXECUTION_REPO_SHA,
        },
        "n_original": n_original,
        "measurements": summaries,
        "support_interpretation": support_notes,
        "d2_authorization": (
            "NOT authorized by this follow-up. Original D1 gate "
            f"{ORIGINAL_D1_GATE} preserved."
        ),
        "environment": _package_versions(),
        "code_sha": get_current_repo_sha(),
        "working_tree": _working_tree_status(),
        "stage_timings": timings,
        "total_seconds": float(time.perf_counter() - t0),
    }
    timings["total"] = _progress("total", t0)

    return WithinDateFollowupResult(
        paired_observations=paired_df,
        excluded_dates=excluded_df,
        measurement_summaries=summaries,
        report=report,
        stage_timings=timings,
        panel_dev=panel_dev,
        n_original=n_original,
    )


def export_followup_evidence(
    *,
    result: WithinDateFollowupResult,
    evidence_dir: Path,
    command: str,
) -> Path:
    """Write paired observations and readable report artifacts."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    paired_path = evidence_dir / "within_date_paired_observations.parquet"
    excluded_path = evidence_dir / "within_date_excluded_dates.parquet"
    result.paired_observations.to_parquet(paired_path, index=False)
    result.excluded_dates.to_parquet(excluded_path, index=False)

    # CSV copies for easy reading
    result.paired_observations.to_csv(
        evidence_dir / "within_date_paired_observations.csv", index=False
    )
    result.excluded_dates.to_csv(
        evidence_dir / "within_date_excluded_dates.csv", index=False
    )

    report = dict(result.report)
    report["command"] = command
    report["evidence_dir"] = str(evidence_dir)
    report["exported_utc"] = datetime.now(timezone.utc).isoformat()

    (evidence_dir / "within_date_followup_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (evidence_dir / "within_date_measurement_summaries.json").write_text(
        json.dumps(result.measurement_summaries, indent=2, default=str),
        encoding="utf-8",
    )

    # Human-readable markdown summary (results known only at export time).
    lines = [
        "# Sprint 008 D1 within-date follow-up — report",
        "",
        f"- Evidence dir: `{evidence_dir}`",
        f"- Executing SHA: `{report.get('code_sha')}`",
        f"- Working tree: `{report.get('working_tree')}`",
        f"- Command: `{command}`",
        f"- Runtime (s): `{report.get('total_seconds')}`",
        f"- Original D1 gate (preserved): `{ORIGINAL_D1_GATE}`",
        f"- Post-hoc disclosure: {report.get('post_hoc_disclosure')}",
        "",
        "## Pins",
        "",
        "```json",
        json.dumps(report.get("pins", {}), indent=2, default=str),
        "```",
        "",
        "## Per-measurement results",
        "",
    ]
    for m in FOLLOWUP_MEASUREMENTS:
        s = result.measurement_summaries.get(m, {})
        lines.extend(
            [
                f"### {m}",
                "",
                f"- Eligible dates: {s.get('n_eligible_dates')}",
                f"- Excluded dates: {s.get('n_excluded_dates')} ({s.get('exclusion_reasons')})",
                f"- Mean L_t: {s.get('mean_L')}",
                f"- Mean U_t: {s.get('mean_U')}",
                f"- Mean d_t: {s.get('mean_d')}",
                f"- Paired t: {s.get('paired_ttest')}",
                f"- HAC p_raw: {s.get('hac', {}).get('p_raw')}",
                f"- Bonferroni adjusted p: {s.get('p_adjusted_bonferroni')}",
                f"- Ordinary 95% HAC CI: {s.get('hac', {}).get('ci_ordinary')}",
                f"- Adjusted 97.5% HAC CI: {s.get('hac', {}).get('ci_adjusted')}",
                f"- Half 2020–2021 mean d: {s.get('half_2020_2021')}",
                f"- Half 2022–2023 mean d: {s.get('half_2022_2023')}",
                f"- vs original D1: {s.get('original_d1_comparison')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            *[f"- {note}" for note in report.get("support_interpretation", [])],
            "",
            f"- D2: {report.get('d2_authorization')}",
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(report.get("environment", {}), indent=2),
            "```",
            "",
        ]
    )
    (evidence_dir / "within_date_followup_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    receipt = {
        "code_sha": report.get("code_sha"),
        "working_tree": report.get("working_tree"),
        "command": command,
        "environment": report.get("environment"),
        "official_run_dir": str(OFFICIAL_RUN_DIR),
        "official_execution_sha": OFFICIAL_EXECUTION_REPO_SHA,
        "seed_n_a": "deterministic_no_bootstrap",
        "protocol": report.get("protocol"),
        "preserves_original_d1_gate": ORIGINAL_D1_GATE,
        "total_seconds": report.get("total_seconds"),
        "stage_timings": result.stage_timings,
    }
    (evidence_dir / "execution_receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str), encoding="utf-8"
    )
    return evidence_dir
