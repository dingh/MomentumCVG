"""Sprint 008 D2 — frozen M1/M2 exclude-U retrospective validation.

Accepted amendment (reviewed commit c2ba972). Does not change historical
D1 findings or STOP_NO_THRESHOLDS. Not a threshold search.
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
import pyarrow.parquet as pq

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_RUN_DIR,
    get_current_repo_sha,
)
from src.backtest.sprint008_d0_input_readiness import (
    FEES,
    MAX_NAMES,
    load_full_long_trade_log,
    reconstruct_capped_long_n,
)
from src.backtest.sprint008_d1_cost_diagnosis import (
    CostDiagnosisError,
    attach_decomposition_columns,
    build_portfolio_comparison,
    enforce_d0_required_inputs,
    require_all_executed_outcomes,
)
from src.backtest.sprint008_d1_measurement_validation import (
    EVAL_START,
    PRIMARY_H,
    attach_scenario_economics,
    build_d1_base_panel,
    is_development_date,
    reconcile_economics,
)
from src.backtest.sprint008_d1_within_date_followup import (
    FOLLOWUP_MEASUREMENTS,
    HAC_KERNEL,
    HAC_MAXLAGS,
    HAC_USE_CORRECTION,
    FollowupValidationError,
    bonferroni_adjust_p,
    newey_west_intercept_inference,
    scored_candidates_for_date,
    select_within_date_groups,
)
from src.backtest.surface_decision_report import PRIMARY_END, filter_to_window
from src.backtest.surface_runner import DATE_STATUS_COLUMNS

EVAL_END = PRIMARY_END
FAMILY_SIZE = 2
ALPHA = 0.05
ORDINARY_CI_LEVEL = 0.95
ADJUSTED_CI_LEVEL = 1.0 - ALPHA / FAMILY_SIZE  # 0.975
EXPECTED_DATE_STATUS_ROWS = 403
ORIGINAL_D1_GATE = "STOP_NO_THRESHOLDS"
DESIGN_PATH = "docs/tmp/sprint008_d2_design.md"
REVIEWED_DESIGN_SHA = "c2ba972"

D2_REPORTING_PERIODS: dict[str, tuple[date, date]] = {
    "2024": (date(2024, 1, 1), date(2024, 12, 31)),
    "2025": (date(2025, 1, 1), date(2025, 12, 31)),
    "2026_partial": (date(2026, 1, 1), EVAL_END),
}
D2_PERIOD_LABELS = {
    "2024": "full year",
    "2025": "full year",
    "2026_partial": "partial year",
}

# Copied from reviewed cost-diagnosis evidence. Not recomputed and not pooled.
PUBLISHED_DEVELOPMENT = {
    "window": "2020-01-01 through 2023-12-31",
    "source": "docs/tmp/sprint008_d1_cost_diagnosis_evidence.md",
    "evidence_dir": "C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_20260912T211530Z",
    "n_dates": 209,
    "baseline_pnl": 6628.20,
    "M1": {
        "filtered_pnl": 20823.08,
        "improvement": 14194.88,
        "mean_weekly_uplift": 0.0068,
        "losses_avoided": 121603.70,
        "winning_profits_sacrificed": 107408.82,
        "winning_profit_retention": 0.838,
        "top5_profit_retention": 0.800,
        "top10_profit_retention": 0.880,
        "baseline_drawdown": -63649.91,
        "filtered_drawdown": -43314.51,
        "hac_se": 0.0072,
        "ci95": [-0.0074, 0.0210],
        "p_raw": 0.3464,
        "p_adjusted_family4": 1.0,
        "label": "inconclusive",
    },
    "M2": {
        "filtered_pnl": 25088.18,
        "improvement": 18459.98,
        "mean_weekly_uplift": 0.0088,
        "losses_avoided": 124638.89,
        "winning_profits_sacrificed": 106178.91,
        "winning_profit_retention": 0.840,
        "top5_profit_retention": 0.510,
        "top10_profit_retention": 0.629,
        "baseline_drawdown": -63649.91,
        "filtered_drawdown": -41965.84,
        "hac_se": 0.0073,
        "ci95": [-0.0055, 0.0232],
        "p_raw": 0.2273,
        "p_adjusted_family4": 0.9092,
        "label": "inconclusive",
    },
}


class D2ValidationError(FollowupValidationError):
    """Hard failure for D2 calendar or evaluation-window construction."""


@dataclass
class D2ValidationResult:
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_level: pd.DataFrame = field(default_factory=pd.DataFrame)
    date_portfolio: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    summaries: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    calendar_validation: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)


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
    print(f"[D2 validation] {stage}: {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def is_d2_evaluation_date(value: Any) -> bool:
    d = _as_date(value)
    return EVAL_START <= d <= EVAL_END


def _reason_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "<na>"}:
        return None
    return text


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def filter_evaluation_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep 2024-01-01 through 2026-07-10 inclusive. Does not invent cash dates."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    out["trade_date"] = out["trade_date"].map(_as_date)
    return out.loc[out["trade_date"].map(is_d2_evaluation_date)].reset_index(drop=True)


def assert_no_development_rows(frame: pd.DataFrame, *, context: str) -> None:
    if frame.empty or "trade_date" not in frame.columns:
        return
    n_dev = int(frame["trade_date"].map(is_development_date).sum())
    if n_dev:
        raise D2ValidationError(f"{context}: found {n_dev} development-period rows")
    late = frame["trade_date"].map(_as_date)
    n_late = int((late > EVAL_END).sum())
    if n_late:
        raise D2ValidationError(f"{context}: found {n_late} dates after {EVAL_END}")


def interpret_adjusted_interval(lo: float, hi: float) -> str:
    if lo > 0.0:
        return "relative_benefit"
    if hi < 0.0:
        return "relative_harm"
    return "inconclusive"


def assign_eval_u_labels(
    panel_econ: pd.DataFrame,
    measurement: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label evaluation-window U groups. Membership ignores future returns."""
    work = panel_econ.copy()
    work["trade_date"] = work["trade_date"].map(_as_date)
    label_col = f"group_{measurement}"
    if label_col not in work.columns:
        work[label_col] = pd.NA
    paired_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for trade_date, day_all in work.groupby("trade_date", sort=True):
        td = _as_date(trade_date)
        if not is_d2_evaluation_date(td):
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
        high = selection["high"]
        bad = ~(high["assoc_valid"].astype(bool) & high["r"].map(_finite))
        if bool(bad.any()):
            raise D2ValidationError(
                f"{measurement} {td} U: missing outcomes {high.loc[bad, 'ticker'].tolist()}"
            )
        work.loc[high.index, label_col] = "U"
        n_scored = int(selection["n_scored"])
        k = int(selection["k"])
        paired_rows.append(
            {
                "measurement": measurement,
                "trade_date": td,
                "n_scored": n_scored,
                "k": k,
                "group_fraction": float(k / n_scored) if n_scored else 0.0,
                "high_cutoff_tie": bool(selection["high_cutoff_tie"]),
                "low_cutoff_tie": bool(selection["low_cutoff_tie"]),
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


def classify_evaluation_calendar(
    mid_status: pd.DataFrame,
    cross_status: pd.DataFrame,
    funnel: pd.DataFrame,
    long_trades: pd.DataFrame,
    *,
    window_start: date = EVAL_START,
    window_end: date = EVAL_END,
) -> pd.DataFrame:
    """Classify each evaluation calendar date. Never treats missing data as cash.

    ``date_status`` is the calendar. Funnel long counts plus reconstructed
    ``in_N`` distinguish verified N=0 from missing or failed data.
    """
    for name, frame in (("mid date_status", mid_status), ("cross date_status", cross_status)):
        missing = [c for c in DATE_STATUS_COLUMNS if c not in frame.columns]
        if missing:
            raise D2ValidationError(f"{name} missing columns: {missing}")
    funnel_need = [
        "trade_date",
        "date_status",
        "date_reason",
        "n_post_signal_long",
        "n_constructable_long",
        "n_included_long",
        "n_included_short",
    ]
    missing_funnel = [c for c in funnel_need if c not in funnel.columns]
    if missing_funnel:
        raise D2ValidationError(f"funnel missing columns: {missing_funnel}")

    mid = mid_status.copy()
    cross = cross_status.copy()
    fun = funnel.copy()
    mid["trade_date"] = mid["trade_date"].map(_as_date)
    cross["trade_date"] = cross["trade_date"].map(_as_date)
    fun["trade_date"] = fun["trade_date"].map(_as_date)
    if mid["trade_date"].duplicated().any() or mid["trade_date"].isna().any():
        raise D2ValidationError("mid date_status trade_date is not unique and non-null")
    if cross["trade_date"].duplicated().any():
        raise D2ValidationError("cross date_status trade_date is not unique")
    if fun["trade_date"].duplicated().any():
        raise D2ValidationError("funnel trade_date is not unique")

    allowed = {"traded", "valid_no_trade", "failed"}
    mid_w, _, _ = filter_to_window(
        mid,
        pd.DataFrame(columns=["trade_date"]),
        pd.DataFrame(columns=["trade_date"]),
        window_start,
        window_end,
    )
    cross_w, _, _ = filter_to_window(
        cross,
        pd.DataFrame(columns=["trade_date"]),
        pd.DataFrame(columns=["trade_date"]),
        window_start,
        window_end,
    )
    if mid_w.empty:
        raise D2ValidationError("evaluation date_status window is empty")
    statuses = set(mid_w["status"].astype(str))
    if not statuses.issubset(allowed):
        raise D2ValidationError(f"date_status has unknown status values: {statuses - allowed}")

    mid_key = mid_w.assign(
        status=mid_w["status"].astype(str),
        reason=mid_w["reason"].map(_reason_text),
    )[["trade_date", "status", "reason"]].sort_values("trade_date")
    cross_key = cross_w.assign(
        status=cross_w["status"].astype(str),
        reason=cross_w["reason"].map(_reason_text),
    )[["trade_date", "status", "reason"]].sort_values("trade_date")
    if not mid_key.reset_index(drop=True).equals(cross_key.reset_index(drop=True)):
        raise D2ValidationError(
            "mid and cross date_status disagree on evaluation trade_date, status, or reason"
        )

    fun_w = fun.loc[fun["trade_date"].isin(set(mid_key["trade_date"]))].copy()
    if set(fun_w["trade_date"]) != set(mid_key["trade_date"]) or fun_w["trade_date"].duplicated().any():
        missing_dates = sorted(set(mid_key["trade_date"]) - set(fun_w["trade_date"]))
        raise D2ValidationError(
            f"funnel is not one-to-one with the evaluation calendar; missing={missing_dates[:10]}"
        )

    reconstructed = reconstruct_capped_long_n(long_trades.copy() if long_trades is not None else pd.DataFrame())
    if reconstructed.empty:
        reconstructed = pd.DataFrame(columns=["trade_date", "structure_ok", "in_N"])
    else:
        reconstructed = reconstructed.copy()
        reconstructed["trade_date"] = reconstructed["trade_date"].map(_as_date)
    in_window = reconstructed["trade_date"].map(
        lambda d: window_start <= _as_date(d) <= window_end
    ) if not reconstructed.empty else pd.Series(dtype=bool)
    recon_w = reconstructed.loc[in_window].copy() if not reconstructed.empty else reconstructed
    cal_dates = set(mid_key["trade_date"])
    if not recon_w.empty:
        stray = sorted(
            {
                _as_date(d)
                for d in recon_w.loc[
                    (recon_w["in_N"] == True) | (recon_w["structure_ok"] == True),  # noqa: E712
                    "trade_date",
                ]
                if _as_date(d) not in cal_dates
            }
        )
        if stray:
            raise D2ValidationError(
                "reconstructed in_N or structure_ok dates are absent from the "
                f"evaluation calendar: {stray[:10]}"
            )

    joined = mid_key.merge(fun_w, on="trade_date", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for rec in joined.itertuples(index=False):
        td = _as_date(rec.trade_date)
        status = str(rec.status)
        reason = _reason_text(rec.reason)
        fun_status = str(rec.date_status) if rec.date_status is not None and not pd.isna(rec.date_status) else None
        fun_reason = _reason_text(rec.date_reason)
        if fun_status != status or fun_reason != reason:
            raise D2ValidationError(
                f"{td}: date_status/reason disagree with funnel "
                f"({status}/{reason} vs {fun_status}/{fun_reason})"
            )
        n_post = _int_or_none(rec.n_post_signal_long)
        n_con = _int_or_none(rec.n_constructable_long)
        n_inc = _int_or_none(rec.n_included_long)
        n_short = _int_or_none(rec.n_included_short)
        if status == "failed" or reason == "missing_features" or None in (n_post, n_con, n_inc, n_short):
            raise D2ValidationError(
                f"{td}: missing or failed required data (status={status}, reason={reason}); "
                "not cash"
            )
        day = recon_w.loc[recon_w["trade_date"] == td] if not recon_w.empty else recon_w
        n_ok = int((day["structure_ok"] == True).sum()) if not day.empty else 0  # noqa: E712
        n_in = int((day["in_N"] == True).sum()) if not day.empty else 0  # noqa: E712
        if n_inc > 0 and n_con == 0:
            raise D2ValidationError(
                f"{td}: n_included_long={n_inc} while n_constructable_long=0"
            )
        if n_con == 0:
            if n_ok != 0 or n_in != 0 or n_inc != 0:
                raise D2ValidationError(
                    f"{td}: verified-zero funnel counts disagree with reconstructed longs "
                    f"(structure_ok={n_ok}, in_N={n_in}, included={n_inc})"
                )
            if status == "traded" and n_short <= 0:
                raise D2ValidationError(
                    f"{td}: traded status with zero long candidates requires n_included_short>0"
                )
            if reason == "empty_signals" and n_post != 0:
                raise D2ValidationError(
                    f"{td}: empty_signals requires n_post_signal_long=0"
                )
            rows.append(
                {
                    "trade_date": td,
                    "status": status,
                    "reason": reason,
                    "n_in_N": 0,
                    "long_book_class": "verified_zero_long",
                    "n_constructable_long": 0,
                    "n_included_long": 0,
                    "n_post_signal_long": n_post,
                    "n_included_short": n_short,
                }
            )
            continue
        expected_n = min(int(n_con), int(MAX_NAMES))
        if n_ok != int(n_con) or n_in != expected_n:
            raise D2ValidationError(
                f"{td}: reconstructed in_N {n_in} (structure_ok {n_ok}) != "
                f"min(n_constructable_long={n_con}, MAX_NAMES={MAX_NAMES})"
            )
        rows.append(
            {
                "trade_date": td,
                "status": status,
                "reason": reason,
                "n_in_N": expected_n,
                "long_book_class": "has_long_candidates",
                "n_constructable_long": int(n_con),
                "n_included_long": int(n_inc),
                "n_post_signal_long": n_post,
                "n_included_short": n_short,
            }
        )
    calendar = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    return calendar


def load_d2_evaluation_calendar(run_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the pinned official calendar and classify long-book dates.

    Does not read evaluation P&L.
    """
    run_dir = Path(run_dir or OFFICIAL_RUN_DIR)
    mid_path = run_dir / "date_status_sprint006_baseline_v1_mid.parquet"
    cross_path = run_dir / "date_status_sprint006_baseline_v1_cross.parquet"
    funnel_path = run_dir / "funnel_summary_sprint006_baseline_v1_mid.parquet"
    for path in (mid_path, cross_path, funnel_path):
        if not path.exists():
            raise D2ValidationError(f"missing calendar input: {path}")
    mid_rows = int(pq.ParquetFile(mid_path).metadata.num_rows)
    cross_rows = int(pq.ParquetFile(cross_path).metadata.num_rows)
    if mid_rows != EXPECTED_DATE_STATUS_ROWS or cross_rows != EXPECTED_DATE_STATUS_ROWS:
        raise D2ValidationError(
            f"date_status metadata row count mid={mid_rows} cross={cross_rows}; "
            f"expected {EXPECTED_DATE_STATUS_ROWS}"
        )
    mid_schema = set(pq.read_schema(mid_path).names)
    if not set(DATE_STATUS_COLUMNS).issubset(mid_schema):
        raise D2ValidationError(f"mid date_status columns {sorted(mid_schema)}")

    mid = pd.read_parquet(mid_path, columns=list(DATE_STATUS_COLUMNS))
    cross = pd.read_parquet(cross_path, columns=list(DATE_STATUS_COLUMNS))
    funnel = pd.read_parquet(funnel_path)
    long_trades = load_full_long_trade_log(run_dir, fill_label="mid")
    mid_dates = mid["trade_date"].map(_as_date)
    n_after = int((mid_dates > EVAL_END).sum())
    calendar = classify_evaluation_calendar(mid, cross, funnel, long_trades)
    assert_no_development_rows(calendar, context="evaluation_calendar")
    validation = {
        "source_run_dir": str(run_dir),
        "date_status_file": mid_path.name,
        "funnel_file": funnel_path.name,
        "metadata_row_count_mid": mid_rows,
        "metadata_row_count_cross": cross_rows,
        "n_dates_after_primary_end_excluded": n_after,
        "n_calendar_dates": int(len(calendar)),
        "n_verified_zero_long": int((calendar["long_book_class"] == "verified_zero_long").sum()),
        "n_has_long_candidates": int((calendar["long_book_class"] == "has_long_candidates").sum()),
        "window": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "fees": FEES,
        "max_names": MAX_NAMES,
    }
    return calendar, validation


def run_d2_inference(portfolios: dict[str, pd.DataFrame], calendar: pd.DataFrame) -> dict[str, Any]:
    cal_dates = [ _as_date(d) for d in calendar["trade_date"].tolist() ]
    if len(cal_dates) < 2:
        raise D2ValidationError(f"Need >=2 calendar dates for HAC; got {len(cal_dates)}")
    gaps = [(cal_dates[i] - cal_dates[i - 1]).days for i in range(1, len(cal_dates))]
    out: dict[str, Any] = {
        "family_size": FAMILY_SIZE,
        "ordinary_ci_level": ORDINARY_CI_LEVEL,
        "adjusted_ci_level": ADJUSTED_CI_LEVEL,
        "maxlags": HAC_MAXLAGS,
        "kernel": HAC_KERNEL,
        "use_correction": HAC_USE_CORRECTION,
        "reference": "student_t",
        "df_rule": "T-1",
        "calendar_n": len(cal_dates),
        "calendar_gap_days_max": int(max(gaps)) if gaps else 0,
        "calendar_gap_days_median": float(np.median(gaps)) if gaps else 0.0,
        "contrasts": {},
    }
    series_dates: list[list[date]] = []
    for measurement in FOLLOWUP_MEASUREMENTS:
        port = portfolios[measurement].sort_values("trade_date")
        dates = [_as_date(d) for d in port["trade_date"].tolist()]
        series_dates.append(dates)
        if dates != cal_dates:
            raise D2ValidationError(
                f"{measurement} uplift series is not the complete evaluation calendar"
            )
        inferred = newey_west_intercept_inference(
            port["uplift"].to_numpy(dtype=float),
            maxlags=HAC_MAXLAGS,
            use_correction=HAC_USE_CORRECTION,
            ordinary_level=ORDINARY_CI_LEVEL,
            adjusted_level=ADJUSTED_CI_LEVEL,
        )
        p_adj = bonferroni_adjust_p(inferred["p_raw"], family_size=FAMILY_SIZE)
        lo, hi = inferred["ci_adjusted"]
        out["contrasts"][measurement] = {
            **inferred,
            "p_adjusted": p_adj,
            "label": interpret_adjusted_interval(float(lo), float(hi)),
            "n_dates": int(len(port)),
        }
    if series_dates[0] != series_dates[1]:
        raise D2ValidationError("M1 and M2 uplift series calendars differ")
    return out


def run_fixed_exclusion_validation(*, run_dir: Path | None = None) -> D2ValidationResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    run_dir = Path(run_dir or OFFICIAL_RUN_DIR)

    panel = build_d1_base_panel(run_dir=run_dir)
    timings["build_panel"] = _progress("build_panel", t0, note=f"rows={len(panel)}")

    t_cal = time.perf_counter()
    calendar, calendar_validation = load_d2_evaluation_calendar(run_dir)
    timings["calendar"] = _progress(
        "calendar",
        t_cal,
        note=(
            f"dates={calendar_validation['n_calendar_dates']} "
            f"zero={calendar_validation['n_verified_zero_long']}"
        ),
    )

    t_filt = time.perf_counter()
    panel_eval = filter_evaluation_panel(panel)
    assert_no_development_rows(panel_eval, context="panel_eval")
    # Firewall check is one-way: development analysis must not see eval rows.
    # This call confirms the helper still treats eval dates as eval.
    if not panel_eval.empty and not panel_eval["trade_date"].map(lambda d: not is_development_date(d)).all():
        raise D2ValidationError("development dates remain in evaluation panel")
    timings["filter_eval"] = _progress("filter_eval", t_filt, note=f"rows={len(panel_eval)}")

    t_d0 = time.perf_counter()
    d0_check = enforce_d0_required_inputs(panel_eval)
    timings["d0_checks"] = _progress("d0_checks", t_d0, note=str(d0_check))

    t_econ = time.perf_counter()
    panel_econ = attach_scenario_economics(panel_eval, PRIMARY_H)
    recon = reconcile_economics(panel_econ)
    if not recon["passed"]:
        raise D2ValidationError(f"Economics reconciliation failed: {recon}")
    panel_econ = attach_decomposition_columns(panel_econ)
    require_all_executed_outcomes(panel_econ)
    timings["economics"] = _progress("economics", t_econ, note=f"recon={recon}")

    trade_level = panel_econ.copy()
    paired_all = []
    excluded_all = []
    portfolios: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for measurement in FOLLOWUP_MEASUREMENTS:
        tm = time.perf_counter()
        trade_level, paired, excluded = assign_eval_u_labels(trade_level, measurement)
        paired_all.append(paired)
        excluded_all.append(excluded)
        port, summary = build_portfolio_comparison(
            trade_level,
            paired,
            measurement,
            window_start=EVAL_START,
            window_end=EVAL_END,
            entry_calendar=calendar,
            reporting_periods=D2_REPORTING_PERIODS,
        )
        assert_no_development_rows(port, context=f"portfolio[{measurement}]")
        if list(port["trade_date"].map(_as_date)) != list(calendar["trade_date"].map(_as_date)):
            raise D2ValidationError(f"{measurement} portfolio calendar mismatch")
        for name, metrics in summary["reporting_periods"].items():
            metrics["period_label"] = D2_PERIOD_LABELS[name]
        portfolios[measurement] = port
        summaries[measurement] = summary
        timings[measurement] = _progress(
            measurement,
            tm,
            note=(
                f"mean_uplift={summary['mean_uplift']:.6f} "
                f"pnl_imp={summary['total_pnl_improvement']:.2f}"
            ),
        )

    if portfolios["M1"]["trade_date"].map(_as_date).tolist() != portfolios["M2"]["trade_date"].map(_as_date).tolist():
        raise D2ValidationError("M1 and M2 date-level calendars differ")

    t_inf = time.perf_counter()
    inference = run_d2_inference(portfolios, calendar)
    timings["inference"] = _progress(
        "inference",
        t_inf,
        note=" ".join(
            f"{m}={inference['contrasts'][m]['label']}" for m in FOLLOWUP_MEASUREMENTS
        ),
    )

    paired_df = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
    excluded_df = pd.concat(excluded_all, ignore_index=True) if excluded_all else pd.DataFrame()
    port_df = pd.concat(
        [portfolios[m] for m in FOLLOWUP_MEASUREMENTS], ignore_index=True
    )
    assert_no_development_rows(trade_level, context="trade_level")
    assert_no_development_rows(port_df, context="date_portfolio")

    report = {
        "protocol": DESIGN_PATH,
        "status": "retrospective_validation_awaiting_review",
        "characterization": (
            "Retrospective validation of already frozen rules. "
            "Not a pristine holdout or independent confirmation."
        ),
        "preserves_original_d1_gate": ORIGINAL_D1_GATE,
        "reviewed_design_sha": REVIEWED_DESIGN_SHA,
        "pins": {
            "window": [EVAL_START.isoformat(), EVAL_END.isoformat()],
            "budget_b": 10000.0,
            "fees": FEES,
            "h": PRIMARY_H,
            "family_size": FAMILY_SIZE,
            "ordinary_ci_level": ORDINARY_CI_LEVEL,
            "adjusted_ci_level": ADJUSTED_CI_LEVEL,
            "hac_maxlags": HAC_MAXLAGS,
            "kernel": HAC_KERNEL,
            "small_sample_correction": HAC_USE_CORRECTION,
        },
        "calendar_validation": calendar_validation,
        "economics_reconciliation": recon,
        "d0_checks": d0_check,
        "summaries": summaries,
        "inference": inference,
        "published_development_comparison": PUBLISHED_DEVELOPMENT,
        "no_pooled_significance": True,
        "no_automatic_promotion": True,
        "timings": timings,
    }
    return D2ValidationResult(
        calendar=calendar,
        trade_level=trade_level,
        date_portfolio=port_df,
        excluded_dates=excluded_df,
        summaries=summaries,
        inference=inference,
        calendar_validation=calendar_validation,
        report=report,
        stage_timings=timings,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


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
        "src/backtest/sprint008_d2_fixed_exclusion_validation.py",
        "src/backtest/sprint008_d1_cost_diagnosis.py",
        "src/backtest/sprint008_d1_within_date_followup.py",
        "src/backtest/sprint008_d1_measurement_validation.py",
        "src/backtest/sprint008_d0_input_readiness.py",
    ]
    hashes = {}
    for rel in paths:
        p = root / rel
        if p.exists():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    dirty = _working_tree_status()
    diff_sha = None
    if dirty == "dirty":
        try:
            diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
            diff_sha = hashlib.sha256(diff).hexdigest() if diff else "no_diff"
        except (OSError, subprocess.CalledProcessError):
            diff_sha = "unavailable"
    return {
        "code_sha": get_current_repo_sha(),
        "working_tree": dirty,
        "file_sha256": hashes,
        "diff_sha256": diff_sha,
    }


def _plot_cumulative(port_df: pd.DataFrame, measurement: str, path: Path) -> None:
    sub = port_df.loc[port_df["measurement"] == measurement].sort_values("trade_date")
    if sub.empty:
        return
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
    ax.set_title(f"Cumulative fixed-budget $ P&L — {measurement} (leading 0)")
    ax.set_ylabel("Cumulative $")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt_pp(value: Any) -> str:
    if value is None or not _finite(value):
        return "NA"
    return f"{100.0 * float(value):.2f} pp"


def _fmt_money(value: Any) -> str:
    if value is None or not _finite(value):
        return "NA"
    return f"${float(value):,.2f}"


def _fmt_pct(value: Any) -> str:
    if value is None or not _finite(value):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def render_d2_report_md(result: D2ValidationResult) -> str:
    cal = result.calendar_validation
    lines = [
        "# Sprint 008 D2 — retrospective validation evidence",
        "",
        "Retrospective validation of already frozen M1 and M2 exclude-U rules. "
        "Not a pristine holdout or independent confirmation. "
        "Historical D1 `STOP_NO_THRESHOLDS` is preserved. "
        "A relative-inference label is not a promotion of a filter.",
        "",
        f"- Window: `{EVAL_START.isoformat()}` through `{EVAL_END.isoformat()}` inclusive",
        f"- Calendar dates: {cal.get('n_calendar_dates')}",
        f"- Verified zero-long dates: {cal.get('n_verified_zero_long')}",
        f"- Dates with long candidates: {cal.get('n_has_long_candidates')}",
        f"- Dates after {EVAL_END.isoformat()} excluded from the source calendar: "
        f"{cal.get('n_dates_after_primary_end_excluded')}",
        f"- Fees: {FEES} (disclosed; unmodeled)",
        "",
        "## Inference",
        "",
        "Family size 2. HAC maxlags 3, Bartlett, small-sample correction, Student-t with T-1. "
        f"Adjusted p = min(1, 2 x raw p). Adjusted interval is {ADJUSTED_CI_LEVEL:.1%}.",
        "",
        "| Rule | Mean uplift | HAC SE | 95% CI | Raw p | Adjusted p | 97.5% CI | Label |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for measurement in FOLLOWUP_MEASUREMENTS:
        inf = result.inference["contrasts"][measurement]
        lines.append(
            "| {m} | {mean} | {se} | [{lo}, {hi}] | {praw:.4f} | {padj:.4f} | [{alo}, {ahi}] | `{label}` |".format(
                m=measurement,
                mean=_fmt_pp(inf["mean"]),
                se=_fmt_pp(inf["se_hac"]),
                lo=_fmt_pp(inf["ci_ordinary"][0]),
                hi=_fmt_pp(inf["ci_ordinary"][1]),
                praw=float(inf["p_raw"]),
                padj=float(inf["p_adjusted"]),
                alo=_fmt_pp(inf["ci_adjusted"][0]),
                ahi=_fmt_pp(inf["ci_adjusted"][1]),
                label=inf["label"],
            )
        )
    lines.extend(
        [
            "",
            f"Calendar gaps (successive evaluation dates): median {result.inference['calendar_gap_days_median']:.1f} days, "
            f"max {result.inference['calendar_gap_days_max']} days.",
            "",
            "M1's relative_benefit label is statistically supported relative improvement "
            "in this retrospective evaluation. Neither that label nor a positive incremental "
            "P&L means the filtered book is profitable over the full window; read absolute "
            "filtered P&L separately. An inconclusive label does not establish relative "
            "benefit or harm. These contrasts do not directly test M1 against M2 or isolate "
            "selection benefit from reduced exposure.",
            "",
        ]
    )
    for measurement in FOLLOWUP_MEASUREMENTS:
        summary = result.summaries[measurement]
        weekly = summary["weekly_uplift"]
        lines.extend(
            [
                f"## {measurement}",
                "",
                f"- Baseline P&L: {_fmt_money(summary['total_pnl_baseline'])}",
                f"- Filtered P&L: {_fmt_money(summary['total_pnl_filtered'])}",
                f"- Incremental P&L: {_fmt_money(summary['total_pnl_improvement'])}",
                f"- Mean return on original B, baseline / filtered: "
                f"{_fmt_pp(summary['mean_R_baseline'])} / {_fmt_pp(summary['mean_R_filtered'])}",
                f"- Losses avoided: {_fmt_money(summary['losses_avoided'])}",
                f"- Winning profits sacrificed: {_fmt_money(summary['winning_profits_sacrificed'])}",
                f"- Winning-profit retention: {_fmt_pct(summary['winning_profit_retention'])}",
                f"- Top-5 / top-10 winner-profit retention: "
                f"{_fmt_pct(summary.get('top5_profit_retention'))} / "
                f"{_fmt_pct(summary.get('top10_profit_retention'))}",
                f"- Dates exclusion applied: {summary['n_dates_exclusion_applied']} of {summary['n_dates']}",
                f"- Mean invested / cash fraction (filtered): "
                f"{_fmt_pct(summary['mean_invested_frac_filtered'])} / "
                f"{_fmt_pct(summary['mean_cash_frac_filtered'])}",
                f"- Baseline / filtered drawdown: "
                f"{_fmt_money(summary['peak_to_trough_baseline'])} / "
                f"{_fmt_money(summary['peak_to_trough_filtered'])}",
                f"- Period reconcile: {summary.get('period_reconcile_ok')}",
                "",
                "| Period | Label | Dates | Baseline $ | Filtered $ | Mean uplift | Losses avoided | Sacrificed | Winner retention |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in summary["reporting_periods"].items():
            lines.append(
                f"| {name} | {metrics.get('period_label')} | {metrics['n_dates']} | "
                f"{_fmt_money(metrics['pnl_baseline'])} | {_fmt_money(metrics['pnl_filtered'])} | "
                f"{_fmt_pp(metrics['mean_uplift'])} | {_fmt_money(metrics['losses_avoided'])} | "
                f"{_fmt_money(metrics['winning_profits_sacrificed'])} | "
                f"{_fmt_pct(metrics['winning_profit_retention'])} |"
            )
        lines.extend(
            [
                "",
                f"Weekly uplift (n={weekly.get('n_dates')}): mean {_fmt_pp(weekly.get('mean'))}, "
                f"median {_fmt_pp(weekly.get('median'))}, std {_fmt_pp(weekly.get('std'))}.",
                f"Fractions positive / zero / negative: "
                f"{_fmt_pct(weekly.get('frac_positive'))} / {_fmt_pct(weekly.get('frac_zero'))} / "
                f"{_fmt_pct(weekly.get('frac_negative'))}.",
                "",
                "Largest positive weeks:",
                "",
            ]
        )
        for week in weekly.get("five_largest_positive_weeks") or []:
            lines.append(
                f"- {week['trade_date']}: {_fmt_money(week['dollar_contribution'])} ({_fmt_pp(week['uplift'])})"
            )
        lines.extend(["", "Largest negative weeks:", ""])
        for week in weekly.get("five_largest_negative_weeks") or []:
            lines.append(
                f"- {week['trade_date']}: {_fmt_money(week['dollar_contribution'])} ({_fmt_pp(week['uplift'])})"
            )
        lines.append("")

    lines.extend(
        [
            "## Comparison with published development findings",
            "",
            "Copied from the reviewed cost-diagnosis evidence. Periods are not pooled. "
            "Development inference used family size 4 and is not a D2 p-value.",
            "",
            "| | Development M1 | Development M2 |",
            "|---|---:|---:|",
            f"| Baseline P&L | {_fmt_money(PUBLISHED_DEVELOPMENT['baseline_pnl'])} | {_fmt_money(PUBLISHED_DEVELOPMENT['baseline_pnl'])} |",
        ]
    )
    for key, label in (
        ("filtered_pnl", "Filtered P&L"),
        ("improvement", "Incremental P&L"),
        ("mean_weekly_uplift", "Mean weekly uplift"),
        ("p_raw", "Development raw p (family 4 context)"),
        ("label", "Development label"),
    ):
        left = PUBLISHED_DEVELOPMENT["M1"][key]
        right = PUBLISHED_DEVELOPMENT["M2"][key]
        if key == "mean_weekly_uplift":
            left_s, right_s = _fmt_pp(left), _fmt_pp(right)
        elif key in {"filtered_pnl", "improvement"}:
            left_s, right_s = _fmt_money(left), _fmt_money(right)
        else:
            left_s, right_s = str(left), str(right)
        lines.append(f"| {label} | {left_s} | {right_s} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Retrospective window; earlier sprints inspected this history.",
            "- Fees remain 0.",
            "- Quote-based full-cross results do not establish attainable live fills.",
            "- Drawdown is peak-to-trough of cumulative fixed-budget dollar P&L, including the initial zero. Not compounded equity and not intraholding-period risk.",
            "- No cutoff was tuned after results. No automatic next experiment. D3 remains the closeout.",
            "",
        ]
    )
    return "\n".join(lines)


def export_d2_evidence(
    *,
    result: D2ValidationResult,
    evidence_dir: Path,
    command: str,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=False)
    result.calendar.to_parquet(evidence_dir / "evaluation_calendar.parquet", index=False)
    result.calendar.to_csv(evidence_dir / "evaluation_calendar.csv", index=False)
    result.trade_level.to_parquet(evidence_dir / "trade_level.parquet", index=False)
    result.date_portfolio.to_parquet(evidence_dir / "date_portfolio.parquet", index=False)
    result.date_portfolio.to_csv(evidence_dir / "date_portfolio.csv", index=False)
    result.excluded_dates.to_parquet(evidence_dir / "excluded_dates.parquet", index=False)
    (evidence_dir / "summaries.json").write_text(
        json.dumps(_json_ready(result.summaries), indent=2), encoding="utf-8"
    )
    (evidence_dir / "inference.json").write_text(
        json.dumps(_json_ready(result.inference), indent=2), encoding="utf-8"
    )
    report_md = render_d2_report_md(result)
    (evidence_dir / "report.md").write_text(report_md, encoding="utf-8")
    for measurement in FOLLOWUP_MEASUREMENTS:
        _plot_cumulative(result.date_portfolio, measurement, evidence_dir / f"cum_pnl_{measurement}.png")
    receipt = {
        "command": command,
        "evidence_dir": str(evidence_dir),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": DESIGN_PATH,
        "reviewed_design_sha": REVIEWED_DESIGN_SHA,
        "provenance": _source_provenance(),
        "pins": result.report["pins"],
        "timings": result.stage_timings,
        "calendar_validation": result.calendar_validation,
        "labels": {
            m: result.inference["contrasts"][m]["label"] for m in FOLLOWUP_MEASUREMENTS
        },
    }
    import sys

    import numpy
    import pandas
    import pyarrow
    import scipy

    receipt["versions"] = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": pyarrow.__version__,
    }
    (evidence_dir / "execution_receipt.json").write_text(
        json.dumps(_json_ready(receipt), indent=2), encoding="utf-8"
    )
