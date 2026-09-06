"""
Sprint 007 D3 — required execution envelope.

Read-only post-pass. Path R is the primary resized envelope; Path F is a
fixed-quantity diagnostic and must not bind Path R. No h_req. No SurfaceRunner.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from src.backtest.pipeline import _apply_tier_a_sizing, _at_risk_per_share
from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_RUN_DIR,
    TRADE_KEY,
    get_current_repo_sha,
    run_d0_validation,
    sha256_file,
)
from src.backtest.sprint007_d1_gross_margin import VERDICT_CONTINUE, run_d1_analysis
from src.backtest.sprint007_d2_shortfall_bridge import (
    CLASS_EXECUTION,
    compute_bridge_terms,
    load_accepted_primary_block,
    load_fill_primary_tables,
)
from src.backtest.sprint007_d2b_package_tradability import run_d2b_analysis
from src.backtest.surface_decision_report import PRIMARY_END, PRIMARY_START, compute_view_a
from src.backtest.surface_metrics import build_date_summary

EVIDENCE_DIR_ENV = "SPRINT007_D3_EVIDENCE_DIR"

H_TOL = 1e-4
H_DET_STEP = 0.01
H_VIS = tuple(i / 20 for i in range(21))
H_DET = tuple(i / 100 for i in range(101))
PATH_F_ALPHAS = (0.50, 0.25, 0.00)
PNL_ABS_TOLERANCE = 0.01
PNL_REL_TOLERANCE = 1e-9
CAR_TOLERANCE = 1e-9
Q_ABS_TOLERANCE = 1e-6
Q_REL_TOLERANCE = 1e-9
EXPECTED_INCLUDED_TRADES = 9212
EXPECTED_TRADED_DATES = 341
TIER_A_CONFIG = SimpleNamespace(
    tier_a_mode="equal_max_loss",
    tier_a_short_budget=10000.0,
    tier_a_long_budget=10000.0,
)
VERDICT_BLOCKED = "D3_BLOCKED"
VERDICT_ENVELOPE = "D3_ENVELOPE"
NO_CROSSING = "no_crossing"


def _d3_log(message: str) -> None:
    print(f"[D3] {message}", flush=True)


class D3AnalysisError(Exception):
    """Raised when the envelope cannot be formed from persisted artifacts."""


def dollar_tolerance(reference: float) -> float:
    return max(PNL_ABS_TOLERANCE, PNL_REL_TOLERANCE * abs(float(reference)))


def within_dollar(value: float, reference: float) -> bool:
    return abs(float(value) - float(reference)) <= dollar_tolerance(reference)


def quantity_tolerance(reference: float) -> float:
    return max(Q_ABS_TOLERANCE, Q_REL_TOLERANCE * abs(float(reference)))


def fill_price_at_h(bid: float, ask: float, unit_quantity: float, h: float) -> float:
    if not 0.0 <= float(h) <= 1.0:
        raise ValueError(f"h must be in [0, 1], got {h!r}")
    qty = float(unit_quantity)
    if qty == 0.0:
        raise ValueError("unit_quantity must be non-zero")
    mid = float(bid) + 0.5 * (float(ask) - float(bid))
    cross = float(ask) if qty > 0.0 else float(bid)
    return mid + float(h) * (cross - mid)


def package_entry_cost_at_h(leg_rows: pd.DataFrame, h: float) -> float:
    total = 0.0
    for row in leg_rows.itertuples(index=False):
        fill = fill_price_at_h(row.bid, row.ask, row.unit_quantity, h)
        cash = fill * abs(float(row.unit_quantity))
        total += cash if float(row.unit_quantity) > 0.0 else -cash
    return float(total)


def path_f_pnl_crossing(alpha: float, p_mid: float, delta_price: float) -> float:
    if float(alpha) not in PATH_F_ALPHAS:
        raise ValueError(f"alpha must be one of {PATH_F_ALPHAS}, got {alpha!r}")
    if float(delta_price) >= 0.0:
        raise D3AnalysisError("Path F closed form requires Δ_price < 0")
    return (1.0 - float(alpha)) * float(p_mid) / (-float(delta_price))


def first_adverse_crossing(eval_fn: Callable[[float], float], target: float) -> float | None:
    """H_det + midpoint miss-check + bisection to H_TOL. Not H_vis interpolation."""
    m0 = float(eval_fn(0.0))
    if m0 <= float(target):
        raise D3AnalysisError("m(0) already at or below target")
    values: dict[float, float] = {h: float(eval_fn(h)) for h in H_DET}
    extra: list[float] = []
    for left, right in zip(H_DET, H_DET[1:]):
        if values[left] > target and values[right] > target:
            mid = 0.5 * (left + right)
            extra.append(mid)
            values[mid] = float(eval_fn(mid))
    ordered = sorted(values)
    bracket: tuple[float, float] | None = None
    for a, b in zip(ordered, ordered[1:]):
        if values[a] > target and values[b] <= target:
            bracket = (a, b)
            break
    if bracket is None:
        return None
    h_lo, h_hi = bracket
    while h_hi - h_lo > H_TOL:
        mid = 0.5 * (h_lo + h_hi)
        if float(eval_fn(mid)) <= target:
            h_hi = mid
        else:
            h_lo = mid
    return float(h_hi)


def monotonic_nonincreasing(eval_fn: Callable[[float], float], slack: float) -> bool:
    prev = float(eval_fn(H_DET[0]))
    for h in H_DET[1:]:
        cur = float(eval_fn(h))
        if cur > prev + slack:
            return False
        prev = cur
    return True


def per_share_economics_at_h(
    *,
    direction: str,
    legs: pd.DataFrame,
    h: float,
    p_mid: float,
    p_cross: float,
    wing_width: float | None,
) -> dict[str, float]:
    entry_cost = package_entry_cost_at_h(legs, h)
    net_credit = -entry_cost
    if direction == "long":
        max_loss = abs(entry_cost)
    else:
        if wing_width is None or pd.isna(wing_width):
            raise D3AnalysisError("short trade missing wing_width")
        max_loss = max(float(wing_width) - net_credit, 0.0)
    return {
        "entry_cost_per_share": entry_cost,
        "net_credit_per_share": net_credit,
        "max_loss_per_share": max_loss,
        "pnl_per_share": (1.0 - float(h)) * float(p_mid) + float(h) * float(p_cross),
    }


def size_book_at_h(trades: pd.DataFrame, h: float, config: Any | None = None) -> pd.DataFrame:
    """Apply existing Tier-A sizing by trade date. ``h`` is recorded only."""
    _ = h
    work = trades.copy()
    work["included_in_portfolio"] = True
    work["quantity"] = float("nan")
    cfg = config or TIER_A_CONFIG
    for _, idx in work.groupby("trade_date", sort=True).groups.items():
        day = work.loc[list(idx)].copy()
        _apply_tier_a_sizing(day, cfg)
        work.loc[day.index, "quantity"] = day["quantity"]
        work.loc[day.index, "included_in_portfolio"] = day["included_in_portfolio"]
    dropped = work.loc[work["included_in_portfolio"] != True]  # noqa: E712
    if not dropped.empty or work["quantity"].isna().any():
        raise D3AnalysisError("Path R sizing would drop a frozen key")
    return work


def _at_risk_row(direction: str, entry_cost: float, net_credit: float, max_loss: float) -> float:
    risk = _at_risk_per_share(
        pd.Series(
            {
                "direction": direction,
                "entry_cost_per_share": entry_cost,
                "net_credit_per_share": net_credit,
                "max_loss_per_share": max_loss,
            }
        )
    )
    if risk is None or risk <= 0.0:
        raise D3AnalysisError("non-positive at-risk per share")
    return float(risk)


def evaluate_path_at_h(book: "D3Book", h: float, *, path: str) -> dict[str, Any]:
    cache_key = (path, float(h))
    cached = book.eval_cache.get(cache_key)
    if cached is not None:
        return cached
    rows = []
    for trade in book.trades.itertuples(index=False):
        trade_key = (trade.trade_date, trade.ticker, trade.direction)
        legs = book.legs_by_key[trade_key]
        econ = per_share_economics_at_h(
            direction=trade.direction,
            legs=legs,
            h=h,
            p_mid=trade.pnl_per_share_mid,
            p_cross=trade.pnl_per_share_cross,
            wing_width=getattr(trade, "wing_width", None),
        )
        rows.append({**trade._asdict(), **econ, "included_in_portfolio": True})
    priced = pd.DataFrame(rows)
    if path == "F":
        priced["quantity"] = priced["quantity_mid"]
    elif path == "R":
        priced = size_book_at_h(priced, h)
    else:
        raise ValueError(path)
    qty = priced["quantity"].abs().astype(float)
    priced["pnl_total"] = qty * priced["pnl_per_share"].astype(float)
    priced["capital_at_risk_dollars"] = [
        abs(float(q))
        * _at_risk_row(
            str(row.direction),
            float(row.entry_cost_per_share),
            float(row.net_credit_per_share),
            float(row.max_loss_per_share),
        )
        for q, row in zip(priced["quantity"], priced.itertuples(index=False))
    ]
    summary = build_date_summary(priced)
    car = float(compute_view_a(book.date_status, summary)["mean_cycle_car"])
    side = priced.groupby("direction", dropna=False)["pnl_total"].sum().to_dict()
    result = {
        "h": float(h),
        "path": path,
        "pnl": float(priced["pnl_total"].sum()),
        "car": car,
        "sum_abs_qty": float(qty.sum()),
        "capital": float(priced["capital_at_risk_dollars"].sum()),
        "n_trades": int(len(priced)),
        "n_dates": int(priced["trade_date"].nunique()),
        "pnl_long": float(side.get("long", 0.0)),
        "pnl_short": float(side.get("short", 0.0)),
        "quantities": priced.set_index(list(TRADE_KEY))["quantity"],
    }
    book.eval_cache[cache_key] = result
    book.eval_calls += 1
    return result


def _precompute_path_grid(
    book: "D3Book",
    path: str,
    grid: tuple[float, ...],
    *,
    label: str,
) -> None:
    total = len(grid)
    _d3_log(f"{label} start n={total} cached_calls={book.eval_calls}")
    for i, h in enumerate(grid, start=1):
        evaluate_path_at_h(book, h, path=path)
        if i == 1 or i == total or i % 10 == 0:
            _d3_log(f"{label} {i}/{total} h={h:.2f} eval_calls={book.eval_calls}")
    _d3_log(f"{label} done eval_calls={book.eval_calls}")


def evaluate_paths(book: "D3Book", grid: tuple[float, ...] = H_VIS) -> pd.DataFrame:
    records = []
    for h in grid:
        for path in ("R", "F"):
            metrics = evaluate_path_at_h(book, h, path=path)
            records.append({k: metrics[k] for k in metrics if k not in {"quantities", "priced"}})
    return pd.DataFrame(records)


@dataclass
class D3Book:
    trades: pd.DataFrame
    legs: pd.DataFrame
    date_status: pd.DataFrame
    legs_by_key: dict[tuple, pd.DataFrame] = field(default_factory=dict)
    car_mid_ref: float | None = None
    car_cross_ref: float | None = None
    eval_cache: dict[tuple[str, float], dict[str, Any]] = field(default_factory=dict, repr=False)
    eval_calls: int = 0

    def __post_init__(self) -> None:
        if not self.legs_by_key:
            grouped: dict[tuple, pd.DataFrame] = {}
            for key, frame in self.legs.groupby(list(TRADE_KEY), sort=False):
                grouped[key] = frame
            self.legs_by_key = grouped


@dataclass
class Crossing:
    path: str
    role: str
    metric: str
    target: float
    h: float | None
    method: str

    @property
    def no_crossing(self) -> bool:
        return self.h is None

    def as_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "metric": self.metric,
            "target": self.target,
            "h": self.h,
            "no_crossing": self.no_crossing,
            "method": self.method,
        }


@dataclass
class D3Result:
    verdict: str
    blocked: bool
    blocker: str | None
    h_R_50: float | None = None
    h_R_25: float | None = None
    h_R_P0: float | None = None
    h_R_CAR0: float | None = None
    h_F_50: float | None = None
    h_F_25: float | None = None
    h_F_P0: float | None = None
    h_F_CAR0: float | None = None
    headroom_50_to_25: float | None | str = None
    headroom_25_to_P0: float | None | str = None
    distance_P0_to_cross: float | None | str = None
    distance_CAR0_to_cross: float | None | str = None
    resize_gaps: dict[str, float | None] = field(default_factory=dict)
    monotonicity: dict[str, bool] = field(default_factory=dict)
    crossings: list[Crossing] = field(default_factory=list)
    curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    side_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "verdict": self.verdict,
            "blocked": self.blocked,
            "blocker": self.blocker,
            "path_r_envelope": {
                "h_R_50": self.h_R_50,
                "h_R_25": self.h_R_25,
                "h_R_P0": self.h_R_P0,
                "h_R_CAR0": self.h_R_CAR0,
            },
            "path_f_diagnostic": {
                "h_F_50": self.h_F_50,
                "h_F_25": self.h_F_25,
                "h_F_P0": self.h_F_P0,
                "h_F_CAR0": self.h_F_CAR0,
            },
            "headroom_50_to_25": self.headroom_50_to_25,
            "headroom_25_to_P0": self.headroom_25_to_P0,
            "distance_P0_to_cross": self.distance_P0_to_cross,
            "distance_CAR0_to_cross": self.distance_CAR0_to_cross,
            "resize_gaps": self.resize_gaps,
            "monotonicity": self.monotonicity,
            "root_finder": {"H_TOL": H_TOL, "H_DET_STEP": H_DET_STEP, "H_vis_step": 0.05},
            "crossings": [row.as_record() for row in self.crossings],
            "reconciliation": self.reconciliation,
            "manifest": self.manifest,
        }
        if "h_req" in payload:
            raise D3AnalysisError("h_req is forbidden")
        return payload


def _headroom(later: float | None, earlier: float | None) -> float | str:
    if later is None or earlier is None:
        return NO_CROSSING
    return float(later) - float(earlier)


def _distance_to_cross(h_star: float | None) -> float | str:
    if h_star is None:
        return NO_CROSSING
    return 1.0 - float(h_star)


def assemble_envelope(
    *,
    h_R_50: float | None,
    h_R_25: float | None,
    h_R_P0: float | None,
    h_R_CAR0: float | None,
    h_F_50: float | None,
    h_F_25: float | None,
    h_F_P0: float | None,
    h_F_CAR0: float | None,
    crossings: list[Crossing],
    curves: pd.DataFrame,
    reconciliation: list[dict[str, Any]],
    monotonicity: dict[str, bool],
    manifest: dict[str, Any],
    side_snapshot: pd.DataFrame | None = None,
) -> D3Result:
    """Path R fields only come from Path R crossings. Path F cannot bind."""
    result = D3Result(
        verdict=VERDICT_ENVELOPE,
        blocked=False,
        blocker=None,
        h_R_50=h_R_50,
        h_R_25=h_R_25,
        h_R_P0=h_R_P0,
        h_R_CAR0=h_R_CAR0,
        h_F_50=h_F_50,
        h_F_25=h_F_25,
        h_F_P0=h_F_P0,
        h_F_CAR0=h_F_CAR0,
        headroom_50_to_25=_headroom(h_R_25, h_R_50),
        headroom_25_to_P0=_headroom(h_R_P0, h_R_25),
        distance_P0_to_cross=_distance_to_cross(h_R_P0),
        distance_CAR0_to_cross=_distance_to_cross(h_R_CAR0),
        resize_gaps={
            "h_R_50_minus_h_F_50": None if h_R_50 is None or h_F_50 is None else h_R_50 - h_F_50,
            "h_R_25_minus_h_F_25": None if h_R_25 is None or h_F_25 is None else h_R_25 - h_F_25,
            "h_R_P0_minus_h_F_P0": None if h_R_P0 is None or h_F_P0 is None else h_R_P0 - h_F_P0,
            "h_R_CAR0_minus_h_F_CAR0": None
            if h_R_CAR0 is None or h_F_CAR0 is None
            else h_R_CAR0 - h_F_CAR0,
        },
        monotonicity=monotonicity,
        crossings=crossings,
        curves=curves,
        side_snapshot=side_snapshot if side_snapshot is not None else pd.DataFrame(),
        reconciliation=reconciliation,
        manifest=manifest,
    )
    if hasattr(result, "h_req"):
        raise D3AnalysisError("h_req is forbidden")
    return result


def check_prerequisites(
    *,
    d0_passed: bool,
    d1_verdict: str,
    d2_final_class: str,
) -> str | None:
    if not d0_passed:
        return "D0 prerequisite failed"
    if d1_verdict != VERDICT_CONTINUE:
        return f"D1 did not continue ({d1_verdict})"
    if d2_final_class != CLASS_EXECUTION:
        return f"D2 class is {d2_final_class}, not {CLASS_EXECUTION}"
    return None


def _recon_row(metric: str, recomputed: float, reference: float, tolerance: float) -> dict[str, Any]:
    delta = float(recomputed) - float(reference)
    return {
        "metric": metric,
        "recomputed": float(recomputed),
        "reference": float(reference),
        "delta": delta,
        "tolerance": float(tolerance),
        "passed": abs(delta) <= float(tolerance),
    }


def reconcile_d3_endpoints(book: D3Book, *, official: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mid = evaluate_path_at_h(book, 0.0, path="F")
    hybrid = evaluate_path_at_h(book, 1.0, path="F")
    r0 = evaluate_path_at_h(book, 0.0, path="R")
    r1 = evaluate_path_at_h(book, 1.0, path="R")
    p_mid = float(book.trades["pnl_total_mid"].sum())
    p_cross = float(book.trades["pnl_total_cross"].sum())
    p_hybrid = float((book.trades["quantity_mid"].abs() * book.trades["pnl_per_share_cross"]).sum())
    rows.append(_recon_row("F_h0_pnl", mid["pnl"], p_mid, dollar_tolerance(p_mid)))
    rows.append(_recon_row("F_h1_pnl", hybrid["pnl"], p_hybrid, dollar_tolerance(p_hybrid)))
    rows.append(_recon_row("R_h0_pnl", r0["pnl"], p_mid, dollar_tolerance(p_mid)))
    rows.append(_recon_row("R_h1_pnl", r1["pnl"], p_cross, dollar_tolerance(p_cross)))
    if official and (book.car_mid_ref is None or book.car_cross_ref is None):
        raise D3AnalysisError("official CAR references missing")
    if book.car_mid_ref is not None:
        rows.append(_recon_row("F_h0_car", mid["car"], book.car_mid_ref, CAR_TOLERANCE))
        rows.append(_recon_row("R_h0_car", r0["car"], book.car_mid_ref, CAR_TOLERANCE))
    if book.car_cross_ref is not None:
        rows.append(_recon_row("R_h1_car", r1["car"], book.car_cross_ref, CAR_TOLERANCE))
    for label, got, col in (("R_h0_Q", r0["quantities"], "quantity_mid"), ("R_h1_Q", r1["quantities"], "quantity_cross")):
        ref = book.trades.set_index(list(TRADE_KEY))[col]
        aligned = got.reindex(ref.index)
        worst = float((aligned.astype(float) - ref.astype(float)).abs().max())
        tol = max(quantity_tolerance(float(ref.abs().max())), Q_ABS_TOLERANCE)
        rows.append(_recon_row(label, worst, 0.0, tol))
    if official:
        if int(len(book.trades)) != EXPECTED_INCLUDED_TRADES:
            rows.append(
                _recon_row("n_trades", float(len(book.trades)), float(EXPECTED_INCLUDED_TRADES), 0.0)
            )
        n_dates = int(book.date_status.loc[book.date_status["status"] == "traded", "trade_date"].nunique())
        if n_dates != EXPECTED_TRADED_DATES:
            rows.append(_recon_row("n_traded_dates", float(n_dates), float(EXPECTED_TRADED_DATES), 0.0))
    return rows


def _blocked(blocker: str, reconciliation: list[dict[str, Any]] | None = None) -> D3Result:
    return D3Result(
        verdict=VERDICT_BLOCKED,
        blocked=True,
        blocker=blocker,
        reconciliation=reconciliation or [],
    )


def run_d3_from_book(
    book: D3Book,
    *,
    d0_passed: bool,
    d1_verdict: str,
    d2_final_class: str,
    official: bool = False,
    p_mid_ref: float | None = None,
    delta_price: float | None = None,
) -> D3Result:
    blocker = check_prerequisites(
        d0_passed=d0_passed, d1_verdict=d1_verdict, d2_final_class=d2_final_class
    )
    if blocker:
        return _blocked(blocker)
    _d3_log(f"endpoint reconciliation official={official} n_trades={len(book.trades)}")
    try:
        reconciliation = reconcile_d3_endpoints(book, official=official)
    except D3AnalysisError as exc:
        return _blocked(str(exc))
    if not all(row["passed"] for row in reconciliation):
        failed = [row["metric"] for row in reconciliation if not row["passed"]]
        return _blocked(f"endpoint reconciliation failed: {failed}", reconciliation)
    _d3_log("endpoint reconciliation passed")

    p_mid = float(p_mid_ref if p_mid_ref is not None else book.trades["pnl_total_mid"].sum())
    if delta_price is None:
        delta_price = float(
            (book.trades["quantity_mid"].abs() * book.trades["pnl_per_share_cross"]).sum() - p_mid
        )
    try:
        h_f_50 = path_f_pnl_crossing(0.50, p_mid, delta_price)
        h_f_25 = path_f_pnl_crossing(0.25, p_mid, delta_price)
        h_f_p0 = path_f_pnl_crossing(0.00, p_mid, delta_price)
    except D3AnalysisError as exc:
        return _blocked(str(exc), reconciliation)
    _d3_log(
        f"Path F P&L closed form h_F_50={h_f_50:.6f} h_F_25={h_f_25:.6f} h_F_P0={h_f_p0:.6f}"
    )

    def pnl_r(h: float) -> float:
        return float(evaluate_path_at_h(book, h, path="R")["pnl"])

    def car_r(h: float) -> float:
        return float(evaluate_path_at_h(book, h, path="R")["car"])

    def car_f(h: float) -> float:
        return float(evaluate_path_at_h(book, h, path="F")["car"])

    _precompute_path_grid(book, "R", H_DET, label="Path R H_det")
    _precompute_path_grid(book, "F", H_DET, label="Path F H_det")

    try:
        _d3_log("Path R first-adverse P<=0.50M")
        h_r_50 = first_adverse_crossing(pnl_r, 0.50 * p_mid)
        _d3_log(f"h_R_50={h_r_50}")
        _d3_log("Path R first-adverse P<=0.25M")
        h_r_25 = first_adverse_crossing(pnl_r, 0.25 * p_mid)
        _d3_log(f"h_R_25={h_r_25}")
        _d3_log("Path R first-adverse P<=0")
        h_r_p0 = first_adverse_crossing(pnl_r, 0.0)
        _d3_log(f"h_R_P0={h_r_p0}")
        _d3_log("Path R first-adverse CAR<=0")
        h_r_car0 = first_adverse_crossing(car_r, 0.0)
        _d3_log(f"h_R_CAR0={h_r_car0}")
        _d3_log("Path F first-adverse CAR<=0")
        h_f_car0 = first_adverse_crossing(car_f, 0.0)
        _d3_log(f"h_F_CAR0={h_f_car0}")
    except D3AnalysisError as exc:
        return _blocked(str(exc), reconciliation)

    _d3_log("monotonicity on cached H_det")
    monotonicity = {
        "P_R": monotonic_nonincreasing(pnl_r, dollar_tolerance(p_mid)),
        "CAR_R": monotonic_nonincreasing(car_r, CAR_TOLERANCE),
    }
    _d3_log(f"monotonicity {monotonicity}")
    _d3_log("H_vis curves")

    crossings = [
        Crossing("R", "primary", "pnl_50", 0.50 * p_mid, h_r_50, "bracket_bisection"),
        Crossing("R", "primary", "pnl_25", 0.25 * p_mid, h_r_25, "bracket_bisection"),
        Crossing("R", "primary", "pnl_0", 0.0, h_r_p0, "bracket_bisection"),
        Crossing("R", "primary", "car_0", 0.0, h_r_car0, "bracket_bisection"),
        Crossing("F", "diagnostic", "pnl_50", 0.50 * p_mid, h_f_50, "closed_form"),
        Crossing("F", "diagnostic", "pnl_25", 0.25 * p_mid, h_f_25, "closed_form"),
        Crossing("F", "diagnostic", "pnl_0", 0.0, h_f_p0, "closed_form"),
        Crossing("F", "diagnostic", "car_0", 0.0, h_f_car0, "bracket_bisection"),
    ]
    curves = evaluate_paths(book, H_VIS)
    _d3_log("exact-root side snapshots")
    snapshot = side_snapshots_at_roots(
        book,
        {
            "h=0": 0.0,
            "h_R_50": h_r_50,
            "h_R_25": h_r_25,
            "h_R_P0": h_r_p0,
            "h=1": 1.0,
        },
    )
    _d3_log(f"envelope complete eval_calls={book.eval_calls} cache={len(book.eval_cache)}")
    return assemble_envelope(
        h_R_50=h_r_50,
        h_R_25=h_r_25,
        h_R_P0=h_r_p0,
        h_R_CAR0=h_r_car0,
        h_F_50=h_f_50,
        h_F_25=h_f_25,
        h_F_P0=h_f_p0,
        h_F_CAR0=h_f_car0,
        crossings=crossings,
        curves=curves,
        reconciliation=reconciliation,
        monotonicity=monotonicity,
        manifest={"d3_code_commit_sha": get_current_repo_sha(), "official": official},
        side_snapshot=snapshot,
    )


def _wing_width(direction: str, max_loss: float, net_credit: float) -> float | None:
    if direction != "short":
        return None
    return float(max_loss) + float(net_credit)


def join_paired_trades(mid_trades: pd.DataFrame, cross_trades: pd.DataFrame) -> pd.DataFrame:
    """Join mid/cross trades. ``direction`` lives on ``TRADE_KEY``, so recover it from the index."""
    mid = mid_trades.set_index(list(TRADE_KEY))
    cross = cross_trades.set_index(list(TRADE_KEY))
    if set(mid.index) != set(cross.index):
        raise D3AnalysisError("mid/cross included keys do not match")
    if "direction" in mid.columns:
        raise D3AnalysisError("direction must come from TRADE_KEY, not a leftover column")
    joined = pd.DataFrame(
        {
            "quantity_mid": mid["quantity"],
            "quantity_cross": cross["quantity"],
            "pnl_per_share_mid": mid["pnl_per_share"],
            "pnl_per_share_cross": cross["pnl_per_share"],
            "pnl_total_mid": mid["pnl_total"],
            "pnl_total_cross": cross["pnl_total"],
            "entry_cost_mid": mid["entry_cost_per_share"],
            "entry_cost_cross": cross["entry_cost_per_share"],
            "net_credit_mid": mid["net_credit_per_share"],
            "net_credit_cross": cross["net_credit_per_share"],
            "max_loss_mid": mid["max_loss_per_share"],
            "max_loss_cross": cross["max_loss_per_share"],
            "instrument_type": mid["instrument_type"],
        }
    ).reset_index()
    if "direction" not in joined.columns:
        raise D3AnalysisError("direction missing after TRADE_KEY index reset")
    widths = []
    for row in joined.itertuples(index=False):
        mid_w = _wing_width(row.direction, row.max_loss_mid, row.net_credit_mid)
        cross_w = _wing_width(row.direction, row.max_loss_cross, row.net_credit_cross)
        if mid_w is not None and cross_w is not None and not within_dollar(mid_w, cross_w):
            raise D3AnalysisError("wing_width mid vs cross mismatch")
        widths.append(mid_w)
    joined["wing_width"] = widths
    return joined


def filter_primary_date_status(
    date_status: pd.DataFrame,
    *,
    require_traded_dates: int | None = None,
) -> pd.DataFrame:
    status = date_status.copy()
    status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.date
    status = status[
        (status["trade_date"] >= PRIMARY_START) & (status["trade_date"] <= PRIMARY_END)
    ].copy()
    n_traded = int(status.loc[status["status"] == "traded", "trade_date"].nunique())
    if require_traded_dates is not None and n_traded != int(require_traded_dates):
        raise D3AnalysisError(
            f"primary traded dates={n_traded}, expected {require_traded_dates}"
        )
    return status.reset_index(drop=True)


def side_snapshots_at_roots(
    book: D3Book,
    marks: dict[str, float | None],
) -> pd.DataFrame:
    """Evaluate Path R long/short P&L at exact root values, not on H_vis."""
    rows = []
    for label, h in marks.items():
        if h is None:
            continue
        metrics = evaluate_path_at_h(book, float(h), path="R")
        rows.append(
            {
                "mark": label,
                "h": float(h),
                "pnl_long": metrics["pnl_long"],
                "pnl_short": metrics["pnl_short"],
                "pnl": metrics["pnl"],
            }
        )
    return pd.DataFrame(rows)


def load_official_book(run_dir: Path) -> D3Book:
    mid_trades, mid_legs = load_fill_primary_tables(run_dir, "mid")
    cross_trades, _cross_legs = load_fill_primary_tables(run_dir, "cross")
    joined = join_paired_trades(mid_trades, cross_trades)
    date_status = pd.read_parquet(
        run_dir / "date_status_sprint006_baseline_v1_mid.parquet",
        columns=["trade_date", "status", "reason"],
    )
    date_status = filter_primary_date_status(
        date_status, require_traded_dates=EXPECTED_TRADED_DATES
    )
    try:
        car_mid = float(
            load_accepted_primary_block(run_dir, "mid")["view_a_conditional"]["mean_cycle_car"]
        )
        car_cross = float(
            load_accepted_primary_block(run_dir, "cross")["view_a_conditional"]["mean_cycle_car"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise D3AnalysisError(f"official CAR references missing: {exc}") from exc
    return D3Book(
        trades=joined,
        legs=mid_legs,
        date_status=date_status,
        car_mid_ref=car_mid,
        car_cross_ref=car_cross,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def resolve_evidence_dir() -> Path:
    override = os.environ.get(EVIDENCE_DIR_ENV)
    if override:
        path = Path(override)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(f"C:/MomentumCVG_env/runs/sprint007_d3_{stamp}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_d3_envelope(result: D3Result, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if "h_req" in payload:
        raise D3AnalysisError("h_req is forbidden")
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return output_path


def write_d3_tables(result: D3Result, evidence_dir: Path) -> dict[str, Path]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "envelope": evidence_dir / "d3_envelope.json",
        "curves": evidence_dir / "d3_curves.csv",
        "crossings": evidence_dir / "d3_crossings.csv",
        "side_snapshot": evidence_dir / "d3_side_snapshot.csv",
    }
    write_d3_envelope(result, paths["envelope"])
    result.curves.to_csv(paths["curves"], index=False)
    pd.DataFrame([row.as_record() for row in result.crossings]).to_csv(
        paths["crossings"], index=False
    )
    result.side_snapshot.to_csv(paths["side_snapshot"], index=False)
    return paths


def write_execution_receipt(
    *,
    evidence_dir: Path,
    executed_notebook: Path,
    html_export: Path,
    d3_code_commit_sha: str | None = None,
) -> Path:
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "d3_code_commit_sha": d3_code_commit_sha or get_current_repo_sha(),
        "sprint006_execution_repo_sha": "e205b9acc5d0400aa38169de721acb7fb8268f29",
        "executed_notebook": str(executed_notebook),
        "executed_notebook_sha256": sha256_file(executed_notebook),
        "html_export": str(html_export),
        "html_export_sha256": sha256_file(html_export),
    }
    path = evidence_dir / "execution_receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def export_d3_evidence(
    result: D3Result | None = None,
    evidence_dir: Path | None = None,
    *,
    clean_notebook: Path | None = None,
    d3_code_commit_sha: str | None = None,
) -> dict[str, Path] | Path:
    """Write design artifacts. Execute the clean notebook only when asked."""
    evidence_dir = evidence_dir or resolve_evidence_dir()
    if clean_notebook is None:
        if result is None:
            raise D3AnalysisError("result is required when not executing a notebook")
        return write_d3_tables(result, evidence_dir)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    executed = evidence_dir / "d3_execution_envelope.executed.ipynb"
    html_path = evidence_dir / "d3_execution_envelope.html"
    d3_code_commit_sha = d3_code_commit_sha or get_current_repo_sha()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env[EVIDENCE_DIR_ENV] = str(evidence_dir)
    python = Path("C:/MomentumCVG_env/venv/Scripts/python.exe")
    if not python.exists():
        python = Path(sys.executable)
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
            "--ExecutePreprocessor.timeout=-1",
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
    if result is not None:
        write_d3_tables(result, evidence_dir)
    write_execution_receipt(
        evidence_dir=evidence_dir,
        executed_notebook=executed,
        html_export=html_path,
        d3_code_commit_sha=d3_code_commit_sha,
    )
    return evidence_dir


def run_d3_analysis(
    *,
    run_dir: Path | None = None,
    book: D3Book | None = None,
    d0_passed: bool | None = None,
    d1_verdict: str | None = None,
    d2_final_class: str | None = None,
    official: bool | None = None,
) -> D3Result:
    """Official artifacts are loaded only when ``book`` is omitted."""
    if book is None:
        run_dir = run_dir or OFFICIAL_RUN_DIR
        official = True if official is None else official
        _d3_log(f"official run_dir={run_dir}")
        _d3_log("D0 validation")
        d0 = run_d0_validation(run_dir=run_dir)
        _d3_log(f"D0 all_passed={d0.all_passed}")
        _d3_log("D1 analysis")
        d1 = run_d1_analysis(run_dir=run_dir, d0_result=d0)
        _d3_log(f"D1 verdict={d1.verdict}")
        _d3_log("D2B class check")
        d2b = run_d2b_analysis(run_dir=run_dir, d0_result=d0, d1_result=d1)
        _d3_log(f"D2B class={d2b.final_d3_class}")
        blocker = check_prerequisites(
            d0_passed=d0.all_passed,
            d1_verdict=d1.verdict,
            d2_final_class=d2b.final_d3_class,
        )
        if blocker:
            return _blocked(blocker)
        try:
            _d3_log("loading official book")
            book = load_official_book(run_dir)
            mid_trades, _ = load_fill_primary_tables(run_dir, "mid")
            cross_trades, _ = load_fill_primary_tables(run_dir, "cross")
            bridge = compute_bridge_terms(mid_trades, cross_trades)
            _d3_log(f"official book loaded n_trades={len(book.trades)}")
        except (D3AnalysisError, KeyError, OSError, ValueError) as exc:
            return _blocked(f"persisted artifacts cannot support D3: {exc}")
        return run_d3_from_book(
            book,
            d0_passed=True,
            d1_verdict=d1.verdict,
            d2_final_class=d2b.final_d3_class,
            official=True,
            p_mid_ref=float(bridge.p_mid),
            delta_price=float(bridge.delta_price),
        )
    return run_d3_from_book(
        book,
        d0_passed=True if d0_passed is None else d0_passed,
        d1_verdict=d1_verdict or VERDICT_CONTINUE,
        d2_final_class=d2_final_class or CLASS_EXECUTION,
        official=bool(official),
    )
