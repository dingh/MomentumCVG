"""
Sprint 007 D2A — midpoint-to-cross dollar-P&L bridge.

Read-only post-pass over accepted Sprint 006 paired artifacts. Computes the
frozen Laspeyres (Q_mid) bridge, side/year dollar slices, one order-sensitivity
statistic, the D2B branch recommendation, and a provisional D3 class.

D2B diagnostics, D3 design, and SurfaceRunner are out of scope.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.backtest.sprint007_artifact_validation import (
    D0ValidationResult,
    LEG_LOG_COLUMNS,
    OFFICIAL_RUN_DIR,
    TRADE_KEY,
    TRADE_LOG_COLUMNS,
    get_current_repo_sha,
    run_d0_validation,
)
from src.backtest.sprint007_d1_gross_margin import (
    DATE_STATUS_COLUMNS,
    DATE_SUMMARY_COLUMNS,
    VERDICT_CONTINUE,
    D1Result,
    ReconciliationRow,
    run_d1_analysis,
    select_included_traded_rows,
)
from src.backtest.surface_decision_report import (
    PRIMARY_END,
    PRIMARY_START,
    assert_report_preconditions,
    filter_to_window,
)

TRADE_COLUMNS = TRADE_LOG_COLUMNS + ("pnl_per_share",)
LEG_COLUMNS = LEG_LOG_COLUMNS

PNL_ABS_TOLERANCE = 0.01
PNL_REL_TOLERANCE = 1e-9
MATERIALITY_SHARE = 0.25
SIDE_CONCENTRATION = 0.70
SET_ABS_TOLERANCE = 0.01

VERDICT_BLOCKED = "D2_BLOCKED"
CLASS_MIXED = "D3_MIXED_MECHANISM"
CLASS_SIZING = "D3_SIZING_AWARE"
CLASS_STRUCTURE = "D3_STRUCTURE_CONDITIONED"
CLASS_EXECUTION = "D3_EXECUTION_FOCUSED"

BRANCH_NONE = "none"
BRANCH_SIZING = "sizing_chain"
BRANCH_BODY_WING = "body_wing"
BRANCH_TRADABILITY = "package_tradability"

EVIDENCE_DIR_ENV = "SPRINT007_D2A_EVIDENCE_DIR"

SHORT_IRON_TYPES = frozenset({"iron_fly"})


class D2AnalysisError(Exception):
    """Raised when a D2A precondition makes the bridge uninterpretable."""


def dollar_tolerance(reference: float) -> float:
    return max(PNL_ABS_TOLERANCE, PNL_REL_TOLERANCE * abs(float(reference)))


def within_dollar(value: float, reference: float) -> bool:
    return abs(float(value) - float(reference)) <= dollar_tolerance(reference)


def residual_tolerance(p_mid: float, p_cross: float) -> float:
    return max(PNL_ABS_TOLERANCE, PNL_REL_TOLERANCE * (abs(float(p_mid)) + abs(float(p_cross))))


@dataclass
class IntegrityCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class BridgeTerms:
    p_mid: float
    p_cross: float
    p_cross_at_q_mid: float
    gap: float
    delta_price: float
    delta_size: float
    delta_set: float
    residual: float
    n_intersection: int
    n_mid_only: int
    n_cross_only: int
    interaction: float
    delta_price_paasche: float
    delta_size_dual: float
    s_order: float

    @property
    def unmatched_keys(self) -> int:
        return int(self.n_mid_only + self.n_cross_only)


@dataclass
class D2AClassification:
    price_material: bool
    size_material: bool
    dominant: str
    order_sensitive: bool
    concentrating_side: str | None
    structure: bool
    d2b_branch: str
    provisional_d3_class: str


@dataclass
class D2AResult:
    verdict: str
    blocked: bool
    blocker: str | None
    bridge: BridgeTerms | None
    classification: D2AClassification | None
    integrity: list[IntegrityCheck] = field(default_factory=list)
    side_bridge: list[dict[str, Any]] = field(default_factory=list)
    yearly_bridge: list[dict[str, Any]] = field(default_factory=list)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


def _artifact_path(run_dir: Path, role: str, fill: str) -> Path:
    return run_dir / f"{role}_sprint006_baseline_v1_{fill}.parquet"


def _read_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pq.read_table(path, columns=list(columns)).to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    if "expiry_date" in frame.columns:
        frame["expiry_date"] = pd.to_datetime(frame["expiry_date"]).dt.date
    return frame.reset_index(drop=True)


def _trade_index(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    indexed["_key"] = list(zip(indexed["trade_date"], indexed["ticker"], indexed["direction"]))
    return indexed


def load_fill_primary_tables(run_dir: Path, fill_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window-filter one fill and return included trades plus included legs."""
    date_status = _read_columns(_artifact_path(run_dir, "date_status", fill_label), DATE_STATUS_COLUMNS)
    date_summary = _read_columns(_artifact_path(run_dir, "date_summary", fill_label), DATE_SUMMARY_COLUMNS)
    trade_log = _read_columns(_artifact_path(run_dir, "trade_log", fill_label), TRADE_COLUMNS)
    labels = {str(value) for value in trade_log["fill_label"].dropna().unique()}
    unexpected = labels - {fill_label}
    if unexpected:
        raise D2AnalysisError(f"{fill_label} trade log has unexpected fill labels: {sorted(unexpected)}")
    status, summary, log = filter_to_window(
        date_status, date_summary, trade_log, PRIMARY_START, PRIMARY_END
    )
    assert_report_preconditions(status, summary, log)
    included = select_included_traded_rows(log, status)
    for column in ("pnl_total", "pnl_per_share", "quantity"):
        included[column] = pd.to_numeric(included[column], errors="coerce")

    legs = _read_columns(_artifact_path(run_dir, "leg_log", fill_label), LEG_COLUMNS)
    status_w, _, legs_w = filter_to_window(
        status, date_summary, legs, PRIMARY_START, PRIMARY_END
    )
    traded = set(status_w.loc[status_w["status"] == "traded", "trade_date"].tolist())
    included_keys = set(
        zip(included["trade_date"], included["ticker"], included["direction"])
    )
    legs_w = legs_w.copy()
    legs_w["_key"] = list(zip(legs_w["trade_date"], legs_w["ticker"], legs_w["direction"]))
    legs_included = legs_w.loc[
        (legs_w["included_in_portfolio"] == True)  # noqa: E712
        & legs_w["trade_date"].isin(traded)
        & legs_w["_key"].isin(included_keys)
    ].copy()
    for column in ("pnl_total_leg", "pnl_per_unit"):
        legs_included[column] = pd.to_numeric(legs_included[column], errors="coerce")
    return included, legs_included.drop(columns=["_key"])


def load_accepted_primary_block(run_dir: Path, fill_label: str) -> dict[str, Any]:
    report = json.loads((run_dir / "decision_report.json").read_text(encoding="utf-8"))
    return report["by_fill"][fill_label]["primary"]


def report_total_pnl(primary_block: dict[str, Any]) -> float:
    long_short = primary_block["long_short"]
    return float(long_short["long"]["pnl_total"]) + float(long_short["short"]["pnl_total"])


def check_leg_to_trade(trades: pd.DataFrame, legs: pd.DataFrame) -> list[IntegrityCheck]:
    """Per-trade sum(pnl_total_leg)==pnl_total and sum(pnl_per_unit)==pnl_per_share."""
    checks: list[IntegrityCheck] = []
    if trades.empty:
        return [IntegrityCheck("leg_to_trade", True, "no included trades")]
    keyed_legs = _trade_index(legs)
    grouped = keyed_legs.groupby("_key", sort=False)
    for row in trades.itertuples(index=False):
        key = (row.trade_date, row.ticker, row.direction)
        if key not in grouped.groups:
            checks.append(
                IntegrityCheck("leg_to_trade", False, f"missing legs for {key}")
            )
            continue
        matched = grouped.get_group(key)
        pnl_leg = float(matched["pnl_total_leg"].sum())
        pnl_unit = float(matched["pnl_per_unit"].sum())
        if not within_dollar(pnl_leg, float(row.pnl_total)):
            checks.append(
                IntegrityCheck(
                    "leg_to_trade",
                    False,
                    f"{key} Σ pnl_total_leg {pnl_leg} != pnl_total {row.pnl_total}",
                )
            )
        if not within_dollar(pnl_unit, float(row.pnl_per_share)):
            checks.append(
                IntegrityCheck(
                    "leg_to_trade",
                    False,
                    f"{key} Σ pnl_per_unit {pnl_unit} != pnl_per_share {row.pnl_per_share}",
                )
            )
    if not checks:
        checks.append(IntegrityCheck("leg_to_trade", True, f"n_trades={len(trades)}"))
    return checks


def compute_bridge_terms(mid: pd.DataFrame, cross: pd.DataFrame) -> BridgeTerms:
    """Official Laspeyres bridge on included trade rows. Unmatched keys go to Δ_set."""
    mid_i = _trade_index(mid)
    cross_i = _trade_index(cross)
    mid_keys = set(mid_i["_key"])
    cross_keys = set(cross_i["_key"])
    both = mid_keys & cross_keys
    mid_only = mid_keys - cross_keys
    cross_only = cross_keys - mid_keys

    mid_both = mid_i.loc[mid_i["_key"].isin(both)].set_index("_key").sort_index()
    cross_both = cross_i.loc[cross_i["_key"].isin(both)].set_index("_key").sort_index()
    aligned = mid_both.index.intersection(cross_both.index)
    mid_both = mid_both.loc[aligned]
    cross_both = cross_both.loc[aligned]

    q_mid = mid_both["quantity"].abs().astype(float)
    q_cross = cross_both["quantity"].abs().astype(float)
    p_mid = mid_both["pnl_per_share"].astype(float)
    p_cross = cross_both["pnl_per_share"].astype(float)

    p_mid_total = float(pd.to_numeric(mid["pnl_total"], errors="coerce").sum())
    p_cross_total = float(pd.to_numeric(cross["pnl_total"], errors="coerce").sum())
    p_cross_at_q_mid = float((q_mid * p_cross).sum())
    p_mid_at_q_mid = float((q_mid * p_mid).sum())
    p_cross_at_q_cross = float((q_cross * p_cross).sum())
    p_mid_at_q_cross = float((q_cross * p_mid).sum())

    delta_price = p_cross_at_q_mid - p_mid_at_q_mid
    delta_size = p_cross_at_q_cross - p_cross_at_q_mid
    delta_set = float(
        pd.to_numeric(cross_i.loc[cross_i["_key"].isin(cross_only), "pnl_total"], errors="coerce").sum()
    ) - float(
        pd.to_numeric(mid_i.loc[mid_i["_key"].isin(mid_only), "pnl_total"], errors="coerce").sum()
    )
    gap = p_cross_total - p_mid_total
    residual = gap - (delta_price + delta_size + delta_set)
    interaction = float(((p_cross - p_mid) * (q_cross - q_mid)).sum())
    delta_price_paasche = p_cross_at_q_cross - p_mid_at_q_cross
    delta_size_dual = p_mid_at_q_cross - p_mid_at_q_mid
    s_order = abs(interaction) / abs(gap) if gap != 0.0 else 0.0
    return BridgeTerms(
        p_mid=p_mid_total,
        p_cross=p_cross_total,
        p_cross_at_q_mid=p_cross_at_q_mid,
        gap=gap,
        delta_price=delta_price,
        delta_size=delta_size,
        delta_set=delta_set,
        residual=residual,
        n_intersection=len(aligned),
        n_mid_only=len(mid_only),
        n_cross_only=len(cross_only),
        interaction=interaction,
        delta_price_paasche=delta_price_paasche,
        delta_size_dual=delta_size_dual,
        s_order=s_order,
    )


def is_material(delta: float, gap: float) -> bool:
    if gap == 0.0:
        return False
    return abs(float(delta)) / abs(float(gap)) >= MATERIALITY_SHARE


def dominant_term(delta_price: float, delta_size: float) -> str:
    if abs(delta_price) > abs(delta_size):
        return "price"
    if abs(delta_size) > abs(delta_price):
        return "size"
    return "tie"


def concentrating_price_side(side_delta_price: dict[str, float], delta_price: float) -> str | None:
    if delta_price == 0.0:
        return None
    hits = [
        side
        for side, value in side_delta_price.items()
        if abs(float(value)) / abs(float(delta_price)) >= SIDE_CONCENTRATION
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _short_is_iron_fly(short_trades: pd.DataFrame) -> bool:
    if short_trades.empty:
        return False
    if "instrument_type" not in short_trades.columns:
        return True
    types = {str(v) for v in short_trades["instrument_type"].dropna().unique()}
    return bool(types & SHORT_IRON_TYPES)


def classify_d2a(
    bridge: BridgeTerms,
    *,
    blocked: bool,
    side_delta_price: dict[str, float],
    short_is_iron_fly: bool,
) -> D2AClassification:
    price_material = is_material(bridge.delta_price, bridge.gap)
    size_material = is_material(bridge.delta_size, bridge.gap)
    official_dom = dominant_term(bridge.delta_price, bridge.delta_size)
    dual_price_mat = is_material(bridge.delta_price_paasche, bridge.gap)
    dual_size_mat = is_material(bridge.delta_size_dual, bridge.gap)
    dual_dom = dominant_term(bridge.delta_price_paasche, bridge.delta_size_dual)
    order_sensitive = (
        dual_price_mat != price_material
        or dual_size_mat != size_material
        or dual_dom != official_dom
    )
    side = concentrating_price_side(side_delta_price, bridge.delta_price)
    structure = side is not None

    if blocked:
        branch = BRANCH_NONE
        provisional = VERDICT_BLOCKED
    else:
        if size_material:
            branch = BRANCH_SIZING
        elif official_dom == "price" and side == "short" and short_is_iron_fly:
            branch = BRANCH_BODY_WING
        elif official_dom == "price":
            branch = BRANCH_TRADABILITY
        else:
            branch = BRANCH_NONE

        mixed = (price_material and size_material) or order_sensitive
        if mixed:
            provisional = CLASS_MIXED
        elif size_material:
            provisional = CLASS_SIZING
        elif structure:
            provisional = CLASS_STRUCTURE
        else:
            provisional = CLASS_EXECUTION

    return D2AClassification(
        price_material=price_material,
        size_material=size_material,
        dominant=official_dom,
        order_sensitive=order_sensitive,
        concentrating_side=side,
        structure=structure,
        d2b_branch=branch,
        provisional_d3_class=provisional,
    )


def assign_final_d3_class(
    classification: D2AClassification | None,
    *,
    blocked: bool,
) -> str:
    """Frozen D3 precedence after D2B. Branch 3 does not set ``structure``."""
    if blocked or classification is None:
        return VERDICT_BLOCKED
    mixed = (
        classification.price_material and classification.size_material
    ) or classification.order_sensitive
    if mixed:
        return CLASS_MIXED
    if classification.size_material:
        return CLASS_SIZING
    if classification.structure:
        return CLASS_STRUCTURE
    return CLASS_EXECUTION


def _slice_bridge(mid: pd.DataFrame, cross: pd.DataFrame, label: str) -> dict[str, Any]:
    terms = compute_bridge_terms(mid, cross)
    return {
        "slice": label,
        "p_mid": terms.p_mid,
        "p_cross": terms.p_cross,
        "gap": terms.gap,
        "delta_price": terms.delta_price,
        "delta_size": terms.delta_size,
        "delta_set": terms.delta_set,
        "residual": terms.residual,
        "n_intersection": terms.n_intersection,
        "n_mid_only": terms.n_mid_only,
        "n_cross_only": terms.n_cross_only,
    }


def side_bridge_rows(mid: pd.DataFrame, cross: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for side in ("long", "short"):
        rows.append(
            _slice_bridge(
                mid.loc[mid["direction"] == side],
                cross.loc[cross["direction"] == side],
                side,
            )
        )
    return rows


def yearly_bridge_rows(mid: pd.DataFrame, cross: pd.DataFrame) -> list[dict[str, Any]]:
    years = sorted(
        {
            d.year
            for d in list(mid["trade_date"]) + list(cross["trade_date"])
        }
    )
    rows = []
    for year in years:
        mid_y = mid.loc[mid["trade_date"].map(lambda d: d.year) == year]
        cross_y = cross.loc[cross["trade_date"].map(lambda d: d.year) == year]
        rows.append(_slice_bridge(mid_y, cross_y, str(year)))
    return rows


def format_d2a_conclusion(result: D2AResult) -> str:
    if result.blocked or result.bridge is None or result.classification is None:
        return (
            f"Supports: none — D2A is blocked ({result.blocker}). "
            f"Weakens: mechanism attribution. "
            f"Unknown: the entire mid→cross gap until the calculation/evidence problem is fixed. "
            f"Provisional D3 class: {VERDICT_BLOCKED}; D2B branch: {BRANCH_NONE}."
        )
    bridge = result.bridge
    cls = result.classification
    gap = bridge.gap
    price_share = abs(bridge.delta_price) / abs(gap) if gap else 0.0
    size_share = abs(bridge.delta_size) / abs(gap) if gap else 0.0
    supports = []
    if cls.price_material:
        supports.append(f"direct concession (Δ_price share {price_share:.1%} of G)")
    if cls.size_material:
        supports.append(f"sizing/financing (Δ_size share {size_share:.1%} of G)")
    if cls.structure:
        supports.append(f"side concentration ({cls.concentrating_side} ≥ 70% of Δ_price)")
    if not supports:
        supports.append("no material isolated mechanism under the frozen thresholds")
    weakens = []
    if bridge.unmatched_keys == 0:
        weakens.append("trade-set / opportunity difference (Δ_set keys = 0)")
    if cls.size_material:
        weakens.append("a pure per-share execution tax")
    if not cls.price_material:
        weakens.append("direct quote concession as the sole driver")
    unknown = (
        "package fill probability; whether any required quality is attainable; "
        "counterfactual structures; D2B not run"
    )
    return (
        f"Supports: {'; '.join(supports)}. "
        f"Weakens: {'; '.join(weakens) if weakens else 'none from the identity alone'}. "
        f"Unknown: {unknown}. "
        f"Provisional D3 class: {cls.provisional_d3_class}; D2B branch: {cls.d2b_branch}. "
        f"Final class is assigned only after D2B."
    )


def _blocked_result(
    *,
    blocker: str,
    integrity: list[IntegrityCheck],
    d0_result: D0ValidationResult | None,
    d1_result: D1Result | None,
    run_dir: Path,
    bridge: BridgeTerms | None = None,
    side_bridge: list[dict[str, Any]] | None = None,
    yearly_bridge: list[dict[str, Any]] | None = None,
    reconciliation: list[dict[str, Any]] | None = None,
) -> D2AResult:
    result = D2AResult(
        verdict=VERDICT_BLOCKED,
        blocked=True,
        blocker=blocker,
        bridge=bridge,
        classification=None,
        integrity=integrity,
        side_bridge=side_bridge or [],
        yearly_bridge=yearly_bridge or [],
        reconciliation=reconciliation or [],
    )
    if bridge is not None:
        result.classification = classify_d2a(
            bridge,
            blocked=True,
            side_delta_price={
                row["slice"]: float(row["delta_price"]) for row in result.side_bridge
            },
            short_is_iron_fly=False,
        )
    result.conclusion = format_d2a_conclusion(result)
    result.manifest = _build_manifest(
        result=result,
        run_dir=run_dir,
        d0_result=d0_result,
        d1_result=d1_result,
    )
    return result


def run_d2a_analysis(
    *,
    run_dir: Path | None = None,
    d0_result: D0ValidationResult | None = None,
    d1_result: D1Result | None = None,
    d2_code_commit_sha: str | None = None,
) -> D2AResult:
    """D0 → D1 continue → paired bridge. Stops before any D2B diagnostic."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    d2_code_commit_sha = d2_code_commit_sha or get_current_repo_sha()
    integrity: list[IntegrityCheck] = []

    if d0_result is None:
        d0_result = run_d0_validation(run_dir=run_dir)
    if not d0_result.all_passed:
        failed = [g.gate_id for g in d0_result.gates if not g.passed]
        return _blocked_result(
            blocker=f"D0 prerequisite failed: {failed or ['no D0 gates recorded']}",
            integrity=integrity,
            d0_result=d0_result,
            d1_result=None,
            run_dir=run_dir,
        )

    if d1_result is None:
        d1_result = run_d1_analysis(run_dir=run_dir, d0_result=d0_result)
    if d1_result.verdict != VERDICT_CONTINUE:
        return _blocked_result(
            blocker=f"D1 verdict is {d1_result.verdict}, not {VERDICT_CONTINUE}",
            integrity=integrity,
            d0_result=d0_result,
            d1_result=d1_result,
            run_dir=run_dir,
        )

    try:
        mid_trades, mid_legs = load_fill_primary_tables(run_dir, "mid")
        cross_trades, cross_legs = load_fill_primary_tables(run_dir, "cross")
        mid_block = load_accepted_primary_block(run_dir, "mid")
        cross_block = load_accepted_primary_block(run_dir, "cross")
    except Exception as exc:
        return _blocked_result(
            blocker=f"{type(exc).__name__}: {exc}",
            integrity=integrity,
            d0_result=d0_result,
            d1_result=d1_result,
            run_dir=run_dir,
        )

    integrity.extend(check_leg_to_trade(mid_trades, mid_legs))
    integrity.extend(check_leg_to_trade(cross_trades, cross_legs))
    if any(not item.passed for item in integrity):
        return _blocked_result(
            blocker="leg-to-trade reconciliation failed",
            integrity=integrity,
            d0_result=d0_result,
            d1_result=d1_result,
            run_dir=run_dir,
        )

    bridge = compute_bridge_terms(mid_trades, cross_trades)
    side_rows = side_bridge_rows(mid_trades, cross_trades)
    year_rows = yearly_bridge_rows(mid_trades, cross_trades)

    ref_mid = report_total_pnl(mid_block)
    ref_cross = report_total_pnl(cross_block)
    recon = [
        ReconciliationRow(
            metric="P_mid",
            recomputed=bridge.p_mid,
            reference=ref_mid,
            delta=bridge.p_mid - ref_mid,
            tolerance=dollar_tolerance(ref_mid),
            passed=within_dollar(bridge.p_mid, ref_mid),
        ),
        ReconciliationRow(
            metric="P_cross",
            recomputed=bridge.p_cross,
            reference=ref_cross,
            delta=bridge.p_cross - ref_cross,
            tolerance=dollar_tolerance(ref_cross),
            passed=within_dollar(bridge.p_cross, ref_cross),
        ),
        ReconciliationRow(
            metric="residual",
            recomputed=bridge.residual,
            reference=0.0,
            delta=bridge.residual,
            tolerance=residual_tolerance(bridge.p_mid, bridge.p_cross),
            passed=abs(bridge.residual) <= residual_tolerance(bridge.p_mid, bridge.p_cross),
        ),
        ReconciliationRow(
            metric="delta_set_keys",
            recomputed=float(bridge.unmatched_keys),
            reference=0.0,
            delta=float(bridge.unmatched_keys),
            tolerance=0.0,
            passed=bridge.unmatched_keys == 0,
        ),
        ReconciliationRow(
            metric="delta_set_dollars",
            recomputed=bridge.delta_set,
            reference=0.0,
            delta=bridge.delta_set,
            tolerance=SET_ABS_TOLERANCE,
            passed=abs(bridge.delta_set) <= SET_ABS_TOLERANCE and bridge.unmatched_keys == 0,
        ),
        ReconciliationRow(
            metric="interaction_identity",
            recomputed=abs(bridge.interaction),
            reference=abs(bridge.delta_price - bridge.delta_price_paasche),
            delta=abs(bridge.interaction) - abs(bridge.delta_price - bridge.delta_price_paasche),
            tolerance=dollar_tolerance(bridge.interaction),
            passed=within_dollar(
                abs(bridge.interaction),
                abs(bridge.delta_price - bridge.delta_price_paasche),
            ),
        ),
    ]
    recon_records = [
        {
            "metric": row.metric,
            "recomputed": row.recomputed,
            "reference": row.reference,
            "delta": row.delta,
            "tolerance": row.tolerance,
            "passed": row.passed,
        }
        for row in recon
    ]
    if not all(row.passed for row in recon):
        failed = [row.metric for row in recon if not row.passed]
        return _blocked_result(
            blocker=f"bridge reconciliation failed: {failed}",
            integrity=integrity,
            d0_result=d0_result,
            d1_result=d1_result,
            run_dir=run_dir,
            bridge=bridge,
            side_bridge=side_rows,
            yearly_bridge=year_rows,
            reconciliation=recon_records,
        )

    short_trades = mid_trades.loc[mid_trades["direction"] == "short"]
    classification = classify_d2a(
        bridge,
        blocked=False,
        side_delta_price={row["slice"]: float(row["delta_price"]) for row in side_rows},
        short_is_iron_fly=_short_is_iron_fly(short_trades),
    )
    result = D2AResult(
        verdict=classification.provisional_d3_class,
        blocked=False,
        blocker=None,
        bridge=bridge,
        classification=classification,
        integrity=integrity,
        side_bridge=side_rows,
        yearly_bridge=year_rows,
        reconciliation=recon_records,
    )
    result.conclusion = format_d2a_conclusion(result)
    result.manifest = _build_manifest(
        result=result,
        run_dir=run_dir,
        d0_result=d0_result,
        d1_result=d1_result,
        d2_code_commit_sha=d2_code_commit_sha,
    )
    return result


def _bridge_record(bridge: BridgeTerms | None) -> dict[str, Any] | None:
    if bridge is None:
        return None
    return {
        "p_mid": bridge.p_mid,
        "p_cross": bridge.p_cross,
        "p_cross_at_q_mid": bridge.p_cross_at_q_mid,
        "gap": bridge.gap,
        "delta_price": bridge.delta_price,
        "delta_size": bridge.delta_size,
        "delta_set": bridge.delta_set,
        "residual": bridge.residual,
        "n_intersection": bridge.n_intersection,
        "n_mid_only": bridge.n_mid_only,
        "n_cross_only": bridge.n_cross_only,
        "interaction": bridge.interaction,
        "delta_price_paasche": bridge.delta_price_paasche,
        "delta_size_dual": bridge.delta_size_dual,
        "s_order": bridge.s_order,
    }


def _class_record(classification: D2AClassification | None) -> dict[str, Any] | None:
    if classification is None:
        return None
    return {
        "price_material": classification.price_material,
        "size_material": classification.size_material,
        "dominant": classification.dominant,
        "order_sensitive": classification.order_sensitive,
        "concentrating_side": classification.concentrating_side,
        "structure": classification.structure,
        "d2b_branch": classification.d2b_branch,
        "provisional_d3_class": classification.provisional_d3_class,
    }


def _build_manifest(
    *,
    result: D2AResult,
    run_dir: Path,
    d0_result: D0ValidationResult | None,
    d1_result: D1Result | None,
    d2_code_commit_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_run_dir": str(run_dir),
        "window_start": str(PRIMARY_START),
        "window_end": str(PRIMARY_END),
        "d2_code_commit_sha": d2_code_commit_sha or get_current_repo_sha(),
        "verdict": result.verdict,
        "blocked": result.blocked,
        "blocker": result.blocker,
        "scope": "D2A",
        "d2b_executed": False,
        "final_d3_class": None,
        "d0_prerequisite": (
            {"verdict": d0_result.verdict, "all_passed": d0_result.all_passed}
            if d0_result is not None
            else None
        ),
        "d1_prerequisite": (
            {"verdict": d1_result.verdict} if d1_result is not None else None
        ),
        "bridge": _bridge_record(result.bridge),
        "classification": _class_record(result.classification),
        "reconciliation": result.reconciliation,
        "conclusion": result.conclusion,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return str(value)
    return value


def resolve_evidence_dir() -> Path:
    override = os.environ.get(EVIDENCE_DIR_ENV)
    if override:
        path = Path(override)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(f"C:/MomentumCVG_env/runs/sprint007_d2a_{stamp}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_d2a_bridge(result: D2AResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": result.verdict,
        "blocked": result.blocked,
        "blocker": result.blocker,
        "bridge": _bridge_record(result.bridge),
        "classification": _class_record(result.classification),
        "reconciliation": result.reconciliation,
        "conclusion": result.conclusion,
        "d2b_executed": False,
        "final_d3_class": None,
    }
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return output_path


def write_d2a_tables(result: D2AResult, evidence_dir: Path) -> dict[str, Path]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "side": evidence_dir / "d2a_side_bridge.csv",
        "year": evidence_dir / "d2a_yearly_bridge.csv",
        "manifest": evidence_dir / "d2a_bridge.json",
    }
    pd.DataFrame(result.side_bridge).to_csv(paths["side"], index=False)
    pd.DataFrame(result.yearly_bridge).to_csv(paths["year"], index=False)
    write_d2a_bridge(result, paths["manifest"])
    (evidence_dir / "d2a_manifest.json").write_text(
        json.dumps(_jsonable(result.manifest), indent=2), encoding="utf-8"
    )
    return paths


__all__ = [
    "BRANCH_BODY_WING",
    "BRANCH_NONE",
    "BRANCH_SIZING",
    "BRANCH_TRADABILITY",
    "CLASS_EXECUTION",
    "CLASS_MIXED",
    "CLASS_SIZING",
    "CLASS_STRUCTURE",
    "D2AResult",
    "VERDICT_BLOCKED",
    "assign_final_d3_class",
    "check_leg_to_trade",
    "classify_d2a",
    "compute_bridge_terms",
    "format_d2a_conclusion",
    "run_d2a_analysis",
    "side_bridge_rows",
    "write_d2a_tables",
    "yearly_bridge_rows",
]
