"""
Sprint 007 D2B — package-tradability diagnostic (accepted branch).

Reuses accepted D2A Laspeyres terms. Does not retune D2A, search cutoffs,
drop trades, or design D3. Selective-friction facts inform D4 later; they
do not add a sixth class.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.sprint007_artifact_validation import (
    LEG_KEY,
    OFFICIAL_RUN_DIR,
    TRADE_KEY,
)
from src.backtest.sprint007_d2_shortfall_bridge import (
    BRANCH_TRADABILITY,
    D2AClassification,
    D2AResult,
    VERDICT_BLOCKED,
    assign_final_d3_class,
    dollar_tolerance,
    load_fill_primary_tables,
    run_d2a_analysis,
)

TERCILE_CHEAP = 1
TERCILE_MID = 2
TERCILE_EXPENSIVE = 3
ZERO_CASHFLOW = "skipped_zero_cashflow"


class D2BAnalysisError(Exception):
    """Raised when the tradability diagnostic cannot be formed."""


@dataclass
class D2BResult:
    d2a: D2AResult
    final_d3_class: str
    blocked: bool
    blocker: str | None
    n_intersection: int = 0
    n_skipped_zero_cashflow: int = 0
    n_ranked: int = 0
    group_terciles: list[dict[str, Any]] = field(default_factory=list)
    book_terciles: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, Any] = field(default_factory=dict)
    expensive_share_of_delta_price: float | None = None
    mid_pnl_outside_expensive: float | None = None
    mid_margin_outside_expensive: bool | None = None
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)


def midpoint_fill(bid: float, ask: float) -> float:
    return float(bid) + 0.5 * (float(ask) - float(bid))


def package_half_spread(unit_quantity: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(unit_quantity) * (ask - bid)))


def midpoint_package_cashflow(
    unit_quantity: np.ndarray, bid: np.ndarray, ask: np.ndarray
) -> float:
    return float(np.sum(unit_quantity * (bid + 0.5 * (ask - bid))))


def package_width_to_cashflow(half_spread: float, cashflow: float) -> float | None:
    if cashflow == 0.0:
        return None
    return float(half_spread) / abs(float(cashflow))


def _numeric_legs(legs: pd.DataFrame) -> pd.DataFrame:
    frame = legs.copy()
    for column in ("bid", "ask", "unit_quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def check_shared_quotes(mid_legs: pd.DataFrame, cross_legs: pd.DataFrame) -> None:
    """bid / ask / unit_quantity must match on LEG_KEY (D0 pairing)."""
    cols = list(LEG_KEY) + ["bid", "ask", "unit_quantity"]
    left = _numeric_legs(mid_legs)[cols].sort_values(list(LEG_KEY)).reset_index(drop=True)
    right = _numeric_legs(cross_legs)[cols].sort_values(list(LEG_KEY)).reset_index(drop=True)
    if len(left) != len(right):
        raise D2BAnalysisError(
            f"leg-row count differs mid={len(left)} cross={len(right)}"
        )
    key_left = list(map(tuple, left[list(LEG_KEY)].itertuples(index=False, name=None)))
    key_right = list(map(tuple, right[list(LEG_KEY)].itertuples(index=False, name=None)))
    if key_left != key_right:
        raise D2BAnalysisError("mid/cross leg keys are not identically ordered after sort")
    for column in ("bid", "ask"):
        if not np.allclose(left[column].to_numpy(), right[column].to_numpy(), atol=1e-12, equal_nan=True):
            raise D2BAnalysisError(f"shared quote mismatch on {column}")
    if not np.array_equal(
        left["unit_quantity"].to_numpy(), right["unit_quantity"].to_numpy()
    ):
        raise D2BAnalysisError("shared unit_quantity mismatch")


def package_metrics_by_trade(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per TRADE_KEY from shared quotes."""
    if legs.empty:
        return pd.DataFrame(
            columns=[
                *TRADE_KEY,
                "package_half_spread",
                "midpoint_package_cashflow",
                "package_width_to_cashflow",
                "n_legs",
            ]
        )
    frame = _numeric_legs(legs)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(TRADE_KEY), sort=False)
    for key, grp in grouped:
        uq = grp["unit_quantity"].to_numpy(dtype=float)
        bid = grp["bid"].to_numpy(dtype=float)
        ask = grp["ask"].to_numpy(dtype=float)
        if np.isnan(uq).any() or np.isnan(bid).any() or np.isnan(ask).any():
            raise D2BAnalysisError(f"non-finite quote fields for {key}")
        half = package_half_spread(uq, bid, ask)
        cash = midpoint_package_cashflow(uq, bid, ask)
        rows.append(
            {
                "trade_date": key[0],
                "ticker": key[1],
                "direction": key[2],
                "package_half_spread": half,
                "midpoint_package_cashflow": cash,
                "package_width_to_cashflow": package_width_to_cashflow(half, cash),
                "n_legs": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def per_trade_delta_price(mid: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Intersection rows: Δ_price_i = |Q_mid| × (p_cross − p_mid)."""
    mid_i = mid.copy()
    cross_i = cross.copy()
    mid_i["_key"] = list(zip(mid_i["trade_date"], mid_i["ticker"], mid_i["direction"]))
    cross_i["_key"] = list(zip(cross_i["trade_date"], cross_i["ticker"], cross_i["direction"]))
    both = set(mid_i["_key"]) & set(cross_i["_key"])
    mid_b = mid_i.loc[mid_i["_key"].isin(both)].set_index("_key").sort_index()
    cross_b = cross_i.loc[cross_i["_key"].isin(both)].set_index("_key").sort_index()
    aligned = mid_b.index.intersection(cross_b.index)
    q_mid = mid_b.loc[aligned, "quantity"].abs().astype(float)
    p_mid = mid_b.loc[aligned, "pnl_per_share"].astype(float)
    p_cross = cross_b.loc[aligned, "pnl_per_share"].astype(float)
    inst = (
        mid_b.loc[aligned, "instrument_type"].astype(str)
        if "instrument_type" in mid_b.columns
        else pd.Series("unknown", index=aligned)
    )
    inst = inst.replace({"nan": "unknown", "None": "unknown"}).fillna("unknown")
    out = pd.DataFrame(
        {
            "trade_date": mid_b.loc[aligned].index.map(lambda k: k[0]),
            "ticker": mid_b.loc[aligned].index.map(lambda k: k[1]),
            "direction": mid_b.loc[aligned].index.map(lambda k: k[2]),
            "instrument_type": inst.to_numpy(),
            "pnl_total_mid": pd.to_numeric(mid_b.loc[aligned, "pnl_total"], errors="coerce").to_numpy(),
            "delta_price": (q_mid * (p_cross - p_mid)).to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def assign_within_group_terciles(ranked: pd.DataFrame) -> pd.DataFrame:
    """Tercile = 1 + min(2, floor(3i / n)) after a deterministic sort. 3 = expensive."""
    if ranked.empty:
        frame = ranked.copy()
        frame["tercile"] = pd.Series(dtype=int)
        return frame
    n = len(ranked)
    index = np.arange(n, dtype=int)
    tercile = 1 + np.minimum(2, (3 * index) // n)
    out = ranked.copy()
    out["tercile"] = tercile
    return out


def attach_terciles(trades: pd.DataFrame) -> pd.DataFrame:
    """Rank non-zero-cashflow rows within (direction, instrument_type)."""
    frame = trades.copy()
    frame["tercile"] = pd.NA
    ranked_mask = frame["package_width_to_cashflow"].notna()
    pieces: list[pd.DataFrame] = []
    skipped = frame.loc[~ranked_mask].copy()
    skipped["tercile"] = pd.NA
    groups = frame.loc[ranked_mask].groupby(
        ["direction", "instrument_type"], sort=True, dropna=False
    )
    for _, grp in groups:
        ordered = grp.sort_values(
            ["package_width_to_cashflow", "trade_date", "ticker"],
            kind="mergesort",
        )
        pieces.append(assign_within_group_terciles(ordered))
    ranked = pd.concat(pieces, axis=0) if pieces else frame.loc[[]].copy()
    combined = pd.concat([ranked, skipped], axis=0, ignore_index=True)
    return combined.sort_values(
        ["direction", "instrument_type", "trade_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


def _agg_slice(frame: pd.DataFrame, **labels: Any) -> dict[str, Any]:
    return {
        **labels,
        "n_trades": int(len(frame)),
        "pnl_total_mid": float(pd.to_numeric(frame["pnl_total_mid"], errors="coerce").sum())
        if len(frame)
        else 0.0,
        "delta_price": float(pd.to_numeric(frame["delta_price"], errors="coerce").sum())
        if len(frame)
        else 0.0,
    }


def summarize_terciles(trades: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ranked = trades.loc[trades["tercile"].notna()]
    skipped = trades.loc[trades["tercile"].isna()]
    group_rows: list[dict[str, Any]] = []
    if not ranked.empty:
        for (direction, inst, tercile), grp in ranked.groupby(
            ["direction", "instrument_type", "tercile"], sort=True
        ):
            group_rows.append(
                _agg_slice(
                    grp,
                    direction=str(direction),
                    instrument_type=str(inst),
                    tercile=int(tercile),
                )
            )
    book_rows = [
        _agg_slice(ranked.loc[ranked["tercile"] == t], tercile=t)
        for t in (TERCILE_CHEAP, TERCILE_MID, TERCILE_EXPENSIVE)
    ]
    skipped_row = _agg_slice(skipped, bucket=ZERO_CASHFLOW)
    return group_rows, book_rows, skipped_row


def _recon_row(metric: str, recomputed: float, reference: float) -> dict[str, Any]:
    tol = dollar_tolerance(reference)
    return {
        "metric": metric,
        "recomputed": recomputed,
        "reference": reference,
        "delta": recomputed - reference,
        "tolerance": tol,
        "passed": abs(recomputed - reference) <= tol,
    }


def run_d2b_tradability(
    *,
    mid_trades: pd.DataFrame,
    cross_trades: pd.DataFrame,
    mid_legs: pd.DataFrame,
    cross_legs: pd.DataFrame,
    d2a_delta_price: float,
    d2a_p_mid: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    check_shared_quotes(mid_legs, cross_legs)
    packages = package_metrics_by_trade(mid_legs)
    economics = per_trade_delta_price(mid_trades, cross_trades)
    trades = economics.merge(packages, on=list(TRADE_KEY), how="left", validate="one_to_one")
    if trades["package_half_spread"].isna().any():
        raise D2BAnalysisError("included trade missing package legs")
    trades = attach_terciles(trades)
    groups, book, skipped = summarize_terciles(trades)
    recon = [
        _recon_row("delta_price_per_trade", float(trades["delta_price"].sum()), float(d2a_delta_price)),
        _recon_row("p_mid_per_trade", float(trades["pnl_total_mid"].sum()), float(d2a_p_mid)),
        _recon_row(
            "delta_price_terciles_plus_skipped",
            float(sum(row["delta_price"] for row in book) + skipped["delta_price"]),
            float(d2a_delta_price),
        ),
    ]
    if not all(row["passed"] for row in recon):
        failed = [row["metric"] for row in recon if not row["passed"]]
        raise D2BAnalysisError(f"tradability reconciliation failed: {failed}")
    return trades, groups, book, skipped, recon


def run_d2b_analysis(
    *,
    d2a_result: D2AResult | None = None,
    run_dir: Path | None = None,
    d0_result: Any = None,
    d1_result: Any = None,
) -> D2BResult:
    """Run accepted D2A (if needed), then the package-tradability follow-up."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    if d2a_result is None:
        d2a_result = run_d2a_analysis(
            run_dir=run_dir, d0_result=d0_result, d1_result=d1_result
        )
    if d2a_result.blocked or d2a_result.bridge is None or d2a_result.classification is None:
        return D2BResult(
            d2a=d2a_result,
            final_d3_class=VERDICT_BLOCKED,
            blocked=True,
            blocker=d2a_result.blocker or "D2A blocked",
        )
    classification: D2AClassification = d2a_result.classification
    if classification.d2b_branch != BRANCH_TRADABILITY:
        return D2BResult(
            d2a=d2a_result,
            final_d3_class=assign_final_d3_class(classification, blocked=False),
            blocked=False,
            blocker=f"D2B tradability not selected (branch={classification.d2b_branch})",
        )

    try:
        mid_trades, mid_legs = load_fill_primary_tables(run_dir, "mid")
        cross_trades, cross_legs = load_fill_primary_tables(run_dir, "cross")
        trades, groups, book, skipped, recon = run_d2b_tradability(
            mid_trades=mid_trades,
            cross_trades=cross_trades,
            mid_legs=mid_legs,
            cross_legs=cross_legs,
            d2a_delta_price=d2a_result.bridge.delta_price,
            d2a_p_mid=d2a_result.bridge.p_mid,
        )
    except Exception as exc:
        return D2BResult(
            d2a=d2a_result,
            final_d3_class=assign_final_d3_class(classification, blocked=True),
            blocked=True,
            blocker=f"{type(exc).__name__}: {exc}",
        )

    expensive = next(row for row in book if row["tercile"] == TERCILE_EXPENSIVE)
    outside = trades.loc[trades["tercile"].isin([TERCILE_CHEAP, TERCILE_MID])]
    total_dp = float(d2a_result.bridge.delta_price)
    expensive_share = (
        abs(expensive["delta_price"]) / abs(total_dp) if total_dp != 0.0 else None
    )
    mid_outside = float(outside["pnl_total_mid"].sum()) if len(outside) else 0.0
    return D2BResult(
        d2a=d2a_result,
        final_d3_class=assign_final_d3_class(classification, blocked=False),
        blocked=False,
        blocker=None,
        n_intersection=int(len(trades)),
        n_skipped_zero_cashflow=int(skipped["n_trades"]),
        n_ranked=int(len(trades) - skipped["n_trades"]),
        group_terciles=groups,
        book_terciles=book,
        skipped=skipped,
        expensive_share_of_delta_price=expensive_share,
        mid_pnl_outside_expensive=mid_outside,
        mid_margin_outside_expensive=mid_outside > 0.0,
        trades=trades,
        reconciliation=recon,
    )


def write_d2b_tables(result: D2BResult, evidence_dir: Path) -> dict[str, Path]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "final_d3_class": result.final_d3_class,
        "blocked": result.blocked,
        "blocker": result.blocker,
        "d2a_provisional_class": (
            result.d2a.classification.provisional_d3_class
            if result.d2a.classification
            else None
        ),
        "d2b_branch": (
            result.d2a.classification.d2b_branch if result.d2a.classification else None
        ),
        "n_intersection": result.n_intersection,
        "n_skipped_zero_cashflow": result.n_skipped_zero_cashflow,
        "n_ranked": result.n_ranked,
        "group_terciles": result.group_terciles,
        "book_terciles": result.book_terciles,
        "skipped": result.skipped,
        "expensive_share_of_delta_price": result.expensive_share_of_delta_price,
        "mid_pnl_outside_expensive": result.mid_pnl_outside_expensive,
        "mid_margin_outside_expensive": result.mid_margin_outside_expensive,
        "reconciliation": result.reconciliation,
        "d2b_executed": True,
        "threshold_search": False,
        "filtered_strategy": False,
    }
    paths = {
        "summary": evidence_dir / "d2b_tradability.json",
        "groups": evidence_dir / "d2b_group_terciles.csv",
        "book": evidence_dir / "d2b_book_terciles.csv",
    }
    paths["summary"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    pd.DataFrame(result.group_terciles).to_csv(paths["groups"], index=False)
    pd.DataFrame(result.book_terciles).to_csv(paths["book"], index=False)
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return str(value)
    return value


__all__ = [
    "D2BResult",
    "assign_within_group_terciles",
    "package_half_spread",
    "package_metrics_by_trade",
    "package_width_to_cashflow",
    "midpoint_package_cashflow",
    "per_trade_delta_price",
    "run_d2b_analysis",
    "run_d2b_tradability",
]
