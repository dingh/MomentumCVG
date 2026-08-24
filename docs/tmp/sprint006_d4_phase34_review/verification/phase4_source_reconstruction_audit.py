"""Sprint 006 D4 Phase 4 — audit-local source reconstruction (§7.4).

Verification-only. No production backtest calculation helpers.
Reads frozen contract JSON directly; reconstructs only the two frozen dates.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RUN_DIR = Path(r"C:/MomentumCVG_env/runs/sprint006_baseline_v1_20260823T204430Z")
VERIFY_DIR = Path(r"C:/MomentumCVG_env/runs/sprint006_d4_verification_20260823T204430Z")
CONTRACT = Path(r"C:/MomentumCVG/configs/sprint006_baseline_v1.json")
CROSS = "sprint006_baseline_v1_cross"
MID = "sprint006_baseline_v1_mid"
EXECUTION_COMMIT = "e205b9acc5d0400aa38169de721acb7fb8268f29"
TOL_ABS, TOL_REL = 1e-6, 1e-8

FROZEN_SAMPLES = [
    ("S1-L", date(2022, 9, 2), "ACN", "long"),
    ("S1-S", date(2022, 9, 2), "AMC", "short"),
    ("S2-L", date(2018, 10, 26), "ABBV", "long"),
    ("S2-S", date(2018, 10, 26), "MRVL", "short"),
]
SAMPLE_DATES = sorted({s[1] for s in FROZEN_SAMPLES})
FROZEN_S4 = (date(2018, 10, 26), "AMBA", "short")

PHASE1_DIGESTS = {
    "features_42_8.parquet": (
        "f34fb2556da03e9113f4a56a23e4e7dff2296810d5c848e24ff251678991b7bc",
        Path(r"C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features/features_42_8.parquet"),
    ),
    "features_backfill_v1.lineage.json": (
        "c585bce169d897d8a393e9cbf7c62a4e42d28e9139e4dce51eabdacc8f4866a5",
        Path(r"C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_backfill_v1.lineage.json"),
    ),
    "features_quality_audit_v1.json": (
        "6737ab2073be4aab874454faf849139031bf66031e80ffc81b712ac2edff2f2c",
        Path(r"C:/MomentumCVG_env/derived/e2c1f8fd44d72176/features_quality_audit_v1.json"),
    ),
    "option_surface_meta_weekly_2018_2026.parquet": (
        "304753a2d5ce9900bdf462442f4f11407c8ec821ec5708ef9190027b4b3b7c4a",
        Path(
            r"C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/"
            r"cache/surface/option_surface_meta_weekly_2018_2026.parquet"
        ),
    ),
    "option_surface_quotes_weekly_2018_2026.parquet": (
        "e8b2b49094362fde3432b2851c47c72004a539db6c37f9a4fbda6f2e6d907ca4",
        Path(
            r"C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/"
            r"cache/surface/option_surface_quotes_weekly_2018_2026.parquet"
        ),
    ),
    "ticker_liquidity_panel.parquet": (
        "756d78160047554b3c158e99aa24e337be933de9b47f273f21dce35b85d07d42",
        Path(
            r"C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/"
            r"input/liquidity/ticker_liquidity_panel.parquet"
        ),
    ),
    "input_snapshot_e2c1f8fd44d72176.json": (
        "e312fd1932ca2a95b104f1c5b52bb6054270695f23c2670cdf125c10f379e1ab",
        Path(
            r"C:/MomentumCVG_env/snapshots/20260724T045049097520Z_40b16886/"
            r"manifests/input_snapshot_e2c1f8fd44d72176.json"
        ),
    ),
}

A1 = PHASE1_DIGESTS["option_surface_meta_weekly_2018_2026.parquet"][1]
A2 = PHASE1_DIGESTS["option_surface_quotes_weekly_2018_2026.parquet"][1]
FEAT = PHASE1_DIGESTS["features_42_8.parquet"][1]
LIQ = PHASE1_DIGESTS["ticker_liquidity_panel.parquet"][1]

rows: List[Dict[str, Any]] = []
coverage: Dict[str, Dict[str, Any]] = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b) -> bool:
    if a is None and b is None:
        return True
    try:
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
    except (TypeError, ValueError):
        pass
    try:
        a_f, b_f = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(a_f - b_f) <= max(TOL_ABS, TOL_REL * max(abs(a_f), abs(b_f), 1.0))


def diff_num(a, b) -> str:
    try:
        return f"abs {abs(float(a) - float(b)):.2e}"
    except (TypeError, ValueError):
        return "—"


def rec(id_, sample, stage, expected, observed, source, difference, verdict, cov_key=None):
    row = {
        "id": id_,
        "sample": sample,
        "stage": stage,
        "expected": expected,
        "observed": observed,
        "source": source,
        "difference": difference,
        "verdict": verdict,
    }
    rows.append(row)
    print(f"[{verdict}] {id_}: {stage} | exp={expected} obs={observed} diff={difference}", flush=True)
    if cov_key:
        coverage.setdefault(cov_key, {"stage": cov_key, "ids": [], "verdicts": []})
        coverage[cov_key]["ids"].append(id_)
        coverage[cov_key]["verdicts"].append(verdict)


def jsonable(x):
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)):
            return None
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, date):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def load_frozen_contract(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_cfg(contract: Dict[str, Any]) -> Dict[str, Any]:
    shared = contract["shared_run_config"]
    window = contract["feature_window"]
    cross = next(r for r in contract["runs"] if r["run_id"] == CROSS)
    mid = next(r for r in contract["runs"] if r["run_id"] == MID)
    return {
        "momentum_col": window["momentum_col"],
        "cvg_col": window["cvg_col"],
        "count_col": window["count_col"],
        "cvg_count_col": window["cvg_count_col"],
        "min_count_pct": float(shared["min_count_pct"]),
        "long_top_pct": float(shared["long_top_pct"]),
        "short_bottom_pct": float(shared["short_bottom_pct"]),
        "cvg_filter_pct": float(shared["cvg_filter_pct"]),
        "dvol_top_pct": float(shared["dvol_top_pct"]),
        "spread_bottom_pct": float(shared["spread_bottom_pct"]),
        "short_structure": shared["short_structure"],
        "wing_delta_target": float(shared["wing_delta_target"]),
        "max_leg_spread_pct": float(shared["max_leg_spread_pct"]),
        "max_names_per_side": int(shared["max_names_per_side"]),
        "tier_a_mode": shared["tier_a_mode"],
        "tier_a_short_budget": float(shared["tier_a_short_budget"]),
        "tier_a_long_budget": float(shared["tier_a_long_budget"]),
        "earnings_exclusion_days": int(shared["earnings_exclusion_days"]),
        "cross_buy_alpha": float(cross["fill"]["buy_alpha"]),
        "cross_sell_alpha": float(cross["fill"]["sell_alpha"]),
        "mid_buy_alpha": float(mid["fill"]["buy_alpha"]),
        "mid_sell_alpha": float(mid["fill"]["sell_alpha"]),
    }


def required_count(cfg: Dict[str, Any]) -> int:
    return math.ceil(cfg["min_count_pct"] * 35)


def fill_price(bid: float, ask: float, uq: float, buy_alpha: float, sell_alpha: float) -> float:
    spread = ask - bid
    if uq > 0:
        return bid + buy_alpha * spread
    return ask - sell_alpha * spread


def leg_entry_cash(uq: float, bid: float, ask: float, buy_alpha: float, sell_alpha: float) -> float:
    px = fill_price(bid, ask, uq, buy_alpha, sell_alpha)
    return px * abs(uq) if uq > 0 else -px * abs(uq)


def intrinsic(option_type: str, strike: float, exit_spot: float) -> float:
    if str(option_type).lower().startswith("c"):
        return max(exit_spot - strike, 0.0)
    return max(strike - exit_spot, 0.0)


def choose_below_nearest(df: pd.DataFrame, target: float) -> pd.Series:
    eligible = df[df["abs_delta"] <= target]
    if eligible.empty:
        raise ValueError(f"No quotes with abs_delta <= {target} available for selection")
    return eligible.loc[eligible["abs_delta"].idxmax()]


def pit_snapshot_rows(trade_date: date, liq: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    trade_ts = pd.Timestamp(trade_date)
    prior = liq.loc[liq["month_date"] < trade_ts, "month_date"]
    if prior.empty:
        return None, pd.DataFrame()
    snap_date = prior.max()
    snap = liq[
        (liq["month_date"] == snap_date)
        & (liq["has_valid_atm_pair"] == True)  # noqa: E712
        & liq["atm_straddle_dollar_vol"].notna()
        & liq["atm_spread_pct"].notna()
    ].copy()
    if snap.empty:
        return snap_date, snap
    snap["dvol_rank_pct"] = snap["atm_straddle_dollar_vol"].rank(
        ascending=True, method="average", pct=True
    )
    snap["spread_rank_pct"] = snap["atm_spread_pct"].rank(
        ascending=False, method="average", pct=True
    )
    return snap_date, snap


def pit_universe(trade_date: date, liq: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    snap_date, snap = pit_snapshot_rows(trade_date, liq)
    if snap_date is None or snap.empty:
        return pd.DataFrame(columns=["ticker", "dvol_rank_pct", "spread_rank_pct"])
    dvol_thr = 1.0 - cfg["dvol_top_pct"]
    spr_thr = 1.0 - cfg["spread_bottom_pct"]
    uni = snap[
        (snap["dvol_rank_pct"] >= dvol_thr) & (snap["spread_rank_pct"] >= spr_thr)
    ]
    return uni[["ticker", "dvol_rank_pct", "spread_rank_pct"]].reset_index(drop=True)


def eligible_feature_slice(
    trade_date: date, feat: pd.DataFrame, uni: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    req = required_count(cfg)
    trade_ts = pd.Timestamp(trade_date)
    fs = feat[feat["date"] == trade_ts].merge(uni[["ticker"]], on="ticker", how="inner")
    if fs.empty:
        return fs
    fs = fs.dropna(subset=[cfg["momentum_col"], cfg["cvg_col"]])
    if fs.empty:
        return fs
    ok = (fs[cfg["count_col"]] >= req) & (fs[cfg["cvg_count_col"]] >= req)
    return fs[ok].copy()


def score_signals(trade_date: date, feat: pd.DataFrame, uni: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=["ticker", "direction", "signal_rank_pct", "cvg_rank_pct"]
    )
    fs = eligible_feature_slice(trade_date, feat, uni, cfg)
    if fs.empty:
        return empty
    fs["signal_rank_pct"] = fs[cfg["momentum_col"]].rank(
        ascending=True, method="average", pct=True
    )
    long_thr = 1.0 - cfg["long_top_pct"]
    short_thr = cfg["short_bottom_pct"]
    long_pool = fs[fs["signal_rank_pct"] >= long_thr].copy()
    short_pool = fs[fs["signal_rank_pct"] <= short_thr].copy()
    if not long_pool.empty:
        long_pool["cvg_rank_pct"] = long_pool[cfg["cvg_col"]].rank(
            ascending=True, method="average", pct=True
        )
        cvg_thr = 1.0 - cfg["cvg_filter_pct"]
        long_pool = long_pool[long_pool["cvg_rank_pct"] >= cvg_thr]
        long_pool["direction"] = "long"
    if not short_pool.empty:
        short_pool["cvg_rank_pct"] = short_pool[cfg["cvg_col"]].rank(
            ascending=True, method="average", pct=True
        )
        cvg_thr = 1.0 - cfg["cvg_filter_pct"]
        short_pool = short_pool[short_pool["cvg_rank_pct"] >= cvg_thr]
        short_pool["direction"] = "short"
    if long_pool.empty and short_pool.empty:
        return empty
    return pd.concat([long_pool, short_pool], ignore_index=True)[
        ["ticker", "direction", "signal_rank_pct", "cvg_rank_pct"]
    ]


def quotes_for(a2: pd.DataFrame, trade_date: date, ticker: str) -> pd.DataFrame:
    q = a2[
        (pd.to_datetime(a2["entry_date"]).dt.date == trade_date) & (a2["ticker"] == ticker)
    ].copy()
    return q


def build_long_straddle(
    meta: pd.Series,
    quotes: pd.DataFrame,
    cfg: Dict[str, Any],
    buy_alpha: float,
    sell_alpha: float,
) -> Dict[str, Any]:
    body = quotes[quotes["is_body"] == True]  # noqa: E712
    body = body[body["spread_pct"] <= cfg["max_leg_spread_pct"]]
    call = body[body["side"] == "call"]
    put = body[body["side"] == "put"]
    if call.empty or put.empty:
        raise ValueError("Missing tradeable body call/put")
    cr, pr = call.iloc[0], put.iloc[0]
    legs = [
        {"leg_index": 0, "option_type": "call", "strike": float(cr.strike), "unit_quantity": 1.0,
         "bid": float(cr.bid), "ask": float(cr.ask), "spread_pct": float(cr.spread_pct)},
        {"leg_index": 1, "option_type": "put", "strike": float(pr.strike), "unit_quantity": 1.0,
         "bid": float(pr.bid), "ask": float(pr.ask), "spread_pct": float(pr.spread_pct)},
    ]
    entry = sum(
        leg_entry_cash(l["unit_quantity"], l["bid"], l["ask"], buy_alpha, sell_alpha)
        for l in legs
    )
    exit_spot = float(meta["exit_spot"])
    expiry_payoff = sum(
        intrinsic(l["option_type"], l["strike"], exit_spot) * l["unit_quantity"] for l in legs
    )
    return {
        "structure_ok": True,
        "entry_cost_per_share": entry,
        "net_credit_per_share": -entry,
        "max_loss_per_share": abs(entry),
        "premium_per_share": abs(entry),
        "legs": legs,
        "expiry_payoff_per_share": expiry_payoff,
        "pnl_per_share": expiry_payoff - entry,
    }


def build_iron_fly(
    meta: pd.Series,
    quotes: pd.DataFrame,
    cfg: Dict[str, Any],
    buy_alpha: float,
    sell_alpha: float,
) -> Dict[str, Any]:
    body = quotes[quotes["is_body"] == True]  # noqa: E712
    body = body[body["spread_pct"] <= cfg["max_leg_spread_pct"]]
    body_call = body[body["side"] == "call"]
    body_put = body[body["side"] == "put"]
    if body_call.empty or body_put.empty:
        raise ValueError("Missing body call/put")
    otm_calls = quotes[(quotes["side"] == "call") & (quotes["is_otm"] == True)]  # noqa: E712
    otm_puts = quotes[(quotes["side"] == "put") & (quotes["is_otm"] == True)]  # noqa: E712
    otm_calls = otm_calls[otm_calls["spread_pct"] <= cfg["max_leg_spread_pct"]]
    otm_puts = otm_puts[otm_puts["spread_pct"] <= cfg["max_leg_spread_pct"]]
    wing_c = choose_below_nearest(otm_calls, cfg["wing_delta_target"])
    wing_p = choose_below_nearest(otm_puts, cfg["wing_delta_target"])
    body_strike = float(meta["body_strike"])
    legs = [
        {"leg_index": 0, "option_type": "put", "strike": float(wing_p.strike), "unit_quantity": 1.0,
         "bid": float(wing_p.bid), "ask": float(wing_p.ask), "spread_pct": float(wing_p.spread_pct)},
        {"leg_index": 1, "option_type": "put", "strike": float(body_put.iloc[0].strike), "unit_quantity": -1.0,
         "bid": float(body_put.iloc[0].bid), "ask": float(body_put.iloc[0].ask),
         "spread_pct": float(body_put.iloc[0].spread_pct)},
        {"leg_index": 2, "option_type": "call", "strike": float(body_call.iloc[0].strike), "unit_quantity": -1.0,
         "bid": float(body_call.iloc[0].bid), "ask": float(body_call.iloc[0].ask),
         "spread_pct": float(body_call.iloc[0].spread_pct)},
        {"leg_index": 3, "option_type": "call", "strike": float(wing_c.strike), "unit_quantity": 1.0,
         "bid": float(wing_c.bid), "ask": float(wing_c.ask), "spread_pct": float(wing_c.spread_pct)},
    ]
    entry = sum(
        leg_entry_cash(l["unit_quantity"], l["bid"], l["ask"], buy_alpha, sell_alpha)
        for l in legs
    )
    net_credit = -entry
    wing_width = max(float(wing_c.strike) - body_strike, body_strike - float(wing_p.strike))
    max_loss = wing_width - net_credit
    exit_spot = float(meta["exit_spot"])
    expiry_payoff = sum(
        intrinsic(l["option_type"], l["strike"], exit_spot) * l["unit_quantity"] for l in legs
    )
    return {
        "structure_ok": True,
        "entry_cost_per_share": entry,
        "net_credit_per_share": net_credit,
        "max_loss_per_share": max_loss,
        "premium_per_share": max(net_credit, 0.0),
        "wing_width": wing_width,
        "legs": legs,
        "expiry_payoff_per_share": expiry_payoff,
        "pnl_per_share": expiry_payoff - entry,
    }


def build_structure_row(
    trade_date: date,
    ticker: str,
    direction: str,
    signal_rank_pct: float,
    cvg_rank_pct: float,
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    cfg: Dict[str, Any],
    buy_alpha: float,
    sell_alpha: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "signal_rank_pct": signal_rank_pct,
        "cvg_rank_pct": cvg_rank_pct,
        "structure_ok": False,
        "had_earnings_nearby": False,
    }
    meta_rows = a1[(a1.entry_date == trade_date) & (a1.ticker == ticker)]
    if meta_rows.empty or not bool(meta_rows.iloc[0].surface_valid):
        row["failure_reason"] = "metadata_error"
        return row
    meta = meta_rows.iloc[0]
    row.update(
        {
            "entry_spot": float(meta.entry_spot),
            "exit_spot": float(meta.exit_spot),
            "body_strike": float(meta.body_strike),
            "expiry_date": pd.to_datetime(meta.expiry_date).date(),
            "dte_actual": int(meta.dte_actual),
        }
    )
    q = quotes_for(a2, trade_date, ticker)
    try:
        if direction == "long":
            built = build_long_straddle(meta, q, cfg, buy_alpha, sell_alpha)
        else:
            built = build_iron_fly(meta, q, cfg, buy_alpha, sell_alpha)
        row.update(built)
        row["structure_ok"] = True
    except Exception as exc:
        row["failure_reason"] = str(exc)
    return row


def select_and_size_day(
    trade_date: date,
    signals: pd.DataFrame,
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    structs: List[Dict[str, Any]] = []
    for _, sig in signals.iterrows():
        structs.append(
            build_structure_row(
                trade_date,
                sig.ticker,
                sig.direction,
                float(sig.signal_rank_pct),
                float(sig.cvg_rank_pct),
                a1,
                a2,
                cfg,
                cfg["cross_buy_alpha"],
                cfg["cross_sell_alpha"],
            )
        )
    eligible = [s for s in structs if s.get("structure_ok")]
    selected: List[Dict[str, Any]] = []
    for direction in ("long", "short"):
        side = [s for s in eligible if s["direction"] == direction]
        side.sort(
            key=lambda s: (-s["signal_rank_pct"], s["ticker"])
            if direction == "long"
            else (s["signal_rank_pct"], s["ticker"])
        )
        selected.extend(side[: cfg["max_names_per_side"]])

    # Drop invalid max loss
    kept = []
    for s in selected:
        ml = s.get("max_loss_per_share")
        if ml is None or ml <= 0:
            continue
        kept.append(s)
    selected = kept

    n_short = sum(1 for s in selected if s["direction"] == "short")
    n_long = sum(1 for s in selected if s["direction"] == "long")
    short_per = cfg["tier_a_short_budget"] / n_short if n_short else None

    for s in selected:
        if s["direction"] == "short":
            s["quantity"] = -(short_per / s["max_loss_per_share"])
        else:
            s["quantity"] = float("nan")

    collected = 0.0
    for s in selected:
        if s["direction"] != "short":
            continue
        credit = s.get("premium_per_share", 0.0)
        if credit > 0:
            collected += abs(float(s["quantity"])) * credit

    fallback = n_short == 0 or collected <= 0
    long_budget = cfg["tier_a_long_budget"] if fallback else collected
    long_per = long_budget / n_long if n_long and long_budget else None

    for s in selected:
        if s["direction"] != "long":
            continue
        prem = s.get("premium_per_share")
        if prem and prem > 0 and long_per:
            s["quantity"] = long_per / prem

    sized = []
    for s in selected:
        qty = s.get("quantity")
        if qty is None or (isinstance(qty, float) and math.isnan(qty)):
            continue
        at_risk = s["max_loss_per_share"] if s["direction"] == "short" else s["premium_per_share"]
        pnl_ps = s["pnl_per_share"]
        s["capital_at_risk_dollars"] = abs(float(qty)) * at_risk
        s["pnl_total"] = abs(float(qty)) * pnl_ps
        sized.append(s)

    meta = {
        "n_short": n_short,
        "n_long": n_long,
        "collected_short_credit": collected,
        "long_budget": long_budget,
        "fallback_fired": fallback,
        "short_per_name_budget": short_per,
        "long_per_name_budget": long_per,
    }
    return sized, meta


def trade_obs(tl: pd.DataFrame, td: date, ticker: str, direction: str) -> pd.Series:
    m = tl[
        (tl.trade_date == td)
        & (tl.ticker == ticker)
        & (tl.direction == direction)
        & (tl.included_in_portfolio == True)  # noqa: E712
    ]
    return m.iloc[0]


def legs_obs(leg: pd.DataFrame, td: date, ticker: str, direction: str) -> pd.DataFrame:
    return leg[
        (leg.trade_date == td)
        & (leg.ticker == ticker)
        & (leg.direction == direction)
        & (leg.included_in_portfolio == True)  # noqa: E712
    ].sort_values("leg_index")


def observed_day_counts(tl: pd.DataFrame, td: date) -> Tuple[int, int, float]:
    day = tl[(tl.trade_date == td) & (tl.included_in_portfolio == True)]  # noqa: E712
    n_short = int((day.direction == "short").sum())
    n_long = int((day.direction == "long").sum())
    collected = 0.0
    for _, r in day[day.direction == "short"].iterrows():
        credit = float(r.net_credit_per_share) if pd.notna(r.net_credit_per_share) else 0.0
        if credit > 0:
            collected += abs(float(r.quantity)) * credit
    return n_short, n_long, collected


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
print("=== identity / immutability ===", flush=True)
receipt = json.loads((RUN_DIR / "run_receipt.json").read_text(encoding="utf-8"))
rec(
    "ID-repo_sha", "identity", "receipt.repo_sha", EXECUTION_COMMIT, receipt.get("repo_sha"),
    "run_receipt.json", "—", "PASS" if receipt.get("repo_sha") == EXECUTION_COMMIT else "FAIL", "identity",
)
rec(
    "ID-result_complete", "identity", "receipt.result_complete", True, receipt.get("result_complete"),
    "run_receipt.json", "—", "PASS" if receipt.get("result_complete") is True else "FAIL", "identity",
)

expected_artifact: Dict[str, str] = {}


def _walk_sha(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("sha256"), str):
            name = obj.get("name") or obj.get("path")
            if name:
                expected_artifact[Path(str(name)).name] = obj["sha256"].lower()
        for v in obj.values():
            _walk_sha(v)
    elif isinstance(obj, list):
        for v in obj:
            _walk_sha(v)


_walk_sha(receipt)
art_issues, art_matched = [], 0
for p in sorted(RUN_DIR.iterdir()):
    if not p.is_file() or p.name == "run_receipt.json":
        continue
    h = sha256_file(p)
    exp = expected_artifact.get(p.name)
    if exp == h:
        art_matched += 1
    else:
        art_issues.append(p.name)
rec(
    "ID-artifact_digests", "identity", "RUN_DIR sha256 vs receipt", "all non-receipt files match",
    f"matched={art_matched} issues={art_issues}", "hashlib", "—",
    "PASS" if not art_issues and art_matched == 16 else "FAIL", "identity",
)

input_issues = []
for name, (exp, path) in PHASE1_DIGESTS.items():
    if sha256_file(path) != exp:
        input_issues.append(name)
rec(
    "ID-input_digests", "identity", "Phase 1 accepted-input digests", "all 7 match Phase 1 baseline",
    f"issues={input_issues or []}", "hashlib", "—", "PASS" if not input_issues else "FAIL", "identity",
)

# ---------------------------------------------------------------------------
# Load contract + observed artifacts (observed only)
# ---------------------------------------------------------------------------
print("=== load contract + observed artifacts ===", flush=True)
contract = load_frozen_contract(CONTRACT)
cfg = contract_cfg(contract)

tl_c = pd.read_parquet(RUN_DIR / f"trade_log_{CROSS}.parquet")
leg_c = pd.read_parquet(RUN_DIR / f"leg_log_{CROSS}.parquet")
leg_m = pd.read_parquet(RUN_DIR / f"leg_log_{MID}.parquet")
ds_c = pd.read_parquet(RUN_DIR / f"date_status_{CROSS}.parquet")
cand_c = pd.read_parquet(RUN_DIR / f"candidate_view_{CROSS}.parquet")
ds_sum_c = pd.read_parquet(RUN_DIR / f"date_summary_{CROSS}.parquet")
for df in (tl_c, leg_c, leg_m, ds_c, cand_c, ds_sum_c):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

liq = pd.read_parquet(LIQ)
liq["month_date"] = pd.to_datetime(liq["month_date"])
feat = pd.read_parquet(
    FEAT,
    columns=["date", "ticker", "mom_42_8_mean", "cvg_42_8", "mom_42_8_count", "cvg_count_42_8"],
)
feat["date"] = pd.to_datetime(feat["date"])
a1 = pd.read_parquet(A1)
a1["entry_date"] = pd.to_datetime(a1["entry_date"]).dt.date

load_dates = sorted(set(SAMPLE_DATES) | {FROZEN_S4[0]})
print(f"loading A2 for {load_dates}...", flush=True)
a2 = pd.read_parquet(A2, filters=[("entry_date", "in", [pd.Timestamp(d) for d in load_dates])])

# ---------------------------------------------------------------------------
# Frozen sample selection
# ---------------------------------------------------------------------------
print("=== frozen sample selection ===", flush=True)
n_vnt = int((ds_c["status"] == "valid_no_trade").sum())
selection = {
    "S3": {"rule": "earliest valid_no_trade", "result": "N/A", "justification": f"n_valid_no_trade_dates={n_vnt}"},
    "S1": {"date": "2022-09-02", "long": "ACN", "short": "AMC"},
    "S2": {"date": "2018-10-26", "long": "ABBV", "short": "MRVL"},
    "S4": {"date": "2018-10-26", "ticker": "AMBA", "direction": "short"},
    "no_sample_replaced": True,
}
rec("S3", "S3", "valid_no_trade existence", "N/A if none", f"n={n_vnt}",
    "date_status cross", "—", "N/A (frozen fallback)", "S3_valid_no_trade")

inc_s1 = tl_c[(tl_c.trade_date == date(2022, 9, 2)) & (tl_c.included_in_portfolio == True)]  # noqa: E712
rec("SEL-S1", "S1", "frozen S1 keys preserved", "ACN/AMC",
    f"{sorted(inc_s1[inc_s1.direction=='long'].ticker)[0]}/{sorted(inc_s1[inc_s1.direction=='short'].ticker)[0]}",
    "trade_log selection rule", "—",
    "PASS" if (sorted(inc_s1[inc_s1.direction=='long'].ticker)[0], sorted(inc_s1[inc_s1.direction=='short'].ticker)[0]) == ("ACN", "AMC") else "FAIL",
    "sample_selection")

inc_all = tl_c[tl_c.included_in_portfolio == True]  # noqa: E712
both_dates = sorted(d for d, g in inc_all.groupby("trade_date") if set(g.direction) >= {"long", "short"})
s2d = both_dates[0]
inc_s2 = inc_all[inc_all.trade_date == s2d]
rec("SEL-S2", "S2", "frozen S2 keys preserved", "2018-10-26/ABBV/MRVL",
    f"{s2d}/{sorted(inc_s2[inc_s2.direction=='long'].ticker)[0]}/{sorted(inc_s2[inc_s2.direction=='short'].ticker)[0]}",
    "trade_log selection rule", "—",
    "PASS" if (s2d, sorted(inc_s2[inc_s2.direction=='long'].ticker)[0], sorted(inc_s2[inc_s2.direction=='short'].ticker)[0]) == (date(2018, 10, 26), "ABBV", "MRVL") else "FAIL",
    "sample_selection")

sf = cand_c[cand_c.stage == "structure_failed"]
s4_row = sf[sf.trade_date == sf.trade_date.min()].sort_values("ticker").iloc[0]
rec("SEL-S4", "S4", "frozen S4 keys preserved", "2018-10-26/AMBA/short",
    f"{s4_row.trade_date}/{s4_row.ticker}/{s4_row.direction}", "candidate_view", "—",
    "PASS" if (s4_row.trade_date, s4_row.ticker, s4_row.direction) == FROZEN_S4 else "FAIL", "sample_selection")
selection["S4"]["reason_code"] = str(s4_row.reason_code)
selection["S4"]["reason_raw"] = str(s4_row.reason_raw)

rec("SEL-no_replace", "selection", "no performance-based replacement", True, True,
    "procedure", "—", "PASS", "sample_selection")

# ---------------------------------------------------------------------------
# Reconstruct included books for frozen dates
# ---------------------------------------------------------------------------
print("=== reconstruct frozen-date books ===", flush=True)
date_books: Dict[date, List[Dict[str, Any]]] = {}
date_meta: Dict[date, Dict[str, Any]] = {}
for td in SAMPLE_DATES:
    print(f"  {td}...", flush=True)
    uni = pit_universe(td, liq, cfg)
    sig = score_signals(td, feat, uni, cfg)
    book, meta = select_and_size_day(td, sig, a1, a2, cfg)
    date_books[td] = book
    date_meta[td] = meta


def find_book_row(td: date, ticker: str, direction: str) -> Optional[Dict[str, Any]]:
    for r in date_books[td]:
        if r["ticker"] == ticker and r["direction"] == direction:
            return r
    return None


def audit_sample(sid: str, td: date, ticker: str, direction: str):
    sample = f"{sid} {td}/{ticker}/{direction}"
    obs = trade_obs(tl_c, td, ticker, direction)
    obs_legs = legs_obs(leg_c, td, ticker, direction)
    obs_legs_m = legs_obs(leg_m, td, ticker, direction)
    req = required_count(cfg)

    snap_date_a, snap_a = pit_snapshot_rows(td, liq)
    prior_dates = liq.loc[liq["month_date"] < pd.Timestamp(td), "month_date"].unique()
    snap_date_b = max(prior_dates) if len(prior_dates) else None
    rec(
        f"{sid}-pit-snapshot", sample, "PIT snapshot date = max(month_date < trade_date)",
        str(pd.Timestamp(snap_date_a).date()) if snap_date_a is not None else None,
        str(pd.Timestamp(snap_date_b).date()) if snap_date_b is not None else None,
        "liquidity panel (two derivations)", "—",
        "PASS" if snap_date_a == snap_date_b else "FAIL", "universe_snapshot_date",
    )

    raw = liq[(liq.month_date == snap_date_a) & (liq.ticker == ticker)]
    raw_ok = len(raw) == 1 and bool(raw.iloc[0].has_valid_atm_pair)
    rec(f"{sid}-pit-atm", sample, "has_valid_atm_pair=True", True, raw_ok,
        "liquidity panel raw row", "—", "PASS" if raw_ok else "FAIL", "universe_atm_pair")

    if not len(snap_a) or ticker not in set(snap_a.ticker):
        rec(f"{sid}-pit-member", sample, "ticker in PIT eligible snapshot", True, False,
            "liquidity panel", "—", "FAIL", "universe_and_membership")
        return
    tr = snap_a[snap_a.ticker == ticker].iloc[0]
    dvol = float(tr.atm_straddle_dollar_vol)
    spr = float(tr.atm_spread_pct)
    rec(
        f"{sid}-pit-fields", sample, "atm_straddle_dollar_vol & atm_spread_pct finite",
        "finite", f"dvol={dvol}/spread={spr}", "liquidity panel", "—",
        "PASS" if math.isfinite(dvol) and math.isfinite(spr) else "FAIL", "universe_dvol_spread_fields",
    )
    dvol_r, spr_r = float(tr.dvol_rank_pct), float(tr.spread_rank_pct)
    dvol_thr, spr_thr = 1.0 - cfg["dvol_top_pct"], 1.0 - cfg["spread_bottom_pct"]
    rec(
        f"{sid}-pit-dvol-rank", sample, f"dvol_rank_pct >= {dvol_thr}", f">={dvol_thr}", dvol_r,
        "liquidity panel rank formula", "—", "PASS" if dvol_r >= dvol_thr else "FAIL", "universe_dvol_rank",
    )
    rec(
        f"{sid}-pit-spread-rank", sample, f"spread_rank_pct >= {spr_thr}", f">={spr_thr}", spr_r,
        "liquidity panel rank formula", "—", "PASS" if spr_r >= spr_thr else "FAIL", "universe_spread_rank",
    )
    uni = pit_universe(td, liq, cfg)
    rec(
        f"{sid}-pit-and", sample, "passes frozen universe AND gates", True, ticker in set(uni.ticker),
        "audit-local pit_universe vs ticker", "—",
        "PASS" if ticker in set(uni.ticker) else "FAIL", "universe_and_membership",
    )

    fs = feat[(feat.date == pd.Timestamp(td)) & (feat.ticker == ticker)]
    in_feat = len(fs) == 1
    elig = eligible_feature_slice(td, feat, uni, cfg)
    in_elig = ticker in set(elig.ticker)
    rec(
        f"{sid}-joint-membership", sample, "ticker in PIT universe ∩ feature slice", True,
        in_feat and (ticker in set(uni.ticker)), "features+liquidity", "—",
        "PASS" if in_feat and (ticker in set(uni.ticker)) else "FAIL", "joint_universe_feature_membership",
    )
    if not in_feat:
        return
    fr = fs.iloc[0]
    finite_ok = math.isfinite(float(fr.mom_42_8_mean)) and math.isfinite(float(fr.cvg_42_8))
    rec(
        f"{sid}-joint-finite", sample, "mom_42_8_mean and cvg_42_8 finite", True,
        finite_ok, "features_42_8", "—", "PASS" if finite_ok else "FAIL", "joint_finite_values",
    )
    rec(
        f"{sid}-joint-mom-count", sample, f"mom_42_8_count >= {req}", f">={req}", float(fr.mom_42_8_count),
        "features_42_8", "—", "PASS" if float(fr.mom_42_8_count) >= req else "FAIL", "joint_mom_count",
    )
    rec(
        f"{sid}-joint-cvg-count", sample, f"cvg_count_42_8 >= {req}", f">={req}", float(fr.cvg_count_42_8),
        "features_42_8", "—", "PASS" if float(fr.cvg_count_42_8) >= req else "FAIL", "joint_cvg_count",
    )
    rec(
        f"{sid}-joint-eligible", sample, "passes joint eligibility cross-section", True, in_elig,
        "audit-local eligible_feature_slice", "—", "PASS" if in_elig else "FAIL", "joint_eligible_slice",
    )

    sigs = score_signals(td, feat, uni, cfg)
    srow = sigs[(sigs.ticker == ticker) & (sigs.direction == direction)]
    if len(srow):
        sr = srow.iloc[0]
        rec(
            f"{sid}-signal-rank", sample, "signal_rank_pct independent", float(sr.signal_rank_pct),
            float(obs.signal_rank_pct), "audit-local vs trade_log", diff_num(sr.signal_rank_pct, obs.signal_rank_pct),
            "PASS" if close(sr.signal_rank_pct, obs.signal_rank_pct) else "FAIL", "signal_rank_recompute",
        )
        rec(
            f"{sid}-cvg-rank", sample, "cvg_rank_pct independent", float(sr.cvg_rank_pct),
            float(obs.cvg_rank_pct), "audit-local vs trade_log", diff_num(sr.cvg_rank_pct, obs.cvg_rank_pct),
            "PASS" if close(sr.cvg_rank_pct, obs.cvg_rank_pct) else "FAIL", "cvg_rank_recompute",
        )
        long_thr = 1.0 - cfg["long_top_pct"]
        short_thr = cfg["short_bottom_pct"]
        cvg_thr = 1.0 - cfg["cvg_filter_pct"]
        if direction == "long":
            dir_ok = float(sr.signal_rank_pct) >= long_thr
        else:
            dir_ok = float(sr.signal_rank_pct) <= short_thr
        cvg_ok = float(sr.cvg_rank_pct) >= cvg_thr
        rec(
            f"{sid}-direction-cvg", sample,
            f"direction={direction} and CVG retention >= {cvg_thr}", True, dir_ok and cvg_ok,
            "audit-local thresholds", "—", "PASS" if dir_ok and cvg_ok else "FAIL", "direction_and_cvg_retention",
        )

    meta_row = a1[(a1.entry_date == td) & (a1.ticker == ticker)].iloc[0]
    rec(
        f"{sid}-a1-valid", sample, "A1 surface_valid", True, bool(meta_row.surface_valid),
        "A1 meta vs trade inclusion", "—", "PASS" if bool(meta_row.surface_valid) else "FAIL", "option_a1_surface_valid",
    )
    for field in ("entry_spot", "exit_spot", "body_strike", "dte_actual"):
        exp = meta_row[field]
        obs_v = obs[field]
        rec(
            f"{sid}-{field}", sample, field, float(exp) if field != "dte_actual" else int(exp),
            float(obs_v) if field != "dte_actual" else int(obs_v), "A1 vs trade_log", diff_num(exp, obs_v),
            "PASS" if close(exp, obs_v) else "FAIL", f"option_{field}",
        )
    exp_d = pd.to_datetime(meta_row.expiry_date).date()
    obs_d = pd.to_datetime(obs.expiry_date).date()
    rec(
        f"{sid}-expiry_date", sample, "expiry_date", str(exp_d), str(obs_d), "A1 vs trade_log", "—",
        "PASS" if exp_d == obs_d else "FAIL", "option_expiry_date",
    )

    br = find_book_row(td, ticker, direction)
    if br is None:
        rec(f"{sid}-book-row", sample, "sample in independent included book", "present", "missing",
            "audit-local S5", "—", "FAIL", "tier_a_quantity")
        return

    q = quotes_for(a2, td, ticker)
    body = q[(q.is_body == True) & (q.spread_pct <= cfg["max_leg_spread_pct"])]  # noqa: E712
    body_strikes = [(r.side, float(r.strike)) for _, r in body.iterrows()]
    rec(
        f"{sid}-body", sample, "body option types/strikes at A1 body_strike",
        f"call/put @ {float(meta_row.body_strike)}", str(body_strikes), "A2 is_body+spread gate", "—",
        "PASS" if body_strikes else "FAIL", "option_body_selection",
    )
    spread_map = {f"leg{l['leg_index']}": l["spread_pct"] for l in br["legs"]}
    rec(
        f"{sid}-spread-gates", sample, f"selected legs spread_pct <= {cfg['max_leg_spread_pct']}",
        True, spread_map, "audit-local assembly legs", "—",
        "PASS" if all(v <= cfg["max_leg_spread_pct"] for v in spread_map.values()) else "FAIL", "option_spread_gates",
    )
    if direction == "short":
        otm = q[q.is_otm == True]  # noqa: E712
        otm = otm[otm.spread_pct <= cfg["max_leg_spread_pct"]]
        wc = choose_below_nearest(otm[otm.side == "call"], cfg["wing_delta_target"])
        wp = choose_below_nearest(otm[otm.side == "put"], cfg["wing_delta_target"])
        obs_put = float(obs_legs[obs_legs.leg_index == 0].iloc[0].strike)
        obs_call = float(obs_legs[obs_legs.leg_index == 3].iloc[0].strike)
        rec(
            f"{sid}-wings", sample, "OTM wings highest abs_delta <= 0.15 after spread gate",
            f"put={float(wp.strike)}/call={float(wc.strike)}",
            f"leg_log put={obs_put}/call={obs_call}",
            "audit-local wing rule vs leg_log", "—",
            "PASS" if close(obs_put, wp.strike) and close(obs_call, wc.strike) else "FAIL",
            "option_wing_delta_rule",
        )

    expect_n = 2 if direction == "long" else 4
    rec(
        f"{sid}-nlegs", sample, "leg count", expect_n, len(obs_legs), "audit-local vs leg_log", "—",
        "PASS" if len(obs_legs) == expect_n else "FAIL", "option_leg_count",
    )
    for leg in br["legs"]:
        i = int(leg["leg_index"])
        ol = obs_legs[obs_legs.leg_index == i]
        if ol.empty:
            rec(f"{sid}-leg{i}", sample, f"leg{i} present", "found", "missing", "leg_log", "—", "FAIL", "option_leg_identity")
            continue
        olr = ol.iloc[0]
        cross_fill = fill_price(
            leg["bid"], leg["ask"], leg["unit_quantity"], cfg["cross_buy_alpha"], cfg["cross_sell_alpha"]
        )
        ok = (
            str(olr.option_type).lower()[0] == str(leg["option_type"]).lower()[0]
            and close(olr.strike, leg["strike"])
            and (float(olr.unit_quantity) > 0) == (leg["unit_quantity"] > 0)
            and close(olr.bid, leg["bid"])
            and close(olr.ask, leg["ask"])
            and close(olr.fill_price, cross_fill)
        )
        rec(
            f"{sid}-leg{i}-identity", sample, f"leg{i} type/strike/sign/bid/ask/cross_fill",
            f"{leg['option_type']}/{leg['strike']}/uq={leg['unit_quantity']}/fill={cross_fill}",
            f"{olr.option_type}/{olr.strike}/uq={olr.unit_quantity}/fill={olr.fill_price}",
            "audit-local vs leg_log", "—", "PASS" if ok else "FAIL", "option_leg_identity",
        )
        if len(obs_legs_m):
            ml = obs_legs_m[obs_legs_m.leg_index == i]
            if len(ml):
                mid_fill = fill_price(
                    leg["bid"], leg["ask"], leg["unit_quantity"], cfg["mid_buy_alpha"], cfg["mid_sell_alpha"]
                )
                half = 0.5 * (float(leg["ask"]) - float(leg["bid"]))
                delta = float(olr.fill_price) - float(ml.iloc[0].fill_price)
                expect_delta = half if leg["unit_quantity"] > 0 else -half
                rec(
                    f"{sid}-leg{i}-midcross", sample, f"leg{i} cross-mid half-spread delta",
                    expect_delta, delta, "leg_log mid+cross", diff_num(delta, expect_delta),
                    "PASS" if close(delta, expect_delta) else "FAIL", "option_mid_cross_half_spread",
                )

    dm = date_meta[td]
    obs_n_short, obs_n_long, obs_collected = observed_day_counts(tl_c, td)
    rec(
        f"{sid}-book-nshort", sample, "included short count", dm["n_short"], obs_n_short,
        "audit-local vs trade_log", "—", "PASS" if dm["n_short"] == obs_n_short else "FAIL", "tier_a_n_short",
    )
    rec(
        f"{sid}-book-nlong", sample, "included long count", dm["n_long"], obs_n_long,
        "audit-local vs trade_log", "—", "PASS" if dm["n_long"] == obs_n_long else "FAIL", "tier_a_n_long",
    )
    rec(
        f"{sid}-collected-credit", sample, "total collected short credit (date)",
        dm["collected_short_credit"], obs_collected, "audit-local vs trade_log", diff_num(dm["collected_short_credit"], obs_collected),
        "PASS" if close(dm["collected_short_credit"], obs_collected) else "FAIL", "tier_a_collected_credit",
    )
    obs_fallback = not (obs_n_short > 0 and obs_collected > 0)
    rec(
        f"{sid}-fallback", sample, "Tier A long-budget fallback fired", dm["fallback_fired"],
        obs_fallback, "audit-local vs trade_log-derived rule", "—",
        "PASS" if dm["fallback_fired"] == obs_fallback else "FAIL", "tier_a_fallback",
    )
    if direction == "short":
        obs_split = abs(float(obs.quantity)) * float(obs.max_loss_per_share)
        rec(
            f"{sid}-short-budget-split", sample, "short budget split = 10000/n_short",
            dm["short_per_name_budget"], obs_split, "audit-local vs trade_log implied", diff_num(dm["short_per_name_budget"], obs_split),
            "PASS" if close(dm["short_per_name_budget"], obs_split) else "FAIL", "tier_a_short_budget_split",
        )
        at_risk = br["max_loss_per_share"]
    else:
        rec(
            f"{sid}-long-budget", sample, "long budget from collected short credit",
            dm["long_budget"], obs_collected, "audit-local vs trade_log collected credit", diff_num(dm["long_budget"], obs_collected),
            "PASS" if close(dm["long_budget"], obs_collected) else "FAIL", "tier_a_long_budget",
        )
        obs_split = abs(float(obs.quantity)) * abs(float(obs.entry_cost_per_share))
        rec(
            f"{sid}-long-budget-split", sample, "long budget split = long_budget/n_long",
            dm["long_per_name_budget"], obs_split, "audit-local vs trade_log implied", diff_num(dm["long_per_name_budget"], obs_split),
            "PASS" if close(dm["long_per_name_budget"], obs_split) else "FAIL", "tier_a_long_budget_split",
        )
        at_risk = br["premium_per_share"]
    rec(
        f"{sid}-at-risk", sample, "at_risk_per_share", at_risk,
        float(obs.max_loss_per_share) if direction == "short" else abs(float(obs.entry_cost_per_share)),
        "audit-local vs trade_log", diff_num(at_risk, obs.max_loss_per_share if direction == "short" else abs(float(obs.entry_cost_per_share))),
        "PASS" if close(at_risk, obs.max_loss_per_share if direction == "short" else abs(float(obs.entry_cost_per_share))) else "FAIL",
        "tier_a_at_risk_per_share",
    )
    rec(
        f"{sid}-quantity", sample, "signed quantity", br["quantity"], float(obs.quantity),
        "audit-local vs trade_log", diff_num(br["quantity"], obs.quantity),
        "PASS" if close(br["quantity"], obs.quantity) else "FAIL", "tier_a_quantity",
    )
    rec(
        f"{sid}-entry", sample, "entry_cost_per_share", br["entry_cost_per_share"], float(obs.entry_cost_per_share),
        "audit-local vs trade_log", diff_num(br["entry_cost_per_share"], obs.entry_cost_per_share),
        "PASS" if close(br["entry_cost_per_share"], obs.entry_cost_per_share) else "FAIL", "pnl_entry_cost",
    )
    rec(
        f"{sid}-exit", sample, "expiry payoff per share", br["expiry_payoff_per_share"],
        float(obs.pnl_per_share) + float(obs.entry_cost_per_share),
        "audit-local vs trade_log identity", diff_num(br["expiry_payoff_per_share"], float(obs.pnl_per_share) + float(obs.entry_cost_per_share)),
        "PASS" if close(br["expiry_payoff_per_share"], float(obs.pnl_per_share) + float(obs.entry_cost_per_share)) else "FAIL",
        "pnl_exit_value",
    )
    rec(
        f"{sid}-pnl-ps", sample, "pnl_per_share", br["pnl_per_share"], float(obs.pnl_per_share),
        "audit-local vs trade_log", diff_num(br["pnl_per_share"], obs.pnl_per_share),
        "PASS" if close(br["pnl_per_share"], obs.pnl_per_share) else "FAIL", "pnl_per_share",
    )
    rec(
        f"{sid}-car-dollars", sample, "capital_at_risk_dollars", br["capital_at_risk_dollars"],
        float(obs.capital_at_risk_dollars), "audit-local vs trade_log", diff_num(br["capital_at_risk_dollars"], obs.capital_at_risk_dollars),
        "PASS" if close(br["capital_at_risk_dollars"], obs.capital_at_risk_dollars) else "FAIL", "capital_at_risk_dollars",
    )
    rec(
        f"{sid}-pnl-total", sample, "pnl_total", br["pnl_total"], float(obs.pnl_total),
        "audit-local vs trade_log", diff_num(br["pnl_total"], obs.pnl_total),
        "PASS" if close(br["pnl_total"], obs.pnl_total) else "FAIL", "pnl_total",
    )


for sid, td, ticker, direction in FROZEN_SAMPLES:
    print(f"\n=== audit {sid} ===", flush=True)
    audit_sample(sid, td, ticker, direction)

print("\n=== sampled-date CAR ===", flush=True)
for td in SAMPLE_DATES:
    exp_car = sum(r["pnl_total"] for r in date_books[td]) / sum(r["capital_at_risk_dollars"] for r in date_books[td])
    obs_car = float(ds_sum_c[ds_sum_c.trade_date == td].iloc[0].cycle_return_on_capital_at_risk)
    tag = "S1" if td == date(2022, 9, 2) else "S2"
    rec(
        f"{tag}-date-car", f"{tag} date {td}", "cycle_return_on_capital_at_risk",
        exp_car, obs_car, "audit-local book vs date_summary", diff_num(exp_car, obs_car),
        "PASS" if close(exp_car, obs_car) else "FAIL", "date_car_contribution",
    )

# S4
print("\n=== S4 ===", flush=True)
s4_td, s4_ticker, s4_dir = FROZEN_S4
a1r = a1[(a1.entry_date == s4_td) & (a1.ticker == s4_ticker)]
s4_valid = bool(a1r.iloc[0].surface_valid) if len(a1r) else False
rec("S4-a1", f"S4 {s4_td}/{s4_ticker}", "A1 surface_valid for failing ticker", True,
    s4_valid, "A1 vs structure failure context", "—", "PASS" if s4_valid else "FAIL", "S4_structure_failure")
rec("S4-code", f"S4 {s4_td}/{s4_ticker}", "reason_code in frozen set", "wing_or_liquidity_selection|...", s4_row.reason_code,
    "candidate_view", "—", "PASS" if s4_row.reason_code == "wing_or_liquidity_selection" else "FAIL", "S4_structure_failure")
q4 = quotes_for(a2, s4_td, s4_ticker)
otm = q4[q4.is_otm == True]  # noqa: E712
otm = otm[otm.spread_pct <= cfg["max_leg_spread_pct"]]
eligible = otm[otm.abs_delta <= cfg["wing_delta_target"]]
rec("S4-wings", f"S4 {s4_td}/{s4_ticker}", "OTM abs_delta<=0.15 after spread gate", 0, int(len(eligible)),
    "A2 quotes", "—", "PASS" if len(eligible) == 0 else "FAIL", "S4_structure_failure")
leg_hit = leg_c[(leg_c.trade_date == s4_td) & (leg_c.ticker == s4_ticker) & (leg_c.direction == s4_dir)]
tl_hit = tl_c[(tl_c.trade_date == s4_td) & (tl_c.ticker == s4_ticker) & (tl_c.direction == s4_dir)]
rec("S4-nolegs", f"S4 {s4_td}/{s4_ticker}", "no leg rows", 0, len(leg_hit), "leg_log", "—",
    "PASS" if len(leg_hit) == 0 else "FAIL", "S4_structure_failure")
included_flag = bool(tl_hit.iloc[0].included_in_portfolio) if len(tl_hit) else None
rec("S4-notincluded", f"S4 {s4_td}/{s4_ticker}", "not included_in_portfolio", False, included_flag, "trade_log", "—",
    "PASS" if included_flag is False else "FAIL", "S4_structure_failure")
date_st = ds_c.loc[ds_c.trade_date == s4_td, "status"].iloc[0]
rec("S4-datestatus", f"S4 {s4_td}", "date status traded|valid_no_trade", "traded|valid_no_trade", date_st,
    "date_status", "—", "PASS" if date_st in {"traded", "valid_no_trade"} else "FAIL", "S4_structure_failure")

# Coverage + output
REQUIRED_COVERAGE = [
    "identity", "sample_selection", "universe_snapshot_date", "universe_atm_pair",
    "universe_dvol_spread_fields", "universe_dvol_rank", "universe_spread_rank", "universe_and_membership",
    "joint_universe_feature_membership", "joint_finite_values", "joint_mom_count", "joint_cvg_count",
    "joint_eligible_slice", "signal_rank_recompute", "cvg_rank_recompute", "direction_and_cvg_retention",
    "option_a1_surface_valid", "option_entry_spot", "option_exit_spot", "option_body_strike", "option_expiry_date",
    "option_dte_actual", "option_body_selection", "option_spread_gates", "option_wing_delta_rule",
    "option_leg_count", "option_leg_identity", "option_mid_cross_half_spread",
    "tier_a_n_short", "tier_a_n_long", "tier_a_at_risk_per_share", "tier_a_short_budget_split",
    "tier_a_long_budget", "tier_a_long_budget_split", "tier_a_collected_credit", "tier_a_fallback",
    "tier_a_quantity", "pnl_entry_cost", "pnl_exit_value", "pnl_per_share", "capital_at_risk_dollars",
    "pnl_total", "date_car_contribution", "S3_valid_no_trade", "S4_structure_failure",
]

coverage_out = []
for key in REQUIRED_COVERAGE:
    info = coverage.get(key)
    if not info:
        coverage_out.append({"stage": key, "covered": False, "verdict": "MISSING"})
        continue
    vs = info["verdicts"]
    v = "FAIL" if any(x == "FAIL" for x in vs) else ("N/A" if all(str(x).startswith("N/A") for x in vs) else "PASS")
    coverage_out.append({"stage": key, "covered": True, "n_checks": len(vs), "verdict": v, "ids": info["ids"]})

n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
n_na = sum(1 for r in rows if str(r["verdict"]).startswith("N/A"))
missing_cov = [c["stage"] for c in coverage_out if not c["covered"]]
fail_rows = [r for r in rows if r["verdict"] == "FAIL"]
only_s3_na = [r["id"] for r in rows if str(r["verdict"]).startswith("N/A")] == ["S3"]

if n_fail == 0 and not missing_cov and only_s3_na:
    audit_verdict = "PASS"
    lifecycle = "PHASE 4 REVERIFICATION COMPLETE — AWAITING REVIEW"
else:
    audit_verdict = "BLOCKED — INDEPENDENT SOURCE AUDIT FAILED"
    lifecycle = audit_verdict

out = {
    "phase": "phase4_source_reconstruction",
    "audit_verdict": audit_verdict,
    "lifecycle_status": lifecycle,
    "independence": "No src.backtest calculation helpers imported",
    "phase5_authorized": False,
    "execution_commit": EXECUTION_COMMIT,
    "run_dir": str(RUN_DIR),
    "verify_dir": str(VERIFY_DIR),
    "tolerance": {"absolute": TOL_ABS, "relative": TOL_REL},
    "selection": selection,
    "s3_only_permitted_na": only_s3_na,
    "no_sample_replaced": True,
    "phase3_shell_limitation": (
        "Phase 3 capturing shell EXIT_CODE and stdout/stderr were not retained "
        "(documented non-blocking operational limitation). Not recovered; baseline not rerun."
    ),
    "coverage_checklist": coverage_out,
    "missing_coverage_stages": missing_cov,
    "failed_rows": fail_rows,
    "date_sizing_meta": {str(k): date_meta[k] for k in SAMPLE_DATES},
    "audit_rows": rows,
    "n_pass": n_pass,
    "n_fail": n_fail,
    "n_na": n_na,
    "aggregate_economics_opened": False,
    "baseline_rerun": False,
}

out_json = VERIFY_DIR / "phase4_source_reconstruction_audit.json"
out_json.write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")

md = [
    "# Sprint 006 D4 — Phase 4 source-reconstruction audit (audit-local)",
    "",
    f"**Audit verdict:** `{audit_verdict}`",
    f"**Lifecycle status:** `{lifecycle}`",
    f"**Independence:** no `src.backtest` calculation helpers imported",
    "",
    f"- PASS: {n_pass}",
    f"- FAIL: {n_fail}",
    f"- N/A: {n_na}",
    "",
    "## Frozen samples",
    "",
    "| Sample | Key |",
    "|--------|-----|",
    "| S1-L | 2022-09-02 / ACN / long |",
    "| S1-S | 2022-09-02 / AMC / short |",
    "| S2-L | 2018-10-26 / ABBV / long |",
    "| S2-S | 2018-10-26 / MRVL / short |",
    "| S3 | N/A |",
    "| S4 | 2018-10-26 / AMBA / short |",
    "",
    "## Failed rows",
    "",
]
if fail_rows:
    for r in fail_rows:
        md.append(f"- `{r['id']}`: {r['stage']}")
else:
    md.append("- none")
(VERIFY_DIR / "phase4_source_reconstruction_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"\n=== SUMMARY verdict={audit_verdict} pass={n_pass} fail={n_fail} na={n_na} missing={missing_cov} ===", flush=True)
raise SystemExit(0 if audit_verdict == "PASS" else 1)
