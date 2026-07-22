"""
Build weekly point-in-time ticker liquidity panel for step1_get_universe().

Pipeline (Sprint 004 C4, plus security-type stage):
  daily raw observations → candidate ticker/date discovery → security
  dictionary update (date-specific ORATS Core assetType observation) →
  company-equity daily filter → weekly liquidity observations → rolling
  N-week panel → liquid_tickers.csv

Non-company securities (ETF / index / VIX-style, ORATS Core assetType 4-9)
are removed from daily data before any weekly / panel / rank construction.
Each missing ticker is classified from one Core observation at its latest
observed trade date (bounded valid-empty fallback to earlier observed dates);
the persistent dictionary (see src/data/security_types.py) means Core is
queried at most once per ticker across backfills. Each build also writes a
snapshot-local security_classification.parquet next to the other artifacts.

Reads ORATS **raw** daily ZIPs from ORATS_Data (no split-adjusted cache required).
Liquidity uses raw bid/ask/volume columns only; downstream surface/backtest still
uses ORATS_Adjusted after scoped split adjustment on liquid names.

See docs/tmp/c4_liquidity_panel_design_plan.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.input_snapshot import generate_build_id  # noqa: E402
from src.data.orats_core_client import OratsCoreClient, OratsCoreError  # noqa: E402
from src.data.paths import DEFAULT_SECURITY_TYPES_PATH  # noqa: E402
from src.data.security_types import (  # noqa: E402
    SecurityTypesError,
    classification_digest,
    company_equity_tickers,
    ensure_security_types,
    snapshot_classification,
)

DEFAULT_DATA_ROOT = Path("C:/ORATS/data/ORATS_Data")
DEFAULT_CACHE_DIR = Path("C:/MomentumCVG_env/cache")
DEFAULT_START_YEAR = 2017
DEFAULT_END_YEAR = 2026
DEFAULT_LOOKBACK_WEEKS = 12
DEFAULT_MIN_VALID_QUOTE_WEEKS = 3
DEFAULT_DTE_MIN = 5
DEFAULT_DTE_MAX = 60
DEFAULT_DVOL_TOP_PCT = 0.20
DEFAULT_SPREAD_BOT_PCT = 1.0

DAILY_FILENAME = "ticker_liquidity_daily_observations.parquet"
WEEKLY_FILENAME = "ticker_liquidity_weekly_observations.parquet"
PANEL_FILENAME = "ticker_liquidity_panel.parquet"
LIQUID_TICKERS_FILENAME = "liquid_tickers.csv"
SECURITY_CLASSIFICATION_FILENAME = "security_classification.parquet"
STAGING_DIRNAME = ".liquidity_panel_staging"
LIQUIDITY_SOURCE = "raw_option_bid_x_volume_sum_dte_5_60"

RAW_REQUIRED_COLS = (
    "ticker",
    "expirDate",
    "stkPx",
    "strike",
    "cBidPx",
    "cAskPx",
    "pBidPx",
    "pAskPx",
    "cVolu",
    "pVolu",
)

DAILY_REQUIRED_COLS = (
    "trade_date",
    "ticker",
    "daily_atm_straddle_dollar_vol",
    "daily_atm_spread_pct",
    "daily_has_valid_quote",
    "n_candidate_expiries",
    "n_expiries_total",
    "no_expiry_in_band",
    "liquidity_source",
)
WEEKLY_REQUIRED_COLS = (
    "week_end_date",
    "ticker",
    "weekly_atm_straddle_dollar_vol",
    "weekly_atm_spread_pct",
    "weekly_valid_quote_days",
    "weekly_has_valid_quote",
)
PANEL_STEP1_COLS = (
    "month_date",
    "ticker",
    "atm_straddle_dollar_vol",
    "atm_spread_pct",
    "has_valid_atm_pair",
)
PANEL_BUILD_PARAM_COLS = (
    "lookback_weeks",
    "min_valid_quote_weeks",
    "dte_min",
    "dte_max",
    "dvol_top_pct",
    "spread_bot_pct",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LiquidityPanelError(Exception):
    """Blocking failure building or extending the liquidity panel."""


@dataclass
class BuildResult:
    daily: pd.DataFrame
    weekly: pd.DataFrame
    panel: pd.DataFrame
    files_read: int
    warnings: list[str] = field(default_factory=list)
    mode: str = "backfill"
    classification: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class IncrementalState:
    daily_watermark: date
    weekly_watermark: date
    panel_watermark: date
    daily: pd.DataFrame
    weekly: pd.DataFrame
    panel: pd.DataFrame
    lookback_weeks: int
    min_valid_quote_weeks: int
    dte_min: int
    dte_max: int


# ── ORATS raw ZIP I/O ─────────────────────────────────────────────────────────


def orats_raw_zip_path(data_root: Path | str, day: date) -> Path:
    """Path to one ORATS raw daily SMV ZIP (same layout as apply_split_adjustment raw_root)."""
    if isinstance(day, datetime) or not isinstance(day, date):
        raise ValueError(f"Expected date, got {day!r}")
    root = Path(data_root)
    return root / f"{day.year:04d}" / f"ORATS_SMV_Strikes_{day.strftime('%Y%m%d')}.zip"


def load_raw_day_from_zip(data_root: Path | str, trade_date: date) -> pd.DataFrame:
    """Load one day's wide-format ORATS chain from raw ZIP; no split adjustment."""
    zip_path = orats_raw_zip_path(data_root, trade_date)
    if not zip_path.is_file():
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith((".csv", ".txt"))]
            if not csv_names:
                raise LiquidityPanelError(f"No CSV/TXT inside {zip_path.name}")
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f, dtype={"ticker": str})
    except LiquidityPanelError:
        raise
    except Exception as exc:
        raise LiquidityPanelError(f"Failed to read {zip_path}: {exc}") from exc


def make_raw_zip_loader(data_root: Path) -> Callable[[date], pd.DataFrame]:
    """Return a load_day_fn that reads ORATS_Data ZIPs (one read per trade_date)."""

    def _load(trade_date: date) -> pd.DataFrame:
        return load_raw_day_from_zip(data_root, trade_date)

    return _load


# ── date discovery ────────────────────────────────────────────────────────────


def discover_orats_trading_dates(
    data_root: Path,
    start_date: date,
    end_date: date,
) -> list[date]:
    """List ORATS raw ZIP trade dates in [start_date, end_date], sorted."""
    dates: list[date] = []
    for fp in sorted(data_root.glob("*/ORATS_SMV_Strikes_*.zip")):
        try:
            d = datetime.strptime(fp.stem.split("_")[-1], "%Y%m%d").date()
        except ValueError:
            continue
        if start_date <= d <= end_date:
            dates.append(d)
    dates.sort()
    return dates


def build_week_calendar(trading_dates: Sequence[date]) -> list[tuple[date, list[date]]]:
    """Map each ISO week to (week_end_date, ORATS trading days in that week)."""
    sorted_dates = sorted(set(trading_dates))
    week_days: dict[date, set[date]] = {}
    for d in sorted_dates:
        monday = d - timedelta(days=d.weekday())
        sunday = monday + timedelta(days=6)
        days_in_week = [x for x in sorted_dates if monday <= x <= sunday]
        if not days_in_week:
            continue
        week_end = max(days_in_week)
        week_days.setdefault(week_end, set()).update(days_in_week)
    return sorted((we, sorted(days)) for we, days in week_days.items())


def resolve_weekly_snapshot_dates(
    trading_dates: Sequence[date],
    start_date: date,
    end_date: date,
) -> list[date]:
    """Completed-week snapshot dates (week_end_date) within the output range."""
    return [
        week_end
        for week_end, _ in build_week_calendar(trading_dates)
        if start_date <= week_end <= end_date
    ]


def compute_dates_needed(
    trading_dates: Sequence[date],
    snapshot_dates: Sequence[date],
    lookback_weeks: int,
) -> list[date]:
    """ORATS days required to build panel snapshots (union of rolling windows)."""
    calendar = build_week_calendar(trading_dates)
    week_end_to_days = dict(calendar)
    all_week_ends = sorted(week_end_to_days.keys())
    needed: set[date] = set()
    for snap in snapshot_dates:
        weeks_le = [w for w in all_week_ends if w <= snap]
        window = weeks_le[-lookback_weeks:] if weeks_le else []
        for w in window:
            needed.update(week_end_to_days[w])
    return sorted(needed)


# ── Stage 1: daily ────────────────────────────────────────────────────────────


def _valid_leg_quote(bid: float, ask: float) -> bool:
    return bool(np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid)


def _leg_spread_pct(bid: float, ask: float) -> float:
    if not _valid_leg_quote(bid, ask):
        return float("nan")
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def validate_raw_columns(day_df: pd.DataFrame) -> None:
    missing = [c for c in RAW_REQUIRED_COLS if c not in day_df.columns]
    if missing:
        raise LiquidityPanelError(
            f"ORATS raw ZIP missing columns required for liquidity: {missing}. "
            "Expected native ORATS wide-format columns (stkPx, cBidPx, …); "
            "do not use adj_* or ORATS_Adjusted as input."
        )


def candidate_expiries(
    expiries: Sequence[date],
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> list[date]:
    return sorted(
        e for e in expiries if dte_min <= (e - trade_date).days <= dte_max
    )


def select_atm_row(g_exp: pd.DataFrame) -> pd.Series:
    """ATM row using raw stkPx and raw strike."""
    g = g_exp.copy()
    g["_dist"] = (g["strike"] - g["stkPx"]).abs()
    return g.sort_values(["_dist", "strike"], kind="mergesort").iloc[0]


def compute_expiry_atm_liquidity(atm: pd.Series) -> tuple[float, float]:
    """
    Returns (expiry_atm_straddle_dollar_vol, expiry_atm_spread_pct).
    Uses raw bid × volume when both call and put quotes are valid (bid/ask > 0, ask >= bid).
    """
    c_bid = float(atm["cBidPx"])
    p_bid = float(atm["pBidPx"])
    c_ask = float(atm["cAskPx"])
    p_ask = float(atm["pAskPx"])
    c_vol = 0.0 if pd.isna(atm["cVolu"]) else float(atm["cVolu"])
    p_vol = 0.0 if pd.isna(atm["pVolu"]) else float(atm["pVolu"])

    if not (_valid_leg_quote(c_bid, c_ask) and _valid_leg_quote(p_bid, p_ask)):
        return 0.0, float("nan")

    call_bid_dollar = 100.0 * c_bid * c_vol
    put_bid_dollar = 100.0 * p_bid * p_vol
    expiry_vol = min(call_bid_dollar, put_bid_dollar)

    call_sp = _leg_spread_pct(c_bid, c_ask)
    put_sp = _leg_spread_pct(p_bid, p_ask)
    expiry_spread = float(max(call_sp, put_sp))

    return expiry_vol, expiry_spread


def compute_ticker_daily_observation(
    g_all: pd.DataFrame,
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> dict:
    expiries = sorted(g_all["expirDate"].dropna().unique())
    n_total = len(expiries)
    candidates = candidate_expiries(expiries, trade_date, dte_min=dte_min, dte_max=dte_max)

    if not candidates:
        return {
            "daily_atm_straddle_dollar_vol": 0.0,
            "daily_atm_spread_pct": np.nan,
            "daily_has_valid_quote": False,
            "n_candidate_expiries": 0,
            "n_expiries_total": n_total,
            "no_expiry_in_band": True,
            "liquidity_source": LIQUIDITY_SOURCE,
        }

    total_vol = 0.0
    spread_num = 0.0
    spread_den = 0.0
    for expiry in candidates:
        g_exp = g_all[g_all["expirDate"] == expiry]
        if g_exp.empty:
            continue
        atm = select_atm_row(g_exp)
        exp_vol, exp_spread = compute_expiry_atm_liquidity(atm)
        total_vol += exp_vol
        if exp_vol > 0 and np.isfinite(exp_spread):
            spread_num += exp_spread * exp_vol
            spread_den += exp_vol

    daily_spread = spread_num / spread_den if spread_den > 0 else float("nan")
    has_valid = total_vol > 0 and np.isfinite(daily_spread)

    return {
        "daily_atm_straddle_dollar_vol": total_vol,
        "daily_atm_spread_pct": daily_spread,
        "daily_has_valid_quote": has_valid,
        "n_candidate_expiries": len(candidates),
        "n_expiries_total": n_total,
        "no_expiry_in_band": False,
        "liquidity_source": LIQUIDITY_SOURCE,
    }


def compute_daily_liquidity_observations(
    day_df: pd.DataFrame,
    trade_date: date,
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> pd.DataFrame:
    validate_raw_columns(day_df)
    df = day_df.copy()
    df["expirDate"] = pd.to_datetime(df["expirDate"]).dt.date

    records: list[dict] = []
    for ticker, g_all in df.groupby("ticker", sort=False):
        obs = compute_ticker_daily_observation(
            g_all, trade_date, dte_min=dte_min, dte_max=dte_max
        )
        records.append({"trade_date": trade_date, "ticker": ticker, **obs})

    if not records:
        return pd.DataFrame(columns=list(DAILY_REQUIRED_COLS))
    return pd.DataFrame(records)


def extract_daily_observations(
    dates: Sequence[date],
    load_day_fn: Callable[[date], pd.DataFrame],
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Read each trade_date once via load_day_fn; return slim daily observations."""
    frames: list[pd.DataFrame] = []
    sorted_dates = sorted(dates)
    day_iter = tqdm(
        sorted_dates,
        desc="Daily liquidity ZIPs",
        unit="day",
        disable=not show_progress or not sys.stderr.isatty(),
    )
    for trade_date in day_iter:
        day_iter.set_postfix_str(trade_date.isoformat(), refresh=False)
        day_df = load_day_fn(trade_date)
        if day_df is None or day_df.empty:
            continue
        frames.append(
            compute_daily_liquidity_observations(
                day_df, trade_date, dte_min=dte_min, dte_max=dte_max
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(DAILY_REQUIRED_COLS)), len(dates)
    return pd.concat(frames, ignore_index=True), len(dates)


# ── Stage 1b: security classification (company-equity filter) ────────────────


def candidate_ticker_dates_from_daily(daily: pd.DataFrame) -> dict[str, list[date]]:
    """Map each normalized candidate ticker to its distinct observed trade
    dates, sorted newest first.

    Deterministic regardless of daily row order: tickers are sorted
    alphabetically and dates are de-duplicated and sorted descending.
    """
    frame = daily[["ticker", "trade_date"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["ticker"] != ""]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date

    out: dict[str, list[date]] = {}
    for ticker, grp in frame.groupby("ticker", sort=True):
        out[str(ticker)] = sorted(set(grp["trade_date"]), reverse=True)
    return out


def filter_daily_to_tickers(df: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    """Keep only rows whose normalized ticker is in ``tickers``."""
    mask = df["ticker"].astype(str).str.strip().str.upper().isin(tickers)
    return df[mask].reset_index(drop=True)


def make_core_classifier(
    security_types_path: Path,
    fetch_observation_fn: Callable[[str, date], pd.DataFrame] | None = None,
    *,
    progress_path: Path | None = None,
) -> Callable[[dict[str, list[date]]], pd.DataFrame]:
    """Return a classify_fn backed by the persistent security-type dictionary.

    ``candidates`` maps each ticker to its observed trade dates (newest
    first). The shared dictionary at ``security_types_path`` is updated with
    one date-specific Core observation per missing ticker (latest observed
    date first, bounded valid-empty / Core-404 fallback). Tickers that remain
    unresolved are omitted from the dictionary and from the returned
    classification subset (not treated as company equity; retried when still
    missing on a later run). Existing dictionary tickers make no API request.
    Optional ``progress_path`` is the snapshot building root for
    ``run_progress.json`` updates during Core classification.
    """
    if fetch_observation_fn is None:
        fetch_observation_fn = OratsCoreClient().fetch_asset_type_at_date

    def _classify(candidates: dict[str, list[date]]) -> pd.DataFrame:
        dictionary = ensure_security_types(
            candidates,
            security_types_path,
            fetch_observation_fn=fetch_observation_fn,
            progress_path=progress_path,
        )
        covered = set(dictionary["ticker"])
        classified_keys = [
            ticker for ticker in candidates if str(ticker).strip().upper() in covered
        ]
        unresolved = sorted(
            {
                str(ticker).strip().upper()
                for ticker in candidates
            }
            - covered
        )
        if unresolved:
            logger.warning(
                "Excluding %d unresolved Core ticker(s) from equity filter "
                "(not in security-type dictionary): %s%s",
                len(unresolved),
                unresolved[:20],
                " ..." if len(unresolved) > 20 else "",
            )
        return snapshot_classification(dictionary, classified_keys)

    return _classify


# ── Stage 2: weekly ───────────────────────────────────────────────────────────


def aggregate_weekly_liquidity_observations(
    daily_obs: pd.DataFrame,
    week_calendar: Sequence[tuple[date, list[date]]],
) -> pd.DataFrame:
    if daily_obs.empty:
        return pd.DataFrame(columns=list(WEEKLY_REQUIRED_COLS))

    daily = daily_obs.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    lookup: dict[tuple[date, str], pd.Series] = {
        (row.trade_date, row.ticker): row
        for row in daily.itertuples(index=False)
    }

    records: list[dict] = []
    for week_end, orats_days in week_calendar:
        if not orats_days:
            continue
        tickers = daily.loc[daily["trade_date"].isin(orats_days), "ticker"].unique()
        for ticker in tickers:
            vols: list[float] = []
            spreads: list[float] = []
            valid_days = 0
            for d in orats_days:
                row = lookup.get((d, ticker))
                if row is None:
                    vols.append(0.0)
                else:
                    v = float(row.daily_atm_straddle_dollar_vol)
                    vols.append(v if np.isfinite(v) else 0.0)
                    if bool(row.daily_has_valid_quote) and np.isfinite(row.daily_atm_spread_pct):
                        spreads.append(float(row.daily_atm_spread_pct))
                        valid_days += 1
            weekly_vol = sum(vols) / len(orats_days)
            weekly_spread = float(np.mean(spreads)) if spreads else float("nan")
            records.append(
                {
                    "week_end_date": week_end,
                    "ticker": ticker,
                    "weekly_atm_straddle_dollar_vol": weekly_vol,
                    "weekly_atm_spread_pct": weekly_spread,
                    "weekly_valid_quote_days": valid_days,
                    "weekly_has_valid_quote": valid_days >= 1,
                }
            )

    if not records:
        return pd.DataFrame(columns=list(WEEKLY_REQUIRED_COLS))
    return pd.DataFrame(records)


# ── Stage 3: rolling panel ───────────────────────────────────────────────────


def _window_week_ends(all_week_ends: Sequence[date], snapshot: date, lookback_weeks: int) -> list[date]:
    eligible = [w for w in all_week_ends if w <= snapshot]
    return eligible[-lookback_weeks:] if eligible else []


def aggregate_rolling_weekly_panel(
    weekly_obs: pd.DataFrame,
    snapshot_dates: Sequence[date],
    all_week_ends: Sequence[date],
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_valid_quote_weeks: int = DEFAULT_MIN_VALID_QUOTE_WEEKS,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
) -> pd.DataFrame:
    if not snapshot_dates:
        return pd.DataFrame(columns=list(PANEL_STEP1_COLS))

    weekly = weekly_obs.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.date
    week_lookup: dict[tuple[date, str], pd.Series] = {
        (row.week_end_date, row.ticker): row
        for row in weekly.itertuples(index=False)
    }
    sorted_week_ends = sorted(set(all_week_ends))

    records: list[dict] = []
    for snap in sorted(snapshot_dates):
        window = _window_week_ends(sorted_week_ends, snap, lookback_weeks)
        window_shortfall = max(0, lookback_weeks - len(window))
        window_start = window[0] if window else snap

        tickers: set[str] = set()
        for w in window:
            tickers.update(weekly.loc[weekly["week_end_date"] == w, "ticker"].tolist())

        for ticker in sorted(tickers):
            vol_sum = 0.0
            spreads: list[float] = []
            valid_weeks = 0
            zero_vol_weeks = 0
            for w in window:
                row = week_lookup.get((w, ticker))
                if row is None:
                    zero_vol_weeks += 1
                    continue
                wvol = float(row.weekly_atm_straddle_dollar_vol)
                vol_sum += wvol if np.isfinite(wvol) else 0.0
                if wvol == 0 or not np.isfinite(wvol):
                    zero_vol_weeks += 1
                if bool(row.weekly_has_valid_quote) and np.isfinite(row.weekly_atm_spread_pct):
                    spreads.append(float(row.weekly_atm_spread_pct))
                    valid_weeks += 1

            atm_dvol = vol_sum / lookback_weeks
            atm_spread = float(np.mean(spreads)) if spreads else float("nan")
            has_valid = valid_weeks >= min_valid_quote_weeks

            records.append(
                {
                    "month_date": pd.Timestamp(snap),
                    "snapshot_date": pd.Timestamp(snap),
                    "ticker": ticker,
                    "atm_straddle_dollar_vol": atm_dvol,
                    "atm_spread_pct": atm_spread,
                    "has_valid_atm_pair": has_valid,
                    "lookback_weeks": lookback_weeks,
                    "min_valid_quote_weeks": min_valid_quote_weeks,
                    "dte_min": dte_min,
                    "dte_max": dte_max,
                    "valid_quote_weeks": valid_weeks,
                    "zero_volume_weeks": zero_vol_weeks,
                    "window_start_date": pd.Timestamp(window_start),
                    "window_end_date": pd.Timestamp(snap),
                    "window_shortfall": window_shortfall,
                    "liquidity_source": LIQUIDITY_SOURCE,
                }
            )

    return pd.DataFrame(records)


# ── incremental ───────────────────────────────────────────────────────────────


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise LiquidityPanelError(
            f"Missing required artifact: {path}. Run with --mode backfill first."
        )
    return pd.read_parquet(path)


def _assert_columns(df: pd.DataFrame, required: Sequence[str], artifact: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LiquidityPanelError(
            f"{artifact} schema mismatch — missing columns {missing}. Run --mode backfill."
        )

def _max_date(series: pd.Series, col: str) -> date:
    return pd.to_datetime(series).max().date()


def _assert_build_params_match(
    panel: pd.DataFrame,
    *,
    lookback_weeks: int,
    min_valid_quote_weeks: int,
    dte_min: int,
    dte_max: int,
    dvol_top_pct: float,
    spread_bot_pct: float,
) -> None:
    """Fail incremental if CLI build/universe params differ from values stored at backfill."""
    int_expected = {
        "lookback_weeks": lookback_weeks,
        "min_valid_quote_weeks": min_valid_quote_weeks,
        "dte_min": dte_min,
        "dte_max": dte_max,
    }
    float_expected = {
        "dvol_top_pct": float(dvol_top_pct),
        "spread_bot_pct": float(spread_bot_pct),
    }
    missing = [c for c in PANEL_BUILD_PARAM_COLS if c not in panel.columns]
    if missing:
        raise LiquidityPanelError(
            f"{PANEL_FILENAME} missing build-param columns {missing}. Run --mode backfill."
        )
    mismatches: list[str] = [
        f"{col}: artifact={int(panel[col].iloc[0])} CLI={val}"
        for col, val in int_expected.items()
        if int(panel[col].iloc[0]) != val
    ]
    mismatches.extend(
        f"{col}: artifact={float(panel[col].iloc[0])} CLI={val}"
        for col, val in float_expected.items()
        if float(panel[col].iloc[0]) != val
    )
    if mismatches:
        raise LiquidityPanelError(
            f"Build params mismatch ({'; '.join(mismatches)}). Run --mode backfill."
        )


def _assert_daily_liquidity_source(daily: pd.DataFrame) -> None:
    sources = daily["liquidity_source"].dropna().unique()
    if len(sources) != 1 or str(sources[0]) != LIQUIDITY_SOURCE:
        raise LiquidityPanelError(
            f"Daily liquidity_source {list(sources)} != expected {LIQUIDITY_SOURCE!r}. "
            "Run --mode backfill."
        )


def validate_incremental_artifacts(
    cache_dir: Path,
    *,
    lookback_weeks: int,
    min_valid_quote_weeks: int,
    dte_min: int,
    dte_max: int,
    dvol_top_pct: float,
    spread_bot_pct: float,
) -> IncrementalState:
    daily_path = cache_dir / DAILY_FILENAME
    weekly_path = cache_dir / WEEKLY_FILENAME
    panel_path = cache_dir / PANEL_FILENAME

    daily = _read_parquet(daily_path)
    weekly = _read_parquet(weekly_path)
    panel = _read_parquet(panel_path)

    _assert_columns(daily, DAILY_REQUIRED_COLS, DAILY_FILENAME)
    _assert_columns(weekly, WEEKLY_REQUIRED_COLS, WEEKLY_FILENAME)
    _assert_columns(panel, PANEL_STEP1_COLS, PANEL_FILENAME)

    if panel.empty:
        raise LiquidityPanelError("Panel artifact is empty; run --mode backfill.")

    _assert_daily_liquidity_source(daily)
    _assert_build_params_match(
        panel,
        lookback_weeks=lookback_weeks,
        min_valid_quote_weeks=min_valid_quote_weeks,
        dte_min=dte_min,
        dte_max=dte_max,
        dvol_top_pct=dvol_top_pct,
        spread_bot_pct=spread_bot_pct,
    )

    return IncrementalState(
        daily_watermark=_max_date(daily["trade_date"], "trade_date"),
        weekly_watermark=_max_date(weekly["week_end_date"], "week_end_date"),
        panel_watermark=_max_date(panel["month_date"], "month_date"),
        daily=daily,
        weekly=weekly,
        panel=panel,
        lookback_weeks=lookback_weeks,
        min_valid_quote_weeks=min_valid_quote_weeks,
        dte_min=dte_min,
        dte_max=dte_max,
    )


def run_incremental(
    data_root: Path,
    cache_dir: Path,
    state: IncrementalState,
    load_day_fn: Callable[[date], pd.DataFrame],
    all_trading_dates: Sequence[date],
    *,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
    show_progress: bool = True,
    classify_fn: Callable[[dict[str, list[date]]], pd.DataFrame] | None = None,
) -> BuildResult:
    calendar = build_week_calendar(all_trading_dates)
    new_weeks = [(we, days) for we, days in calendar if we > state.panel_watermark]
    if not new_weeks:
        raise LiquidityPanelError(
            "Nothing to append: no completed week with week_end_date > panel watermark. "
            "Wait for EOD or run backfill."
        )
    if len(new_weeks) > 1:
        raise LiquidityPanelError(
            f"Incremental expects one new week; found {len(new_weeks)} "
            f"({new_weeks[0][0]} … {new_weeks[-1][0]}). Run --mode backfill."
        )

    week_end, week_days = new_weeks[0]
    if week_end <= state.weekly_watermark:
        raise LiquidityPanelError("New week_end_date must be > weekly watermark.")

    new_daily_dates = [d for d in week_days if d > state.daily_watermark]
    if not new_daily_dates:
        raise LiquidityPanelError(
            "No ORATS days > daily watermark in new week; nothing to append."
        )

    for d in week_days:
        if d <= state.daily_watermark and d not in set(
            pd.to_datetime(state.daily["trade_date"]).dt.date
        ):
            raise LiquidityPanelError(
                f"Gap detected: ORATS day {d} <= daily watermark but missing from daily parquet. "
                "Run --mode backfill."
            )

    new_daily, files_read = extract_daily_observations(
        new_daily_dates, load_day_fn, dte_min=dte_min, dte_max=dte_max, show_progress=show_progress
    )
    if new_daily.empty:
        raise LiquidityPanelError("No daily observations extracted for incremental week.")

    new_daily_min = pd.to_datetime(new_daily["trade_date"]).min().date()
    if new_daily_min <= state.daily_watermark:
        raise LiquidityPanelError(
            "Incremental daily rows must be strictly after daily watermark; run backfill."
        )

    merged_daily = pd.concat([state.daily, new_daily], ignore_index=True)

    # Security-type stage: the candidate set is every ticker in the merged
    # daily data (prior + new). Prior artifacts are filtered too so
    # non-company tickers never survive into weekly / panel / rank outputs,
    # even when the prior artifacts predate the security-type filter.
    classification = pd.DataFrame()
    state_weekly = state.weekly
    state_panel = state.panel
    if classify_fn is not None:
        candidates = candidate_ticker_dates_from_daily(merged_daily)
        classification = classify_fn(candidates)
        equity = company_equity_tickers(classification)
        merged_daily = filter_daily_to_tickers(merged_daily, equity)
        new_daily = filter_daily_to_tickers(new_daily, equity)
        state_weekly = filter_daily_to_tickers(state.weekly, equity)
        state_panel = filter_daily_to_tickers(state.panel, equity)
        if new_daily.empty:
            raise LiquidityPanelError(
                "No company-equity daily rows remain in the incremental week "
                "after security-type filter."
            )

    new_weekly = aggregate_weekly_liquidity_observations(new_daily, [(week_end, week_days)])
    if new_weekly.empty:
        raise LiquidityPanelError("No weekly observations for incremental week.")

    merged_weekly = pd.concat([state_weekly, new_weekly], ignore_index=True)
    all_week_ends = sorted(
        set(pd.to_datetime(merged_weekly["week_end_date"]).dt.date)
    )

    new_panel = aggregate_rolling_weekly_panel(
        merged_weekly,
        [week_end],
        all_week_ends,
        lookback_weeks=state.lookback_weeks,
        min_valid_quote_weeks=state.min_valid_quote_weeks,
        dte_min=state.dte_min,
        dte_max=state.dte_max,
    )
    merged_panel = pd.concat([state_panel, new_panel], ignore_index=True)

    warnings: list[str] = []
    if int(new_panel["window_shortfall"].max()) > 0:
        warnings.append(f"Incremental snapshot {week_end} has window_shortfall > 0")

    return BuildResult(
        daily=merged_daily,
        weekly=merged_weekly,
        panel=merged_panel,
        files_read=files_read,
        warnings=warnings,
        mode="incremental",
        classification=classification,
    )


# ── backfill ──────────────────────────────────────────────────────────────────


def run_backfill(
    data_root: Path,
    start_date: date,
    end_date: date,
    load_day_fn: Callable[[date], pd.DataFrame],
    all_trading_dates: Sequence[date],
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_valid_quote_weeks: int = DEFAULT_MIN_VALID_QUOTE_WEEKS,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
    show_progress: bool = True,
    classify_fn: Callable[[dict[str, list[date]]], pd.DataFrame] | None = None,
) -> BuildResult:
    if not all_trading_dates:
        raise LiquidityPanelError(
            f"No ORATS raw ZIP files found under {data_root} for {start_date} → {end_date}."
        )

    snapshot_dates = resolve_weekly_snapshot_dates(all_trading_dates, start_date, end_date)
    if not snapshot_dates:
        raise LiquidityPanelError(
            f"No weekly snapshot dates in range {start_date} → {end_date}."
        )

    earliest_snap = min(snapshot_dates)
    hist_start = earliest_snap - timedelta(weeks=lookback_weeks + 1)
    hist_dates = [d for d in all_trading_dates if hist_start <= d <= end_date]
    dates_needed = compute_dates_needed(hist_dates, snapshot_dates, lookback_weeks)

    daily, files_read = extract_daily_observations(
        dates_needed,
        load_day_fn,
        dte_min=dte_min,
        dte_max=dte_max,
        show_progress=show_progress,
    )
    if daily.empty:
        raise LiquidityPanelError("Daily observation stage produced no rows.")

    # Security-type stage: discover candidates from raw daily metrics, update
    # the persistent dictionary, and filter daily rows to company equities
    # BEFORE any weekly / panel / rank construction.
    classification = pd.DataFrame()
    if classify_fn is not None:
        candidates = candidate_ticker_dates_from_daily(daily)
        classification = classify_fn(candidates)
        equity = company_equity_tickers(classification)
        daily = filter_daily_to_tickers(daily, equity)
        if daily.empty:
            raise LiquidityPanelError(
                "No company-equity daily rows remain after security-type filter."
            )

    week_calendar = build_week_calendar(dates_needed)
    weekly = aggregate_weekly_liquidity_observations(daily, week_calendar)
    all_week_ends = sorted({we for we, _ in week_calendar})
    panel = aggregate_rolling_weekly_panel(
        weekly,
        snapshot_dates,
        all_week_ends,
        lookback_weeks=lookback_weeks,
        min_valid_quote_weeks=min_valid_quote_weeks,
        dte_min=dte_min,
        dte_max=dte_max,
    )

    warnings: list[str] = []
    if not panel.empty and (panel["window_shortfall"] > 0).any():
        n = int((panel["window_shortfall"] > 0).sum())
        warnings.append(f"{n} panel snapshot(s) with window_shortfall > 0")

    return BuildResult(
        daily=daily,
        weekly=weekly,
        panel=panel,
        files_read=files_read,
        warnings=warnings,
        mode="backfill",
        classification=classification,
    )


# ── liquid tickers + report ───────────────────────────────────────────────────


LIQUID_TICKERS_COLUMNS = ("Ticker", "snapshots_qualified", "months_qualified")


def build_liquid_tickers(
    panel: pd.DataFrame,
    dvol_top_pct: float,
    spread_bot_pct: float,
) -> pd.DataFrame:
    valid = panel[panel["has_valid_atm_pair"]].copy()
    qual_counts: dict[str, int] = {}

    for _, grp in valid.groupby("month_date"):
        both_valid = grp[
            grp["atm_straddle_dollar_vol"].notna() & grp["atm_spread_pct"].notna()
        ]
        if len(both_valid) < 5:
            continue
        dvol_thresh = both_valid["atm_straddle_dollar_vol"].quantile(1.0 - dvol_top_pct)
        sp_thresh = both_valid["atm_spread_pct"].quantile(spread_bot_pct)
        qualifiers = both_valid.loc[
            (both_valid["atm_straddle_dollar_vol"] >= dvol_thresh)
            & (both_valid["atm_spread_pct"] <= sp_thresh),
            "ticker",
        ]
        for t in qualifiers:
            qual_counts[t] = qual_counts.get(t, 0) + 1

    if not qual_counts:
        return pd.DataFrame(columns=list(LIQUID_TICKERS_COLUMNS))

    qual_df = (
        pd.Series(qual_counts, name="snapshots_qualified")
        .sort_values(ascending=False)
        .rename_axis("ticker")
        .reset_index()
    )
    out = (
        qual_df.rename(columns={"ticker": "Ticker"})[["Ticker", "snapshots_qualified"]]
        .sort_values("Ticker")
        .reset_index(drop=True)
    )
    out["months_qualified"] = out["snapshots_qualified"]
    return out[list(LIQUID_TICKERS_COLUMNS)]


def write_liquidity_panel_report(
    path: Path,
    *,
    build_id: str,
    mode: str,
    data_root: Path,
    cache_dir: Path,
    start_date: date,
    end_date: date,
    lookback_weeks: int,
    min_valid_quote_weeks: int,
    dte_min: int,
    dte_max: int,
    dvol_top_pct: float,
    spread_bot_pct: float,
    result: BuildResult,
    liquidity_source: str = LIQUIDITY_SOURCE,
    watermarks_before: dict[str, str] | None = None,
    watermarks_after: dict[str, str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = result.panel
    daily = result.daily

    status = "PASS"
    if result.warnings:
        status = "WARN"
    if panel.empty:
        status = "FAIL"

    no_expiry = daily[daily.get("no_expiry_in_band", False) == True]  # noqa: E712
    valid_pct = (
        float(panel["has_valid_atm_pair"].mean()) if not panel.empty else 0.0
    )

    lines = [
        "# Liquidity panel report",
        "",
        f"- build_id: `{build_id}`",
        f"- mode: `{mode}`",
        f"- date range: {start_date} → {end_date}",
        f"- status: **{status}**",
        "",
        "## Inputs",
        f"- data_root: `{data_root}`",
        f"- cache_dir: `{cache_dir}`",
        f"- lookback_weeks: {lookback_weeks}",
        f"- min_valid_quote_weeks: {min_valid_quote_weeks}",
        f"- dte_min: {dte_min}",
        f"- dte_max: {dte_max}",
        f"- dvol_top_pct: {dvol_top_pct}",
        f"- spread_bot_pct: {spread_bot_pct}",
        f"- liquidity_source: `{liquidity_source}`",
        "",
        "## Execution",
        f"- ORATS raw ZIP files read: {result.files_read}",
        f"- daily rows: {len(daily)}",
        f"- weekly rows: {len(result.weekly)}",
        f"- panel rows: {len(panel)}",
    ]
    if watermarks_before and watermarks_after:
        lines.extend(
            [
                f"- watermarks before: {watermarks_before}",
                f"- watermarks after: {watermarks_after}",
            ]
        )
    lines.extend(
        [
            "",
            "## Coverage",
            f"- snapshots: {panel['month_date'].nunique() if not panel.empty else 0}",
            f"- tickers: {panel['ticker'].nunique() if not panel.empty else 0}",
            f"- has_valid_atm_pair rate: {valid_pct:.1%}",
        ]
    )
    if not result.classification.empty:
        cls = result.classification
        n_company = int((cls["classification"] == "company_equity").sum())
        lines.extend(
            [
                "",
                "## Security classification",
                f"- candidate tickers: {len(cls)}",
                f"- company_equity: {n_company}",
                f"- non_company_equity: {len(cls) - n_company}",
                f"- classification_digest: `{classification_digest(cls)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Debug",
            f"- no_expiry_in_band daily rows: {len(no_expiry)}",
        ]
    )
    if not no_expiry.empty:
        sample = no_expiry.head(5)[["trade_date", "ticker"]].to_string(index=False)
        lines.append(f"- sample:\n```\n{sample}\n```")
    if result.warnings:
        lines.extend(["", "## Warnings", *[f"- {w}" for w in result.warnings]])
    lines.extend(["", "## Summary", f"Liquidity panel build {status.lower()}."])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_artifacts(
    cache_dir: Path,
    result: BuildResult,
    *,
    liquid_tickers: pd.DataFrame,
) -> None:
    """Write panel artifacts atomically via staging dir (all files, then replace)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = cache_dir / STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    names = [DAILY_FILENAME, WEEKLY_FILENAME, PANEL_FILENAME, LIQUID_TICKERS_FILENAME]
    try:
        result.daily.to_parquet(staging / DAILY_FILENAME, index=False)
        result.weekly.to_parquet(staging / WEEKLY_FILENAME, index=False)
        result.panel.to_parquet(staging / PANEL_FILENAME, index=False)
        liquid_tickers.to_csv(staging / LIQUID_TICKERS_FILENAME, index=False)
        if not result.classification.empty:
            result.classification.to_parquet(
                staging / SECURITY_CLASSIFICATION_FILENAME, index=False
            )
            names.append(SECURITY_CLASSIFICATION_FILENAME)

        for name in names:
            os.replace(staging / name, cache_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stamp_panel_universe_params(
    panel: pd.DataFrame,
    *,
    dvol_top_pct: float,
    spread_bot_pct: float,
) -> pd.DataFrame:
    """Persist liquid_tickers thresholds on panel rows for incremental param checks."""
    out = panel.copy()
    out["dvol_top_pct"] = float(dvol_top_pct)
    out["spread_bot_pct"] = float(spread_bot_pct)
    return out


def build_panel(
    data_root: Path,
    cache_dir: Path,
    start_date: date,
    end_date: date,
    *,
    mode: Literal["backfill", "incremental"] = "backfill",
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_valid_quote_weeks: int = DEFAULT_MIN_VALID_QUOTE_WEEKS,
    dte_min: int = DEFAULT_DTE_MIN,
    dte_max: int = DEFAULT_DTE_MAX,
    dvol_top_pct: float = DEFAULT_DVOL_TOP_PCT,
    spread_bot_pct: float = DEFAULT_SPREAD_BOT_PCT,
    load_day_fn: Callable[[date], pd.DataFrame] | None = None,
    show_progress: bool = True,
    security_types_path: Path = DEFAULT_SECURITY_TYPES_PATH,
    fetch_observation_fn: Callable[[str, date], pd.DataFrame] | None = None,
) -> BuildResult:
    if load_day_fn is None:
        load_day_fn = make_raw_zip_loader(data_root)
    classify_fn = make_core_classifier(
        Path(security_types_path), fetch_observation_fn=fetch_observation_fn
    )

    hist_start = start_date - timedelta(weeks=lookback_weeks + 2)
    all_trading = discover_orats_trading_dates(data_root, hist_start, end_date)

    if mode == "backfill":
        return run_backfill(
            data_root,
            start_date,
            end_date,
            load_day_fn,
            all_trading,
            lookback_weeks=lookback_weeks,
            min_valid_quote_weeks=min_valid_quote_weeks,
            dte_min=dte_min,
            dte_max=dte_max,
            show_progress=show_progress,
            classify_fn=classify_fn,
        )

    state = validate_incremental_artifacts(
        cache_dir,
        lookback_weeks=lookback_weeks,
        min_valid_quote_weeks=min_valid_quote_weeks,
        dte_min=dte_min,
        dte_max=dte_max,
        dvol_top_pct=dvol_top_pct,
        spread_bot_pct=spread_bot_pct,
    )
    return run_incremental(
        data_root,
        cache_dir,
        state,
        load_day_fn,
        all_trading,
        dte_min=dte_min,
        dte_max=dte_max,
        show_progress=show_progress,
        classify_fn=classify_fn,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build weekly PIT ticker liquidity panel (Sprint 004 C4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="ORATS raw ZIP root (ORATS_Data)",
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--mode", choices=("backfill", "incremental"), default="backfill")
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--lookback-weeks", type=int, default=DEFAULT_LOOKBACK_WEEKS)
    p.add_argument("--min-valid-quote-weeks", type=int, default=DEFAULT_MIN_VALID_QUOTE_WEEKS)
    p.add_argument("--dte-min", type=int, default=DEFAULT_DTE_MIN)
    p.add_argument("--dte-max", type=int, default=DEFAULT_DTE_MAX)
    p.add_argument("--dvol-top-pct", type=float, default=DEFAULT_DVOL_TOP_PCT)
    p.add_argument("--spread-bot-pct", type=float, default=DEFAULT_SPREAD_BOT_PCT)
    p.add_argument(
        "--security-types-path",
        type=Path,
        default=DEFAULT_SECURITY_TYPES_PATH,
        help=(
            "Persistent ORATS security-type dictionary parquet. One "
            "date-specific ORATS Core observation (ORATS_API_TOKEN env var) "
            "is fetched per candidate ticker absent from this dictionary."
        ),
    )
    p.add_argument("--build-id", type=str, default=None)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar on daily ZIP processing",
    )
    return p.parse_args(argv)


def _resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.start_date and args.end_date:
        return date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    return date(args.start_year, 1, 1), date(args.end_year, 12, 31)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start_date, end_date = _resolve_date_range(args)
    build_id = args.build_id or generate_build_id(
        now=datetime.utcnow(), command="build_liquidity_panel"
    )

    try:
        result = build_panel(
            args.data_root,
            args.cache_dir,
            start_date,
            end_date,
            mode=args.mode,
            lookback_weeks=args.lookback_weeks,
            min_valid_quote_weeks=args.min_valid_quote_weeks,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            dvol_top_pct=args.dvol_top_pct,
            spread_bot_pct=args.spread_bot_pct,
            show_progress=not args.no_progress,
            security_types_path=args.security_types_path,
        )
    except (LiquidityPanelError, SecurityTypesError, OratsCoreError) as exc:
        logger.error("%s", exc)
        return 1

    result.panel = stamp_panel_universe_params(
        result.panel,
        dvol_top_pct=args.dvol_top_pct,
        spread_bot_pct=args.spread_bot_pct,
    )
    universe_df = build_liquid_tickers(
        result.panel,
        dvol_top_pct=args.dvol_top_pct,
        spread_bot_pct=args.spread_bot_pct,
    )
    write_artifacts(args.cache_dir, result, liquid_tickers=universe_df)

    report_path = args.cache_dir / "manifests" / "reports" / f"liquidity_panel_{build_id}.md"
    write_liquidity_panel_report(
        report_path,
        build_id=build_id,
        mode=result.mode,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        start_date=start_date,
        end_date=end_date,
        lookback_weeks=args.lookback_weeks,
        min_valid_quote_weeks=args.min_valid_quote_weeks,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        dvol_top_pct=args.dvol_top_pct,
        spread_bot_pct=args.spread_bot_pct,
        result=result,
    )

    universe_path = args.cache_dir / LIQUID_TICKERS_FILENAME

    logger.info(
        "Saved panel → %s (%d rows); liquid_tickers → %s (%d tickers); report → %s",
        args.cache_dir / PANEL_FILENAME,
        len(result.panel),
        universe_path,
        len(universe_df),
        report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
