"""
Audit a filtered adjusted ORATS data layer (Sprint 004 C5.8).

Validates split history, adjusted output inventory, universe containment,
adjusted-column structure, and raw-vs-adjusted math on sampled files.

Usage:
    python scripts/audit_adjusted_liquid.py \\
        --raw-root C:/ORATS/data/ORATS_Data \\
        --adj-root C:/MomentumCVG_env/input/adjusted_liquid \\
        --splits C:/MomentumCVG_env/input/adjusted_liquid/splits_hist_liquid.parquet \\
        --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv \\
        --years 2020 \\
        --report-path docs/tmp/c5_8_adjusted_liquid_audit_report.md
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.ticker_universe import load_ticker_universe  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
REQUIRED_SPLIT_COLUMNS = ("ticker", "split_date", "divisor")
REQUIRED_ADJ_COLUMNS = (
    "ticker",
    "trade_date",
    "split_factor",
    "stkPx",
    "strike",
    "adj_stkPx",
    "adj_strike",
    "spot_px",
)
OPTIONAL_ADJ_PRICE_MAP = {
    "cBidPx": "adj_cBidPx",
    "cAskPx": "adj_cAskPx",
    "pBidPx": "adj_pBidPx",
    "pAskPx": "adj_pAskPx",
}
JOIN_KEYS = ("ticker", "expirDate", "strike")
ZIP_PREFIX = "ORATS_SMV_Strikes_"
MATH_RTOL = 1e-8
MATH_ATOL = 1e-8
SPOT_RTOL = 1e-8
SPOT_ATOL = 1e-8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class CategoryResult:
    name: str
    status: str  # PASS | FAIL | WARN
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit filtered adjusted ORATS liquid data layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root directory containing per-year raw ORATS ZIP files.",
    )
    p.add_argument(
        "--adj-root",
        type=Path,
        required=True,
        help="Root directory containing per-year adjusted parquet files.",
    )
    p.add_argument(
        "--splits",
        type=Path,
        required=True,
        dest="splits_path",
        help="Parquet split history (ticker, split_date, divisor).",
    )
    p.add_argument(
        "--ticker-universe",
        type=Path,
        required=True,
        dest="ticker_universe",
        help="C4 liquid ticker universe CSV or parquet.",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        metavar="YEAR",
        help="Years to audit.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Markdown report output path.",
    )
    p.add_argument(
        "--sample-files",
        type=int,
        default=10,
        help="Max adjusted parquet files to sample for raw-vs-adjusted math.",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=20000,
        help="Max rows to sample per file for raw-vs-adjusted math.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=57,
        help="Random seed for deterministic file/row sampling.",
    )
    return p.parse_args(argv)


def _date_str_from_filename(name: str) -> str | None:
    """Extract YYYYMMDD from ORATS_SMV_Strikes_YYYYMMDD.{zip,parquet}."""
    stem = Path(name).stem
    if not stem.startswith(ZIP_PREFIX):
        return None
    date_str = stem[len(ZIP_PREFIX) :]
    if len(date_str) == 8 and date_str.isdigit():
        return date_str
    return None


def _inventory_dates(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    dates: set[str] = set()
    for path in directory.glob(f"{ZIP_PREFIX}*{suffix}"):
        date_str = _date_str_from_filename(path.name)
        if date_str:
            dates.add(date_str)
    return dates


def audit_split_file(
    splits_path: Path,
    universe: set[str],
) -> CategoryResult:
    result = CategoryResult(name="Split file audit", status="PASS")

    if not splits_path.is_file():
        result.status = "FAIL"
        result.failures.append(f"Split file not found: {splits_path}")
        return result

    df = pd.read_parquet(splits_path)
    missing_cols = [c for c in REQUIRED_SPLIT_COLUMNS if c not in df.columns]
    if missing_cols:
        result.status = "FAIL"
        result.failures.append(
            f"Missing required columns: {missing_cols}; "
            f"expected {list(REQUIRED_SPLIT_COLUMNS)}"
        )
        result.metrics["split_row_count"] = len(df)
        return result

    work = df.copy()
    raw_ticker = work["ticker"].astype(str).str.strip()
    null_ticker_mask = raw_ticker.isna() | (raw_ticker == "") | (raw_ticker == "nan")
    null_ticker_count = int(null_ticker_mask.sum())
    if null_ticker_count:
        result.status = "FAIL"
        result.failures.append(f"{null_ticker_count} split row(s) have null/blank ticker")

    work["ticker"] = raw_ticker.str.upper()
    not_uppercase = int((work["ticker"] != raw_ticker.str.upper()).sum())
    if not_uppercase:
        result.status = "FAIL"
        result.failures.append(
            f"{not_uppercase} split row(s) are not uppercase after normalization"
        )

    work["split_date"] = pd.to_datetime(work["split_date"], errors="coerce")
    bad_dates = int(work["split_date"].isna().sum())
    if bad_dates:
        result.status = "FAIL"
        result.failures.append(f"{bad_dates} split row(s) have unparseable split_date")

    work["divisor"] = pd.to_numeric(work["divisor"], errors="coerce")
    null_divisor_count = int(work["divisor"].isna().sum())
    nonpositive_divisor_count = int((work["divisor"] <= 0).sum())
    if null_divisor_count:
        result.status = "FAIL"
        result.failures.append(f"{null_divisor_count} split row(s) have null divisor")
    if nonpositive_divisor_count:
        result.status = "FAIL"
        result.failures.append(
            f"{nonpositive_divisor_count} split row(s) have nonpositive divisor"
        )

    duplicate_key_count = int(
        work.duplicated(subset=["ticker", "split_date"], keep=False).sum()
    )
    conflicting = (
        work.groupby(["ticker", "split_date"])["divisor"]
        .nunique(dropna=False)
        .reset_index(name="n_divisors")
    )
    conflicting_duplicate_count = int((conflicting["n_divisors"] > 1).sum())
    if conflicting_duplicate_count:
        result.status = "FAIL"
        result.failures.append(
            f"{conflicting_duplicate_count} (ticker, split_date) key(s) have "
            "conflicting divisor values"
        )

    split_tickers = set(work["ticker"].dropna().unique())
    outside = sorted(split_tickers - universe)
    outside_universe_count = len(outside)
    if outside_universe_count:
        result.status = "FAIL"
        result.failures.append(
            f"{outside_universe_count} split ticker(s) outside universe: "
            f"{outside[:10]}"
        )

    valid_dates = work["split_date"].dropna()
    result.metrics = {
        "split_row_count": len(work),
        "split_unique_ticker_count": work["ticker"].nunique(),
        "split_date_min": valid_dates.min() if not valid_dates.empty else None,
        "split_date_max": valid_dates.max() if not valid_dates.empty else None,
        "null_divisor_count": null_divisor_count,
        "nonpositive_divisor_count": nonpositive_divisor_count,
        "duplicate_key_count": duplicate_key_count,
        "conflicting_duplicate_count": conflicting_duplicate_count,
        "outside_universe_split_ticker_count": outside_universe_count,
    }
    return result


def audit_inventory(
    raw_root: Path,
    adj_root: Path,
    years: list[int],
) -> CategoryResult:
    result = CategoryResult(name="Adjusted output inventory audit", status="PASS")
    year_results: list[dict[str, Any]] = []

    for year in years:
        raw_year = raw_root / str(year)
        adj_year = adj_root / str(year)
        yr: dict[str, Any] = {"year": year, "status": "PASS"}

        raw_exists = raw_year.is_dir()
        adj_exists = adj_year.is_dir()
        if not raw_exists:
            yr["status"] = "FAIL"
            result.status = "FAIL"
            result.failures.append(f"Raw year directory missing: {raw_year}")
        if not adj_exists:
            yr["status"] = "FAIL"
            result.status = "FAIL"
            result.failures.append(f"Adjusted year directory missing: {adj_year}")

        raw_dates = _inventory_dates(raw_year, ".zip") if raw_exists else set()
        adj_dates = _inventory_dates(adj_year, ".parquet") if adj_exists else set()
        missing = sorted(raw_dates - adj_dates)
        extra = sorted(adj_dates - raw_dates)

        yr.update(
            {
                "raw_zip_count": len(raw_dates),
                "adjusted_parquet_count": len(adj_dates),
                "missing_adjusted_count": len(missing),
                "extra_adjusted_count": len(extra),
                "raw_date_min": min(raw_dates) if raw_dates else None,
                "raw_date_max": max(raw_dates) if raw_dates else None,
                "adj_date_min": min(adj_dates) if adj_dates else None,
                "adj_date_max": max(adj_dates) if adj_dates else None,
            }
        )

        if missing:
            yr["status"] = "FAIL"
            result.status = "FAIL"
            result.failures.append(
                f"Year {year}: {len(missing)} raw ZIP date(s) missing adjusted "
                f"parquet (examples: {missing[:5]})"
            )
        if extra:
            yr["status"] = "WARN" if yr["status"] == "PASS" else yr["status"]
            if result.status == "PASS":
                result.status = "WARN"
            result.warnings.append(
                f"Year {year}: {len(extra)} adjusted parquet file(s) without "
                f"matching raw ZIP (examples: {extra[:5]})"
            )

        year_results.append(yr)

    result.metrics["years"] = year_results
    return result


def _collect_adj_parquet_paths(adj_root: Path, years: list[int]) -> list[Path]:
    paths: list[Path] = []
    for year in years:
        adj_year = adj_root / str(year)
        if adj_year.is_dir():
            paths.extend(sorted(adj_year.glob(f"{ZIP_PREFIX}*.parquet")))
    return paths


def audit_universe_containment(
    adj_root: Path,
    years: list[int],
    universe: set[str],
) -> CategoryResult:
    result = CategoryResult(name="Universe containment audit", status="PASS")
    all_tickers: set[str] = set()
    outside_examples: list[str] = []

    for path in _collect_adj_parquet_paths(adj_root, years):
        df = pd.read_parquet(path, columns=["ticker"])
        tickers = df["ticker"].astype(str).str.strip().str.upper().unique()
        all_tickers.update(tickers)

    outside = sorted(t for t in all_tickers if t not in universe)
    if outside:
        result.status = "FAIL"
        outside_examples = outside[:10]
        result.failures.append(
            f"{len(outside)} adjusted output ticker(s) outside universe "
            f"(examples: {outside_examples})"
        )

    result.metrics = {
        "adjusted_output_unique_ticker_count": len(all_tickers),
        "outside_universe_ticker_count": len(outside),
        "outside_universe_examples": outside_examples,
    }
    return result


def audit_adjusted_structure(
    adj_root: Path,
    years: list[int],
) -> CategoryResult:
    result = CategoryResult(name="Adjusted-column structural audit", status="PASS")
    files_checked = 0
    missing_required_columns = 0
    bad_split_factor_count = 0
    bad_adjusted_price_count = 0
    spot_px_mismatch_count = 0
    trade_date_mismatch_count = 0

    for path in _collect_adj_parquet_paths(adj_root, years):
        files_checked += 1
        df = pd.read_parquet(path)
        missing = [c for c in REQUIRED_ADJ_COLUMNS if c not in df.columns]
        if missing:
            missing_required_columns += 1
            result.status = "FAIL"
            result.failures.append(
                f"{path.name}: missing required columns {missing}"
            )
            continue

        file_date_str = _date_str_from_filename(path.name)
        expected_trade_date = (
            pd.Timestamp(file_date_str) if file_date_str else pd.NaT
        )

        sf = pd.to_numeric(df["split_factor"], errors="coerce")
        bad_sf = (
            sf.isna()
            | ~np.isfinite(sf)
            | (sf <= 0)
        )
        n_bad_sf = int(bad_sf.sum())
        if n_bad_sf:
            bad_split_factor_count += n_bad_sf
            result.status = "FAIL"
            result.failures.append(
                f"{path.name}: {n_bad_sf} row(s) with bad split_factor"
            )

        for col in ("adj_stkPx", "adj_strike"):
            vals = pd.to_numeric(df[col], errors="coerce")
            n_bad = int((~np.isfinite(vals)).sum())
            if n_bad:
                bad_adjusted_price_count += n_bad
                result.status = "FAIL"
                result.failures.append(
                    f"{path.name}: {n_bad} row(s) with nonfinite {col}"
                )

        spot = pd.to_numeric(df["spot_px"], errors="coerce")
        adj_stk = pd.to_numeric(df["adj_stkPx"], errors="coerce")
        spot_mismatch = ~np.isclose(spot, adj_stk, rtol=SPOT_RTOL, atol=SPOT_ATOL)
        n_spot = int(spot_mismatch.sum())
        if n_spot:
            spot_px_mismatch_count += n_spot
            result.status = "FAIL"
            result.failures.append(
                f"{path.name}: {n_spot} row(s) where spot_px != adj_stkPx"
            )

        trade_dates = pd.to_datetime(df["trade_date"], errors="coerce")
        n_bad_td = int(trade_dates.isna().sum())
        if n_bad_td:
            trade_date_mismatch_count += n_bad_td
            result.status = "FAIL"
            result.failures.append(
                f"{path.name}: {n_bad_td} row(s) with unparseable trade_date"
            )

        if file_date_str is not None and expected_trade_date is not pd.NaT:
            td_mismatch = trade_dates.dt.normalize() != expected_trade_date.normalize()
            n_td = int(td_mismatch.sum())
            if n_td:
                trade_date_mismatch_count += n_td
                result.status = "FAIL"
                result.failures.append(
                    f"{path.name}: {n_td} row(s) where trade_date != file date "
                    f"{file_date_str}"
                )

    result.metrics = {
        "files_checked": files_checked,
        "missing_required_columns": missing_required_columns,
        "bad_split_factor_count": bad_split_factor_count,
        "bad_adjusted_price_count": bad_adjusted_price_count,
        "spot_px_mismatch_count": spot_px_mismatch_count,
        "trade_date_mismatch_count": trade_date_mismatch_count,
    }
    return result


def _read_raw_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith((".csv", ".txt"))]
        if not csv_names:
            raise ValueError(f"No CSV inside {zip_path.name}")
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f, dtype={"ticker": str})


def _filter_raw_to_universe(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    ticker_series = df["ticker"].astype(str).str.strip().str.upper()
    filtered = df[ticker_series.isin(universe)].copy()
    filtered["ticker"] = ticker_series.loc[filtered.index].values
    return filtered


def _math_mismatch_mask(
    raw_vals: pd.Series,
    adj_vals: pd.Series,
    split_factor: pd.Series,
) -> pd.Series:
    expected = pd.to_numeric(raw_vals, errors="coerce") / pd.to_numeric(
        split_factor, errors="coerce"
    )
    actual = pd.to_numeric(adj_vals, errors="coerce")
    return ~np.isclose(expected, actual, rtol=MATH_RTOL, atol=MATH_ATOL)


def audit_raw_math_sample(
    raw_root: Path,
    adj_root: Path,
    years: list[int],
    universe: set[str],
    *,
    sample_files: int,
    sample_rows: int,
    seed: int,
) -> CategoryResult:
    result = CategoryResult(name="Raw-vs-adjusted math spot-check", status="PASS")
    all_adj = _collect_adj_parquet_paths(adj_root, years)
    rng = random.Random(seed)
    if len(all_adj) <= sample_files:
        sampled_paths = list(all_adj)
    else:
        sampled_paths = rng.sample(all_adj, sample_files)

    sampled_files = len(sampled_paths)
    sampled_rows_total = 0
    matched_rows = 0
    unmatched_rows = 0
    math_mismatch_count = 0

    join_keys = list(JOIN_KEYS)

    for adj_path in sampled_paths:
        date_str = _date_str_from_filename(adj_path.name)
        if not date_str:
            result.warnings.append(f"Cannot parse date from {adj_path.name}; skipped")
            continue

        year = date_str[:4]
        zip_path = raw_root / year / f"{ZIP_PREFIX}{date_str}.zip"
        if not zip_path.is_file():
            result.warnings.append(f"Matching raw ZIP not found: {zip_path}")
            continue

        adj_df = pd.read_parquet(adj_path)
        if len(adj_df) == 0:
            continue

        n_sample = min(sample_rows, len(adj_df))
        adj_sample = adj_df.sample(n=n_sample, random_state=seed)
        sampled_rows_total += len(adj_sample)

        try:
            raw_df = _read_raw_csv_from_zip(zip_path)
        except Exception as exc:
            result.warnings.append(f"Failed to read {zip_path.name}: {exc}")
            continue

        raw_df = _filter_raw_to_universe(raw_df, universe)

        available_keys = [k for k in join_keys if k in raw_df.columns and k in adj_sample.columns]
        if len(available_keys) < len(join_keys):
            missing = set(join_keys) - set(available_keys)
            result.status = "FAIL"
            result.failures.append(
                f"{adj_path.name}: missing join key columns {sorted(missing)}"
            )
            continue

        raw_keys = raw_df[available_keys].copy()
        for col in available_keys:
            if col == "ticker":
                raw_keys[col] = raw_keys[col].astype(str).str.strip().str.upper()
            if col == "strike":
                raw_keys[col] = pd.to_numeric(raw_keys[col], errors="coerce")

        adj_keys = adj_sample[available_keys].copy()
        for col in available_keys:
            if col == "ticker":
                adj_keys[col] = adj_keys[col].astype(str).str.strip().str.upper()
            if col == "strike":
                adj_keys[col] = pd.to_numeric(adj_keys[col], errors="coerce")

        raw_for_merge = raw_df.copy()
        raw_for_merge["ticker"] = raw_for_merge["ticker"].astype(str).str.strip().str.upper()
        raw_for_merge["strike"] = pd.to_numeric(raw_for_merge["strike"], errors="coerce")

        merged = adj_sample.merge(
            raw_for_merge,
            on=available_keys,
            how="left",
            suffixes=("_adj", "_raw"),
            indicator=True,
        )

        matched = merged[merged["_merge"] == "both"]
        unmatched = merged[merged["_merge"] != "both"]
        matched_rows += len(matched)
        unmatched_rows += len(unmatched)

        if len(unmatched) > 0:
            result.warnings.append(
                f"{adj_path.name}: {len(unmatched)} sampled row(s) unmatched to raw"
            )

        if matched.empty:
            continue

        sf = matched["split_factor"]
        checks = [("stkPx", "adj_stkPx"), ("strike", "adj_strike")]
        for raw_col, adj_col in checks:
            raw_col_name = raw_col if raw_col in matched.columns else f"{raw_col}_raw"
            if raw_col_name not in matched.columns:
                continue
            mismatch = _math_mismatch_mask(matched[raw_col_name], matched[adj_col], sf)
            n = int(mismatch.sum())
            if n:
                math_mismatch_count += n
                result.status = "FAIL"
                result.failures.append(
                    f"{adj_path.name}: {n} row(s) fail {adj_col} == "
                    f"{raw_col} / split_factor"
                )

        for raw_col, adj_col in OPTIONAL_ADJ_PRICE_MAP.items():
            if raw_col in matched.columns and adj_col in matched.columns:
                mismatch = _math_mismatch_mask(matched[raw_col], matched[adj_col], sf)
                n = int(mismatch.sum())
                if n:
                    math_mismatch_count += n
                    result.status = "FAIL"
                    result.failures.append(
                        f"{adj_path.name}: {n} row(s) fail {adj_col} == "
                        f"{raw_col} / split_factor"
                    )

    if unmatched_rows > 0 and result.status == "PASS":
        result.status = "WARN"
        result.warnings.append(
            f"{unmatched_rows} total sampled row(s) could not be matched to raw "
            "(investigate duplicate keys or missing raw rows)"
        )

    result.metrics = {
        "sampled_files": sampled_files,
        "sampled_rows": sampled_rows_total,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "math_mismatch_count": math_mismatch_count,
    }
    return result


def _overall_verdict(categories: list[CategoryResult]) -> str:
    if any(c.status == "FAIL" for c in categories):
        return "FAIL"
    if any(c.status == "WARN" for c in categories):
        return "PASS WITH WARNINGS"
    return "PASS"


def write_markdown_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    universe_count: int,
    categories: list[CategoryResult],
    overall: str,
    all_failures: list[str],
    all_warnings: list[str],
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Adjusted Liquid Data Layer Audit Report",
        "",
        f"**Generated:** {ts}",
        "",
        "## Input paths",
        "",
        f"- **raw-root:** `{args.raw_root}`",
        f"- **adj-root:** `{args.adj_root}`",
        f"- **splits:** `{args.splits_path}`",
        f"- **ticker-universe:** `{args.ticker_universe}`",
        "",
        "## Years audited",
        "",
        ", ".join(str(y) for y in args.years),
        "",
        "## Audit configuration",
        "",
        f"- sample-files: {args.sample_files}",
        f"- sample-rows: {args.sample_rows}",
        f"- seed: {args.seed}",
        f"- universe ticker count: {universe_count}",
        "",
        f"## Overall verdict: **{overall}**",
        "",
    ]

    for cat in categories:
        lines.extend([f"## {cat.name}", "", f"**Status:** {cat.status}", ""])
        for key, val in cat.metrics.items():
            if key == "years":
                lines.append("### Per-year inventory")
                lines.append("")
                for yr in val:
                    lines.append(f"#### Year {yr.get('year')}")
                    lines.append("")
                    for yk, yv in yr.items():
                        if yk != "year":
                            lines.append(f"- {yk}: {yv}")
                    lines.append("")
            else:
                lines.append(f"- {key}: {val}")
        lines.append("")
        if cat.failures:
            lines.append("Failures:")
            for f in cat.failures:
                lines.append(f"- {f}")
            lines.append("")
        if cat.warnings:
            lines.append("Warnings:")
            for w in cat.warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.extend(["## Warnings", ""])
    if all_warnings:
        for w in all_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.extend(["", "## Failures", ""])
    if all_failures:
        for f in all_failures:
            lines.append(f"- {f}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Next recommended action",
            "",
        ]
    )
    if overall == "FAIL":
        lines.append(
            "Fix blocking failures above, then re-run this audit before using "
            "the adjusted_liquid layer downstream."
        )
    elif overall == "PASS WITH WARNINGS":
        lines.append(
            "Review warnings (extra adjusted files, unmatched sample rows). "
            "Re-run after investigation if warnings persist."
        )
    else:
        lines.append(
            "All checks passed. Safe to proceed with downstream input-layer "
            "validation on this adjusted root."
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    universe_list = load_ticker_universe(args.ticker_universe)
    universe = set(universe_list)

    logger.info("Auditing adjusted liquid layer")
    logger.info("  raw-root         : %s", args.raw_root)
    logger.info("  adj-root         : %s", args.adj_root)
    logger.info("  splits           : %s", args.splits_path)
    logger.info("  ticker-universe  : %s (%d tickers)", args.ticker_universe, len(universe))
    logger.info("  years            : %s", args.years)
    logger.info("  report-path      : %s", args.report_path)

    categories = [
        audit_split_file(args.splits_path, universe),
        audit_inventory(args.raw_root, args.adj_root, args.years),
        audit_universe_containment(args.adj_root, args.years, universe),
        audit_adjusted_structure(args.adj_root, args.years),
        audit_raw_math_sample(
            args.raw_root,
            args.adj_root,
            args.years,
            universe,
            sample_files=args.sample_files,
            sample_rows=args.sample_rows,
            seed=args.seed,
        ),
    ]

    overall = _overall_verdict(categories)
    all_failures: list[str] = []
    all_warnings: list[str] = []
    for cat in categories:
        all_failures.extend(cat.failures)
        all_warnings.extend(cat.warnings)

    write_markdown_report(
        args.report_path,
        args=args,
        universe_count=len(universe),
        categories=categories,
        overall=overall,
        all_failures=all_failures,
        all_warnings=all_warnings,
    )

    logger.info("─" * 60)
    logger.info("Overall verdict: %s", overall)
    for cat in categories:
        logger.info("  %-40s %s", cat.name, cat.status)
    logger.info("Report written: %s", args.report_path)
    logger.info("─" * 60)

    if overall == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
