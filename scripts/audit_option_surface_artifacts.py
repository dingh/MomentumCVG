"""
Read-only audit for option surface A1/A2 parquet artifacts (Sprint 004 C6.2).

Usage:
    python scripts/audit_option_surface_artifacts.py \\
        --meta-path C:/MomentumCVG_env/cache/c6_2_surface_smoke/option_surface_meta_weekly_2024_2024.parquet \\
        --quotes-path C:/MomentumCVG_env/cache/c6_2_surface_smoke/option_surface_quotes_weekly_2024_2024.parquet \\
        --frequency weekly \\
        --data-root C:/MomentumCVG_env/input/adjusted_liquid \\
        --start-date 2024-01-01 \\
        --end-date 2024-01-31 \\
        --output-report docs/tmp/c6_2_surface_artifact_contract_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.option_surface_contract import (  # noqa: E402
    ContractCheckResult,
    compute_overall_verdict,
    filter_artifacts,
    run_contract_checks,
    _to_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit for option surface A1/A2 parquet artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--meta-path", type=Path, required=True, help="A1 metadata parquet path")
    parser.add_argument("--quotes-path", type=Path, required=True, help="A2 quotes parquet path")
    parser.add_argument(
        "--frequency",
        choices=("weekly", "monthly"),
        default="weekly",
        help="Surface frequency for date-alignment checks",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Adjusted-liquid root for weekly schedule resolution (weekly only)",
    )
    parser.add_argument("--start-date", type=_parse_iso_date, default=None, help="Audit window start")
    parser.add_argument("--end-date", type=_parse_iso_date, default=None, help="Audit window end")
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Markdown report output path",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap metadata rows audited")
    parser.add_argument(
        "--sample-tickers",
        nargs="+",
        default=None,
        help="Optional ticker subset",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit 1 when overall verdict is WARN",
    )
    return parser.parse_args(argv)


def _artifact_inventory(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    meta_path: Path,
    quotes_path: Path,
) -> dict[str, Any]:
    entry_dates = [d for d in (_to_date(v) for v in meta_df.get("entry_date", [])) if d]
    expiry_dates = [d for d in (_to_date(v) for v in meta_df.get("expiry_date", [])) if d]
    return {
        "meta_path": str(meta_path),
        "quotes_path": str(quotes_path),
        "meta_exists": meta_path.exists(),
        "quotes_exists": quotes_path.exists(),
        "meta_row_count": len(meta_df),
        "quotes_row_count": len(quotes_df),
        "ticker_count": int(meta_df["ticker"].nunique()) if "ticker" in meta_df.columns else 0,
        "entry_date_min": min(entry_dates).isoformat() if entry_dates else None,
        "entry_date_max": max(entry_dates).isoformat() if entry_dates else None,
        "expiry_date_min": min(expiry_dates).isoformat() if expiry_dates else None,
        "expiry_date_max": max(expiry_dates).isoformat() if expiry_dates else None,
    }


def _result_by_name(results: list[ContractCheckResult], name: str) -> ContractCheckResult | None:
    for result in results:
        if result.name == name:
            return result
    return None


def write_markdown_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    results: list[ContractCheckResult],
    overall: str,
    tests_run: list[str],
    files_changed: list[str],
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    schema = _result_by_name(results, "schema_checks")
    invariant = _result_by_name(results, "surface_valid_invariant")
    vocabulary = _result_by_name(results, "failure_vocabulary")
    settlement = _result_by_name(results, "settlement_readiness")
    join = _result_by_name(results, "a1_a2_join_integrity")
    grain = _result_by_name(results, "quote_grain")
    alignment = _result_by_name(results, "date_alignment")

    lines: list[str] = [
        "# C6.2 — Surface Artifact Contract and Audit Foundation",
        "",
        f"**Generated:** {ts}",
        "",
        "## Verdict",
        "",
        f"**{overall}**",
        "",
        "## Scope",
        "",
        "### Files changed",
        "",
    ]
    for path in files_changed:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "### Artifacts audited",
            "",
            f"- Meta: `{inventory['meta_path']}`",
            f"- Quotes: `{inventory['quotes_path']}`",
            "",
            "### Read-only guarantee",
            "",
            "This audit only reads parquet inputs and writes a markdown report. "
            "No parquet mutation, no backfill, no raw ORATS access.",
            "",
            "## Artifact inventory",
            "",
            f"- Meta path: `{inventory['meta_path']}`",
            f"- Quotes path: `{inventory['quotes_path']}`",
            f"- Meta exists: {inventory['meta_exists']}",
            f"- Quotes exists: {inventory['quotes_exists']}",
            f"- Meta row count: {inventory['meta_row_count']}",
            f"- Quotes row count: {inventory['quotes_row_count']}",
            f"- Ticker count: {inventory['ticker_count']}",
            f"- Entry-date range: {inventory['entry_date_min']} .. {inventory['entry_date_max']}",
            f"- Expiry-date range: {inventory['expiry_date_min']} .. {inventory['expiry_date_max']}",
            "",
            "## Schema checks",
            "",
        ]
    )
    if schema:
        lines.append(f"- Required A1 columns present: {schema.metrics.get('meta_columns_present')}")
        lines.append(f"- Required A2 columns present: {schema.metrics.get('quotes_columns_present')}")
        if schema.metrics.get("missing_meta_columns"):
            lines.append(f"- Missing A1 columns: {schema.metrics['missing_meta_columns']}")
        if schema.metrics.get("missing_quotes_columns"):
            lines.append(f"- Missing A2 columns: {schema.metrics['missing_quotes_columns']}")
    lines.extend(["", "## surface_valid invariant", ""])
    if invariant:
        lines.append(f"- Status: {invariant.status}")
        lines.append(f"- Pass count: {invariant.metrics.get('pass_count')}")
        lines.append(f"- Violation count: {invariant.metrics.get('violation_count')}")
        if invariant.examples:
            lines.append("- Examples:")
            for ex in invariant.examples:
                lines.append(f"  - {ex}")
    lines.extend(["", "## Failure vocabulary", ""])
    if vocabulary:
        lines.append(f"- Known tags: {vocabulary.metrics.get('known_tags')}")
        lines.append(f"- Unknown tag count: {vocabulary.metrics.get('unknown_tag_count')}")
        lines.append(f"- Failure breakdown: {vocabulary.metrics.get('failure_breakdown')}")
        if vocabulary.warnings:
            lines.append("- Warnings:")
            for warn in vocabulary.warnings:
                lines.append(f"  - {warn}")
    lines.extend(["", "## Settlement readiness", ""])
    if settlement:
        lines.append(f"- Valid row count: {settlement.metrics.get('valid_row_count')}")
        lines.append(f"- dte_actual mismatch count: {settlement.metrics.get('dte_mismatch_count')}")
        for key, val in settlement.metrics.items():
            if key.startswith("null_") and val:
                lines.append(f"- {key}: {val}")
        if settlement.examples:
            lines.append("- Examples:")
            for ex in settlement.examples:
                lines.append(f"  - {ex}")
    lines.extend(["", "## A1/A2 join integrity", ""])
    if join:
        lines.append(f"- Orphan quote rows: {join.metrics.get('orphan_quote_count')}")
        lines.append(
            f"- Valid metadata rows without quote rows: "
            f"{join.metrics.get('valid_meta_without_quotes_count')}"
        )
        lines.append(
            f"- Invalid metadata rows with quote rows (informational): "
            f"{join.metrics.get('invalid_meta_with_quotes_count')}"
        )
        if join.examples:
            lines.append("- Examples:")
            for ex in join.examples:
                lines.append(f"  - {ex}")
    lines.extend(["", "## Quote grain", ""])
    if grain:
        lines.append(f"- Grain: {grain.metrics.get('grain')}")
        lines.append(f"- Duplicate key count: {grain.metrics.get('duplicate_key_count')}")
        if grain.examples:
            lines.append("- Examples:")
            for ex in grain.examples:
                lines.append(f"  - {ex}")
    lines.extend(["", "## Date alignment", ""])
    if alignment:
        if alignment.metrics.get("skipped"):
            lines.append(f"- Skipped: {alignment.metrics.get('reason')}")
        else:
            lines.append(f"- Status: {alignment.status}")
            lines.append(f"- Policy: {alignment.metrics.get('policy')}")
            lines.append(
                f"- Misaligned entry dates: {alignment.metrics.get('misaligned_entry_count')}"
            )
            if alignment.examples:
                lines.append("- Examples:")
                for ex in alignment.examples:
                    lines.append(f"  - {ex}")
    lines.extend(["", "## Tests", ""])
    for cmd in tests_run:
        lines.append(f"```powershell\n{cmd}\n```")
    lines.extend(
        [
            "",
            "## Remaining limitations",
            "",
            "- C6.3 assembly-readiness metrics (straddle_ready, ironfly_candidate_ready) deferred.",
            "- C6.4 broader coverage / validity-rate thresholds deferred.",
            "- C6.1D failure-taxonomy cleanup for null failure_reason on legacy invalid rows deferred.",
            "- C7 PIT universe harness deferred.",
            "",
            "## Check summary",
            "",
        ]
    )
    for result in results:
        lines.append(f"- **{result.name}**: {result.status}")
        for failure in result.failures:
            lines.append(f"  - FAIL: {failure}")
        for warning in result.warnings:
            lines.append(f"  - WARN: {warning}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_artifacts(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    frequency: str,
    data_root: Path | None,
    start_date: date | None,
    end_date: date | None,
) -> list[ContractCheckResult]:
    if frequency == "weekly" and data_root is None:
        raise ValueError("--data-root is required for weekly frequency audits")
    return run_contract_checks(
        meta_df,
        quotes_df,
        frequency=frequency,
        data_root=data_root,
        start_date=start_date,
        end_date=end_date,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.meta_path.exists():
        logger.error("Meta artifact not found: %s", args.meta_path)
        return 2
    if not args.quotes_path.exists():
        logger.error("Quotes artifact not found: %s", args.quotes_path)
        return 2
    if args.frequency == "weekly" and args.data_root is None:
        logger.error("--data-root is required when --frequency weekly")
        return 2

    meta_df = pd.read_parquet(args.meta_path)
    quotes_df = pd.read_parquet(args.quotes_path)
    meta_df, quotes_df = filter_artifacts(
        meta_df,
        quotes_df,
        start_date=args.start_date,
        end_date=args.end_date,
        sample_tickers=args.sample_tickers,
        max_rows=args.max_rows,
    )

    results = audit_artifacts(
        meta_df,
        quotes_df,
        frequency=args.frequency,
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    overall = compute_overall_verdict(results)
    inventory = _artifact_inventory(
        meta_df,
        quotes_df,
        meta_path=args.meta_path,
        quotes_path=args.quotes_path,
    )

    tests_run = [
        "C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest "
        "tests/unit/test_option_surface_weekly_expiry.py "
        "tests/unit/test_precompute_option_surface_cli.py "
        "tests/unit/test_diagnose_weekly_expiry_policy.py -q",
        "C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest "
        "tests/unit/test_option_surface_contract.py "
        "tests/unit/test_audit_option_surface_artifacts.py -q",
    ]
    files_changed = [
        "src/features/option_surface_contract.py",
        "scripts/audit_option_surface_artifacts.py",
        "src/features/option_surface_analyzer.py",
        "tests/unit/test_option_surface_contract.py",
        "tests/unit/test_audit_option_surface_artifacts.py",
    ]

    write_markdown_report(
        args.output_report,
        args=args,
        inventory=inventory,
        results=results,
        overall=overall,
        tests_run=tests_run,
        files_changed=files_changed,
    )
    logger.info("Wrote report: %s", args.output_report)
    logger.info("Overall verdict: %s", overall)

    if overall == "FAIL":
        return 1
    if overall == "WARN" and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
