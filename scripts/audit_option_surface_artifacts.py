"""
Read-only audit for option surface A1/A2 parquet artifacts (Sprint 004 C6.2/C6.3).

Usage (C6.2 contract only):
    python scripts/audit_option_surface_artifacts.py \\
        --meta-path C:/MomentumCVG_env/cache/c6_2_surface_smoke/option_surface_meta_weekly_2024_2024.parquet \\
        --quotes-path C:/MomentumCVG_env/cache/c6_2_surface_smoke/option_surface_quotes_weekly_2024_2024.parquet \\
        --frequency weekly \\
        --data-root C:/MomentumCVG_env/input/adjusted_liquid \\
        --start-date 2024-01-01 \\
        --end-date 2024-01-31 \\
        --output-report docs/tmp/c6_2_surface_artifact_contract_report.md

Usage (C6.3 assembly-readiness phase after contract gate):
    python scripts/audit_option_surface_artifacts.py \\
        ... \\
        --include-assembly-readiness \\
        --output-report docs/tmp/c6_3_surface_assembly_readiness_report.md
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
from src.features.option_surface_readiness import (  # noqa: E402
    DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
    ReadinessAuditResult,
    group_quotes_by_surface,
    run_readiness_audit,
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
    parser.add_argument(
        "--c6-2-commit",
        default="ffb6ae94d33515971905f54b7a9db5af5bec6ff9",
        help="Baseline C6.2 commit SHA recorded in the report",
    )
    parser.add_argument(
        "--c61-regression-result",
        default="(not run)",
        help="Pytest result line for C6.1 regression suite",
    )
    parser.add_argument(
        "--c62-test-result",
        default="(not run)",
        help="Pytest result line for C6.2 contract/audit tests",
    )
    parser.add_argument(
        "--include-assembly-readiness",
        action="store_true",
        help="Run C6.3 assembly-readiness phase after C6.2 contract gate",
    )
    parser.add_argument(
        "--ironfly-symmetry-tolerance",
        type=float,
        default=DEFAULT_IRONFLY_SYMMETRY_TOLERANCE,
        help="Max strike-distance asymmetry for iron-fly wing pairs",
    )
    parser.add_argument(
        "--c63-commit",
        default=None,
        help="C6.3 commit SHA recorded in readiness report",
    )
    parser.add_argument(
        "--c63-test-result",
        default="(not run)",
        help="Pytest result line for C6.3 readiness tests",
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
    tests_run: list[tuple[str, str]],
    files_changed: list[str],
    c6_2_commit: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    schema = _result_by_name(results, "schema_checks")
    invariant = _result_by_name(results, "surface_valid_invariant")
    vocabulary = _result_by_name(results, "failure_vocabulary")
    settlement = _result_by_name(results, "settlement_readiness")
    meta_grain = _result_by_name(results, "meta_grain")
    join = _result_by_name(results, "a1_a2_join_integrity")
    grain = _result_by_name(results, "quote_grain")
    alignment = _result_by_name(results, "date_alignment")

    lines: list[str] = [
        "# C6.2 — Surface Artifact Contract and Audit Foundation",
        "",
        f"**Generated:** {ts}",
    ]
    if c6_2_commit:
        lines.extend(["", f"**C6.2 commit:** `{c6_2_commit}`"])
    lines.extend(
        [
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
    )
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
            "No parquet mutation, no backfill. "
            "No raw ORATS access; weekly date alignment reads adjusted-liquid file presence only.",
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
    lines.extend(["", "## A1 metadata grain", ""])
    if meta_grain:
        lines.append(f"- Grain: {meta_grain.metrics.get('grain')}")
        lines.append(f"- Metadata row count: {meta_grain.metrics.get('meta_row_count')}")
        lines.append(f"- Duplicate row count: {meta_grain.metrics.get('duplicate_row_count')}")
        lines.append(f"- Duplicate key count: {meta_grain.metrics.get('duplicate_key_count')}")
        if meta_grain.examples:
            lines.append("- Examples:")
            for ex in meta_grain.examples:
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
    for cmd, result_line in tests_run:
        lines.append(f"```powershell\n{cmd}\n```")
        lines.append(f"Result: {result_line}")
        lines.append("")
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


DOWNSTREAM_ASSEMBLY_CONTRACT = """
### Straddle leg requirements
- ``build_straddle_from_surface`` requires one body call and one body put at ``body_strike`` from A2.
- v1 readiness: ``straddle_ready == body_pair_ready`` (quotable body call + quotable body put).

### Iron-fly leg requirements
- ``build_ironfly_from_surface``: long OTM put / short ATM put / short ATM call / long OTM call.
- Body legs from ``is_body`` quotes; wings from ``is_otm`` quotes on each side.
- Wing selection uses ``abs_delta`` vs ``wing_target_delta`` at assembly time (not a C6.3 gate).
- C6.3 structural rule: body pair + at least one quotable OTM call wing + at least one
  quotable OTM put wing. Symmetric wing distance is informational only.

### Iron-condor leg requirements
- ``build_ironcondor_from_surface``: long OTM put / short nearer put / short nearer call / long OTM call.
- Short-leg candidates include body (``is_body | is_otm``); long legs must be further OTM than shorts.
- Delta targets applied at assembly; C6.3 uses conservative structural rule:
  body pair + quotable puts/calls forming at least one put vertical and one call vertical spread.
""".strip()


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def write_c63_markdown_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    contract_results: list[ContractCheckResult],
    contract_overall: str,
    readiness: ReadinessAuditResult | None,
    tests_run: list[tuple[str, str]],
    files_changed: list[str],
    c63_commit: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    readiness_overall = readiness.status if readiness else "SKIPPED"
    if contract_overall == "FAIL":
        overall = "FAIL"
    elif readiness and readiness.status == "FAIL":
        overall = "FAIL"
    elif contract_overall == "WARN" or (readiness and readiness.status == "WARN"):
        overall = "WARN"
    else:
        overall = "PASS" if readiness else contract_overall

    lines: list[str] = [
        "# C6.3 — Surface Assembly-Readiness Audit",
        "",
        f"**Generated:** {ts}",
    ]
    if c63_commit:
        lines.extend(["", f"**Commit:** `{c63_commit}`"])
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{overall}**",
            "",
            "## Scope",
            "",
            f"- Commit baseline (C6.2 follow-up): `{args.c6_2_commit}`",
            f"- Meta: `{inventory['meta_path']}`",
            f"- Quotes: `{inventory['quotes_path']}`",
            f"- Sample window: {args.start_date} .. {args.end_date}",
            f"- Ticker scope: {args.sample_tickers or 'all in filtered sample'}",
            f"- Iron-fly symmetry tolerance: {args.ironfly_symmetry_tolerance}",
            "",
            "### Read-only guarantee",
            "",
            "This audit only reads parquet inputs and writes a markdown report. "
            "No parquet mutation, no backfill, no strategy assembly or backtest.",
            "",
            "## Downstream assembly contract",
            "",
            DOWNSTREAM_ASSEMBLY_CONTRACT,
            "",
            "### Unresolved ambiguity",
            "",
            "- Iron-condor readiness does not enforce delta-bucket targets; S3 may reject "
            "structurally valid surfaces when ``_choose_nearest`` cannot match targets.",
            "- Spread filters (``max_leg_spread_pct``, ``max_spread_cost_ratio``) are assembly-time only.",
            "",
            "## C6.2 prerequisite",
            "",
            f"- Contract verdict: **{contract_overall}**",
        ]
    )
    for result in contract_results:
        lines.append(f"- **{result.name}**: {result.status}")
        for failure in result.failures[:3]:
            lines.append(f"  - FAIL: {failure}")
    if readiness and readiness.blocked:
        lines.extend(
            [
                "",
                "- Readiness evaluation **blocked** because C6.2 contract checks failed.",
                f"- Block reason: {readiness.block_reason}",
            ]
        )

    lines.extend(["", "## Body-pair consistency", ""])
    if readiness and not readiness.blocked:
        body_mismatch = sum(
            1
            for row in readiness.rows
            if row.consistency_failures
            and any(
                f in row.consistency_failures
                for f in (
                    "body_flag_mismatch",
                    "body_strike_mismatch",
                    "surface_valid_body_contradiction",
                    "duplicate_body_call",
                    "duplicate_body_put",
                )
            )
        )
        lines.append(
            "- A1 has_body_call and has_body_put must agree exactly with quotable "
            "A2 body-leg availability in both directions."
        )
        lines.append(f"- Surfaces with body/A1 inconsistencies: {body_mismatch}")
        for ex in readiness.examples[:5]:
            lines.append(f"- Example: {ex}")
    else:
        lines.append("- Not computed (contract gate failed or readiness skipped).")

    metrics = readiness.metrics if readiness else {}
    lines.extend(
        [
            "",
            "## Straddle readiness",
            "",
            f"- Surface count: {metrics.get('surface_count', inventory['meta_row_count'])}",
            f"- body_pair_ready: {metrics.get('body_pair_ready_count', 'n/a')} "
            f"({_format_rate(metrics.get('body_pair_ready_rate'))})",
            f"- straddle_ready: {metrics.get('straddle_ready_count', 'n/a')} "
            f"({_format_rate(metrics.get('straddle_ready_rate'))})",
            f"- Conditional (among surface_valid): "
            f"{_format_rate(metrics.get('straddle_ready_among_surface_valid_rate'))}",
            "",
            "## OTM wing availability",
            "",
            f"- otm_call_wing_available: {metrics.get('otm_call_wing_available_count', 'n/a')} "
            f"({_format_rate(metrics.get('otm_call_wing_available_rate'))})",
            f"- otm_put_wing_available: {metrics.get('otm_put_wing_available_count', 'n/a')} "
            f"({_format_rate(metrics.get('otm_put_wing_available_rate'))})",
            f"- otm_wing_pair_available: {metrics.get('otm_wing_pair_available_count', 'n/a')} "
            f"({_format_rate(metrics.get('otm_wing_pair_available_rate'))})",
            "",
            "## Iron-fly candidate readiness",
            "",
            "### Definition",
            "",
            "body_pair_ready AND at least one quotable OTM call wing AND at least one "
            "quotable OTM put wing.",
            "",
            f"- ironfly_candidate_ready: {metrics.get('ironfly_candidate_ready_count', 'n/a')} "
            f"({_format_rate(metrics.get('ironfly_candidate_ready_rate'))})",
            f"- Conditional (among surface_valid): "
            f"{_format_rate(metrics.get('ironfly_candidate_ready_among_surface_valid_rate'))}",
            "",
            "### Symmetric-wing informational metric",
            "",
            "Symmetric wing distance is not required by the current downstream iron-fly assembler. "
            "It is reported only as an informational structural characteristic.",
            "",
            f"- symmetric_ironfly_pair_available: "
            f"{metrics.get('symmetric_ironfly_pair_available_count', 'n/a')} "
            f"({_format_rate(metrics.get('symmetric_ironfly_pair_available_rate'))})",
            f"- symmetric_ironfly_pair_count: "
            f"{metrics.get('symmetric_ironfly_pair_count', 'n/a')}",
            f"- Symmetry tolerance: {args.ironfly_symmetry_tolerance}",
            "",
            "## Iron-condor candidate readiness",
            "",
            "### Definition derived from S3",
            "",
            "body_pair_ready AND ≥1 quotable put vertical spread AND ≥1 quotable call vertical spread "
            "(long further OTM than short on each side; body may be inner short leg).",
            "",
            f"- ironcondor_candidate_ready: {metrics.get('ironcondor_candidate_ready_count', 'n/a')} "
            f"({_format_rate(metrics.get('ironcondor_candidate_ready_rate'))})",
            f"- Conditional (among surface_valid): "
            f"{_format_rate(metrics.get('ironcondor_candidate_ready_among_surface_valid_rate'))}",
            "",
            "### Limitations",
            "",
            "- Does not apply delta targets or spread filters from ``BacktestRunConfig``.",
            "- Candidate count = put_vertical_pairs × call_vertical_pairs (structural only).",
            "",
            "## Readiness failure breakdown",
            "",
        ]
    )
    if readiness and readiness.failure_reason_breakdown:
        for reason, count in list(readiness.failure_reason_breakdown.items())[:15]:
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    if readiness:
        for warn in readiness.warnings[:8]:
            lines.append(f"- WARN: {warn}")
        for fail in readiness.failures:
            lines.append(f"- FAIL: {fail}")

    lines.extend(["", "## Tests", ""])
    for cmd, result_line in tests_run:
        lines.append(f"```powershell\n{cmd}\n```")
        lines.append(f"Result: {result_line}")
        lines.append("")

    lines.extend(
        [
            "",
            "## Remaining limitations",
            "",
            "- No pricing beyond quotable bid/ask/mid checks.",
            "- No fill simulation or transaction costs.",
            "- No strategy ranking, backtest, or Sharpe/profitability conclusion.",
            "- C6.4 broader coverage thresholds deferred.",
            "",
            "### Files changed (C6.3)",
            "",
        ]
    )
    for path in files_changed:
        lines.append(f"- `{path}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_readiness_phase(
    meta_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    contract_overall: str,
    ironfly_symmetry_tolerance: float,
) -> ReadinessAuditResult:
    contract_passed = contract_overall != "FAIL"
    quotes_grouped = group_quotes_by_surface(quotes_df.to_dict(orient="records"))
    meta_records = meta_df.to_dict(orient="records")
    return run_readiness_audit(
        meta_records,
        quotes_grouped,
        contract_passed=contract_passed,
        ironfly_symmetry_tolerance=ironfly_symmetry_tolerance,
    )


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

    readiness: ReadinessAuditResult | None = None
    if args.include_assembly_readiness:
        readiness = run_readiness_phase(
            meta_df,
            quotes_df,
            contract_overall=overall,
            ironfly_symmetry_tolerance=args.ironfly_symmetry_tolerance,
        )
        if overall == "FAIL":
            final_overall = "FAIL"
        elif readiness.status == "FAIL":
            final_overall = "FAIL"
        elif overall == "WARN" or readiness.status == "WARN":
            final_overall = "WARN"
        else:
            final_overall = "PASS"
    else:
        final_overall = overall

    tests_run = [
        (
            "C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest "
            "tests/unit/test_option_surface_weekly_expiry.py "
            "tests/unit/test_precompute_option_surface_cli.py "
            "tests/unit/test_diagnose_weekly_expiry_policy.py -q",
            args.c61_regression_result,
        ),
        (
            "C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest "
            "tests/unit/test_option_surface_contract.py "
            "tests/unit/test_audit_option_surface_artifacts.py -q",
            args.c62_test_result,
        ),
    ]
    if args.include_assembly_readiness:
        tests_run.append(
            (
                "C:/MomentumCVG_env/venv/Scripts/python.exe -m pytest "
                "tests/unit/test_option_surface_readiness.py -q",
                args.c63_test_result,
            )
        )

    if args.include_assembly_readiness:
        files_changed = [
            "src/features/option_surface_readiness.py",
            "scripts/audit_option_surface_artifacts.py",
            "tests/unit/test_option_surface_readiness.py",
            "tests/unit/test_audit_option_surface_artifacts.py",
            "docs/tmp/c6_3_surface_assembly_readiness_report.md",
        ]
        write_c63_markdown_report(
            args.output_report,
            args=args,
            inventory=inventory,
            contract_results=results,
            contract_overall=overall,
            readiness=readiness,
            tests_run=tests_run,
            files_changed=files_changed,
            c63_commit=args.c63_commit,
        )
    else:
        files_changed = [
            "src/features/option_surface_contract.py",
            "scripts/audit_option_surface_artifacts.py",
            "tests/unit/test_option_surface_contract.py",
            "docs/tmp/c6_2_surface_artifact_contract_report.md",
        ]
        write_markdown_report(
            args.output_report,
            args=args,
            inventory=inventory,
            results=results,
            overall=overall,
            tests_run=tests_run,
            files_changed=files_changed,
            c6_2_commit=args.c6_2_commit,
        )

    logger.info("Wrote report: %s", args.output_report)
    logger.info("Overall verdict: %s", final_overall)

    if final_overall == "FAIL":
        return 1
    if final_overall == "WARN" and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
