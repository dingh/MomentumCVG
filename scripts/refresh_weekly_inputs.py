"""Weekly Stage A input refresh CLI skeleton (Sprint 004 C2).

Purpose
-------
Operator entrypoint for refreshing and validating the *input layer* artifacts
(splits, spot, liquidity panel, option surface) that feed backtesting. C2 delivers
only the shell: argparse, ``--as-of`` resolution, a provisional ``plan``, and a
fixed exit-code contract.

What C2 does
------------
* ``plan`` — print a short, clearly provisional Stage A step list (no execution).
* ``refresh --dry-run`` — same plan plus a dry-run banner.
* Blocked stubs — ``validate``, ``split-audit``, ``surface-audit``, and non-dry
  ``refresh`` return exit **2** so they are not mistaken for success.

What C2 does **not** do (deferred commits)
------------------------------------------
* Subprocess wiring to Stage A scripts (C8).
* Manifest writes / ``write_manifest`` (C8).
* Real validation or audit reports (C3/C6); split audit script exists standalone (C5 ✓); CLI wiring → C8.
* Reading cache artifacts or rebuilding liquidity (C3–C8).

Exit codes
----------
0 = ``plan`` or ``refresh --dry-run`` success
1 = blocking validation failure (reserved for C3+; unused in C2)
2 = usage/config error, not-implemented stub, or blocked refresh

See ``docs/tmp/c2_cli_design_plan.md`` for the full contract.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# ── resolve project root so src/ imports work regardless of cwd ──────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.input_snapshot import (  # noqa: E402
    ARTIFACT_LIQUIDITY_PANEL,
    ARTIFACT_OPTION_SURFACE_META,
    ARTIFACT_OPTION_SURFACE_QUOTES,
    ARTIFACT_SPLITS,
    ARTIFACT_SPOT_PRICES,
    DEFAULT_DATA_SOURCE,
    INPUT_SNAPSHOT_SCHEMA_VERSION,
    compute_snapshot_id,
)
from src.data.paths import DEFAULT_ADJUSTED_LIQUID_ROOT
from src.data.trading_day import resolve_as_of_trading_day  # noqa: E402

# ── Defaults (match existing Stage A scripts; overridable via CLI flags) ───────
DEFAULT_CACHE_DIR = Path("C:/MomentumCVG_env/cache")
DEFAULT_ORATS_ADJ_ROOT = DEFAULT_ADJUSTED_LIQUID_ROOT

# Cache-relative artifact paths for the weekly input receipt. Kept here (not in C1)
# so C1 stays a minimal receipt module; CLI owns operator-facing path defaults.
DEFAULT_ARTIFACT_REL_PATHS: dict[str, str] = {
    ARTIFACT_SPLITS: "splits_hist.parquet",
    ARTIFACT_SPOT_PRICES: "spot_prices_adjusted.parquet",
    ARTIFACT_LIQUIDITY_PANEL: "ticker_liquidity_panel.parquet",
    ARTIFACT_OPTION_SURFACE_META: "option_surface_meta_weekly_2018_2026.parquet",
    ARTIFACT_OPTION_SURFACE_QUOTES: "option_surface_quotes_weekly_2018_2026.parquet",
}

# Identity params mirrored from C1 receipt schema — used only for snapshot_id preview.
DEFAULT_RECEIPT_PARAMS: dict[str, Any] = {
    "rolling_months": 3,
    "universe_rule": "top_20_pct_and_filter",
    "feature_branch": "deferred_to_sprint005",
}


@dataclass
class CliContext:
    """Resolved state shared by all subcommand handlers after argparse parsing.

    Separates *requested* ``--as-of`` from *resolved* trading day so plan output
    and future manifests can show both (operator may pass a weekend date).
    """

    command: str
    as_of_requested: date
    as_of_resolved_trading_day: date
    cache_dir: Path
    orats_adj_root: Path
    mode: str  # incremental | backfill | repair — display-only in C2


def _parse_iso_date(value: str) -> date:
    """Argparse type converter; surfaces bad CLI dates as exit code 2 via SystemExit."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from exc


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags required on every subcommand (HD-004-2 as-of resolution needs ORATS root)."""
    parser.add_argument(
        "--as-of",
        required=True,
        type=_parse_iso_date,
        help="As-of calendar date (YYYY-MM-DD); resolves to last ORATS day on or before it",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache root (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--orats-adj-root",
        type=Path,
        default=DEFAULT_ORATS_ADJ_ROOT,
        help=f"ORATS adjusted parquet root (default: {DEFAULT_ORATS_ADJ_ROOT})",
    )


def _add_plan_refresh_args(parser: argparse.ArgumentParser) -> None:
    """Operator mode flags — parsed and echoed in plan; enforcement deferred to C8."""
    parser.add_argument(
        "--mode",
        choices=("incremental", "backfill", "repair"),
        default="incremental",
        help="Operator mode (display-only in C2; default: incremental)",
    )
    parser.add_argument("--start-date", type=_parse_iso_date, default=None)
    parser.add_argument("--end-date", type=_parse_iso_date, default=None)
    parser.add_argument(
        "--sample-tickers",
        default=None,
        help="Comma-separated tickers for repair scope (display-only in C2)",
    )
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build argparse tree; ``argv`` hook makes ``main()`` testable without subprocess."""
    parser = argparse.ArgumentParser(
        description="Weekly Stage A input refresh operator CLI (Sprint 004).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Show provisional refresh plan")
    _add_common_args(plan_parser)
    _add_plan_refresh_args(plan_parser)

    refresh_parser = subparsers.add_parser("refresh", help="Execute or dry-run refresh")
    _add_common_args(refresh_parser)
    _add_plan_refresh_args(refresh_parser)
    refresh_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not execute subprocesses",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate input artifacts")
    _add_common_args(validate_parser)
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN as FAIL (deferred until C3)",
    )

    split_audit_parser = subparsers.add_parser("split-audit", help="Run split audit")
    _add_common_args(split_audit_parser)

    surface_audit_parser = subparsers.add_parser("surface-audit", help="Run surface audit")
    _add_common_args(surface_audit_parser)

    return parser.parse_args(argv)


def build_cli_context(
    args: argparse.Namespace,
    *,
    exists_fn=None,
) -> CliContext:
    """Parse args into resolved ``CliContext``; raises ``ValueError`` if as-of cannot resolve.

    ``exists_fn`` is forwarded to ``resolve_as_of_trading_day`` for unit tests only.
    Production leaves it ``None`` so resolution uses real ``Path.is_file`` checks.
    """
    resolved = resolve_as_of_trading_day(
        args.as_of,
        args.orats_adj_root,
        exists_fn=exists_fn,
    )
    # validate / audit subcommands do not define --mode; default incremental for context.
    mode = getattr(args, "mode", "incremental")
    return CliContext(
        command=args.command,
        as_of_requested=args.as_of,
        as_of_resolved_trading_day=resolved,
        cache_dir=Path(args.cache_dir),
        orats_adj_root=Path(args.orats_adj_root),
        mode=mode,
    )


def _snapshot_id_preview(ctx: CliContext) -> str:
    """Compute receipt ``snapshot_id`` for display only — no manifest write in C2."""
    identity = {
        "schema_version": INPUT_SNAPSHOT_SCHEMA_VERSION,
        "as_of_resolved_trading_day": ctx.as_of_resolved_trading_day,
        "data_source": DEFAULT_DATA_SOURCE,
        "artifacts": DEFAULT_ARTIFACT_REL_PATHS,
        "params": DEFAULT_RECEIPT_PARAMS,
    }
    return compute_snapshot_id(identity)


def render_plan(ctx: CliContext, args: argparse.Namespace) -> str:
    """Return plain-text provisional plan (stdout for ``plan`` and ``refresh --dry-run``).

    Step order is intentional: establish candidate universe scope *before* splits,
    adjustment, and liquidity rebuild so later C3–C8 wiring can be universe-aware.
    Wording is explicitly provisional — not an approved architecture doc.

    Sprint 004 closeout (C9): remove C2-era scaffolding from operator-facing output.
    See ``current_sprint.md`` closeout blocker #13. Strings marked TEMP below must go
    once the referenced commit lands; update ``test_refresh_weekly_inputs_cli.py`` in
    the same change.
    """
    # TEMP (C2–C7): commit labels and bracket notes in plan body — not permanent UX.
    # Keep step *names*; drop "deferred to C3–C8", "no rebuild in C2", "implementation
    # deferred", and "Provisional" header when C8 wires real execution. Feature-branch
    # Sprint 005 line may remain until 005 wires A4.
    lines: list[str] = [
        "=== Weekly input refresh plan (no execution) ===",  # TEMP: drop "(no execution)" / "Provisional" at closeout
        f"as_of_requested:             {ctx.as_of_requested.isoformat()}",
        f"as_of_resolved_trading_day:  {ctx.as_of_resolved_trading_day.isoformat()}",
        f"mode:                        {ctx.mode}",
        f"cache_dir:                   {ctx.cache_dir}",
        f"orats_adj_root:              {ctx.orats_adj_root}",
        f"data_source:                 {DEFAULT_DATA_SOURCE}",
        "",
        "Provisional high-level Stage A plan:",  # TEMP: rename when plan is no longer provisional
        # Step 0: read-only scope resolution in future commits; no cache reads in C2.
        "  0. resolve_candidate_universe_scope   [read existing liquidity panel / liquid_tickers / master list; no rebuild in C2]",  # TEMP bracket
        "  1. fetch_splits",
        "  2. apply_split_adjustment",
        "  3. build_liquidity_panel              [rebuild/refine rolling panel; implementation deferred]",  # TEMP bracket (C4)
        "  4. extract_spot_prices",
        "  5. precompute_option_surface",
        "",
        "Note: candidate universe scope should be established early so later steps can become universe-aware.",
        "      Exact storage scope, master-universe filtering, rolling liquidity rebuild behavior,",
        "      and subprocess wiring are deferred to C3–C8.",  # TEMP: remove entire note block at closeout
        "",
        "Feature branch (straddle history, build_features, A4): deferred to Sprint 005",  # TEMP until Sprint 005
        "",
        "Logical receipt artifacts (cache-relative):",
        f"  {ARTIFACT_SPLITS + ':':<22} {DEFAULT_ARTIFACT_REL_PATHS[ARTIFACT_SPLITS]}",
        f"  {ARTIFACT_SPOT_PRICES + ':':<22} {DEFAULT_ARTIFACT_REL_PATHS[ARTIFACT_SPOT_PRICES]}",
        f"  {ARTIFACT_LIQUIDITY_PANEL + ':':<22} {DEFAULT_ARTIFACT_REL_PATHS[ARTIFACT_LIQUIDITY_PANEL]}",
        f"  {ARTIFACT_OPTION_SURFACE_META + ':':<22} {DEFAULT_ARTIFACT_REL_PATHS[ARTIFACT_OPTION_SURFACE_META]}",
        f"  {ARTIFACT_OPTION_SURFACE_QUOTES + ':':<22} {DEFAULT_ARTIFACT_REL_PATHS[ARTIFACT_OPTION_SURFACE_QUOTES]}",
        "",
        f"snapshot_id (preview, display-only):  {_snapshot_id_preview(ctx)}",
        "",
        "execution: none",  # TEMP: plan/dry-run only; refresh (C8) shows planned vs actual execution
    ]

    # C2 warns but does not fail — real enforcement arrives with refresh execution (C8).
    # TEMP: WARN lines mentioning "deferred until ... (C8)" — remove or replace at C8 closeout.
    if ctx.mode == "backfill" and (args.start_date is None or args.end_date is None):
        lines.insert(
            lines.index("Provisional high-level Stage A plan:"),
            "WARN: backfill date-window enforcement deferred until refresh execution (C8)",
        )
        lines.insert(
            lines.index("Provisional high-level Stage A plan:"),
            "",
        )

    if ctx.mode == "repair" and not args.sample_tickers:
        insert_at = lines.index("Provisional high-level Stage A plan:")
        lines.insert(insert_at, "WARN: repair ticker-scope enforcement deferred until refresh execution (C8)")
        lines.insert(insert_at, "")

    skip_notes: list[str] = []
    if getattr(args, "skip_surface", False):
        skip_notes.append("--skip-surface set (surface precompute omitted from planned execution)")
    if getattr(args, "skip_splits", False):
        skip_notes.append("--skip-splits set (split fetch/adjust omitted from planned execution)")
    if skip_notes:
        insert_at = lines.index("execution: none")
        for note in reversed(skip_notes):
            lines.insert(insert_at, note)
        lines.insert(insert_at, "")

    return "\n".join(lines)


def cmd_plan(ctx: CliContext, args: argparse.Namespace) -> int:
    """Print provisional plan; always succeeds in C2 (exit 0)."""
    print(render_plan(ctx, args))
    return 0


def cmd_refresh(ctx: CliContext, args: argparse.Namespace) -> int:
    """Dry-run reuses plan; real execution blocked until C8 subprocess wiring."""
    if args.dry_run:
        print(render_plan(ctx, args))
        print("DRY-RUN: no subprocesses executed")
        return 0
    # Non-dry refresh must not return 0 — would imply a completed pipeline run.
    print("refresh execution not implemented until C8", file=sys.stderr)
    return 2


def cmd_validate(_ctx: CliContext, _args: argparse.Namespace) -> int:
    """Stub: exit 2 so callers do not treat C2 as a passing validation run."""
    print("validate not implemented until C3", file=sys.stderr)
    return 2


def cmd_split_audit(_ctx: CliContext, _args: argparse.Namespace) -> int:
    """Stub: exit 2 until C8 wires ``scripts/audit_adjusted_liquid.py`` (C5 audit script exists)."""
    print(
        "split-audit CLI not wired until C8; run scripts/audit_adjusted_liquid.py "
        "(see docs/sprint_memos/004_c5_adjusted_liquid.md)",
        file=sys.stderr,
    )
    return 2


def cmd_surface_audit(_ctx: CliContext, _args: argparse.Namespace) -> int:
    """Stub: exit 2 until C6 surface audit report exists."""
    print("surface-audit not implemented until C6", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns int exit code for tests (``if __name__`` uses SystemExit).

    Flow: parse → resolve context (as-of + paths) → dispatch handler.
    argparse raises SystemExit on usage errors; we convert that to return code 2.
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)

    try:
        ctx = build_cli_context(args)
    except ValueError as exc:
        # Unresolvable ORATS day or resolver validation failure → operator-facing stderr.
        print(exc, file=sys.stderr)
        return 2

    handlers = {
        "plan": cmd_plan,
        "refresh": cmd_refresh,
        "validate": cmd_validate,
        "split-audit": cmd_split_audit,
        "surface-audit": cmd_surface_audit,
    }
    handler = handlers[ctx.command]
    return handler(ctx, args)


if __name__ == "__main__":
    raise SystemExit(main())
