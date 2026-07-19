"""Weekly input snapshot operator CLI (Sprint 004 C8.3B — permanent contract).

Purpose
-------
Operator entrypoint for the resumable cold input backfill: a fresh snapshot is
rebuilt from raw ORATS ZIPs through four stages — liquidity → adjusted → spot →
surface — then atomically published (``<BUILD_ID>.building`` → ``<BUILD_ID>``).

Command shapes (permanent C8.3B contract)
-----------------------------------------
New run::

    refresh --mode backfill --snapshots-root DIR --raw-root DIR \
        --start-date D --as-of D --workers N

Resume::

    refresh --resume BUILD_ID --snapshots-root DIR --workers N

What this commit executes
-------------------------
Only ``plan`` and ``refresh --dry-run`` succeed (exit 0). A non-dry ``refresh``
parses and validates its arguments, then exits 2 with a clear message: the
producer, marker, and publication wiring lands in the next C8.3B commit. The
blocked path creates **no** ``.building`` root, run config, raw inventory, or
lock state. The snapshot preparation/resume functions live in
``src.data.snapshot_orchestrator`` and are exercised directly by unit tests.

* ``--mode backfill`` is the only executable mode; ``incremental`` and
  ``repair`` are deferred (exit 2).
* Resume rejects explicitly supplied identity-defining flags (``--mode``,
  ``--raw-root``, ``--start-date``, ``--as-of``); ``--workers`` may differ.
* ``validate`` / ``split-audit`` / ``surface-audit`` stay blocked (exit 2) and
  point to the existing standalone audit scripts.
* The ORATS API token is read only from ``ORATS_API_TOKEN`` — there is no
  token flag, and no copy-source / evidence-ID / scope / skip-stage /
  security-master flags.

Exit codes
----------
0   = ``plan`` or ``refresh --dry-run`` success
1   = runtime / producer / gate failure (reserved for the next commit)
2   = usage error, config/corruption failure, or unsupported operation
130 = interrupted (KeyboardInterrupt)

Design: docs/tmp/c8_3b_resumable_cold_backfill_design.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# ── resolve project root so src/ imports work regardless of cwd ──────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# NOTE: this module deliberately does not import src.data.snapshot_orchestrator
# yet. The blocked non-dry refresh must not be able to scan inventory, create
# a .building root, or acquire a lock; the wiring arrives in the next commit.

EXIT_OK = 0
EXIT_RUNTIME = 1  # reserved: producer/gate/runtime failure (next commit)
EXIT_USAGE = 2  # usage / config / corruption / unsupported operation
EXIT_INTERRUPT = 130

_RESUME_IDENTITY_FLAGS = ("--mode", "--raw-root", "--start-date", "--as-of")

DRY_RUN_BANNER = "DRY-RUN: no inventory scan, writes, lock, or producers executed"


def _parse_iso_date(value: str) -> date:
    """Argparse type converter; surfaces bad CLI dates as exit code 2."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from exc


def _add_refresh_like_args(parser: argparse.ArgumentParser) -> None:
    """Shared flags for ``plan`` and ``refresh`` (all optional at parse time).

    Requiredness depends on new-run vs resume vs dry-run, so it is enforced in
    the command handlers (not by argparse) to keep error text precise.
    """
    parser.add_argument(
        "--mode",
        choices=("incremental", "backfill", "repair"),
        default=None,
        help="Refresh mode; only 'backfill' is executable in C8.3B",
    )
    parser.add_argument(
        "--resume",
        metavar="BUILD_ID",
        default=None,
        help="Resume the named <BUILD_ID>.building run (refresh only)",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        default=None,
        help="Directory that holds <BUILD_ID>.building / final snapshot roots",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Raw ORATS ZIP root (*/ORATS_SMV_Strikes_YYYYMMDD.zip)",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Requested output start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=None,
        help="Requested as-of date (YYYY-MM-DD); resolved from frozen raw inventory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker process count (positive; may differ on resume)",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse tree; ``argv`` hook makes ``main()`` testable."""
    parser = argparse.ArgumentParser(
        description=(
            "Weekly input snapshot operator CLI (Sprint 004 C8.3B cold backfill)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="Describe the four-stage cold backfill plan (no execution)"
    )
    _add_refresh_like_args(plan_parser)

    refresh_parser = subparsers.add_parser(
        "refresh", help="Run or resume a cold input backfill"
    )
    _add_refresh_like_args(refresh_parser)
    refresh_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cold backfill plan only; perform no scan, write, or lock",
    )

    subparsers.add_parser(
        "validate", help="Blocked: use the standalone audit scripts (exit 2)"
    )
    subparsers.add_parser(
        "split-audit", help="Blocked: use scripts/audit_adjusted_liquid.py (exit 2)"
    )
    subparsers.add_parser(
        "surface-audit",
        help="Blocked: use scripts/audit_option_surface_artifacts.py (exit 2)",
    )

    return parser.parse_args(argv)


def render_plan(args: argparse.Namespace) -> str:
    """Permanent four-stage cold backfill plan (``plan`` and ``refresh --dry-run``)."""
    mode = args.mode or "backfill"
    lines: list[str] = [
        "=== Weekly input snapshot plan - cold backfill (C8.3B) ===",
        f"mode:            {mode}"
        + ("" if mode == "backfill" else "  [execute-unsupported: exit 2]"),
        f"snapshots_root:  {args.snapshots_root if args.snapshots_root else '(required for execution)'}",
        f"raw_root:        {args.raw_root if args.raw_root else '(required for a new run)'}",
        f"start_date:      {args.start_date.isoformat() if args.start_date else '(required for a new run)'}",
        f"as_of:           {args.as_of.isoformat() if args.as_of else '(required for a new run)'}",
        f"workers:         {args.workers if args.workers is not None else 1}",
        f"resume:          {args.resume if args.resume else '(new run)'}",
        "",
        "Source: fresh raw ORATS ZIPs only (no copy of existing C4/C5 outputs).",
        "Freeze: raw inventory + immutable run_config.json in <BUILD_ID>.building,",
        "        guarded by the sibling OS lock <BUILD_ID>.lock.",
        "",
        "Stages (rebuilt inside <BUILD_ID>.building):",
        "  1. liquidity  - C4 rebuild from frozen raw ZIPs, equity-only"
        " classification, strict C7 gate",
        "  2. adjusted   - split fetch + exact-ZIP split adjustment, C5 audit gate",
        "  3. spot       - spot extraction from certified adjusted chains, Gate SP",
        "  4. surface    - weekly A1/A2 option-surface precompute, Gate SF + C6 audit",
        "",
        "Then: final cross-stage validation -> schema-v1 manifest ->"
        " atomic publish (rename <BUILD_ID>.building -> <BUILD_ID>).",
        "",
        "Modes: only 'backfill' is executable in C8.3B;"
        " 'incremental' and 'repair' are execute-unsupported (exit 2).",
        "Resume: refresh --resume BUILD_ID reuses the frozen run intent;"
        " identity flags are rejected, --workers may change.",
        "Token: ORATS_API_TOKEN environment variable only (no token flag).",
    ]
    return "\n".join(lines)


def _validate_workers(args: argparse.Namespace) -> int | None:
    """Return exit code on invalid ``--workers``; ``None`` when acceptable."""
    if args.workers is not None and args.workers <= 0:
        print(
            f"--workers must be a positive integer; got {args.workers}",
            file=sys.stderr,
        )
        return EXIT_USAGE
    return None


def cmd_plan(args: argparse.Namespace) -> int:
    """Print the permanent four-stage cold plan; performs no filesystem work."""
    failed = _validate_workers(args)
    if failed is not None:
        return failed
    print(render_plan(args))
    return EXIT_OK


def cmd_refresh(args: argparse.Namespace) -> int:
    """Validate the permanent refresh contract; execution is deferred.

    Validation order: workers → resume identity-flag rejection → dry-run →
    mode support → new-run required flags → blocked-execution message. The
    blocked path never creates ``.building``, config, inventory, or lock state.
    """
    failed = _validate_workers(args)
    if failed is not None:
        return failed

    if args.resume is not None:
        supplied = [
            flag
            for flag, value in (
                ("--mode", args.mode),
                ("--raw-root", args.raw_root),
                ("--start-date", args.start_date),
                ("--as-of", args.as_of),
            )
            if value is not None
        ]
        if supplied:
            print(
                "--resume rejects identity-defining flags "
                f"({', '.join(supplied)}); the frozen run_config.json of "
                f"{args.resume} is authoritative. Only --snapshots-root and "
                "--workers are accepted on resume.",
                file=sys.stderr,
            )
            return EXIT_USAGE

    if args.dry_run:
        print(render_plan(args))
        print(DRY_RUN_BANNER)
        return EXIT_OK

    if args.resume is not None:
        if args.snapshots_root is None:
            print("--resume requires --snapshots-root", file=sys.stderr)
            return EXIT_USAGE
        print(
            f"refresh --resume {args.resume} is validated but blocked: "
            "producer, marker, and publication wiring is deferred to the next "
            "C8.3B commit. No snapshot state was opened or modified.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.mode is None:
        print(
            "refresh requires --mode backfill (or --resume BUILD_ID)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.mode in ("incremental", "repair"):
        print(
            f"refresh --mode {args.mode} is deferred: C8.3B executes cold "
            "backfill only. Use --mode backfill.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    missing = [
        flag
        for flag, value in (
            ("--snapshots-root", args.snapshots_root),
            ("--raw-root", args.raw_root),
            ("--start-date", args.start_date),
            ("--as-of", args.as_of),
        )
        if value is None
    ]
    if missing:
        print(
            f"refresh --mode backfill requires {', '.join(missing)}",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Arguments are valid, but a partial refresh must not masquerade as a
    # completed backfill: block before any inventory scan, write, or lock.
    print(
        "refresh --mode backfill is validated but blocked: producer, marker, "
        "and publication wiring is deferred to the next C8.3B commit. No "
        ".building root, run config, inventory, or lock was created.",
        file=sys.stderr,
    )
    return EXIT_USAGE


def cmd_validate(_args: argparse.Namespace) -> int:
    """Blocked: point to the standalone audits instead of a silent global scan."""
    print(
        "validate is not wired in C8.3B; use the standalone audits "
        "(scripts/audit_adjusted_liquid.py, scripts/audit_option_surface_artifacts.py, "
        "scripts/audit_pit_universe.py)",
        file=sys.stderr,
    )
    return EXIT_USAGE


def cmd_split_audit(_args: argparse.Namespace) -> int:
    """Blocked: the standalone C5 audit script is the supported path."""
    print(
        "split-audit is not wired in C8.3B; run scripts/audit_adjusted_liquid.py "
        "(see docs/sprint_memos/004_c5_adjusted_liquid.md)",
        file=sys.stderr,
    )
    return EXIT_USAGE


def cmd_surface_audit(_args: argparse.Namespace) -> int:
    """Blocked: the standalone C6 audit script is the supported path."""
    print(
        "surface-audit is not wired in C8.3B; run "
        "scripts/audit_option_surface_artifacts.py "
        "(see docs/sprint_memos/004_c6_option_surface.md)",
        file=sys.stderr,
    )
    return EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns an int exit code (``if __name__`` raises SystemExit)."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_OK
        return int(code)

    handlers = {
        "plan": cmd_plan,
        "refresh": cmd_refresh,
        "validate": cmd_validate,
        "split-audit": cmd_split_audit,
        "surface-audit": cmd_surface_audit,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPT


if __name__ == "__main__":
    raise SystemExit(main())
