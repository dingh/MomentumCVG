"""Sprint 006 D1 — documented command for the frozen baseline runner.

Thin CLI over ``src.backtest.sprint006_baseline``. Every frozen contract run
(diagnostic mid and primary cross) always executes; there is no fill selector and
no override for any frozen economic parameter.

Examples
--------
    python scripts/run_sprint006_baseline.py --output-dir C:/MomentumCVG_env/runs/d1_check --dry-run
    python scripts/run_sprint006_baseline.py --output-dir C:/MomentumCVG_env/runs/sprint006_baseline_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.backtest.sprint006_baseline import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    ContractError,
    load_contract,
    preflight,
    run_baseline,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Sprint 006 baseline (mid + cross) on accepted artifacts."
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=str(DEFAULT_CONTRACT_PATH),
        help="Path to the frozen D0 contract JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run output directory; must not already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate contract, configs, accepted paths, and identity only; run nothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    command = [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])]

    try:
        if args.dry_run:
            checked = preflight(load_contract(args.contract))
            print(f"contract: {checked.contract.path}")
            print(f"contract identity: {checked.contract.contract_id} "
                  f"v{checked.contract.contract_version} ({checked.contract.status})")
            print(f"contract sha256: {checked.contract.sha256}")
            for config in checked.configs:
                print(f"run: {config.run_id} fill={config.fill.label} "
                      f"dates={config.start_date}..{config.end_date}")
            print(f"features dir: {checked.data_paths.resolved_features_dir}")
            print(f"surface meta: {checked.data_paths.resolved_surface_meta_path}")
            print(f"surface quotes: {checked.data_paths.resolved_surface_quotes_path}")
            print(f"liquidity panel: {checked.data_paths.resolved_liquidity_panel_path}")
            print("dry run: no economic execution performed")
            return 0

        outcome = run_baseline(
            contract_path=args.contract,
            output_dir=args.output_dir,
            command=command,
        )
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"run dir: {outcome['run_dir']}")
    for run in outcome["runs"]:
        print(f"run: {run['run_id']} fill={run['fill_label']} "
              f"trade_log_rows={run['trade_log_rows']}")
        for name, path in sorted(run["outputs"].items()):
            print(f"  {name}: {path}")
    print(f"receipt: {outcome['receipt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
