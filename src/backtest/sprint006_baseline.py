"""Sprint 006 D1 — thin frozen-contract adapter for the canonical Surface runner.

This module maps the accepted D0 contract (``configs/sprint006_baseline_v1.json``)
onto existing objects and nothing else:

    contract JSON → BacktestRunConfig (one per frozen run) + SurfaceDataPaths
                  → SurfaceRunner.run_single_config()
                  → existing trade_log / date_summary / run_summary + a light receipt

It contains **no** signal, selection, pricing, sizing, settlement, or metric
logic; ``SurfaceRunner.run_single_config`` remains the only economic engine.
Both frozen runs (diagnostic ``mid`` and primary ``cross``) always execute — the
run set comes entirely from the contract's ``runs`` list.

Identity handling is deliberately proportional: clean Git HEAD, contract
id/version/status, the effective configs, and a recorded digest of the contract
bytes as they exist on disk. The digest is *recorded*, not compared against the
D0 line-ending-normalised value, so no portability machinery is needed here.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.backtest.option_surface import FillAssumption
from src.backtest.run_config import BacktestRunConfig
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner, SurfaceRunResult

EXPERIMENT_ID = "sprint006_baseline_v1"
DEFAULT_CONTRACT_PATH = Path("configs/sprint006_baseline_v1.json")

# Producer cache root that must never serve as an accepted input (D0
# ``mutable_cache_forbidden``). Containment is checked against this root only:
# accepted snapshot artifacts legitimately live under ``<snapshot>/cache/surface``.
MUTABLE_CACHE_ROOT = Path(r"C:/MomentumCVG_env/cache")

# Fill labels that must both be present so mid (diagnostic) and cross (primary)
# always run together.
REQUIRED_FILL_LABELS = frozenset({"mid", "cross"})

# Work that D1 explicitly does not deliver; recorded in the receipt so a reader
# never mistakes these outputs for a complete Sprint 006 result.
DEFERRED_TO_LATER_DELIVERABLES = (
    "joint Momentum+CVG count eligibility (D2)",
    "A1 expected-date calendar and date-status table (D2)",
    "all-leg max_leg_spread_pct on iron-fly bodies (D2)",
    "decision-quality report and dual return views (D3)",
    "real-data smoke, manual trade sample, and full-history execution (D4)",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CONFIG_FIELD_NAMES = frozenset(f.name for f in fields(BacktestRunConfig))
_FEATURE_COLUMN_FIELDS = ("momentum_col", "cvg_col", "count_col")
_REQUIRED_ACCEPTED_INPUT_FILES = (
    "baseline_feature_file",
    "a1_surface_meta",
    "a2_surface_quotes",
    "liquidity_panel",
    "manifest",
    "d3_receipt",
)


class ContractError(ValueError):
    """The frozen contract, its accepted inputs, or the run identity is unusable."""


@dataclass(frozen=True)
class LoadedContract:
    """The frozen D0 contract plus the identity fields recorded in the receipt."""

    path: Path
    sha256: str
    contract_id: str
    contract_version: int
    status: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class BaselinePreflight:
    """Validated contract, twin run configs, and accepted data paths."""

    contract: LoadedContract
    configs: Tuple[BacktestRunConfig, ...]
    data_paths: SurfaceDataPaths
    accepted_inputs: Mapping[str, Optional[Path]]


# ---------------------------------------------------------------------------
# Contract loading and identity
# ---------------------------------------------------------------------------

def load_contract(
    contract_path: Path | str,
    *,
    expected_contract_id: str = EXPERIMENT_ID,
) -> LoadedContract:
    """Load and identity-check the frozen contract; record its on-disk digest."""
    path = Path(contract_path).resolve()
    if not path.is_file():
        raise ContractError(f"contract file not found: {path}")

    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"contract is not valid UTF-8 JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"contract must be a JSON object: {path}")

    contract_id = payload.get("contract_id")
    if contract_id != expected_contract_id:
        raise ContractError(
            f"contract_id must be {expected_contract_id!r}, got {contract_id!r}"
        )

    version = payload.get("contract_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ContractError(f"contract_version must be an integer >= 1, got {version!r}")

    status = payload.get("status")
    if status != "accepted":
        raise ContractError(f"contract status must be 'accepted', got {status!r}")

    return LoadedContract(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        contract_id=contract_id,
        contract_version=version,
        status=status,
        payload=payload,
    )


def clean_repo_sha() -> str:
    """Return the HEAD SHA only when the working tree is clean."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        pending = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot determine repository revision from Git: {exc}") from exc

    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head.lower()):
        raise ContractError(f"git HEAD is not a 40-character commit SHA: {head!r}")
    if pending:
        raise ContractError(
            "refusing to write baseline artifacts from a dirty working tree; "
            f"commit or stash first (HEAD {head}):\n{pending}"
        )
    return head


# ---------------------------------------------------------------------------
# Contract → BacktestRunConfig
# ---------------------------------------------------------------------------

def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"contract {key!r} must be a JSON object")
    return value


def _contract_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ContractError(f"contract {label} must be an ISO date string, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"contract {label} is not an ISO date: {value!r}") from exc


def _fill_from_contract(run: Mapping[str, Any], run_label: str) -> FillAssumption:
    fill = run.get("fill")
    if not isinstance(fill, dict):
        raise ContractError(f"contract run {run_label} must define a 'fill' object")
    try:
        return FillAssumption(
            buy_alpha=float(fill["buy_alpha"]),
            sell_alpha=float(fill["sell_alpha"]),
            label=str(fill["label"]),
        )
    except KeyError as exc:
        raise ContractError(
            f"contract run {run_label} fill is missing required key {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ContractError(f"contract run {run_label} has an invalid fill: {exc}") from exc


def build_run_configs(contract: LoadedContract) -> Tuple[BacktestRunConfig, ...]:
    """Build one ``BacktestRunConfig`` per frozen run, in contract order.

    Only keys that are recognized ``BacktestRunConfig`` fields are mapped; the
    contract's ``*_note`` / ``*_intent`` / ``*_role`` prose keys are ignored by
    construction rather than unpacked.
    """
    payload = contract.payload
    shared = _require_mapping(payload, "shared_run_config")
    window = _require_mapping(payload, "feature_window")
    periods = _require_mapping(payload, "periods")

    if window.get("search") is not False:
        raise ContractError("contract feature_window.search must be false (no parameter search)")

    base: Dict[str, Any] = {
        key: value for key, value in shared.items() if key in _RUN_CONFIG_FIELD_NAMES
    }
    for field_name in _FEATURE_COLUMN_FIELDS:
        if not isinstance(window.get(field_name), str):
            raise ContractError(f"contract feature_window.{field_name} must be a string")
        base[field_name] = window[field_name]

    base["start_date"] = _contract_date(shared.get("start_date"), "shared_run_config.start_date")
    base["end_date"] = _contract_date(shared.get("end_date"), "shared_run_config.end_date")
    run_start = _contract_date(periods.get("run_start_date"), "periods.run_start_date")
    run_end = _contract_date(periods.get("run_end_date"), "periods.run_end_date")
    if (base["start_date"], base["end_date"]) != (run_start, run_end):
        raise ContractError(
            "contract run dates disagree: shared_run_config "
            f"{base['start_date']}..{base['end_date']} vs periods {run_start}..{run_end}"
        )

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ContractError("contract 'runs' must be a non-empty list")

    configs: List[BacktestRunConfig] = []
    labels: List[str] = []
    primary_count = 0
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ContractError(f"contract runs[{index}] must be a JSON object")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ContractError(f"contract runs[{index}].run_id must be a non-empty string")

        fill = _fill_from_contract(run, run_id)
        labels.append(fill.label)
        if run.get("primary_decision_view") is True:
            primary_count += 1

        kwargs = dict(base)
        kwargs["run_id"] = run_id
        kwargs["fill"] = fill
        if "cost_model" in run:
            kwargs["cost_model"] = run["cost_model"]
        try:
            configs.append(BacktestRunConfig(**kwargs))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"contract runs[{index}] ({run_id}) is not constructible: {exc}") from exc

    if len(set(labels)) != len(labels):
        raise ContractError(f"contract run fill labels must be unique, got {labels}")
    if not REQUIRED_FILL_LABELS.issubset(labels):
        raise ContractError(
            f"contract runs must cover fill labels {sorted(REQUIRED_FILL_LABELS)}, got {labels}"
        )
    if primary_count != 1:
        raise ContractError(
            f"contract must mark exactly one run primary_decision_view=true, got {primary_count}"
        )
    return tuple(configs)


# ---------------------------------------------------------------------------
# Accepted input paths
# ---------------------------------------------------------------------------

def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _accepted_path(accepted_inputs: Mapping[str, Any], key: str) -> Path:
    value = accepted_inputs.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"contract accepted_inputs.{key} must be a non-empty path string")
    path = Path(value).resolve()
    if _is_inside(path, MUTABLE_CACHE_ROOT.resolve()):
        raise ContractError(
            f"refusing accepted_inputs.{key} under the mutable producer cache root "
            f"{MUTABLE_CACHE_ROOT}: {path}"
        )
    return path


def resolve_accepted_inputs(contract: LoadedContract) -> Dict[str, Optional[Path]]:
    """Resolve accepted input paths, refusing the mutable producer cache root."""
    accepted = _require_mapping(contract.payload, "accepted_inputs")
    if accepted.get("mutable_cache_forbidden") is not True:
        raise ContractError("contract accepted_inputs.mutable_cache_forbidden must be true")
    if accepted.get("earnings_path") is not None:
        raise ContractError(
            "contract accepted_inputs.earnings_path must be null (earnings filtering is off)"
        )

    resolved: Dict[str, Optional[Path]] = {"earnings_path": None}
    for key in ("features_dir", *_REQUIRED_ACCEPTED_INPUT_FILES):
        resolved[key] = _accepted_path(accepted, key)

    if not resolved["features_dir"].is_dir():
        raise ContractError(f"accepted features_dir does not exist: {resolved['features_dir']}")
    for key in _REQUIRED_ACCEPTED_INPUT_FILES:
        if not resolved[key].is_file():
            raise ContractError(f"accepted input {key} does not exist: {resolved[key]}")
    return resolved


def preflight(contract: LoadedContract) -> BaselinePreflight:
    """Validate contract, twin configs, and accepted paths without running anything."""
    configs = build_run_configs(contract)
    accepted_inputs = resolve_accepted_inputs(contract)

    data_paths = SurfaceDataPaths(
        # Every artifact path is explicit, so cache_dir is never consulted; it is
        # still pointed away from the mutable producer-cache default.
        cache_dir=accepted_inputs["features_dir"].parent,
        features_dir=accepted_inputs["features_dir"],
        liquidity_panel_path=accepted_inputs["liquidity_panel"],
        surface_meta_path=accepted_inputs["a1_surface_meta"],
        surface_quotes_path=accepted_inputs["a2_surface_quotes"],
        earnings_path=None,
    )

    expected_feature_file = accepted_inputs["baseline_feature_file"]
    for config in configs:
        resolved = data_paths.features_path_for_config(config).resolve()
        if resolved != expected_feature_file:
            raise ContractError(
                f"run {config.run_id} resolves feature file {resolved}, "
                f"which is not the accepted baseline feature file {expected_feature_file}"
            )
    return BaselinePreflight(
        contract=contract,
        configs=configs,
        data_paths=data_paths,
        accepted_inputs=accepted_inputs,
    )


# ---------------------------------------------------------------------------
# Output writing and receipt
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, int):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    # JSON has no NaN/Infinity; record them as null rather than emitting invalid JSON.
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None


def effective_config_dump(config: BacktestRunConfig) -> Dict[str, Any]:
    """JSON-ready dump of the config actually handed to the runner."""
    return _jsonable(asdict(config))


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite existing artifact: {path}")


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    _refuse_existing(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def create_run_dir(output_dir: Path | str) -> Path:
    """Create the run output directory, refusing an existing one."""
    run_dir = Path(output_dir).resolve()
    if run_dir.exists():
        raise ContractError(f"refusing to overwrite existing run output directory: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def write_run_outputs(result: SurfaceRunResult, run_dir: Path) -> Dict[str, Path]:
    """Persist the existing result frames/summary; refuse to overwrite artifacts."""
    run_id = result.config.run_id
    targets = {
        "trade_log": run_dir / f"trade_log_{run_id}.parquet",
        "date_summary": run_dir / f"date_summary_{run_id}.parquet",
        "run_summary": run_dir / f"run_summary_{run_id}.json",
    }
    for path in targets.values():
        _refuse_existing(path)

    result.trade_log.to_parquet(targets["trade_log"], index=False)
    result.date_summary.to_parquet(targets["date_summary"], index=False)
    _write_json(_jsonable(dict(result.run_summary)), targets["run_summary"])
    return targets


def build_receipt(
    *,
    preflight_result: BaselinePreflight,
    repo_sha: str,
    run_outputs: Sequence[Tuple[BacktestRunConfig, Dict[str, Path]]],
    command: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble the light run receipt (identity, effective configs, outputs)."""
    contract = preflight_result.contract
    return {
        "deliverable": "sprint006_d1",
        "experiment_id": EXPERIMENT_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": list(command) if command is not None else None,
        "repo_sha": repo_sha,
        "contract": {
            "path": str(contract.path),
            "sha256": contract.sha256,
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "status": contract.status,
        },
        "accepted_inputs": {
            key: (str(path) if path is not None else None)
            for key, path in sorted(preflight_result.accepted_inputs.items())
        },
        "runs": [
            {
                "run_id": config.run_id,
                "fill_label": config.fill.label,
                "effective_config": effective_config_dump(config),
                "outputs": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in sorted(outputs.items())
                },
            }
            for config, outputs in run_outputs
        ],
        "deferred": list(DEFERRED_TO_LATER_DELIVERABLES),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_baseline(
    *,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    output_dir: Path | str,
    command: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run every frozen contract run through ``SurfaceRunner.run_single_config``.

    Returns a summary of written paths and row counts — deliberately no economic
    metrics, so D1 execution cannot double as P&L inspection.
    """
    contract = load_contract(contract_path)
    checked = preflight(contract)
    repo_sha = clean_repo_sha()
    run_dir = create_run_dir(output_dir)

    runner = SurfaceRunner(data_paths=checked.data_paths)
    run_outputs: List[Tuple[BacktestRunConfig, Dict[str, Path]]] = []
    row_counts: Dict[str, int] = {}
    for config in checked.configs:
        result = runner.run_single_config(config)
        run_outputs.append((config, write_run_outputs(result, run_dir)))
        row_counts[config.run_id] = int(len(result.trade_log))

    receipt = build_receipt(
        preflight_result=checked,
        repo_sha=repo_sha,
        run_outputs=run_outputs,
        command=command,
    )
    receipt_path = _write_json(receipt, run_dir / "run_receipt.json")
    return {
        "run_dir": run_dir,
        "receipt_path": receipt_path,
        "runs": [
            {
                "run_id": config.run_id,
                "fill_label": config.fill.label,
                "trade_log_rows": row_counts[config.run_id],
                "outputs": outputs,
            }
            for config, outputs in run_outputs
        ],
    }
