"""Sprint 005 D3 — standalone weekly Momentum/CVG feature backfill.

Block 1 implements configuration loading, accepted-D2 input validation, and
clean-Git provenance helpers only. Later blocks add calculation and publication.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.snapshot_foundation import sha256_file
from src.features.straddle_observations import a1_key_digest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMMIT_SHA_PATTERN_LEN = 40
_MUTABLE_CACHE_ROOT = Path(r"C:/MomentumCVG_env/cache")


@dataclass(frozen=True)
class FeatureBackfillConfig:
    """Validated semantics from ``feature_backfill_v1.json``."""

    spec_version: str
    spec_id: str
    windows: list[tuple[int, int]]
    baseline_window: tuple[int, int]
    momentum_min_periods: int
    cvg_min_periods: int
    required_columns: list[str]
    output_columns_per_window: list[str]
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class ValidatedD2Input:
    """Accepted D2 observations plus the lineage fields used for later receipts."""

    observations: pd.DataFrame
    observations_path: Path
    d2_lineage_path: Path
    snapshot_id: str
    build_id: str
    file_sha256: str
    row_count: int
    key_count: int
    output_key_digest: str


def _require_mapping(obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj


def _require_int(obj: dict[str, Any], key: str, label: str) -> int:
    if key not in obj:
        raise ValueError(f"{label} missing required field {key!r}")
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer, got {value!r}")
    return value


def _require_bool(obj: dict[str, Any], key: str, label: str) -> bool:
    if key not in obj:
        raise ValueError(f"{label} missing required field {key!r}")
    value = obj[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean, got {value!r}")
    return value


def _require_str(obj: dict[str, Any], key: str, label: str) -> str:
    if key not in obj:
        raise ValueError(f"{label} missing required field {key!r}")
    value = obj[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string, got {value!r}")
    return value


def _require_str_list(obj: dict[str, Any], key: str, label: str) -> list[str]:
    if key not in obj:
        raise ValueError(f"{label} missing required field {key!r}")
    value = obj[key]
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{label}.{key} must be a non-empty list of strings")
    return list(value)


def expand_windows_from_bounds(
    *,
    min_lag_start: int,
    min_lag_end: int,
    max_lag_start: int,
    max_lag_end: int,
    step: int,
    require_max_gt_min: bool,
) -> list[tuple[int, int]]:
    """Expand the frozen grid with every bound supplied explicitly."""
    if step <= 0:
        raise ValueError(f"window step must be > 0, got {step}")
    windows: list[tuple[int, int]] = []
    for max_lag in range(max_lag_start, max_lag_end + 1, step):
        for min_lag in range(min_lag_start, min_lag_end + 1, step):
            if require_max_gt_min and max_lag <= min_lag:
                continue
            windows.append((max_lag, min_lag))
    return windows


def load_feature_backfill_config(config_path: Path | str) -> FeatureBackfillConfig:
    """Load and validate ``feature_backfill_v1.json``; no path default."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ValueError(f"feature config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = _require_mapping(raw, "feature config")

    spec_version = _require_str(raw, "spec_version", "feature config")
    if spec_version != "feature_backfill_v1":
        raise ValueError(
            f"unsupported spec_version {spec_version!r}; expected 'feature_backfill_v1'"
        )
    spec_id = _require_str(raw, "spec_id", "feature config")

    windows_cfg = _require_mapping(raw.get("windows"), "feature config.windows")
    min_lag_start = _require_int(windows_cfg, "min_lag_start", "windows")
    min_lag_end = _require_int(windows_cfg, "min_lag_end", "windows")
    max_lag_start = _require_int(windows_cfg, "max_lag_start", "windows")
    max_lag_end = _require_int(windows_cfg, "max_lag_end", "windows")
    step = _require_int(windows_cfg, "step", "windows")
    require_max_gt_min = _require_bool(windows_cfg, "require_max_gt_min", "windows")
    order = _require_str(windows_cfg, "order", "windows")
    expected_count = _require_int(windows_cfg, "expected_count", "windows")

    if order != "max_lag_outer_min_lag_inner":
        raise ValueError(
            f"windows.order must be 'max_lag_outer_min_lag_inner', got {order!r}"
        )
    if not require_max_gt_min:
        raise ValueError("windows.require_max_gt_min must be true")

    windows = expand_windows_from_bounds(
        min_lag_start=min_lag_start,
        min_lag_end=min_lag_end,
        max_lag_start=max_lag_start,
        max_lag_end=max_lag_end,
        step=step,
        require_max_gt_min=require_max_gt_min,
    )
    if len(windows) != expected_count:
        raise ValueError(
            f"expanded window count {len(windows)} != windows.expected_count "
            f"{expected_count}"
        )
    if expected_count != 281:
        raise ValueError(
            f"windows.expected_count must be 281 for feature_backfill_v1, "
            f"got {expected_count}"
        )
    if len(set(windows)) != len(windows):
        raise ValueError("expanded window list contains duplicates")
    if any(max_lag <= min_lag for max_lag, min_lag in windows):
        raise ValueError("every window must satisfy max_lag > min_lag")

    baseline_cfg = _require_mapping(
        raw.get("baseline_window"), "feature config.baseline_window"
    )
    baseline = (
        _require_int(baseline_cfg, "max_lag", "baseline_window"),
        _require_int(baseline_cfg, "min_lag", "baseline_window"),
    )
    if baseline != (42, 8):
        raise ValueError(f"baseline_window must be (42, 8), got {baseline}")
    if baseline not in windows:
        raise ValueError(f"baseline window {baseline} missing from expanded grid")

    momentum_cfg = _require_mapping(raw.get("momentum"), "feature config.momentum")
    cvg_cfg = _require_mapping(raw.get("cvg"), "feature config.cvg")
    momentum_min_periods = _require_int(momentum_cfg, "min_periods", "momentum")
    cvg_min_periods = _require_int(cvg_cfg, "min_periods", "cvg")

    input_schema = _require_mapping(
        raw.get("input_schema"), "feature config.input_schema"
    )
    required_columns = _require_str_list(
        input_schema, "required_columns", "input_schema"
    )
    output_columns = _require_str_list(
        raw, "output_columns_per_window", "feature config"
    )

    return FeatureBackfillConfig(
        spec_version=spec_version,
        spec_id=spec_id,
        windows=windows,
        baseline_window=baseline,
        momentum_min_periods=momentum_min_periods,
        cvg_min_periods=cvg_min_periods,
        required_columns=required_columns,
        output_columns_per_window=output_columns,
        config_path=path,
        config_sha256=sha256_file(path),
    )


def _path_has_cache_segment(path: Path) -> bool:
    return any(part.lower() == "cache" for part in path.parts)


def _refuse_mutable_cache_path(observations_path: Path) -> None:
    resolved = observations_path.resolve()
    try:
        resolved.relative_to(_MUTABLE_CACHE_ROOT.resolve())
        under_env_cache = True
    except ValueError:
        under_env_cache = False
    if under_env_cache or _path_has_cache_segment(resolved):
        raise ValueError(
            "refusing observations path under a mutable cache location: "
            f"{resolved}"
        )


def validate_d2_input(
    *,
    observations_path: Path | str,
    d2_lineage_path: Path | str,
    expected_snapshot_id: str,
    expected_build_id: str,
    required_columns: Sequence[str],
) -> ValidatedD2Input:
    """Validate accepted D2 parquet + lineage against explicit expected identity."""
    obs_path = Path(observations_path).resolve()
    lineage_path = Path(d2_lineage_path).resolve()

    if not obs_path.is_file():
        raise ValueError(f"observations file not found: {obs_path}")
    if not lineage_path.is_file():
        raise ValueError(f"D2 lineage file not found: {lineage_path}")
    if obs_path.parent != lineage_path.parent:
        raise ValueError(
            "observations and D2 lineage must be siblings under the same directory; "
            f"got parents {obs_path.parent} and {lineage_path.parent}"
        )
    _refuse_mutable_cache_path(obs_path)

    with lineage_path.open(encoding="utf-8") as handle:
        lineage = json.load(handle)
    lineage = _require_mapping(lineage, "D2 lineage")

    artifact = _require_str(lineage, "artifact", "D2 lineage")
    if artifact != "straddle_observations_weekly":
        raise ValueError(
            f"D2 lineage.artifact must be 'straddle_observations_weekly', "
            f"got {artifact!r}"
        )

    snapshot_id = _require_str(lineage, "snapshot_id", "D2 lineage")
    build_id = _require_str(lineage, "build_id", "D2 lineage")
    if snapshot_id != expected_snapshot_id:
        raise ValueError(
            f"snapshot_id mismatch: lineage has {snapshot_id!r}, "
            f"expected {expected_snapshot_id!r}"
        )
    if build_id != expected_build_id:
        raise ValueError(
            f"build_id mismatch: lineage has {build_id!r}, "
            f"expected {expected_build_id!r}"
        )

    output = _require_mapping(lineage.get("output"), "D2 lineage.output")
    file_sha256 = _require_str(output, "file_sha256", "output")
    row_count = _require_int(output, "row_count", "output")
    key_count = _require_int(output, "key_count", "output")
    output_key_digest = _require_str(output, "output_key_digest", "output")

    actual_sha = sha256_file(obs_path)
    if actual_sha != file_sha256:
        raise ValueError(
            f"observations file digest mismatch: computed {actual_sha}, "
            f"lineage output.file_sha256 {file_sha256}"
        )

    if not required_columns:
        raise ValueError("required_columns must be a non-empty sequence")

    observations = pd.read_parquet(obs_path)
    missing = [c for c in required_columns if c not in observations.columns]
    if missing:
        raise ValueError(f"observations missing required columns: {missing}")

    if len(observations) != row_count:
        raise ValueError(
            f"observations row count {len(observations)} != "
            f"lineage output.row_count {row_count}"
        )

    if observations["ticker"].isna().any() or observations["entry_date"].isna().any():
        raise ValueError("observations contain null ticker or entry_date keys")

    key_frame = observations.loc[:, ["ticker", "entry_date"]].copy()
    if key_frame.duplicated().any():
        raise ValueError("observations contain duplicate (ticker, entry_date) keys")

    unique_key_count = len(key_frame.drop_duplicates())
    if unique_key_count != key_count:
        raise ValueError(
            f"unique key count {unique_key_count} != "
            f"lineage output.key_count {key_count}"
        )

    recomputed_digest = a1_key_digest(observations)
    if recomputed_digest != output_key_digest:
        raise ValueError(
            f"key digest mismatch: a1_key_digest() produced {recomputed_digest}, "
            f"lineage output.output_key_digest {output_key_digest}"
        )

    return ValidatedD2Input(
        observations=observations,
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        snapshot_id=snapshot_id,
        build_id=build_id,
        file_sha256=file_sha256,
        row_count=row_count,
        key_count=key_count,
        output_key_digest=output_key_digest,
    )


def _git_output(*args: str) -> str:
    """Run a read-only git command in the repository root and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def require_clean_repo_sha() -> str:
    """Return the full HEAD SHA only when the working tree is clean.

    Does not accept a caller-supplied SHA. Does not run at import time.
    """
    try:
        head = _git_output("rev-parse", "HEAD")
        pending = _git_output("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot determine repository revision from Git: {exc}"
        ) from exc

    if len(head) != _COMMIT_SHA_PATTERN_LEN or any(
        ch not in "0123456789abcdef" for ch in head.lower()
    ):
        raise RuntimeError(
            f"git HEAD resolved to {head!r}, which is not a 40-character commit SHA"
        )
    if pending:
        raise RuntimeError(
            "refusing to proceed with a dirty working tree; "
            f"commit or stash changes first (HEAD {head}):\n{pending}"
        )
    return head
