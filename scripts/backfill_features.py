"""Sprint 005 D3 — standalone weekly Momentum/CVG feature backfill.

Block 1: configuration loading, accepted-D2 input validation, clean-Git provenance.
Block 2: one-window Momentum/CVG computation and staging-file write.
Later blocks add the 281-window loop, publication, and receipt.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.snapshot_foundation import sha256_file
from src.features.base import FeatureDataContext
from src.features.cvg_calculator import CVGCalculator
from src.features.momentum_calculator import MomentumCalculator
from src.features.straddle_observations import a1_key_digest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMMIT_SHA_PATTERN_LEN = 40
_MUTABLE_CACHE_ROOT = Path(r"C:/MomentumCVG_env/cache")

# Exact approved schemas from configs/feature_backfill_v1.json (order matters).
_APPROVED_REQUIRED_COLUMNS = [
    "ticker",
    "entry_date",
    "return_pct",
    "entry_iv",
    "realized_volatility",
    "vol_gap",
    "expiry_date",
]
_APPROVED_OUTPUT_COLUMNS_PER_WINDOW = [
    "ticker",
    "date",
    "mom_{max}_{min}_mean",
    "mom_{max}_{min}_count",
    "cvg_{max}_{min}",
    "cvg_count_{max}_{min}",
]


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
    if windows[0] != (6, 2):
        raise ValueError(
            f"frozen grid must start with (6, 2), got {windows[0]}"
        )
    if windows[-1] != (60, 24):
        raise ValueError(
            f"frozen grid must end with (60, 24), got {windows[-1]}"
        )

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
    if momentum_min_periods != 1:
        raise ValueError(
            f"momentum.min_periods must be 1, got {momentum_min_periods}"
        )
    if cvg_min_periods != 1:
        raise ValueError(f"cvg.min_periods must be 1, got {cvg_min_periods}")

    input_schema = _require_mapping(
        raw.get("input_schema"), "feature config.input_schema"
    )
    required_columns = _require_str_list(
        input_schema, "required_columns", "input_schema"
    )
    output_columns = _require_str_list(
        raw, "output_columns_per_window", "feature config"
    )
    if required_columns != _APPROVED_REQUIRED_COLUMNS:
        raise ValueError(
            "input_schema.required_columns must exactly match the approved "
            f"v1 schema {_APPROVED_REQUIRED_COLUMNS!r}, got {required_columns!r}"
        )
    if output_columns != _APPROVED_OUTPUT_COLUMNS_PER_WINDOW:
        raise ValueError(
            "output_columns_per_window must exactly match the approved "
            f"v1 schema {_APPROVED_OUTPUT_COLUMNS_PER_WINDOW!r}, "
            f"got {output_columns!r}"
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

    observations = observations.loc[:, list(required_columns)].copy()

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


def render_output_columns(
    templates: Sequence[str], max_lag: int, min_lag: int
) -> list[str]:
    """Render approved per-window column templates for one window."""
    rendered: list[str] = []
    for template in templates:
        if "{max}" in template or "{min}" in template:
            rendered.append(
                template.replace("{max}", str(max_lag)).replace("{min}", str(min_lag))
            )
        else:
            rendered.append(template)
    return rendered


def _normalized_key_pairs(
    frame: pd.DataFrame, *, ticker_col: str, date_col: str, label: str
) -> list[tuple[str, pd.Timestamp]]:
    """Return ordered unique `(ticker, normalized date)` pairs or fail."""
    if ticker_col not in frame.columns or date_col not in frame.columns:
        raise ValueError(f"{label} missing key columns {ticker_col!r}/{date_col!r}")
    if frame[ticker_col].isna().any() or frame[date_col].isna().any():
        raise ValueError(f"{label} contains null {ticker_col}/{date_col} keys")
    tickers = frame[ticker_col].astype(str)
    dates = pd.to_datetime(frame[date_col]).dt.normalize()
    pairs = list(zip(tickers.tolist(), dates.tolist(), strict=True))
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{label} contains duplicate ({ticker_col}, {date_col}) keys")
    return pairs


def _assert_key_set_equal(
    actual_pairs: Sequence[tuple[str, pd.Timestamp]],
    expected_pairs: Sequence[tuple[str, pd.Timestamp]],
    *,
    label: str,
) -> None:
    actual_set = set(actual_pairs)
    expected_set = set(expected_pairs)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} keys are not exactly equal to the canonical D2 key set; "
            f"missing={missing[:5]!r} unexpected={unexpected[:5]!r}"
        )
    if len(actual_pairs) != len(expected_pairs):
        raise ValueError(
            f"{label} key count {len(actual_pairs)} != canonical D2 key count "
            f"{len(expected_pairs)}"
        )


def compute_one_window_features(
    observations: pd.DataFrame,
    window: tuple[int, int],
    config: FeatureBackfillConfig,
) -> pd.DataFrame:
    """Compute one window's six-column publish frame from a validated D2 panel."""
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError(f"window must be a (max_lag, min_lag) tuple, got {window!r}")
    max_lag, min_lag = int(window[0]), int(window[1])
    if max_lag <= min_lag:
        raise ValueError(f"window must satisfy max_lag > min_lag, got {(max_lag, min_lag)}")
    if (max_lag, min_lag) not in config.windows:
        raise ValueError(
            f"window {(max_lag, min_lag)} is not in the approved config window grid"
        )
    if list(observations.columns) != list(config.required_columns):
        raise ValueError(
            "observations columns must exactly match the validated required_columns "
            f"{config.required_columns!r}, got {list(observations.columns)!r}"
        )

    canonical_pairs = _normalized_key_pairs(
        observations, ticker_col="ticker", date_col="entry_date", label="D2 observations"
    )
    start_date = pd.to_datetime(observations["entry_date"]).min()
    end_date = pd.to_datetime(observations["entry_date"]).max()
    context = FeatureDataContext(straddle_history=observations)

    momentum = MomentumCalculator(
        windows=[(max_lag, min_lag)],
        min_periods=config.momentum_min_periods,
    ).calculate_bulk(
        context,
        start_date=start_date,
        end_date=end_date,
        tickers=None,
    )
    cvg = CVGCalculator(
        windows=[(max_lag, min_lag)],
        min_periods=config.cvg_min_periods,
    ).calculate_bulk(
        context,
        start_date=start_date,
        end_date=end_date,
        tickers=None,
    )

    mom_pairs = _normalized_key_pairs(
        momentum, ticker_col="ticker", date_col="date", label="Momentum output"
    )
    cvg_pairs = _normalized_key_pairs(
        cvg, ticker_col="ticker", date_col="date", label="CVG output"
    )
    _assert_key_set_equal(mom_pairs, canonical_pairs, label="Momentum output")
    _assert_key_set_equal(cvg_pairs, canonical_pairs, label="CVG output")

    publish_columns = render_output_columns(
        config.output_columns_per_window, max_lag, min_lag
    )
    mom_mean_col = f"mom_{max_lag}_{min_lag}_mean"
    mom_count_col = f"mom_{max_lag}_{min_lag}_count"
    cvg_col = f"cvg_{max_lag}_{min_lag}"
    cvg_count_col = f"cvg_count_{max_lag}_{min_lag}"
    for col in (mom_mean_col, mom_count_col):
        if col not in momentum.columns:
            raise ValueError(f"Momentum output missing required column {col!r}")
    for col in (cvg_col, cvg_count_col):
        if col not in cvg.columns:
            raise ValueError(f"CVG output missing required column {col!r}")

    mom_part = momentum.loc[:, ["ticker", "date", mom_mean_col, mom_count_col]].copy()
    mom_part["date"] = pd.to_datetime(mom_part["date"]).dt.normalize()
    cvg_part = cvg.loc[:, ["ticker", "date", cvg_col, cvg_count_col]].copy()
    cvg_part["date"] = pd.to_datetime(cvg_part["date"]).dt.normalize()

    merged = mom_part.merge(
        cvg_part,
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged_pairs = _normalized_key_pairs(
        merged, ticker_col="ticker", date_col="date", label="merged features"
    )
    _assert_key_set_equal(merged_pairs, canonical_pairs, label="merged features")
    if len(merged) != len(canonical_pairs):
        raise ValueError(
            f"merged feature row count {len(merged)} != canonical D2 key count "
            f"{len(canonical_pairs)}"
        )

    out = merged.loc[:, publish_columns].sort_values(
        ["ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)
    if list(out.columns) != publish_columns:
        raise ValueError(
            f"published columns {list(out.columns)!r} != required {publish_columns!r}"
        )
    return out


def write_staging_feature_file(
    frame: pd.DataFrame,
    staging_dir: Path | str,
    window: tuple[int, int],
) -> Path:
    """Write one six-column window frame into an existing staging directory."""
    staging = Path(staging_dir)
    if not staging.is_dir():
        raise ValueError(f"staging directory does not exist: {staging}")
    max_lag, min_lag = int(window[0]), int(window[1])
    target = staging / f"features_{max_lag}_{min_lag}.parquet"
    if target.exists():
        raise ValueError(f"refusing to overwrite existing staging file: {target}")
    frame.to_parquet(target, index=False, compression="snappy")
    return target.resolve()
