"""Sprint 005 D4 — read-only Momentum/CVG feature quality audit.

Block 1: startup identity gate, per-window coverage/missingness, ready interval,
compact PIT proof, and deterministic JSON output.
Block 2 (later): production execution against accepted D3 artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.snapshot_foundation import sha256_file
from src.features.straddle_observations import a1_key_digest


def _load_backfill_helpers():
    """Load D3 script helpers without requiring ``scripts`` to be a package."""
    path = _REPO_ROOT / "scripts" / "backfill_features.py"
    module_name = "backfill_features_for_d4_audit"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load backfill helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_bf = _load_backfill_helpers()


def load_feature_backfill_config(config_path):
    return _bf.load_feature_backfill_config(config_path)


def require_clean_repo_sha() -> str:
    return _bf.require_clean_repo_sha()


FeatureBackfillConfig = _bf.FeatureBackfillConfig


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

_D2_COLUMNS = ["ticker", "entry_date", "expiry_date", "return_pct", "vol_gap"]
_SENTINEL_WINDOWS = ((6, 2), (42, 8), (60, 24))
_BASELINE_WINDOW = (42, 8)
_MAX_VIOLATION_EXAMPLES = 10
_PIT_MIN_LAG = 2
_EXPECTED_WINDOW_COUNT = 281


def _require_mapping(obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj


def _iso_date(value: Any) -> str:
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def _as_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError(f"rate denominator must be > 0, got {denominator}")
    return float(numerator) / float(denominator)


def available_slots_for_positions(
    positions: np.ndarray, *, max_lag: int, min_lag: int
) -> np.ndarray:
    """Vectorized ``min(window_size, max(0, i - min_lag + 1))``."""
    window_size = int(max_lag) - int(min_lag) + 1
    return np.minimum(
        window_size, np.maximum(0, positions.astype(np.int64) - int(min_lag) + 1)
    )


def compute_baseline_ready_interval(
    common_dates: Sequence[pd.Timestamp],
) -> dict[str, Any]:
    """Return frozen (42, 8) ready interval: 43rd → final common ordered date."""
    dates = [pd.Timestamp(d).normalize() for d in common_dates]
    if len(dates) < 43:
        raise ValueError(
            f"need at least 43 common ordered D2 dates for baseline ready interval, "
            f"got {len(dates)}"
        )
    ready_start = dates[42]  # 43rd date (1-based)
    ready_end = dates[-1]
    return {
        "window": [42, 8],
        "window_size": 35,
        "ready_start": _iso_date(ready_start),
        "ready_end": _iso_date(ready_end),
        "common_date_count": len(dates),
    }


def compact_pit_proof(
    observations: pd.DataFrame,
    *,
    economic_col: str,
    label: str,
) -> dict[str, Any]:
    """Compact exhaustive PIT: expiry[j] < entry_date[j+2] for finite economics."""
    if economic_col not in observations.columns:
        raise ValueError(f"{label} missing economic column {economic_col!r}")

    eligible = 0
    checked = 0
    violations = 0
    min_gap_days: int | None = None
    examples: list[dict[str, Any]] = []

    for ticker, group in observations.groupby("ticker", sort=True):
        g = group.reset_index(drop=True)
        n = len(g)
        if n <= _PIT_MIN_LAG:
            continue
        entry = pd.to_datetime(g["entry_date"]).dt.normalize()
        expiry = pd.to_datetime(g["expiry_date"], errors="coerce")
        econ = pd.to_numeric(g[economic_col], errors="coerce")
        for j in range(n - _PIT_MIN_LAG):
            if not np.isfinite(econ.iloc[j]):
                continue
            eligible += 1
            checked += 1
            exp = expiry.iloc[j]
            target = entry.iloc[j + _PIT_MIN_LAG]
            if pd.isna(exp):
                violations += 1
                if len(examples) < _MAX_VIOLATION_EXAMPLES:
                    examples.append(
                        {
                            "ticker": str(ticker),
                            "source_entry_date": _iso_date(entry.iloc[j]),
                            "source_position": int(j),
                            "reason": "missing_or_invalid_expiry",
                        }
                    )
                continue
            exp_n = pd.Timestamp(exp).normalize()
            if not (exp_n < target):
                violations += 1
                if len(examples) < _MAX_VIOLATION_EXAMPLES:
                    examples.append(
                        {
                            "ticker": str(ticker),
                            "source_entry_date": _iso_date(entry.iloc[j]),
                            "source_position": int(j),
                            "expiry_date": _iso_date(exp_n),
                            "feature_date_j_plus_2": _iso_date(target),
                            "reason": "expiry_not_strictly_before_j_plus_2",
                        }
                    )
                continue
            gap = int((target - exp_n).days)
            if min_gap_days is None or gap < min_gap_days:
                min_gap_days = gap

    if checked != eligible:
        raise ValueError(
            f"{label} PIT checked_observations {checked} != eligible {eligible}"
        )
    return {
        "signal": label,
        "eligible_observations": int(eligible),
        "checked_observations": int(checked),
        "violations": int(violations),
        "minimum_safety_gap_days": min_gap_days,
        "violation_examples": examples,
    }


def _count_summary(
    counts: pd.Series, *, window_size: int, n_rows: int
) -> dict[str, Any]:
    finite = pd.to_numeric(counts, errors="coerce")
    finite_mask = np.isfinite(finite.to_numpy(dtype=float))
    finite_vals = finite.to_numpy(dtype=float)[finite_mask]
    full_n = int(np.sum(finite_vals == float(window_size))) if finite_vals.size else 0
    if finite_vals.size == 0:
        return {
            "finite_count_n": 0,
            "count_min": None,
            "count_median": None,
            "count_max": None,
            "full_window_count_n": 0,
            "full_window_count_rate": _as_rate(0, n_rows),
        }
    return {
        "finite_count_n": int(finite_vals.size),
        "count_min": float(np.min(finite_vals)),
        "count_median": float(np.median(finite_vals)),
        "count_max": float(np.max(finite_vals)),
        "full_window_count_n": full_n,
        "full_window_count_rate": _as_rate(full_n, n_rows),
    }


def _validate_counts_and_features(
    feature: pd.Series,
    counts: pd.Series,
    available_slots: np.ndarray,
    *,
    label: str,
) -> None:
    feat = pd.to_numeric(feature, errors="coerce").to_numpy(dtype=float)
    cnt = pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float)
    if np.isinf(feat).any():
        raise ValueError(f"{label} contains infinite feature values")
    if np.isinf(cnt).any():
        raise ValueError(f"{label} contains infinite count values")

    for i in range(len(feat)):
        slots = int(available_slots[i])
        c = cnt[i]
        f = feat[i]
        if slots == 0:
            if np.isfinite(c):
                raise ValueError(
                    f"{label} count must be null when available_slots==0 "
                    f"(row {i}, count={c})"
                )
            continue
        if not np.isfinite(c):
            raise ValueError(
                f"{label} count must be finite when available_slots>0 "
                f"(row {i}, slots={slots})"
            )
        if abs(c - round(c)) > 1e-9:
            raise ValueError(f"{label} count is non-integral at row {i}: {c}")
        c_int = int(round(c))
        if c_int < 0:
            raise ValueError(f"{label} count is negative at row {i}: {c_int}")
        if c_int > slots:
            raise ValueError(
                f"{label} count {c_int} exceeds available_slots {slots} at row {i}"
            )
        if c_int > 0 and not np.isfinite(f):
            raise ValueError(
                f"{label} count>0 requires finite feature at row {i} (min_periods=1)"
            )
        if c_int == 0 and np.isfinite(f):
            raise ValueError(
                f"{label} count==0 requires null feature at row {i} (min_periods=1)"
            )


def attribute_missingness(
    counts: pd.Series,
    available_slots: np.ndarray,
    *,
    window_size: int,
    n_rows: int,
    label: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (structural_counts, economic_counts) with reconciliations."""
    slots = available_slots.astype(np.int64)
    if len(slots) != n_rows:
        raise ValueError(f"{label} available_slots length mismatch")

    structural = {
        "no_slots": int(np.sum(slots == 0)),
        "truncated_window": int(np.sum((slots > 0) & (slots < window_size))),
        "full_window": int(np.sum(slots == window_size)),
    }
    if (
        structural["no_slots"]
        + structural["truncated_window"]
        + structural["full_window"]
        != n_rows
    ):
        raise ValueError(f"{label} structural missingness does not reconcile to n_rows")

    cnt = pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float)
    active = slots > 0
    c_active = cnt[active]
    s_active = slots[active]
    # counts validated earlier; use rounded ints
    c_int = np.rint(c_active).astype(np.int64)
    economic = {
        "zero_finite": int(np.sum(c_int == 0)),
        "partial_finite": int(np.sum((c_int > 0) & (c_int < s_active))),
        "all_available_finite": int(np.sum(c_int == s_active)),
    }
    if (
        economic["zero_finite"]
        + economic["partial_finite"]
        + economic["all_available_finite"]
        != n_rows - structural["no_slots"]
    ):
        raise ValueError(
            f"{label} economic missingness does not reconcile to n_rows - no_slots"
        )
    return structural, economic


def analyze_one_window_frame(
    frame: pd.DataFrame,
    *,
    max_lag: int,
    min_lag: int,
    date_to_pos: dict[pd.Timestamp, int],
    expected_row_count: int,
) -> dict[str, Any]:
    """Compute coverage, counts, and missingness for one feature frame."""
    window_size = int(max_lag) - int(min_lag) + 1
    mom_mean = f"mom_{max_lag}_{min_lag}_mean"
    mom_count = f"mom_{max_lag}_{min_lag}_count"
    cvg_col = f"cvg_{max_lag}_{min_lag}"
    cvg_count = f"cvg_count_{max_lag}_{min_lag}"
    required = ["ticker", "date", mom_mean, mom_count, cvg_col, cvg_count]
    if list(frame.columns) != required:
        raise ValueError(
            f"features_{max_lag}_{min_lag} columns {list(frame.columns)!r} "
            f"!= required {required!r}"
        )
    n_rows = len(frame)
    if n_rows != expected_row_count:
        raise ValueError(
            f"features_{max_lag}_{min_lag} row count {n_rows} != expected "
            f"{expected_row_count}"
        )

    dates = pd.to_datetime(frame["date"]).dt.normalize()
    try:
        positions = dates.map(lambda d: date_to_pos[pd.Timestamp(d).normalize()])
    except KeyError as exc:
        raise ValueError(
            f"features_{max_lag}_{min_lag} contains date not in D2 common index: {exc}"
        ) from exc
    pos_arr = positions.to_numpy(dtype=np.int64)
    slots = available_slots_for_positions(pos_arr, max_lag=max_lag, min_lag=min_lag)

    _validate_counts_and_features(
        frame[mom_mean], frame[mom_count], slots, label=f"Momentum {max_lag}_{min_lag}"
    )
    _validate_counts_and_features(
        frame[cvg_col], frame[cvg_count], slots, label=f"CVG {max_lag}_{min_lag}"
    )

    mom_finite = np.isfinite(pd.to_numeric(frame[mom_mean], errors="coerce").to_numpy(dtype=float))
    cvg_finite = np.isfinite(pd.to_numeric(frame[cvg_col], errors="coerce").to_numpy(dtype=float))
    both = mom_finite & cvg_finite
    mom_n = int(mom_finite.sum())
    cvg_n = int(cvg_finite.sum())
    both_n = int(both.sum())

    structural, mom_econ = attribute_missingness(
        frame[mom_count],
        slots,
        window_size=window_size,
        n_rows=n_rows,
        label=f"Momentum {max_lag}_{min_lag}",
    )
    _, cvg_econ = attribute_missingness(
        frame[cvg_count],
        slots,
        window_size=window_size,
        n_rows=n_rows,
        label=f"CVG {max_lag}_{min_lag}",
    )

    result: dict[str, Any] = {
        "max_lag": int(max_lag),
        "min_lag": int(min_lag),
        "window_size": window_size,
        "filename": f"features_{max_lag}_{min_lag}.parquet",
        "n_rows": n_rows,
        "mom_nonnull_n": mom_n,
        "mom_nonnull_rate": _as_rate(mom_n, n_rows),
        "cvg_nonnull_n": cvg_n,
        "cvg_nonnull_rate": _as_rate(cvg_n, n_rows),
        "both_nonnull_n": both_n,
        "both_nonnull_rate": _as_rate(both_n, n_rows),
        "mom_count_summary": _count_summary(
            frame[mom_count], window_size=window_size, n_rows=n_rows
        ),
        "cvg_count_summary": _count_summary(
            frame[cvg_count], window_size=window_size, n_rows=n_rows
        ),
        "structural_missingness": structural,
        "momentum_economic_missingness": mom_econ,
        "cvg_economic_missingness": cvg_econ,
    }
    return result


def sentinel_date_coverage(
    frame: pd.DataFrame, *, max_lag: int, min_lag: int
) -> list[dict[str, Any]]:
    """Per-date coverage series for one sentinel window."""
    mom_mean = f"mom_{max_lag}_{min_lag}_mean"
    cvg_col = f"cvg_{max_lag}_{min_lag}"
    out: list[dict[str, Any]] = []
    tmp = frame.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
    for date, group in tmp.groupby("date", sort=True):
        n = len(group)
        mom_n = int(
            np.isfinite(
                pd.to_numeric(group[mom_mean], errors="coerce").to_numpy(dtype=float)
            ).sum()
        )
        cvg_n = int(
            np.isfinite(
                pd.to_numeric(group[cvg_col], errors="coerce").to_numpy(dtype=float)
            ).sum()
        )
        both_n = int(
            (
                np.isfinite(
                    pd.to_numeric(group[mom_mean], errors="coerce").to_numpy(dtype=float)
                )
                & np.isfinite(
                    pd.to_numeric(group[cvg_col], errors="coerce").to_numpy(dtype=float)
                )
            ).sum()
        )
        out.append(
            {
                "date": _iso_date(date),
                "n_rows": n,
                "mom_nonnull_n": mom_n,
                "mom_nonnull_rate": _as_rate(mom_n, n),
                "cvg_nonnull_n": cvg_n,
                "cvg_nonnull_rate": _as_rate(cvg_n, n),
                "both_nonnull_n": both_n,
                "both_nonnull_rate": _as_rate(both_n, n),
            }
        )
    return out


def baseline_interval_coverage(
    frame: pd.DataFrame,
    *,
    ready_start: str,
    ready_end: str,
    max_lag: int = 42,
    min_lag: int = 8,
) -> dict[str, Any]:
    """Coverage inside the fixed baseline ready interval (inclusive)."""
    mom_mean = f"mom_{max_lag}_{min_lag}_mean"
    cvg_col = f"cvg_{max_lag}_{min_lag}"
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    start = pd.Timestamp(ready_start)
    end = pd.Timestamp(ready_end)
    mask = (dates >= start) & (dates <= end)
    sub = frame.loc[mask]
    n = len(sub)
    if n == 0:
        raise ValueError("baseline ready interval contains no feature rows")
    mom_n = int(
        np.isfinite(pd.to_numeric(sub[mom_mean], errors="coerce").to_numpy(dtype=float)).sum()
    )
    cvg_n = int(
        np.isfinite(pd.to_numeric(sub[cvg_col], errors="coerce").to_numpy(dtype=float)).sum()
    )
    both_n = int(
        (
            np.isfinite(pd.to_numeric(sub[mom_mean], errors="coerce").to_numpy(dtype=float))
            & np.isfinite(pd.to_numeric(sub[cvg_col], errors="coerce").to_numpy(dtype=float))
        ).sum()
    )
    return {
        "n_rows": n,
        "mom_nonnull_n": mom_n,
        "mom_nonnull_rate": _as_rate(mom_n, n),
        "cvg_nonnull_n": cvg_n,
        "cvg_nonnull_rate": _as_rate(cvg_n, n),
        "both_nonnull_n": both_n,
        "both_nonnull_rate": _as_rate(both_n, n),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _require_mapping(json.load(handle), label)


def validate_startup_identities(
    *,
    features_dir: Path,
    d3_receipt_path: Path,
    observations_path: Path,
    d2_lineage_path: Path,
    config_path: Path,
    output_json: Path,
    expected_snapshot_id: str,
    expected_build_id: str,
    expected_d3_repo_sha: str,
) -> tuple[FeatureBackfillConfig, dict[str, Any], str, dict[str, str]]:
    """Lightweight gate before reading feature/D2 panels. Returns config, receipt, d4 sha, digests."""
    features_dir = features_dir.resolve()
    d3_receipt_path = d3_receipt_path.resolve()
    observations_path = observations_path.resolve()
    d2_lineage_path = d2_lineage_path.resolve()
    config_path = config_path.resolve()
    output_json = output_json.resolve()

    if output_json.exists():
        raise ValueError(f"refusing to overwrite existing output JSON: {output_json}")
    if not output_json.parent.is_dir():
        raise ValueError(
            f"output JSON parent directory does not exist: {output_json.parent}"
        )
    if _path_is_inside(output_json, features_dir):
        raise ValueError(
            f"refusing output JSON inside features directory: {output_json}"
        )

    d4_audit_repo_sha = require_clean_repo_sha()

    config = load_feature_backfill_config(config_path)
    if len(config.windows) != _EXPECTED_WINDOW_COUNT:
        raise ValueError(
            f"config must expand to {_EXPECTED_WINDOW_COUNT} windows, "
            f"got {len(config.windows)}"
        )
    min_lags = [min_lag for _, min_lag in config.windows]
    if min(min_lags) != _PIT_MIN_LAG:
        raise ValueError(
            f"global minimum configured min_lag must be {_PIT_MIN_LAG}, "
            f"got {min(min_lags)}"
        )

    receipt = _load_json(d3_receipt_path, "D3 receipt")
    if receipt.get("artifact") != "features_backfill_v1":
        raise ValueError(
            f"D3 receipt.artifact must be 'features_backfill_v1', "
            f"got {receipt.get('artifact')!r}"
        )
    if receipt.get("status") != "complete":
        raise ValueError(
            f"D3 receipt.status must be 'complete', got {receipt.get('status')!r}"
        )
    if receipt.get("snapshot_id") != expected_snapshot_id:
        raise ValueError(
            f"snapshot_id mismatch: receipt {receipt.get('snapshot_id')!r}, "
            f"expected {expected_snapshot_id!r}"
        )
    if receipt.get("build_id") != expected_build_id:
        raise ValueError(
            f"build_id mismatch: receipt {receipt.get('build_id')!r}, "
            f"expected {expected_build_id!r}"
        )
    if receipt.get("repo_sha") != expected_d3_repo_sha:
        raise ValueError(
            f"D3 repo_sha mismatch: receipt {receipt.get('repo_sha')!r}, "
            f"expected {expected_d3_repo_sha!r}"
        )
    if receipt.get("feature_config_sha256") != config.config_sha256:
        raise ValueError(
            "feature_config_sha256 mismatch between receipt and loaded config bytes"
        )

    receipt_windows = receipt.get("windows")
    if (
        not isinstance(receipt_windows, list)
        or len(receipt_windows) != _EXPECTED_WINDOW_COUNT
    ):
        raise ValueError(
            f"D3 receipt.windows must be a list of length {_EXPECTED_WINDOW_COUNT}"
        )
    expected_windows = [[int(a), int(b)] for a, b in config.windows]
    if receipt_windows != expected_windows:
        raise ValueError("D3 receipt.windows do not match config window order")

    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != _EXPECTED_WINDOW_COUNT:
        raise ValueError(
            f"D3 receipt.files must be a list of length {_EXPECTED_WINDOW_COUNT}"
        )
    for idx, (window, record) in enumerate(zip(config.windows, files, strict=True)):
        if not isinstance(record, dict):
            raise ValueError(f"D3 receipt.files[{idx}] must be an object")
        max_lag, min_lag = window
        expected_name = f"features_{max_lag}_{min_lag}.parquet"
        if record.get("filename") != expected_name:
            raise ValueError(
                f"D3 receipt.files[{idx}].filename {record.get('filename')!r} "
                f"!= {expected_name!r}"
            )
        if int(record.get("max_lag")) != max_lag or int(record.get("min_lag")) != min_lag:
            raise ValueError(f"D3 receipt.files[{idx}] lag identity mismatch")

    receipt_features_dir = Path(str(receipt.get("features_dir"))).resolve()
    if receipt_features_dir != features_dir:
        raise ValueError(
            f"features-dir mismatch: CLI {features_dir}, receipt {receipt_features_dir}"
        )
    receipt_obs = Path(str(receipt.get("observations_path"))).resolve()
    receipt_d2 = Path(str(receipt.get("d2_lineage_path"))).resolve()
    if receipt_obs != observations_path:
        raise ValueError(
            f"observations path mismatch: CLI {observations_path}, receipt {receipt_obs}"
        )
    if receipt_d2 != d2_lineage_path:
        raise ValueError(
            f"d2-lineage path mismatch: CLI {d2_lineage_path}, receipt {receipt_d2}"
        )

    if not features_dir.is_dir():
        raise ValueError(f"features directory does not exist: {features_dir}")
    staging = features_dir.parent / "features.building"
    if staging.exists():
        raise ValueError(f"refusing audit while staging directory exists: {staging}")

    d2_lineage = _load_json(d2_lineage_path, "D2 lineage")
    if d2_lineage.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("D2 lineage snapshot_id mismatch")
    if d2_lineage.get("build_id") != expected_build_id:
        raise ValueError("D2 lineage build_id mismatch")
    d2_output = _require_mapping(d2_lineage.get("output"), "D2 lineage.output")
    for key in ("file_sha256", "row_count", "key_count", "output_key_digest"):
        if key not in d2_output:
            raise ValueError(f"D2 lineage.output missing {key!r}")

    if d2_output["file_sha256"] != receipt.get("observations_file_sha256"):
        raise ValueError("D2 file_sha256 does not match D3 receipt")
    if int(d2_output["row_count"]) != int(receipt.get("observations_row_count")):
        raise ValueError("D2 row_count does not match D3 receipt")
    if int(d2_output["key_count"]) != int(receipt.get("observations_key_count")):
        raise ValueError("D2 key_count does not match D3 receipt")
    if d2_output["output_key_digest"] != receipt.get("observations_output_key_digest"):
        raise ValueError("D2 output_key_digest does not match D3 receipt")

    obs_digest = sha256_file(observations_path)
    if obs_digest != d2_output["file_sha256"]:
        raise ValueError(
            f"observations file digest mismatch: computed {obs_digest}, "
            f"lineage {d2_output['file_sha256']}"
        )

    digests = {
        "feature_config_sha256": config.config_sha256,
        "d3_receipt_sha256": sha256_file(d3_receipt_path),
        "observations_file_sha256": obs_digest,
    }
    return config, receipt, d4_audit_repo_sha, digests


def load_d2_panel(observations_path: Path | str) -> pd.DataFrame:
    """Load the five audit columns, normalize, and sort once."""
    path = Path(observations_path).resolve()
    frame = pd.read_parquet(path, columns=_D2_COLUMNS)
    missing = [c for c in _D2_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"observations missing columns: {missing}")
    frame = frame.loc[:, _D2_COLUMNS].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"]).dt.normalize()
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"], errors="coerce")
    frame = frame.sort_values(["ticker", "entry_date"], kind="mergesort").reset_index(
        drop=True
    )
    return frame


def write_audit_json(path: Path | str, payload: dict[str, Any]) -> Path:
    """Atomically write deterministic JSON; refuse overwrite."""
    target = Path(path).resolve()
    if target.exists():
        raise ValueError(f"refusing to overwrite existing output JSON: {target}")
    temp_path = target.with_name(target.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    return target


def run_feature_quality_audit(
    *,
    features_dir: Path | str,
    d3_receipt_path: Path | str,
    observations_path: Path | str,
    d2_lineage_path: Path | str,
    config_path: Path | str,
    output_json: Path | str,
    expected_snapshot_id: str,
    expected_build_id: str,
    expected_d3_repo_sha: str,
) -> Path:
    """Execute the full read-only audit and write the JSON result."""
    features_dir_p = Path(features_dir).resolve()
    output_json_p = Path(output_json).resolve()

    config, receipt, d4_sha, digests = validate_startup_identities(
        features_dir=features_dir_p,
        d3_receipt_path=Path(d3_receipt_path),
        observations_path=Path(observations_path),
        d2_lineage_path=Path(d2_lineage_path),
        config_path=Path(config_path),
        output_json=output_json_p,
        expected_snapshot_id=expected_snapshot_id,
        expected_build_id=expected_build_id,
        expected_d3_repo_sha=expected_d3_repo_sha,
    )

    observations = load_d2_panel(observations_path)
    expected_rows = int(receipt["observations_row_count"])
    if len(observations) != expected_rows:
        raise ValueError(
            f"loaded observations rows {len(observations)} != receipt "
            f"observations_row_count {expected_rows}"
        )
    key_digest = a1_key_digest(observations)
    if key_digest != receipt["observations_output_key_digest"]:
        raise ValueError("loaded observations key digest mismatch vs D3 receipt")

    common_dates = sorted(pd.to_datetime(observations["entry_date"]).dt.normalize().unique())
    date_to_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(common_dates)}
    ready = compute_baseline_ready_interval(common_dates)

    mom_pit = compact_pit_proof(
        observations, economic_col="return_pct", label="momentum"
    )
    cvg_pit = compact_pit_proof(observations, economic_col="vol_gap", label="cvg")
    if mom_pit["violations"] != 0:
        raise ValueError(
            f"Momentum PIT violations={mom_pit['violations']}; "
            f"examples={mom_pit['violation_examples'][:3]!r}"
        )
    if cvg_pit["violations"] != 0:
        raise ValueError(
            f"CVG PIT violations={cvg_pit['violations']}; "
            f"examples={cvg_pit['violation_examples'][:3]!r}"
        )

    per_window: list[dict[str, Any]] = []
    sentinel_series: dict[str, list[dict[str, Any]]] = {}
    baseline_coverage: dict[str, Any] | None = None

    for max_lag, min_lag in config.windows:
        path = features_dir_p / f"features_{max_lag}_{min_lag}.parquet"
        if not path.is_file():
            raise ValueError(f"missing feature file: {path}")
        frame = pd.read_parquet(path)
        try:
            window_result = analyze_one_window_frame(
                frame,
                max_lag=max_lag,
                min_lag=min_lag,
                date_to_pos=date_to_pos,
                expected_row_count=expected_rows,
            )
            per_window.append(window_result)
            if (max_lag, min_lag) in _SENTINEL_WINDOWS:
                key = f"{max_lag}_{min_lag}"
                sentinel_series[key] = sentinel_date_coverage(
                    frame, max_lag=max_lag, min_lag=min_lag
                )
            if (max_lag, min_lag) == _BASELINE_WINDOW:
                baseline_coverage = baseline_interval_coverage(
                    frame,
                    ready_start=ready["ready_start"],
                    ready_end=ready["ready_end"],
                )
        finally:
            del frame

    if baseline_coverage is None:
        raise ValueError("baseline window (42, 8) was not present in config windows")

    ready_payload = {
        **ready,
        "coverage_inside_interval": baseline_coverage,
    }

    payload = {
        "schema_version": "1",
        "artifact": "features_quality_audit_v1",
        "status": "complete",
        "snapshot_id": expected_snapshot_id,
        "build_id": expected_build_id,
        "d3_producer_repo_sha": expected_d3_repo_sha,
        "d4_audit_repo_sha": d4_sha,
        "feature_config_sha256": digests["feature_config_sha256"],
        "d3_receipt_sha256": digests["d3_receipt_sha256"],
        "observations_file_sha256": digests["observations_file_sha256"],
        "paths": {
            "features_dir": str(features_dir_p),
            "d3_receipt": str(Path(d3_receipt_path).resolve()),
            "observations": str(Path(observations_path).resolve()),
            "d2_lineage": str(Path(d2_lineage_path).resolve()),
            "config": str(Path(config_path).resolve()),
            "output_json": str(output_json_p),
        },
        "window_count": len(per_window),
        "windows": per_window,
        "sentinel_date_coverage": {
            "windows": [[a, b] for a, b in _SENTINEL_WINDOWS],
            "series": sentinel_series,
        },
        "baseline_ready_interval": ready_payload,
        "pit_momentum": mom_pit,
        "pit_cvg": cvg_pit,
    }

    return write_audit_json(output_json_p, payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 005 D4 — read-only feature quality audit."
    )
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--d3-receipt", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--d2-lineage", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--expected-snapshot-id", required=True)
    parser.add_argument("--expected-build-id", required=True)
    parser.add_argument("--expected-d3-repo-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_feature_quality_audit(
            features_dir=args.features_dir,
            d3_receipt_path=args.d3_receipt,
            observations_path=args.observations,
            d2_lineage_path=args.d2_lineage,
            config_path=args.config,
            output_json=args.output_json,
            expected_snapshot_id=args.expected_snapshot_id,
            expected_build_id=args.expected_build_id,
            expected_d3_repo_sha=args.expected_d3_repo_sha,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
