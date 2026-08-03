"""
Sprint 005 D1 — feature_backfill_v1 contract tests (G1–G2).

Configuration-level only: no D2 artifact, no Windows data paths.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.features.cvg_calculator import CVGCalculator
from src.features.momentum_calculator import MomentumCalculator

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "feature_backfill_v1.json"
BUILD_FEATURES_PATH = REPO_ROOT / "scripts" / "build_features.py"


def _load_spec() -> dict:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_generate_momentum_windows():
    spec = importlib.util.spec_from_file_location(
        "build_features_for_contract", BUILD_FEATURES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.generate_momentum_windows


def _oracle_windows(
    *,
    min_lag_start: int,
    min_lag_end: int,
    max_lag_start: int,
    max_lag_end: int,
    step: int,
) -> list[tuple[int, int]]:
    return [
        (max_lag, min_lag)
        for max_lag in range(max_lag_start, max_lag_end + 1, step)
        for min_lag in range(min_lag_start, min_lag_end + 1, step)
        if max_lag > min_lag
    ]


def _windows_from_spec(spec: dict) -> list[tuple[int, int]]:
    """Expand the frozen grid with every bound supplied explicitly from the spec."""
    windows_cfg = spec["windows"]
    generate = _load_generate_momentum_windows()
    return generate(
        short_range=(windows_cfg["min_lag_start"], windows_cfg["min_lag_end"]),
        long_range=(windows_cfg["max_lag_start"], windows_cfg["max_lag_end"]),
        step=windows_cfg["step"],
    )


def _render_output_columns(templates: list[str], max_lag: int, min_lag: int) -> list[str]:
    rendered = []
    for template in templates:
        if "{max}" in template or "{min}" in template:
            rendered.append(
                template.replace("{max}", str(max_lag)).replace("{min}", str(min_lag))
            )
        else:
            rendered.append(template)
    return rendered


# ---------------------------------------------------------------------------
# G1 — Exact 281-window expansion
# ---------------------------------------------------------------------------


def test_g1_grid_expands_to_281_unique_windows_in_deterministic_order():
    spec = _load_spec()
    windows_cfg = spec["windows"]

    assert windows_cfg["min_lag_start"] == 2
    assert windows_cfg["min_lag_end"] == 24
    assert windows_cfg["max_lag_start"] == 6
    assert windows_cfg["max_lag_end"] == 60
    assert windows_cfg["step"] == 2
    assert windows_cfg["require_max_gt_min"] is True
    assert windows_cfg["order"] == "max_lag_outer_min_lag_inner"
    assert windows_cfg["expected_count"] == 281

    windows = _windows_from_spec(spec)
    oracle = _oracle_windows(
        min_lag_start=windows_cfg["min_lag_start"],
        min_lag_end=windows_cfg["min_lag_end"],
        max_lag_start=windows_cfg["max_lag_start"],
        max_lag_end=windows_cfg["max_lag_end"],
        step=windows_cfg["step"],
    )

    assert len(windows) == 281
    assert len(set(windows)) == 281
    assert windows == oracle
    assert windows[0] == (6, 2)
    assert windows[:5] == [(6, 2), (6, 4), (8, 2), (8, 4), (8, 6)]
    assert windows[-5:] == [(60, 16), (60, 18), (60, 20), (60, 22), (60, 24)]
    assert windows[-1] == (60, 24)
    assert all(max_lag > min_lag for max_lag, min_lag in windows)
    assert (42, 8) in windows


# ---------------------------------------------------------------------------
# G2 — Configuration authority
# ---------------------------------------------------------------------------


def test_g2_spec_is_sole_authority_for_windows_min_periods_baseline_and_columns():
    spec = _load_spec()
    windows = _windows_from_spec(spec)

    # Baseline 42:8 is present and explicitly designated.
    baseline = spec["baseline_window"]
    assert baseline == {"max_lag": 42, "min_lag": 8}
    assert (baseline["max_lag"], baseline["min_lag"]) in windows

    # Both minimum-period values are explicitly 1.
    assert spec["momentum"]["min_periods"] == 1
    assert spec["cvg"]["min_periods"] == 1

    mom = MomentumCalculator(
        windows=windows,
        min_periods=spec["momentum"]["min_periods"],
    )
    cvg = CVGCalculator(
        windows=windows,
        min_periods=spec["cvg"]["min_periods"],
    )
    assert mom.min_periods == 1
    assert cvg.min_periods == 1
    assert mom.min_periods == spec["momentum"]["min_periods"]
    assert cvg.min_periods == spec["cvg"]["min_periods"]

    # Required output-column templates render correctly, including for 42:8.
    templates = spec["output_columns_per_window"]
    assert "mom_{max}_{min}_mean" in templates
    assert "mom_{max}_{min}_count" in templates
    assert "cvg_{max}_{min}" in templates
    assert "cvg_count_{max}_{min}" in templates

    rendered_baseline = _render_output_columns(
        templates, baseline["max_lag"], baseline["min_lag"]
    )
    assert rendered_baseline == [
        "ticker",
        "date",
        "mom_42_8_mean",
        "mom_42_8_count",
        "cvg_42_8",
        "cvg_count_42_8",
    ]

    # Semantics-only: lineage / machine-specific fields must not be spec keys.
    forbidden_keys = {
        "snapshot_id",
        "repo_sha",
        "file_sha256",
        "a1_key_digest",
        "build_id",
        "observation_path",
        "derived_path",
        "cache_path",
    }

    def _collect_keys(obj, found: set[str]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                found.add(key)
                _collect_keys(value, found)
        elif isinstance(obj, list):
            for item in obj:
                _collect_keys(item, found)

    present_keys: set[str] = set()
    _collect_keys(spec, present_keys)
    assert forbidden_keys.isdisjoint(present_keys)


def test_g2_grid_generation_uses_explicit_spec_bounds_not_function_defaults():
    """All bounds come from the spec; helper defaults are non-authoritative."""
    spec = _load_spec()
    windows_cfg = spec["windows"]
    generate = _load_generate_momentum_windows()

    # Explicit call matching the frozen contract.
    from_spec = generate(
        short_range=(windows_cfg["min_lag_start"], windows_cfg["min_lag_end"]),
        long_range=(windows_cfg["max_lag_start"], windows_cfg["max_lag_end"]),
        step=windows_cfg["step"],
    )
    assert len(from_spec) == windows_cfg["expected_count"] == 281

    # Function defaults (long_range starts at 12) yield a different grid.
    from_defaults = generate()
    assert len(from_defaults) != 281
    assert len(from_defaults) == 272

    # Spec authority must not silently inherit the non-281 default grid.
    assert from_spec != from_defaults
    assert from_spec[0] == (6, 2)
    assert from_defaults[0] == (12, 2)
