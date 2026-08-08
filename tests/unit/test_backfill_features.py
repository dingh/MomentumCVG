"""Sprint 005 D3 Block 1 — feature backfill config, D2 input, and Git checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.snapshot_foundation import sha256_file
from src.features.straddle_observations import a1_key_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_features.py"
SPEC_PATH = REPO_ROOT / "configs" / "feature_backfill_v1.json"

REQUIRED_COLUMNS = [
    "ticker",
    "entry_date",
    "return_pct",
    "entry_iv",
    "realized_volatility",
    "vol_gap",
    "expiry_date",
]


def _load_module():
    module_name = "backfill_features_block1"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bf():
    return _load_module()


def _oracle_windows() -> list[tuple[int, int]]:
    return [
        (max_lag, min_lag)
        for max_lag in range(6, 61, 2)
        for min_lag in range(2, 25, 2)
        if max_lag > min_lag
    ]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _synthetic_observations(n_tickers: int = 2, n_dates: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    rows = []
    for ticker in [f"T{i}" for i in range(n_tickers)]:
        for entry in dates:
            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": entry,
                    "return_pct": 1.0,
                    "entry_iv": 0.2,
                    "realized_volatility": 0.25,
                    "vol_gap": 0.05,
                    "expiry_date": entry + pd.Timedelta(days=7),
                }
            )
    return pd.DataFrame(rows)


def _write_d2_pair(
    tmp_path: Path,
    df: pd.DataFrame,
    *,
    snapshot_id: str = "snaptest01",
    build_id: str = "buildtest01",
    mutate_lineage: dict | None = None,
    obs_name: str = "straddle_observations_weekly.parquet",
    lineage_name: str = "straddle_observations_weekly.lineage.json",
) -> tuple[Path, Path]:
    root = tmp_path / "derived" / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    obs_path = root / obs_name
    lineage_path = root / lineage_name
    df.to_parquet(obs_path, index=False)
    lineage = {
        "schema_version": "1",
        "artifact": "straddle_observations_weekly",
        "snapshot_id": snapshot_id,
        "build_id": build_id,
        "output": {
            "row_count": len(df),
            "key_count": len(df.drop_duplicates(["ticker", "entry_date"])),
            "output_key_digest": a1_key_digest(df),
            "file_sha256": sha256_file(obs_path),
            "content_digest": "unused-in-block1",
            "column_order": list(df.columns),
        },
    }
    if mutate_lineage:
        # shallow/deep patches for negative tests
        for key, value in mutate_lineage.items():
            if key == "output" and isinstance(value, dict):
                lineage["output"].update(value)
            else:
                lineage[key] = value
        # if file digest intentionally left stale, do not recompute
    _write_json(lineage_path, lineage)
    return obs_path, lineage_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_real_config_expands_to_281_ordered_windows(bf):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    oracle = _oracle_windows()

    assert cfg.spec_version == "feature_backfill_v1"
    assert len(cfg.windows) == 281
    assert len(set(cfg.windows)) == 281
    assert cfg.windows == oracle
    assert cfg.windows[0] == (6, 2)
    assert cfg.windows[-1] == (60, 24)
    assert all(max_lag > min_lag for max_lag, min_lag in cfg.windows)
    assert cfg.baseline_window == (42, 8)
    assert (42, 8) in cfg.windows
    assert cfg.momentum_min_periods == 1
    assert cfg.cvg_min_periods == 1
    assert cfg.required_columns == REQUIRED_COLUMNS
    assert cfg.output_columns_per_window == [
        "ticker",
        "date",
        "mom_{max}_{min}_mean",
        "mom_{max}_{min}_count",
        "cvg_{max}_{min}",
        "cvg_count_{max}_{min}",
    ]
    assert cfg.config_sha256 == sha256_file(SPEC_PATH)
    assert cfg.config_path == SPEC_PATH.resolve()


def test_config_missing_field_fails_without_defaults(bf, tmp_path):
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    del raw["momentum"]["min_periods"]
    bad = _write_json(tmp_path / "bad_config.json", raw)
    with pytest.raises(ValueError, match="min_periods"):
        bf.load_feature_backfill_config(bad)


def test_config_malformed_expected_count_fails(bf, tmp_path):
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw["windows"]["expected_count"] = 280
    bad = _write_json(tmp_path / "bad_count.json", raw)
    with pytest.raises(ValueError, match="expected_count"):
        bf.load_feature_backfill_config(bad)


def test_shifted_281_grid_fails_frozen_endpoints(bf, tmp_path):
    """A 281-window grid like (4,0)…(58,22) must not pass as the frozen v1 grid."""
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw["windows"]["min_lag_start"] = 0
    raw["windows"]["min_lag_end"] = 22
    raw["windows"]["max_lag_start"] = 4
    raw["windows"]["max_lag_end"] = 58
    bad = _write_json(tmp_path / "shifted_grid.json", raw)
    with pytest.raises(ValueError, match=r"must start with \(6, 2\)"):
        bf.load_feature_backfill_config(bad)


@pytest.mark.parametrize(
    "section, value",
    [
        ("momentum", 3),
        ("momentum", 0),
        ("cvg", 5),
        ("cvg", 2),
    ],
)
def test_min_periods_must_be_exactly_one(bf, tmp_path, section, value):
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw[section]["min_periods"] = value
    bad = _write_json(tmp_path / f"bad_{section}_min_{value}.json", raw)
    with pytest.raises(ValueError, match=rf"{section}\.min_periods must be 1"):
        bf.load_feature_backfill_config(bad)


@pytest.mark.parametrize(
    "field, value, match",
    [
        (
            "required_columns",
            ["ticker"],
            "required_columns must exactly match",
        ),
        (
            "output_columns_per_window",
            ["ticker"],
            "output_columns_per_window must exactly match",
        ),
        (
            "required_columns",
            [
                "ticker",
                "entry_date",
                "return_pct",
                "entry_iv",
                "realized_volatility",
                "vol_gap",
                "extra_col",
            ],
            "required_columns must exactly match",
        ),
        (
            "output_columns_per_window",
            [
                "date",
                "ticker",
                "mom_{max}_{min}_mean",
                "mom_{max}_{min}_count",
                "cvg_{max}_{min}",
                "cvg_count_{max}_{min}",
            ],
            "output_columns_per_window must exactly match",
        ),
    ],
)
def test_exact_v1_schemas_enforced(bf, tmp_path, field, value, match):
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if field == "required_columns":
        raw["input_schema"]["required_columns"] = value
    else:
        raw["output_columns_per_window"] = value
    bad = _write_json(tmp_path / f"bad_{field}.json", raw)
    with pytest.raises(ValueError, match=match):
        bf.load_feature_backfill_config(bad)


def test_config_path_required_no_default(bf):
    with pytest.raises(TypeError):
        bf.load_feature_backfill_config()


def test_script_does_not_import_build_features(bf):
    assert "build_features" not in getattr(bf, "__file__", "")
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "build_features" not in source
    assert "generate_momentum_windows" not in source


# ---------------------------------------------------------------------------
# D2 validation
# ---------------------------------------------------------------------------


def test_valid_synthetic_d2_passes_and_uses_a1_key_digest(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    validated = bf.validate_d2_input(
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        expected_snapshot_id="snaptest01",
        expected_build_id="buildtest01",
        required_columns=REQUIRED_COLUMNS,
    )
    assert len(validated.observations) == len(df)
    assert list(validated.observations.columns) == REQUIRED_COLUMNS
    assert validated.output_key_digest == a1_key_digest(df)
    assert validated.file_sha256 == sha256_file(obs_path)
    assert validated.row_count == len(df)
    assert validated.key_count == len(df)
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "features.building").exists()
    assert not list(tmp_path.rglob("features_backfill_v1.lineage.json"))


def test_validated_d2_frame_keeps_only_required_columns(bf, tmp_path):
    df = _synthetic_observations()
    df["extra_noise"] = 123.0
    df["another_extra"] = "x"
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    validated = bf.validate_d2_input(
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        expected_snapshot_id="snaptest01",
        expected_build_id="buildtest01",
        required_columns=REQUIRED_COLUMNS,
    )
    assert list(validated.observations.columns) == REQUIRED_COLUMNS
    assert "extra_noise" not in validated.observations.columns
    assert "another_extra" not in validated.observations.columns
    assert len(validated.observations) == len(df)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"expected_snapshot_id": "wrong"}, "snapshot_id mismatch"),
        ({"expected_build_id": "wrong"}, "build_id mismatch"),
    ],
)
def test_wrong_identity_fails(bf, tmp_path, kwargs, match):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    args = dict(
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        expected_snapshot_id="snaptest01",
        expected_build_id="buildtest01",
        required_columns=REQUIRED_COLUMNS,
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bf.validate_d2_input(**args)


def test_incorrect_file_digest_fails(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(
        tmp_path,
        df,
        mutate_lineage={"output": {"file_sha256": "0" * 64}},
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_incorrect_row_count_fails(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(
        tmp_path, df, mutate_lineage={"output": {"row_count": 1}}
    )
    with pytest.raises(ValueError, match="row count"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_incorrect_key_count_fails(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(
        tmp_path, df, mutate_lineage={"output": {"key_count": 1}}
    )
    with pytest.raises(ValueError, match="key count"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_incorrect_key_digest_fails(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(
        tmp_path,
        df,
        mutate_lineage={"output": {"output_key_digest": "f" * 64}},
    )
    with pytest.raises(ValueError, match="key digest mismatch"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_missing_required_column_fails(bf, tmp_path):
    df = _synthetic_observations().drop(columns=["vol_gap"])
    root = tmp_path / "derived" / "snaptest01"
    root.mkdir(parents=True)
    obs_path = root / "straddle_observations_weekly.parquet"
    df.to_parquet(obs_path, index=False)
    lineage = {
        "schema_version": "1",
        "artifact": "straddle_observations_weekly",
        "snapshot_id": "snaptest01",
        "build_id": "buildtest01",
        "output": {
            "row_count": len(df),
            "key_count": len(df),
            "output_key_digest": a1_key_digest(df),
            "file_sha256": sha256_file(obs_path),
        },
    }
    lineage_path = _write_json(
        root / "straddle_observations_weekly.lineage.json", lineage
    )
    with pytest.raises(ValueError, match="missing required columns"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_null_key_fails(bf, tmp_path):
    df = _synthetic_observations()
    df.loc[0, "ticker"] = None
    # Write without going through helper digest on nulls — build lineage from clean copy
    clean = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, clean)
    df.to_parquet(obs_path, index=False)
    # refresh digest/row to match corrupted file so we hit null-key check
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["output"]["file_sha256"] = sha256_file(obs_path)
    lineage["output"]["row_count"] = len(df)
    _write_json(lineage_path, lineage)
    with pytest.raises(ValueError, match="null ticker or entry_date"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_duplicate_key_fails(bf, tmp_path):
    df = _synthetic_observations()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    root = tmp_path / "derived" / "snaptest01"
    root.mkdir(parents=True)
    obs_path = root / "straddle_observations_weekly.parquet"
    df.to_parquet(obs_path, index=False)
    lineage = {
        "schema_version": "1",
        "artifact": "straddle_observations_weekly",
        "snapshot_id": "snaptest01",
        "build_id": "buildtest01",
        "output": {
            "row_count": len(df),
            "key_count": len(df),  # deliberately wrong unique count path hits duplicate first
            "output_key_digest": "0" * 64,
            "file_sha256": sha256_file(obs_path),
        },
    }
    lineage_path = _write_json(
        root / "straddle_observations_weekly.lineage.json", lineage
    )
    with pytest.raises(ValueError, match="duplicate"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_non_sibling_paths_fail(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    other = tmp_path / "other"
    other.mkdir()
    moved = other / lineage_path.name
    moved.write_text(lineage_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="siblings"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=moved,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_cache_path_observations_fail(bf, tmp_path, monkeypatch):
    df = _synthetic_observations()
    cache_root = tmp_path / "MomentumCVG_env" / "cache" / "derived"
    cache_root.mkdir(parents=True)
    obs_path = cache_root / "straddle_observations_weekly.parquet"
    lineage_path = cache_root / "straddle_observations_weekly.lineage.json"
    df.to_parquet(obs_path, index=False)
    lineage = {
        "schema_version": "1",
        "artifact": "straddle_observations_weekly",
        "snapshot_id": "snaptest01",
        "build_id": "buildtest01",
        "output": {
            "row_count": len(df),
            "key_count": len(df),
            "output_key_digest": a1_key_digest(df),
            "file_sha256": sha256_file(obs_path),
        },
    }
    _write_json(lineage_path, lineage)
    monkeypatch.setattr(bf, "_MUTABLE_CACHE_ROOT", tmp_path / "MomentumCVG_env" / "cache")
    with pytest.raises(ValueError, match="mutable cache"):
        bf.validate_d2_input(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
            required_columns=REQUIRED_COLUMNS,
        )


def test_validation_writes_no_feature_outputs(bf, tmp_path):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    bf.validate_d2_input(
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        expected_snapshot_id="snaptest01",
        expected_build_id="buildtest01",
        required_columns=REQUIRED_COLUMNS,
    )
    bf.load_feature_backfill_config(SPEC_PATH)
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "features.building").exists()
    feature_files = [
        p for p in tmp_path.rglob("*.parquet") if p.name.startswith("features_")
    ]
    assert feature_files == []
    assert not list(tmp_path.rglob("features_backfill_v1.lineage.json"))


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------


def test_clean_git_returns_actual_sha(bf, monkeypatch):
    expected = "a" * 40

    def fake_git(*args: str) -> str:
        return {("rev-parse", "HEAD"): expected, ("status", "--porcelain"): ""}[args]

    monkeypatch.setattr(bf, "_git_output", fake_git)
    assert bf.require_clean_repo_sha() == expected


def test_dirty_git_fails(bf, monkeypatch):
    def fake_git(*args: str) -> str:
        return {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): " M scripts/backfill_features.py",
        }[args]

    monkeypatch.setattr(bf, "_git_output", fake_git)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        bf.require_clean_repo_sha()


def test_unavailable_git_fails(bf, monkeypatch):
    def fake_git(*args: str) -> str:
        raise subprocess.CalledProcessError(1, ["git", *args])

    monkeypatch.setattr(bf, "_git_output", fake_git)
    with pytest.raises(RuntimeError, match="cannot determine repository revision"):
        bf.require_clean_repo_sha()


def test_git_helper_not_run_at_import(bf):
    # Importing the module (fixture) must not require a clean tree; this file can
    # exist while the suite itself dirties nothing related to require_clean_repo_sha.
    assert callable(bf.require_clean_repo_sha)
