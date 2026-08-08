"""Sprint 005 D3 — feature backfill helpers (Blocks 1–3)."""

from __future__ import annotations

import importlib.util
import json
import os
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


def test_script_cli_imports_without_pythonpath():
    """Production `python scripts/backfill_features.py` must resolve `src` itself."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--observations" in completed.stdout


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


# ---------------------------------------------------------------------------
# Block 2 — one-window compute + staging write
# ---------------------------------------------------------------------------


def _panel_for_window(n_dates: int = 12) -> pd.DataFrame:
    """Weekly panel large enough for window (6, 2) with independent miss patterns."""
    dates = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    rows = []
    for ticker, returns, gaps in (
        (
            "AAA",
            [10.0, None, 20.0, None, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            [0.10] * n_dates,
        ),
        (
            "BBB",
            [1.0] * n_dates,
            [None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        ),
    ):
        for entry, ret, gap in zip(dates, returns, gaps, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": entry,
                    "return_pct": ret,
                    "entry_iv": 0.2,
                    "realized_volatility": 0.25 if gap is None else 0.2 + gap,
                    "vol_gap": gap,
                    "expiry_date": entry + pd.Timedelta(days=7),
                }
            )
    return pd.DataFrame(rows)


def test_compute_one_window_end_to_end_with_production_calculators(bf, monkeypatch):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    canonical = bf.build_canonical_key_index(observations)
    window = (6, 2)
    mom_calls: list[dict] = []
    cvg_calls: list[dict] = []

    orig_mom = bf.MomentumCalculator.calculate_bulk
    orig_cvg = bf.CVGCalculator.calculate_bulk

    def spy_mom(self, context, start_date, end_date, tickers=None):
        mom_calls.append(
            {
                "windows": list(self.windows),
                "min_periods": self.min_periods,
                "tickers": tickers,
                "n_rows": len(context.get("straddle_history")),
                "n_tickers": context.get("straddle_history")["ticker"].nunique(),
            }
        )
        return orig_mom(self, context, start_date, end_date, tickers=tickers)

    def spy_cvg(self, context, start_date, end_date, tickers=None):
        cvg_calls.append(
            {
                "windows": list(self.windows),
                "min_periods": self.min_periods,
                "tickers": tickers,
                "n_rows": len(context.get("straddle_history")),
                "n_tickers": context.get("straddle_history")["ticker"].nunique(),
            }
        )
        return orig_cvg(self, context, start_date, end_date, tickers=tickers)

    monkeypatch.setattr(bf.MomentumCalculator, "calculate_bulk", spy_mom)
    monkeypatch.setattr(bf.CVGCalculator, "calculate_bulk", spy_cvg)

    out = bf.compute_one_window_features(observations, window, cfg, canonical)

    assert len(mom_calls) == 1 and len(cvg_calls) == 1
    assert mom_calls[0]["windows"] == [(6, 2)]
    assert cvg_calls[0]["windows"] == [(6, 2)]
    assert mom_calls[0]["min_periods"] == 1
    assert cvg_calls[0]["min_periods"] == 1
    assert mom_calls[0]["tickers"] is None
    assert cvg_calls[0]["tickers"] is None
    assert mom_calls[0]["n_rows"] == len(observations)
    assert cvg_calls[0]["n_rows"] == len(observations)
    assert mom_calls[0]["n_tickers"] == observations["ticker"].nunique()

    expected_cols = [
        "ticker",
        "date",
        "mom_6_2_mean",
        "mom_6_2_count",
        "cvg_6_2",
        "cvg_count_6_2",
    ]
    assert list(out.columns) == expected_cols
    assert "entry_date" not in out.columns
    assert len(out) == len(observations)
    assert out.duplicated(["ticker", "date"]).sum() == 0
    assert out["ticker"].isna().sum() == 0
    assert out["date"].isna().sum() == 0
    assert out.equals(out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True))

    # NA features must not drop rows (early history / sparse slots remain).
    assert out["mom_6_2_mean"].isna().any()
    assert len(out) == len(observations)

    # Independent Momentum vs CVG counts: AAA feature date index 6 uses
    # returns [10, None, 20, None, 30] (count 3) and five finite vol_gaps.
    feature_date = pd.to_datetime(observations["entry_date"]).drop_duplicates().iloc[6]
    row = out.loc[(out["ticker"] == "AAA") & (out["date"] == feature_date)].iloc[0]
    assert row["mom_6_2_count"] == 3
    assert row["cvg_count_6_2"] == 5


def test_canonical_key_index_built_once_and_reused(bf):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    canonical = bf.build_canonical_key_index(observations)
    assert isinstance(canonical, pd.MultiIndex)
    assert len(canonical) == len(observations)
    assert canonical.is_unique

    out_a = bf.compute_one_window_features(observations, (6, 2), cfg, canonical)
    out_b = bf.compute_one_window_features(observations, (8, 2), cfg, canonical)
    assert len(out_a) == len(canonical)
    assert len(out_b) == len(canonical)
    assert list(out_a.columns) == [
        "ticker",
        "date",
        "mom_6_2_mean",
        "mom_6_2_count",
        "cvg_6_2",
        "cvg_count_6_2",
    ]
    assert list(out_b.columns) == [
        "ticker",
        "date",
        "mom_8_2_mean",
        "mom_8_2_count",
        "cvg_8_2",
        "cvg_count_8_2",
    ]


def test_compute_accepts_reordered_calculator_keys(bf, monkeypatch):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    canonical = bf.build_canonical_key_index(observations)
    orig_mom = bf.MomentumCalculator.calculate_bulk
    orig_cvg = bf.CVGCalculator.calculate_bulk

    def shuffle_mom(self, context, start_date, end_date, tickers=None):
        out = orig_mom(self, context, start_date, end_date, tickers=tickers)
        return out.sample(frac=1.0, random_state=0).reset_index(drop=True)

    def shuffle_cvg(self, context, start_date, end_date, tickers=None):
        out = orig_cvg(self, context, start_date, end_date, tickers=tickers)
        return out.sample(frac=1.0, random_state=1).reset_index(drop=True)

    monkeypatch.setattr(bf.MomentumCalculator, "calculate_bulk", shuffle_mom)
    monkeypatch.setattr(bf.CVGCalculator, "calculate_bulk", shuffle_cvg)

    out = bf.compute_one_window_features(observations, (6, 2), cfg, canonical)
    assert len(out) == len(canonical)
    assert out.equals(
        out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "which, mutator, match",
    [
        (
            "momentum",
            lambda df: df.iloc[:-1].copy(),
            "Momentum output keys",
        ),
        (
            "cvg",
            lambda df: df.iloc[:-1].copy(),
            "CVG output keys",
        ),
        (
            "momentum",
            lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (
            "cvg",
            lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (
            "momentum",
            lambda df: df.assign(ticker=df["ticker"].where(df.index != 0, "ZZZ")),
            "Momentum output keys",
        ),
        (
            "cvg",
            lambda df: df.assign(ticker=df["ticker"].where(df.index != 0, "ZZZ")),
            "CVG output keys",
        ),
        (
            "momentum",
            lambda df: df.assign(ticker=df["ticker"].where(df.index != 0, None)),
            "null",
        ),
        (
            "cvg",
            lambda df: df.assign(ticker=df["ticker"].where(df.index != 0, None)),
            "null",
        ),
    ],
)
def test_compute_fails_on_malformed_calculator_keys(bf, monkeypatch, which, mutator, match):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    canonical = bf.build_canonical_key_index(observations)
    window = (6, 2)

    if which == "momentum":
        orig = bf.MomentumCalculator.calculate_bulk

        def bad(self, context, start_date, end_date, tickers=None):
            return mutator(orig(self, context, start_date, end_date, tickers=tickers))

        monkeypatch.setattr(bf.MomentumCalculator, "calculate_bulk", bad)
    else:
        orig = bf.CVGCalculator.calculate_bulk

        def bad(self, context, start_date, end_date, tickers=None):
            return mutator(orig(self, context, start_date, end_date, tickers=tickers))

        monkeypatch.setattr(bf.CVGCalculator, "calculate_bulk", bad)

    with pytest.raises(ValueError, match=match):
        bf.compute_one_window_features(observations, window, cfg, canonical)


def test_write_staging_feature_file_schema_and_no_overwrite(bf, tmp_path):
    from src.backtest.surface_run_config import SurfaceDataPaths

    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    canonical = bf.build_canonical_key_index(observations)
    window = (6, 2)
    frame = bf.compute_one_window_features(observations, window, cfg, canonical)

    staging = tmp_path / "staging"
    staging.mkdir()
    written = bf.write_staging_feature_file(frame, staging, window)
    assert written.name == "features_6_2.parquet"
    assert written.parent == staging.resolve()

    paths = SurfaceDataPaths(cache_dir=tmp_path, features_dir=staging)
    assert paths.resolved_features_dir / "features_6_2.parquet" == written

    roundtrip = pd.read_parquet(written)
    assert list(roundtrip.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(
        roundtrip.sort_values(["ticker", "date"]).reset_index(drop=True),
        frame.sort_values(["ticker", "date"]).reset_index(drop=True),
        check_dtype=False,
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        bf.write_staging_feature_file(frame, staging, window)

    assert not (tmp_path / "features").exists()
    assert not list(tmp_path.rglob("features_backfill_v1.lineage.json"))


def test_write_staging_requires_existing_directory(bf, tmp_path):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _panel_for_window()
    frame = bf.compute_one_window_features(
        observations, (6, 2), cfg, bf.build_canonical_key_index(observations)
    )
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="staging directory does not exist"):
        bf.write_staging_feature_file(frame, missing, (6, 2))


# ---------------------------------------------------------------------------
# Block 3 — orchestration, staging validation, publication, receipt, CLI
# ---------------------------------------------------------------------------


def _synthetic_feature_frame(
    bf, observations: pd.DataFrame, window: tuple[int, int], config
) -> pd.DataFrame:
    max_lag, min_lag = window
    cols = bf.render_output_columns(config.output_columns_per_window, max_lag, min_lag)
    n = len(observations)
    frame = pd.DataFrame(
        {
            "ticker": observations["ticker"].astype(str).to_numpy(),
            "date": pd.to_datetime(observations["entry_date"]).dt.normalize().to_numpy(),
            cols[2]: [1.0] * n,
            cols[3]: [1] * n,
            cols[4]: [0.5] * n,
            cols[5]: [1] * n,
        }
    )
    frame.loc[0, cols[2]] = pd.NA
    return frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def _write_valid_staging_set(
    bf,
    staging: Path,
    observations: pd.DataFrame,
    windows: list[tuple[int, int]],
    config,
) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for window in windows:
        frame = _synthetic_feature_frame(bf, observations, window, config)
        bf.write_staging_feature_file(frame, staging, window)


@pytest.mark.parametrize(
    "existing",
    ["features", "features.building", "features_backfill_v1.lineage.json"],
)
def test_refuse_existing_output_paths(bf, tmp_path, existing):
    output_root = tmp_path / "out"
    output_root.mkdir()
    target = output_root / existing
    if existing.endswith(".json"):
        target.write_text("{}\n", encoding="utf-8")
    else:
        target.mkdir()
    with pytest.raises(ValueError, match="already exist"):
        bf.refuse_existing_outputs(output_root)


def test_run_refuses_existing_empty_staging_before_mutation(bf, tmp_path, monkeypatch):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "features.building").mkdir()
    monkeypatch.setattr(bf, "require_clean_repo_sha", lambda: "a" * 40)
    counts = {"config": 0, "d2": 0, "git": 0}

    orig_config = bf.load_feature_backfill_config
    orig_d2 = bf.validate_d2_input
    orig_git = bf.require_clean_repo_sha

    def count_config(path):
        counts["config"] += 1
        return orig_config(path)

    def count_d2(**kwargs):
        counts["d2"] += 1
        return orig_d2(**kwargs)

    def count_git():
        counts["git"] += 1
        return orig_git()

    monkeypatch.setattr(bf, "load_feature_backfill_config", count_config)
    monkeypatch.setattr(bf, "validate_d2_input", count_d2)
    monkeypatch.setattr(bf, "require_clean_repo_sha", count_git)

    with pytest.raises(ValueError, match="already exist"):
        bf.run_feature_backfill(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            config_path=SPEC_PATH,
            output_root=output_root,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
        )
    assert counts == {"config": 1, "d2": 1, "git": 1}
    assert not (output_root / "features").exists()
    assert not (output_root / "features_backfill_v1.lineage.json").exists()
    assert list((output_root / "features.building").iterdir()) == []
    assert cfg.windows[0] == (6, 2)


def test_orchestration_reuses_inputs_and_runs_all_windows_in_order(
    bf, tmp_path, monkeypatch
):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    output_root = tmp_path / "out"
    output_root.mkdir()
    repo_sha = "b" * 40

    counts = {"config": 0, "d2": 0, "git": 0, "canonical": 0}
    orig_config = bf.load_feature_backfill_config
    orig_d2 = bf.validate_d2_input
    orig_canonical = bf.build_canonical_key_index

    def count_config(path):
        counts["config"] += 1
        return orig_config(path)

    def count_d2(**kwargs):
        counts["d2"] += 1
        return orig_d2(**kwargs)

    def count_git():
        counts["git"] += 1
        return repo_sha

    def count_canonical(observations):
        counts["canonical"] += 1
        return orig_canonical(observations)

    monkeypatch.setattr(bf, "load_feature_backfill_config", count_config)
    monkeypatch.setattr(bf, "validate_d2_input", count_d2)
    monkeypatch.setattr(bf, "require_clean_repo_sha", count_git)
    monkeypatch.setattr(bf, "build_canonical_key_index", count_canonical)

    tracker: dict = {}

    def fake_compute(obs, window, config, canonical_key_index):
        tracker.setdefault("windows", []).append(window)
        tracker.setdefault("obs_ids", []).append(id(obs))
        tracker.setdefault("key_ids", []).append(id(canonical_key_index))
        return _synthetic_feature_frame(bf, obs, window, config)

    monkeypatch.setattr(bf, "compute_one_window_features", fake_compute)

    receipt_path = bf.run_feature_backfill(
        observations_path=obs_path,
        d2_lineage_path=lineage_path,
        config_path=SPEC_PATH,
        output_root=output_root,
        expected_snapshot_id="snaptest01",
        expected_build_id="buildtest01",
    )

    assert counts == {"config": 1, "d2": 1, "git": 1, "canonical": 1}
    assert tracker["windows"] == cfg.windows
    assert len(tracker["windows"]) == 281
    assert len(set(tracker["obs_ids"])) == 1
    assert len(set(tracker["key_ids"])) == 1
    assert receipt_path.is_file()
    assert (output_root / "features").is_dir()
    assert not (output_root / "features.building").exists()
    assert len(list((output_root / "features").glob("features_*.parquet"))) == 281


def test_validate_staging_accepts_valid_directory_with_na_features(bf, tmp_path):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _synthetic_observations()
    canonical = bf.build_canonical_key_index(observations)
    windows = [(6, 2), (6, 4)]
    staging = tmp_path / "features.building"
    _write_valid_staging_set(bf, staging, observations, windows, cfg)

    records = bf.validate_staging_directory(
        staging,
        windows=windows,
        output_columns_per_window=cfg.output_columns_per_window,
        canonical_key_index=canonical,
        expected_row_count=len(observations),
    )
    assert len(records) == 2
    assert records[0]["filename"] == "features_6_2.parquet"
    assert records[1]["filename"] == "features_6_4.parquet"
    assert all(len(r["file_sha256"]) == 64 for r in records)
    frame = pd.read_parquet(staging / "features_6_2.parquet")
    assert frame["mom_6_2_mean"].isna().any()


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing", "filenames do not match"),
        ("extra", "filenames do not match"),
        ("unreadable", "not a readable Parquet"),
        ("wrong_schema", "columns"),
        ("wrong_row_count", "row count"),
        ("null_key", "null"),
        ("duplicate_key", "duplicate"),
        ("key_mismatch", "keys are not exactly equal"),
        ("unsorted", "not deterministically sorted"),
    ],
)
def test_validate_staging_rejects_defects(bf, tmp_path, defect, match):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    observations = _synthetic_observations()
    canonical = bf.build_canonical_key_index(observations)
    windows = [(6, 2), (6, 4)]
    staging = tmp_path / "features.building"
    _write_valid_staging_set(bf, staging, observations, windows, cfg)
    target = staging / "features_6_2.parquet"

    if defect == "missing":
        target.unlink()
    elif defect == "extra":
        (staging / "features_999_1.parquet").write_bytes(target.read_bytes())
    elif defect == "unreadable":
        target.write_text("not-parquet", encoding="utf-8")
    elif defect == "wrong_schema":
        bad = pd.read_parquet(target).rename(columns={"date": "entry_date"})
        bad.to_parquet(target, index=False)
    elif defect == "wrong_row_count":
        bad = pd.read_parquet(target).iloc[:-1]
        bad.to_parquet(target, index=False)
    elif defect == "null_key":
        bad = pd.read_parquet(target)
        bad.loc[0, "ticker"] = None
        bad.to_parquet(target, index=False)
    elif defect == "duplicate_key":
        bad = pd.read_parquet(target)
        bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
        bad.to_parquet(target, index=False)
    elif defect == "key_mismatch":
        bad = pd.read_parquet(target)
        bad.loc[0, "ticker"] = "ZZZ"
        bad.to_parquet(target, index=False)
    elif defect == "unsorted":
        bad = pd.read_parquet(target).sort_values(
            ["ticker", "date"], ascending=[False, False]
        )
        bad.to_parquet(target, index=False)

    with pytest.raises(ValueError, match=match):
        bf.validate_staging_directory(
            staging,
            windows=windows,
            output_columns_per_window=cfg.output_columns_per_window,
            canonical_key_index=canonical,
            expected_row_count=len(observations),
        )


def test_pre_rename_failure_leaves_staging_only(bf, tmp_path, monkeypatch):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    output_root = tmp_path / "out"
    output_root.mkdir()
    monkeypatch.setattr(bf, "require_clean_repo_sha", lambda: "c" * 40)

    calls = {"n": 0}

    def fail_midway(obs, window, config, canonical_key_index):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("boom before rename")
        return _synthetic_feature_frame(bf, obs, window, config)

    monkeypatch.setattr(bf, "compute_one_window_features", fail_midway)

    with pytest.raises(RuntimeError, match="boom before rename"):
        bf.run_feature_backfill(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            config_path=SPEC_PATH,
            output_root=output_root,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
        )

    staging = output_root / "features.building"
    assert staging.is_dir()
    assert len(list(staging.glob("features_*.parquet"))) == 3
    assert not (output_root / "features").exists()
    assert not (output_root / "features_backfill_v1.lineage.json").exists()


def test_receipt_failure_keeps_published_features(bf, tmp_path, monkeypatch):
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    output_root = tmp_path / "out"
    output_root.mkdir()
    monkeypatch.setattr(bf, "require_clean_repo_sha", lambda: "d" * 40)

    def fake_compute(obs, window, config, canonical_key_index):
        return _synthetic_feature_frame(bf, obs, window, config)

    monkeypatch.setattr(bf, "compute_one_window_features", fake_compute)

    def boom_receipt(path, payload):
        raise OSError("receipt write failed")

    monkeypatch.setattr(bf, "write_completion_receipt", boom_receipt)

    with pytest.raises(OSError, match="receipt write failed"):
        bf.run_feature_backfill(
            observations_path=obs_path,
            d2_lineage_path=lineage_path,
            config_path=SPEC_PATH,
            output_root=output_root,
            expected_snapshot_id="snaptest01",
            expected_build_id="buildtest01",
        )

    assert (output_root / "features").is_dir()
    assert not (output_root / "features.building").exists()
    assert len(list((output_root / "features").glob("features_*.parquet"))) == 281
    assert not (output_root / "features_backfill_v1.lineage.json").exists()


def test_successful_cli_publishes_features_and_atomic_receipt(bf, tmp_path, monkeypatch):
    cfg = bf.load_feature_backfill_config(SPEC_PATH)
    df = _synthetic_observations()
    obs_path, lineage_path = _write_d2_pair(tmp_path, df)
    output_root = tmp_path / "out"
    output_root.mkdir()
    repo_sha = "e" * 40
    monkeypatch.setattr(bf, "require_clean_repo_sha", lambda: repo_sha)

    def fake_compute(obs, window, config, canonical_key_index):
        return _synthetic_feature_frame(bf, obs, window, config)

    monkeypatch.setattr(bf, "compute_one_window_features", fake_compute)

    rc = bf.main(
        [
            "--observations",
            str(obs_path),
            "--d2-lineage",
            str(lineage_path),
            "--config",
            str(SPEC_PATH),
            "--output-root",
            str(output_root),
            "--expected-snapshot-id",
            "snaptest01",
            "--expected-build-id",
            "buildtest01",
        ]
    )
    assert rc == 0

    features_dir = output_root / "features"
    receipt_path = output_root / "features_backfill_v1.lineage.json"
    assert features_dir.is_dir()
    assert not (output_root / "features.building").exists()
    feature_files = sorted(features_dir.glob("features_*.parquet"))
    assert len(feature_files) == 281
    assert receipt_path.is_file()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["artifact"] == "features_backfill_v1"
    assert receipt["schema_version"] == "1"
    assert receipt["repo_sha"] == repo_sha
    assert receipt["spec_version"] == cfg.spec_version
    assert receipt["spec_id"] == cfg.spec_id
    assert receipt["feature_config_sha256"] == cfg.config_sha256
    assert receipt["snapshot_id"] == "snaptest01"
    assert receipt["build_id"] == "buildtest01"
    assert receipt["observations_path"] == str(obs_path.resolve())
    assert receipt["d2_lineage_path"] == str(lineage_path.resolve())
    assert receipt["observations_file_sha256"] == sha256_file(obs_path)
    assert receipt["observations_row_count"] == len(df)
    assert receipt["observations_key_count"] == len(df)
    assert receipt["observations_output_key_digest"] == a1_key_digest(df)
    assert receipt["window_count"] == 281
    assert receipt["windows"] == [[m, n] for m, n in cfg.windows]
    assert receipt["baseline_window"] == {"max_lag": 42, "min_lag": 8}
    assert receipt["momentum_min_periods"] == 1
    assert receipt["cvg_min_periods"] == 1
    assert receipt["output_root"] == str(output_root.resolve())
    assert receipt["features_dir"] == str(features_dir.resolve())
    assert len(receipt["files"]) == 281
    assert receipt["files"][0]["filename"] == "features_6_2.parquet"
    assert receipt["files"][-1]["filename"] == "features_60_24.parquet"
    assert all(len(item["file_sha256"]) == 64 for item in receipt["files"])
    assert all(
        (features_dir / item["filename"]).is_file() for item in receipt["files"]
    )
    assert not list(output_root.glob("*.tmp"))


def test_cli_nonzero_on_error(bf, tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "require_clean_repo_sha", lambda: "f" * 40)
    rc = bf.main(
        [
            "--observations",
            str(tmp_path / "missing.parquet"),
            "--d2-lineage",
            str(tmp_path / "missing.lineage.json"),
            "--config",
            str(SPEC_PATH),
            "--output-root",
            str(tmp_path),
            "--expected-snapshot-id",
            "snaptest01",
            "--expected-build-id",
            "buildtest01",
        ]
    )
    assert rc == 1
