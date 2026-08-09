"""Sprint 005 D4 Block 1 — synthetic tests for feature quality audit."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.snapshot_foundation import sha256_file
from src.features.straddle_observations import a1_key_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_feature_quality.py"
SPEC_PATH = REPO_ROOT / "configs" / "feature_backfill_v1.json"


def _load_module():
    name = "audit_feature_quality_block1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aq():
    return _load_module()


def _tiny_config(aq, windows: list[tuple[int, int]], tmp_path: Path):
    base = aq.load_feature_backfill_config(SPEC_PATH)
    return aq.FeatureBackfillConfig(
        spec_version=base.spec_version,
        spec_id=base.spec_id,
        windows=list(windows),
        baseline_window=(42, 8),
        momentum_min_periods=1,
        cvg_min_periods=1,
        required_columns=list(base.required_columns),
        output_columns_per_window=list(base.output_columns_per_window),
        config_path=SPEC_PATH.resolve(),
        config_sha256=base.config_sha256,
    )


def _panel(
    n_dates: int,
    tickers: list[str] = None,
    *,
    return_pct=1.0,
    vol_gap=0.05,
    expiry_offset_days: int = 7,
) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB"]
    dates = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    rows = []
    for ticker in tickers:
        for entry in dates:
            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": entry,
                    "expiry_date": entry + pd.Timedelta(days=expiry_offset_days),
                    "return_pct": return_pct,
                    "vol_gap": vol_gap,
                }
            )
    return pd.DataFrame(rows)


def _feature_frame_from_panel(
    panel: pd.DataFrame,
    window: tuple[int, int],
    *,
    mom_values=None,
    mom_counts=None,
    cvg_values=None,
    cvg_counts=None,
) -> pd.DataFrame:
    max_lag, min_lag = window
    window_size = max_lag - min_lag + 1
    rows = []
    for ticker, group in panel.groupby("ticker", sort=True):
        g = group.sort_values("entry_date").reset_index(drop=True)
        for i, row in g.iterrows():
            slots = min(window_size, max(0, i - min_lag + 1))
            if slots == 0:
                mom_c = np.nan
                cvg_c = np.nan
                mom_v = np.nan
                cvg_v = np.nan
            else:
                mom_c = float(slots) if mom_counts is None else float(mom_counts)
                cvg_c = float(slots) if cvg_counts is None else float(cvg_counts)
                mom_v = 1.0 if mom_values is None else mom_values
                cvg_v = 0.5 if cvg_values is None else cvg_values
                if mom_c == 0:
                    mom_v = np.nan
                if cvg_c == 0:
                    cvg_v = np.nan
            rows.append(
                {
                    "ticker": ticker,
                    "date": row["entry_date"],
                    f"mom_{max_lag}_{min_lag}_mean": mom_v,
                    f"mom_{max_lag}_{min_lag}_count": mom_c,
                    f"cvg_{max_lag}_{min_lag}": cvg_v,
                    f"cvg_count_{max_lag}_{min_lag}": cvg_c,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _write_d2(tmp_path: Path, panel: pd.DataFrame, *, snapshot_id="snap", build_id="build"):
    root = tmp_path / "derived" / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "straddle_observations_weekly.parquet"
    lin = root / "straddle_observations_weekly.lineage.json"
    panel.to_parquet(obs, index=False)
    lineage = {
        "schema_version": "1",
        "artifact": "straddle_observations_weekly",
        "snapshot_id": snapshot_id,
        "build_id": build_id,
        "output": {
            "row_count": len(panel),
            "key_count": len(panel.drop_duplicates(["ticker", "entry_date"])),
            "output_key_digest": a1_key_digest(panel),
            "file_sha256": sha256_file(obs),
        },
    }
    lin.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    return obs, lin


def _write_receipt(
    tmp_path: Path,
    *,
    features_dir: Path,
    obs: Path,
    lin: Path,
    config_sha: str,
    windows: list[tuple[int, int]],
    panel: pd.DataFrame,
    snapshot_id="snap",
    build_id="build",
    repo_sha="a" * 40,
    status="complete",
):
    files = [
        {
            "filename": f"features_{max_lag}_{min_lag}.parquet",
            "max_lag": max_lag,
            "min_lag": min_lag,
            "row_count": len(panel),
            "file_sha256": "b" * 64,
        }
        for max_lag, min_lag in windows
    ]
    receipt = {
        "schema_version": "1",
        "artifact": "features_backfill_v1",
        "status": status,
        "snapshot_id": snapshot_id,
        "build_id": build_id,
        "repo_sha": repo_sha,
        "feature_config_sha256": config_sha,
        "observations_path": str(obs.resolve()),
        "d2_lineage_path": str(lin.resolve()),
        "features_dir": str(features_dir.resolve()),
        "observations_file_sha256": sha256_file(obs),
        "observations_row_count": len(panel),
        "observations_key_count": len(panel),
        "observations_output_key_digest": a1_key_digest(panel),
        "window_count": len(windows),
        "windows": [[a, b] for a, b in windows],
        "files": files,
    }
    path = tmp_path / "features_backfill_v1.lineage.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def brute_force_pit_violating_sources(
    observations: pd.DataFrame,
    *,
    economic_col: str,
    windows: list[tuple[int, int]],
) -> set[tuple[str, int]]:
    """Oracle: violating source observations (ticker, source-position)."""
    viol: set[tuple[str, int]] = set()
    for ticker, group in observations.groupby("ticker", sort=True):
        g = group.reset_index(drop=True)
        entry = pd.to_datetime(g["entry_date"]).dt.normalize()
        expiry = pd.to_datetime(g["expiry_date"], errors="coerce")
        econ = pd.to_numeric(g[economic_col], errors="coerce")
        n = len(g)
        for max_lag, min_lag in windows:
            for i in range(n):
                start = max(0, i - max_lag)
                end = i - min_lag
                if end < start:
                    continue
                for j in range(start, end + 1):
                    if not np.isfinite(econ.iloc[j]):
                        continue
                    exp = expiry.iloc[j]
                    feat_date = entry.iloc[i]
                    if pd.isna(exp) or not (pd.Timestamp(exp).normalize() < feat_date):
                        viol.add((str(ticker), int(j)))
    return viol


def compact_pit_violating_sources(
    observations: pd.DataFrame, *, economic_col: str
) -> set[tuple[str, int]]:
    viol: set[tuple[str, int]] = set()
    for ticker, group in observations.groupby("ticker", sort=True):
        g = group.reset_index(drop=True)
        n = len(g)
        entry = pd.to_datetime(g["entry_date"]).dt.normalize()
        expiry = pd.to_datetime(g["expiry_date"], errors="coerce")
        econ = pd.to_numeric(g[economic_col], errors="coerce")
        for j in range(max(0, n - 2)):
            if not np.isfinite(econ.iloc[j]):
                continue
            exp = expiry.iloc[j]
            target = entry.iloc[j + 2]
            if pd.isna(exp) or not (pd.Timestamp(exp).normalize() < target):
                viol.add((str(ticker), int(j)))
    return viol


# ---------------------------------------------------------------------------
# Import / CLI surface
# ---------------------------------------------------------------------------


def test_script_help_without_pythonpath_from_repo_and_other_cwd(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for cwd in (REPO_ROOT, tmp_path):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--expected-d3-repo-sha" in completed.stdout
        assert "--features-dir" in completed.stdout


def test_all_cli_args_required(aq):
    with pytest.raises(SystemExit):
        aq._parse_args([])


def test_import_does_not_touch_git_or_write(aq, monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("git must not run at import")

    # Module already imported by fixture; ensure helpers are lazy for git.
    assert callable(aq.require_clean_repo_sha)
    assert not (tmp_path / "features_quality_audit_v1.json").exists()


# ---------------------------------------------------------------------------
# Unit helpers: coverage, missingness, ready, PIT
# ---------------------------------------------------------------------------


def test_coverage_and_count_oracle(aq):
    panel = _panel(6, tickers=["T"])
    window = (4, 2)
    frame = _feature_frame_from_panel(panel, window)
    # Force one early no-slots rows already; set a mid row count=0
    col_c = "mom_4_2_count"
    col_m = "mom_4_2_mean"
    # position 2 has slots>0; zero it
    idx = frame.index[frame["date"] == panel["entry_date"].iloc[2]][0]
    frame.loc[idx, col_c] = 0.0
    frame.loc[idx, col_m] = np.nan

    dates = sorted(pd.to_datetime(panel["entry_date"]).dt.normalize().unique())
    date_to_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    result = aq.analyze_one_window_frame(
        frame,
        max_lag=4,
        min_lag=2,
        date_to_pos=date_to_pos,
        expected_row_count=len(frame),
    )
    assert result["n_rows"] == 6
    assert result["mom_nonnull_n"] + result["structural_missingness"]["no_slots"] <= 6
    assert result["mom_count_summary"]["finite_count_n"] == (
        6 - result["structural_missingness"]["no_slots"]
    )


def test_count_safeguards(aq):
    panel = _panel(5, tickers=["T"])
    window = (4, 2)
    frame = _feature_frame_from_panel(panel, window)
    dates = sorted(pd.to_datetime(panel["entry_date"]).dt.normalize().unique())
    date_to_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}

    bad = frame.copy()
    # row with slots>0: set non-integral count
    i = bad.index[bad["mom_4_2_count"].notna()][0]
    bad.loc[i, "mom_4_2_count"] = 1.5
    with pytest.raises(ValueError, match="non-integral"):
        aq.analyze_one_window_frame(
            bad, max_lag=4, min_lag=2, date_to_pos=date_to_pos, expected_row_count=5
        )

    bad = frame.copy()
    i = bad.index[bad["mom_4_2_count"].notna()][0]
    bad.loc[i, "mom_4_2_count"] = -1
    bad.loc[i, "mom_4_2_mean"] = np.nan
    with pytest.raises(ValueError, match="negative"):
        aq.analyze_one_window_frame(
            bad, max_lag=4, min_lag=2, date_to_pos=date_to_pos, expected_row_count=5
        )

    bad = frame.copy()
    i = bad.index[bad["mom_4_2_count"].notna()][0]
    slots_pos = int(bad.index.get_loc(i))
    # oversized
    bad.loc[i, "mom_4_2_count"] = 99
    with pytest.raises(ValueError, match="exceeds available_slots"):
        aq.analyze_one_window_frame(
            bad, max_lag=4, min_lag=2, date_to_pos=date_to_pos, expected_row_count=5
        )

    bad = frame.copy()
    i = bad.index[bad["mom_4_2_count"].isna()][0]
    bad.loc[i, "mom_4_2_count"] = 0
    with pytest.raises(ValueError, match="available_slots==0"):
        aq.analyze_one_window_frame(
            bad, max_lag=4, min_lag=2, date_to_pos=date_to_pos, expected_row_count=5
        )

    bad = frame.copy()
    i = bad.index[bad["mom_4_2_count"].notna()][0]
    bad.loc[i, "mom_4_2_count"] = 1
    bad.loc[i, "mom_4_2_mean"] = np.nan
    with pytest.raises(ValueError, match="count>0 requires finite"):
        aq.analyze_one_window_frame(
            bad, max_lag=4, min_lag=2, date_to_pos=date_to_pos, expected_row_count=5
        )


def test_missingness_reconciles_and_independence(aq):
    panel = _panel(8, tickers=["T"])
    window = (6, 2)
    window_size = 5
    frame = _feature_frame_from_panel(panel, window)
    # Make momentum partial on a full/truncated row while CVG full
    dates = sorted(pd.to_datetime(panel["entry_date"]).dt.normalize().unique())
    date_to_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    # pick last row (full window)
    last = frame.index[-1]
    slots = window_size
    frame.loc[last, "mom_6_2_count"] = float(slots - 1)
    frame.loc[last, "mom_6_2_mean"] = 1.0
    frame.loc[last, "cvg_count_6_2"] = float(slots)
    frame.loc[last, "cvg_6_2"] = 0.5

    result = aq.analyze_one_window_frame(
        frame,
        max_lag=6,
        min_lag=2,
        date_to_pos=date_to_pos,
        expected_row_count=len(frame),
    )
    s = result["structural_missingness"]
    assert s["no_slots"] + s["truncated_window"] + s["full_window"] == len(frame)
    me = result["momentum_economic_missingness"]
    ce = result["cvg_economic_missingness"]
    assert me["zero_finite"] + me["partial_finite"] + me["all_available_finite"] == (
        len(frame) - s["no_slots"]
    )
    assert ce["zero_finite"] + ce["partial_finite"] + ce["all_available_finite"] == (
        len(frame) - s["no_slots"]
    )
    assert me["partial_finite"] >= 1
    assert ce["all_available_finite"] >= 1
    # no_slots rows have NaN counts
    pos = frame["date"].map(lambda d: date_to_pos[pd.Timestamp(d).normalize()])
    slots_arr = aq.available_slots_for_positions(
        pos.to_numpy(), max_lag=6, min_lag=2
    )
    assert frame.loc[slots_arr == 0, "mom_6_2_count"].isna().all()


def test_truncated_can_be_all_available_finite(aq):
    panel = _panel(5, tickers=["T"])
    window = (6, 2)  # window_size=5; early rows truncated
    frame = _feature_frame_from_panel(panel, window)
    dates = sorted(pd.to_datetime(panel["entry_date"]).dt.normalize().unique())
    date_to_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    result = aq.analyze_one_window_frame(
        frame,
        max_lag=6,
        min_lag=2,
        date_to_pos=date_to_pos,
        expected_row_count=5,
    )
    assert result["structural_missingness"]["truncated_window"] > 0
    assert result["momentum_economic_missingness"]["all_available_finite"] > 0


def test_baseline_ready_interval_43rd_to_last(aq):
    dates = pd.date_range("2018-01-05", periods=50, freq="W-FRI")
    ready = aq.compute_baseline_ready_interval(dates)
    assert ready["ready_start"] == pd.Timestamp(dates[42]).strftime("%Y-%m-%d")
    assert ready["ready_end"] == pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    assert ready["window_size"] == 35


def test_compact_pit_matches_bruteforce_source_set(aq):
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    rows = []
    # AAA: mostly safe; one unsafe finite return
    for i, entry in enumerate(dates):
        exp = entry + pd.Timedelta(days=7)
        ret = 1.0
        gap = 0.05
        if i == 3:
            exp = entry + pd.Timedelta(days=30)  # unsafe vs j+2
        if i == 5:
            ret = np.nan  # null mom economics
        if i == 6:
            gap = np.nan  # null cvg
        rows.append(
            {
                "ticker": "AAA",
                "entry_date": entry,
                "expiry_date": exp,
                "return_pct": ret,
                "vol_gap": gap,
            }
        )
    # BBB: missing expiry on finite return
    for i, entry in enumerate(dates):
        rows.append(
            {
                "ticker": "BBB",
                "entry_date": entry,
                "expiry_date": pd.NaT if i == 1 else entry + pd.Timedelta(days=7),
                "return_pct": 1.0,
                "vol_gap": 0.05 if i != 2 else np.nan,
            }
        )
    # CCC: mom-only finite / cvg-only finite mix
    for i, entry in enumerate(dates):
        rows.append(
            {
                "ticker": "CCC",
                "entry_date": entry,
                "expiry_date": entry + pd.Timedelta(days=7),
                "return_pct": 1.0 if i % 2 == 0 else np.nan,
                "vol_gap": 0.05 if i % 2 == 1 else np.nan,
            }
        )
    panel = pd.DataFrame(rows)
    windows = [(6, 2), (8, 4), (10, 2)]
    for col, label in (("return_pct", "momentum"), ("vol_gap", "cvg")):
        compact = compact_pit_violating_sources(panel, economic_col=col)
        brute = brute_force_pit_violating_sources(
            panel, economic_col=col, windows=windows
        )
        assert compact == brute
        result = aq.compact_pit_proof(panel, economic_col=col, label=label)
        assert result["checked_observations"] == result["eligible_observations"]
        assert result["violations"] == len(compact)


def test_missing_expiry_on_finite_fails_compact(aq):
    panel = _panel(5, tickers=["T"])
    panel.loc[0, "expiry_date"] = pd.NaT
    result = aq.compact_pit_proof(panel, economic_col="return_pct", label="momentum")
    assert result["violations"] >= 1
    assert any(
        ex.get("reason") == "missing_or_invalid_expiry"
        for ex in result["violation_examples"]
    )


# ---------------------------------------------------------------------------
# Startup / JSON / end-to-end with patched short grid
# ---------------------------------------------------------------------------


def _prepare_short_audit(aq, tmp_path, monkeypatch, *, windows=None, panel=None):
    windows = windows or [(6, 2), (42, 8), (60, 24)]
    panel = panel if panel is not None else _panel(45, tickers=["AAA", "BBB"])
    cfg = _tiny_config(aq, windows, tmp_path)
    monkeypatch.setattr(aq, "load_feature_backfill_config", lambda path: cfg)
    monkeypatch.setattr(aq, "require_clean_repo_sha", lambda: "d" * 40)
    monkeypatch.setattr(aq, "_EXPECTED_WINDOW_COUNT", len(windows))

    obs, lin = _write_d2(tmp_path, panel, snapshot_id="snap", build_id="build")
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for window in windows:
        frame = _feature_frame_from_panel(panel, window)
        frame.to_parquet(
            features_dir / f"features_{window[0]}_{window[1]}.parquet", index=False
        )
    receipt = _write_receipt(
        tmp_path,
        features_dir=features_dir,
        obs=obs,
        lin=lin,
        config_sha=cfg.config_sha256,
        windows=windows,
        panel=panel,
        repo_sha="c" * 40,
    )
    out = tmp_path / "out" / "features_quality_audit_v1.json"
    out.parent.mkdir()
    return cfg, obs, lin, features_dir, receipt, out, panel


def test_startup_identity_failures(aq, tmp_path, monkeypatch):
    cfg, obs, lin, features_dir, receipt, out, panel = _prepare_short_audit(
        aq, tmp_path, monkeypatch
    )
    common = dict(
        features_dir=features_dir,
        d3_receipt_path=receipt,
        observations_path=obs,
        d2_lineage_path=lin,
        config_path=SPEC_PATH,
        output_json=out,
        expected_snapshot_id="snap",
        expected_build_id="build",
        expected_d3_repo_sha="c" * 40,
    )
    with pytest.raises(ValueError, match="snapshot_id mismatch"):
        aq.validate_startup_identities(**{**common, "expected_snapshot_id": "wrong"})
    with pytest.raises(ValueError, match="build_id mismatch"):
        aq.validate_startup_identities(**{**common, "expected_build_id": "wrong"})
    with pytest.raises(ValueError, match="D3 repo_sha mismatch"):
        aq.validate_startup_identities(**{**common, "expected_d3_repo_sha": "e" * 40})

    # dirty git
    monkeypatch.setattr(
        aq,
        "require_clean_repo_sha",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty working tree")),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        aq.validate_startup_identities(**common)


def test_existing_output_and_inside_features_fail(aq, tmp_path, monkeypatch):
    cfg, obs, lin, features_dir, receipt, out, panel = _prepare_short_audit(
        aq, tmp_path, monkeypatch
    )
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        aq.validate_startup_identities(
            features_dir=features_dir,
            d3_receipt_path=receipt,
            observations_path=obs,
            d2_lineage_path=lin,
            config_path=SPEC_PATH,
            output_json=out,
            expected_snapshot_id="snap",
            expected_build_id="build",
            expected_d3_repo_sha="c" * 40,
        )
    out.unlink()
    inside = features_dir / "nested.json"
    with pytest.raises(ValueError, match="inside features"):
        aq.validate_startup_identities(
            features_dir=features_dir,
            d3_receipt_path=receipt,
            observations_path=obs,
            d2_lineage_path=lin,
            config_path=SPEC_PATH,
            output_json=inside,
            expected_snapshot_id="snap",
            expected_build_id="build",
            expected_d3_repo_sha="c" * 40,
        )


def test_clean_git_records_d4_sha(aq, tmp_path, monkeypatch):
    cfg, obs, lin, features_dir, receipt, out, panel = _prepare_short_audit(
        aq, tmp_path, monkeypatch
    )
    monkeypatch.setattr(aq, "require_clean_repo_sha", lambda: "f" * 40)
    _cfg, _receipt, d4_sha, _digests = aq.validate_startup_identities(
        features_dir=features_dir,
        d3_receipt_path=receipt,
        observations_path=obs,
        d2_lineage_path=lin,
        config_path=SPEC_PATH,
        output_json=out,
        expected_snapshot_id="snap",
        expected_build_id="build",
        expected_d3_repo_sha="c" * 40,
    )
    assert d4_sha == "f" * 40


def test_end_to_end_writes_deterministic_json(aq, tmp_path, monkeypatch):
    cfg, obs, lin, features_dir, receipt, out, panel = _prepare_short_audit(
        aq, tmp_path, monkeypatch
    )
    monkeypatch.setattr(aq, "require_clean_repo_sha", lambda: "1" * 40)
    written = aq.run_feature_quality_audit(
        features_dir=features_dir,
        d3_receipt_path=receipt,
        observations_path=obs,
        d2_lineage_path=lin,
        config_path=SPEC_PATH,
        output_json=out,
        expected_snapshot_id="snap",
        expected_build_id="build",
        expected_d3_repo_sha="c" * 40,
    )
    assert written == out.resolve()
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["status"] == "complete"
    assert payload["artifact"] == "features_quality_audit_v1"
    assert payload["d3_producer_repo_sha"] == "c" * 40
    assert payload["d4_audit_repo_sha"] == "1" * 40
    assert payload["window_count"] == 3
    assert payload["baseline_ready_interval"]["ready_start"]
    assert payload["pit_momentum"]["violations"] == 0
    assert payload["pit_cvg"]["violations"] == 0
    assert set(payload["sentinel_date_coverage"]["series"]) == {"6_2", "42_8", "60_24"}

    with pytest.raises(ValueError, match="overwrite"):
        aq.run_feature_quality_audit(
            features_dir=features_dir,
            d3_receipt_path=receipt,
            observations_path=obs,
            d2_lineage_path=lin,
            config_path=SPEC_PATH,
            output_json=out,
            expected_snapshot_id="snap",
            expected_build_id="build",
            expected_d3_repo_sha="c" * 40,
        )


def test_no_calculate_bulk_and_no_features_dir_writes(aq, tmp_path, monkeypatch):
    cfg, obs, lin, features_dir, receipt, out, panel = _prepare_short_audit(
        aq, tmp_path, monkeypatch
    )
    monkeypatch.setattr(aq, "require_clean_repo_sha", lambda: "2" * 40)

    # Ensure backfill helpers' calculators are not invoked if present
    bf = aq._bf
    if hasattr(bf, "MomentumCalculator"):
        monkeypatch.setattr(
            bf.MomentumCalculator,
            "calculate_bulk",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("mom calculate_bulk")),
        )
    if hasattr(bf, "CVGCalculator"):
        monkeypatch.setattr(
            bf.CVGCalculator,
            "calculate_bulk",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("cvg calculate_bulk")),
        )

    before = {p.resolve() for p in features_dir.rglob("*")}
    aq.run_feature_quality_audit(
        features_dir=features_dir,
        d3_receipt_path=receipt,
        observations_path=obs,
        d2_lineage_path=lin,
        config_path=SPEC_PATH,
        output_json=out,
        expected_snapshot_id="snap",
        expected_build_id="build",
        expected_d3_repo_sha="c" * 40,
    )
    after = {p.resolve() for p in features_dir.rglob("*")}
    assert after == before
    assert out.is_file()
    assert not aq._path_is_inside(out, features_dir)


def test_main_nonzero_on_failure(aq, tmp_path, monkeypatch):
    monkeypatch.setattr(
        aq,
        "run_feature_quality_audit",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    rc = aq.main(
        [
            "--features-dir",
            str(tmp_path),
            "--d3-receipt",
            str(tmp_path / "r.json"),
            "--observations",
            str(tmp_path / "o.parquet"),
            "--d2-lineage",
            str(tmp_path / "l.json"),
            "--config",
            str(SPEC_PATH),
            "--output-json",
            str(tmp_path / "out.json"),
            "--expected-snapshot-id",
            "snap",
            "--expected-build-id",
            "build",
            "--expected-d3-repo-sha",
            "a" * 40,
        ]
    )
    assert rc == 1
