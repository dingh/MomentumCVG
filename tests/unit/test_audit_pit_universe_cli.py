"""CLI tests for scripts/audit_pit_universe.py (Sprint 004 C7.3).

Synthetic tmp_path fixtures only — does not read production paths under
C:/MomentumCVG_env or C:/ORATS.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.pit_universe_audit import (
    recompute_rolling_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "audit_pit_universe.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("audit_pit_universe_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

WEEKS = [
    pd.Timestamp("2024-01-05"),
    pd.Timestamp("2024-01-12"),
    pd.Timestamp("2024-01-19"),
    pd.Timestamp("2024-01-26"),
    pd.Timestamp("2024-02-02"),
    pd.Timestamp("2024-02-09"),
    pd.Timestamp("2024-02-16"),
    pd.Timestamp("2024-03-01"),  # gap > 7 days from 2024-02-16
    pd.Timestamp("2024-03-08"),
]
# Extra weekly candidate after last panel snapshot for discovery mapping.
POST_LAST = pd.Timestamp("2024-03-15")

LOOKBACK = 3
MIN_VQW = 2
BUILD = dict(
    lookback_weeks=LOOKBACK,
    min_valid_quote_weeks=MIN_VQW,
    dte_min=5,
    dte_max=60,
    dvol_top_pct=0.20,
    spread_bot_pct=1.0,
    liquidity_source="raw_option_bid_x_volume_sum_dte_5_60",
)


def _wrow(week, ticker, vol, spread, valid=True):
    return dict(
        week_end_date=week,
        ticker=ticker,
        weekly_atm_straddle_dollar_vol=vol,
        weekly_atm_spread_pct=spread,
        weekly_has_valid_quote=valid,
    )


def _build_weekly(tickers_vols: dict[str, float], weeks: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for w in weeks:
        for ticker, base in tickers_vols.items():
            rows.append(_wrow(w, ticker, base, 0.02, True))
    # Optional weak / invalid ticker for missing-liquidity discovery.
    return pd.DataFrame(rows)


def _panel_from_weekly(
    weekly: pd.DataFrame,
    snapshots: list[pd.Timestamp],
    tickers: list[str],
    *,
    inject_new_ticker_on: pd.Timestamp | None = None,
    new_ticker: str = "NEW1",
) -> pd.DataFrame:
    """Build a panel whose stored metrics match independent rolling recompute."""
    rows = []
    for s in snapshots:
        checked = list(tickers)
        if inject_new_ticker_on is not None and s >= inject_new_ticker_on:
            if new_ticker not in checked:
                checked = checked + [new_ticker]
        rec = recompute_rolling_snapshot(
            s, checked, weekly, LOOKBACK, MIN_VQW
        )
        for ticker in checked:
            r = rec.loc[ticker]
            spread = r["atm_spread_pct"]
            rows.append(
                dict(
                    month_date=s,
                    ticker=ticker,
                    atm_straddle_dollar_vol=float(r["atm_straddle_dollar_vol"]),
                    atm_spread_pct=(float(spread) if pd.notna(spread) else float("nan")),
                    has_valid_atm_pair=bool(r["has_valid_atm_pair"]),
                    valid_quote_weeks=int(r["valid_quote_weeks"]),
                    zero_volume_weeks=int(r["zero_volume_weeks"]),
                    window_start_date=r["window_start_date"],
                    window_end_date=r["window_end_date"],
                    window_shortfall=int(r["window_shortfall"]),
                    **BUILD,
                )
            )
    return pd.DataFrame(rows)


def _write_artifacts(
    tmp_path: Path,
    panel: pd.DataFrame,
    weekly: pd.DataFrame,
    liquid: list[str],
) -> tuple[Path, Path, Path, Path]:
    panel_path = tmp_path / "panel.parquet"
    weekly_path = tmp_path / "weekly.parquet"
    liquid_path = tmp_path / "liquid_tickers.csv"
    report_path = tmp_path / "out" / "report.md"
    panel.to_parquet(panel_path, index=False)
    weekly.to_parquet(weekly_path, index=False)
    pd.DataFrame({"Ticker": liquid}).to_csv(liquid_path, index=False)
    return panel_path, weekly_path, liquid_path, report_path


def _base_argv(
    panel_path: Path,
    weekly_path: Path,
    liquid_path: Path,
    report_path: Path,
    *extra: str,
) -> list[str]:
    return [
        "--panel-path",
        str(panel_path),
        "--weekly-path",
        str(weekly_path),
        "--liquid-tickers-path",
        str(liquid_path),
        "--output-report",
        str(report_path),
        *extra,
    ]


def _pass_fixture(tmp_path: Path, *, early_shortfall_sample: bool = False):
    """Full PASS-capable artifacts; optional early snapshot for WARN shortfall."""
    tickers = ["AAA", "BBB", "CCC"]
    vols = {"AAA": 300.0, "BBB": 200.0, "CCC": 100.0}
    # Include a low-quality ticker present from an intermediate snapshot for discovery.
    weeks = list(WEEKS) + [POST_LAST]
    weekly_rows = []
    for w in weeks:
        for t, v in vols.items():
            weekly_rows.append(_wrow(w, t, v, 0.02, True))
        # Invalid ATM pair source weeks for DDD (missing/new liquidity case).
        weekly_rows.append(_wrow(w, "DDD", 50.0, 0.02, False))
        # New ticker appears only late.
        if w >= WEEKS[6]:
            weekly_rows.append(_wrow(w, "NEW1", 80.0, 0.03, True))
    weekly = pd.DataFrame(weekly_rows)

    snapshots = list(WEEKS)
    panel = _panel_from_weekly(
        weekly,
        snapshots,
        tickers + ["DDD"],
        inject_new_ticker_on=WEEKS[6],
        new_ticker="NEW1",
    )
    # Ensure NEW1 weekly exists for recompute on late snaps already handled.
    liquid = sorted({*tickers, "DDD", "NEW1"})
    paths = _write_artifacts(tmp_path, panel, weekly, liquid)
    sample = "2024-01-19" if early_shortfall_sample else "2024-03-15"
    return paths, sample, panel, weekly


# ---------------------------------------------------------------------------
# C1 — PASS fixture exits 0
# ---------------------------------------------------------------------------


def test_c1_pass_fixture_exits_0(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `PASS`" in text


# ---------------------------------------------------------------------------
# C2 — artifact FAIL exits 1
# ---------------------------------------------------------------------------


def test_c2_artifact_fail_exits_1(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, weekly = _pass_fixture(
        tmp_path
    )
    # Duplicate grain → FAIL
    bad = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    bad.to_parquet(panel_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 1
    assert report_path.is_file()
    assert "overall status: `FAIL`" in report_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C3 — missing path exits 2
# ---------------------------------------------------------------------------


def test_c3_missing_path_exits_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(
            tmp_path / "missing_panel.parquet",
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2
    assert not report_path.exists()


# ---------------------------------------------------------------------------
# C4 / C5 — WARN exit codes
# ---------------------------------------------------------------------------


def test_c4_warn_exits_0_without_strict(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, _, _ = _pass_fixture(
        tmp_path, early_shortfall_sample=True
    )
    # Early trade date resolves to first/early snapshot with window_shortfall > 0.
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            "2024-01-12",
            "--sample-date",
            "2024-03-15",
        )
    )
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `WARN`" in text
    assert code == 0


def test_c5_warn_exits_1_with_strict(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, _, _ = _pass_fixture(
        tmp_path, early_shortfall_sample=True
    )
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            "2024-01-12",
            "--sample-date",
            "2024-03-15",
            "--strict",
        )
    )
    assert code == 1
    assert "overall status: `WARN`" in report_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C6 / C7 — report created + byte-deterministic
# ---------------------------------------------------------------------------


def test_c6_markdown_report_created(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    for section in (
        "# C7 PIT Universe Audit",
        "## Verdict",
        "## Scope and parameters",
        "## Artifact inventory",
        "## Artifact checks",
        "## Supported parameter envelope",
        "## Sample discovery",
        "## PIT sample results",
        "## Rolling provenance",
        "## Sample superset coverage",
        "## Full-history superset coverage",
        "## Blocking failures",
        "## Warnings",
    ):
        assert section in text


def test_c7_report_rendering_byte_deterministic(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    argv = _base_argv(
        panel_path,
        weekly_path,
        liquid_path,
        report_path,
        "--sample-date",
        sample,
    )
    cli.main(argv)
    first = report_path.read_bytes()
    report_path.unlink()
    cli.main(argv)
    second = report_path.read_bytes()
    assert first == second
    assert first.endswith(b"\n")


# ---------------------------------------------------------------------------
# C8 / C9 — empty artifacts cannot PASS
# ---------------------------------------------------------------------------


def test_c8_empty_panel_cannot_pass(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, weekly = _pass_fixture(
        tmp_path
    )
    empty = panel.iloc[0:0].copy()
    empty.to_parquet(panel_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `PASS`" not in text
    assert "overall status: `FAIL`" in text


def test_c9_empty_weekly_cannot_pass(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, weekly = _pass_fixture(
        tmp_path
    )
    empty = weekly.iloc[0:0].copy()
    empty.to_parquet(weekly_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 1
    assert "overall status: `FAIL`" in report_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C10 / C20 — liquid ticker CSV usage errors
# ---------------------------------------------------------------------------


def test_c10_empty_liquid_ticker_csv_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    pd.DataFrame({"Ticker": []}).to_csv(liquid_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2


def test_c20_missing_ticker_column_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    pd.DataFrame({"Symbol": ["AAA"]}).to_csv(liquid_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2


# ---------------------------------------------------------------------------
# C11 — unsupported dvol envelope
# ---------------------------------------------------------------------------


def test_c11_unsupported_dvol_envelope_exits_1(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
            "--dvol-top-pct",
            "0.50",
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `FAIL`" in text
    assert "supported: `False`" in text


# ---------------------------------------------------------------------------
# C12 — repeatable explicit dates dedupe
# ---------------------------------------------------------------------------


def test_c12_repeatable_explicit_sample_dates_dedupe(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            "2024-03-15",
            "--sample-date",
            "2024-02-16",
            "--sample-date",
            "2024-03-15",
        )
    )
    assert code in (0, 1)  # may WARN on early shortfall if included; here both late
    text = report_path.read_text(encoding="utf-8")
    # Two distinct trade dates after dedupe (chronological).
    assert text.count("trade_date: `2024-02-16`") == 1
    assert text.count("trade_date: `2024-03-15`") == 1


# ---------------------------------------------------------------------------
# C13 — neither sample mode → 2
# ---------------------------------------------------------------------------


def test_c13_neither_sample_date_nor_discover_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(panel_path, weekly_path, liquid_path, report_path)
    )
    assert code == 2


# ---------------------------------------------------------------------------
# C14–C18 — discovery
# ---------------------------------------------------------------------------


def test_c14_discovery_maps_target_to_later_t(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, panel, weekly = _pass_fixture(
        tmp_path
    )
    specs, check = cli.discover_audit_samples(panel, weekly)
    assert check.name == "sample_discovery"
    for spec in specs:
        assert spec.target_snapshot_date is not None
        assert spec.trade_date > spec.target_snapshot_date
        resolved = cli._resolve_prior_snapshot(
            spec.trade_date, cli._panel_snapshots(panel)
        )
        assert resolved == spec.target_snapshot_date

    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--discover-samples",
        )
    )
    assert code in (0, 1)
    text = report_path.read_text(encoding="utf-8")
    blocks = text.split("### Sample ")
    for block in blocks[1:]:
        if "missing_or_new_liquidity" not in block and "normal" not in block and "boundary" not in block:
            continue
        lines = block.splitlines()
        target = next(
            (ln for ln in lines if ln.startswith("- target_snapshot_date:")), ""
        )
        trade = next((ln for ln in lines if ln.startswith("- trade_date:")), "")
        resolved = next(
            (ln for ln in lines if ln.startswith("- resolved_snapshot_date:")), ""
        )
        tval = target.split("`")[1]
        rval = resolved.split("`")[1]
        trade_val = trade.split("`")[1]
        assert tval != "None"
        assert trade_val > tval
        assert tval == rval


def test_c15_discovery_never_uses_trade_date_eq_target(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, panel, weekly = _pass_fixture(
        tmp_path
    )
    specs, check = cli.discover_audit_samples(panel, weekly)
    assert check.name == "sample_discovery"
    for spec in specs:
        assert spec.target_snapshot_date is not None
        assert spec.trade_date > spec.target_snapshot_date
        assert spec.trade_date != spec.target_snapshot_date


def test_c16_discovery_deterministic_labels_order(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, panel, weekly = _pass_fixture(
        tmp_path
    )
    a, ca = cli.discover_audit_samples(panel, weekly)
    b, cb = cli.discover_audit_samples(panel, weekly)
    assert [(s.trade_date, s.target_snapshot_date, s.labels) for s in a] == [
        (s.trade_date, s.target_snapshot_date, s.labels) for s in b
    ]
    assert ca.status == cb.status
    assert ca.details == cb.details


def test_c17_fewer_than_three_discoverable_cases_warn(tmp_path: Path):
    # Two snapshots with clean tickers → normal + baseline missing map (2 cases, WARN).
    weeks = [WEEKS[0], WEEKS[1], WEEKS[2]]
    weekly = pd.DataFrame(
        [
            _wrow(weeks[0], "AAA", 300.0, 0.02, True),
            _wrow(weeks[1], "AAA", 300.0, 0.02, True),
            _wrow(weeks[2], "AAA", 300.0, 0.02, True),
            _wrow(weeks[0], "BBB", 200.0, 0.02, True),
            _wrow(weeks[1], "BBB", 200.0, 0.02, True),
            _wrow(weeks[2], "BBB", 200.0, 0.02, True),
        ]
    )
    panel = _panel_from_weekly(weekly, [weeks[1], weeks[2]], ["AAA", "BBB"])
    panel_path, weekly_path, liquid_path, report_path = _write_artifacts(
        tmp_path, panel, weekly, ["AAA", "BBB"]
    )
    _, discovery_check = cli.discover_audit_samples(panel, weekly)
    assert discovery_check.status == "WARN"
    assert discovery_check.details["mapped_case_count"] in (1, 2)

    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--discover-samples",
            "--sample-date",
            "2024-01-26",
        )
    )
    text = report_path.read_text(encoding="utf-8")
    assert "`sample_discovery`: `WARN`" in text
    assert discovery_check.details["mapped_case_count"] in (1, 2)
    assert code in (0, 1)


def test_c18_zero_discovered_samples_produces_fail(tmp_path: Path):
    # Panel with one snapshot and weekly candidates that cannot map (no T > S in weekly).
    snap = pd.Timestamp("2024-06-01")
    weekly = pd.DataFrame(
        [
            _wrow(pd.Timestamp("2024-05-01"), "AAA", 100.0, 0.02, True),
            _wrow(pd.Timestamp("2024-05-08"), "AAA", 100.0, 0.02, True),
            _wrow(pd.Timestamp("2024-05-15"), "AAA", 100.0, 0.02, True),
        ]
    )
    # Build panel snapshot after all weekly dates so no T > S exists in weekly.
    snap = pd.Timestamp("2024-06-01")
    panel = pd.DataFrame(
        [
            dict(
                month_date=snap,
                ticker="AAA",
                atm_straddle_dollar_vol=100.0,
                atm_spread_pct=0.02,
                has_valid_atm_pair=True,
                valid_quote_weeks=3,
                zero_volume_weeks=0,
                window_start_date=pd.Timestamp("2024-05-01"),
                window_end_date=snap,
                window_shortfall=0,
                **BUILD,
            )
        ]
    )
    panel_path, weekly_path, liquid_path, report_path = _write_artifacts(
        tmp_path, panel, weekly, ["AAA"]
    )
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--discover-samples",
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `FAIL`" in text
    assert "sample_discovery" in text or "zero mapped" in text.lower()


# ---------------------------------------------------------------------------
# C19 — zero checked tickers cannot PASS
# ---------------------------------------------------------------------------


def test_c19_zero_checked_tickers_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)

    def _empty(*_a, **_k):
        return [], None

    monkeypatch.setattr(cli, "select_checked_tickers_for_sample", _empty)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "overall status: `PASS`" not in text
    assert "rolling_ticker_selection" in text or "overall status: `FAIL`" in text


# ---------------------------------------------------------------------------
# C21 / C22 — output parent + FAIL still writes
# ---------------------------------------------------------------------------


def test_c21_output_parent_is_created(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = _pass_fixture(tmp_path)
    nested = tmp_path / "a" / "b" / "c" / "report.md"
    assert not nested.parent.exists()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            nested,
            "--sample-date",
            sample,
        )
    )
    assert code == 0
    assert nested.is_file()


def test_c22_completed_fail_audit_still_writes_report(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, _ = _pass_fixture(
        tmp_path
    )
    bad = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    bad.to_parquet(panel_path, index=False)
    assert not report_path.exists()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 1
    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------------
# Extra guards used by discovery/C1
# ---------------------------------------------------------------------------


def test_discovery_target_equals_resolved_in_report(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), _, _, _ = _pass_fixture(tmp_path)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--discover-samples",
        )
    )
    assert code in (0, 1)
    text = report_path.read_text(encoding="utf-8")
    # Extract sample blocks and verify target == resolved when target present.
    blocks = text.split("### Sample ")
    checked = 0
    for block in blocks[1:]:
        lines = block.splitlines()
        target = next((ln for ln in lines if ln.startswith("- target_snapshot_date:")), None)
        resolved = next((ln for ln in lines if ln.startswith("- resolved_snapshot_date:")), None)
        if target is None or resolved is None:
            continue
        tval = target.split("`")[1]
        rval = resolved.split("`")[1]
        if tval != "None":
            assert tval == rval
            checked += 1
    assert checked >= 1


# ---------------------------------------------------------------------------
# C23–C44 — C7.3A defensive fixes
# ---------------------------------------------------------------------------


def _artifact_paths_and_sample(tmp_path: Path):
    paths, sample, panel, weekly = _pass_fixture(tmp_path)
    return paths, sample, panel, weekly


def test_c23_output_report_equal_panel_path_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, _), sample, panel, _ = _artifact_paths_and_sample(
        tmp_path
    )
    before = panel_path.read_bytes()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            panel_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2
    assert panel_path.read_bytes() == before


def test_c24_output_report_equal_weekly_path_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, _), sample, _, weekly = _artifact_paths_and_sample(
        tmp_path
    )
    before = weekly_path.read_bytes()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            weekly_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2
    assert weekly_path.read_bytes() == before


def test_c25_output_report_equal_ticker_csv_returns_2(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, _), sample, _, _ = _artifact_paths_and_sample(
        tmp_path
    )
    before = liquid_path.read_bytes()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            liquid_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 2
    assert liquid_path.read_bytes() == before


def test_c26_symlink_alias_output_collision_returns_2(tmp_path: Path):
    if os.name == "nt":
        pytest.importorskip("ntpath")
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = (
        _artifact_paths_and_sample(tmp_path)
    )
    alias = tmp_path / "panel_alias.md"
    try:
        os.symlink(panel_path, alias)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not supported on this platform")
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            alias,
            "--sample-date",
            sample,
        )
    )
    assert code == 2


def test_c27_write_probe_does_not_overwrite_existing_file(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = (
        _artifact_paths_and_sample(tmp_path)
    )
    parent = report_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    sentinel = parent / ".audit_pit_universe_probe_sentinel"
    sentinel.write_bytes(b"KEEP")
    before = sentinel.read_bytes()
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
        )
    )
    assert code == 0
    assert sentinel.read_bytes() == before


def _fail_panel_fixture(tmp_path: Path, mutate) -> tuple[Path, Path, Path, Path, str]:
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, weekly = (
        _pass_fixture(tmp_path)
    )
    bad = panel.copy()
    mutate(bad)
    bad.to_parquet(panel_path, index=False)
    return panel_path, weekly_path, liquid_path, report_path, sample


def _stringify_date_column(frame: pd.DataFrame, column: str) -> None:
    frame[column] = frame[column].apply(
        lambda x: x.date().isoformat() if pd.notna(x) else None
    )


def test_c28_malformed_window_start_date_fail_report(tmp_path: Path):
    def _mut(p):
        _stringify_date_column(p, "window_start_date")
        p.loc[0, "window_start_date"] = "not-a-date"

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "panel_provenance_dates" in text
    assert "overall status: `FAIL`" in text


def test_c29_malformed_window_end_date_fail_report(tmp_path: Path):
    def _mut(p):
        _stringify_date_column(p, "window_end_date")
        p.loc[0, "window_end_date"] = "bad-date"

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_dates" in report_path.read_text(encoding="utf-8")


def test_c30_nonnumeric_window_shortfall_fail_report(tmp_path: Path):
    def _mut(p):
        p["window_shortfall"] = p["window_shortfall"].astype(str)
        p.loc[0, "window_shortfall"] = "nan"

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_integers" in report_path.read_text(encoding="utf-8")


def test_c31_fractional_valid_quote_weeks_fail_report(tmp_path: Path):
    def _mut(p):
        p["valid_quote_weeks"] = p["valid_quote_weeks"].astype(float)
        p.loc[0, "valid_quote_weeks"] = 1.5

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_integers" in report_path.read_text(encoding="utf-8")


def test_c32_invalid_lookback_weeks_zero_fail(tmp_path: Path):
    def _mut(p):
        p["lookback_weeks"] = 0

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_integers" in report_path.read_text(encoding="utf-8")


def test_c33_min_valid_quote_weeks_gt_lookback_fail(tmp_path: Path):
    def _mut(p):
        p["lookback_weeks"] = 2
        p["min_valid_quote_weeks"] = 3

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_integers" in report_path.read_text(encoding="utf-8")


def test_c34_dte_min_gt_dte_max_fail(tmp_path: Path):
    def _mut(p):
        p["dte_min"] = 60
        p["dte_max"] = 5

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_integers" in report_path.read_text(encoding="utf-8")


def test_c35_non_boolean_has_valid_atm_pair_fail(tmp_path: Path):
    def _mut(p):
        p["has_valid_atm_pair"] = p["has_valid_atm_pair"].astype(str)
        p.loc[0, "has_valid_atm_pair"] = "1"

    panel_path, weekly_path, liquid_path, report_path, sample = _fail_panel_fixture(
        tmp_path, _mut
    )
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    assert "panel_provenance_boolean" in report_path.read_text(encoding="utf-8")


def test_c36_runtime_artifact_exception_converted(tmp_path: Path, monkeypatch):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = (
        _pass_fixture(tmp_path)
    )

    def _boom(**_kwargs):
        raise ValueError("synthetic artifact validation failure")

    monkeypatch.setattr(cli, "evaluate_pit_sample", _boom)
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "audit_runtime_validation" in text
    assert "overall status: `FAIL`" in text


def test_c37_unexpected_internal_exception_exit_2(tmp_path: Path, monkeypatch):
    (panel_path, weekly_path, liquid_path, report_path), sample, _, _ = (
        _pass_fixture(tmp_path)
    )

    def _boom(**_kwargs):
        raise RuntimeError("unexpected internal")

    monkeypatch.setattr(cli, "run_audit", _boom)
    code = cli.main(
        _base_argv(
            panel_path, weekly_path, liquid_path, report_path, "--sample-date", sample
        )
    )
    assert code == 2
    assert not report_path.exists()


def _clean_two_snap_panel(tmp_path: Path):
    weeks = [WEEKS[0], WEEKS[1], WEEKS[2]]
    weekly = pd.DataFrame(
        [
            _wrow(weeks[0], "AAA", 300.0, 0.02, True),
            _wrow(weeks[1], "AAA", 300.0, 0.02, True),
            _wrow(weeks[2], "AAA", 300.0, 0.02, True),
            _wrow(weeks[0], "BBB", 200.0, 0.02, True),
            _wrow(weeks[1], "BBB", 200.0, 0.02, True),
            _wrow(weeks[2], "BBB", 200.0, 0.02, True),
        ]
    )
    # Use later snapshots so rolling windows are mature (avoid short-history P1).
    panel = _panel_from_weekly(weekly, [weeks[1], weeks[2]], ["AAA", "BBB"])
    return panel, weekly, weeks


def test_c38_missing_new_not_first_snapshot_baseline_only(tmp_path: Path):
    panel, weekly, weeks = _clean_two_snap_panel(tmp_path)
    snap, notes = cli._discover_missing_or_new_snapshot(panel, [weeks[1], weeks[2]])
    assert snap == weeks[1]
    assert any("baseline_initial_population" in n for n in notes)


def test_c39_later_first_seen_ticker_selected(tmp_path: Path):
    weeks = [WEEKS[0], WEEKS[1], WEEKS[2]]
    weekly = pd.DataFrame(
        [
            _wrow(weeks[0], "AAA", 300.0, 0.02, True),
            _wrow(weeks[1], "AAA", 300.0, 0.02, True),
            _wrow(weeks[2], "AAA", 300.0, 0.02, True),
            _wrow(weeks[0], "BBB", 200.0, 0.02, True),
            _wrow(weeks[1], "BBB", 200.0, 0.02, True),
            _wrow(weeks[2], "BBB", 200.0, 0.02, True),
            _wrow(weeks[1], "NEW2", 150.0, 0.02, True),
            _wrow(weeks[2], "NEW2", 150.0, 0.02, True),
        ]
    )
    panel = _panel_from_weekly(weekly, [weeks[1], weeks[2]], ["AAA", "BBB"])
    rec = recompute_rolling_snapshot(weeks[2], ["NEW2"], weekly, LOOKBACK, MIN_VQW)
    r = rec.loc["NEW2"]
    extra = pd.DataFrame(
        [
            dict(
                month_date=weeks[2],
                ticker="NEW2",
                atm_straddle_dollar_vol=float(r["atm_straddle_dollar_vol"]),
                atm_spread_pct=float(r["atm_spread_pct"]),
                has_valid_atm_pair=bool(r["has_valid_atm_pair"]),
                valid_quote_weeks=int(r["valid_quote_weeks"]),
                zero_volume_weeks=int(r["zero_volume_weeks"]),
                window_start_date=r["window_start_date"],
                window_end_date=r["window_end_date"],
                window_shortfall=int(r["window_shortfall"]),
                **BUILD,
            )
        ]
    )
    panel = pd.concat([panel, extra], ignore_index=True)
    snap, notes = cli._discover_missing_or_new_snapshot(panel, [weeks[1], weeks[2]])
    assert snap == weeks[2]
    assert not any("baseline_initial_population" in n for n in notes)


def test_c40_invalid_liquidity_priority_over_new_ticker(tmp_path: Path):
    weeks = [WEEKS[0], WEEKS[1], WEEKS[2]]
    weekly = pd.DataFrame(
        [
            _wrow(weeks[0], "AAA", 300.0, 0.02, True),
            _wrow(weeks[1], "AAA", 300.0, 0.02, True),
            _wrow(weeks[2], "AAA", 300.0, 0.02, True),
            _wrow(weeks[0], "BAD1", 50.0, 0.02, False),
            _wrow(weeks[1], "BAD1", 50.0, 0.02, False),
            _wrow(weeks[2], "BAD1", 50.0, 0.02, False),
            _wrow(weeks[2], "NEW3", 80.0, 0.02, True),
        ]
    )
    panel = _panel_from_weekly(weekly, [weeks[0], weeks[1], weeks[2]], ["AAA", "BAD1"])
    snap, _ = cli._discover_missing_or_new_snapshot(panel, [weeks[0], weeks[1], weeks[2]])
    assert snap == weeks[0]


def test_c41_baseline_fallback_note_deterministic(tmp_path: Path):
    panel, weekly, weeks = _clean_two_snap_panel(tmp_path)
    a, na = cli._discover_missing_or_new_snapshot(panel, [weeks[1], weeks[2]])
    b, nb = cli._discover_missing_or_new_snapshot(panel, [weeks[1], weeks[2]])
    assert a == b
    assert na == nb
    assert na == [
        "missing_or_new_liquidity used baseline_initial_population fallback"
    ]


def test_c42_artifact_fail_details_in_markdown_capped(tmp_path: Path):
    (panel_path, weekly_path, liquid_path, report_path), sample, panel, _ = (
        _pass_fixture(tmp_path)
    )
    bad = panel.copy()
    bad["has_valid_atm_pair"] = bad["has_valid_atm_pair"].astype(str)
    bad.loc[0, "has_valid_atm_pair"] = "True"
    bad.loc[1, "has_valid_atm_pair"] = "yes"
    bad.to_parquet(panel_path, index=False)
    code = cli.main(
        _base_argv(
            panel_path,
            weekly_path,
            liquid_path,
            report_path,
            "--sample-date",
            sample,
            "--max-examples",
            "5",
        )
    )
    assert code == 1
    text = report_path.read_text(encoding="utf-8")
    assert "panel_provenance_boolean" in text
    assert "invalid_count" in text
    assert "examples" in text


def test_c43_revised_c14_target_resolved(tmp_path: Path):
    test_c14_discovery_maps_target_to_later_t(tmp_path)


def test_c44_revised_c17_exact_warn(tmp_path: Path):
    test_c17_fewer_than_three_discoverable_cases_warn(tmp_path)
