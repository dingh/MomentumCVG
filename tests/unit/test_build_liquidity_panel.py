"""Unit tests for scripts/build_liquidity_panel.py (Sprint 004 C4)."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = PROJECT_ROOT / "scripts" / "build_liquidity_panel.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_liquidity_panel", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_liquidity_panel"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


blp = _load_module()


def _orats_row(
    ticker: str,
    trade_date: date,
    expiry: date,
    strike: float,
    stk_px: float,
    *,
    c_bid: float = 1.0,
    c_ask: float = 1.1,
    p_bid: float = 1.0,
    p_ask: float = 1.1,
    c_vol: float = 10.0,
    p_vol: float = 10.0,
) -> dict:
    return {
        "ticker": ticker,
        "expirDate": expiry,
        "strike": strike,
        "stkPx": stk_px,
        "cBidPx": c_bid,
        "cAskPx": c_ask,
        "pBidPx": p_bid,
        "pAskPx": p_ask,
        "cVolu": c_vol,
        "pVolu": p_vol,
        "adj_strike": strike * 10,
        "adj_stkPx": stk_px * 10,
        "adj_cBidPx": c_bid * 10,
        "adj_cAskPx": c_ask * 10,
        "adj_pBidPx": p_bid * 10,
        "adj_pAskPx": p_ask * 10,
    }


class TestDailyLiquidity:
    def test_uses_raw_not_adjusted_for_atm_and_bid_dollar(self):
        trade = date(2024, 1, 5)
        expiry = date(2024, 1, 12)
        rows = [
            _orats_row(
                "AAA", trade, expiry, strike=100.0, stk_px=100.0,
                c_bid=2.0, c_ask=2.1, p_bid=2.0, p_ask=2.1, c_vol=5, p_vol=5,
            ),
            _orats_row(
                "AAA", trade, expiry, strike=1000.0, stk_px=100.0,
                c_bid=20.0, c_ask=20.1, p_bid=20.0, p_ask=20.1, c_vol=5, p_vol=5,
            ),
        ]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        row = obs.loc[obs["ticker"] == "AAA"].iloc[0]
        assert row["daily_atm_straddle_dollar_vol"] == pytest.approx(1000.0)

    def test_atm_one_row_per_expiry_not_all_strikes(self):
        trade = date(2024, 1, 5)
        expiry = date(2024, 1, 12)
        rows = [
            _orats_row("AAA", trade, expiry, strike=100.0, stk_px=100.0, c_bid=1.0, p_bid=1.0, c_vol=10, p_vol=10),
            _orats_row("AAA", trade, expiry, strike=110.0, stk_px=100.0, c_bid=99.0, p_bid=99.0, c_vol=1, p_vol=1),
        ]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert obs.iloc[0]["daily_atm_straddle_dollar_vol"] == pytest.approx(1000.0)

    def test_per_expiry_bid_dollar_min_legs(self):
        atm = pd.Series(
            {"cBidPx": 2.0, "pBidPx": 1.0, "cAskPx": 2.2, "pAskPx": 1.2, "cVolu": 10, "pVolu": 20}
        )
        vol, _ = blp.compute_expiry_atm_liquidity(atm)
        assert vol == pytest.approx(min(100 * 2 * 10, 100 * 1 * 20))

    def test_expiry_spread_nan_unless_both_legs_valid(self):
        atm_ok = pd.Series(
            {"cBidPx": 2.0, "pBidPx": 1.0, "cAskPx": 2.2, "pAskPx": 1.2, "cVolu": 10, "pVolu": 10}
        )
        _, spread_ok = blp.compute_expiry_atm_liquidity(atm_ok)
        assert np.isfinite(spread_ok)

        atm_bad_call = pd.Series(
            {"cBidPx": 2.0, "pBidPx": 1.0, "cAskPx": 1.0, "pAskPx": 1.2, "cVolu": 10, "pVolu": 10}
        )
        vol, spread = blp.compute_expiry_atm_liquidity(atm_bad_call)
        assert vol == 0.0
        assert np.isnan(spread)

    def test_crossed_quote_does_not_inflate_daily_vol_with_valid_expiry(self):
        trade = date(2024, 1, 5)
        exp_good = date(2024, 1, 12)
        exp_bad = date(2024, 2, 9)
        rows = [
            _orats_row(
                "AAA", trade, exp_good, 100, 100,
                c_bid=1.0, c_ask=1.1, p_bid=1.0, p_ask=1.1, c_vol=10, p_vol=10,
            ),
            _orats_row(
                "AAA", trade, exp_bad, 100, 100,
                c_bid=99.0, c_ask=1.0, p_bid=99.0, p_ask=1.0, c_vol=100, p_vol=100,
            ),
        ]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert obs.iloc[0]["daily_atm_straddle_dollar_vol"] == pytest.approx(1000.0)
        assert np.isfinite(obs.iloc[0]["daily_atm_spread_pct"])
        assert obs.iloc[0]["daily_has_valid_quote"]

    def test_daily_sums_multiple_expiries_in_band(self):
        trade = date(2024, 1, 5)
        exp1 = date(2024, 1, 12)
        exp2 = date(2024, 2, 9)
        exp_far = date(2024, 4, 5)
        rows = [
            _orats_row("AAA", trade, exp1, 100, 100, c_bid=1, p_bid=1, c_vol=10, p_vol=10),
            _orats_row("AAA", trade, exp2, 100, 100, c_bid=1, p_bid=1, c_vol=10, p_vol=10),
            _orats_row("AAA", trade, exp_far, 100, 100, c_bid=99, p_bid=99, c_vol=99, p_vol=99),
        ]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert obs.iloc[0]["daily_atm_straddle_dollar_vol"] == pytest.approx(2000.0)

    def test_zero_bid_dollar_gives_nan_spread(self):
        trade = date(2024, 1, 5)
        expiry = date(2024, 6, 1)
        rows = [_orats_row("AAA", trade, expiry, 100, 100)]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert obs.iloc[0]["daily_atm_straddle_dollar_vol"] == 0.0
        assert np.isnan(obs.iloc[0]["daily_atm_spread_pct"])

    def test_inverted_ask_bid_gives_nan_spread(self):
        trade = date(2024, 1, 5)
        expiry = date(2024, 1, 12)
        rows = [_orats_row("AAA", trade, expiry, 100, 100, c_bid=2.0, c_ask=1.0, p_bid=2.0, p_ask=1.0)]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert obs.iloc[0]["daily_atm_straddle_dollar_vol"] == 0.0
        assert np.isnan(obs.iloc[0]["daily_atm_spread_pct"])
        assert not obs.iloc[0]["daily_has_valid_quote"]

    def test_no_expiry_in_band_emits_debug_row(self):
        trade = date(2024, 1, 5)
        expiry = date(2024, 6, 1)
        rows = [_orats_row("AAA", trade, expiry, 100, 100)]
        obs = blp.compute_daily_liquidity_observations(pd.DataFrame(rows), trade)
        assert bool(obs.iloc[0]["no_expiry_in_band"]) is True


class TestWeeklyAndPanel:
    def test_weekly_mean_of_daily_with_missing_day_zero(self):
        daily = pd.DataFrame(
            [
                {
                    "trade_date": date(2024, 1, 3),
                    "ticker": "AAA",
                    "daily_atm_straddle_dollar_vol": 1000.0,
                    "daily_atm_spread_pct": 0.05,
                    "daily_has_valid_quote": True,
                    "n_candidate_expiries": 1,
                    "n_expiries_total": 1,
                    "no_expiry_in_band": False,
                    "liquidity_source": blp.LIQUIDITY_SOURCE,
                },
                {
                    "trade_date": date(2024, 1, 4),
                    "ticker": "AAA",
                    "daily_atm_straddle_dollar_vol": 3000.0,
                    "daily_atm_spread_pct": 0.07,
                    "daily_has_valid_quote": True,
                    "n_candidate_expiries": 1,
                    "n_expiries_total": 1,
                    "no_expiry_in_band": False,
                    "liquidity_source": blp.LIQUIDITY_SOURCE,
                },
            ]
        )
        week_cal = [(date(2024, 1, 5), [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)])]
        weekly = blp.aggregate_weekly_liquidity_observations(daily, week_cal)
        assert weekly.iloc[0]["weekly_atm_straddle_dollar_vol"] == pytest.approx(4000 / 3)

    def test_rolling_panel_fixed_denominator(self):
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": date(2024, 1, 5),
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 100.0,
                    "weekly_atm_spread_pct": 0.1,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
                {
                    "week_end_date": date(2024, 1, 12),
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 200.0,
                    "weekly_atm_spread_pct": 0.2,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
            ]
        )
        all_weeks = [date(2024, 1, 5), date(2024, 1, 12)]
        panel = blp.aggregate_rolling_weekly_panel(
            weekly, [date(2024, 1, 12)], all_weeks, lookback_weeks=12, min_valid_quote_weeks=1
        )
        assert panel.iloc[0]["atm_straddle_dollar_vol"] == pytest.approx(300 / 12)

    def test_zero_volume_weeks_counts_missing_ticker_week_rows(self):
        weeks = [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19), date(2024, 1, 26)]
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": weeks[-1],
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 400.0,
                    "weekly_atm_spread_pct": 0.1,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
            ]
        )
        panel = blp.aggregate_rolling_weekly_panel(
            weekly,
            [weeks[-1]],
            weeks,
            lookback_weeks=4,
            min_valid_quote_weeks=1,
        )
        row = panel.iloc[0]
        assert row["atm_straddle_dollar_vol"] == pytest.approx(100.0)
        assert row["zero_volume_weeks"] == 3

    def test_valid_quote_weeks_gates_has_valid_atm_pair(self):
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": date(2024, 1, 5),
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 1.0,
                    "weekly_atm_spread_pct": 0.1,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
            ]
        )
        panel = blp.aggregate_rolling_weekly_panel(
            weekly,
            [date(2024, 1, 5)],
            [date(2024, 1, 5)],
            lookback_weeks=12,
            min_valid_quote_weeks=3,
        )
        assert bool(panel.iloc[0]["has_valid_atm_pair"]) is False

    def test_panel_preserves_step1_columns(self):
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": date(2024, 1, 5),
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 1.0,
                    "weekly_atm_spread_pct": 0.1,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
            ]
        )
        panel = blp.aggregate_rolling_weekly_panel(
            weekly,
            [date(2024, 1, 5)],
            [date(2024, 1, 5)],
            lookback_weeks=12,
            min_valid_quote_weeks=1,
        )
        for col in blp.PANEL_STEP1_COLS:
            assert col in panel.columns


class TestBackfillPipeline:
    def test_backfill_end_to_end_synthetic(self, tmp_path: Path):
        trade1 = date(2024, 1, 3)
        trade2 = date(2024, 1, 4)
        trade3 = date(2024, 1, 5)
        expiry = date(2024, 1, 12)
        store = {
            td: pd.DataFrame([
                _orats_row("AAA", td, expiry, 100, 100, c_bid=1, p_bid=1, c_vol=10, p_vol=10),
            ])
            for td in (trade1, trade2, trade3)
        }
        load_calls: list[date] = []

        def load_fn(d: date) -> pd.DataFrame:
            load_calls.append(d)
            return store[d]

        result = blp.run_backfill(
            tmp_path,
            date(2024, 1, 5),
            date(2024, 1, 5),
            load_fn,
            sorted(store.keys()),
            lookback_weeks=12,
            min_valid_quote_weeks=1,
        )
        assert not result.panel.empty
        assert len(set(load_calls)) == len(load_calls)

    def test_empty_source_fails(self, tmp_path: Path):
        with pytest.raises(blp.LiquidityPanelError, match="No ORATS raw ZIP files"):
            blp.run_backfill(
                tmp_path,
                date(2024, 1, 1),
                date(2024, 1, 31),
                lambda d: pd.DataFrame(),
                [],
            )


class TestIncremental:
    _DVOL_TOP = 0.20
    _SPREAD_BOT = 0.20

    def _validate_kwargs(self, **overrides):
        base = {
            "lookback_weeks": 12,
            "min_valid_quote_weeks": 1,
            "dte_min": 5,
            "dte_max": 60,
            "dvol_top_pct": self._DVOL_TOP,
            "spread_bot_pct": self._SPREAD_BOT,
        }
        base.update(overrides)
        return base

    def _make_prior_artifacts(self, cache: Path) -> None:
        daily = pd.DataFrame(
            [
                {
                    "trade_date": date(2024, 1, 3),
                    "ticker": "AAA",
                    "daily_atm_straddle_dollar_vol": 1000.0,
                    "daily_atm_spread_pct": 0.05,
                    "daily_has_valid_quote": True,
                    "n_candidate_expiries": 1,
                    "n_expiries_total": 1,
                    "no_expiry_in_band": False,
                    "liquidity_source": blp.LIQUIDITY_SOURCE,
                },
            ]
        )
        weekly = pd.DataFrame(
            [
                {
                    "week_end_date": date(2024, 1, 5),
                    "ticker": "AAA",
                    "weekly_atm_straddle_dollar_vol": 1000.0,
                    "weekly_atm_spread_pct": 0.05,
                    "weekly_valid_quote_days": 1,
                    "weekly_has_valid_quote": True,
                },
            ]
        )
        panel = blp.aggregate_rolling_weekly_panel(
            weekly,
            [date(2024, 1, 5)],
            [date(2024, 1, 5)],
            lookback_weeks=12,
            min_valid_quote_weeks=1,
            dte_min=5,
            dte_max=60,
        )
        panel = blp.stamp_panel_universe_params(
            panel,
            dvol_top_pct=self._DVOL_TOP,
            spread_bot_pct=self._SPREAD_BOT,
        )
        daily.to_parquet(cache / blp.DAILY_FILENAME, index=False)
        weekly.to_parquet(cache / blp.WEEKLY_FILENAME, index=False)
        panel.to_parquet(cache / blp.PANEL_FILENAME, index=False)

    def test_validate_passes_when_build_params_match(self, tmp_path: Path):
        self._make_prior_artifacts(tmp_path)
        state = blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs())
        assert state.dte_min == 5
        assert state.dte_max == 60

    def test_missing_artifacts_fail(self, tmp_path: Path):
        with pytest.raises(blp.LiquidityPanelError, match="Missing required artifact"):
            blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs())

    def test_schema_mismatch_fail(self, tmp_path: Path):
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / blp.DAILY_FILENAME, index=False)
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / blp.WEEKLY_FILENAME, index=False)
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / blp.PANEL_FILENAME, index=False)
        with pytest.raises(blp.LiquidityPanelError, match="schema mismatch"):
            blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs())

    def test_build_params_mismatch_fail(self, tmp_path: Path):
        self._make_prior_artifacts(tmp_path)
        with pytest.raises(blp.LiquidityPanelError, match="Build params mismatch"):
            blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs(dte_min=10))

    def test_universe_params_mismatch_fail(self, tmp_path: Path):
        self._make_prior_artifacts(tmp_path)
        with pytest.raises(blp.LiquidityPanelError, match="dvol_top_pct"):
            blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs(dvol_top_pct=0.50))

    def test_missing_build_param_columns_fail(self, tmp_path: Path):
        self._make_prior_artifacts(tmp_path)
        panel = pd.read_parquet(tmp_path / blp.PANEL_FILENAME)
        panel = panel.drop(columns=["dte_min", "dte_max"])
        panel.to_parquet(tmp_path / blp.PANEL_FILENAME, index=False)
        with pytest.raises(blp.LiquidityPanelError, match="missing build-param columns"):
            blp.validate_incremental_artifacts(tmp_path, **self._validate_kwargs())


class TestWriteArtifacts:
    def test_write_artifacts_commits_all_outputs(self, tmp_path: Path):
        daily = pd.DataFrame([{"trade_date": date(2024, 1, 3), "ticker": "AAA"}])
        weekly = pd.DataFrame([{"week_end_date": date(2024, 1, 5), "ticker": "AAA"}])
        panel = pd.DataFrame([{"month_date": pd.Timestamp("2024-01-05"), "ticker": "AAA"}])
        result = blp.BuildResult(daily=daily, weekly=weekly, panel=panel, files_read=1)
        liquid = pd.DataFrame(
            [{"Ticker": "AAA", "snapshots_qualified": 1, "months_qualified": 1}]
        )

        blp.write_artifacts(tmp_path, result, liquid_tickers=liquid)

        assert (tmp_path / blp.DAILY_FILENAME).is_file()
        assert (tmp_path / blp.WEEKLY_FILENAME).is_file()
        assert (tmp_path / blp.PANEL_FILENAME).is_file()
        assert (tmp_path / blp.LIQUID_TICKERS_FILENAME).is_file()
        assert not (tmp_path / blp.STAGING_DIRNAME).exists()


class TestLiquidityPanelReport:
    def test_report_inputs_include_build_params(self, tmp_path: Path):
        daily = pd.DataFrame(
            [
                {
                    "trade_date": date(2024, 1, 3),
                    "ticker": "AAA",
                    "no_expiry_in_band": False,
                }
            ]
        )
        result = blp.BuildResult(
            daily=daily,
            weekly=pd.DataFrame(),
            panel=pd.DataFrame(),
            files_read=0,
        )
        report_path = tmp_path / "report.md"
        blp.write_liquidity_panel_report(
            report_path,
            build_id="test-build",
            mode="backfill",
            data_root=Path("C:/ORATS/data/ORATS_Data"),
            cache_dir=tmp_path,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            lookback_weeks=12,
            min_valid_quote_weeks=3,
            dte_min=5,
            dte_max=60,
            dvol_top_pct=0.20,
            spread_bot_pct=0.20,
            result=result,
        )
        text = report_path.read_text(encoding="utf-8")
        assert "dte_min: 5" in text
        assert "dte_max: 60" in text
        assert "dvol_top_pct: 0.2" in text
        assert "spread_bot_pct: 0.2" in text
        assert f"liquidity_source: `{blp.LIQUIDITY_SOURCE}`" in text


class TestBuildLiquidTickers:
    def test_build_liquid_tickers_runs(self):
        panel = pd.DataFrame(
            [
                {
                    "month_date": pd.Timestamp("2024-01-05"),
                    "ticker": "AAA",
                    "atm_straddle_dollar_vol": 1e6,
                    "atm_spread_pct": 0.01,
                    "has_valid_atm_pair": True,
                },
                {
                    "month_date": pd.Timestamp("2024-01-05"),
                    "ticker": "BBB",
                    "atm_straddle_dollar_vol": 1e3,
                    "atm_spread_pct": 0.5,
                    "has_valid_atm_pair": True,
                },
            ]
        )
        out = blp.build_liquid_tickers(panel, 0.5, 0.5)
        assert list(out.columns) == list(blp.LIQUID_TICKERS_COLUMNS)
        assert (out["months_qualified"] == out["snapshots_qualified"]).all()


class TestRawZipIO:
    def test_orats_raw_zip_path(self):
        d = date(2024, 3, 15)
        p = blp.orats_raw_zip_path(Path("C:/ORATS/data/ORATS_Data"), d)
        assert p == Path("C:/ORATS/data/ORATS_Data/2024/ORATS_SMV_Strikes_20240315.zip")

    def test_discover_orats_trading_dates_from_zips(self, tmp_path: Path):
        for d in (date(2024, 1, 3), date(2024, 1, 5)):
            zdir = tmp_path / f"{d.year:04d}"
            zdir.mkdir(parents=True, exist_ok=True)
            (zdir / f"ORATS_SMV_Strikes_{d.strftime('%Y%m%d')}.zip").touch()
        (tmp_path / "2024" / "ORATS_SMV_Strikes_20240104.parquet").touch()

        found = blp.discover_orats_trading_dates(
            tmp_path, date(2024, 1, 1), date(2024, 1, 31)
        )
        assert found == [date(2024, 1, 3), date(2024, 1, 5)]

    def test_load_raw_day_from_zip(self, tmp_path: Path):
        trade = date(2024, 1, 5)
        csv = (
            "ticker,expirDate,strike,stkPx,cBidPx,cAskPx,pBidPx,pAskPx,cVolu,pVolu\n"
            "AAA,2024-01-12,100,100,1,1.1,1,1.1,10,10\n"
        )
        zpath = tmp_path / "2024" / f"ORATS_SMV_Strikes_{trade.strftime('%Y%m%d')}.zip"
        zpath.parent.mkdir(parents=True)
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("strikes.csv", csv)

        df = blp.load_raw_day_from_zip(tmp_path, trade)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAA"

    def test_load_raw_day_missing_zip_returns_empty(self, tmp_path: Path):
        df = blp.load_raw_day_from_zip(tmp_path, date(2024, 1, 5))
        assert df.empty
