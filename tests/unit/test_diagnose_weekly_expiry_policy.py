"""Unit tests for C6.1B weekly expiry policy diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.orats_provider import OptionQuote

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "diagnose_weekly_expiry_policy.py"


@pytest.fixture
def diag_module():
    spec = importlib.util.spec_from_file_location("diagnose_weekly_expiry_policy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quote(
    *,
    strike: str = "100",
    option_type: str = "call",
    bid: str = "1.0",
    ask: str = "1.2",
) -> OptionQuote:
    bid_dec = Decimal(bid)
    ask_dec = Decimal(ask)
    return OptionQuote(
        ticker="AAPL",
        trade_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        strike=Decimal(strike),
        option_type=option_type,
        bid=bid_dec,
        ask=ask_dec,
        mid=(bid_dec + ask_dec) / 2,
        iv=0.2,
        delta=0.5,
        gamma=0.01,
        vega=0.1,
        theta=-0.05,
        volume=100,
        open_interest=1000,
    )


def _obs(
    diag_module,
    *,
    ticker: str = "AAPL",
    listed: bool = True,
    body_pair: bool = True,
    match: bool = True,
) -> object:
    target = date(2024, 1, 12)
    chain = target if match else date(2024, 1, 10)
    return diag_module.DiagnosedObservation(
        ticker=ticker,
        entry_date=date(2024, 1, 5),
        expiry_chain_scanned=chain,
        expiry_target_weekly=target,
        target_listed_on_chain=listed,
        target_body_call_quotable=body_pair,
        target_body_put_quotable=body_pair,
        target_body_pair_quotable=body_pair,
        dte_chain=(chain - date(2024, 1, 5)).days,
        dte_target=7,
        dte_delta=(chain - target).days,
        expiries_match=match,
    )


def _sample_result(diag_module, diagnosed, skipped=None, tickers=None):
    tickers = tickers or ["AAPL"]
    skipped = skipped or []
    entry_dates = [date(2024, 1, 5)]
    attempted = len(tickers) * len(entry_dates)
    metrics = diag_module.aggregate_metrics(diagnosed, skipped, attempted)
    return diag_module.SampleResult(
        label="sample_a",
        tickers=tickers,
        entry_dates=entry_dates,
        diagnosed=list(diagnosed),
        skipped=list(skipped),
        metrics=metrics,
        sampling_method="test",
    )


class TestHelperFunctions:
    def test_compute_dte_delta(self, diag_module) -> None:
        entry = date(2024, 1, 5)
        chain_exp = date(2024, 1, 12)
        target_exp = date(2024, 1, 19)
        dte_chain, dte_target, dte_delta = diag_module.compute_dte_fields(
            entry, chain_exp, target_exp
        )
        assert dte_chain == 7
        assert dte_target == 14
        assert dte_delta == -7

    def test_is_body_quote_quotable_matches_producer(self, diag_module) -> None:
        assert diag_module.is_body_quote_quotable(_quote()) is True
        assert diag_module.is_body_quote_quotable(_quote(bid="0")) is False
        assert diag_module.is_body_quote_quotable(None) is False

    def test_select_entry_dates_respects_max(self, diag_module) -> None:
        schedule = [date(2024, 1, d) for d in (5, 12, 19, 26)]
        selected = diag_module.select_entry_dates(
            schedule, date(2024, 1, 1), date(2024, 1, 31), max_entry_dates=2
        )
        assert selected == [date(2024, 1, 5), date(2024, 1, 12)]

    def test_sample_coverage_tickers_stratified(self, diag_module) -> None:
        ranked = [f"T{i:03d}" for i in range(90)]
        selected = diag_module.sample_coverage_tickers(ranked, max_tickers=9)
        assert len(selected) == 9
        assert selected[0] == "T000"
        assert "T089" in selected

    def test_classify_verdict_pass_with_policy(self, diag_module) -> None:
        metrics = {
            "attempted": 10,
            "diagnosed_count": 10,
            "skipped_count": 0,
            "weekly_tradable_rate": 0.95,
            "match_rate": 0.95,
        }
        assert diag_module.classify_verdict(metrics) == diag_module.VERDICT_PASS_POLICY

    def test_classify_verdict_fail_when_no_diagnosed(self, diag_module) -> None:
        metrics = {
            "attempted": 5,
            "diagnosed_count": 0,
            "skipped_count": 5,
            "weekly_tradable_rate": None,
            "match_rate": None,
        }
        assert diag_module.classify_verdict(metrics) == "FAIL"

    def test_low_broad_coverage_does_not_block_c61c(self, diag_module) -> None:
        sample_a = {
            "attempted": 60,
            "diagnosed_count": 60,
            "skipped_count": 0,
            "weekly_tradable_rate": 1.0,
            "match_rate": 1.0,
        }
        sample_b = {
            "attempted": 600,
            "diagnosed_count": 500,
            "skipped_count": 100,
            "weekly_tradable_rate": 0.20,
            "match_rate": 0.20,
        }
        assert (
            diag_module.classify_verdict(sample_a, sample_b_metrics=sample_b)
            == diag_module.VERDICT_PASS_POLICY
        )

    def test_missing_target_is_not_weekly_tradable(self, diag_module) -> None:
        obs = _obs(diag_module, listed=False, body_pair=False, match=False)
        assert obs.weekly_tradable is False
        metrics = diag_module.aggregate_metrics([obs], [], attempted=1)
        assert metrics["missing_target_rate"] == 1.0
        assert metrics["weekly_tradable_rate"] == 0.0


class TestDiagnoseObservation:
    def test_uses_target_helper_only_for_target_expiry(self, diag_module) -> None:
        schedule = [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19)]
        builder = MagicMock()
        builder._find_best_expiry.return_value = date(2024, 1, 12)

        provider = MagicMock()
        provider.get_available_expiries.return_value = [
            date(2024, 1, 12),
            date(2024, 1, 19),
        ]
        provider.get_spot_price.return_value = Decimal("100")
        provider.get_option_chain.return_value = [
            _quote(strike="100", option_type="call"),
            _quote(strike="100", option_type="put"),
        ]

        result = diag_module.diagnose_observation(
            builder,
            provider,
            "AAPL",
            date(2024, 1, 5),
            schedule,
        )

        assert isinstance(result, diag_module.DiagnosedObservation)
        assert result.expiry_chain_scanned == date(2024, 1, 12)
        assert result.expiry_target_weekly == date(2024, 1, 12)
        builder._find_best_expiry.assert_called_once_with(
            "AAPL", date(2024, 1, 5), diag_module.WEEKLY_DTE_TARGET
        )

    def test_missing_entry_parquet_is_skip_not_crash(self, diag_module) -> None:
        schedule = [date(2024, 1, 5), date(2024, 1, 12)]
        builder = MagicMock()
        provider = MagicMock()
        provider.get_available_expiries.side_effect = FileNotFoundError("missing")

        result = diag_module.diagnose_observation(
            builder,
            provider,
            "AAPL",
            date(2024, 1, 5),
            schedule,
        )
        assert isinstance(result, diag_module.SkippedObservation)
        assert result.reason == "entry_parquet_missing"
        builder._find_best_expiry.assert_not_called()


class TestReportRendering:
    def test_report_contains_required_policy_language(self, diag_module) -> None:
        diagnosed = [_obs(diag_module)]
        sample_a = _sample_result(diag_module, diagnosed)
        sample_b = _sample_result(
            diag_module,
            [_obs(diag_module, ticker="XYZ", listed=False, body_pair=False, match=False)],
            tickers=["XYZ"],
        )
        sample_b = diag_module.SampleResult(
            label="sample_b",
            tickers=sample_b.tickers,
            entry_dates=sample_b.entry_dates,
            diagnosed=sample_b.diagnosed,
            skipped=sample_b.skipped,
            metrics=sample_b.metrics,
            sampling_method="stratified test",
        )
        verdict = diag_module.classify_verdict(
            sample_a.metrics, sample_b_metrics=sample_b.metrics
        )
        report = diag_module.render_markdown_report(
            repo_commit="abc123",
            input_root=Path("C:/MomentumCVG_env/input/adjusted_liquid"),
            sample_start=date(2024, 1, 1),
            sample_end=date(2024, 3, 31),
            sample_a=sample_a,
            sample_b=sample_b,
            verdict=verdict,
        )

        assert diag_module.NO_PRODUCER_CHANGE_STATEMENT in report
        assert diag_module.NO_FALLBACK_STATEMENT in report
        assert "No fallback to nearest-DTE expiry is allowed." in report
        assert "Sample A — known-weekly sanity check" in report
        assert "Sample B — broad-universe coverage check" in report
        assert "informational coverage, not a C6.1C correctness blocker" in report
        assert "Missing exact target weekly expiry means no weekly trade." in report
        assert "expected for non-weekly-option names" in report
        assert "must not silently substitute a nearby expiry" in report
        assert "should not define weekly strategy semantics" in report
        assert "deciding policy semantics, not proving broad weekly-option coverage" in report
        assert "opportunity count/capacity, not correctness" in report
        assert "threshold >= 90%" not in report
        assert verdict == diag_module.VERDICT_PASS_POLICY
        assert "Proceed to C6.1C with strict calendar-paired weekly expiry" in report

    def test_recommendation_does_not_require_broad_90(self, diag_module) -> None:
        sample_a = {
            "attempted": 60,
            "diagnosed_count": 60,
            "skipped_count": 0,
            "weekly_tradable_rate": 1.0,
            "match_rate": 1.0,
        }
        sample_b = {
            "attempted": 600,
            "diagnosed_count": 500,
            "skipped_count": 100,
            "weekly_tradable_rate": 0.15,
            "match_rate": 0.15,
        }
        text = diag_module.recommendation_text(
            diag_module.VERDICT_PASS_POLICY, sample_a, sample_b
        )
        assert "No fallback to nearest-DTE expiry is allowed" in text
        assert "informational coverage only" in text
        assert "does not block C6.1C" in text
        assert ">= 90%" not in text


class TestCliSafety:
    def test_defaults_are_bounded(self, diag_module) -> None:
        args = diag_module.parse_args([])
        assert args.tickers is None
        assert args.tickers_file is None
        assert args.include_coverage_sample is False
        assert args.coverage_max_tickers == diag_module.DEFAULT_COVERAGE_MAX_TICKERS
        assert diag_module.resolve_tickers(args) == list(diag_module.DEFAULT_TICKERS)
        assert args.start_date == diag_module.DEFAULT_START_DATE
        assert args.end_date == diag_module.DEFAULT_END_DATE
        assert args.max_entry_dates == diag_module.DEFAULT_MAX_ENTRY_DATES
        assert args.output_report == diag_module.DEFAULT_OUTPUT_REPORT

    def test_main_writes_markdown_only(
        self, diag_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_report = tmp_path / "report.md"
        fake_entry_dates = [date(2024, 1, 5)]
        fake_diagnosed = [_obs(diag_module)]

        monkeypatch.setattr(
            diag_module,
            "run_diagnostic",
            lambda **kwargs: (fake_entry_dates, fake_diagnosed, []),
        )

        input_root = tmp_path / "adjusted"
        input_root.mkdir()
        code = diag_module.main(
            [
                "--input-root",
                str(input_root),
                "--tickers",
                "AAPL",
                "--output-report",
                str(output_report),
                "--commit",
                "deadbeef",
            ]
        )

        assert code == 0
        assert output_report.exists()
        text = output_report.read_text(encoding="utf-8")
        assert diag_module.NO_PRODUCER_CHANGE_STATEMENT in text
        assert diag_module.NO_FALLBACK_STATEMENT in text
        assert list(tmp_path.glob("*.parquet")) == []

    def test_main_uses_normalized_unique_tickers_in_report(
        self, diag_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_report = tmp_path / "report.md"
        captured: dict[str, object] = {}

        def fake_run_diagnostic(**kwargs):
            captured["tickers"] = kwargs["tickers"]
            return (
                [date(2024, 1, 5)],
                [_obs(diag_module)],
                [],
            )

        monkeypatch.setattr(diag_module, "run_diagnostic", fake_run_diagnostic)
        input_root = tmp_path / "adjusted"
        input_root.mkdir()
        diag_module.main(
            [
                "--input-root",
                str(input_root),
                "--tickers",
                "aapl",
                "AAPL",
                "msft",
                "--output-report",
                str(output_report),
                "--commit",
                "deadbeef",
            ]
        )
        assert captured["tickers"] == ["AAPL", "MSFT"]

    def test_empty_normalized_tickers_exit_2(self, diag_module, tmp_path: Path) -> None:
        code = diag_module.main(
            [
                "--input-root",
                str(tmp_path / "adjusted"),
                "--tickers",
                "  ",
                "",
            ]
        )
        assert code == 2

    def test_default_tickers_file_normalizes(
        self, diag_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tickers_file = tmp_path / "tickers.csv"
        pd.DataFrame({"Ticker": ["aapl", " AAPL ", "msft", "", None]}).to_csv(
            tickers_file, index=False
        )
        captured: dict[str, object] = {}

        def fake_run_diagnostic(**kwargs):
            captured["tickers"] = kwargs["tickers"]
            return ([], [], [])

        monkeypatch.setattr(diag_module, "run_diagnostic", fake_run_diagnostic)
        input_root = tmp_path / "adjusted"
        input_root.mkdir()
        diag_module.main(
            [
                "--input-root",
                str(input_root),
                "--tickers-file",
                str(tickers_file),
                "--output-report",
                str(tmp_path / "report.md"),
                "--commit",
                "deadbeef",
            ]
        )
        assert captured["tickers"] == ["AAPL", "MSFT"]
