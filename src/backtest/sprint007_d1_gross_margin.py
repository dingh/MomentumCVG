"""
Sprint 007 D1 — gross economics of the frozen midpoint trade expression.

Read-only post-pass over the accepted Sprint 006 midpoint artifacts. Reconciles
to ``decision_report.json`` (``by_fill.mid.primary``) and evaluates the four
frozen continue/stop gate parts from ``docs/tmp/sprint007_d1_design.md``.

Cross-fill artifacts, bridge math, execution assumptions, and filters are out of
scope for D1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.backtest.sprint007_artifact_validation import (
    OFFICIAL_RUN_DIR,
    get_current_repo_sha,
    sha256_file,
)
from src.backtest.surface_decision_report import (
    PRIMARY_END,
    PRIMARY_START,
    assert_report_preconditions,
    compute_long_short_attribution,
    compute_view_a,
    count_date_classes,
    filter_to_window,
)

FILL_LABEL = "mid"
EXPECTED_INCLUDED_TRADES = 9212

PNL_ABS_TOLERANCE = 0.01
PNL_REL_TOLERANCE = 1e-9
CAR_TOLERANCE = 1e-9

BREADTH_TOP_N = 5
LOCATION_MIN_TRADE_SHARE = 0.10
MIN_POSITIVE_YEARS = 2

VERDICT_CONTINUE = "D1_CONTINUE_TO_D2"
VERDICT_STOP = "D1_STOP_CURRENT_EXPRESSION"
VERDICT_BLOCKED = "D1_BLOCKED"

EVIDENCE_DIR_ENV = "SPRINT007_D1_EVIDENCE_DIR"

TRADE_LOG_COLUMNS = (
    "trade_date",
    "ticker",
    "direction",
    "included_in_portfolio",
    "pnl_total",
    "capital_at_risk_dollars",
    "fill_label",
)
DATE_STATUS_COLUMNS = ("trade_date", "status", "reason")
DATE_SUMMARY_COLUMNS = (
    "trade_date",
    "cycle_pnl_total",
    "cycle_capital_at_risk",
    "cycle_return_on_capital_at_risk",
    "short_cycle_pnl_total",
    "short_cycle_return",
    "long_cycle_pnl_total",
    "long_cycle_return",
)


class D1AnalysisError(Exception):
    """Raised when a D1 precondition makes the gate uninterpretable."""


@dataclass
class MidPrimaryBundle:
    run_dir: Path
    date_status: pd.DataFrame
    date_summary: pd.DataFrame
    trade_log: pd.DataFrame
    included: pd.DataFrame


@dataclass
class ReconciliationRow:
    metric: str
    recomputed: Any
    reference: Any
    delta: Any
    tolerance: Any
    passed: bool


@dataclass
class ReconciliationResult:
    rows: list[ReconciliationRow] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "metric": row.metric,
                "recomputed": row.recomputed,
                "reference": row.reference,
                "delta": row.delta,
                "tolerance": row.tolerance,
                "passed": row.passed,
            }
            for row in self.rows
        ]


@dataclass
class GatePart:
    part_id: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class D1Scorecard:
    verdict: str
    parts: list[GatePart] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    side_attribution: list[dict[str, Any]] = field(default_factory=list)
    breadth_exclusions: list[dict[str, Any]] = field(default_factory=list)
    yearly_pnl: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_gates_passed(self) -> bool:
        return all(part.passed for part in self.parts)


@dataclass
class D1Result:
    verdict: str
    scorecard: D1Scorecard
    reconciliation: ReconciliationResult
    manifest: dict[str, Any] = field(default_factory=dict)


def _trade_log_path(run_dir: Path) -> Path:
    return run_dir / f"trade_log_sprint006_baseline_v1_{FILL_LABEL}.parquet"


def _date_status_path(run_dir: Path) -> Path:
    return run_dir / f"date_status_sprint006_baseline_v1_{FILL_LABEL}.parquet"


def _date_summary_path(run_dir: Path) -> Path:
    return run_dir / f"date_summary_sprint006_baseline_v1_{FILL_LABEL}.parquet"


def _read_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pq.read_table(path, columns=list(columns)).to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    return frame.sort_values("trade_date").reset_index(drop=True)


def select_included_traded_rows(
    trade_log: pd.DataFrame,
    date_status: pd.DataFrame,
) -> pd.DataFrame:
    """Included portfolio rows on traded dates — the D1 population."""
    if trade_log.empty:
        return trade_log.copy()
    traded_dates = set(
        date_status.loc[date_status["status"] == "traded", "trade_date"].tolist()
    )
    mask = (trade_log["included_in_portfolio"] == True) & (  # noqa: E712
        trade_log["trade_date"].isin(traded_dates)
    )
    included = trade_log.loc[mask].copy()
    included["pnl_total"] = pd.to_numeric(included["pnl_total"], errors="coerce")
    return included.sort_values(["trade_date", "ticker", "direction"]).reset_index(drop=True)


def load_mid_primary_tables(run_dir: Path | None = None) -> MidPrimaryBundle:
    """Load mid artifacts, filter to the primary window, and select included rows."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    date_status = _read_columns(_date_status_path(run_dir), DATE_STATUS_COLUMNS)
    date_summary = _read_columns(_date_summary_path(run_dir), DATE_SUMMARY_COLUMNS)
    trade_log = _read_columns(_trade_log_path(run_dir), TRADE_LOG_COLUMNS)

    # Non-included candidate rows carry a null fill_label; only labelled rows are checked.
    labels = {str(value) for value in trade_log["fill_label"].dropna().unique()}
    unexpected = labels - {FILL_LABEL}
    if unexpected:
        raise D1AnalysisError(f"trade log carries non-mid fill labels: {sorted(unexpected)}")

    status, summary, log = filter_to_window(
        date_status, date_summary, trade_log, PRIMARY_START, PRIMARY_END
    )
    assert_report_preconditions(status, summary, log)
    return MidPrimaryBundle(
        run_dir=run_dir,
        date_status=status,
        date_summary=summary,
        trade_log=log,
        included=select_included_traded_rows(log, status),
    )


def load_accepted_mid_primary_report(run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or OFFICIAL_RUN_DIR
    report = json.loads((run_dir / "decision_report.json").read_text(encoding="utf-8"))
    return report["by_fill"][FILL_LABEL]["primary"]


def _pnl_tolerance(reference: float) -> float:
    return max(PNL_ABS_TOLERANCE, PNL_REL_TOLERANCE * abs(float(reference)))


def reconcile_mid_primary(
    bundle: MidPrimaryBundle,
    report: dict[str, Any],
) -> ReconciliationResult:
    """Compare recomputed mid-primary aggregates to the accepted Sprint 006 report."""
    long_short = report["long_short"]
    ref_pnl = float(long_short["long"]["pnl_total"]) + float(long_short["short"]["pnl_total"])
    ref_trades = int(long_short["long"]["n_traded_rows"]) + int(long_short["short"]["n_traded_rows"])
    ref_car = float(report["view_a_conditional"]["mean_cycle_car"])
    ref_traded_dates = int(report["date_class_counts"]["n_traded_dates"])

    total_pnl = float(bundle.included["pnl_total"].sum())
    n_included = int(len(bundle.included))
    mean_car = float(compute_view_a(bundle.date_status, bundle.date_summary)["mean_cycle_car"])
    n_traded_dates = int(count_date_classes(bundle.date_status)["n_traded_dates"])

    pnl_tol = _pnl_tolerance(ref_pnl)
    rows = [
        ReconciliationRow(
            metric="total_pnl",
            recomputed=total_pnl,
            reference=ref_pnl,
            delta=total_pnl - ref_pnl,
            tolerance=pnl_tol,
            passed=abs(total_pnl - ref_pnl) <= pnl_tol,
        ),
        ReconciliationRow(
            metric="view_a_mean_cycle_car",
            recomputed=mean_car,
            reference=ref_car,
            delta=mean_car - ref_car,
            tolerance=CAR_TOLERANCE,
            passed=abs(mean_car - ref_car) <= CAR_TOLERANCE,
        ),
        ReconciliationRow(
            metric="n_included_trades",
            recomputed=n_included,
            reference=ref_trades,
            delta=n_included - ref_trades,
            tolerance=0,
            passed=n_included == ref_trades,
        ),
        ReconciliationRow(
            metric="n_traded_dates",
            recomputed=n_traded_dates,
            reference=ref_traded_dates,
            delta=n_traded_dates - ref_traded_dates,
            tolerance=0,
            passed=n_traded_dates == ref_traded_dates,
        ),
        ReconciliationRow(
            metric="expected_included_trades",
            recomputed=n_included,
            reference=EXPECTED_INCLUDED_TRADES,
            delta=n_included - EXPECTED_INCLUDED_TRADES,
            tolerance=0,
            passed=n_included == EXPECTED_INCLUDED_TRADES,
        ),
    ]
    return ReconciliationResult(rows=rows)


def _group_pnl(included: pd.DataFrame, column: str) -> pd.DataFrame:
    """Grouped P&L sorted by descending P&L with a deterministic key tie-break."""
    grouped = included.groupby(column, sort=False)["pnl_total"].sum().reset_index()
    return grouped.sort_values(
        ["pnl_total", column], ascending=[False, True]
    ).reset_index(drop=True)


def pnl_excluding_top_groups(
    included: pd.DataFrame,
    column: str,
    n_exclude: int = BREADTH_TOP_N,
) -> tuple[float, list[Any]]:
    """Total P&L after removing the ``n_exclude`` highest-P&L groups of ``column``."""
    if included.empty:
        return 0.0, []
    ranked = _group_pnl(included, column)
    excluded = ranked.head(n_exclude)[column].tolist()
    remaining = included[~included[column].isin(excluded)]
    return float(remaining["pnl_total"].sum()), excluded


def yearly_pnl_table(included: pd.DataFrame) -> pd.DataFrame:
    if included.empty:
        return pd.DataFrame(columns=["year", "year_pnl", "n_trades"])
    frame = included.copy()
    frame["year"] = frame["trade_date"].map(lambda d: d.year)
    table = (
        frame.groupby("year", sort=True)
        .agg(year_pnl=("pnl_total", "sum"), n_trades=("pnl_total", "size"))
        .reset_index()
    )
    table["year_pnl"] = table["year_pnl"].astype(float)
    table["n_trades"] = table["n_trades"].astype(int)
    return table


def compute_d1_gate_scorecard(bundle: MidPrimaryBundle) -> D1Scorecard:
    """Evaluate the four frozen gate parts. Formulas must not change after output."""
    included = bundle.included
    total_pnl = float(included["pnl_total"].sum())
    n_included = int(len(included))
    view_a = compute_view_a(bundle.date_status, bundle.date_summary)
    mean_car = float(view_a["mean_cycle_car"])

    attribution = compute_long_short_attribution(
        bundle.date_status, bundle.date_summary, bundle.trade_log
    )
    side_rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        block = attribution[side]
        n_trades = int(block["n_traded_rows"])
        side_rows.append(
            {
                "side": side,
                "pnl_total": float(block["pnl_total"]),
                "n_trades": n_trades,
                "trade_share": (n_trades / n_included) if n_included else 0.0,
                "capital_at_risk_dollars": float(block["capital_at_risk_dollars"]),
            }
        )

    pnl_excl_dates, excluded_dates = pnl_excluding_top_groups(included, "trade_date")
    pnl_excl_tickers, excluded_tickers = pnl_excluding_top_groups(included, "ticker")

    years = yearly_pnl_table(included)
    n_positive_years = int((years["year_pnl"] > 0.0).sum()) if not years.empty else 0
    if years.empty:
        best_year: int | None = None
        pnl_excl_best_year = 0.0
    else:
        ranked_years = years.sort_values(
            ["year_pnl", "year"], ascending=[False, True]
        ).reset_index(drop=True)
        best_year = int(ranked_years.loc[0, "year"])
        pnl_excl_best_year = float(
            included[included["trade_date"].map(lambda d: d.year) != best_year]["pnl_total"].sum()
        )

    qualifying_sides = [
        row["side"]
        for row in side_rows
        if row["pnl_total"] > 0.0 and row["trade_share"] >= LOCATION_MIN_TRADE_SHARE
    ]

    parts = [
        GatePart(
            part_id="G-Sign",
            passed=total_pnl > 0.0 and mean_car > 0.0,
            detail=f"total_pnl={total_pnl:.2f} view_a_mean_cycle_car={mean_car:.9f}",
            metrics={"total_pnl": total_pnl, "view_a_mean_cycle_car": mean_car},
        ),
        GatePart(
            part_id="G-Breadth",
            passed=pnl_excl_dates > 0.0 and pnl_excl_tickers > 0.0,
            detail=(
                f"excl_top{BREADTH_TOP_N}_dates={pnl_excl_dates:.2f} "
                f"excl_top{BREADTH_TOP_N}_tickers={pnl_excl_tickers:.2f}"
            ),
            metrics={
                "total_pnl_excl_top5_dates": pnl_excl_dates,
                "total_pnl_excl_top5_tickers": pnl_excl_tickers,
                "excluded_dates": [str(d) for d in excluded_dates],
                "excluded_tickers": [str(t) for t in excluded_tickers],
            },
        ),
        GatePart(
            part_id="G-Location",
            passed=bool(qualifying_sides),
            detail=(
                f"qualifying_sides={qualifying_sides or 'none'} "
                f"(min_share={LOCATION_MIN_TRADE_SHARE:.2f})"
            ),
            metrics={"qualifying_sides": qualifying_sides, "sides": side_rows},
        ),
        GatePart(
            part_id="G-Stability",
            passed=n_positive_years >= MIN_POSITIVE_YEARS and pnl_excl_best_year > 0.0,
            detail=(
                f"positive_years={n_positive_years} best_year={best_year} "
                f"excl_best_year_pnl={pnl_excl_best_year:.2f}"
            ),
            metrics={
                "n_years_with_positive_pnl": n_positive_years,
                "best_year": best_year,
                "total_pnl_excl_best_year": pnl_excl_best_year,
            },
        ),
    ]

    verdict = VERDICT_CONTINUE if all(p.passed for p in parts) else VERDICT_STOP
    return D1Scorecard(
        verdict=verdict,
        parts=parts,
        metrics={
            "total_pnl": total_pnl,
            "n_included_trades": n_included,
            "view_a_mean_cycle_car": mean_car,
            "view_a_sharpe": view_a["annualized_sharpe"],
            "view_a_max_drawdown": view_a["max_drawdown"],
            "n_traded_dates": int(view_a["n_traded_dates"]),
            "capital_at_risk_dollars": float(
                pd.to_numeric(included["capital_at_risk_dollars"], errors="coerce").sum()
            ),
        },
        side_attribution=side_rows,
        breadth_exclusions=[
            {
                "basis": "baseline",
                "excluded": "",
                "pnl_total": total_pnl,
            },
            {
                "basis": f"exclude_top{BREADTH_TOP_N}_dates",
                "excluded": ";".join(str(d) for d in excluded_dates),
                "pnl_total": pnl_excl_dates,
            },
            {
                "basis": f"exclude_top{BREADTH_TOP_N}_tickers",
                "excluded": ";".join(str(t) for t in excluded_tickers),
                "pnl_total": pnl_excl_tickers,
            },
            {
                "basis": "exclude_best_year",
                "excluded": str(best_year) if best_year is not None else "",
                "pnl_total": pnl_excl_best_year,
            },
        ],
        yearly_pnl=years.to_dict("records"),
    )


def run_d1_analysis(
    *,
    run_dir: Path | None = None,
    d1_code_commit_sha: str | None = None,
) -> D1Result:
    """Load mid-primary artifacts, reconcile, and evaluate the frozen D1 gate."""
    run_dir = run_dir or OFFICIAL_RUN_DIR
    d1_code_commit_sha = d1_code_commit_sha or get_current_repo_sha()

    try:
        bundle = load_mid_primary_tables(run_dir)
        report = load_accepted_mid_primary_report(run_dir)
    except Exception as exc:  # precondition failure is a blocker, not a stop
        scorecard = D1Scorecard(verdict=VERDICT_BLOCKED)
        reconciliation = ReconciliationResult()
        manifest = _build_manifest(
            run_dir=run_dir,
            d1_code_commit_sha=d1_code_commit_sha,
            verdict=VERDICT_BLOCKED,
            scorecard=scorecard,
            reconciliation=reconciliation,
            blocker=f"{type(exc).__name__}: {exc}",
        )
        return D1Result(VERDICT_BLOCKED, scorecard, reconciliation, manifest)

    reconciliation = reconcile_mid_primary(bundle, report)
    scorecard = compute_d1_gate_scorecard(bundle)
    if not reconciliation.passed:
        scorecard.verdict = VERDICT_BLOCKED

    manifest = _build_manifest(
        run_dir=run_dir,
        d1_code_commit_sha=d1_code_commit_sha,
        verdict=scorecard.verdict,
        scorecard=scorecard,
        reconciliation=reconciliation,
        blocker=None,
    )
    return D1Result(scorecard.verdict, scorecard, reconciliation, manifest)


def _build_manifest(
    *,
    run_dir: Path,
    d1_code_commit_sha: str | None,
    verdict: str,
    scorecard: D1Scorecard,
    reconciliation: ReconciliationResult,
    blocker: str | None,
) -> dict[str, Any]:
    report_path = run_dir / "decision_report.json"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_run_dir": str(run_dir),
        "decision_report_sha256": sha256_file(report_path) if report_path.exists() else None,
        "fill_label": FILL_LABEL,
        "window_start": str(PRIMARY_START),
        "window_end": str(PRIMARY_END),
        "expected_included_trades": EXPECTED_INCLUDED_TRADES,
        "d1_code_commit_sha": d1_code_commit_sha,
        "verdict": verdict,
        "blocker": blocker,
        "reconciliation": reconciliation.as_records(),
        "metrics": scorecard.metrics,
        "gates": [
            {
                "part_id": part.part_id,
                "passed": part.passed,
                "detail": part.detail,
                "metrics": part.metrics,
            }
            for part in scorecard.parts
        ],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return str(value)
    return value


def write_d1_manifest(result: D1Result, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(result.manifest), indent=2), encoding="utf-8"
    )
    return output_path


def write_d1_scorecard(result: D1Result, output_path: Path) -> Path:
    payload = {
        "verdict": result.verdict,
        "reconciliation_passed": result.reconciliation.passed,
        "reconciliation": result.reconciliation.as_records(),
        "gates": [
            {
                "part_id": part.part_id,
                "passed": part.passed,
                "detail": part.detail,
                "metrics": part.metrics,
            }
            for part in result.scorecard.parts
        ],
        "metrics": result.scorecard.metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return output_path


def write_d1_tables(result: D1Result, evidence_dir: Path) -> dict[str, Path]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "side_attribution": evidence_dir / "d1_side_attribution.csv",
        "breadth_exclusions": evidence_dir / "d1_breadth_exclusions.csv",
        "yearly_pnl": evidence_dir / "d1_yearly_pnl.csv",
    }
    pd.DataFrame(result.scorecard.side_attribution).to_csv(paths["side_attribution"], index=False)
    pd.DataFrame(result.scorecard.breadth_exclusions).to_csv(
        paths["breadth_exclusions"], index=False
    )
    pd.DataFrame(result.scorecard.yearly_pnl).to_csv(paths["yearly_pnl"], index=False)
    return paths


def resolve_evidence_dir() -> Path:
    """Evidence directory: honour the exporter's env override, else timestamp a new one."""
    override = os.environ.get(EVIDENCE_DIR_ENV)
    if override:
        path = Path(override)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(f"C:/MomentumCVG_env/runs/sprint007_d1_{stamp}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_execution_receipt(
    *,
    evidence_dir: Path,
    executed_notebook: Path,
    html_export: Path,
    d1_code_commit_sha: str | None = None,
) -> Path:
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "d1_code_commit_sha": d1_code_commit_sha or get_current_repo_sha(),
        "sprint006_execution_repo_sha": "e205b9acc5d0400aa38169de721acb7fb8268f29",
        "executed_notebook": str(executed_notebook),
        "executed_notebook_sha256": sha256_file(executed_notebook),
        "html_export": str(html_export),
        "html_export_sha256": sha256_file(html_export),
    }
    path = evidence_dir / "execution_receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def export_d1_evidence(
    *,
    result: D1Result,
    clean_notebook: Path,
    evidence_dir: Path | None = None,
    d1_code_commit_sha: str | None = None,
) -> Path:
    """Execute the D1 notebook on a fresh kernel and write all evidence artifacts."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = evidence_dir or Path(f"C:/MomentumCVG_env/runs/sprint007_d1_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    executed = evidence_dir / "d1_gross_margin.executed.ipynb"
    html_path = evidence_dir / "d1_gross_margin.html"
    d1_code_commit_sha = d1_code_commit_sha or get_current_repo_sha()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env[EVIDENCE_DIR_ENV] = str(evidence_dir)
    python = Path("C:/MomentumCVG_env/venv/Scripts/python.exe")
    if not python.exists():
        python = Path(sys.executable)
    jupyter = [str(python), "-m", "jupyter"]

    subprocess.run(
        [
            *jupyter,
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(clean_notebook),
            "--output",
            executed.name,
            "--output-dir",
            str(evidence_dir),
            "--ExecutePreprocessor.kernel_name=momentumcvg",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [
            *jupyter,
            "nbconvert",
            "--to",
            "html",
            str(executed),
            "--output",
            html_path.name,
            "--output-dir",
            str(evidence_dir),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    write_d1_manifest(result, evidence_dir / "d1_gross_margin_manifest.json")
    write_d1_scorecard(result, evidence_dir / "d1_scorecard.json")
    write_d1_tables(result, evidence_dir)
    write_execution_receipt(
        evidence_dir=evidence_dir,
        executed_notebook=executed,
        html_export=html_path,
        d1_code_commit_sha=d1_code_commit_sha,
    )
    return evidence_dir


__all__ = [
    "D1AnalysisError",
    "D1Result",
    "D1Scorecard",
    "GatePart",
    "MidPrimaryBundle",
    "ReconciliationResult",
    "compute_d1_gate_scorecard",
    "export_d1_evidence",
    "load_accepted_mid_primary_report",
    "load_mid_primary_tables",
    "pnl_excluding_top_groups",
    "reconcile_mid_primary",
    "resolve_evidence_dir",
    "run_d1_analysis",
    "select_included_traded_rows",
    "write_d1_manifest",
    "write_d1_scorecard",
    "write_d1_tables",
    "yearly_pnl_table",
]
