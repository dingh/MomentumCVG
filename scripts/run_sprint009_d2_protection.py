"""Execute Sprint 009 D2 against the accepted D1 development panel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint009_d2_protection_comparison import (
    ACCEPTED_D1_DIR,
    SUPERSEDED_D1_DIR,
    ComparisonResult,
    accepted_d1_hashes,
    compare_development,
    d1_handoff_problems,
    render_report_md,
)


def _progress(stage: str) -> None:
    print(f"stage {stage}", flush=True)


def _write_cumulative(result: ComparisonResult, path: Path) -> None:
    dates = result.dates.sort_values("trade_date")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates["trade_date"], dates["cumulative_body_cross"], label="Body-only cross")
    ax.plot(dates["trade_date"], dates["cumulative_fly_cross"], label="Iron-fly cross")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylabel("Dollars")
    ax.set_title("Development cumulative P&L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_annual(result: ComparisonResult, path: Path) -> None:
    annual = result.annual.sort_values("year")
    years = [str(int(value)) for value in annual["year"]]
    index = range(len(years))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([item - width / 2 for item in index], annual["p_body_cross"], width, label="Body-only cross")
    ax.bar([item + width / 2 for item in index], annual["p_fly_cross"], width, label="Iron-fly cross")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(list(index))
    ax.set_xticklabels(years)
    ax.set_ylabel("Dollars")
    ax.set_title("Development annual P&L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_blocked(evidence: Path, inventory: dict, problems: list[str], command: str, code_sha: str) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "input_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    payload = {
        "verdict": "BLOCKED",
        "provenance_problems": problems,
        "code_sha": code_sha,
        "d1_dir": str(ACCEPTED_D1_DIR),
        "invocation": command,
    }
    (evidence / "d2_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (evidence / "d2_report.md").write_text(
        "# Sprint 009 D2 protection comparison\n\n**Verdict:** `BLOCKED`\n\n"
        + "\n".join(f"- {item}" for item in problems)
        + "\n\nNo interpretation. A failed provenance check is a named blocker, not a partial economic result.\n",
        encoding="utf-8",
    )
    (evidence / "execution_receipt.json").write_text(
        json.dumps(
            {
                "generated_utc": inventory["generated_utc"],
                "verdict": "BLOCKED",
                "command": command,
                "code_sha": code_sha,
                "d1_dir": str(ACCEPTED_D1_DIR),
                "evidence_dir": str(evidence),
                "input_hashes": inventory["files"],
                "provenance_problems": problems,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint009_d2_{ts}")
    if evidence.resolve() in {ACCEPTED_D1_DIR.resolve(), SUPERSEDED_D1_DIR.resolve()}:
        raise SystemExit("refusing to write into a D1 directory")
    code_sha = get_current_repo_sha()
    command = "C:/MomentumCVG_env/venv/Scripts/python.exe scripts/run_sprint009_d2_protection.py"
    print("evidence_dir", evidence, flush=True)
    print("code_sha", code_sha, flush=True)
    print("d1_dir", ACCEPTED_D1_DIR, flush=True)
    _progress("inventory")
    receipt = json.loads((ACCEPTED_D1_DIR / "execution_receipt.json").read_text(encoding="utf-8"))
    hashes = accepted_d1_hashes(ACCEPTED_D1_DIR)
    inventory = {
        "d1_dir": str(ACCEPTED_D1_DIR),
        "d1_code_sha": receipt.get("code_sha"),
        "d1_verdict": receipt.get("verdict"),
        "code_sha": code_sha,
        "command": command,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes,
    }
    problems = d1_handoff_problems(receipt, ACCEPTED_D1_DIR, hashes)
    inventory["provenance_problems"] = problems
    if problems:
        _write_blocked(evidence, inventory, problems, command, code_sha or "")
        print("VERDICT BLOCKED", flush=True)
        for item in problems:
            print(f"FAIL provenance: {item}", flush=True)
        print("exported", evidence, flush=True)
        return
    trades = pd.read_parquet(ACCEPTED_D1_DIR / "trade_decomposition.parquet")
    dates = pd.read_parquet(ACCEPTED_D1_DIR / "date_decomposition.parquet")
    annual = pd.read_parquet(ACCEPTED_D1_DIR / "annual_decomposition.parquet")
    aggregate = json.loads((ACCEPTED_D1_DIR / "aggregate_dollars.json").read_text(encoding="utf-8"))
    _progress("comparison")
    result = compare_development(
        trades,
        dates,
        annual,
        aggregate,
        require_official_coverage=True,
        handoff_problems=problems,
    )
    inventory["development_trades"] = int(len(result.trades))
    inventory["development_dates"] = int(len(result.dates))
    _progress("reconciliation")
    _progress("calendar")
    _progress("output")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "input_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    result.trades.to_parquet(evidence / "trade_comparison.parquet", index=False)
    result.dates.to_parquet(evidence / "date_comparison.parquet", index=False)
    result.annual.to_parquet(evidence / "annual_comparison.parquet", index=False)
    result.worst.to_parquet(evidence / "worst_events.parquet", index=False)
    (evidence / "frequency.json").write_text(json.dumps(result.frequency, indent=2, default=str), encoding="utf-8")
    (evidence / "concentration.json").write_text(json.dumps(result.concentration, indent=2, default=str), encoding="utf-8")
    (evidence / "drawdown.json").write_text(json.dumps(result.drawdown, indent=2, default=str), encoding="utf-8")
    payload = {
        **result.report,
        "code_sha": code_sha,
        "d1_dir": str(ACCEPTED_D1_DIR),
        "invocation": command,
        "gates": [{"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail} for gate in result.gates],
        "totals": result.totals,
        "frequency": result.frequency,
        "concentration": result.concentration,
        "drawdown": result.drawdown,
    }
    (evidence / "d2_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (evidence / "d2_report.md").write_text(render_report_md(result), encoding="utf-8")
    (evidence / "execution_receipt.json").write_text(
        json.dumps(
            {
                "generated_utc": inventory["generated_utc"],
                "verdict": result.verdict,
                "command": command,
                "code_sha": code_sha,
                "d1_dir": str(ACCEPTED_D1_DIR),
                "d1_code_sha": receipt.get("code_sha"),
                "evidence_dir": str(evidence),
                "input_hashes": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.verdict == "READY":
        _write_cumulative(result, evidence / "cumulative_cross.png")
        _write_annual(result, evidence / "annual_cross.png")
    print("VERDICT", result.verdict, flush=True)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"{status} {gate.gate_id}: {gate.detail}", flush=True)
    print("exported", evidence, flush=True)


if __name__ == "__main__":
    main()
