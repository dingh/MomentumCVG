"""Execute Sprint 009 D1 against the accepted D0 development panel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from src.backtest.sprint007_artifact_validation import get_current_repo_sha, sha256_file
from src.backtest.sprint009_d1_body_wing_decomposition import (
    ACCEPTED_D0_DIR,
    SUPERSEDED_D0_DIR,
    DecompositionResult,
    decompose_development,
    render_report_md,
)


def _progress(stage: str) -> None:
    print(f"stage {stage}", flush=True)


def _write_waterfall(result: DecompositionResult, path: Path) -> None:
    labels = ["B_mid", "-H_body", "-W_mid", "-H_wing", "+W_pay", "P_fly_cross"]
    steps = [
        float(result.dollars["b_mid"]),
        -float(result.dollars["h_body"]),
        -float(result.dollars["w_mid"]),
        -float(result.dollars["h_wing"]),
        float(result.dollars["w_pay"]),
    ]
    total = float(result.dollars["p_fly_cross"])
    bases: list[float] = []
    heights: list[float] = []
    running = 0.0
    for value in steps:
        if value >= 0:
            bases.append(running)
            heights.append(value)
        else:
            bases.append(running + value)
            heights.append(-value)
        running += value
    bases.append(0.0)
    heights.append(total)
    colors = ["#2c7bb6", "#d7191c", "#d7191c", "#d7191c", "#1a9641", "#4d4d4d"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, heights, bottom=bases, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylabel("Dollars")
    ax.set_title("Development iron-fly decomposition")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint009_d1_{ts}")
    if evidence.resolve() in {ACCEPTED_D0_DIR.resolve(), SUPERSEDED_D0_DIR.resolve()}:
        raise SystemExit("refusing to write into a D0 directory")
    code_sha = get_current_repo_sha()
    matched_path = ACCEPTED_D0_DIR / "matched_short_iron_flies.parquet"
    calendar_path = ACCEPTED_D0_DIR / "short_calendar.parquet"
    receipt_path = ACCEPTED_D0_DIR / "execution_receipt.json"
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint009_d1_decomposition.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", code_sha, flush=True)
    print("d0_dir", ACCEPTED_D0_DIR, flush=True)
    _progress("inventory")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory = {
        "d0_dir": str(ACCEPTED_D0_DIR),
        "d0_code_sha": receipt.get("code_sha"),
        "d0_verdict": receipt.get("verdict"),
        "code_sha": code_sha,
        "command": command,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "matched_short_iron_flies.parquet": sha256_file(matched_path),
            "short_calendar.parquet": sha256_file(calendar_path),
        },
    }
    matched = pd.read_parquet(matched_path)
    calendar = pd.read_parquet(calendar_path)
    _progress("decomposition")
    result = decompose_development(matched, calendar, require_official_coverage=True)
    inventory["development_trades"] = int(len(result.trades))
    inventory["development_dates"] = int(len(result.dates))
    _progress("reconciliation")
    _progress("calendar")
    _progress("output")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "input_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    result.trades.to_parquet(evidence / "trade_decomposition.parquet", index=False)
    result.dates.to_parquet(evidence / "date_decomposition.parquet", index=False)
    result.annual.to_parquet(evidence / "annual_decomposition.parquet", index=False)
    (evidence / "aggregate_dollars.json").write_text(json.dumps(result.dollars, indent=2, default=str), encoding="utf-8")
    (evidence / "aggregate_ratios.json").write_text(json.dumps(result.ratios, indent=2, default=str), encoding="utf-8")
    payload = {
        **result.report,
        "code_sha": code_sha,
        "d0_dir": str(ACCEPTED_D0_DIR),
        "invocation": command,
        "gates": [{"gate_id": gate.gate_id, "passed": gate.passed, "detail": gate.detail} for gate in result.gates],
        "dollars": result.dollars,
        "ratios": result.ratios,
    }
    (evidence / "d1_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (evidence / "d1_report.md").write_text(render_report_md(result), encoding="utf-8")
    (evidence / "execution_receipt.json").write_text(
        json.dumps(
            {
                "generated_utc": inventory["generated_utc"],
                "verdict": result.verdict,
                "command": command,
                "code_sha": code_sha,
                "d0_dir": str(ACCEPTED_D0_DIR),
                "d0_code_sha": receipt.get("code_sha"),
                "evidence_dir": str(evidence),
                "input_hashes": inventory["files"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.verdict == "READY":
        _write_waterfall(result, evidence / "waterfall_development.png")
    print("VERDICT", result.verdict, flush=True)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"{status} {gate.gate_id}: {gate.detail}", flush=True)
    print("exported", evidence, flush=True)


if __name__ == "__main__":
    main()
