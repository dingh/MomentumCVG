"""Execute Sprint 008 D0 readiness against official Sprint 006 artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint008_d0_input_readiness import (
    export_d0_readiness_evidence,
    run_d0_readiness,
)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint008_d0_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint008_d0_readiness.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", get_current_repo_sha(), flush=True)
    result = run_d0_readiness()
    print("VERDICT", result.verdict, flush=True)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"{status} {gate.gate_id}: {gate.detail}", flush=True)
    print("coverage", json.dumps(result.coverage, indent=2, default=str), flush=True)
    print("timings", json.dumps(result.stage_timings, indent=2), flush=True)
    export_d0_readiness_evidence(
        result=result,
        evidence_dir=evidence,
        execute_notebook=False,
        d0_code_commit_sha=get_current_repo_sha(),
    )
    manifest_path = evidence / "d0_readiness_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command"] = command
    payload["evidence_dir"] = str(evidence)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    receipt_path = evidence / "execution_receipt.json"
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": result.verdict,
        "command": command,
        "code_sha": get_current_repo_sha(),
        "evidence_dir": str(evidence),
        "official_run_dir": payload.get("official_run_dir"),
        "sprint006_execution_repo_sha": payload.get("sprint006_execution_repo_sha"),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("exported", evidence, flush=True)
    print("files", sorted(p.name for p in evidence.iterdir()), flush=True)


if __name__ == "__main__":
    main()
