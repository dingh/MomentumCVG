"""Execute Sprint 009 D0 readiness against official Sprint 006 artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import OFFICIAL_RUN_DIR, get_current_repo_sha
from src.backtest.sprint009_d0_body_wing_readiness import export_d0_evidence, run_d0_readiness


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint009_d0_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint009_d0_readiness.py"
    )
    code_sha = get_current_repo_sha()
    print("evidence_dir", evidence, flush=True)
    print("code_sha", code_sha, flush=True)
    print("official_run_dir", OFFICIAL_RUN_DIR, flush=True)
    result = run_d0_readiness()
    print("stage output", flush=True)
    export_d0_evidence(result, evidence, code_sha=code_sha)
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": result.verdict,
        "command": command,
        "code_sha": code_sha,
        "evidence_dir": str(evidence),
        "official_run_dir": str(OFFICIAL_RUN_DIR),
    }
    (evidence / "execution_receipt.json").write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )
    print("VERDICT", result.verdict, flush=True)
    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"{status} {gate.gate_id}: {gate.detail}", flush=True)
    print("exported", evidence, flush=True)


if __name__ == "__main__":
    main()
