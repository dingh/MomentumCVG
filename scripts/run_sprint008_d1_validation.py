"""Execute Sprint 008 D1 measurement validation against official Sprint 006 artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint008_d1_measurement_validation import (
    export_d1_evidence,
    run_d1_validation,
)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint008_d1_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint008_d1_validation.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", get_current_repo_sha(), flush=True)
    result = run_d1_validation()
    print("GATE", result.gate.get("decision"), flush=True)
    print("labels", json.dumps(result.labels, indent=2), flush=True)
    print("predicates", json.dumps(result.predicates, indent=2, default=str), flush=True)
    print("coverage", json.dumps(result.coverage, indent=2, default=str), flush=True)
    print("timings", json.dumps(result.stage_timings, indent=2), flush=True)
    export_d1_evidence(result=result, evidence_dir=evidence, command=command)
    print("exported", evidence, flush=True)
    print("files", sorted(p.name for p in evidence.iterdir()), flush=True)


if __name__ == "__main__":
    main()
