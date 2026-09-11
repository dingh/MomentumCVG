"""Execute Sprint 008 D1 cost diagnosis and fixed U-exclusion follow-up."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint008_d1_cost_diagnosis import (
    export_cost_diagnosis_evidence,
    run_cost_diagnosis,
)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint008_d1_cost_diagnosis_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint008_d1_cost_diagnosis.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", get_current_repo_sha(), flush=True)
    result = run_cost_diagnosis()
    print(
        "reconciliation",
        json.dumps(result.reconciliation, indent=2, default=str),
        flush=True,
    )
    print(
        "exclusion",
        json.dumps(result.exclusion_summary, indent=2, default=str),
        flush=True,
    )
    print(
        "recommendation",
        json.dumps(result.interpretation.get("recommendation"), indent=2),
        flush=True,
    )
    print("timings", json.dumps(result.stage_timings, indent=2), flush=True)
    export_cost_diagnosis_evidence(result=result, evidence_dir=evidence, command=command)
    print("exported", evidence, flush=True)
    print("files", sorted(p.name for p in evidence.iterdir()), flush=True)


if __name__ == "__main__":
    main()
