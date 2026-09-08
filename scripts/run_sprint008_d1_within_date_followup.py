"""Execute Sprint 008 D1 within-date L vs U follow-up (M1/M2, development only)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint008_d1_within_date_followup import (
    export_followup_evidence,
    run_within_date_followup,
)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint008_d1_within_date_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint008_d1_within_date_followup.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", get_current_repo_sha(), flush=True)
    result = run_within_date_followup()
    print(
        "summaries",
        json.dumps(result.measurement_summaries, indent=2, default=str),
        flush=True,
    )
    print("timings", json.dumps(result.stage_timings, indent=2), flush=True)
    print(
        "interpretation",
        json.dumps(result.report.get("support_interpretation"), indent=2),
        flush=True,
    )
    export_followup_evidence(result=result, evidence_dir=evidence, command=command)
    print("exported", evidence, flush=True)
    print("files", sorted(p.name for p in evidence.iterdir()), flush=True)


if __name__ == "__main__":
    main()
