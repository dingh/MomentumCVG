"""Execute Sprint 008 D2 frozen-rule retrospective validation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.backtest.sprint007_artifact_validation import get_current_repo_sha
from src.backtest.sprint008_d2_fixed_exclusion_validation import (
    export_d2_evidence,
    run_fixed_exclusion_validation,
)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = Path(f"C:/MomentumCVG_env/runs/sprint008_d2_{ts}")
    command = (
        "C:/MomentumCVG_env/venv/Scripts/python.exe "
        "scripts/run_sprint008_d2_fixed_exclusion_validation.py"
    )
    print("evidence_dir", evidence, flush=True)
    print("code_sha", get_current_repo_sha(), flush=True)
    result = run_fixed_exclusion_validation()
    print("calendar", result.calendar_validation, flush=True)
    for measurement, contrast in result.inference["contrasts"].items():
        print(
            measurement,
            "label",
            contrast["label"],
            "mean",
            contrast["mean"],
            "p_adjusted",
            contrast["p_adjusted"],
            flush=True,
        )
    print("timings", result.stage_timings, flush=True)
    export_d2_evidence(result=result, evidence_dir=evidence, command=command)
    print("exported", evidence, flush=True)
    print("files", sorted(p.name for p in evidence.iterdir()), flush=True)


if __name__ == "__main__":
    main()
