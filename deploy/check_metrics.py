"""Release gate: block a deployment if the freshly trained model is not good enough.

Usage: python deploy/check_metrics.py [reports/metrics.json]
Exit code 1 lists the reasons. In GitHub Actions the result is added to the job summary.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

BASELINE = "logistic_regression"
MIN_TEST_PR_AUC = 0.85   # current model: 0.895, random ranking: about 0.37
MIN_SYSTEM_RECALL = 0.85  # rule + model on the test years; the threshold targets 0.90
MAX_TEST_ECE = 0.03       # after calibration; raw model: 0.034


def check(m: Dict) -> List[str]:
    """Return the reasons to block the release; an empty list means it may ship."""
    problems = []
    selected = m["selected"]
    pr_auc = m["models"][selected]["test"]["pr_auc"]
    if pr_auc < MIN_TEST_PR_AUC:
        problems.append(f"test PR-AUC {pr_auc} is below {MIN_TEST_PR_AUC}")
    if selected != BASELINE:
        base = m["models"][BASELINE]["test"]["pr_auc"]
        if pr_auc <= base:
            problems.append(f"{selected} (test PR-AUC {pr_auc}) does not beat the baseline ({base})")
    recall = m["triage_system_test"]["recall"]
    if recall < MIN_SYSTEM_RECALL:
        problems.append(f"triage recall {recall} is below {MIN_SYSTEM_RECALL}")
    calibration = m.get("calibration")
    if not calibration:
        problems.append("the model is not calibrated")
    elif calibration["test_after"]["ece"] > MAX_TEST_ECE:
        problems.append(f"test ECE {calibration['test_after']['ece']} is above {MAX_TEST_ECE}")
    return problems


def main(argv: List[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "reports/metrics.json")
    problems = check(json.loads(path.read_text()))
    lines = ["## Release gate", ""]
    lines += [f"- BLOCKED: {p}" for p in problems] or ["- passed: PR-AUC, baseline, recall and calibration"]
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
