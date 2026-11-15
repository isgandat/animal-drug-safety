import copy
import json

import pytest

from deploy.check_metrics import check, main

GOOD = {
    "selected": "mlp_neural_net",
    "models": {
        "logistic_regression": {"test": {"pr_auc": 0.80}},
        "mlp_neural_net": {"test": {"pr_auc": 0.89}},
    },
    "triage_system_test": {"recall": 0.91},
    "calibration": {"test_before": {"ece": 0.034}, "test_after": {"ece": 0.015}},
}


def with_change(path, value):
    m = copy.deepcopy(GOOD)
    *keys, last = path
    node = m
    for k in keys:
        node = node[k]
    if value is None:
        del node[last]
    else:
        node[last] = value
    return m


def test_good_model_passes():
    assert check(GOOD) == []


@pytest.mark.parametrize("path, value, reason", [
    (["models", "mlp_neural_net", "test", "pr_auc"], 0.84, "PR-AUC"),
    (["models", "logistic_regression", "test", "pr_auc"], 0.90, "baseline"),
    (["triage_system_test", "recall"], 0.80, "recall"),
    (["calibration", "test_after", "ece"], 0.05, "ECE"),
    (["calibration"], None, "calibrat"),
])
def test_each_rule_blocks_a_release(path, value, reason):
    problems = check(with_change(path, value))
    assert len(problems) == 1
    assert reason in problems[0]


def test_main_exit_code_and_job_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(GOOD))
    bad.write_text(json.dumps(with_change(["triage_system_test", "recall"], 0.5)))

    assert main(["check_metrics.py", str(good)]) == 0
    assert main(["check_metrics.py", str(bad)]) == 1
    text = summary.read_text()
    assert "Release gate" in text and "recall" in text
