import json

import joblib
import numpy as np
import pytest

from etl import db
from ml.features import load_cases, rule_serious
from ml.train import evaluate, pick_threshold, run


def test_pick_threshold_reaches_the_target_recall():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    t = pick_threshold(y, p, 0.8)
    assert evaluate(y, p, t)["recall"] >= 0.8
    assert t == pytest.approx(0.4)  # the highest threshold that still keeps recall >= 0.8


def test_models_learn_the_synthetic_pattern(trained):
    m = trained["metrics"]
    assert set(m["models"]) == {"logistic_regression", "mlp_neural_net"}
    for result in m["models"].values():
        assert result["validation"]["roc_auc"] > 0.75
        assert result["validation"]["recall"] >= 0.9
    assert m["selected"] == max(m["models"], key=lambda n: m["models"][n]["validation"]["pr_auc"])


def test_rule_cases_are_kept_out_of_the_model_data(trained, synthetic_db):
    conn = db.connect(synthetic_db, read_only=True)
    expected = sum(1 for e in load_cases(conn)
                   if e.year <= 2022 and not rule_serious(e.case["reactions"], e.case["outcomes"]))
    assert trained["metrics"]["metadata"]["rows"]["train"] == expected
    assert trained["metrics"]["triage_system_test"]["rule_precision"] == 1.0


def test_artifacts_are_versioned_and_loadable(trained):
    bundle = joblib.load(trained["model"])
    meta = bundle["metadata"]
    assert meta["model_version"].endswith(meta["data_fingerprint"][:8])
    assert {"threshold", "split", "rows", "sklearn"} <= set(meta)
    saved = json.loads((trained["reports"] / "metrics.json").read_text())
    assert saved["metadata"] == meta
    assert "Model comparison" in (trained["reports"] / "model_card.md").read_text()


def test_empty_split_stops_with_a_clear_message(synthetic_db, tmp_path):
    with pytest.raises(SystemExit, match="split 'test' is empty"):
        run(synthetic_db, tmp_path / "m", tmp_path / "r", train_to=2022, val_year=2023, test_from=2030)
