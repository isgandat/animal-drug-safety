"""Train and evaluate the seriousness-triage model.

    python -m ml.train --db data/ads.db

Time-based split (no shuffling across years): train on older reports, choose
the model and alert threshold on the validation year, report on later years.
Reports that mention death or euthanasia are handled by a rule and are left
out of model training and model metrics (they would leak the label).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from etl import db
from ml import drift
from ml.features import Example, load_cases, report_features, rule_serious

log = logging.getLogger("train")


def make_models(seed: int) -> Dict:
    return {
        "logistic_regression": make_pipeline(
            MaxAbsScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "mlp_neural_net": make_pipeline(
            MaxAbsScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, batch_size=1024,
                                          learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                                          n_iter_no_change=4, random_state=seed)),
    }


def pick_threshold(y: Sequence[int], p: Sequence[float], target_recall: float) -> float:
    """Highest score threshold that still catches `target_recall` of serious reports."""
    _, recall, thresholds = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def evaluate(y: Sequence[int], p: Sequence[float], threshold: float = None) -> Dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }
    if threshold is not None:
        flag = p >= threshold
        tp = int((flag & (y == 1)).sum())
        out.update({
            "threshold": round(threshold, 4),
            "recall": round(tp / max(int(y.sum()), 1), 4),
            "precision": round(tp / max(int(flag.sum()), 1), 4),
            "flagged_share": round(float(flag.mean()), 4),
        })
    return out


def vectorize(train_dicts: List[Dict], min_count: int) -> DictVectorizer:
    vec = DictVectorizer(sparse=True)
    x = vec.fit_transform(train_dicts)
    support = np.asarray((x != 0).sum(axis=0)).ravel() >= min_count
    return vec.restrict(support)


def data_fingerprint(conn) -> str:
    rows = conn.execute("SELECT key, sha256 FROM partitions ORDER BY key").fetchall()
    return hashlib.sha256(json.dumps([tuple(r) for r in rows]).encode()).hexdigest()


def top_coefficients(pipeline, vec: DictVectorizer, k: int = 15) -> Dict[str, List]:
    coef = pipeline[-1].coef_.ravel()
    names = np.asarray(vec.get_feature_names_out())
    order = np.argsort(coef)
    return {
        "raises_risk": [[str(n), round(float(c), 3)] for n, c in zip(names[order[::-1][:k]], coef[order[::-1][:k]])],
        "lowers_risk": [[str(n), round(float(c), 3)] for n, c in zip(names[order[:k]], coef[order[:k]])],
    }


def run(db_path: Path, out_dir: Path, report_dir: Path, train_to: int, val_year: int,
        test_from: int, min_count: int = 50, target_recall: float = 0.90, seed: int = 0) -> Dict:
    started = time.time()
    conn = db.connect(db_path, read_only=True)
    examples = load_cases(conn)
    log.info("loaded %d labelled reports in %.0fs", len(examples), time.time() - started)

    rule_flags = np.array([rule_serious(e.case["reactions"], e.case["outcomes"]) for e in examples])
    years = np.array([e.year for e in examples])
    labels = np.array([e.label for e in examples])
    splits = {"train": years <= train_to, "validation": years == val_year, "test": years >= test_from}
    for name, mask in splits.items():
        if not (mask & ~rule_flags).any():
            raise SystemExit(f"split '{name}' is empty - check the year arguments")

    def subset(name: str) -> List[Example]:
        return [e for e, keep in zip(examples, splits[name] & ~rule_flags) if keep]

    data = {name: subset(name) for name in splits}
    feats = {name: [report_features(e.case) for e in rows] for name, rows in data.items()}
    y = {name: np.array([e.label for e in rows]) for name, rows in data.items()}
    vec = vectorize(feats["train"], min_count)
    x = {name: vec.transform(rows) for name, rows in feats.items()}
    log.info("features: %d (min_count=%d); rows train=%d val=%d test=%d", len(vec.feature_names_),
             min_count, x["train"].shape[0], x["validation"].shape[0], x["test"].shape[0])

    results, fitted = {}, {}
    for name, model in make_models(seed).items():
        t0 = time.time()
        model.fit(x["train"], y["train"])
        p_val = model.predict_proba(x["validation"])[:, 1]
        p_test = model.predict_proba(x["test"])[:, 1]
        threshold = pick_threshold(y["validation"], p_val, target_recall)
        results[name] = {
            "fit_seconds": round(time.time() - t0, 1),
            "validation": evaluate(y["validation"], p_val, threshold),
            "test": evaluate(y["test"], p_test, threshold),
        }
        fitted[name] = (model, threshold, p_val, p_test)
        log.info("%s: val PR-AUC %.3f ROC-AUC %.3f | test PR-AUC %.3f ROC-AUC %.3f (%.0fs)", name,
                 results[name]["validation"]["pr_auc"], results[name]["validation"]["roc_auc"],
                 results[name]["test"]["pr_auc"], results[name]["test"]["roc_auc"],
                 results[name]["fit_seconds"])

    best = max(results, key=lambda n: results[n]["validation"]["pr_auc"])  # chosen on validation only
    model, threshold, p_val, p_test = fitted[best]

    # Whole triage system on the test years: rule-flagged reports get score 1.0.
    test_all = splits["test"]
    sys_scores = np.ones(int(test_all.sum()))
    sys_scores[~rule_flags[test_all]] = p_test
    system = evaluate(labels[test_all], sys_scores, threshold)
    system["rule_routed_share"] = round(float(rule_flags[test_all].mean()), 4)
    system["rule_precision"] = round(float(labels[test_all & rule_flags].mean()), 4)

    prob_true, prob_pred = calibration_curve(y["test"], p_test, n_bins=10, strategy="quantile")
    cases = {name: [e.case for e in rows] for name, rows in data.items()}
    input_drift = drift.drift_report(cases["train"], cases["test"])
    score_psi = drift.psi_numeric(list(p_val), list(p_test))

    fingerprint = data_fingerprint(conn)
    trained_at = datetime.now(timezone.utc)
    version = f"{trained_at:%Y%m%d}-{fingerprint[:8]}"
    metadata = {
        "model_version": version,
        "model": best,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "threshold": round(threshold, 6),
        "target_recall": target_recall,
        "split": {"train": f"<= {train_to}", "validation": str(val_year), "test": f">= {test_from}"},
        "rows": {name: int(len(v)) for name, v in y.items()},
        "n_features": len(vec.feature_names_),
        "data_fingerprint": fingerprint,
        "partitions": conn.execute("SELECT COUNT(*) FROM partitions").fetchone()[0],
        "sklearn": sklearn.__version__,
        "python": sys.version.split()[0],
    }
    metrics = {
        "metadata": metadata,
        "models": results,
        "selected": best,
        "triage_system_test": system,
        "calibration_test": {"mean_predicted": [round(float(v), 4) for v in prob_pred],
                             "observed_rate": [round(float(v), 4) for v in prob_true]},
        "drift_train_vs_test": input_drift,
        "score_psi_validation_vs_test": round(score_psi, 4),
        "top_coefficients": top_coefficients(fitted["logistic_regression"][0], vec),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": model, "threshold": threshold, "metadata": metadata},
                out_dir / "model.joblib", compress=3)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (report_dir / "model_card.md").write_text(model_card(metrics))
    log.info("selected %s, version %s, total %.0fs", best, version, time.time() - started)
    return metrics


def model_card(m: Dict) -> str:
    md, sel = m["metadata"], m["selected"]
    lines = [
        "# Model card - adverse-event seriousness triage", "",
        f"*Generated by `ml/train.py` - version `{md['model_version']}`, trained {md['trained_at']}.*", "",
        "**Task.** Score incoming animal-drug adverse-event reports by the chance they are *serious*,",
        "so reviewers can open the riskiest ones first. Reports that already mention death or",
        "euthanasia are flagged by a rule and never reach the model.", "",
        "**Data.** openFDA animal & veterinary adverse events (public, unvalidated reports).",
        f"Split by receive year: train {md['split']['train']}, validation {md['split']['validation']},",
        f"test {md['split']['test']}. Rows: train {md['rows']['train']:,}, validation"
        f" {md['rows']['validation']:,}, test {md['rows']['test']:,}. Features: {md['n_features']:,}.", "",
        "## Model comparison (model population, rule cases excluded)", "",
        "| Model | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier | Test recall | Test precision | Flagged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in m["models"].items():
        v, t = r["validation"], r["test"]
        mark = " (selected)" if name == sel else ""
        lines.append(f"| {name}{mark} | {v['pr_auc']} | {v['roc_auc']} | {t['pr_auc']} | {t['roc_auc']} | "
                     f"{t['brier']} | {t['recall']} | {t['precision']} | {t['flagged_share']} |")
    s = m["triage_system_test"]
    lines += [
        "", f"Test prevalence of serious reports: {m['models'][sel]['test']['prevalence']}.",
        f"Threshold {md['threshold']:.4f} was set on validation to reach recall >= {md['target_recall']}.", "",
        "## Whole triage system on the test years (rule + model)", "",
        f"- Reports: {s['n']}, serious: {s['prevalence']}",
        f"- Routed by the death/euthanasia rule: {s['rule_routed_share']} (rule precision {s['rule_precision']})",
        f"- ROC-AUC {s['roc_auc']}, PR-AUC {s['pr_auc']}, recall {s['recall']}, precision {s['precision']},"
        f" flagged {s['flagged_share']}", "",
        "## Calibration on the test years (10 quantile bins)", "",
        "| Mean predicted | " + " | ".join(f"{v:.2f}" for v in m["calibration_test"]["mean_predicted"]) + " |",
        "|---" * (len(m["calibration_test"]["mean_predicted"]) + 1) + "|",
        "| Observed rate | " + " | ".join(f"{v:.2f}" for v in m["calibration_test"]["observed_rate"]) + " |", "",
        "## Drift (PSI, train vs test inputs)", "",
        "| Input | PSI | Status |", "|---|---|---|",
    ]
    lines += [f"| {d['feature']} | {d['psi']} | {d['status']} |" for d in m["drift_train_vs_test"]]
    lines += [f"| model score (val vs test) | {m['score_psi_validation_vs_test']} | "
              f"{drift.status(m['score_psi_validation_vs_test'])} |", "",
              "## Limitations", "",
              "- Spontaneous reports are unvalidated and under-reported; the label is the reporter's",
              "  seriousness flag, not a clinical review.",
              "- The model ranks reports for review. It says nothing about whether a drug caused an event.",
              "- Not for clinical or regulatory decisions.", ""]
    return "\n".join(lines)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/ads.db"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--train-to", type=int, default=2022)
    ap.add_argument("--val-year", type=int, default=2023)
    ap.add_argument("--test-from", type=int, default=2024)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--target-recall", type=float, default=0.90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.db, args.out, args.reports, args.train_to, args.val_year, args.test_from,
        args.min_count, args.target_recall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
