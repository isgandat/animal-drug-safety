"""Post-hoc probability calibration of the triage score.

The MLP ranks reports well but is over-confident: on the test years a predicted 0.86 meant an
observed 0.72. Platt scaling on the log-odds fixes the scale: a one-feature logistic regression
maps the model's logit to a calibrated probability. A positive slope makes the mapping strictly
increasing, so the ranking (PR-AUC, ROC-AUC) and the order of the review queue do not change.
Isotonic regression gave a lower ECE on this data but merges scores into ties, which cost PR-AUC.

Fit it on the validation year only; the test years stay untouched.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def _logit(p: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """calibrated = sigmoid(slope * logit(score) + intercept)."""

    def __init__(self) -> None:
        self.slope, self.intercept = 1.0, 0.0

    def fit(self, y: Sequence[int], p: Sequence[float]) -> "Calibrator":
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration needs both classes in the fitting data")
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        self.slope, self.intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if self.slope <= 0:
            raise ValueError(f"calibration slope {self.slope:.3f} <= 0: the scores do not rank the classes")
        return self

    def predict(self, p: Sequence[float]) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.slope * _logit(p) + self.intercept)))

    def to_dict(self) -> Dict:
        return {"method": "platt_logit", "slope": round(self.slope, 4), "intercept": round(self.intercept, 4)}


def reliability(y: Sequence[int], p: Sequence[float], n_bins: int = 10,
                by: Sequence[float] = None) -> Tuple[List[float], List[float], List[int]]:
    """Mean score, observed rate and size of equal-count bins, from the lowest scores up.

    `by` sorts the reports into bins (default: `p`), so raw and calibrated scores can share bins.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(np.asarray(p if by is None else by, dtype=float), kind="stable")
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    return ([round(float(p[b].mean()), 4) for b in bins],
            [round(float(y[b].mean()), 4) for b in bins],
            [int(len(b)) for b in bins])


def expected_calibration_error(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    """Size-weighted mean gap between the mean score and the observed rate per bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = [b for b in np.array_split(np.argsort(p, kind="stable"), n_bins) if len(b)]
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins))
