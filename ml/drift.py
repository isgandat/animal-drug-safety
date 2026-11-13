"""Population Stability Index (PSI) for monitoring input and score drift.

Rule of thumb: PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-4


def _psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(expected: Sequence[Optional[float]], actual: Sequence[Optional[float]],
                bins: int = 10) -> float:
    """Quantile bins from `expected`; missing values get their own bucket."""
    exp = np.array([v for v in expected if v is not None and not math.isnan(v)], dtype=float)
    act = np.array([v for v in actual if v is not None and not math.isnan(v)], dtype=float)
    if len(exp) == 0 or len(act) == 0:
        return 0.0
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([exp[0] - 0.5, exp[0] + 0.5])
    e_hist = np.histogram(np.clip(exp, edges[0], edges[-1]), edges)[0]
    a_hist = np.histogram(np.clip(act, edges[0], edges[-1]), edges)[0]
    e_missing = len(expected) - len(exp)
    a_missing = len(actual) - len(act)
    return _psi(np.append(e_hist, e_missing) / len(expected),
                np.append(a_hist, a_missing) / len(actual))


def psi_categorical(expected: Sequence[Optional[str]], actual: Sequence[Optional[str]]) -> float:
    e_counts = Counter(v or "missing" for v in expected)
    a_counts = Counter(v or "missing" for v in actual)
    keys = sorted(set(e_counts) | set(a_counts))
    return _psi([e_counts[k] / len(expected) for k in keys],
                [a_counts[k] / len(actual) for k in keys])


def status(value: float) -> str:
    return "stable" if value < 0.10 else "moderate" if value <= 0.25 else "major"


NUMERIC = ("age_years", "weight_kg", "n_drugs")
CATEGORICAL = ("species", "reporter", "type_info", "health_prior")


def drift_report(reference: List[Dict], current: List[Dict]) -> List[Dict]:
    """PSI per monitored input, comparing two lists of cases."""
    rows = []
    for key in NUMERIC:
        value = psi_numeric([c.get(key) for c in reference], [c.get(key) for c in current])
        rows.append({"feature": key, "psi": round(value, 4), "status": status(value)})
    for key in CATEGORICAL:
        value = psi_categorical([c.get(key) for c in reference], [c.get(key) for c in current])
        rows.append({"feature": key, "psi": round(value, 4), "status": status(value)})
    reactions = psi_numeric([len(c.get("reactions") or []) for c in reference],
                            [len(c.get("reactions") or []) for c in current])
    rows.append({"feature": "n_reactions", "psi": round(reactions, 4), "status": status(reactions)})
    return rows
