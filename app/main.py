"""REST API: triage score for a report, signal lookup and dataset stats.

    uvicorn app.main:app --port 8000
Settings: ADS_MODEL (default models/model.joblib), ADS_DB (default data/ads.db).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from etl import db
from ml.features import report_features, rule_serious

app = FastAPI(title="Animal drug safety", version="0.1.0",
              description="Triage of openFDA animal adverse-event reports. Not for clinical use.")


def _model_path() -> Path:
    return Path(os.environ.get("ADS_MODEL", "models/model.joblib"))


def _db_path() -> Path:
    return Path(os.environ.get("ADS_DB", "data/ads.db"))


@lru_cache(maxsize=4)
def _load_model(path: str, mtime: float) -> Dict:  # mtime in the key -> reload after retraining
    return joblib.load(path)


def model_bundle() -> Optional[Dict]:
    path = _model_path()
    if not path.exists():
        return None
    return _load_model(str(path), path.stat().st_mtime)


def _connect():
    path = _db_path()
    if not path.exists():
        raise HTTPException(503, "database not built yet - run `python -m etl.run`")
    return db.connect(path, read_only=True)


class Report(BaseModel):
    species: Optional[str] = Field(None, examples=["Dog"])
    gender: Optional[str] = None
    repro_status: Optional[str] = None
    health_prior: Optional[str] = Field(None, examples=["Good"])
    reporter: Optional[str] = Field(None, examples=["Veterinarian"])
    type_info: Optional[str] = Field(None, examples=["Safety Issue"])
    age_years: Optional[float] = Field(None, ge=0, le=60)
    weight_kg: Optional[float] = Field(None, gt=0, le=3000)
    treated: Optional[bool] = None
    used_per_label: Optional[bool] = Field(None, description="False if any drug was used off label")
    n_affected: Optional[int] = Field(None, ge=0)
    n_drugs: int = Field(1, ge=0, le=50)
    routes: List[str] = Field(default_factory=list, examples=[["Oral"]])
    administered_by: List[str] = Field(default_factory=list)
    off_label: List[str] = Field(default_factory=list)
    atc_codes: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(default_factory=list, examples=[["Meloxicam"]])
    reactions: List[str] = Field(default_factory=list, max_length=200, examples=[["Vomiting", "Lethargy"]])
    outcomes: List[str] = Field(default_factory=list, description="Only used by the death/euthanasia rule")


class Triage(BaseModel):
    serious_probability: float
    flagged: bool
    route: str
    threshold: Optional[float]
    model_version: Optional[str]


@app.get("/health")
def health() -> Dict:
    bundle = model_bundle()
    return {"status": "ok",
            "model_loaded": bundle is not None,
            "model_version": bundle["metadata"]["model_version"] if bundle else None,
            "database": _db_path().exists()}


@app.post("/triage", response_model=Triage)
def triage(report: Report) -> Triage:
    case = report.model_dump()
    bundle = model_bundle()
    version = bundle["metadata"]["model_version"] if bundle else None
    if rule_serious(case["reactions"], case["outcomes"]):
        return Triage(serious_probability=1.0, flagged=True, route="rule: death or euthanasia reported",
                      threshold=None, model_version=version)
    if bundle is None:
        raise HTTPException(503, "model not trained yet - run `python -m ml.train`")
    x = bundle["vectorizer"].transform([report_features(case)])
    p = float(bundle["model"].predict_proba(x)[0, 1])
    return Triage(serious_probability=round(p, 4), flagged=p >= bundle["threshold"], route="model",
                  threshold=round(bundle["threshold"], 4), model_version=version)


@app.get("/signals")
def signals(ingredient: str = Query(..., min_length=3, max_length=80),
            scope: str = "species=all;from=all",
            only_flagged: bool = True,
            limit: int = Query(20, ge=1, le=200)) -> Dict:
    conn = _connect()
    try:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE name = 'signals'").fetchone()
        if not exists:
            raise HTTPException(503, "signals not computed yet - run `python -m stats.signals`")
        rows = conn.execute(
            "SELECT term, a, prr, ror, ror_lo, ror_hi, chi2, signal FROM signals "
            "WHERE scope = ? AND ingredient = ? AND (signal = 1 OR ? = 0) "
            "ORDER BY signal DESC, prr DESC LIMIT ?",
            (scope, ingredient.strip().lower(), int(only_flagged), limit)).fetchall()
    finally:
        conn.close()
    return {"ingredient": ingredient.strip().lower(), "scope": scope,
            "note": "Disproportionality is a screening statistic, not evidence of causation.",
            "results": [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in r.keys()}
                        for r in rows]}


@app.get("/stats")
def stats() -> Dict:
    conn = _connect()
    try:
        by_year = conn.execute("SELECT receive_year AS year, COUNT(*) AS reports, "
                               "ROUND(AVG(serious), 3) AS serious_rate FROM reports "
                               "GROUP BY receive_year ORDER BY receive_year").fetchall()
        species = conn.execute("SELECT COALESCE(species, 'missing') AS species, COUNT(*) AS reports "
                               "FROM reports GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchall()
    finally:
        conn.close()
    return {"by_year": [dict(r) for r in by_year], "top_species": [dict(r) for r in species]}
