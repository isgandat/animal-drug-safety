from __future__ import annotations

import json
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest

from etl import db
from etl.run import load_file

TRAIN_YEARS = list(range(2018, 2025))  # 2018-2022 train, 2023 validation, 2024 test


def raw_record(aer_id: str = "A-1", received: str = "20230115", **overrides) -> Dict:
    """One record in the openFDA animal-event JSON shape (synthetic values)."""
    record = {
        "unique_aer_id_number": aer_id,
        "report_id": "N000001",
        "original_receive_date": received,
        "primary_reporter": "Veterinarian",
        "type_of_information": "Safety Issue",
        "serious_ae": "false",
        "treated_for_ae": "true",
        "number_of_animals_affected": "1",
        "number_of_animals_treated": "1",
        "foreign_or_domestic": "Domestic",
        "health_assessment_prior_to_exposure": {"condition": "Good", "assessed_by": "Veterinarian"},
        "animal": {
            "species": "Dog", "gender": "Female", "reproductive_status": "Neutered",
            "breed": {"is_crossbred": "false"},
            "age": {"min": "5.00", "unit": "Year", "qualifier": "Measured"},
            "weight": {"min": "20.000", "unit": "Kilogram", "qualifier": "Measured"},
        },
        "drug": [{
            "route": "Oral", "administered_by": "Animal Owner", "used_according_to_label": "true",
            "previous_exposure_to_drug": "false", "dosage_form": "Tablet", "atc_vet_code": "QM01AC06",
            "active_ingredients": [{"name": "Meloxicam 1.5 mg"}],
        }],
        "reaction": [{"veddra_version": "11", "veddra_term_code": "1", "veddra_term_name": "Vomiting"}],
        "outcome": [{"medical_status": "Recovered/Normal", "number_of_animals_affected": "1"}],
    }
    record.update(overrides)
    return record


def write_zip(path: Path, records: List[Dict]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("animalandveterinary-event-0001-of-0001.json",
                    json.dumps({"meta": {}, "results": records}))
    return path


def synthetic_year(year: int, n: int, seed: int = 7) -> List[Dict]:
    """Records whose seriousness depends on known inputs, so a model can learn it."""
    rng = random.Random(seed * 10000 + year)
    records = []
    for i in range(n):
        seizure, iv, old, no_effect = (rng.random() < 0.3, rng.random() < 0.25,
                                       rng.random() < 0.4, rng.random() < 0.3)
        logit = -1.0 + 2.5 * seizure + 1.5 * iv + 1.0 * old - 1.5 * no_effect
        serious = rng.random() < 1 / (1 + math.exp(-logit))
        reactions = ["Seizure NOS" if seizure else "Vomiting"]
        if no_effect:
            reactions.append("Lack of efficacy - NOS")
        if rng.random() < 0.05:
            reactions.append("Death")
            serious = True
        drug = {
            "route": "Intravenous" if iv else "Oral", "administered_by": "Veterinarian",
            "used_according_to_label": "true", "atc_vet_code": "QP54AB02" if i % 2 else "QM01AC06",
            "active_ingredients": [{"name": "Ivermectin" if i % 2 else "Meloxicam"}],
        }
        records.append(raw_record(
            f"SYN-{year}-{i:05d}", f"{year}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
            serious_ae=None if i % 25 == 0 else ("true" if serious else "false"),
            animal={"species": "Cat" if i % 3 == 0 else "Dog",
                    "age": {"min": str(12 if old else 3), "unit": "Year"},
                    "weight": {"min": "10", "unit": "Kilogram"}},
            drug=[drug],
            reaction=[{"veddra_term_name": t} for t in reactions],
        ))
    return records


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    conn = db.connect(root / "ads.db")
    for year in TRAIN_YEARS:
        path = write_zip(root / f"{year}q1-0001.json.zip", synthetic_year(year, 400))
        load_file(conn, f"{year}q1-0001", path)
    conn.close()
    return root / "ads.db"


@pytest.fixture(scope="session")
def trained(synthetic_db, tmp_path_factory) -> Dict:
    from ml.train import run

    out = tmp_path_factory.mktemp("model")
    metrics = run(synthetic_db, out / "models", out / "reports", train_to=2022, val_year=2023,
                  test_from=2024, min_count=5)
    return {"metrics": metrics, "model": out / "models" / "model.joblib", "reports": out / "reports",
            "db": synthetic_db}
