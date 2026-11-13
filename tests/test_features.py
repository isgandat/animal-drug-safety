
from app.main import Report
from etl import db
from etl.run import load_file
from ml.features import load_cases, report_features, rule_serious
from tests.conftest import raw_record, write_zip


def test_rule_catches_death_and_euthanasia():
    assert rule_serious(["Vomiting", "Death by euthanasia"])
    assert rule_serious(["Found dead"])
    assert rule_serious(["Vomiting"], ["Euthanized"])
    assert not rule_serious(["Vomiting", "Lethargy"], ["Recovered/Normal", "Ongoing"])


def test_leakage_terms_and_outcomes_never_become_features():
    case = {"reactions": ["Vomiting", "Death", "Sudden death", "Unrelated death"],
            "outcomes": ["Died"]}
    names = report_features(case)
    assert "rx=Vomiting" in names
    assert not [n for n in names if "death" in n.lower() or "died" in n.lower()]
    assert report_features(case) == report_features({**case, "outcomes": ["Recovered/Normal"]})


def test_missing_values_get_explicit_indicators():
    f = report_features({})
    assert f["age_years_missing"] == 1.0 and f["weight_kg_missing"] == 1.0
    assert f["species=missing"] == 1.0 and f["treated=missing"] == 1.0


def test_training_and_api_build_identical_features(tmp_path):
    """Guards against train/serve skew: same report via SQLite and via the API schema."""
    rec = raw_record("A-1", serious_ae="true", reaction=[{"veddra_term_name": "Seizure NOS"},
                                                         {"veddra_term_name": "Ataxia"}])
    rec["drug"].append({"route": ["Subcutaneous", "Oral"], "administered_by": "Veterinarian",
                        "used_according_to_label": "false", "off_label_use": "Overdosed",
                        "atc_vet_code": "QP54AA01", "active_ingredients": [{"name": "Ivermectin"}]})
    conn = db.connect(tmp_path / "t.db")
    load_file(conn, "2023q1-0001", write_zip(tmp_path / "a.zip", [rec]))
    from_db = load_cases(conn)[0].case

    from_api = Report(
        species="Dog", gender="Female", repro_status="Neutered", health_prior="Good",
        reporter="Veterinarian", type_info="Safety Issue", age_years=5, weight_kg=20, treated=True,
        used_per_label=False, n_affected=1, n_drugs=2,
        routes=["Oral", "Subcutaneous"], administered_by=["Animal Owner", "Veterinarian"],
        off_label=["Overdosed"], atc_codes=["QM01AC06", "QP54AA01"],
        ingredients=["Meloxicam 1.5 mg", "Ivermectin"], reactions=["Seizure NOS", "Ataxia"],
    ).model_dump()
    assert report_features(from_db) == report_features(from_api)
