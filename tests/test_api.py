import pytest
from fastapi.testclient import TestClient

from app.main import app
from etl import db
from stats.signals import compute, save

REPORT = {"species": "Dog", "age_years": 12, "routes": ["Intravenous"],
          "ingredients": ["Meloxicam"], "reactions": ["Seizure NOS"]}


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("ADS_MODEL", str(trained["model"]))
    monkeypatch.setenv("ADS_DB", str(trained["db"]))
    return TestClient(app)


def test_health_reports_the_model_version(client, trained):
    body = client.get("/health").json()
    assert body["model_loaded"] and body["database"]
    assert body["model_version"] == trained["metrics"]["metadata"]["model_version"]


def test_triage_scores_with_the_model(client):
    risky = client.post("/triage", json=REPORT).json()
    mild = client.post("/triage", json={**REPORT, "age_years": 3, "routes": ["Oral"],
                                        "reactions": ["Vomiting", "Lack of efficacy - NOS"]}).json()
    assert risky["route"] == "model" and 0 <= risky["serious_probability"] <= 1
    assert risky["serious_probability"] > mild["serious_probability"]
    assert risky["flagged"] == (risky["serious_probability"] >= risky["threshold"])


def test_death_reports_take_the_rule_route(client):
    body = client.post("/triage", json={**REPORT, "reactions": ["Death"]}).json()
    assert body == {**body, "serious_probability": 1.0, "flagged": True}
    assert body["route"].startswith("rule")


def test_invalid_input_is_rejected(client):
    assert client.post("/triage", json={"age_years": -1}).status_code == 422
    assert client.post("/triage", json={"reactions": "Vomiting"}).status_code == 422


def test_missing_model_gives_503_but_rule_still_works(monkeypatch, tmp_path):
    monkeypatch.setenv("ADS_MODEL", str(tmp_path / "none.joblib"))
    monkeypatch.setenv("ADS_DB", str(tmp_path / "none.db"))
    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"] is False
    assert c.post("/triage", json=REPORT).status_code == 503
    assert c.post("/triage", json={"reactions": ["Death"]}).status_code == 200
    assert c.get("/stats").status_code == 503


def test_signals_and_stats(client, trained, tmp_path):
    assert client.get("/signals", params={"ingredient": "meloxicam"}).status_code in (200, 503)
    conn = db.connect(trained["db"])
    save(conn, compute(conn, min_a=3), "species=all;from=all")
    conn.close()
    body = client.get("/signals", params={"ingredient": "Meloxicam", "only_flagged": False}).json()
    assert body["ingredient"] == "meloxicam" and body["results"]
    assert {"term", "a", "prr", "ror", "chi2", "signal"} <= set(body["results"][0])
    years = [row["year"] for row in client.get("/stats").json()["by_year"]]
    assert years == list(range(2018, 2025))
