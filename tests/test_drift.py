import random

from ml.drift import drift_report, psi_categorical, psi_numeric, status


def test_identical_distributions_have_no_drift():
    values = [random.Random(1).gauss(0, 1) for _ in range(5000)]
    assert psi_numeric(values, values) < 1e-9
    assert psi_categorical(["Dog", "Cat"] * 50, ["Cat", "Dog"] * 50) < 1e-9


def test_shifted_distribution_is_major_drift():
    rng = random.Random(2)
    before = [rng.gauss(0, 1) for _ in range(5000)]
    after = [rng.gauss(1.5, 1) for _ in range(5000)]
    assert status(psi_numeric(before, after)) == "major"


def test_missing_values_and_new_categories_count_as_drift():
    assert psi_numeric([1.0, 2.0, 3.0, 4.0] * 50, [1.0, 2.0, None, None] * 50) > 0.25
    assert psi_categorical(["Dog"] * 100, ["Dog"] * 50 + ["Horse"] * 50) > 0.25


def test_report_lists_every_monitored_input():
    cases = [{"age_years": 3, "species": "Dog", "reactions": ["Vomiting"]}] * 20
    names = [row["feature"] for row in drift_report(cases, cases)]
    assert names == ["age_years", "weight_kg", "n_drugs", "species", "reporter", "type_info",
                     "health_prior", "n_reactions"]
