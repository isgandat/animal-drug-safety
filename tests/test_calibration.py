import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from ml.calibration import Calibrator, expected_calibration_error, reliability


def overconfident(n: int, seed: int):
    """Labels drawn from known probabilities, scored by a model that pushes them towards 0 and 1."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < true_p).astype(int)
    raw = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    return y, raw


def test_ece_is_zero_when_scores_match_the_observed_rate():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_ece_is_the_gap_between_predicted_and_observed():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_reliability_uses_equal_count_bins():
    y = np.array([0] * 50 + [1] * 50)
    p = np.linspace(0, 1, 100)
    predicted, observed, counts = reliability(y, p, n_bins=4)
    assert counts == [25, 25, 25, 25]
    assert observed == [0.0, 0.0, 1.0, 1.0]
    assert predicted == sorted(predicted)


def test_calibration_fixes_an_overconfident_model_on_new_data():
    y_fit, p_fit = overconfident(20000, seed=1)
    y_new, p_new = overconfident(20000, seed=2)
    cal = Calibrator().fit(y_fit, p_fit)
    fixed = cal.predict(p_new)
    assert expected_calibration_error(y_new, fixed) < expected_calibration_error(y_new, p_new) / 3
    assert brier_score_loss(y_new, fixed) < brier_score_loss(y_new, p_new)
    assert cal.slope == pytest.approx(1 / 3, abs=0.05)  # undoes the factor 3 in the log-odds


def test_calibration_keeps_the_ranking():
    y, p = overconfident(5000, seed=3)
    cal = Calibrator().fit(y, p)
    assert np.all(np.diff(cal.predict(np.sort(p))) > 0)
    assert average_precision_score(y, cal.predict(p)) == pytest.approx(average_precision_score(y, p))


def test_calibrated_scores_stay_probabilities_for_extreme_inputs():
    cal = Calibrator().fit(*overconfident(2000, seed=4))
    q = cal.predict(np.array([0.0, 1e-12, 0.5, 1 - 1e-12, 1.0]))
    assert np.all((q > 0) & (q < 1))


def test_calibration_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        Calibrator().fit(np.zeros(10), np.linspace(0.1, 0.9, 10))


def test_calibration_refuses_scores_that_rank_backwards():
    y, p = overconfident(2000, seed=5)
    with pytest.raises(ValueError, match="slope"):
        Calibrator().fit(y, 1 - p)
