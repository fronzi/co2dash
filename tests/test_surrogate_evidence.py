"""Tests for hyperparameter estimation in the Bayesian surrogate.

The point of these tests: a hard-coded noise precision `beta` puts a constant
floor beta^-1/2 under every predictive std. When that floor dominates, the std
stops discriminating between points and any ranking built on it (active
learning / EVOI) degenerates. `fit_evidence()` must remove that failure mode.
"""
import numpy as np
import pytest

from co2dash.surrogate import BayesianLinearSurrogate, cv_noise_precision


def _linear_data(n=300, d=4, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + rng.normal(0.0, sigma, n)
    return X, y, sigma


# --------------------------------------------------------------- evidence fit
def test_fit_evidence_recovers_the_true_noise_scale():
    X, y, sigma = _linear_data(sigma=0.05)
    m = BayesianLinearSurrogate(beta=50.0).fit_evidence(X, y)
    sigma_hat = 1.0 / np.sqrt(m.beta)
    assert sigma_hat == pytest.approx(sigma, rel=0.25)


def test_fit_evidence_corrects_a_badly_wrong_starting_beta():
    X, y, sigma = _linear_data(sigma=0.05)
    # beta=50 implies sigma=0.141, ~3x too large for this data
    naive = BayesianLinearSurrogate(beta=50.0).fit(X, y)
    tuned = BayesianLinearSurrogate(beta=50.0).fit_evidence(X, y)
    err_naive = abs(1.0 / np.sqrt(naive.beta) - sigma)
    err_tuned = abs(1.0 / np.sqrt(tuned.beta) - sigma)
    assert err_tuned < err_naive


def test_fit_evidence_is_insensitive_to_the_initial_hyperparameters():
    X, y, _ = _linear_data()
    a = BayesianLinearSurrogate(alpha=1e-4, beta=1.0).fit_evidence(X, y)
    b = BayesianLinearSurrogate(alpha=1e2, beta=5e3).fit_evidence(X, y)
    assert a.beta == pytest.approx(b.beta, rel=0.05)


def test_fit_evidence_keeps_the_predict_contract():
    X, y, _ = _linear_data()
    m = BayesianLinearSurrogate().fit_evidence(X, y)
    mean, std = m.predict(X[:10])
    assert mean.shape == (10,) and std.shape == (10,)
    assert np.all(std > 0)


def test_overconfident_intervals_become_calibrated():
    """A too-small beta (too-wide sigma) over-covers; the evidence fit should
    bring 95% coverage back toward nominal."""
    X, y, _ = _linear_data(sigma=0.05)
    naive = BayesianLinearSurrogate(beta=50.0).fit(X, y)
    tuned = BayesianLinearSurrogate(beta=50.0).fit_evidence(X, y)

    def coverage(model):
        mu, sd = model.predict(X)
        return float(np.mean(np.abs(y - mu) <= 1.96 * sd))

    assert coverage(naive) > 0.99                       # over-wide
    assert abs(coverage(tuned) - 0.95) < abs(coverage(naive) - 0.95)


def test_relative_ranking_by_uncertainty_survives_a_wrong_beta():
    """Documents what a wrong beta does NOT break.

    The epistemic term phi^T S phi with S = (alpha I + beta Phi^T Phi)^-1 scales
    with beta too, so the ORDER of points by predictive std is largely
    beta-invariant. Any active-learning ranking that only uses the order is
    therefore robust. Recorded as a test so the claim is not re-asserted
    incorrectly later.
    """
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(0, 1, (60, 3)) * 6.0])
    y = X @ np.array([1.0, -0.5, 0.25]) + rng.normal(0, 0.02, 120)

    _, sd_naive = BayesianLinearSurrogate(beta=50.0).fit(X, y).predict(X)
    _, sd_tuned = BayesianLinearSurrogate(beta=50.0).fit_evidence(X, y).predict(X)

    rank = lambda v: np.argsort(np.argsort(v))               # noqa: E731
    corr = np.corrcoef(rank(sd_naive), rank(sd_tuned))[0, 1]
    assert corr > 0.9


def test_absolute_uncertainty_scale_is_what_a_wrong_beta_breaks():
    """What a wrong beta DOES break: the size of the error bar in eV.

    Every decision with an absolute threshold depends on it -- conformal
    intervals, 'compute next if sigma > 0.1 eV', and propagating the surrogate's
    sigma into the Monte-Carlo as a cell-voltage uncertainty.
    """
    X, y, sigma = _linear_data(n=250, sigma=0.02)
    naive = BayesianLinearSurrogate(beta=50.0).fit(X, y)
    tuned = BayesianLinearSurrogate(beta=50.0).fit_evidence(X, y)

    def scale_error(model):
        mu, sd = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
        return float(np.mean(sd)) / rmse                     # 1.0 = honest

    assert scale_error(naive) > 3.0                          # overstates by >3x
    assert 0.5 < scale_error(tuned) < 2.0                    # right order of magnitude


def test_rank_deficient_design_does_not_blow_up():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    X = np.hstack([X, X[:, :1]])                             # exactly collinear column
    y = X[:, 0] * 2.0
    m = BayesianLinearSurrogate().fit_evidence(X, y)
    mean, std = m.predict(X)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    assert np.isfinite(m.alpha) and np.isfinite(m.beta) and m.beta > 0


# --------------------------------------------------------------- CV cross-check
def test_cv_noise_precision_agrees_with_the_truth_on_a_well_specified_model():
    X, y, sigma = _linear_data(n=400, sigma=0.05)
    rep = cv_noise_precision(X, y, n_splits=5,
                             factory=lambda: BayesianLinearSurrogate().__class__())
    assert rep["sigma_eV"] == pytest.approx(sigma, rel=0.5)
    assert rep["n"] == 400 and rep["n_splits"] == 5
    assert 0.0 <= rep["coverage_95"] <= 1.0


def test_cv_noise_precision_reports_the_diagnostics_needed_to_judge_it():
    X, y, _ = _linear_data(n=120)
    rep = cv_noise_precision(X, y)
    assert set(rep) >= {"beta", "sigma_eV", "cv_rmse", "mean_epistemic_var",
                        "coverage_95", "n", "n_splits"}
    assert rep["beta"] > 0 and rep["cv_rmse"] > 0


def test_cv_noise_precision_refuses_to_run_on_too_few_samples():
    X, y, _ = _linear_data(n=6)
    with pytest.raises(ValueError, match="at least"):
        cv_noise_precision(X, y, n_splits=5)
