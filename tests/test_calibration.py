"""Piece 3 tests: calibration + conformal prediction."""
import numpy as np
from co2dash import (coverage_report, miscalibration_area, TemperatureScaler,
                     SplitConformal, CalibratedSurrogate)


def _overconfident_predictions(n=4000, seed=0):
    """Synthetic over-confident model: true noise sigma=1.0 but the model reports
    sigma=0.4 -> central intervals are too narrow -> coverage below nominal."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 2, n)
    y_true = mean + rng.normal(0, 1.0, n)     # real spread 1.0
    std_reported = np.full(n, 0.4)            # claimed spread 0.4 (too small)
    return mean, std_reported, y_true


def test_overconfident_model_is_undercovered():
    mean, std, y = _overconfident_predictions()
    cov = coverage_report(mean, std, y, levels=(0.9,))
    assert cov[0.9] < 0.9        # under-covered, as constructed


def test_temperature_scaling_reduces_miscalibration():
    mean, std, y = _overconfident_predictions()
    before = miscalibration_area(mean, std, y)
    ts = TemperatureScaler().fit(mean, std, y)
    m2, s2 = ts.transform(mean, std)
    after = miscalibration_area(m2, s2, y)
    assert ts.s > 1.0                 # must inflate the under-confident std
    assert after < before             # calibration improved
    assert after < 0.05               # and is now close to nominal


def test_split_conformal_achieves_target_coverage():
    mean, std, y = _overconfident_predictions(n=6000, seed=2)
    # split: half to calibrate, half to test
    cut = mean.size // 2
    conf = SplitConformal().fit(mean[:cut], std[:cut], y[:cut])
    lo, hi = conf.interval(mean[cut:], std[cut:], alpha=0.1)
    cov = float(np.mean((y[cut:] >= lo) & (y[cut:] <= hi)))
    assert abs(cov - 0.9) < 0.03      # ~90% coverage, finite-sample

    # effective_std maps the conformal width back to a Gaussian-equivalent std
    eff = conf.effective_std(std[cut:], alpha=0.1)
    assert np.all(eff > std[cut:])    # under-confident model -> widened


def test_calibrated_surrogate_preserves_contract():
    class Dummy:
        def predict(self, X):
            X = np.atleast_2d(X)
            return np.zeros(X.shape[0]), np.full(X.shape[0], 0.4)
    mean, std, y = _overconfident_predictions()
    ts = TemperatureScaler().fit(mean, std, y)
    cs = CalibratedSurrogate(Dummy(), scaler=ts)
    m, s = cs.predict(np.zeros((5, 2)))
    assert m.shape == (5,) and s.shape == (5,)
    assert np.all(s > 0.4)            # uncertainty was inflated by calibration
