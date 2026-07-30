"""Tests for the data-quality guards."""
import numpy as np
from co2dash.quality import data_quality_report


def _good():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)); y = X @ rng.normal(size=5) + rng.normal(0, 0.1, 300)
    return X, y


def test_good_data_is_ok():
    r = data_quality_report(*_good())
    assert r.tier == "ok" and r.usable and not r.warnings


def test_constant_feature_flagged():
    X, y = _good(); X[:, 0] = 2.0
    r = data_quality_report(X, y)
    assert 0 in r.constant_features
    assert any("constant" in w for w in r.warnings)


def test_tiny_n_is_poor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 4)); y = rng.normal(size=10)
    r = data_quality_report(X, y)
    assert r.tier == "poor" and not r.usable


def test_constant_target_is_poor():
    X, y = _good(); y[:] = 0.5
    assert data_quality_report(X, y).tier == "poor"


def test_missing_reported():
    X, y = _good(); X[0, 0] = np.nan
    r = data_quality_report(X, y)
    assert r.frac_missing > 0 and any("missing" in w for w in r.warnings)
