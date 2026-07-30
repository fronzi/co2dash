"""
Calibration harness.

The calibration *gate*: given a surrogate factory and (X, y) data, split into
train / calibration / test, fit the surrogate on train, fit temperature scaling
and split-conformal on the held-out calibration set, and evaluate coverage on
the untouched test set. Returns a before/after report.

Why a synthetic generator lives here: to VALIDATE the calibration procedure
itself you need known ground truth, so you can check that empirical coverage
matches the nominal level after calibration. `make_linear_synthetic` +
`ConstStdSurrogate` build a controllable, deliberately mis-calibrated model for
exactly this purpose. They are clearly synthetic and used only to prove the
harness works; real calibration passes real (descriptor, experimental-FE) data
through the same `calibrate_and_evaluate`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
import numpy as np

from .calibration import (coverage_report, miscalibration_area,
                          TemperatureScaler, SplitConformal)
from .surrogate import BayesianLinearSurrogate

LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


# ------------------------------------------------------------------ splitting
def split_indices(n: int, frac_train=0.5, frac_cal=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_cal = int(frac_train * n), int(frac_cal * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]


# ------------------------------------------------------------------ harness
@dataclass
class CalibrationReport:
    temperature_s: float
    miscal_before: float
    miscal_after: float
    coverage_before: Dict[float, float]
    coverage_after: Dict[float, float]
    conformal_alpha: float
    conformal_coverage: float          # empirical coverage of the (1-alpha) conformal interval on test
    n_train: int
    n_cal: int
    n_test: int
    levels: Sequence[float] = field(default_factory=lambda: LEVELS)

    @property
    def improved(self) -> bool:
        return self.miscal_after <= self.miscal_before

    def summary(self) -> str:
        return (f"temperature s={self.temperature_s:.3f}; miscalibration "
                f"{self.miscal_before:.3f} -> {self.miscal_after:.3f}; "
                f"conformal {1-self.conformal_alpha:.0%} coverage on test = "
                f"{self.conformal_coverage:.0%} "
                f"(n_train={self.n_train}, n_cal={self.n_cal}, n_test={self.n_test})")


def calibrate_and_evaluate(X, y,
                           surrogate_factory: Callable = None,
                           frac_train=0.5, frac_cal=0.25, alpha=0.1,
                           levels: Sequence[float] = LEVELS, seed=0) -> CalibrationReport:
    """Fit on train, calibrate on cal, evaluate on test. Surrogate factory takes
    (X_train, y_train) and returns an object with predict(X)->(mean,std)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if surrogate_factory is None:
        def surrogate_factory(Xt, yt):
            s = BayesianLinearSurrogate(); s.fit(Xt, yt); return s

    itr, ical, ite = split_indices(len(y), frac_train, frac_cal, seed)
    surr = surrogate_factory(X[itr], y[itr])

    m_cal, s_cal = surr.predict(X[ical])
    m_te, s_te = surr.predict(X[ite])

    # before calibration (raw surrogate uncertainty on test)
    cov_before = coverage_report(m_te, s_te, y[ite], levels)
    mis_before = miscalibration_area(m_te, s_te, y[ite], np.asarray(levels))

    # fit temperature scaling + conformal on the held-out calibration set
    scaler = TemperatureScaler().fit(m_cal, s_cal, y[ical])
    conf = SplitConformal().fit(*scaler.transform(m_cal, s_cal), y[ical])

    m_te2, s_te2 = scaler.transform(m_te, s_te)
    cov_after = coverage_report(m_te2, s_te2, y[ite], levels)
    mis_after = miscalibration_area(m_te2, s_te2, y[ite], np.asarray(levels))

    lo, hi = conf.interval(m_te2, s_te2, alpha=alpha)
    conf_cov = float(np.mean((y[ite] >= lo) & (y[ite] <= hi)))

    return CalibrationReport(
        temperature_s=scaler.s, miscal_before=mis_before, miscal_after=mis_after,
        coverage_before=cov_before, coverage_after=cov_after,
        conformal_alpha=alpha, conformal_coverage=conf_cov,
        n_train=len(itr), n_cal=len(ical), n_test=len(ite), levels=levels)


# ------------------------------------------------------------------ synthetic (validation only)
def make_linear_synthetic(n=400, d=4, sigma_true=0.05, seed=0):
    """Ground-truth linear target y = Xw + N(0, sigma_true^2). Clearly synthetic;
    used only to validate the calibration procedure (needs known noise)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + rng.normal(0.0, sigma_true, size=n)
    return X, y, w


class ConstStdSurrogate:
    """A surrogate whose MEAN is a least-squares fit but whose reported STD is a
    fixed value chosen to be deliberately wrong (over/under-confident). Lets us
    verify temperature scaling recovers the true noise scale. Synthetic tool."""
    def __init__(self, reported_std: float):
        self.reported_std = float(reported_std)
        self.w = None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        mean = X @ self.w
        return mean, np.full(mean.shape, self.reported_std)


# ------------------------------------------------------------------ real-data join
def join_labeled(descriptors: Dict[str, Sequence[float]],
                 targets: Dict[str, float], keys: Sequence[str]):
    """Join real DFT descriptors {material_id: vector} to real experimental
    targets {material_id: FE} on shared ids, using `keys` as the descriptor
    order. Returns (X, y, ids). No imputation: ids missing either side are
    dropped. This is the real-data entry point for the calibration gate."""
    ids = [i for i in descriptors if i in targets]
    if not ids:
        raise ValueError("no overlapping material_ids between descriptors and targets")
    X = np.array([[descriptors[i][k] if isinstance(descriptors[i], dict)
                   else descriptors[i][j] for j, k in enumerate(keys)] for i in ids], float)
    y = np.array([targets[i] for i in ids], float)
    return X, y, ids
