"""
Uncertainty calibration + conformal prediction (piece 3).

Sits BETWEEN surrogate.predict() and the Monte-Carlo propagation. A Bayesian
surrogate (or KAN/BNN) that is over/under-confident produces feasibility
probabilities that are confident and WRONG -- the worst failure for a decision
tool. This module:

  1. coverage_report  -- diagnose miscalibration on a held-out set.
  2. TemperatureScaler -- a single global scale s on the predictive std,
     fitted by Gaussian MLE: s^2 = mean[ (y - mu)^2 / sigma^2 ]. Cheap, often
     enough to fix systematic over/under-confidence.
  3. SplitConformal -- normalised split-conformal prediction. Distribution-free,
     finite-sample coverage guarantee: with calibration residuals
         r_i = |y_i - mu_i| / sigma_i,
     the (1-alpha) interval is mu +/- q * sigma, q = ceil((m+1)(1-alpha))/m
     empirical quantile of r. Also exposes `effective_std`, mapping the conformal
     half-width back to a Gaussian-equivalent std so the EXISTING Gaussian MC /
     active-learning machinery consumes calibrated uncertainty unchanged
     (documented approximation: matches interval width, not full shape).

The surrogate contract is preserved: a calibrator wraps any object exposing
predict(X) -> (mean, std).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Callable, Tuple
import numpy as np
from scipy.stats import norm


def coverage_report(mean: np.ndarray, std: np.ndarray, y_true: np.ndarray,
                    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Dict[float, float]:
    """Empirical coverage of central Gaussian intervals at each nominal level.
    coverage << level  => over-confident (intervals too narrow)."""
    mean, std, y_true = map(np.asarray, (mean, std, y_true))
    out = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2.0)
        out[float(lv)] = float(np.mean(np.abs(y_true - mean) <= z * std))
    return out


def miscalibration_area(mean, std, y_true,
                        levels=np.linspace(0.05, 0.95, 19)) -> float:
    """Mean absolute gap between empirical and nominal coverage. 0 = perfect."""
    cov = coverage_report(mean, std, y_true, levels)
    return float(np.mean([abs(cov[l] - l) for l in cov]))


@dataclass
class TemperatureScaler:
    """Global multiplicative correction on predictive std: sigma' = s * sigma."""
    s: float = 1.0

    def fit(self, mean, std, y_true) -> "TemperatureScaler":
        mean, std, y_true = map(np.asarray, (mean, std, y_true))
        std = np.maximum(std, 1e-12)
        self.s = float(np.sqrt(np.mean(((y_true - mean) / std) ** 2)))  # Gaussian MLE
        return self

    def transform(self, mean, std):
        return np.asarray(mean), self.s * np.asarray(std)


@dataclass
class SplitConformal:
    """Normalised split-conformal regressor. Stores calibration residuals."""
    residuals: np.ndarray = None  # type: ignore

    def fit(self, mean, std, y_true) -> "SplitConformal":
        mean, std, y_true = map(np.asarray, (mean, std, y_true))
        std = np.maximum(std, 1e-12)
        self.residuals = np.sort(np.abs(y_true - mean) / std)
        return self

    def _q(self, alpha: float) -> float:
        m = self.residuals.size
        # finite-sample conformal quantile level
        k = int(np.ceil((m + 1) * (1.0 - alpha)))
        k = min(max(k, 1), m)
        return float(self.residuals[k - 1])

    def interval(self, mean, std, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """(1-alpha) prediction interval [lo, hi] with finite-sample coverage."""
        mean, std = np.asarray(mean), np.asarray(std)
        q = self._q(alpha)
        return mean - q * std, mean + q * std

    def effective_std(self, std, alpha: float = 0.1) -> np.ndarray:
        """Gaussian-equivalent std whose central (1-alpha) interval matches the
        conformal half-width q*std. Lets the existing Gaussian MC consume
        calibrated uncertainty without changing its sampler. Approximation:
        matches interval width, not the full predictive shape."""
        q = self._q(alpha)
        z = norm.ppf(0.5 + (1.0 - alpha) / 2.0)
        return (q / z) * np.asarray(std)


class CalibratedSurrogate:
    """Wrap any surrogate with predict(X)->(mean,std), applying a fitted
    calibrator. Preserves the predict(X)->(mean,std) contract so everything
    downstream (MC, active learning) is unchanged."""
    def __init__(self, base_surrogate, scaler: TemperatureScaler | None = None,
                 conformal: SplitConformal | None = None, alpha: float = 0.1):
        self.base = base_surrogate
        self.scaler = scaler
        self.conformal = conformal
        self.alpha = alpha

    def predict(self, X):
        mean, std = self.base.predict(X)
        if self.scaler is not None:
            mean, std = self.scaler.transform(mean, std)
        if self.conformal is not None:
            std = self.conformal.effective_std(std, self.alpha)
        return mean, std
