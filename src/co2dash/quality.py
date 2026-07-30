"""
Data-quality guards.

The whole point of co2dash is honest uncertainty, which matters MOST when the
input data is poor. This module inspects a training set (X, optional y) before
it reaches the surrogate and returns an explicit quality tier + warnings, so
low-quality data produces wide, honest uncertainty and clear caveats rather than
confident nonsense. It never silently "fixes" data — it reports.

Tiers:
  'ok'       — enough clean data for a meaningful calibrated fit
  'marginal' — usable but the surrogate/calibration will be shaky; widen caveats
  'poor'     — do not trust point predictions; treat outputs as exploratory only
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class QualityReport:
    n: int
    d: int
    tier: str
    warnings: List[str] = field(default_factory=list)
    n_over_d: float = 0.0
    constant_features: List[int] = field(default_factory=list)
    frac_missing: float = 0.0
    frac_outliers: float = 0.0

    @property
    def usable(self) -> bool:
        return self.tier != "poor"

    def summary(self) -> str:
        head = f"data quality: {self.tier.upper()} (n={self.n}, features={self.d}, n/d={self.n_over_d:.1f})"
        return head + ("" if not self.warnings else "\n  - " + "\n  - ".join(self.warnings))


def _robust_outlier_frac(A: np.ndarray, thresh: float = 5.0) -> float:
    """Fraction of entries beyond `thresh` robust z-scores (median/MAD) per column."""
    med = np.median(A, axis=0)
    mad = np.median(np.abs(A - med), axis=0)
    mad = np.where(mad < 1e-12, np.nan, mad)
    z = 0.6745 * np.abs(A - med) / mad
    z = np.nan_to_num(z, nan=0.0)
    return float(np.mean(z > thresh))


def data_quality_report(X, y=None, feature_names: Optional[Sequence[str]] = None) -> QualityReport:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    w: List[str] = []

    finite = np.isfinite(X)
    frac_missing = float(1.0 - finite.mean())
    if frac_missing > 0:
        w.append(f"{frac_missing:.0%} of feature entries are missing/non-finite (rows with any missing are dropped downstream)")

    Xc = X[np.all(finite, axis=1)] if frac_missing else X
    stds = Xc.std(axis=0) if len(Xc) else np.zeros(d)
    constant = [i for i, s in enumerate(stds) if s < 1e-9]
    if constant:
        names = ([feature_names[i] for i in constant] if feature_names else constant)
        w.append(f"{len(constant)} constant/near-constant feature(s) carry no information: {names} (will be dropped)")

    n_eff = len(Xc)
    ratio = n_eff / max(1, d)
    if n_eff < 15:
        w.append(f"very few usable rows (n={n_eff}); calibration coverage estimates are unreliable")
    elif n_eff < 30:
        w.append(f"few usable rows (n={n_eff}); train/cal/test split will be noisy")
    if ratio < 3:
        w.append(f"n/features = {ratio:.1f} < 3: high-dimensional regime, surrogate will over-rely on the prior/regulariser")

    frac_out = _robust_outlier_frac(Xc) if n_eff else 0.0
    if frac_out > 0.02:
        w.append(f"{frac_out:.0%} of feature entries are strong outliers (>5 robust-z); check for unit slips / bad DFT points")

    # near-duplicate rows (identical descriptors) — inflate apparent n
    if n_eff > 1:
        uniq = len({tuple(np.round(r, 6)) for r in Xc})
        if uniq < 0.7 * n_eff:
            w.append(f"only {uniq}/{n_eff} rows are distinct: many duplicate descriptor vectors (effective n is smaller)")

    if y is not None:
        y = np.asarray(y, float)
        if np.isfinite(y).mean() < 1.0:
            w.append("target has missing/non-finite values")
        yc = y[np.isfinite(y)]
        if len(yc) and yc.std() < 1e-9:
            w.append("target is (near-)constant: nothing to learn")

    # tier
    tier = "ok"
    if (n_eff < 30) or (ratio < 3) or (frac_out > 0.05):
        tier = "marginal"
    if (n_eff < 15) or (len(constant) == d) or (frac_missing > 0.3) or \
       (y is not None and len(np.atleast_1d(y)) and np.asarray(y, float)[np.isfinite(np.asarray(y, float))].std() < 1e-9):
        tier = "poor"

    return QualityReport(n=n, d=d, tier=tier, warnings=w, n_over_d=ratio,
                         constant_features=constant, frac_missing=frac_missing,
                         frac_outliers=frac_out)
