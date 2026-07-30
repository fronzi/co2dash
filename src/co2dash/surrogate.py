"""
Bayesian surrogate.

Maps cheap, publicly-available DFT descriptors (d-band centre, adsorption
energies of *CO, *COOH, *OCHO from Catalysis-Hub / OC20) to a performance KPI
(e.g. faradaic efficiency) WITH a predictive variance.

Implemented as closed-form Bayesian linear regression to keep the scaffold
dependency-light and fully reproducible. In production this slot is occupied by
your KAN/BNN: the only contract the rest of the platform needs is
    predict(X) -> (mean, std).
Swapping in a BNN changes nothing downstream.

Posterior (Bishop, PRML 3.3):
    S = (alpha I + beta Phi^T Phi)^-1
    m = beta S Phi^T t
    predictive mean  = phi(x)^T m
    predictive var   = beta^-1 + phi(x)^T S phi(x)
"""
from __future__ import annotations
import numpy as np


class BayesianLinearSurrogate:
    def __init__(self, alpha: float = 1e-2, beta: float = 50.0, degree: int = 1):
        self.alpha = alpha          # prior precision on weights
        self.beta = beta            # noise precision (1/sigma^2 of observations)
        self.degree = degree
        self.m = None
        self.S = None
        self._mu = None
        self._sd = None

    # simple polynomial feature map + standardisation
    def _phi(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self._mu is None:
            self._mu, self._sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self._mu) / self._sd
        feats = [np.ones((Xs.shape[0], 1)), Xs]
        for d in range(2, self.degree + 1):
            feats.append(Xs ** d)
        return np.hstack(feats)

    def fit(self, X, y):
        Phi = self._phi(X)
        y = np.asarray(y, float)
        D = Phi.shape[1]
        self.S = np.linalg.inv(self.alpha * np.eye(D) + self.beta * Phi.T @ Phi)
        self.m = self.beta * self.S @ Phi.T @ y
        return self

    def predict(self, X):
        """Returns (mean, std) of the predictive distribution."""
        Phi = self._phi(X)
        mean = Phi @ self.m
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        return mean, np.sqrt(np.maximum(var, 0.0))
