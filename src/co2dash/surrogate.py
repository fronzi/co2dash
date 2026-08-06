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

CHOOSING alpha AND beta
-----------------------
`fit()` uses whatever (alpha, beta) you constructed the object with. Those
defaults are arbitrary, and beta is load-bearing: it fixes the aleatoric floor
at beta^-1/2, which is added to EVERY predictive variance.

What a wrong beta does and does not break:
  * ORDER of points by predictive std -- largely UNAFFECTED. S = (alpha I +
    beta Phi^T Phi)^-1 scales with beta as well, so the epistemic term shrinks
    as the floor grows and the ranking is roughly preserved. Active-learning
    selection that only uses the order is robust. (test_surrogate_evidence.py
    pins this down.)
  * ABSOLUTE size of the error bar -- BROKEN, and that is what most decisions
    actually consume: conformal/calibrated intervals, any 'compute it next if
    sigma > 0.1 eV' rule, and propagating the surrogate sigma into the
    Monte-Carlo as a cell-voltage uncertainty. With beta=50 the floor is
    0.141 eV, which exceeds the measured CV RMSE on all three HEA sheets --
    i.e. the reported uncertainty is a constructor default, not a result.

`fit_evidence()` instead estimates both from the data by type-II maximum
likelihood (evidence maximisation, Bishop PRML 3.5.2):

    gamma  = sum_i  beta*lambda_i / (alpha + beta*lambda_i)      (effective #params)
    alpha <- gamma / (m^T m)
    beta  <- (N - gamma) / ||t - Phi m||^2

iterated to convergence, with lambda_i the eigenvalues of Phi^T Phi. Use it
unless you have a calibrated prior you actually believe.
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

    def fit_evidence(self, X, y, n_iter: int = 200, tol: float = 1e-6):
        """Fit weights AND the hyperparameters (alpha, beta) by evidence
        maximisation. Same contract as fit(): returns self, predict() unchanged.

        Converges to the type-II ML estimate of the noise precision, so the
        aleatoric term beta^-1 reflects the data rather than a constructor
        default. Falls back gracefully (keeps the last valid estimate) when the
        design is rank-deficient or the effective parameter count saturates.
        """
        Phi = self._phi(X)
        y = np.asarray(y, float)
        N, D = Phi.shape
        lam = np.linalg.eigvalsh(Phi.T @ Phi)
        lam = np.clip(lam, 0.0, None)

        alpha, beta = float(self.alpha), float(self.beta)
        for _ in range(n_iter):
            S = np.linalg.inv(alpha * np.eye(D) + beta * Phi.T @ Phi)
            m = beta * S @ Phi.T @ y

            gamma = float(np.sum(beta * lam / (alpha + beta * lam)))
            mtm = float(m @ m)
            rss = float(np.sum((y - Phi @ m) ** 2))

            # guards: keep the previous value if an update is undefined
            alpha_new = gamma / mtm if mtm > 1e-12 else alpha
            dof = N - gamma
            beta_new = dof / rss if (dof > 1e-9 and rss > 1e-12) else beta

            alpha_new = float(np.clip(alpha_new, 1e-10, 1e10))
            beta_new = float(np.clip(beta_new, 1e-10, 1e10))

            converged = (abs(alpha_new - alpha) <= tol * max(alpha, 1.0) and
                         abs(beta_new - beta) <= tol * max(beta, 1.0))
            alpha, beta = alpha_new, beta_new
            if converged:
                break

        self.alpha, self.beta = alpha, beta
        self.S = np.linalg.inv(alpha * np.eye(D) + beta * Phi.T @ Phi)
        self.m = beta * self.S @ Phi.T @ y
        return self

    def predict(self, X):
        """Returns (mean, std) of the predictive distribution."""
        Phi = self._phi(X)
        mean = Phi @ self.m
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        return mean, np.sqrt(np.maximum(var, 0.0))


def cv_noise_precision(X, y, n_splits: int = 5, seed: int = 0,
                       factory=None) -> dict:
    """Independent cross-check on beta, using held-out residuals instead of the
    evidence.

    For a well-specified model the held-out squared error decomposes as

        E[(y - mu)^2] = sigma^2 + phi^T S phi        (aleatoric + epistemic)

    so sigma^2 is estimated by subtracting the mean predicted epistemic variance
    from the mean squared residual. Returns the implied beta together with the
    diagnostics needed to judge it.

    Disagreement between this beta and the evidence beta is informative: it means
    the model is misspecified, not that one estimator is 'wrong'.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float)
    n = len(y)
    if n < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} samples for {n_splits}-fold CV, got {n}")
    factory = factory or (lambda: BayesianLinearSurrogate())

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_splits)

    resid_sq, epistemic, covered = [], [], []
    for k in range(n_splits):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        model = factory().fit(X[tr], y[tr])
        mu, sd = model.predict(X[te])
        resid_sq.append((y[te] - mu) ** 2)
        epistemic.append(np.maximum(sd ** 2 - 1.0 / model.beta, 0.0))
        covered.append(np.abs(y[te] - mu) <= 1.96 * sd)

    mse = float(np.mean(np.concatenate(resid_sq)))
    epi = float(np.mean(np.concatenate(epistemic)))
    sigma2 = max(mse - epi, 1e-12)
    return {"beta": 1.0 / sigma2,
            "sigma_eV": float(np.sqrt(sigma2)),
            "cv_rmse": float(np.sqrt(mse)),
            "mean_epistemic_var": epi,
            "coverage_95": float(np.mean(np.concatenate(covered))),
            "n_splits": n_splits, "n": n}
