"""Hand-written GP used before the scikit-learn backend; kept as an
independent reference for the differential tests in ``test_gpucb.py``."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class _GP:
    """Zero-mean (after centring) GP regression with isotropic RBF kernel."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, noise: float):
        self.X, self.Y, self.noise = X, Y, noise
        self.log_amp, self.log_len = 0.0, np.log(0.3)
        self._fit()

    def _k(self, A, B, log_amp=None, log_len=None):
        la = self.log_amp if log_amp is None else log_amp
        ll = self.log_len if log_len is None else log_len
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(2 * la) * np.exp(-0.5 * d2 / np.exp(2 * ll))

    def _fit(self):
        self.mean = float(self.Y.mean())
        K = self._k(self.X, self.X) + (self.noise + 1e-10) * np.eye(len(self.X))
        self.chol = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.chol, self.Y - self.mean)

    def add(self, x: np.ndarray, y: float):
        self.X = np.vstack([self.X, x])
        self.Y = np.append(self.Y, y)
        self._fit()

    def neg_log_likelihood(self, h):
        la, ll = h
        n = len(self.X)
        K = self._k(self.X, self.X, la, ll) + (self.noise + 1e-10) * np.eye(n)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        r = self.Y - self.Y.mean()
        return 0.5 * r @ cho_solve(c, r) + np.log(np.diag(c[0])).sum()

    def optimise_hyperparameters(self, rng: np.random.Generator, restarts: int):
        bounds = [(-6.0, 6.0), (np.log(1e-3), np.log(10.0))]
        starts = [np.array([np.log(self.Y.std() + 1e-6), np.log(0.3)])]
        starts += [np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])]) for _ in range(restarts)]
        best = None
        for s in starts:
            s = np.clip(s, [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(self.neg_log_likelihood, s, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.log_amp, self.log_len = best.x
        self._fit()

    def predict(self, Xs: np.ndarray):
        Ks = self._k(Xs, self.X)
        mu = self.mean + Ks @ self.alpha
        v = cho_solve(self.chol, Ks.T)
        var = np.exp(2 * self.log_amp) - (Ks * v.T).sum(1)
        return mu, np.sqrt(np.maximum(var, 0.0))
