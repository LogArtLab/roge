"""GP-UCB maximisation of a noisy function on the unit cube (paper §4.2).

Follows the scheme of the original Java ``GPOptimisation.optimise`` (Srinivas et al.
2012 with a heuristic beta schedule):

1. ``n_init`` uniform points, evaluated;
2. GP regression with an RBF kernel (``scikit-learn``), hyperparameters
   fitted by maximum marginal likelihood on the initial points
   (homoskedastic noise variance = ``noise_ratio * range(initial values)``;
   default 0.01, the Java code uses 0.1);
3. at each iteration, sample ``n_grid`` points, take the UCB maximiser
   ``mu + beta * sigma`` with ``beta = beta0 * (1 + 0.1 log(iter))``, refine
   it locally; if its UCB is at least the best value so far, evaluate it.
   "Best value" is the maximum GP posterior mean at the observed points
   (default) or, as in the Java code, the maximum noisy observation
   (``best_reference="observed"``); the two rules accept candidates at
   different rates (see docs/DESIGN.md);
4. stop after ``max_iter`` iterations, ``max_no_improvement`` added points
   without improvement, or ``max_failed`` consecutive failed attempts;
5. return the best observed point (optionally improved by maximising the GP
   mean around it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


@dataclass
class GPUCBOptions:
    n_init: int = 20
    max_iter: int = 50
    n_grid: int = 200
    beta0: float = 2.0
    noise_ratio: float = 0.01
    hyper_restarts: int = 5
    max_no_improvement: int = 100
    improvement_factor: float = 1.01
    max_failed: int = 200
    refine_candidates: bool = True
    final_mean_refinement: bool = True
    # A candidate is evaluated only if its UCB reaches the best value so far:
    # "observed" = max of the noisy observations (original Java implementation),
    # "mean" = max of the GP posterior mean at the observed points.
    best_reference: Literal["observed", "mean"] = "mean"

    @classmethod
    def java(cls, **kw) -> "GPUCBOptions":
        """Settings of the original Java implementation (acceptance on the
        noisy maximum, noise ratio 0.1)."""
        return cls(**{"best_reference": "observed", "noise_ratio": 0.1, **kw})


@dataclass
class GPUCBResult:
    x: np.ndarray
    y: float
    n_evaluations: int
    X: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)


class _GP:
    """GP regression on centred observations: ``scikit-learn`` with an RBF kernel.

    Kernel ``amp^2 * exp(-|x - x'|^2 / (2 len^2))`` (isotropic) plus a fixed
    homoskedastic noise variance on the training points; ``predict`` returns
    the mean and standard deviation of the latent function (noise excluded).
    Hyperparameters ``(amp, len)`` are fitted by maximum marginal likelihood
    (L-BFGS-B with restarts) and then kept fixed when points are added.
    """

    AMP2_BOUNDS = (np.exp(-12.0), np.exp(12.0))
    LEN_BOUNDS = (1e-3, 10.0)

    def __init__(self, X: np.ndarray, Y: np.ndarray, noise: float):
        self.X, self.Y, self.noise = X, Y, noise
        self.kernel = (ConstantKernel(max(float(Y.std()) ** 2, 1e-12), self.AMP2_BOUNDS)
                       * RBF(0.3, self.LEN_BOUNDS))
        self._fit(optimise=False)

    def _fit(self, optimise: bool, restarts: int = 0, seed: int | None = None):
        self.mean = float(self.Y.mean())
        self.model = GaussianProcessRegressor(
            self.kernel, alpha=self.noise + 1e-10, normalize_y=False,
            optimizer="fmin_l_bfgs_b" if optimise else None,
            n_restarts_optimizer=restarts, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(self.X, self.Y - self.mean)
        self.kernel = self.model.kernel_

    def add(self, x: np.ndarray, y: float):
        self.X = np.vstack([self.X, x])
        self.Y = np.append(self.Y, y)
        self._fit(optimise=False)

    def optimise_hyperparameters(self, rng: np.random.Generator, restarts: int):
        self._fit(optimise=True, restarts=restarts, seed=int(rng.integers(2**31 - 1)))

    def predict(self, Xs: np.ndarray):
        mu, sd = self.model.predict(Xs, return_std=True)
        return self.mean + mu, sd


def gp_ucb(
    f: Callable[[np.ndarray], float],
    dim: int,
    rng: np.random.Generator,
    options: GPUCBOptions | None = None,
) -> GPUCBResult:
    """Maximise ``f`` over ``[0, 1]^dim``."""
    o = options or GPUCBOptions()
    if dim == 0:
        y = float(f(np.empty(0)))
        return GPUCBResult(np.empty(0), y, 1, np.empty((1, 0)), np.array([y]))

    X = rng.uniform(size=(o.n_init, dim))
    Y = np.array([f(x) for x in X], dtype=float)
    n_eval = o.n_init
    noise = max(o.noise_ratio * float(Y.max() - Y.min()), 1e-6)
    gp = _GP(X, Y, noise)
    gp.optimise_hyperparameters(rng, o.hyper_restarts)
    bounds = [(0.0, 1.0)] * dim

    def ucb(x, beta):
        mu, sd = gp.predict(np.atleast_2d(x))
        return float(mu[0] + beta * sd[0])

    no_improvement = failed = 0
    for it in range(1, o.max_iter + 1):
        if no_improvement > o.max_no_improvement or failed > o.max_failed:
            break
        beta = o.beta0 * (1 + 0.1 * np.log(it))
        grid = rng.uniform(size=(o.n_grid, dim))
        mu, sd = gp.predict(grid)
        cand = grid[int(np.argmax(mu + beta * sd))]
        if o.refine_candidates:
            res = minimize(lambda x: -ucb(x, beta), cand, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 30})
            cand = np.clip(res.x, 0.0, 1.0)
        best_obs = float(gp.Y.max())
        reference = best_obs if o.best_reference == "observed" else float(gp.predict(gp.X)[0].max())
        if ucb(cand, beta) >= reference:
            failed = 0
            y = float(f(cand))
            n_eval += 1
            gp.add(cand, y)
            # relative improvement, also valid when best_obs < 0
            improved = y >= best_obs + (o.improvement_factor - 1) * abs(best_obs)
            no_improvement = 0 if improved else no_improvement + 1
        else:
            failed += 1

    i = int(np.argmax(gp.Y))
    x_best, y_best = gp.X[i].copy(), float(gp.Y[i])
    if o.final_mean_refinement:
        res = minimize(lambda x: -float(gp.predict(np.atleast_2d(x))[0][0]), x_best,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
        x_ref = np.clip(res.x, 0.0, 1.0)
        y_ref = float(f(x_ref))
        n_eval += 1
        if y_ref > y_best:
            x_best, y_best = x_ref, y_ref
    return GPUCBResult(x_best, y_best, n_eval, gp.X, gp.Y)
