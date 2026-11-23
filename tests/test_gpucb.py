import numpy as np
import pytest

from roge.gpucb import GPUCBOptions, _GP, gp_ucb

from reference_gp import _GP as ReferenceGP


def _problem(rng, n, d):
    X = rng.uniform(size=(n, d))
    Y = np.sin(3 * X).sum(1) + 0.1 * rng.normal(size=n)
    return X, Y, 0.1 * float(Y.max() - Y.min())


@pytest.mark.parametrize("seed", range(10))
def test_gp_matches_reference_with_fixed_hyperparameters(seed):
    rng = np.random.default_rng(seed)
    X, Y, noise = _problem(rng, int(rng.integers(5, 60)), int(rng.integers(1, 7)))
    ref = ReferenceGP(X, Y, noise)
    ref.log_amp, ref.log_len = rng.uniform(-2, 1), rng.uniform(np.log(0.05), np.log(2))
    ref._fit()
    gp = _GP(X, Y, noise)
    gp.kernel = gp.kernel.clone_with_theta(np.array([2 * ref.log_amp, ref.log_len]))
    gp._fit(optimise=False)
    Xs = rng.uniform(size=(50, X.shape[1]))
    for a, b in zip(gp.predict(Xs), ref.predict(Xs)):
        assert np.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_hyperparameter_fit_not_worse_than_reference(seed):
    rng = np.random.default_rng(seed)
    X, Y, noise = _problem(rng, 40, 3)
    ref = ReferenceGP(X, Y, noise)
    ref.optimise_hyperparameters(np.random.default_rng(seed), 5)
    gp = _GP(X, Y, noise)
    gp.optimise_hyperparameters(np.random.default_rng(seed), 5)
    log_amp, log_len = gp.kernel.theta[0] / 2, gp.kernel.theta[1]
    assert ref.neg_log_likelihood((log_amp, log_len)) <= ref.neg_log_likelihood(
        (ref.log_amp, ref.log_len)) + 1e-6


def _branin(u):
    x1, x2 = -5 + 15 * u[0], 15 * u[1]
    return -((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2
             + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


@pytest.mark.parametrize("reference", ["observed", "mean"])
def test_gp_ucb_beats_random_search_on_branin(reference):
    best, rand = [], []
    for seed in range(3):
        res = gp_ucb(_branin, 2, np.random.default_rng(seed), GPUCBOptions(best_reference=reference))
        best.append(res.y)
        U = np.random.default_rng(100 + seed).uniform(size=(res.n_evaluations, 2))
        rand.append(max(_branin(u) for u in U))
    assert np.mean(best) > np.mean(rand)
    assert np.mean(best) > -1.0  # optimum -0.398
