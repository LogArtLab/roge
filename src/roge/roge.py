"""ROGE — RObustness GEnetic algorithm (paper Alg. 1, §4).

Bi-level optimisation: a genetic algorithm on the formula structure (§4.3),
GP-UCB on the parameters of every formula (§4.2), both driven by the
discrimination function G (§4.1, Eq. 1) computed on the **training** data.
The best formula is finally turned into a classifier by translating its
atomic predicates by a threshold alpha (§4.2, last paragraph).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from . import genetic
from .gpucb import GPUCBOptions, gp_ucb
from .stl import (
    Atom,
    Node,
    Not,
    Traces,
    UntilSemantics,
    atoms,
    n_parameters,
    robustness_at_zero,
    set_parameters,
    temporal_nodes,
)

log = logging.getLogger("roge")

G_MAX = 1e3  # cap for G when both standard deviations vanish


# --------------------------------------------------------------------------
# Discrimination function (Eq. 1)
# --------------------------------------------------------------------------


def discrimination(rob_p: np.ndarray, rob_n: np.ndarray) -> float:
    """``G = (E[R|Xp] - E[R|Xn]) / (sigma[R|Xp] + sigma[R|Xn])``.

    Population standard deviations (as in the original Java implementation). If both are 0 the value
    is capped at ``±G_MAX`` (0 when the means coincide).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        mp, mn = float(rob_p.mean()), float(rob_n.mean())
        den = float(rob_p.std() + rob_n.std())
    num = mp - mn
    if not (np.isfinite(num) and np.isfinite(den)):
        # only with until="java" (-MAX_VALUE robustness); the Java code maps NaN to 0
        return 0.0
    if den <= 0:
        return 0.0 if num == 0 else float(np.sign(num) * G_MAX)
    return float(np.clip(num / den, -G_MAX, G_MAX))


# --------------------------------------------------------------------------
# Parameter space K = T x C and learning of the parameters (§4.2)
# --------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """Bounds of the temporal parameters and of the thresholds of each variable."""

    time_bounds: tuple[float, float]
    var_bounds: dict[str, tuple[float, float]]

    @classmethod
    def from_data(cls, *datasets: Traces) -> "ParameterSpace":
        times = datasets[0].times
        bounds = {}
        for i, v in enumerate(datasets[0].variables):
            vals = np.concatenate([d.values[:, i, :].ravel() for d in datasets])
            bounds[v] = (float(vals.min()), float(vals.max()))
        return cls((0.0, float(times[-1] - times[0])), bounds)

    def from_unit(self, phi: Node, u: np.ndarray) -> np.ndarray:
        """Map a point of the unit cube to a parameter vector of ``phi``.

        Each interval ``(u_a, u_b)`` becomes ``a = u_a``, ``b = u_a + u_b (1 - u_a)``
        (so that ``a <= b``) and is then rescaled to ``time_bounds``; thresholds
        are rescaled to the bounds of their variable. The reparametrisation is
        applied exactly once.
        """
        u = np.asarray(u, dtype=float)
        n_t = len(temporal_nodes(phi))
        lo, hi = self.time_bounds
        theta = np.empty_like(u)
        for i in range(n_t):
            ua, ub = u[2 * i], u[2 * i + 1]
            a = ua
            b = ua + ub * (1 - ua)
            theta[2 * i] = lo + a * (hi - lo)
            theta[2 * i + 1] = lo + b * (hi - lo)
        for k, at in enumerate(atoms(phi)):
            vlo, vhi = self.var_bounds[at.var]
            theta[2 * n_t + k] = vlo + u[2 * n_t + k] * (vhi - vlo)
        return theta


@dataclass
class Individual:
    """A formula with learned parameters and its score on the training set."""

    formula: Node
    G: float
    mu_p: float
    sd_p: float
    mu_n: float
    sd_n: float

    @property
    def size(self) -> int:
        return self.formula.size()

    def __str__(self) -> str:
        return f"G={self.G:.4f}  size={self.size}  {self.formula}"


Score = Callable[[np.ndarray, np.ndarray], float]


def evaluate(phi: Node, Dp: Traces, Dn: Traces, until: UntilSemantics = "paper",
             score: Score = discrimination) -> Individual:
    """``Individual`` with ``G = score(rob_p, rob_n)`` (Eq. 1 by default)."""
    rp, rn = robustness_at_zero(phi, Dp, until), robustness_at_zero(phi, Dn, until)
    return Individual(phi, score(rp, rn), float(rp.mean()), float(rp.std()),
                      float(rn.mean()), float(rn.std()))


def learn_parameters(
    phi: Node,
    Dp: Traces,
    Dn: Traces,
    space: ParameterSpace,
    rng: np.random.Generator,
    options: GPUCBOptions | None = None,
    until: UntilSemantics = "paper",
    score: Score = discrimination,
) -> Individual:
    """LEARNINGPARAMETERS for one formula: maximise ``theta -> G(phi_theta)``.

    ``score`` replaces G only in non-ROGE variants (see ``roge.variant``).
    """
    work = phi.copy()

    def objective(u: np.ndarray) -> float:
        set_parameters(work, space.from_unit(work, u))
        rp = robustness_at_zero(work, Dp, until)
        rn = robustness_at_zero(work, Dn, until)
        return score(rp, rn)

    res = gp_ucb(objective, n_parameters(work), rng, options)
    best = phi.copy()
    set_parameters(best, space.from_unit(best, res.x))
    return evaluate(best, Dp, Dn, until, score)


# --------------------------------------------------------------------------
# Genetic algorithm (Alg. 1, §4.3)
# --------------------------------------------------------------------------


@dataclass
class RogeConfig:
    """Inputs of Alg. 1 plus the choices the paper leaves open.

    ``ne=None`` means ``len(templates) + n_random`` (all the templates are
    kept, as the paper suggests for few variables, plus ``n_random`` random
    formulae). Otherwise ``nl = ne - n_random`` templates are sampled.
    """

    ne: int | None = None
    ng: int = 3
    alpha: float = 0.1  # mutation probability (paper: alpha <= 0.1)
    s: int = 5  # maximum size of the random initial formulae
    n_random: int = 10
    # size penalty S(phi) = g * h(size), g = mean G of the current set
    size_penalty: Literal["increasing", "literal", "none"] = "increasing"
    p: float = 0.5 ** (1 / 5)  # p^5 = 0.5 (paper §4.3)
    max_size: int = 15  # children larger than this are discarded (not in the paper)
    gpucb: GPUCBOptions = field(default_factory=GPUCBOptions)
    until: UntilSemantics = "paper"
    seed: int | None = None
    # objective replacing G; None = Eq. 1 (ROGE). Only for non-ROGE variants.
    score: Score | None = None


def size_penalty(pop: list[Individual], cfg: RogeConfig) -> np.ndarray:
    """``S(phi)`` for every individual of ``pop``.

    Paper §4.3: ``S = g * p^size`` with ``p^5 = 0.5`` ("formulae of size 5
    get a 50% penalty") and ``g`` the average discrimination of the current
    generation. Read literally, ``p < 1`` makes the penalty *decrease* with
    the size, contradicting its purpose; ``"increasing"`` (default) uses
    ``g * (1 - p^size)``, which matches both "50% at size 5" and "penalizes
    large sizes". ``g`` is clipped at 0 so that a negative average does not
    turn the penalty into a bonus.
    """
    sizes = np.array([ind.size for ind in pop], dtype=float)
    if cfg.size_penalty == "none":
        return np.zeros(len(pop))
    g = max(float(np.mean([ind.G for ind in pop])), 0.0)
    if cfg.size_penalty == "literal":
        return g * cfg.p**sizes
    return g * (1 - cfg.p**sizes)


def sample(pop: list[Individual], k: int, cfg: RogeConfig) -> list[Individual]:
    """SAMPLE: the ``k`` best individuals according to ``F = G - S``."""
    F = np.array([ind.G for ind in pop]) - size_penalty(pop, cfg)
    order = np.argsort(-F, kind="stable")
    return [pop[i] for i in order[:k]]


def evolve(sub: list[Individual], n: int, variables, cfg: RogeConfig,
           rng: np.random.Generator) -> list[Node]:
    """EVOLVE: ``n`` children; mutation if ``p <= alpha``, else recombination.

    Parents are chosen uniformly at random from ``sub`` (paper), not by a
    roulette on G.
    """
    children: list[Node] = []
    attempts = 0
    while len(children) < n:
        attempts += 1
        if attempts > 100 * n:
            raise RuntimeError("EVOLVE cannot produce children within max_size")
        if rng.random() <= cfg.alpha or len(sub) < 2:
            parent = sub[rng.integers(len(sub))].formula
            new = [genetic.mutate(parent, variables, rng)]
        else:
            i, j = rng.choice(len(sub), size=2, replace=False)
            new = list(genetic.crossover(sub[i].formula, sub[j].formula, rng))
        children.extend(c for c in new if c.size() <= cfg.max_size)
    return children[:n]


def generate_initial_formulae(variables, cfg: RogeConfig, rng: np.random.Generator) -> list[Node]:
    """GENERATEINITIALFORMULAE(Ne, s)."""
    templates = genetic.initial_templates(variables)
    if cfg.ne is None:
        chosen = templates
        nr = cfg.n_random
    else:
        nr = min(cfg.n_random, cfg.ne)
        nl = cfg.ne - nr
        if nl >= len(templates):
            chosen, nr = templates, cfg.ne - len(templates)
        else:
            idx = rng.choice(len(templates), size=nl, replace=False)
            chosen = [templates[i] for i in sorted(idx)]
    randoms = [genetic.random_formula_up_to(variables, cfg.s, rng) for _ in range(nr)]
    return chosen + randoms


@dataclass
class RogeResult:
    population: list[Individual]
    best: Individual
    history: list[dict]
    elapsed: float


def roge(Dp: Traces, Dn: Traces, space: ParameterSpace | None = None,
         cfg: RogeConfig | None = None) -> RogeResult:
    """Alg. 1. ``Dp``: good trajectories, ``Dn``: bad trajectories (training)."""
    cfg = cfg or RogeConfig()
    space = space or ParameterSpace.from_data(Dp, Dn)
    rng = np.random.default_rng(cfg.seed)
    variables = Dp.variables
    t0 = time.perf_counter()

    def learn_all(formulae: list[Node], label: str) -> list[Individual]:
        out = []
        for k, phi in enumerate(formulae):
            out.append(learn_parameters(phi, Dp, Dn, space, rng, cfg.gpucb, cfg.until,
                                        cfg.score or discrimination))
            log.debug("%s %d/%d %s", label, k + 1, len(formulae), out[-1])
        return out

    gen = generate_initial_formulae(variables, cfg, rng)  # line 1
    ne = len(gen)
    gen_theta = learn_all(gen, "init")  # line 2
    history = [_summary(0, gen_theta, t0)]
    log.info("generation 0: best %s", max(gen_theta, key=lambda i: i.G))
    for it in range(1, cfg.ng + 1):  # line 3
        sub = sample(gen_theta, max(ne // 2, 1), cfg)  # line 4
        newg = evolve(sub, ne, variables, cfg, rng)  # line 5
        newg_theta = learn_all(newg, f"gen{it}")  # line 6
        gen_theta = sample(newg_theta + gen_theta, ne, cfg)  # line 7
        history.append(_summary(it, gen_theta, t0))
        log.info("generation %d: best %s", it, max(gen_theta, key=lambda i: i.G))
    best = max(gen_theta, key=lambda i: i.G)  # phi_best = argmax G
    return RogeResult(gen_theta, best, history, time.perf_counter() - t0)


def _summary(it: int, pop: list[Individual], t0: float) -> dict:
    G = np.array([i.G for i in pop])
    best = pop[int(np.argmax(G))]
    return {"generation": it, "best_G": float(G.max()), "mean_G": float(G.mean()),
            "best": str(best.formula), "elapsed": time.perf_counter() - t0}


# --------------------------------------------------------------------------
# Threshold shift and classifier (§4.2, last paragraph)
# --------------------------------------------------------------------------


def shift_formula(phi: Node, alpha: float) -> Node:
    """Return ``phi'`` with ``rho(phi', x, t) = rho(phi, x, t) - alpha`` exactly.

    Every atom is translated by ``alpha`` with the sign given by its polarity
    (number of enclosing negations): the paper's argument (min/max/inf/sup
    commute with translations) needs this correction when ``phi`` contains
    ``!``. No clamping of the thresholds.
    """
    out = phi.copy()

    def rec(n: Node, positive: bool) -> None:
        if isinstance(n, Atom):
            delta = alpha if positive else -alpha  # desired change of rho: -delta
            n.threshold = n.threshold + delta if n.op == ">" else n.threshold - delta
            return
        for c in n.children:
            rec(c, positive ^ isinstance(n, Not))

    rec(out, True)
    return out


AlphaRule = Literal["weighted", "midpoint", "java", "train"]


def choose_alpha(mu_p, sd_p, mu_n, sd_n, rule: AlphaRule = "weighted") -> float:
    """Threshold between the two average robustness values (requires mu_p >= mu_n).

    - ``weighted``: ``(mu_p sd_n + mu_n sd_p) / (sd_p + sd_n)``, the point at
      the same number of standard deviations from both means (always between
      them; then ``G = (mu_p - alpha) / sd_p``);
    - ``midpoint``: ``(mu_p + mu_n) / 2``;
    - ``java``: ``((mu_p - sd_p) + (mu_n + sd_n)) / 2`` (original Java implementation);
    - ``train``: minimum training error, see :func:`train_alpha` (needs the
      training robustness, so it is handled by :func:`make_classifier`).
    """
    if rule == "train":
        raise ValueError("rule 'train' needs the training data: use make_classifier(..., Dp=, Dn=)")
    if rule == "midpoint" or (rule == "weighted" and sd_p + sd_n == 0):
        return (mu_p + mu_n) / 2
    if rule == "weighted":
        return (mu_p * sd_n + mu_n * sd_p) / (sd_p + sd_n)
    return ((mu_p - sd_p) + (mu_n + sd_n)) / 2


def train_alpha(rob_p: np.ndarray, rob_n: np.ndarray, mu_p: float, mu_n: float,
                tie_break: float) -> float:
    """``alpha`` in ``[mu_n, mu_p]`` minimising the training error of ``rho > alpha``.

    The paper only asks for a threshold between the two average robustness
    values (§4.2); this rule picks, within that interval, the one with the
    fewest training misclassifications. Candidates are the interval ends and
    the midpoints between consecutive training robustness values inside it;
    ties are broken by the distance from ``tie_break`` (the ``weighted`` alpha).
    """
    rob_p, rob_n = np.asarray(rob_p, float), np.asarray(rob_n, float)
    inside = np.concatenate([rob_p, rob_n])
    inside = np.unique(inside[np.isfinite(inside) & (inside >= mu_n) & (inside <= mu_p)])
    cand = np.unique(np.concatenate([[mu_n, mu_p], (inside[:-1] + inside[1:]) / 2]))
    # errors: positives with rho <= alpha, negatives with rho > alpha
    err = (rob_p[None, :] <= cand[:, None]).sum(1) + (rob_n[None, :] > cand[:, None]).sum(1)
    best = cand[err == err.min()]
    return float(best[np.argmin(np.abs(best - tie_break))])


@dataclass
class STLClassifier:
    """``x`` is predicted positive iff ``rho(formula, x) > 0``."""

    formula: Node
    until: UntilSemantics = "paper"

    def robustness(self, data: Traces) -> np.ndarray:
        return robustness_at_zero(self.formula, data, self.until)

    def predict(self, data: Traces) -> np.ndarray:
        return np.where(self.robustness(data) > 0, 1, -1)


def make_classifier(ind: Individual, rule: AlphaRule = "weighted",
                    until: UntilSemantics = "paper",
                    Dp: Traces | None = None, Dn: Traces | None = None) -> STLClassifier:
    """Turn a learned formula into a classifier by the alpha-shift of §4.2.

    If the formula characterises the negative class (``mu_p < mu_n``) it is
    negated first, so that positives are the satisfying trajectories.
    ``rule="train"`` needs the training sets ``Dp`` and ``Dn``.
    """
    phi, mp, mn = ind.formula, ind.mu_p, ind.mu_n
    sign = 1.0
    if mp < mn:
        phi, mp, mn, sign = Not(phi.copy()), -mp, -mn, -1.0
    if rule == "train":
        if Dp is None or Dn is None:
            raise ValueError("rule 'train' needs Dp and Dn")
        rp = sign * robustness_at_zero(ind.formula, Dp, until)
        rn = sign * robustness_at_zero(ind.formula, Dn, until)
        alpha = train_alpha(rp, rn, mp, mn, choose_alpha(mp, ind.sd_p, mn, ind.sd_n, "weighted"))
    else:
        alpha = choose_alpha(mp, ind.sd_p, mn, ind.sd_n, rule)
    return STLClassifier(shift_formula(phi, alpha), until)


def misclassification_rate(clf: STLClassifier, data: Traces, labels: np.ndarray) -> float:
    pred = clf.predict(data)
    return float(np.mean(pred != np.where(np.asarray(labels) > 0, 1, -1)))
