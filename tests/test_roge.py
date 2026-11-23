from pathlib import Path

import numpy as np
import pytest

from roge import (
    GPUCBOptions,
    ParameterSpace,
    Traces,
    discrimination,
    gp_ucb,
    load_java_dataset,
    make_classifier,
    misclassification_rate,
    parse,
    robustness,
    shift_formula,
)
from roge import genetic
from roge.roge import RogeConfig, evaluate, evolve, generate_initial_formulae, size_penalty, train_alpha
from roge.stl import Atom, Until, get_parameters, set_parameters

NAVAL = Path(__file__).resolve().parents[1] / "data/naval"
TIMES = np.arange(0, 301, 5.0)


def traces(*signals, variables=("x", "y")):
    v = np.stack([np.asarray(s, dtype=float) for s in signals])[None]
    return Traces(TIMES, v, variables)


# ---------------------------------------------------------------- semantics


def brute_until(r1, r2, a, b, i):
    """Def. 2 at sample i, by enumeration."""
    t = TIMES[i]
    best = -np.inf
    for k, tk in enumerate(TIMES):
        if t + a <= tk <= t + b:
            inf1 = min([r1[j] for j, tj in enumerate(TIMES) if t <= tj < tk], default=np.inf)
            best = max(best, min(r2[k], inf1))
    return best


def test_until_matches_definition_2_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.normal(size=(2, len(TIMES)))
        a = float(rng.uniform(0, 100))
        b = a + float(rng.uniform(5, 150))
        phi = Until(Atom("x", ">", 0.0), Atom("y", ">", 0.0), (a, b))
        rho = robustness(phi, traces(x, y))[0]
        for i in range(0, 61, 7):
            if TIMES[i] + a <= TIMES[-1]:
                assert rho[i] == pytest.approx(brute_until(x, y, a, b, i))


def test_until_cases_where_java_semantics_differs():
    phi = parse("(x > 0) U[49,287] (y > 0)")
    # case A: phi1 violated at t=10 (< a), phi2 true from t=100
    x = np.ones(61); x[2] = -5
    y = np.where(TIMES < 100, -1.0, 2.0)
    assert robustness(phi, traces(x, y))[0, 0] == -5
    assert robustness(phi, traces(x, y), until="java")[0, 0] == 1
    # case B: phi2 true only at t=50 (first sample of the window)
    x = np.ones(61); y = -np.ones(61); y[10] = 3
    assert robustness(phi, traces(x, y))[0, 0] == 1
    assert robustness(phi, traces(x, y), until="java")[0, 0] == -1


def test_eventually_globally_and_boolean():
    x = np.sin(TIMES / 30)
    y = np.cos(TIMES / 50)
    d = traces(x, y)
    w = (TIMES >= 20) & (TIMES <= 80)
    assert robustness(parse("F[20,80] (x > 0.1)"), d)[0, 0] == pytest.approx((x[w] - 0.1).max())
    assert robustness(parse("G[20,80] (x <= 0.1)"), d)[0, 0] == pytest.approx((0.1 - x[w]).min())
    r = robustness(parse("!(x > 0) & (y > 0) | (y <= 1)"), d)[0]
    assert r == pytest.approx(np.maximum(np.minimum(-x, y), 1 - y))


def test_empty_window_holds_last_sample():
    x = np.arange(61.0)
    d = traces(x, x)
    rho = robustness(parse("F[11,13] (x > 0)"), d)[0]
    assert rho[0] == 3.0  # no sample in [11,13]: first sample after 11 (t=15)
    assert rho[-1] == 60.0  # past the end: last sample


def test_parse_and_parameters_roundtrip():
    phi = parse("((x2 > 22.46) U[49,287] (x1 <= 31.65))")
    assert list(get_parameters(phi)) == [49, 287, 22.46, 31.65]
    set_parameters(phi, [1, 2, 3, 4])
    assert str(parse(str(phi))) == str(phi)


# ---------------------------------------------------------------- shift


def test_shift_is_exact_translation_even_under_negation():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 61))
    d = traces(x, y)
    for text in ["!(F[0,50] (x > 0.2)) | G[10,40] (y <= -0.1)",
                 "(!(x > 0) U[5,100] (y > 0.3)) & !(!(y <= 1))"]:
        phi = parse(text)
        assert robustness(shift_formula(phi, 0.37), d) == pytest.approx(robustness(phi, d) - 0.37)


def test_classifier_negates_formula_for_negative_class():
    Dp = Traces(TIMES, np.zeros((5, 2, 61)) - 1, ("x", "y"))
    Dn = Traces(TIMES, np.zeros((5, 2, 61)) + 1, ("x", "y"))
    ind = evaluate(parse("G[0,100] (x > 0)"), Dp, Dn)
    assert ind.G < 0
    clf = make_classifier(ind)
    assert (clf.predict(Dp) == 1).all() and (clf.predict(Dn) == -1).all()


def test_train_alpha_minimises_training_error_within_means():
    rp = np.array([0.0, 2.0, 3.0, 4.0])  # mu_p = 2.25
    rn = np.array([-4.0, -3.0, -2.0, 1.0])  # mu_n = -2
    a = train_alpha(rp, rn, rp.mean(), rn.mean(), tie_break=0.0)
    assert rn.mean() <= a <= rp.mean()
    assert ((rp <= a).sum() + (rn > a).sum()) == 1  # the best achievable here
    # never outside [mu_n, mu_p], even if the error would be lower there
    a = train_alpha(np.array([-5.0, 10.0]), np.array([-6.0, -6.0]), 2.5, -6.0, tie_break=0.0)
    assert -6.0 <= a <= 2.5


def test_make_classifier_train_rule():
    rng = np.random.default_rng(3)
    Dp = Traces(TIMES, rng.normal(0.5, 1.0, (30, 2, 61)), ("x", "y"))
    Dn = Traces(TIMES, rng.normal(-0.5, 1.0, (30, 2, 61)), ("x", "y"))
    ind = evaluate(parse("x > 0"), Dp, Dn)
    data = Traces(TIMES, np.concatenate([Dp.values, Dn.values]), ("x", "y"))
    y = np.r_[np.ones(30), -np.ones(30)]
    e_train = misclassification_rate(make_classifier(ind, "train", Dp=Dp, Dn=Dn), data, y)
    for rule in ("weighted", "midpoint", "java"):
        assert e_train <= misclassification_rate(make_classifier(ind, rule), data, y)
    with pytest.raises(ValueError):
        make_classifier(ind, "train")


# ---------------------------------------------------------------- G and GP-UCB


def test_discrimination():
    assert discrimination(np.array([1.0, 3.0]), np.array([-1.0, -3.0])) == pytest.approx(2.0)
    assert discrimination(np.ones(3), np.zeros(3)) > 100


def test_gp_ucb_finds_maximum():
    f = lambda u: -float(((u - np.array([0.3, 0.7])) ** 2).sum())
    res = gp_ucb(f, 2, np.random.default_rng(0), GPUCBOptions(max_iter=40))
    assert res.y > -0.01


def test_unit_cube_mapping_orders_intervals():
    phi = parse("F[?,?] (x > ?)")
    space = ParameterSpace((0, 300), {"x": (-1, 1)})
    th = space.from_unit(phi, np.array([0.5, 0.5, 0.25]))
    assert th.tolist() == [150, 225, -0.5]


# ---------------------------------------------------------------- GA


def test_templates_count_and_no_duplicates():
    t = genetic.initial_templates(("x", "y"))
    # atoms 4, pairs C(4,2)=6 x {and, or} -> 16 mus x {F, G} + 12 untils
    assert len(t) == 44
    assert len({str(p) for p in t}) == 44


def test_genetic_operators_produce_valid_trees():
    rng = np.random.default_rng(2)
    vs = ("x", "y")
    pop = [genetic.random_formula_up_to(vs, 5, rng) for _ in range(30)]
    assert all(2 <= p.size() <= 5 for p in pop)
    for _ in range(200):
        a, b = pop[rng.integers(30)], pop[rng.integers(30)]
        sa, sb = str(a), str(b)
        c1, c2 = genetic.crossover(a, b, rng)
        m = genetic.mutate(a, vs, rng)
        assert (str(a), str(b)) == (sa, sb)  # parents untouched
        assert c1.size() + c2.size() == a.size() + b.size()
        for c in (c1, c2, m):
            assert parse(str(c)) is not None


def test_evolve_and_size_penalty():
    rng = np.random.default_rng(3)
    cfg = RogeConfig(seed=3)
    gen = generate_initial_formulae(("x", "y"), cfg, rng)
    assert len(gen) == 54
    from roge.roge import Individual
    sub = [Individual(f, 1.0, 0, 1, 0, 1) for f in gen[:10]]
    kids = evolve(sub, 20, ("x", "y"), cfg, rng)
    assert len(kids) == 20 and all(k.size() <= cfg.max_size for k in kids)
    S = size_penalty(sub, cfg)
    sizes = np.array([s.size for s in sub])
    assert np.all(np.diff(S[np.argsort(sizes)]) >= -1e-12)  # non-decreasing in size


# ---------------------------------------------------------------- data


@pytest.mark.skipif(not NAVAL.exists(), reason="naval data not available")
def test_load_naval():
    d, labels = load_java_dataset(NAVAL, "naval", ("x", "y"))
    assert d.values.shape == (2000, 2, 61)
    assert sorted(np.unique(labels)) == [-1, 1] and (labels == 1).sum() == 1000
    assert d.values[0, 0, 0] == pytest.approx(78.086)  # interleaved x, y per sample
    assert d.values[0, 1, 0] == pytest.approx(39.987)
