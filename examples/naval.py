"""Maritime surveillance case study (paper §5.1), dataset in ``data/naval``.

    uv run python examples/naval.py eq2          # paper formula (Eq. 2), both Until semantics
    uv run python examples/naval.py single       # GP-UCB on the structure of Eq. 2
    uv run python examples/naval.py roge         # ROGE, one 80/20 split
    uv run python examples/naval.py cv --folds 10
"""

from __future__ import annotations

import argparse
import logging
import os
import time

# small matrices: multithreaded BLAS only adds overhead
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

from roge import (
    GPUCBOptions,
    ParameterSpace,
    STLClassifier,
    RogeConfig,
    evaluate,
    learn_parameters,
    load_java_dataset,
    make_classifier,
    misclassification_rate,
    parse,
    roge,
    split_by_label,
    stratified_kfold,
    train_test_split,
)

DATA = Path(__file__).resolve().parents[1] / "data/naval"
# the paper calls the coordinates x1, x2; bounds as in ExampleLearning
VARIABLES = ("x1", "x2")
SPACE = ParameterSpace((0.0, 300.0), {"x1": (0.0, 80.0), "x2": (0.0, 45.0)})
EQ2 = "(x2 > 22.46) U[49,287] (x1 <= 31.65)"


def fit_and_score(data, labels, train, test, args, rng):
    Dp, Dn = split_by_label(data.subset(train), labels[train])
    cfg = RogeConfig(ng=args.ng, seed=int(rng.integers(2**31)), until=args.until)
    cfg.gpucb.max_iter = args.gp_iter
    res = roge(Dp, Dn, SPACE, cfg)
    clf = make_classifier(res.best, until=args.until)
    return res, clf, misclassification_rate(clf, data.subset(test), labels[test])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eq2", "single", "roge", "cv"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--gp-iter", type=int, default=50)
    ap.add_argument("--until", choices=["paper", "java"], default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    rng = np.random.default_rng(args.seed)
    data, labels = load_java_dataset(DATA, "naval", VARIABLES)

    if args.mode == "eq2":
        Dp, Dn = split_by_label(data, labels)
        phi = parse(EQ2)
        for sem in ("paper", "java"):
            ind = evaluate(phi, Dp, Dn, sem)
            mis_raw = misclassification_rate(STLClassifier(phi, sem), data, labels)
            mis_shift = misclassification_rate(make_classifier(ind, until=sem), data, labels)
            print(f"[{sem:5}] G={ind.G:.3f}  mu_p={ind.mu_p:.2f}±{ind.sd_p:.2f}  "
                  f"mu_n={ind.mu_n:.2f}±{ind.sd_n:.2f}  mis-rate: Eq.2 as is={mis_raw:.4f}, "
                  f"after alpha-shift={mis_shift:.4f}")
        return

    if args.mode == "single":
        train, test = train_test_split(labels, 0.8, rng)
        Dp, Dn = split_by_label(data.subset(train), labels[train])
        phi = parse("(x2 > ?) U[?,?] (x1 <= ?)")
        t0 = time.perf_counter()
        ind = learn_parameters(phi, Dp, Dn, SPACE, rng, GPUCBOptions(max_iter=args.gp_iter),
                               args.until)
        clf = make_classifier(ind, until=args.until)
        print(f"{ind}\nclassifier: {clf.formula}\n"
              f"test mis-rate {misclassification_rate(clf, data.subset(test), labels[test]):.4f}"
              f"  ({time.perf_counter() - t0:.1f}s)")
        return

    if args.mode == "roge":
        train, test = train_test_split(labels, 0.8, rng)
        res, clf, mis = fit_and_score(data, labels, train, test, args, rng)
        for h in res.history:
            print(h)
        print(f"best: {res.best}\nclassifier: {clf.formula}\n"
              f"test mis-rate {mis:.4f}  ({res.elapsed:.1f}s)")
        return

    rates, times = [], []
    for f, (train, test) in enumerate(stratified_kfold(labels, args.folds, rng)):
        res, clf, mis = fit_and_score(data, labels, train, test, args, rng)
        rates.append(mis)
        times.append(res.elapsed)
        print(f"fold {f}: mis-rate {mis:.4f}  {res.elapsed:.1f}s  {clf.formula}", flush=True)
    print(f"mis-rate {np.mean(rates):.4f} ± {np.std(rates):.4f}   "
          f"time {np.mean(times):.1f} ± {np.std(times):.1f}s")


if __name__ == "__main__":
    main()
