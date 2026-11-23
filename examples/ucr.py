"""Binary UCR datasets (the benchmark used by ROCKET) with the standard train/test split.

    uv run python examples/ucr.py <dir> GunPoint --seeds 5

``<dir>`` contains ``<Name>_TRAIN.txt`` and ``<Name>_TEST.txt`` (UCR archive format).
"""

from __future__ import annotations

import argparse
import logging
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np

from roge import RogeConfig, Traces, load_ucr_txt, make_classifier, misclassification_rate, roge, split_by_label


def with_derivative(d: Traces) -> Traces:
    dx = np.gradient(d.values[:, 0, :], d.times, axis=1)
    return Traces(d.times, np.stack([d.values[:, 0, :], dx], axis=1), ("x", "dx"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("name")
    ap.add_argument("--positive", type=float, default=1, help="label of the positive class")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--derivative", action="store_true",
                    help="add the first derivative dx as a second variable (as flow' in paper §5.2)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    train, ytr = load_ucr_txt(args.directory / f"{args.name}_TRAIN.txt", positive_label=args.positive)
    test, yte = load_ucr_txt(args.directory / f"{args.name}_TEST.txt", positive_label=args.positive)
    if args.derivative:
        train, test = with_derivative(train), with_derivative(test)
    Dp, Dn = split_by_label(train, ytr)
    print(f"{args.name}: train {len(train)} ({len(Dp)}+/{len(Dn)}-), test {len(test)}, length {len(train.times)}")

    errs = []
    for seed in range(args.seeds):
        res = roge(Dp, Dn, cfg=RogeConfig(ng=args.ng, seed=seed))
        clf = make_classifier(res.best)
        tr_err = misclassification_rate(clf, train, ytr)
        te_err = misclassification_rate(clf, test, yte)
        errs.append(te_err)
        print(f"seed {seed}: train err {tr_err:.4f}  test err {te_err:.4f}  "
              f"acc {1 - te_err:.4f}  G={res.best.G:.3f}  {res.elapsed:.1f}s  {clf.formula}", flush=True)
    print(f"test accuracy {1 - np.mean(errs):.4f} ± {np.std(errs):.4f}")


if __name__ == "__main__":
    main()
