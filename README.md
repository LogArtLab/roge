# ROGE

Python implementation of **ROGE** (RObustness GEnetic algorithm), the method of

> L. Nenzi, S. Silvetti, E. Bartocci, L. Bortolussi. *A Robust Genetic Algorithm for Learning Temporal Specifications from Data.* In: Quantitative Evaluation of Systems (QEST 2018), LNCS 11024, pp. 323–338. Springer, 2018. [doi:10.1007/978-3-319-99154-2_20](https://doi.org/10.1007/978-3-319-99154-2_20)

ROGE learns a Signal Temporal Logic (STL) formula that separates two classes of trajectories. It runs a genetic algorithm on the formula structure and GP-UCB on its parameters. Both are guided by the discrimination function G, the normalised difference of the average robustness of the two classes. A final threshold shift turns the learned formula into a classifier.

The paper is the reference; the [original Java implementation](https://github.com/simonesilvetti/learningformulae) was used as a guide for the details the paper leaves open. [`docs/DESIGN.md`](docs/DESIGN.md) documents every implementation choice.

## Installation

Requires Python ≥ 3.11; dependencies: numpy, scipy, scikit-learn.

```bash
uv sync            # or: pip install -e .
uv run pytest
```

## Usage

```python
import numpy as np
from roge import (RogeConfig, load_java_dataset, make_classifier,
                  misclassification_rate, roge, split_by_label, train_test_split)

data, labels = load_java_dataset("data/naval", "naval", ("x1", "x2"))  # labels: +1 / -1
tr, te = train_test_split(labels, 0.8, np.random.default_rng(0))     # stratified 80/20
Dp, Dn = split_by_label(data.subset(tr), labels[tr])                  # positive / negative training trajectories
result = roge(Dp, Dn, cfg=RogeConfig(seed=0))                         # Alg. 1 (a few minutes)
clf = make_classifier(result.best)                                    # threshold shift (paper §4.2)
print(clf.formula)
print(misclassification_rate(clf, data.subset(te), labels[te]))
```

Paths are relative to the repository root.

**Labels.** `+1` marks the positive ("good") trajectories `Dp`, `-1` the negative ones `Dn`. The classifier predicts `+1` for the trajectories that satisfy the learned formula; if the best formula characterises the negative class, `make_classifier` negates it first. With `load_ucr_txt`, `positive_label` chooses which class is positive and all the others become `-1`.

**Your own data.** Build a `Traces(times, values, variables)` with `values` of shape `(n_trajectories, n_variables, n_samples)`, or use `load_ucr_txt` for files in the UCR archive format.

**Reproducibility.** With a fixed `RogeConfig(seed=...)` a run is deterministic. Set `OMP_NUM_THREADS=1` (the example scripts do it): the matrices are small, so multithreaded BLAS only adds overhead, and it gets in the way when several runs execute in parallel.

### Main parameters (`RogeConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `ng` | `3` | number of generations of the genetic algorithm (Alg. 1) |
| `ne` | `None` | size of the evaluated population; `None` = all the initial templates plus `n_random` |
| `n_random` | `10` | random formulae added to the templates in the initial population |
| `alpha` | `0.1` | mutation probability (otherwise crossover; paper: α ≤ 0.1) |
| `s` | `5` | maximum size of the random initial formulae |
| `size_penalty` | `"increasing"` | size penalty S in the fitness `G − S`; also `"literal"`, `"none"` (see below) |
| `max_size` | `15` | children larger than this are discarded (not in the paper) |
| `gpucb` | `GPUCBOptions()` | GP-UCB parameter search: 20 initial points, `max_iter=50` iterations |
| `until` | `"paper"` | Until semantics; `"java"` reproduces the original code, for comparison only |
| `seed` | `None` | random seed |

### Examples

```bash
uv run python examples/naval.py eq2             # formula of the paper (Eq. 2) on the naval dataset
uv run python examples/naval.py roge            # ROGE on an 80/20 split
uv run python examples/naval.py cv --folds 10   # 10-fold cross-validation (paper §5.1)
uv run python examples/ucr.py <dir> GunPoint --seeds 5
```

## Results

Naval dataset (paper §5.1), stratified 10-fold cross-validation: misclassification **0.65 %** with the default settings, 0.45–0.95 % with `GPUCBOptions.java()` depending on the seed (paper: 0). Reproduce with `uv run python examples/naval.py cv --folds 10`.

## Package layout

| Module | Content | Paper |
|---|---|---|
| `roge.stl` | STL syntax tree, parser, vectorised quantitative semantics (robustness) | §2, Def. 1–2 |
| `roge.gpucb` | Gaussian process + GP-UCB optimisation of the parameters | §4.2 |
| `roge.genetic` | initial templates, random formulae, mutation, crossover | §4.3 |
| `roge.roge` | discrimination function G, parameter learning, Alg. 1, threshold shift, classifier | §4, Alg. 1 |
| `roge.data` | dataset loaders, stratified splits and k-fold | §5 |

## Data

`data/naval/` is the maritime surveillance dataset of the paper's §5.1 (2000 trajectories, 2 variables, 61 samples). It was introduced by Bombara et al. (HSCC 2016) and is copied here, unchanged, from the original Java implementation of ROGE. It is not covered by this repository's licence.

## Citation

If you use this software, please cite the paper:

> L. Nenzi, S. Silvetti, E. Bartocci, L. Bortolussi. *A Robust Genetic Algorithm for Learning Temporal Specifications from Data.* In: Quantitative Evaluation of Systems (QEST 2018), LNCS 11024, pp. 323–338. Springer, 2018. [doi:10.1007/978-3-319-99154-2_20](https://doi.org/10.1007/978-3-319-99154-2_20)

```bibtex
@inproceedings{DBLP:conf/qest/NenziSBB18,
  author       = {Laura Nenzi and
                  Simone Silvetti and
                  Ezio Bartocci and
                  Luca Bortolussi},
  title        = {A Robust Genetic Algorithm for Learning Temporal Specifications from
                  Data},
  booktitle    = {{QEST}},
  series       = {Lecture Notes in Computer Science},
  volume       = {11024},
  pages        = {323--338},
  publisher    = {Springer},
  year         = {2018}
}
```

The same information is in machine-readable form in [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository").

## Licence

MIT, see [`LICENSE`](LICENSE).

## About this implementation

This Python implementation was developed by vibe coding with [Claude](https://claude.com/claude-code) (Anthropic): the code and documentation were written by Claude under the author's direction.
