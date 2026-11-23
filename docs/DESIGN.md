# Design notes

The paper (Nenzi et al., QEST 2018, arXiv:1711.06202) is the specification. This file lists:
- where the paper leaves a choice open, and what this implementation does;
- where it differs from the [original Java implementation](https://github.com/simonesilvetti/learningformulae) of ROGE.

## Faithful to the paper

- **Algorithm 1** is applied literally:
  - `gen_theta = sample(newg_theta + gen_theta, Ne)`: children enter the evaluated population;
  - parents are drawn uniformly from the sampled sub-population;
  - each formula gets one GP-UCB run, and its reported G is the G of the returned parameters.
- **G (Eq. 1)** uses only the training set. The test set is used only for the final misclassification rate.
- **Until (Def. 2)** follows the paper's quantitative semantics. Tests check it against a brute-force enumeration of the definition. `until="java"` reproduces the semantics of the original Java implementation, for comparison only; it differs from Def. 2.
- **Fitness `F = G − S`** includes the size penalty S (see below).
- **Threshold shift.** Every atom is translated by α with the sign of its polarity (the number of enclosing negations). As a result `ρ(φ') = ρ(φ) − α` holds exactly even under negation, which the paper's argument needs. Thresholds are not clamped.
- **Parameters** are assigned to the nodes of the syntax tree, never by textual substitution. Unordered pairs are used for the commutative ∧/∨ and ordered pairs for U.
- **Splits.** k-fold cross-validation is stratified.

## Choices where the paper is silent or ambiguous

| Topic | Choice |
|---|---|
| Discrete time | The window `t + [a, b]` is the set of samples inside it. If it is empty, the first sample after `t + a` is used, or the last sample if `t + a` is past the end of the trace (the trace is held constant). Robustness is therefore always finite. |
| Size penalty S | Paper §4.3: `S = g·p^size` with `p^5 = 0.5`, and "penalizes formulae with large sizes". Read literally, `p < 1` makes the penalty *decrease* with size. The default `size_penalty="increasing"` uses `g·(1 − p^size)`, which matches both "50 % at size 5" and the stated intent. `"literal"` and `"none"` are available. `g` is the mean G of the current set, clipped at 0. |
| Initial templates | `F µ`, `G µ` with µ ∈ atoms ∪ {µ1 ∧ µ2, µ1 ∨ µ2 over unordered pairs of distinct atoms}, plus `µ1 U µ2` over atoms (44 templates with 2 variables). With `ne=None`, all templates are kept plus `n_random = 10` random formulae. |
| Random formulae | Size uniform in `[2, s]` (`s = 5`), with at least one temporal operator. |
| Mutation | One random node: flip of the inequality, change of variable, F↔G↔U, ∧↔∨, insertion of ¬/F/G above the node, deletion of an operator. |
| Crossover | Exchange of two uniformly chosen sub-trees. |
| `max_size = 15` | Children larger than this are discarded (anti-bloat limit, **not in the paper**; set a large value to disable). |
| GP-UCB | Scheme of the original implementation: 20 initial points, 50 iterations, 200 candidate points, `β = 2(1 + 0.1 log it)`, final refinement on the GP mean. GP regression with scikit-learn (isotropic RBF, maximum-likelihood hyper-parameters, fixed homoskedastic noise). **Acceptance rule (default):** a candidate is evaluated if its UCB reaches the maximum *posterior mean* at the observed points (`best_reference="mean"`), with noise `0.01 ×` range of the initial observations. The original implementation compares the UCB with the maximum *noisy observation* (`best_reference="observed"`), with noise `0.1 ×` range. The paper (§4.2) describes plain GP-UCB and does not specify this rule. On the naval and GunPoint datasets the default evaluates more candidates and reaches a higher G. `GPUCBOptions.java()` gives the original settings. |
| Parameter space | Search in the unit cube. A temporal interval `(u_a, u_b)` maps to `a = u_a`, `b = u_a + u_b(1 − u_a)` (so that `a ≤ b`) and is then rescaled to the time bounds. Thresholds are rescaled to the bounds of their variable. |
| α (threshold shift) | Default `weighted`: `α = (µp σn + µn σp)/(σp + σn)`, always between the two means; after the shift `G = (µp − α)/σp`. Alternatives: `midpoint`, `java` (original implementation), and `train`, which picks the α in `[µn, µp]` with the fewest training errors, breaking ties by the distance from `weighted`. If `µp < µn`, the formula is negated first. |
| Degenerate G | If `σp + σn = 0`, G is ±1000 (0 if the means coincide). |
| Result of Alg. 1 | `argmax G` over the final population. The extra 100-iteration parameter refinement of the Java code, which the paper does not mention, is not performed. |

## Extension point

`RogeConfig.score` replaces G with another function of the two robustness vectors. Its default `None` means G (Eq. 1). It exists only for experiments with non-ROGE variants: anything that sets it is **not ROGE**.

## Verification

- `uv run pytest` runs 33 test cases. They cover:
  - Until against an enumeration of Def. 2;
  - the F/G/Boolean operators and the parser;
  - exactness of the shift under negation;
  - G and the α rules;
  - the GP against an independent hand-written GP (`tests/reference_gp.py`): identical predictions with fixed hyper-parameters, marginal likelihood not worse after fitting;
  - GP-UCB against random search on Branin, with both acceptance rules;
  - uniqueness of the templates and validity of the genetic operators;
  - the naval dataset loader.
- Naval dataset, 10-fold CV: misclassification 0.65 % with the default GP-UCB (seed = fold index), 0.45–0.95 % with `GPUCBOptions.java()` depending on the seed (paper: 0). Most folds recover the structure of the paper's Eq. 2, `(x2 > ~24) U (x1 <= ~35)`, with similar thresholds.
