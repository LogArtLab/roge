"""ROGE: learning STL classifiers with a robustness-driven genetic algorithm.

Reimplementation of Nenzi, Silvetti, Bartocci, Bortolussi, "A Robust Genetic
Algorithm for Learning Temporal Specifications from Data" (QEST 2018,
arXiv 1711.06202), using the paper as the reference and the original Java
implementation as a guide for the details the paper leaves open
(see docs/DESIGN.md).
"""

from .data import load_java_dataset, load_ucr_txt, split_by_label, stratified_kfold, train_test_split
from .gpucb import GPUCBOptions, gp_ucb
from .roge import (
    Individual,
    ParameterSpace,
    RogeConfig,
    RogeResult,
    STLClassifier,
    discrimination,
    evaluate,
    learn_parameters,
    make_classifier,
    misclassification_rate,
    roge,
    shift_formula,
)
from .stl import Traces, parse, robustness, robustness_at_zero

__all__ = [
    "GPUCBOptions", "Individual", "ParameterSpace", "RogeConfig", "RogeResult",
    "STLClassifier", "Traces", "discrimination", "evaluate", "gp_ucb",
    "learn_parameters", "load_java_dataset", "load_ucr_txt", "make_classifier",
    "misclassification_rate", "parse", "robustness", "robustness_at_zero",
    "roge", "shift_formula", "split_by_label", "stratified_kfold",
    "train_test_split",
]
