"""Dataset loading and splitting.

The format of the original Java implementation (see ``data/naval``) is three CSV files:
``<prefix>Times`` (one row of sample times), ``<prefix>Labels`` (one row of
+1/-1) and ``<prefix>Data`` (one row per trajectory, values **interleaved per
sample**: ``x1(t0), x2(t0), ..., x1(t1), x2(t1), ...``, as read by
``Utils.readMatrixMultiFromFile``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .stl import Traces


def load_java_dataset(directory: str | Path, prefix: str, variables: tuple[str, ...]):
    """Return ``(traces, labels)`` with labels in {+1, -1}."""
    d = Path(directory)
    times = np.loadtxt(d / f"{prefix}Times", delimiter=",", ndmin=1)
    labels = np.loadtxt(d / f"{prefix}Labels", delimiter=",", ndmin=1).astype(int)
    raw = np.loadtxt(d / f"{prefix}Data", delimiter=",", ndmin=2)
    n_traj, n_cols = raw.shape
    n_vars = len(variables)
    if n_cols != n_vars * len(times):
        raise ValueError(f"{n_cols} columns, expected {n_vars} x {len(times)}")
    if n_traj != len(labels):
        raise ValueError(f"{n_traj} trajectories but {len(labels)} labels")
    values = raw.reshape(n_traj, len(times), n_vars).transpose(0, 2, 1)
    return Traces(times, values, tuple(variables)), labels


def split_by_label(traces: Traces, labels: np.ndarray) -> tuple[Traces, Traces]:
    """``(D_p, D_n)``: positive (+1) and negative trajectories."""
    labels = np.asarray(labels)
    return traces.subset(labels > 0), traces.subset(labels <= 0)


def stratified_kfold(labels: np.ndarray, k: int, rng: np.random.Generator):
    """Yield ``(train_idx, test_idx)`` for a stratified k-fold CV."""
    labels = np.asarray(labels)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        for f, part in enumerate(np.array_split(idx, k)):
            folds[f].extend(part.tolist())
    all_idx = np.arange(len(labels))
    for f in range(k):
        test = np.sort(np.asarray(folds[f], dtype=int))
        yield np.setdiff1d(all_idx, test), test


def train_test_split(labels: np.ndarray, train_fraction: float, rng: np.random.Generator):
    """Stratified random split, returns ``(train_idx, test_idx)``."""
    labels = np.asarray(labels)
    train: list[int] = []
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        train.extend(idx[: int(round(train_fraction * len(idx)))].tolist())
    train_idx = np.sort(np.asarray(train, dtype=int))
    return train_idx, np.setdiff1d(np.arange(len(labels)), train_idx)


def load_ucr_txt(path: str | Path, variable: str = "x", positive_label=1):
    """Univariate UCR archive ``*.txt`` file: label followed by the values.

    Sample times are the indices ``0 .. length-1``. Returns ``(traces, labels)``
    with ``+1`` for ``positive_label`` and ``-1`` for every other class.
    """
    raw = np.loadtxt(path, ndmin=2)
    y = raw[:, 0]
    values = raw[:, 1:]
    if np.isnan(values).any():
        raise ValueError("missing values / variable-length series are not supported")
    labels = np.where(y == positive_label, 1, -1)
    times = np.arange(values.shape[1], dtype=float)
    return Traces(times, values[:, None, :], (variable,)), labels
