"""STL / PSTL formulae and their quantitative semantics (paper §2, Def. 1-2).

Formulae are trees of mutable nodes. Parameters live on the nodes:
``Atom.threshold`` and ``Temporal.interval``. A formula's parameter vector is
ordered as all intervals ``(a, b)`` of the temporal nodes in pre-order,
followed by the atom thresholds in pre-order (the same layout as the Java implementation,
``Formula.getParameters``), but values are assigned by position on the tree,
never by string substitution.

Robustness is computed on sampled traces, vectorised over trajectories:
``robustness(phi, data)`` returns an array ``(n_traj, n_samples)`` with
``rho(phi, x, t_i)`` for every sample time ``t_i``.

Discrete-time choices (not specified by the paper, documented in docs/DESIGN.md):
- the window ``t + [a, b]`` is the set of samples with time in that interval;
- if it contains no sample, the first sample after ``t + a`` is used, or the
  last sample if ``t + a`` is past the end of the trace (the trace is held
  constant after its last sample). This keeps robustness finite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

_EPS = 1e-9


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Traces:
    """A set of sampled trajectories sharing the same time grid.

    ``values`` has shape ``(n_traj, n_vars, n_samples)``.
    """

    times: np.ndarray
    values: np.ndarray
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 3:
            raise ValueError("values must have shape (n_traj, n_vars, n_samples)")
        if self.values.shape[1] != len(self.variables):
            raise ValueError("values.shape[1] != len(variables)")
        if self.values.shape[2] != len(self.times):
            raise ValueError("values.shape[2] != len(times)")
        self._index = {v: i for i, v in enumerate(self.variables)}

    def __len__(self) -> int:
        return self.values.shape[0]

    def signal(self, var: str) -> np.ndarray:
        return self.values[:, self._index[var], :]

    def subset(self, idx) -> "Traces":
        return Traces(self.times, self.values[idx], self.variables)


# --------------------------------------------------------------------------
# Syntax
# --------------------------------------------------------------------------


class Node:
    children: list["Node"]

    def copy(self) -> "Node":
        raise NotImplementedError

    def walk(self) -> Iterator["Node"]:
        """Pre-order traversal."""
        yield self
        for c in self.children:
            yield from c.walk()

    def size(self) -> int:
        return sum(1 for _ in self.walk())

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self, digits: int = 4) -> str:
        raise NotImplementedError


class Atom(Node):
    """``var > threshold`` or ``var <= threshold`` (paper §2, PSTL)."""

    def __init__(self, var: str, op: Literal[">", "<="], threshold: float | None = None):
        if op not in (">", "<="):
            raise ValueError(f"unsupported operator {op!r}")
        self.var, self.op, self.threshold = var, op, threshold
        self.children = []

    def copy(self) -> "Atom":
        return Atom(self.var, self.op, self.threshold)

    def to_str(self, digits: int = 4) -> str:
        k = "?" if self.threshold is None else f"{self.threshold:.{digits}g}"
        return f"({self.var} {self.op} {k})"


class Not(Node):
    def __init__(self, child: Node):
        self.children = [child]

    def copy(self) -> "Not":
        return Not(self.children[0].copy())

    def to_str(self, digits: int = 4) -> str:
        return f"!{self.children[0].to_str(digits)}"


class _Binary(Node):
    symbol = ""

    def __init__(self, left: Node, right: Node):
        self.children = [left, right]

    def copy(self):
        return type(self)(self.children[0].copy(), self.children[1].copy())

    def to_str(self, digits: int = 4) -> str:
        l, r = (c.to_str(digits) for c in self.children)
        return f"({l} {self.symbol} {r})"


class And(_Binary):
    symbol = "&"


class Or(_Binary):
    symbol = "|"


class Temporal(Node):
    """Base class of the temporal operators; ``interval = (a, b)``, ``a <= b``."""

    symbol = ""
    interval: tuple[float, float] | None

    def _interval_str(self, digits: int) -> str:
        if self.interval is None:
            return "[?,?]"
        a, b = self.interval
        return f"[{a:.{digits}g},{b:.{digits}g}]"


class Eventually(Temporal):
    symbol = "F"

    def __init__(self, child: Node, interval: tuple[float, float] | None = None):
        self.children = [child]
        self.interval = interval

    def copy(self) -> "Eventually":
        return Eventually(self.children[0].copy(), self.interval)

    def to_str(self, digits: int = 4) -> str:
        return f"F{self._interval_str(digits)} {self.children[0].to_str(digits)}"


class Globally(Temporal):
    symbol = "G"

    def __init__(self, child: Node, interval: tuple[float, float] | None = None):
        self.children = [child]
        self.interval = interval

    def copy(self) -> "Globally":
        return Globally(self.children[0].copy(), self.interval)

    def to_str(self, digits: int = 4) -> str:
        return f"G{self._interval_str(digits)} {self.children[0].to_str(digits)}"


class Until(Temporal):
    symbol = "U"

    def __init__(self, left: Node, right: Node, interval: tuple[float, float] | None = None):
        self.children = [left, right]
        self.interval = interval

    def copy(self) -> "Until":
        return Until(self.children[0].copy(), self.children[1].copy(), self.interval)

    def to_str(self, digits: int = 4) -> str:
        l, r = (c.to_str(digits) for c in self.children)
        return f"({l} U{self._interval_str(digits)} {r})"


UNARY = (Not, Eventually, Globally)
BINARY = (And, Or, Until)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


def temporal_nodes(phi: Node) -> list[Temporal]:
    return [n for n in phi.walk() if isinstance(n, Temporal)]


def atoms(phi: Node) -> list[Atom]:
    return [n for n in phi.walk() if isinstance(n, Atom)]


def n_parameters(phi: Node) -> int:
    return 2 * len(temporal_nodes(phi)) + len(atoms(phi))


def get_parameters(phi: Node) -> np.ndarray:
    out: list[float] = []
    for t in temporal_nodes(phi):
        out.extend(t.interval if t.interval is not None else (np.nan, np.nan))
    for a in atoms(phi):
        out.append(np.nan if a.threshold is None else a.threshold)
    return np.asarray(out, dtype=float)


def set_parameters(phi: Node, theta) -> None:
    theta = np.asarray(theta, dtype=float)
    temps, ats = temporal_nodes(phi), atoms(phi)
    if len(theta) != 2 * len(temps) + len(ats):
        raise ValueError("wrong number of parameters")
    for i, t in enumerate(temps):
        a, b = float(theta[2 * i]), float(theta[2 * i + 1])
        if a > b:
            raise ValueError(f"interval [{a}, {b}] with a > b")
        t.interval = (a, b)
    for k, at in enumerate(ats):
        at.threshold = float(theta[2 * len(temps) + k])


# --------------------------------------------------------------------------
# Quantitative semantics (paper Def. 2)
# --------------------------------------------------------------------------

UntilSemantics = Literal["paper", "java"]


def _window(times: np.ndarray, i: int, a: float, b: float) -> tuple[int, int]:
    """Sample indices ``lo..hi`` (inclusive) with time in ``t_i + [a, b]``."""
    n = len(times)
    t = times[i]
    lo = int(np.searchsorted(times, t + a - _EPS, side="left"))
    hi = int(np.searchsorted(times, t + b + _EPS, side="right")) - 1
    if lo >= n:
        return n - 1, n - 1
    if hi < lo:
        return lo, lo
    return lo, hi


def robustness(phi: Node, data: Traces, until: UntilSemantics = "paper") -> np.ndarray:
    """``rho(phi, x, t_i)`` for every trajectory and sample: shape ``(N, T)``.

    ``until="java"`` reproduces the original Java ``MitlUntil.evaluateValue``
    for comparison purposes only; it differs from Def. 2.
    """
    times = data.times
    T = len(times)

    def rec(n: Node) -> np.ndarray:
        if isinstance(n, Atom):
            if n.threshold is None:
                raise ValueError(f"unbound threshold in {n}")
            x = data.signal(n.var)
            return x - n.threshold if n.op == ">" else n.threshold - x
        if isinstance(n, Not):
            return -rec(n.children[0])
        if isinstance(n, And):
            return np.minimum(rec(n.children[0]), rec(n.children[1]))
        if isinstance(n, Or):
            return np.maximum(rec(n.children[0]), rec(n.children[1]))
        if isinstance(n, Temporal):
            if n.interval is None:
                raise ValueError(f"unbound interval in {n}")
            a, b = n.interval
            if isinstance(n, (Eventually, Globally)):
                r = rec(n.children[0])
                red = np.max if isinstance(n, Eventually) else np.min
                out = np.empty_like(r)
                for i in range(T):
                    lo, hi = _window(times, i, a, b)
                    out[:, i] = red(r[:, lo : hi + 1], axis=1)
                return out
            r1, r2 = rec(n.children[0]), rec(n.children[1])
            if until == "java":
                return _until_java(times, r1, r2, a, b)
            return _until_paper(times, r1, r2, a, b)
        raise TypeError(f"unknown node {type(n).__name__}")

    return rec(phi)


def _until_paper(times, r1, r2, a, b) -> np.ndarray:
    """sup_{t' in t+[a,b]} min(rho2(t'), inf_{t'' in [t,t')} rho1(t''))."""
    N, T = r1.shape
    out = np.empty_like(r1)
    for i in range(T):
        lo, hi = _window(times, i, a, b)
        # prefix[:, m] = min rho1 over samples i .. i+m-1  (m = 0 -> +inf)
        prefix = np.empty((N, hi - i + 1))
        prefix[:, 0] = np.inf
        if hi > i:
            prefix[:, 1:] = np.minimum.accumulate(r1[:, i:hi], axis=1)
        vals = np.minimum(r2[:, lo : hi + 1], prefix[:, lo - i : hi - i + 1])
        out[:, i] = vals.max(axis=1)
    return out


def _until_java(times, r1, r2, a, b) -> np.ndarray:
    """Transcription of the Java ``MitlUntil.evaluateValue``."""
    T = len(times)

    def after(t):  # first sample >= t
        idx = int(np.searchsorted(times, t, side="left"))
        return min(idx, T - 1)

    def until_idx(t):  # last sample <= t
        idx = int(np.searchsorted(times, t, side="right")) - 1
        return max(min(idx, T - 1), 0)

    big = np.finfo(float).max
    out = np.empty_like(r1)
    for t in range(T):
        i1, i2 = after(times[t] + a), until_idx(times[t] + b)
        if i1 > i2 or a == b:
            out[:, t] = np.maximum(r1[:, i2], r2[:, i2])
            continue
        m = np.full(r1.shape[0], -big)
        for i in range(i1 + 1, i2):
            v1 = r1[:, i1:i].min(axis=1)
            m = np.maximum(m, np.minimum(v1, r2[:, i + 1]))
        out[:, t] = m
    return out


def robustness_at_zero(phi: Node, data: Traces, until: UntilSemantics = "paper") -> np.ndarray:
    """``rho(phi, x) = rho(phi, x, 0)`` for every trajectory (shape ``(N,)``)."""
    return robustness(phi, data, until)[:, 0]


# --------------------------------------------------------------------------
# Parser (for tests and for writing formulae by hand)
# --------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<num>-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)|(?P<op><=|>|[()!&|\[\],?])"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_']*))"
)


def parse(text: str) -> Node:
    """Parse e.g. ``(x2 > 22.46) U[49,287] (x1 <= 31.65)``.

    Precedence (high to low): ``!``, ``F[a,b]``, ``G[a,b]`` > ``U[a,b]`` >
    ``&`` > ``|``. ``?`` leaves a parameter unbound.
    """
    toks: list[str] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot tokenise at {text[pos:]!r}")
        toks.append(m.group(m.lastgroup))
        pos = m.end()
    k = 0

    def peek():
        return toks[k] if k < len(toks) else None

    def take(expected=None):
        nonlocal k
        t = peek()
        if t is None or (expected is not None and t != expected):
            raise ValueError(f"expected {expected!r}, got {t!r}")
        k += 1
        return t

    def number():
        t = take()
        return None if t == "?" else float(t)

    def interval():
        take("[")
        a = number()
        take(",")
        b = number()
        take("]")
        return None if a is None or b is None else (a, b)

    def p_or():
        n = p_and()
        while peek() == "|":
            take()
            n = Or(n, p_and())
        return n

    def p_and():
        n = p_until()
        while peek() == "&":
            take()
            n = And(n, p_until())
        return n

    def p_until():
        n = p_unary()
        while peek() == "U":
            take()
            iv = interval()
            n = Until(n, p_unary(), iv)
        return n

    def p_unary():
        t = peek()
        if t == "!":
            take()
            return Not(p_unary())
        if t in ("F", "G") and k + 1 < len(toks) and toks[k + 1] == "[":
            take()
            iv = interval()
            child = p_unary()
            return Eventually(child, iv) if t == "F" else Globally(child, iv)
        if t == "(":
            take()
            n = p_or()
            take(")")
            return n
        var = take()
        op = take()
        if op not in (">", "<="):
            raise ValueError(f"expected '>' or '<=', got {op!r}")
        return Atom(var, op, number())

    phi = p_or()
    if k != len(toks):
        raise ValueError(f"trailing tokens: {toks[k:]}")
    return phi
