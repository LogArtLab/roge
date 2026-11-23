"""Structural operators of the genetic algorithm (paper §4.3).

- ``initial_templates``: ``F_I mu``, ``G_I mu`` and ``mu1 U_I mu2``, where
  ``mu`` is an atom ``x > k`` / ``x <= k`` or a conjunction/disjunction of two
  distinct atoms (unordered pairs, so that commutative duplicates are not
  generated); the Until operands are distinct atoms;
- ``random_formula``: a random tree of size at most ``s``;
- ``mutate``: a random change on one randomly selected node;
- ``crossover``: swap of two random subtrees.

Operators work on copies and leave parameters as they are: the parameters of
every new formula are re-learned by GP-UCB anyway.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from .stl import (
    And,
    Atom,
    Eventually,
    Globally,
    Node,
    Not,
    Or,
    Temporal,
    Until,
)


def all_atoms(variables) -> list[Atom]:
    return [Atom(v, op) for v in variables for op in (">", "<=")]


def initial_templates(variables) -> list[Node]:
    ats = all_atoms(variables)
    mus: list[Node] = [a.copy() for a in ats]
    for a, b in combinations(ats, 2):
        mus.append(And(a.copy(), b.copy()))
        mus.append(Or(a.copy(), b.copy()))
    out: list[Node] = []
    for mu in mus:
        out.append(Eventually(mu.copy()))
        out.append(Globally(mu.copy()))
    for a, b in permutations(ats, 2):
        out.append(Until(a.copy(), b.copy()))
    return out


def random_atom(variables, rng: np.random.Generator) -> Atom:
    return Atom(variables[rng.integers(len(variables))], ">" if rng.random() < 0.5 else "<=")


def random_formula(variables, size: int, rng: np.random.Generator) -> Node:
    """Random formula with exactly ``size`` nodes (no double negation)."""
    if size <= 1:
        return random_atom(variables, rng)
    if size == 2:
        kind = rng.choice(["not", "F", "G"])
    else:
        kind = rng.choice(["not", "F", "G", "and", "or", "U"])
    if kind in ("not", "F", "G"):
        child = random_formula(variables, size - 1, rng)
        if kind == "not":
            return Eventually(child) if isinstance(child, Not) else Not(child)
        return Eventually(child) if kind == "F" else Globally(child)
    left_size = int(rng.integers(1, size - 1))
    left = random_formula(variables, left_size, rng)
    right = random_formula(variables, size - 1 - left_size, rng)
    return {"and": And, "or": Or, "U": Until}[str(kind)](left, right)


def random_formula_up_to(variables, s: int, rng: np.random.Generator) -> Node:
    """Random formula with size uniform in ``[2, s]``, containing a temporal operator."""
    while True:
        phi = random_formula(variables, int(rng.integers(2, max(s, 2) + 1)), rng)
        if any(isinstance(n, Temporal) for n in phi.walk()):
            return phi


# --------------------------------------------------------------------------
# Tree surgery
# --------------------------------------------------------------------------


def _replace(root: Node, target: Node, new: Node) -> Node:
    """Replace ``target`` (identity) with ``new`` inside ``root``; returns the new root."""
    if root is target:
        return new
    for n in root.walk():
        for i, c in enumerate(n.children):
            if c is target:
                n.children[i] = new
                return root
    raise ValueError("target not in tree")


def _nodes(phi: Node) -> list[Node]:
    return list(phi.walk())


def mutate(phi: Node, variables, rng: np.random.Generator) -> Node:
    """Return a mutated copy of ``phi``: one random change on one random node."""
    root = phi.copy()
    n = _nodes(root)[rng.integers(root.size())]
    moves: list[str] = ["insert"]
    if isinstance(n, Atom):
        moves += ["flip", "flip", "variable"] if len(variables) > 1 else ["flip", "flip"]
    elif isinstance(n, (Eventually, Globally)):
        moves += ["temporal", "temporal", "delete"]
    elif isinstance(n, Until):
        moves += ["temporal", "temporal", "delete"]
    elif isinstance(n, (And, Or)):
        moves += ["boolean", "boolean", "delete"]
    elif isinstance(n, Not):
        moves += ["delete", "delete"]
    move = moves[rng.integers(len(moves))]

    if move == "flip":
        n.op = "<=" if n.op == ">" else ">"
        return root
    if move == "variable":
        others = [v for v in variables if v != n.var]
        n.var = others[rng.integers(len(others))]
        return root
    if move == "boolean":
        new = (Or if isinstance(n, And) else And)(*n.children)
        return _replace(root, n, new)
    if move == "temporal":
        if isinstance(n, Until):
            # keep the target sub-formula: phi1 U phi2 -> F phi2 / G phi2
            kind = Eventually if rng.random() < 0.5 else Globally
            return _replace(root, n, kind(n.children[1], n.interval))
        options = ["U", "G" if isinstance(n, Eventually) else "F"]
        choice = options[rng.integers(2)]
        if choice == "U":
            new = Until(random_atom(variables, rng), n.children[0], n.interval)
        else:
            new = (Globally if choice == "G" else Eventually)(n.children[0], n.interval)
        return _replace(root, n, new)
    if move == "delete":
        keep = n.children[rng.integers(len(n.children))]
        return _replace(root, n, keep)
    # insert a unary operator above the node
    kinds = [k for k in (Not, Eventually, Globally) if not isinstance(n, k)]
    kind = kinds[rng.integers(len(kinds))]
    return _replace(root, n, kind(n))


def crossover(phi1: Node, phi2: Node, rng: np.random.Generator) -> tuple[Node, Node]:
    """Swap a random subtree of ``phi1`` with a random subtree of ``phi2`` (copies)."""
    r1, r2 = phi1.copy(), phi2.copy()
    n1 = _nodes(r1)[rng.integers(r1.size())]
    n2 = _nodes(r2)[rng.integers(r2.size())]
    c1 = _replace(r1, n1, n2.copy())
    c2 = _replace(r2, n2, n1)
    return c1, c2
