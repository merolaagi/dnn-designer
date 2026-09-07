"""
dnn_bench/domains/crn_network.py — reaction-network math for the CRN domain.

Pure structure: no torch layers here. Exact deficiency over Q, weak
reversibility, stoichiometric subspace, conservation laws, and generators for
the three arms.

An implicit layer whose well-posedness comes from the TOPOLOGY of a reaction
graph (an integer computed once at design time) rather than from any constraint
on the trainable weights.

Feinberg / Horn-Jackson Deficiency Zero Theorem:
    If a reaction network is weakly reversible and has deficiency
        delta = m - l - s = 0
    (m complexes, l linkage classes, s = dim of the stoichiometric subspace)
    then for ANY positive rate constants there is exactly one positive
    equilibrium in each stoichiometric compatibility class, and it is
    asymptotically stable, with Lyapunov function
        V(x) = sum_i [ x_i log(x_i / x_i*) - x_i + x_i* ].

Consequence for ML: rates are parameterised k = exp(theta) with theta free over
all of R^E. No projection, no spectral normalisation, no monotonicity margin to
monitor -- every point in parameter space is well-posed by construction.

Contrast with monDEQ (arXiv:2006.08591) and friends, which all constrain W.

Layer semantics: the input enters as the INITIAL CONDITION x0 > 0. The output
is the unique equilibrium in x0's compatibility class, x* in x0 + S. So the map
exactly preserves the conserved quantities (left null space of YB) -- a hard
invariance that falls out rather than being bolted on.

Local asymptotic stability is a theorem; global stability is the Global
Attractor Conjecture, proven only for dim(S) < 3. Do not overclaim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# exact linear algebra over Q  (deficiency is an integer; do not trust float SVD)
# ----------------------------------------------------------------------------

def exact_rank(M: np.ndarray) -> int:
    """Rank over Q of an integer matrix, by fraction-based elimination."""
    M = np.asarray(M)
    if M.size == 0:
        return 0
    A = [[Fraction(int(round(v))) for v in row] for row in M]
    rows, cols = len(A), len(A[0])
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c] / pv
                for j in range(c, cols):
                    A[i][j] -= f * A[r][j]
        r += 1
        if r == rows:
            break
    return r


# ----------------------------------------------------------------------------
# reaction network
# ----------------------------------------------------------------------------

@dataclass
class ReactionNetwork:
    """Y: (n, m) non-negative integer complex matrix, columns are complexes.
    edges: list of (source_complex, target_complex) index pairs."""
    Y: np.ndarray
    edges: List[Tuple[int, int]]

    def __post_init__(self):
        self.Y = np.asarray(self.Y, dtype=np.int64)
        # plain Python ints: numpy scalars leak into JSON responses otherwise
        self.edges = [(int(i), int(j)) for i, j in self.edges]
        assert (self.Y >= 0).all(), "stoichiometric coefficients must be >= 0"

    @property
    def n(self) -> int:                 # species  (= hidden width)
        return self.Y.shape[0]

    @property
    def m(self) -> int:                 # complexes
        return self.Y.shape[1]

    @property
    def n_edges(self) -> int:           # reactions (= number of parameters)
        return len(self.edges)

    @property
    def B(self) -> np.ndarray:
        """Incidence matrix, (m, E)."""
        B = np.zeros((self.m, self.n_edges), dtype=np.int64)
        for e, (i, j) in enumerate(self.edges):
            B[i, e] -= 1
            B[j, e] += 1
        return B

    @property
    def YB(self) -> np.ndarray:
        return self.Y @ self.B

    def linkage_classes(self) -> List[List[int]]:
        """Connected components of the UNDIRECTED reaction graph."""
        parent = list(range(self.m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in self.edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        groups: dict = {}
        for v in range(self.m):
            groups.setdefault(find(v), []).append(v)
        return list(groups.values())

    def strongly_connected_components(self) -> List[List[int]]:
        """Iterative Tarjan."""
        adj = [[] for _ in range(self.m)]
        for i, j in self.edges:
            adj[i].append(j)
        index = [None] * self.m
        low = [0] * self.m
        on_stack = [False] * self.m
        stack: List[int] = []
        result: List[List[int]] = []
        counter = 0
        for root in range(self.m):
            if index[root] is not None:
                continue
            work = [(root, 0)]
            while work:
                v, pi = work[-1]
                if pi == 0:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack[v] = True
                recurse = False
                for i in range(pi, len(adj[v])):
                    w = adj[v][i]
                    if index[w] is None:
                        work[-1] = (v, i + 1)
                        work.append((w, 0))
                        recurse = True
                        break
                    elif on_stack[w]:
                        low[v] = min(low[v], index[w])
                if recurse:
                    continue
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp.append(w)
                        if w == v:
                            break
                    result.append(comp)
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
        return result

    def is_weakly_reversible(self) -> bool:
        """Every linkage class is strongly connected."""
        sccs = {frozenset(c) for c in self.strongly_connected_components()}
        return all(frozenset(lc) in sccs for lc in self.linkage_classes())

    def deficiency(self) -> int:
        """delta = m - l - s, computed exactly."""
        return self.m - len(self.linkage_classes()) - exact_rank(self.YB)

    def stoichiometric_basis(self) -> np.ndarray:
        """Orthonormal basis of S = im(YB), shape (n, s)."""
        U, sv, _ = np.linalg.svd(self.YB.astype(np.float64), full_matrices=False)
        s = exact_rank(self.YB)
        return U[:, :s]

    def conservation_laws(self) -> np.ndarray:
        """Orthonormal basis of the left null space of YB, shape (n, n-s).
        Each column is a quantity the layer preserves EXACTLY."""
        U, sv, _ = np.linalg.svd(self.YB.astype(np.float64), full_matrices=True)
        s = exact_rank(self.YB)
        return U[:, s:]

    def summary(self) -> str:
        s = exact_rank(self.YB)
        return (f"n={self.n} species, m={self.m} complexes, "
                f"l={len(self.linkage_classes())} linkage classes, "
                f"s={s}, delta={self.deficiency()}, "
                f"E={self.n_edges} reactions, "
                f"weakly_reversible={self.is_weakly_reversible()}, "
                f"{self.n - s} conservation law(s)")


# ----------------------------------------------------------------------------
# network generation
# ----------------------------------------------------------------------------

def _cycle_edges(nodes: List[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    """A random Hamiltonian cycle -> guarantees strong connectivity."""
    p = list(rng.permutation(nodes))
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def _build_edges(classes: List[List[int]], extra: int,
                 rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for nodes in classes:
        edges += _cycle_edges(nodes, rng) if len(nodes) > 1 else []
    pool = []
    for nodes in classes:
        for a in nodes:
            for b in nodes:
                if a != b and (a, b) not in edges:
                    pool.append((a, b))
    if pool and extra > 0:
        pick = rng.choice(len(pool), size=min(extra, len(pool)), replace=False)
        edges += [pool[i] for i in np.atleast_1d(pick)]
    return edges


def _sample_Y(n: int, m: int, rng: np.random.Generator, max_order: int = 2) -> np.ndarray:
    """Sparse non-negative integer complexes. Total order >= 2 for some columns
    so the kinetics are genuinely nonlinear (order 1 everywhere => linear ODE)."""
    Y = np.zeros((n, m), dtype=np.int64)
    for c in range(m):
        order = int(rng.integers(1, max_order + 1))
        idx = rng.choice(n, size=order, replace=True)
        for i in idx:
            Y[i, c] += 1
    return Y


def generate_network(n: int, class_sizes: List[int], extra_edges: int = 0,
                     target_deficiency: int = 0, seed: int = 0,
                     max_tries: int = 20000) -> ReactionNetwork:
    """Rejection-sample a weakly reversible network with the requested deficiency.

    Because the edge structure (Hamiltonian cycles + `extra_edges`) is fixed by
    class_sizes, delta=0 and delta=1 networks generated this way have IDENTICAL
    parameter counts -- which is what makes the ablation fair.
    """
    m = sum(class_sizes)
    if m - len(class_sizes) > n:
        raise ValueError(f"need s = m - l = {m - len(class_sizes)} <= n = {n}")
    rng = np.random.default_rng(seed)
    classes, k = [], 0
    for sz in class_sizes:
        classes.append(list(range(k, k + sz)))
        k += sz
    edges = _build_edges(classes, extra_edges, rng)

    for _ in range(max_tries):
        Y = _sample_Y(n, m, rng)
        if target_deficiency > 0:
            # force a rank drop: make one complex an affine combination of three
            # others in the same class (coefficients 1,1,-1 sum to 1).
            for _ in range(60):
                cls = classes[int(rng.integers(len(classes)))]
                if len(cls) < 4:
                    continue
                a, b, c, d = rng.choice(cls, size=4, replace=False)
                cand = Y[:, a] + Y[:, b] - Y[:, c]
                if (cand >= 0).all() and cand.sum() > 0:
                    Y = Y.copy()
                    Y[:, d] = cand
                    break
        net = ReactionNetwork(Y, edges)
        if net.deficiency() == target_deficiency and net.is_weakly_reversible():
            if len(set(map(tuple, Y.T))) == m:   # distinct complexes
                return net
    raise RuntimeError(
        f"could not sample delta={target_deficiency} network; "
        f"try different class_sizes or a larger n")


def break_weak_reversibility(net: ReactionNetwork,
                             seed: int = 0) -> ReactionNetwork:
    """Flip ONE edge direction so the network is no longer weakly reversible.

    Reversing an edge negates a column of B, which changes neither rank(B) nor
    rank(YB) -- so the deficiency is unchanged. Complexes, edge count and
    parameter count are all identical. This isolates weak reversibility as the
    single varying factor.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(net.edges))
    for e in order:
        edges = list(net.edges)
        i, j = edges[e]
        edges[e] = (j, i)
        cand = ReactionNetwork(net.Y.copy(), edges)
        if not cand.is_weakly_reversible() and cand.deficiency() == net.deficiency():
            return cand
    raise RuntimeError("no single edge flip broke weak reversibility")


