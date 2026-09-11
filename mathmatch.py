"""Which mathematics in a document is already implemented by a layer here.

A textbook on linear algebra or statistics is mostly definitions, and most of
them have nothing to do with neural networks. A few do, exactly, and the
correspondence is checkable rather than rhetorical: this application already
stores the equation each layer implements, so "the book defines a matrix-vector
product, and `Linear` computes `y = W x + b`" is a statement about two things
that are both written down here.

What this deliberately does not do is claim a paper *justifies* a layer, or
that every equation matched is important. It finds where the same object
appears in both places and shows both, with the layer's own numbers when that
layer is on the canvas. Deciding what follows is the reader's job.

The matching is by vocabulary, which is crude and says so. A book that writes
"inner product" and never "dot product" still matches; a book about convex
optimization matches `convex` for reasons that may be irrelevant. Every match
carries the sentence that produced it so it can be dismissed in a second.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: concept -> (words that name it, layers that implement it, why they are the
#: same thing). The third part is the one that has to be true; the first is
#: only a way of finding candidates.
CONCEPTS: List[Dict[str, Any]] = [
    {
        "key": "matrix_product",
        "name": "Matrix–vector product",
        "words": ["matrix-vector", "matrix vector", "linear map",
                  "linear transformation", "matrix multiplication",
                  "inner product", "dot product"],
        "layers": ["Linear", "Embedding"],
        "why": "A dense layer is exactly a matrix applied to a vector, with a "
               "translation added. Everything a book proves about rank, "
               "nullspace and conditioning of that matrix is true of the layer.",
    },
    {
        "key": "convolution",
        "name": "Convolution",
        "words": ["convolution", "convolve", "cross-correlation", "kernel",
                  "impulse response", "filter bank"],
        "layers": ["Conv2d", "Conv1d", "ConvTranspose2d", "SeparableConv2d"],
        "why": "What the layer computes is cross-correlation rather than the "
               "flipped convolution most books define — the distinction is a "
               "sign on the index, and it matters when comparing to a "
               "textbook formula.",
    },
    {
        "key": "expectation",
        "name": "Mean and variance",
        "words": ["expectation", "expected value", "variance", "standard "
                  "deviation", "sample mean", "moment", "central limit"],
        "layers": ["BatchNorm2d", "BatchNorm1d", "LayerNorm", "GroupNorm"],
        "why": "Normalization subtracts an estimated mean and divides by an "
               "estimated standard deviation. Which sample the estimate is "
               "over is the whole difference between batch, layer and group "
               "normalization.",
    },
    {
        "key": "softmax",
        "name": "Softmax and the exponential family",
        "words": ["softmax", "logistic", "sigmoid", "partition function",
                  "gibbs", "boltzmann distribution", "log-sum-exp"],
        "layers": ["Activation", "Attention", "SelfAttention"],
        "why": "Softmax is the gradient of log-sum-exp and the canonical link "
               "of the exponential family. Attention uses it to turn scores "
               "into weights that sum to one.",
    },
    {
        "key": "gradient",
        "name": "Derivatives and the chain rule",
        "words": ["chain rule", "derivative", "gradient", "jacobian",
                  "partial derivative", "differentiable", "taylor"],
        "layers": [],
        "why": "Backpropagation is the chain rule applied to the graph. No "
               "single layer implements it — every layer participates, which "
               "is why it has no entry of its own.",
    },
    {
        "key": "eigen",
        "name": "Eigenvalues and spectra",
        "words": ["eigenvalue", "eigenvector", "spectral", "singular value",
                  "condition number", "spectral radius", "diagonaliz"],
        "layers": ["ImplicitEquilibrium", "EquilibriumCRN"],
        "why": "An implicit layer solves F(x)=0, and whether that solve "
               "converges is a statement about the eigenvalues of its "
               "Jacobian. The Precision view reports the condition number for "
               "the same reason.",
    },
    {
        "key": "fixed_point",
        "name": "Fixed points and contraction",
        "words": ["fixed point", "banach", "contraction mapping", "converges "
                  "to a unique", "iterative scheme", "picard"],
        "layers": ["ImplicitEquilibrium", "EquilibriumCRN", "ODEBlock"],
        "why": "The contraction mapping theorem is what makes an implicit "
               "layer well defined: one solution, reached from anywhere. This "
               "is the mathematics the Implicit domains page is built on.",
    },
    {
        "key": "ode",
        "name": "Differential equations",
        "words": ["differential equation", "initial value problem",
                  "euler method", "runge-kutta", "vector field", "flow map"],
        "layers": ["ODEBlock", "EquilibriumCRN"],
        "why": "A residual block is one step of Euler's method on a learned "
               "vector field; an ODE block makes that explicit and solves it "
               "properly.",
    },
    {
        "key": "convex",
        "name": "Convexity",
        "words": ["convex", "concave", "jensen", "subgradient", "strongly "
                  "convex", "global minimum"],
        "layers": ["ImplicitEquilibrium"],
        "why": "A strongly convex potential has exactly one stationary point. "
               "That is the guarantee the convex domain carries, and the "
               "reason its control arms break exactly that condition.",
    },
    {
        "key": "divergence",
        "name": "Entropy and divergence",
        "words": ["entropy", "kullback", "kl divergence", "cross entropy",
                  "mutual information", "likelihood"],
        "layers": ["Sampling", "Output"],
        "why": "Cross entropy is the classification loss, and the KL term is "
               "what keeps a variational autoencoder variational. Sampling "
               "provides kl() for exactly this.",
    },
    {
        "key": "markov",
        "name": "Markov chains and stationary distributions",
        "words": ["markov", "transition matrix", "stationary distribution",
                  "detailed balance", "ergodic"],
        "layers": ["RBM", "GraphConv", "MessagePassing"],
        "why": "A restricted Boltzmann machine is sampled by a Markov chain, "
               "and message passing on a graph is one step of a transition "
               "operator.",
    },
    {
        "key": "norm",
        "name": "Norms and distance",
        "words": ["norm", "metric", "cauchy-schwarz", "triangle inequality",
                  "lipschitz", "distance"],
        "layers": ["LayerNorm", "Dropout"],
        "why": "A Lipschitz bound on a layer bounds how far its output can "
               "move when its input does, which is what stability arguments "
               "about deep stacks rest on.",
    },
    {
        "key": "graph",
        "name": "Graphs and adjacency",
        "words": ["adjacency", "incidence matrix", "graph laplacian",
                  "degree matrix", "spanning", "bipartite"],
        "layers": ["GraphConv", "GraphAttention", "MessagePassing"],
        "why": "A graph layer multiplies by a normalized adjacency matrix. "
               "The graph Laplacian is that matrix minus the degree matrix, "
               "and spectral graph theory is about its eigenvalues.",
    },
]


def _sentences(text: str) -> List[str]:
    rough = re.split(r"(?<=[.;:])\s+|\n{2,}", text)
    return [s.strip() for s in rough if 20 < len(s.strip()) < 600]


def find(text: str, graph: Optional[Dict[str, Any]] = None,
         registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Concepts the document names, and the layers that implement them."""
    lowered = text.lower()
    sentences = _sentences(text)
    on_canvas: Dict[str, List[str]] = {}
    for node in (graph or {}).get("nodes", []):
        on_canvas.setdefault(node.get("type", ""), []).append(
            node.get("label") or node.get("type") or "")

    found = []
    for concept in CONCEPTS:
        hits = [w for w in concept["words"] if w in lowered]
        if not hits:
            continue
        # the sentence that produced the match, so it can be dismissed quickly
        quotes = []
        for word in hits[:3]:
            for line in sentences:
                if word in line.lower():
                    quotes.append(line)
                    break
        available = [name for name in concept["layers"]
                     if registry is None or name in registry]
        used = {name: on_canvas[name] for name in available if name in on_canvas}
        found.append({
            "key": concept["key"],
            "name": concept["name"],
            "why": concept["why"],
            "words": hits,
            "mentions": sum(lowered.count(w) for w in hits),
            "quotes": quotes[:2],
            "layers": available,
            "on_canvas": used,
            "score": len(hits) * 2 + min(sum(lowered.count(w) for w in hits), 20)
                     + (6 if used else 0),
        })

    found.sort(key=lambda f: -f["score"])
    covered = {name for f in found for name in f["layers"]}
    return {
        "concepts": found,
        "sentences": len(sentences),
        # what the book does NOT reach: worth saying, because a reader
        # otherwise assumes silence means agreement
        "unmatched_layers": sorted(
            (set(registry or {}) - covered) if registry else set())[:24],
        "on_canvas_unmatched": sorted(
            {t for t in on_canvas if t not in covered
             and t not in ("Input", "Output")}),
    }
