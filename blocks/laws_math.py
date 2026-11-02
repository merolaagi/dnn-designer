"""Mathematics as architecture: convexity, Lipschitz bounds, permutation invariance.

Source is shared with lawsource.py, where laws.py measures what each buys.
"""
from blocks_sdk import Block, Param, ShapeError, install

from lawsource import SOURCES, learnables


def vector_infer(name):
    def infer(p, shapes):
        s = list(shapes[0])
        if not s:
            raise ShapeError(f"{name} needs a vector input; got {s}.")
        return s
    return infer


install(Block(
    name="ConvexPotential",
    category="Laws · mathematics",
    doc="Input-convex network: the output is convex in the input for any "
        "weights (non-negative hidden weights, convex non-decreasing "
        "activations). One minimum, no spurious wells. Amos et al. 2017.",
    params=[
        Param("hidden", "int", 64, min=1),
        Param("layers", "int", 2, min=1),
        Param("out", "int", 1, min=1, help="1 for a scalar potential"),
    ],
    infer=lambda p, ins: list(ins[0][:-1]) + [int(p["out"])],
    learnables=lambda p, ins, out: learnables("ConvexPotential", dim=ins[0][-1],
                                              hidden=int(p["hidden"]), layers=int(p["layers"]), out=int(p["out"])),
    prelude=SOURCES["ConvexPotential"],
    torch_init=lambda p, ins: (f"ConvexPotential({ins[0][-1]}, {int(p['hidden'])}, "
                               f"layers={int(p['layers'])}, out={int(p['out'])})"),
))


install(Block(
    name="LipschitzLinear",
    category="Laws · mathematics",
    doc="A linear layer whose operator norm is held at or below a bound L by "
        "exact spectral normalisation, so ‖f(x)−f(x')‖ ≤ L‖x−x'‖ for any weights. "
        "Stack with ReLU for a certified Lipschitz network. Miyato et al. 2018.",
    params=[
        Param("units", "int", 64, min=1),
        Param("bound", "float", 1.0, help="The Lipschitz constant L this layer may not exceed"),
    ],
    infer=lambda p, ins: list(ins[0][:-1]) + [int(p["units"])],
    learnables=lambda p, ins, out: learnables("LipschitzLinear", dim=ins[0][-1], units=int(p["units"])),
    prelude=SOURCES["LipschitzLinear"],
    torch_init=lambda p, ins: f"LipschitzLinear({ins[0][-1]}, {int(p['units'])}, bound={float(p['bound'])})",
))


def set_infer(p, shapes):
    s = list(shapes[0])
    if len(s) != 2:
        raise ShapeError(f"InvariantPool reads a set as [N, features]; got {s}.")
    return [int(p["out"])]


install(Block(
    name="InvariantPool",
    category="Laws · mathematics",
    doc="Deep Sets: ρ(Σ φ(xᵢ)). Exactly permutation-invariant for any weights, "
        "so the order of the N rows cannot matter and the network never needs "
        "to see a second ordering. Zaheer et al. 2017.",
    params=[
        Param("hidden", "int", 64, min=1),
        Param("out", "int", 16, min=1),
        Param("pool", "select", "sum", options=["sum", "mean", "max"]),
    ],
    infer=set_infer,
    learnables=lambda p, ins, out: learnables("InvariantPool", dim=ins[0][-1],
                                              hidden=int(p["hidden"]), out=int(p["out"])),
    prelude=SOURCES["InvariantPool"],
    torch_init=lambda p, ins: (f"InvariantPool({ins[0][-1]}, {int(p['hidden'])}, "
                               f"out={int(p['out'])}, pool={p['pool']!r})"),
))
