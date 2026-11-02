"""Physics as architecture: the Lagrangian field.

Source is shared with lawsource.py, where laws.py measures how much data the
law saves, so the block on the canvas is the layer that was measured.
"""
from blocks_sdk import Block, Param, ShapeError, install

from lawsource import SOURCES, learnables


def even_infer(name):
    def infer(p, shapes):
        s = list(shapes[0])
        if not s or s[-1] % 2:
            raise ShapeError(f"{name} splits its input into positions and velocities, "
                             f"so the last dimension must be even; got {s}.")
        return s
    return infer


install(Block(
    name="LagrangianField",
    category="Laws · physics",
    doc="Learns a Lagrangian L(q, q̇) and returns the motion Euler–Lagrange "
        "implies, (q̇, q̈). Energy is conserved for any weights, and it works in "
        "measured coordinates — no momenta needed. Cranmer et al. 2020.",
    params=[
        Param("hidden", "int", 64, min=1),
        Param("eps", "float", 1e-3, help="Added to the mass matrix before solving, for stability"),
        Param("activation", "select", "tanh", options=["tanh", "softplus"],
              help="Inside the networks that make L; both are smooth enough to differentiate twice"),
        Param("form", "select", "mechanical", options=["mechanical", "general"],
              help="mechanical: L = ½q̇ᵀM(q)q̇ − V(q) with M positive definite (stable). general: any L"),
    ],
    infer=even_infer("LagrangianField"),
    learnables=lambda p, ins, out: learnables("LagrangianField", dim=ins[0][-1], hidden=int(p["hidden"]),
                                              form=str(p.get("form", "mechanical"))),
    prelude=SOURCES["LagrangianField"],
    torch_init=lambda p, ins: (f"LagrangianField({ins[0][-1]}, {int(p['hidden'])}, eps={float(p['eps'])}, "
                               f"form={str(p.get('form', 'mechanical'))!r}, activation={str(p.get('activation', 'tanh'))!r})"),
))
