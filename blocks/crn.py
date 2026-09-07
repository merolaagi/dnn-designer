"""The Deficiency-Zero equilibrium layer, as a layer you can place.

An implicit layer: instead of computing an output from its input directly, it
solves for the equilibrium of a chemical reaction network seeded by the input.
What makes it unusual is where the guarantee comes from. Most implicit layers
need a constraint on the weights — a spectral norm, a monotonicity margin, a
projection after every step — to be sure a fixed point exists and is unique.
This one gets that from the *topology of the reaction graph*, which is fixed at
design time and never trained. The Feinberg / Horn–Jackson Deficiency Zero
Theorem says that a weakly reversible network with deficiency zero has exactly
one positive equilibrium in each compatibility class, for **any** positive rate
constants. So the rates are parameterised k = exp(theta) with theta free over
all of R, and no projection is ever needed.

The mathematics is in `crn_deq.py`, which came from a separate piece of work and
is vendored unchanged. The deficiency is computed exactly over the rationals
rather than by a floating-point rank, because the whole guarantee turns on that
integer being zero.
"""

from blocks_sdk import Block, Param, ShapeError, install

CRN_PRELUDE = '''
# The reaction-network machinery lives in crn_deq.py, which must sit beside this
# file to run it. It is not inlined because it is several hundred lines and is
# maintained separately.
from crn_deq import CRNBlock, generate_network
'''


def _network_for(p):
    """Build the reaction graph for these settings.

    Deterministic in (species, classes, extra edges, deficiency, seed), so the
    parameter count the canvas reports is the count the generated model has.
    """
    from crn_deq import generate_network

    species = int(p.get("species", 6))
    classes = max(1, int(p.get("classes", 2)))
    base = species // classes
    sizes = [base] * classes
    for i in range(species - base * classes):
        sizes[i] += 1
    sizes = [s for s in sizes if s >= 2] or [species]
    return generate_network(
        species, sizes,
        extra_edges=int(p.get("extra_edges", 0)),
        target_deficiency=int(p.get("deficiency", 0)),
        seed=int(p.get("seed", 0)),
    )


def crn_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 1:
        raise ShapeError(
            f"This layer works on a flat vector; {len(shape)} dimensions "
            f"arrived. Flatten or pool first.")
    if int(p.get("species", 6)) < 2 * max(1, int(p.get("classes", 2))):
        raise ShapeError(
            "Each linkage class needs at least two complexes, so species must "
            "be at least twice the number of classes.")
    return [int(p.get("units", 8))]


def crn_learnables(p, ins, out):
    """encoder + one rate per reaction + decoder.

    The rate count comes from the graph itself, so this builds it. Cheap: a few
    milliseconds, and the result is cached by the loader's parameter panel.
    """
    try:
        net = _network_for(p)
    except Exception:  # noqa: BLE001 - an impossible graph is reported by infer
        return 0
    d_in = ins[0][0] if ins and ins[0] else 0
    d_out = int(p.get("units", 8))
    species = net.n
    return (d_in * species + species) + net.n_edges + (species * d_out + d_out)


install(Block(
    name="EquilibriumCRN",
    category="Implicit",
    doc="Solves a reaction network to equilibrium instead of computing forward. "
        "Its fixed point is guaranteed by the graph's topology — deficiency zero "
        "and weakly reversible — so the rates train unconstrained, with no "
        "projection or spectral norm.",
    params=[
        Param("units", "int", 8, min=1, help="Width leaving the layer"),
        Param("species", "int", 6, min=2, help="Chemical species: the solver's "
                                               "internal width"),
        Param("classes", "int", 2, min=1, help="Linkage classes in the graph"),
        Param("extra_edges", "int", 0, min=0, help="Reactions beyond the "
                                                   "spanning cycles"),
        Param("deficiency", "choice", "0", options=["0", "1"],
              help="0 is the guaranteed case; 1 is for comparison and carries "
                   "no guarantee"),
        Param("seed", "int", 0, min=0, help="Which graph of this shape"),
        Param("tol", "float", 1e-10, help="Equilibrium solver tolerance"),
        Param("max_iter", "int", 200, min=1),
    ],
    infer=crn_infer,
    learnables=crn_learnables,
    prelude=CRN_PRELUDE,
    torch_init=lambda p, ins: (
        f"CRNBlock({ins[0][0]}, {int(p['units'])}, "
        f"generate_network({int(p['species'])}, "
        f"{_sizes_literal(p)}, "
        f"extra_edges={int(p['extra_edges'])}, "
        f"target_deficiency={int(p['deficiency'])}, "
        f"seed={int(p['seed'])}), "
        f"dtype=torch.float64, tol={float(p['tol'])}, "
        f"max_iter={int(p['max_iter'])}, verify=False)"
    ),
))


def _sizes_literal(p) -> str:
    species = int(p.get("species", 6))
    classes = max(1, int(p.get("classes", 2)))
    base = species // classes
    sizes = [base] * classes
    for i in range(species - base * classes):
        sizes[i] += 1
    sizes = [s for s in sizes if s >= 2] or [species]
    return repr(sizes)
