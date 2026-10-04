"""A layer that respects rotation, reflection, translation and relabelling.

The source is shared with equivariant.py, where the symmetry is checked to
machine precision and its value is measured, so the layer on the canvas is the
layer that was tested.
"""
from blocks_sdk import Block, Param, ShapeError, install

from equivariant import SOURCE, learnables as _count


def equivariant_infer(p, shapes):
    s = list(shapes[0])
    d = int(p["coords"])
    if len(s) != 2:
        raise ShapeError(f"EquivariantLayer reads a set of points, [N, coords + "
                         f"features]; got {s}.")
    if s[1] < d:
        raise ShapeError(f"Each point needs at least {d} coordinates, but rows "
                         f"have {s[1]} numbers.")
    return [s[0], d + int(p["out_features"])]


install(Block(
    name="EquivariantLayer",
    category="Graphs",
    doc="E(n)-equivariant message passing (Satorras et al. 2021). Rotating, "
        "reflecting or translating the points moves the output coordinates "
        "the same way and leaves the features unchanged; relabelling the "
        "points relabels the output. Exactly, for any weights.",
    params=[
        Param("coords", "int", 3, min=1,
              help="How many numbers at the start of each row are coordinates"),
        Param("hidden", "int", 32, min=1),
        Param("out_features", "int", 16, min=1),
        Param("update_coords", "bool", True,
              help="Move the points, or only update their features"),
    ],
    infer=equivariant_infer,
    learnables=lambda p, ins, out: _count(
        ins[0][-1] - int(p["coords"]), int(p["hidden"]),
        int(p["out_features"]), bool(p["update_coords"])),
    prelude=SOURCE,
    torch_init=lambda p, ins: (
        "EquivariantLayer({d}, {f}, {h}, {o}, update_coords={u})".format(
            d=int(p["coords"]), f=ins[0][-1] - int(p["coords"]),
            h=int(p["hidden"]), o=int(p["out_features"]),
            u=bool(p["update_coords"]))),
))
