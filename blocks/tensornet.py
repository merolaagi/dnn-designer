"""A linear layer stored as a tensor train: see tensornet.py, where it is
compared with ordinary low-rank compression at matched parameter counts."""
from blocks_sdk import Block, Param, ShapeError, install

from tensornet import SOURCE, learnables as _count


def _modes(text, name):
    try:
        modes = [int(m) for m in str(text).replace(" ", "").split(",") if m]
    except ValueError:
        raise ShapeError(f"{name} must be whole numbers separated by commas.")
    if not modes or any(m <= 0 for m in modes):
        raise ShapeError(f"{name} must be positive whole numbers.")
    return modes


def tt_infer(p, shapes):
    import math

    s = list(shapes[0])
    ins = _modes(p["in_modes"], "in_modes")
    outs = _modes(p["out_modes"], "out_modes")
    if len(ins) != len(outs):
        raise ShapeError(f"in_modes has {len(ins)} factors and out_modes "
                         f"{len(outs)}; they need the same number.")
    if not s or s[-1] != math.prod(ins):
        raise ShapeError(f"in_modes {ins} multiply to {math.prod(ins)}, but the "
                         f"input's last dimension is {s[-1] if s else '?'}.")
    return s[:-1] + [math.prod(outs)]


install(Block(
    name="TensorTrainLinear",
    category="Dense",
    doc="A linear layer whose weight is a chain of small cores (Novikov et "
        "al. 2015): parameters grow with the sum of the mode sizes, not their "
        "product. Measured here, it beats ordinary low-rank compression only "
        "below about 1/16 of the dense size; above that, truncated SVD wins.",
    params=[
        Param("in_modes", "text", "4,4,4,4",
              help="Factors of the input width, e.g. 256 = 4,4,4,4"),
        Param("out_modes", "text", "4,4,2,2",
              help="Factors of the output width, the same number of them"),
        Param("rank", "int", 4, min=1),
        Param("bias", "bool", True),
    ],
    infer=tt_infer,
    learnables=lambda p, ins, out: _count(
        _modes(p["in_modes"], "in_modes"), _modes(p["out_modes"], "out_modes"),
        int(p["rank"]), bool(p["bias"])),
    prelude=SOURCE,
    torch_init=lambda p, ins: "TensorTrainLinear({i}, {o}, rank={r}, bias={b})".format(
        i=_modes(p["in_modes"], "in_modes"), o=_modes(p["out_modes"], "out_modes"),
        r=int(p["rank"]), b=bool(p["bias"])),
))
