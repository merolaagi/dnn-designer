"""Psychology as architecture: a hard working-memory capacity.

Source is shared with lawsource.py, where laws.py measures what it buys.
"""
from blocks_sdk import Block, Param, ShapeError, install

from lawsource import SOURCES, learnables


def wm_infer(p, shapes):
    s = list(shapes[0])
    if len(s) != 2:
        raise ShapeError(f"WorkingMemory reads a sequence as [L, features]; got {s}.")
    return [int(p["slots"]), s[1]]


install(Block(
    name="WorkingMemory",
    category="Laws · psychology",
    doc="A bottleneck of K slots: however long the input, exactly K vectors "
        "leave, chosen by attention. Capacity is architectural (Cowan's 4 ± 1), "
        "which forces chunking downstream. Output [K, features].",
    params=[
        Param("slots", "int", 4, min=1, help="K: how many chunks may be held at once"),
    ],
    infer=wm_infer,
    learnables=lambda p, ins, out: learnables("WorkingMemory", dim=ins[0][-1], slots=int(p["slots"])),
    prelude=SOURCES["WorkingMemory"],
    torch_init=lambda p, ins: f"WorkingMemory({ins[0][-1]}, slots={int(p['slots'])})",
))
