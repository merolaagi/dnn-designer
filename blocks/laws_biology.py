"""Neuroscience as architecture: Dale's law and Hebbian fast weights.

Source is shared with lawsource.py, where laws.py measures what each buys.
"""
from blocks_sdk import Block, Param, ShapeError, install

from lawsource import SOURCES, learnables


install(Block(
    name="DaleLinear",
    category="Laws · biology",
    doc="Dale's law: every input neuron is excitatory or inhibitory, never "
        "both. Each weight column keeps one sign for any weights; the first "
        "frac of inputs are E (cortex is ~80% E). Song, Yang & Wang 2016.",
    params=[
        Param("units", "int", 64, min=1),
        Param("frac", "float", 0.8, help="Fraction of input neurons that are excitatory"),
    ],
    infer=lambda p, ins: list(ins[0][:-1]) + [int(p["units"])],
    learnables=lambda p, ins, out: learnables("DaleLinear", dim=ins[0][-1], units=int(p["units"])),
    prelude=SOURCES["DaleLinear"],
    torch_init=lambda p, ins: f"DaleLinear({ins[0][-1]}, {int(p['units'])}, frac={float(p['frac'])})",
))


def sequence_infer(name):
    def infer(p, shapes):
        s = list(shapes[0])
        if len(s) != 2:
            raise ShapeError(f"{name} reads a sequence as [L, features]; got {s}.")
        return s
    return infer


install(Block(
    name="HebbianFastWeights",
    category="Laws · biology",
    doc="Fast synapses written by a local Hebbian rule, A ← λA + η hhᵀ, never "
        "by the optimizer. Recalls the recent past; λ is forgetting. Causal over "
        "[L, features]. Ba et al. 2016.",
    params=[
        Param("decay", "float", 0.9, help="λ: how much of the fast memory survives each step"),
        Param("rate", "float", 0.5, help="η: how strongly each step is written"),
    ],
    infer=sequence_infer("HebbianFastWeights"),
    learnables=lambda p, ins, out: learnables("HebbianFastWeights", dim=ins[0][-1]),
    prelude=SOURCES["HebbianFastWeights"],
    torch_init=lambda p, ins: (f"HebbianFastWeights({ins[0][-1]}, decay={float(p['decay'])}, "
                               f"rate={float(p['rate'])})"),
))
