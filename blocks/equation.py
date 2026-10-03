"""A layer written as its own mathematics: see equation.py."""
from blocks_sdk import Block, Param, ShapeError, install

import equation as E


def equation_infer(p, shapes):
    try:
        return E.infer(p, list(shapes[0]))
    except E.EquationError as exc:
        raise ShapeError(str(exc))


def equation_learnables(p, ins, out):
    import math

    try:
        _, _, shapes = E.compile_spec(p, int(ins[0][-1]))
    except E.EquationError:
        return 0
    return sum(math.prod(s) for s in shapes.values())


install(Block(
    name="Equation",
    category="Custom",
    doc="A layer written as its own mathematics. Declare parameters, write "
        "the forward pass as an expression in x; the code, the equation in "
        "the Maths tab, the output shape and the parameter count all follow "
        "from that one expression.",
    params=[
        Param("label", "text", "MyEquation", help="Name shown on the node"),
        Param("parameters", "code", "W: [16, in]; b: [16]",
              help="name: [dims] = start. A dimension is a number or `in`, "
                   "the input width"),
        Param("forward", "code", "tanh(W @ x + b)",
              help="An expression in x and your parameters. W @ x is the "
                   "linear map Wx; * is elementwise"),
    ],
    infer=equation_infer,
    learnables=equation_learnables,
    prelude=E.PRELUDE,
    torch_init=lambda p, ins: E.constructor(p, int(ins[0][-1])),
))
