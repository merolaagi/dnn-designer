"""A layer written as its own mathematics.

You declare parameters and write the forward pass as an expression:

    parameters   W: [16, in]    b: [16]    gamma: [1] = 1.0
    forward      tanh(W @ x + b) * sigmoid(gamma * x)

and everything else is derived from that one expression's syntax tree: the
PyTorch code, the equation shown in the Maths tab, the output shape (by
running it), and the parameter count. Because the code and the rendering are
printed from the same tree, they cannot disagree — editing one edits both.

The language is deliberately small. Arithmetic, matrix products, a fixed set
of functions, the input x, your parameters and π. Anything else is refused by
name, for two reasons: an expression the printer cannot render faithfully
would make the equation a lie, and the canvas should not become a place where
arbitrary code runs. x is a column vector per sample, so W @ x means the
linear map Wx; matrices go on the left.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple

FUNCTIONS = {
    # name in the expression: (torch code, how it is written in the maths)
    "tanh": ("torch.tanh", "tanh"),
    "sigmoid": ("torch.sigmoid", "σ"),
    "relu": ("torch.relu", "ReLU"),
    "exp": ("torch.exp", "exp"),
    "log": ("torch.log", "log"),
    "sin": ("torch.sin", "sin"),
    "cos": ("torch.cos", "cos"),
    "sqrt": ("torch.sqrt", "√"),
    "abs": ("torch.abs", "abs"),
    "softplus": ("F.softplus", "softplus"),
    "gelu": ("F.gelu", "GELU"),
}
RESERVED = set(FUNCTIONS) | {"x", "pi", "torch", "F", "nn", "math"}


class EquationError(Exception):
    pass


# --------------------------------------------------------------------------
# reading what was written
# --------------------------------------------------------------------------

def parse_parameters(text: str) -> List[Dict[str, Any]]:
    """`name: [dims] = init`, one per line or separated by semicolons.

    A dimension is a positive integer or `in`, the width of the input.
    """
    found: List[Dict[str, Any]] = []
    for raw in re.split(r"[;\n]", text or ""):
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"([^\W\d]\w*)\s*:\s*\[([^\]]*)\]\s*(?:=\s*(\S+))?",
                         line)
        if not m:
            raise EquationError(
                f"Could not read the parameter “{line}”. Write it as "
                f"name: [dims], for example W: [16, in] or gamma: [1] = 1.0")
        name, dims, init = m.groups()
        if name in RESERVED:
            raise EquationError(f"“{name}” is reserved; call the parameter "
                                f"something else.")
        if any(p["name"] == name for p in found):
            raise EquationError(f"“{name}” is declared twice.")
        shape: List[Any] = []
        for d in [d.strip() for d in dims.split(",") if d.strip()]:
            if d == "in":
                shape.append("in")
            elif d.isdigit() and int(d) > 0:
                shape.append(int(d))
            else:
                raise EquationError(
                    f"{name} has the dimension “{d}”, which is neither a "
                    f"positive whole number nor `in`.")
        if not shape:
            raise EquationError(f"{name} needs at least one dimension.")
        value = None
        if init is not None:
            try:
                value = float(init)
            except ValueError:
                raise EquationError(f"{name} starts at “{init}”, which is not "
                                    f"a number.")
        found.append({"name": name, "shape": shape, "init": value})
    return found


def parse_forward(text: str, names: List[str]) -> ast.expr:
    if not (text or "").strip():
        raise EquationError("The forward expression is empty.")
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise EquationError(f"The forward expression does not read: {exc.msg}.")

    allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult, ast.Pow)

    def check(node):
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_ops):
                raise EquationError(f"The operator "
                                    f"{type(node.op).__name__} is not supported.")
            if isinstance(node.op, ast.MatMult) and not _uses_x(node.right):
                raise EquationError(
                    "In A @ v the right-hand side must depend on x: x is a "
                    "column vector per sample, so write W @ x, not x @ W.")
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            check(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                called = getattr(node.func, "id", ast.unparse(node.func))
                raise EquationError(
                    f"“{called}” is not one of the functions available here: "
                    f"{', '.join(sorted(FUNCTIONS))}.")
            if len(node.args) != 1 or node.keywords:
                raise EquationError(f"{node.func.id} takes exactly one argument.")
            check(node.args[0])
        elif isinstance(node, ast.Name):
            if node.id not in names and node.id not in ("x", "pi"):
                raise EquationError(
                    f"“{node.id}” is used but never declared. Declare it under "
                    f"parameters, or use x for the input.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            pass
        else:
            raise EquationError(
                f"“{ast.unparse(node)}” is not something an equation layer can "
                f"contain: only arithmetic, @, the listed functions, x, your "
                f"parameters, numbers and pi.")

    check(tree)
    if not _uses_x(tree):
        raise EquationError("The expression never uses x, so the layer would "
                            "ignore its input.")
    return tree


def _uses_x(node) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(node))


# --------------------------------------------------------------------------
# one tree, two printers
# --------------------------------------------------------------------------

def to_torch(node) -> str:
    if isinstance(node, ast.BinOp):
        a, b = to_torch(node.left), to_torch(node.right)
        if isinstance(node.op, ast.MatMult):
            return f"_mm({a}, {b})"
        sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
               ast.Pow: "**"}[type(node.op)]
        return f"({a} {sym} {b})"
    if isinstance(node, ast.UnaryOp):
        return f"(-{to_torch(node.operand)})"
    if isinstance(node, ast.Call):
        return f"{FUNCTIONS[node.func.id][0]}({to_torch(node.args[0])})"
    if isinstance(node, ast.Name):
        return "math.pi" if node.id == "pi" else node.id
    return repr(node.value)


_PRECEDENCE = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2,
               ast.MatMult: 3, ast.Pow: 4}
_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
         "epsilon": "ε", "eta": "η", "theta": "θ", "lam": "λ", "mu": "μ",
         "nu": "ν", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
         "omega": "ω"}


def to_maths(node, parent: int = 0) -> str:
    """The same tree, written as mathematics. Wx for a matrix product, ⊙ for
    an elementwise one, superscripts for small integer powers."""
    if isinstance(node, ast.BinOp):
        mine = _PRECEDENCE[type(node.op)]
        a = to_maths(node.left, mine)
        b = to_maths(node.right, mine + (1 if isinstance(
            node.op, (ast.Sub, ast.Div)) else 0))
        if isinstance(node.op, ast.MatMult):
            text = f"{a}{b}" if len(a) == 1 else f"{a} {b}"
        elif isinstance(node.op, ast.Mult):
            scalar = isinstance(node.left, ast.Constant) or \
                isinstance(node.right, ast.Constant)
            text = f"{a}{b}" if scalar and isinstance(node.left, ast.Constant) \
                else f"{a} ⊙ {b}"
        elif isinstance(node.op, ast.Pow):
            power = node.right
            if isinstance(power, ast.Constant) and isinstance(power.value, int):
                return f"{a}{str(power.value).translate(_SUPER)}"
            text = f"{a}^({b})"
        else:
            sym = {ast.Add: "+", ast.Sub: "−", ast.Div: "/"}[type(node.op)]
            text = f"{a} {sym} {b}"
        return f"({text})" if mine < parent else text
    if isinstance(node, ast.UnaryOp):
        return f"−{to_maths(node.operand, 5)}"
    if isinstance(node, ast.Call):
        return f"{FUNCTIONS[node.func.id][1]}({to_maths(node.args[0])})"
    if isinstance(node, ast.Name):
        return "π" if node.id == "pi" else GREEK.get(node.id, node.id)
    value = node.value
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


# --------------------------------------------------------------------------
# what the canvas needs
# --------------------------------------------------------------------------

PRELUDE = '''
def _mm(M, v):
    """W @ x with x a column vector per sample: the linear map Wx."""
    return v @ M.transpose(-1, -2)


class EquationLayer(nn.Module):
    """A layer defined by an expression; see its forward function."""
    def __init__(self, shapes, inits, fn):
        super().__init__()
        self.fn = fn
        self.names = list(shapes)
        for name, shape in shapes.items():
            if name in inits:
                value = torch.full(shape, float(inits[name]))
            elif len(shape) >= 2:
                value = torch.randn(*shape) / math.sqrt(shape[-1])
            else:
                value = torch.zeros(*shape)
            setattr(self, name, nn.Parameter(value))

    def forward(self, x):
        return self.fn(x, **{n: getattr(self, n) for n in self.names})
'''


def compile_spec(params: Dict[str, Any], in_width: int
                 ) -> Tuple[List[Dict[str, Any]], ast.expr, Dict[str, List[int]]]:
    declared = parse_parameters(params.get("parameters", ""))
    tree = parse_forward(params.get("forward", ""),
                         [p["name"] for p in declared])
    shapes = {p["name"]: [in_width if d == "in" else d for d in p["shape"]]
              for p in declared}
    return declared, tree, shapes


def constructor(params: Dict[str, Any], in_width: int) -> str:
    declared, tree, shapes = compile_spec(params, in_width)
    inits = {p["name"]: p["init"] for p in declared if p["init"] is not None}
    args = ", ".join(["x"] + [p["name"] for p in declared])
    return (f"EquationLayer({shapes!r}, {inits!r}, "
            f"lambda {args}: {to_torch(tree)})")


def build(params: Dict[str, Any], in_width: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(PRELUDE, space)  # noqa: S102 - our own fixed text
    return eval(constructor(params, in_width), space)  # noqa: S307 - built from a whitelisted tree


def infer(params: Dict[str, Any], shape: List[int]) -> List[int]:
    """Run it on zeros: the only shape rule that cannot drift from the code."""
    import torch

    try:
        layer = build(params, int(shape[-1]))
        with torch.no_grad():
            out = layer(torch.zeros(1, *[int(d) for d in shape]))
    except EquationError:
        raise
    except Exception as exc:  # noqa: BLE001
        message = str(exc).split("\n")[0]
        raise EquationError(f"It does not run on an input of {list(shape)}: "
                            f"{message}")
    return list(out.shape[1:])


def explain(params: Dict[str, Any], in_shapes, out_shape) -> Dict[str, Any]:
    width = int(in_shapes[0][-1]) if in_shapes and in_shapes[0] else 1
    try:
        declared, tree, shapes = compile_spec(params, width)
    except EquationError as exc:
        return {"family": "equation", "title": params.get("label") or "Equation",
                "equation": "", "shape": "", "symbols": [], "arithmetic": [],
                "freedom": [], "missing": str(exc)}
    count = sum(math.prod(s) for s in shapes.values())
    return {
        "family": "equation",
        "title": params.get("label") or "Your equation",
        "equation": f"y = {to_maths(tree)}",
        "shape": f"x ∈ ℝ^{width} → y ∈ ℝ^{out_shape[-1] if out_shape else '?'}",
        "symbols": [["x", f"the {width} numbers arriving"]] + [
            [p["name"], f"learned, shape {shapes[p['name']]}"
             + (f", starting at {p['init']:g}" if p["init"] is not None else "")]
            for p in declared],
        "arithmetic": [
            ["parameters", " + ".join(
                f"{'×'.join(map(str, s))}" for s in shapes.values())
             + f" = {count}"],
            ["the code it becomes", to_torch(tree)],
        ],
        "freedom": [
            "The equation above and the code generated for this node are "
            "printed from the same expression, so they cannot disagree. Edit "
            "the expression and both change.",
        ],
    }
