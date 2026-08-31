"""Differentiable numerical methods: a linear solve, an ODE integrator, a fixed point.

These are ordinary layers. Every one of them is differentiable end to end, so the
solver sits inside the network and gradients flow through the solution.
"""

from blocks_sdk import Block, Param, ShapeError, install, need_rank

SOLVE = '''
class RidgeSolve(nn.Module):
    """Solves (A + lambda I) x = b for x, with lambda learned.

    Gradients flow through the solve, so the network can shape the system it is
    asked to solve. The ridge term keeps a near-singular A from blowing up.
    """

    def __init__(self, init_lambda: float = 1e-3, learn_lambda: bool = True):
        super().__init__()
        value = torch.tensor(float(math.log(init_lambda)))
        if learn_lambda:
            self.log_lambda = nn.Parameter(value)
        else:
            self.register_buffer("log_lambda", value)

    def forward(self, A, b):
        eye = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
        M = A + self.log_lambda.exp() * eye
        return torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
'''

ODE = '''
class ODEBlock(nn.Module):
    """Integrates dh/dt = f(h) with fixed-step RK4.

    Depth becomes a continuous quantity: more steps cost time, not parameters.
    The field is shared across steps, which is what makes this different from
    stacking the same block several times.
    """

    def __init__(self, field: nn.Module, steps: int = 4, t_end: float = 1.0):
        super().__init__()
        self.field = field
        self.steps = steps
        self.dt = t_end / max(steps, 1)

    def forward(self, h):
        dt = self.dt
        for _ in range(self.steps):
            k1 = self.field(h)
            k2 = self.field(h + 0.5 * dt * k1)
            k3 = self.field(h + 0.5 * dt * k2)
            k4 = self.field(h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


def make_conv_field(channels: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.SiLU(),
        nn.Conv2d(hidden, channels, 3, padding=1), nn.GroupNorm(1, channels), nn.Tanh())


def make_mlp_field(dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim), nn.Tanh())
'''

FIXED = '''
class FixedPoint(nn.Module):
    """Iterates z <- (1 - a) z + a f(z, x) until it settles.

    One shared transform applied until the state stops moving, in the spirit of
    deep equilibrium models. Cheaper in parameters than the equivalent depth, and
    it stops early once the change falls under the tolerance.
    """

    def __init__(self, dim: int, hidden: int = 128, iterations: int = 16,
                 tol: float = 1e-4, alpha: float = 0.5):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        self.iterations = iterations
        self.tol = tol
        self.alpha = alpha
        self.last_iterations = 0

    def forward(self, x):
        z = torch.zeros_like(x)
        a = self.alpha
        for i in range(self.iterations):
            z_next = (1 - a) * z + a * torch.tanh(self.f(torch.cat([z, x], dim=-1)))
            delta = (z_next - z).abs().max()
            z = z_next
            self.last_iterations = i + 1
            if delta < self.tol:
                break
        return z
'''


def solve_infer(p, shapes):
    if len(shapes) != 2:
        raise ShapeError("RidgeSolve takes two inputs: the matrix A, then the vector b")
    A, b = shapes[0], shapes[1]
    need_rank(A, 2, "RidgeSolve", "[N, N] for the matrix")
    if A[0] != A[1]:
        raise ShapeError(f"A must be square, got {A[0]} by {A[1]}")
    need_rank(b, 1, "RidgeSolve", "[N] for the right-hand side")
    if b[0] != A[0]:
        raise ShapeError(f"b has length {b[0]} but A is {A[0]} by {A[1]}")
    return [A[0]]


def ode_infer(p, shapes):
    s = shapes[0]
    if p["field"] == "conv":
        need_rank(s, 3, "ODEBlock with a conv field", "[C, H, W]")
    elif len(s) not in (1, 2):
        raise ShapeError(
            f"ODEBlock with an mlp field needs [F] or [L, C], got {list(s)}. "
            f"Switch the field to conv for images."
        )
    return list(s)


def fixed_infer(p, shapes):
    s = shapes[0]
    if len(s) not in (1, 2):
        raise ShapeError(f"FixedPoint works on [F] or [L, C], got {list(s)}")
    return list(s)


def solve_learnables(p, ins, out):
    return 1 if p["learn_lambda"] else 0


def ode_learnables(p, ins, out):
    h = int(p["hidden"])
    if p["field"] == "conv":
        c = ins[0][0]
        return c * h * 9 + h + 2 * h + h * c * 9 + c + 2 * c
    d = ins[0][-1]
    return d * h + h + h * d + d


def fixed_learnables(p, ins, out):
    d, h = ins[0][-1], int(p["hidden"])
    return 2 * d * h + h + h * d + d


install(Block(
    name="RidgeSolve",
    category="Numerical",
    n_inputs=2,
    doc="Solves a linear system inside the network. First input is the matrix "
        "[N, N], second is the right-hand side [N]. Gradients flow through the "
        "solution, so upstream layers learn to pose a well-conditioned problem.",
    params=[
        Param("init_lambda", "float", 1e-3, help="Starting ridge term"),
        Param("learn_lambda", "bool", True),
    ],
    infer=solve_infer,
    learnables=solve_learnables,
    prelude=SOLVE,
    torch_init=lambda p, ins: (
        f"RidgeSolve(init_lambda={float(p['init_lambda'])}, "
        f"learn_lambda={bool(p['learn_lambda'])})"
    ),
))

install(Block(
    name="ODEBlock",
    category="Numerical",
    doc="Continuous-depth block integrated with RK4. Adding steps deepens the "
        "computation without adding parameters.",
    params=[
        Param("field", "select", "conv", options=["conv", "mlp"],
              help="conv for [C, H, W], mlp for [F] or [L, C]"),
        Param("hidden", "int", 64, min=1),
        Param("steps", "int", 4, min=1, help="RK4 steps; cost is four field calls each"),
        Param("t_end", "float", 1.0, help="Integration horizon"),
    ],
    infer=ode_infer,
    learnables=ode_learnables,
    prelude=ODE,
    torch_init=lambda p, ins: (
        "ODEBlock(make_{f}_field({dim}, {hidden}), steps={steps}, t_end={t})".format(
            f=p["field"],
            dim=ins[0][0] if p["field"] == "conv" else ins[0][-1],
            hidden=int(p["hidden"]), steps=int(p["steps"]), t=float(p["t_end"]))
    ),
))

install(Block(
    name="FixedPoint",
    category="Numerical",
    doc="Runs one transform to convergence instead of stacking copies of it. "
        "Stops early when the state stops changing.",
    params=[
        Param("hidden", "int", 128, min=1),
        Param("iterations", "int", 16, min=1, help="Upper bound on the loop"),
        Param("tol", "float", 1e-4, help="Stop when the largest change falls below this"),
        Param("alpha", "float", 0.5, min=0.05, max=1.0, help="Damping; lower is steadier"),
    ],
    infer=fixed_infer,
    learnables=fixed_learnables,
    prelude=FIXED,
    torch_init=lambda p, ins: (
        f"FixedPoint({ins[0][-1]}, hidden={int(p['hidden'])}, "
        f"iterations={int(p['iterations'])}, tol={float(p['tol'])}, "
        f"alpha={float(p['alpha'])})"
    ),
))
