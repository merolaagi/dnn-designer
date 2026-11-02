"""lawsource.py — layers whose *structure* enforces a law.

Each class below is a law written as architecture: the property holds for every
value of the weights, so it is never learned from data and never violated.
That is what lets these layers train on less — the data no longer has to teach
what the structure already knows.

The sources live here as text, the same way conservation.py holds
HamiltonianField, so the block on the canvas, the generated file, and the
measurement in laws.py all run the identical class.

    LagrangianField     physics     Euler–Lagrange; energy conserved for any L
    ConvexPotential     math        output is convex in the input (ICNN)
    LipschitzLinear     math        operator norm ≤ L, so the layer cannot blow up
    InvariantPool       math        exactly permutation-invariant (Deep Sets)
    DaleLinear          biology     Dale's law: each input neuron is E or I
    HebbianFastWeights  biology     fast synapses change by a local Hebbian rule
    WorkingMemory       psychology  a hard capacity of K slots (Cowan's 4 ± 1)

Use ``build(name)`` to get the class in Python, as laws.py does.
"""

from __future__ import annotations

import math

NAMES = ["LagrangianField", "ConvexPotential", "LipschitzLinear", "InvariantPool",
         "DaleLinear", "HebbianFastWeights", "WorkingMemory", "SetHamiltonian"]

# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

LAGRANGIAN = '''
class MechanicalLagrangian(nn.Module):
    """L(q, q̇) = ½ q̇ᵀ M(q) q̇ − V(q), with M(q) = C(q)C(q)ᵀ + εI and C lower-triangular with a positive
    diagonal: the mass matrix is positive definite for any weights (Lutter et al., Deep Lagrangian
    Networks, 2019), so Euler–Lagrange can always be solved for q̈."""
    def __init__(self, n, hidden, eps, act=nn.Softplus):
        super().__init__()
        self.n, self.eps = n, float(eps)
        self.mass = nn.Sequential(nn.Linear(n, hidden), act(), nn.Linear(hidden, hidden), act(),
                                  nn.Linear(hidden, n * (n + 1) // 2))
        self.potential = nn.Sequential(nn.Linear(n, hidden), act(), nn.Linear(hidden, hidden), act(),
                                       nn.Linear(hidden, 1))
        self.register_buffer("rows", torch.tril_indices(n, n)[0])
        self.register_buffer("cols", torch.tril_indices(n, n)[1])

    def forward(self, z):
        q, qd = z[:, :self.n], z[:, self.n:]
        flat = self.mass(q)
        C = q.new_zeros(q.shape[0], self.n, self.n)
        C[:, self.rows, self.cols] = flat
        diag = torch.arange(self.n, device=q.device)
        C[:, diag, diag] = F.softplus(C[:, diag, diag]) + 1e-3
        M = C @ C.transpose(1, 2) + self.eps * torch.eye(self.n, dtype=q.dtype, device=q.device)
        kinetic = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(-1)).squeeze(-1)
        return kinetic - self.potential(q)


class LagrangianField(nn.Module):
    """Learns a Lagrangian L(q, q̇) and returns the motion Euler–Lagrange implies.

    Cranmer et al., *Lagrangian Neural Networks* (2020). Input z = (q, q̇), split
    in half. The equation d/dt ∂L/∂q̇ = ∂L/∂q is solved for q̈:

        q̈ = (∇²_{q̇q̇} L)⁻¹ [ ∇_q L − (∇²_{q̇q} L) q̇ ]

    and the layer returns (q̇, q̈), i.e. dz/dt. Because the motion is *derived*
    from a single scalar L, it conserves E = q̇·∂L/∂q̇ − L for every value of the
    weights. It works in the coordinates you measure (positions and velocities).

    form="mechanical" (the default) builds L = ½ q̇ᵀM(q)q̇ − V(q) with M positive
    definite, as every mechanical system has; the mass matrix can then always be
    inverted. form="general" lets a network be any L: more general, but nothing
    keeps ∇²_{q̇q̇} L invertible, and trained on a pendulum its rollouts blew up.
    """
    def __init__(self, dim, hidden=64, eps=1e-3, form="mechanical", activation="tanh"):
        super().__init__()
        if dim % 2:
            raise ValueError(f"LagrangianField needs an even width, got {dim}")
        if activation not in ("softplus", "tanh"):
            raise ValueError(f"activation is softplus or tanh, not {activation!r}")
        self.n = dim // 2
        self.eps = float(eps)
        self.form = form
        act = nn.Tanh if activation == "tanh" else nn.Softplus   # both twice differentiable, as the solve needs
        if form == "mechanical":
            self.lagrangian = MechanicalLagrangian(self.n, hidden, eps, act)
        elif form == "general":
            self.lagrangian = nn.Sequential(
                nn.Linear(dim, hidden), act(),
                nn.Linear(hidden, hidden), act(),
                nn.Linear(hidden, 1))
        else:
            raise ValueError(f"form is mechanical or general, not {form!r}")

    def forward(self, z):
        lead = z.shape[:-1]
        z = z.reshape(-1, 2 * self.n)
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.requires_grad_(True)
            n = self.n
            L = self.lagrangian(z).sum()
            g = torch.autograd.grad(L, z, create_graph=True)[0]        # [B, 2n]
            rows = [torch.autograd.grad(g[:, n + i].sum(), z, create_graph=True)[0]
                    for i in range(n)]                                   # each [B, 2n]
            H = torch.stack(rows, dim=1)                                 # [B, n, 2n]
            H_qqd, H_qdqd = H[:, :, :n], H[:, :, n:]
            qd = z[:, n:]
            rhs = g[:, :n] - torch.bmm(H_qqd, qd.unsqueeze(-1)).squeeze(-1)
            reg = 0.0 if self.form == "mechanical" else self.eps        # the mechanical mass already holds εI
            eye = torch.eye(n, dtype=z.dtype, device=z.device)
            qdd = torch.linalg.solve(H_qdqd + reg * eye, rhs.unsqueeze(-1)).squeeze(-1)
            out = torch.cat([qd, qdd], dim=-1)
        return out.reshape(*lead, 2 * self.n)
'''

# ---------------------------------------------------------------------------
# mathematics
# ---------------------------------------------------------------------------

CONVEX = '''
class ConvexPotential(nn.Module):
    """An input-convex neural network: f(x) is convex in x for any weights.

    Amos, Xu & Kolter, *Input Convex Neural Networks* (ICML 2017). Each layer is

        z_{i+1} = σ( W_z⁺ z_i + W_y y + b )

    with W_z⁺ ≥ 0 (softplus of a free parameter) and σ convex and non-decreasing
    (softplus). A non-negative combination of convex functions, passed through a
    convex non-decreasing function, is convex — by induction, so is the output.
    Gradient descent on f therefore has one minimum. Used for energies, optimal
    transport potentials, and value functions that must have no spurious wells.
    """
    def __init__(self, dim, hidden=64, layers=2, out=1):
        super().__init__()
        self.first = nn.Linear(dim, hidden)
        # softplus(W) starts near 1/hidden, so each layer roughly keeps the scale it is given; starting
        # at softplus(0) = 0.69 summed over every unit made the output hundreds of times the target's
        start = math.log(math.expm1(1.0 / hidden))
        self.wz = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden, hidden) * 0.1 + start) for _ in range(layers - 1)]
            + [nn.Parameter(torch.randn(out, hidden) * 0.1 + start)])
        self.wy = nn.ModuleList(
            [nn.Linear(dim, hidden) for _ in range(layers - 1)] + [nn.Linear(dim, out)])
        self.act = nn.Softplus()

    def forward(self, y):
        z = self.act(self.first(y))
        for i, (wz, wy) in enumerate(zip(self.wz, self.wy)):
            z = F.linear(z, F.softplus(wz)) + wy(y)
            if i < len(self.wz) - 1:
                z = self.act(z)
        return z
'''

LIPSCHITZ = '''
class LipschitzLinear(nn.Module):
    """A linear layer whose operator norm never exceeds a bound L.

    Spectral normalisation (Miyato et al., ICLR 2018) used as a hard constraint:
    the weight actually applied is W · min(1, L/σ(W)), where σ(W) is the largest
    singular value computed exactly (not estimated by power iteration, which can
    only under-estimate σ and so lets the bound be broken). Hence ‖W_eff‖₂ ≤ L for
    any weights, and the layer is L-Lipschitz: ‖f(x) − f(x')‖ ≤ L‖x − x'‖. Stack
    these with 1-Lipschitz activations (ReLU, tanh, GroupSort; not GELU, whose
    slope exceeds 1) and the whole network is Lipschitz with the product of the
    bounds: a certificate of robustness, not a hope.
    """
    def __init__(self, dim, units, bound=1.0):
        super().__init__()
        self.linear = nn.Linear(dim, units)
        self.bound = float(bound)

    def effective_weight(self):
        W = self.linear.weight
        sigma = torch.linalg.matrix_norm(W, ord=2)
        return W * torch.clamp(self.bound / (sigma + 1e-12), max=1.0)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.linear.bias)
'''

INVARIANT = '''
class InvariantPool(nn.Module):
    """A function of a set, exactly invariant to the order of its elements.

    Zaheer et al., *Deep Sets* (NeurIPS 2017): every permutation-invariant
    function of a set decomposes as ρ(Σᵢ φ(xᵢ)). This layer is that decomposition,
    with φ and ρ small MLPs and the sum replaceable by mean or max. Reordering the
    rows cannot change the output, for any weights — so the network never needs
    to see the same set in a second order, which a flat MLP must.
    """
    def __init__(self, dim, hidden=64, out=16, pool="sum"):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out))
        self.pool = pool

    def forward(self, x):                     # [B, N, dim]
        h = self.phi(x)
        if self.pool == "sum":
            s = h.sum(dim=-2)
        elif self.pool == "mean":
            s = h.mean(dim=-2)
        else:
            s = h.max(dim=-2).values
        return self.rho(s)
'''

# ---------------------------------------------------------------------------
# biology
# ---------------------------------------------------------------------------

DALE = '''
class DaleLinear(nn.Module):
    """A linear layer obeying Dale's law: every input neuron is excitatory or
    inhibitory, never both.

    Dale (1935); Song, Yang & Wang, *Training Excitatory-Inhibitory Recurrent
    Networks* (PLoS Comp Bio 2016). The weight actually used is

        W_eff = softplus(W) · diag(s),   s_j = +1 (E) or −1 (I)

    so every column has one sign, fixed by which neuron it leaves — for any
    weights. The first ⌈frac·dim⌉ inputs are excitatory (cortex is ~80% E).
    This is the constraint under which real circuits learn, and it changes which
    solutions are reachable: a network cannot flip a synapse's sign to fit noise.
    """
    def __init__(self, dim, units, frac=0.8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(units, dim) * (1.0 / dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(units))
        n_e = int(math.ceil(frac * dim))
        sign = torch.ones(dim)
        sign[n_e:] = -1.0
        self.register_buffer("sign", sign)

    def effective_weight(self):
        return F.softplus(self.weight) * self.sign

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)
'''

HEBBIAN = '''
class HebbianFastWeights(nn.Module):
    """A memory that writes itself by a local Hebbian rule, with no gradient.

    Ba et al., *Using Fast Weights to Attend to the Recent Past* (NeurIPS 2016).
    Slow weights W are trained by backprop. A fast matrix A starts at zero each
    sequence and after every step is updated by

        A ← λ A + η ĥ ĥᵀ          (decay, then “neurons that fire together wire together”)

    where ĥ is the step's own output, layer-normalised before it is written. A's update uses only the activity at its
    two ends — the rule is local, as a synapse's must be — and A is never touched
    by the optimizer. Reading A h recalls recent patterns; the decay λ is the
    forgetting. Input [L, dim], output [L, dim]; causal, so step t sees A from
    steps < t only.
    """
    def __init__(self, dim, decay=0.9, rate=0.5):
        super().__init__()
        self.slow = nn.Linear(dim, dim)
        # what is written is normalised, so one strong step cannot swamp the memory. No learned scale and
        # shift: the write is outside the gradient (that is the locality), so they would never be trained.
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.decay = float(decay)
        self.rate = float(rate)

    def forward(self, x, return_fast=False):  # [B, L, dim]
        B, L, D = x.shape
        A = x.new_zeros(B, D, D)
        outs, trace = [], []
        for t in range(L):
            xt = x[:, t]
            trace.append(A)
            h = torch.tanh(self.slow(xt) + torch.bmm(A, xt.unsqueeze(-1)).squeeze(-1))
            outs.append(h)
            hd = self.norm(h.detach())        # the rule sees activity only: no gradient reaches A
            A = self.decay * A + self.rate * torch.bmm(hd.unsqueeze(-1), hd.unsqueeze(-2))
        out = torch.stack(outs, dim=1)
        return (out, trace) if return_fast else out
'''

# ---------------------------------------------------------------------------
# psychology
# ---------------------------------------------------------------------------

WORKING_MEMORY = '''
class WorkingMemory(nn.Module):
    """A bottleneck with a hard capacity of K slots, whatever the input length.

    Miller (1956) found 7 ± 2; Cowan (2001) revised it to 4 ± 1 chunks. Here K
    learned queries attend over the L input items (Locatello et al. 2020's slot
    attention, one pass) and only the K slot vectors go on. A sequence of 5 or
    5,000 items leaves as exactly K vectors — the capacity is architectural, not
    a regulariser. Downstream layers must therefore work with a compressed,
    chunked summary, which is the pressure that produces chunking in people.
    Input [L, dim], output [K, dim].
    """
    def __init__(self, dim, slots=4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, dim) * 0.1)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                     # [B, L, dim]
        B = x.shape[0]
        q = self.q(self.slots).unsqueeze(0).expand(B, -1, -1)   # [B, K, dim]
        k, v = self.k(x), self.v(x)
        att = torch.softmax(q @ k.transpose(1, 2) / k.shape[-1] ** 0.5, dim=-1)
        return self.norm(self.out(att @ v) + q)
'''

SET_HAMILTONIAN = '''
class SetHamiltonian(nn.Module):
    """Two laws in one layer: energy conservation and permutation invariance.

    The input is a set of N particles, each a row (q, p) of width dim. One
    scalar energy is built the Deep Sets way, so relabelling the particles
    cannot change it:

        H(Z) = Σᵢ φ(zᵢ) + ½ Σ_{i≠j} ψ(zᵢ, zⱼ)

    and the motion is Hamilton's: dqᵢ/dt = ∂H/∂pᵢ, dpᵢ/dt = −∂H/∂qᵢ. Both
    properties hold for any weights: dH/dt = 0 along the motion, and relabelling
    the input particles relabels the output motion the same way (equivariance).
    A plain network on the flattened state has to learn both from data — every
    ordering of the same configuration is a new example to it, and conservation
    is something it can only approximate. Input [N, dim], output [N, dim].
    """
    def __init__(self, dim, hidden=64, pairwise=True):
        super().__init__()
        if dim % 2:
            raise ValueError(f"SetHamiltonian needs an even particle width, got {dim}")
        self.n = dim // 2
        self.one = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
        self.pair = (nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))
                     if pairwise else None)

    def energy(self, z):                       # z: [B, N, dim] -> [B]
        e = self.one(z).sum(dim=(1, 2))
        if self.pair is not None:
            B, N, D = z.shape
            zi = z.unsqueeze(2).expand(B, N, N, D)
            zj = z.unsqueeze(1).expand(B, N, N, D)
            pair = self.pair(torch.cat([zi, zj], dim=-1)).squeeze(-1)   # [B, N, N]
            mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            e = e + 0.5 * (pair * mask).sum(dim=(1, 2))
        return e

    def forward(self, z):
        lead = z.shape[:-2]
        z = z.reshape(-1, z.shape[-2], z.shape[-1])
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.requires_grad_(True)
            H = self.energy(z).sum()
            g = torch.autograd.grad(H, z, create_graph=True)[0]
            out = torch.cat([g[..., self.n:], -g[..., :self.n]], dim=-1)
        return out.reshape(*lead, z.shape[-2], z.shape[-1])
'''

SOURCES = {
    "LagrangianField": LAGRANGIAN,
    "ConvexPotential": CONVEX,
    "LipschitzLinear": LIPSCHITZ,
    "InvariantPool": INVARIANT,
    "DaleLinear": DALE,
    "HebbianFastWeights": HEBBIAN,
    "WorkingMemory": WORKING_MEMORY,
    "SetHamiltonian": SET_HAMILTONIAN,
}


def build(name: str):
    """Return the class, executed from the same text the block emits."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(SOURCES[name], space)  # noqa: S102 - our own source
    return space[name]


# ---------------------------------------------------------------------------
# parameter counts, derived so the canvas total stays honest
# ---------------------------------------------------------------------------

def _mlp(*widths):
    return sum(a * b + b for a, b in zip(widths, widths[1:]))


def learnables(name: str, **k) -> int:
    if name == "LagrangianField":
        if k.get("form", "mechanical") == "general":
            return _mlp(k["dim"], k["hidden"], k["hidden"], 1)
        n, h = k["dim"] // 2, k["hidden"]
        return _mlp(n, h, h, n * (n + 1) // 2) + _mlp(n, h, h, 1)
    if name == "ConvexPotential":
        d, h, L, o = k["dim"], k["hidden"], k["layers"], k["out"]
        return (d * h + h) + (L - 1) * (h * h + d * h + h) + (h * o + d * o + o)
    if name == "LipschitzLinear":
        return k["dim"] * k["units"] + k["units"]
    if name == "InvariantPool":
        d, h, o = k["dim"], k["hidden"], k["out"]
        return _mlp(d, h, h) + _mlp(h, h, o)
    if name == "DaleLinear":
        return k["dim"] * k["units"] + k["units"]
    if name == "HebbianFastWeights":
        return k["dim"] * k["dim"] + k["dim"]
    if name == "WorkingMemory":
        d = k["dim"]
        return k["slots"] * d + 4 * (d * d + d) + 2 * d
    if name == "SetHamiltonian":
        d, h = k["dim"], k["hidden"]
        return _mlp(d, h, h, 1) + (_mlp(2 * d, h, h, 1) if k.get("pairwise", True) else 0)
    raise KeyError(name)
