"""rulesource.py — one network assembled from rules.

lawsource.py holds a layer per law. This holds the step after: a single class, RuleNet, that takes a list of
rules and builds the network in which every one of them is structure, so each holds for any value of the
weights and none has to be learned from data:

    energy        the output is the motion J∇H of one learned scalar H, so H is conserved        (Hamilton)
    permutation   H or the output is a sum over rows, so relabelling the rows cannot matter        (Deep Sets)
    rotation      the network only ever sees dot products, so rotating every vector cannot matter
    translation   positions are read relative to their centroid, so shifting them all cannot matter
    reflection    the scalar is averaged with its mirror image, so it is even
    monotone      weights on the named inputs are held to one sign and later ones are non-negative, so the
                  output can only move one way
    convex        input-convex network: non-negative hidden weights, convex non-decreasing activations
    additive      a sum of one-input networks, so no two inputs can interact

They compose. energy + rotation is a rotation-invariant energy, whose motion therefore conserves angular
momentum (Noether); energy + translation conserves total momentum; energy + permutation is SetHamiltonian.

The source lives here as text, as in lawsource.py, so the block on the canvas, the file the super agent
writes and the measurement all run the identical class.
"""
import math

RULES = ("energy", "permutation", "rotation", "translation", "reflection", "monotone", "convex", "additive")

RULENET = '''
class RuleNet(nn.Module):
    """A network built from rules; each rule is structure, so it holds for any weights.

    shape   (d,) for a row of d numbers, or (N, d) for N rows (particles, points)
    out     outputs per example; ignored with the rule "energy", whose output is the motion, shaped like the input
    rules   any of: energy, permutation, rotation, translation, reflection (rows), monotone, convex, additive,
            reflection (a row of numbers)
    k       coordinates per position vector in a row; a row is q (k numbers), then p (k numbers) if momenta
    signs   for "monotone": +1 / -1 / 0 per input (rises with it, falls with it, unconstrained)
    """
    def __init__(self, shape, out=1, rules=(), hidden=32, k=1, momenta=False, signs=None):
        super().__init__()
        self.shape = tuple(int(s) for s in shape)
        self.rules = tuple(sorted(set(rules)))
        self.out, self.hidden, self.k = int(out), int(hidden), int(k)
        self.is_set = len(self.shape) == 2
        R = set(self.rules)
        self.momenta = bool(momenta) or "energy" in R
        for r in R:
            if r not in ("energy", "permutation", "rotation", "translation", "reflection", "monotone", "convex", "additive"):
                raise ValueError(f"RuleNet has no rule called {r!r}.")
        if R & {"energy", "permutation", "rotation", "translation"} and not self.is_set:
            raise ValueError("energy, permutation, rotation and translation read rows: give a shape of (N, d).")
        if R & {"monotone", "convex", "additive"} and self.is_set:
            raise ValueError("monotone, convex and additive read one row of numbers: give a shape of (d,).")
        if "reflection" in R and "monotone" in R:
            raise ValueError("A function that is even cannot also be monotone.")
        if self.is_set:
            N, d = self.shape
            if d != self.k * (2 if self.momenta else 1):
                raise ValueError(f"Each row should hold {self.k * (2 if self.momenta else 1)} numbers "
                                 f"(k = {self.k}{', positions then momenta' if self.momenta else ''}); it holds {d}.")
            if "rotation" in R and self.k < 2:
                raise ValueError("Rotation needs vectors of at least 2 coordinates (k >= 2).")
            e = 1 if "energy" in R else self.hidden
            vecs = 2 if self.momenta else 1
            if "permutation" in R:
                own = (vecs * (vecs + 1) // 2 + (vecs * vecs if N > 1 else 0)) if "rotation" in R else d
                pair = (1 + (2 if self.momenta else 0)) if "rotation" in R else 2 * d
                self.one = self._mlp(own, e)
                self.pair = self._mlp(pair, e) if N > 1 else None
            else:
                M = N * vecs
                self.flat = self._mlp(M * (M + 1) // 2 if "rotation" in R else N * d, e)
            self.read = None if "energy" in R else nn.Sequential(nn.Linear(e, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.out))
        else:
            d = self.shape[0]
            s = [0] * d if signs is None else [int(v) for v in signs]
            if len(s) != d:
                raise ValueError(f"signs needs one entry per input ({d}); got {len(s)}.")
            if "monotone" in R and not any(s):
                s = [1] * d
            self.register_buffer("signs", torch.tensor(s if "monotone" in R else [0] * d, dtype=torch.float32))
            if "additive" in R:
                self.parts = nn.ModuleList([self._column(1, max(8, self.hidden // 2)) for _ in range(d)])
            else:
                self.parts = nn.ModuleList([self._column(d, self.hidden)])

    # ---- building blocks -------------------------------------------------------------------------
    def _mlp(self, n_in, n_out):
        h = self.hidden
        return nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_out))

    def _column(self, n_in, h):
        """Weights of one two-hidden-layer network on a row; how they are used depends on the rules."""
        m = nn.Module()
        m.a0, m.a1, m.a2 = nn.Linear(n_in, h), nn.Linear(n_in, h, bias=False), nn.Linear(n_in, self.out, bias=False)
        m.w1, m.w2 = nn.Linear(h, h), nn.Linear(h, self.out)
        return m

    def _signed(self, W, sign):
        """Columns with a sign are held to it (softplus is positive); the others are free."""
        s = sign.to(W.dtype)
        return torch.where(s.abs() > 0, F.softplus(W) * s, W)

    @staticmethod
    def _rising(a):
        """Half the units bend upward and half downward; both only ever rise, so with non-negative weights
        the network stays monotone and can still fit curves of either bend (Runje & Shankaranarayana 2023)."""
        h = a.shape[-1] // 2
        return torch.cat([F.softplus(a[..., :h]), -F.softplus(-a[..., h:])], dim=-1)

    def _row(self, m, x, sign):
        R = self.rules
        if "convex" in R:                                  # input-convex: z-weights >= 0, softplus activations
            a0, a1, a2 = (self._signed(l.weight, sign) for l in (m.a0, m.a1, m.a2))
            z = F.softplus(F.linear(x, a0, m.a0.bias))
            z = F.softplus(F.linear(z, F.softplus(m.w1.weight) / m.w1.in_features, m.w1.bias) + F.linear(x, a1))
            return F.linear(z, F.softplus(m.w2.weight) / m.w2.in_features, m.w2.bias) + F.linear(x, a2)
        if "monotone" in R:                                # one sign per constrained input, then >= 0 all the way up
            s = sign.to(x.dtype)
            a0 = torch.where(s.abs() > 0, m.a0.weight.abs() * s, m.a0.weight)
            z = self._rising(F.linear(x, a0, m.a0.bias))
            z = self._rising(F.linear(z, m.w1.weight.abs(), m.w1.bias))
            return F.linear(z, m.w2.weight.abs(), m.w2.bias)
        z = torch.tanh(m.a0(x))
        return m.w2(torch.tanh(m.w1(z)))

    def _vector(self, x):
        if "additive" in self.rules:
            return sum(self._row(m, x[..., i:i + 1], self.signs[i:i + 1]) for i, m in enumerate(self.parts))
        return self._row(self.parts[0], x, self.signs)

    @staticmethod
    def _dots(a, b):
        return (a * b).sum(-1, keepdim=True)

    def _rows(self, z):
        """z: [B, N, d] -> [B, e]. Everything the rules promise is decided by what this lets the network see."""
        R, k = self.rules, self.k
        B, N, _ = z.shape
        q = z[..., :k]
        p = z[..., k:2 * k] if self.momenta else None
        if "translation" in R:
            q = q - q.mean(dim=1, keepdim=True)
        vecs = [q] + ([p] if p is not None else [])
        if "permutation" in R:
            if "rotation" in R:
                own = [self._dots(a, b) for i, a in enumerate(vecs) for b in vecs[i:]]
                if N > 1:
                    own += [self._dots(a, t.sum(dim=1, keepdim=True)) for a in vecs for t in vecs]
                total = self.one(torch.cat(own, dim=-1)).sum(dim=1)
            else:
                total = self.one(torch.cat(vecs, dim=-1)).sum(dim=1)
            if self.pair is not None:
                qi, qj = q.unsqueeze(2), q.unsqueeze(1)
                if "rotation" in R:
                    dq = qi - qj
                    feats = [self._dots(dq, dq)]
                    if p is not None:
                        pi, pj = p.unsqueeze(2), p.unsqueeze(1)
                        feats += [self._dots(pi.expand(B, N, N, k), pj.expand(B, N, N, k)), self._dots(dq, pi - pj)]
                    pf = torch.cat(feats, dim=-1)
                else:
                    x = torch.cat(vecs, dim=-1)
                    pf = torch.cat([x.unsqueeze(2).expand(B, N, N, -1), x.unsqueeze(1).expand(B, N, N, -1)], dim=-1)
                mask = (~torch.eye(N, dtype=torch.bool, device=z.device)).unsqueeze(0).unsqueeze(-1)
                total = total + 0.5 * (self.pair(pf) * mask).sum(dim=(1, 2))
            return total
        if "rotation" in R:
            V = torch.cat(vecs, dim=1)                         # [B, M, k]: every vector of the example
            G = V @ V.transpose(1, 2)
            i, j = torch.triu_indices(G.shape[1], G.shape[1], device=z.device)
            return self.flat(G[:, i, j])
        return self.flat(torch.cat(vecs, dim=-1).flatten(1))

    def scalar(self, x):
        """What the network computes before the motion is read off it: H for "energy", else the output."""
        if not self.is_set:
            y = self._vector(x)
            return 0.5 * (y + self._vector(-x)) if "reflection" in self.rules else y
        y = self._rows(x)
        if "reflection" in self.rules and "rotation" not in self.rules:      # dot products are already even
            y = 0.5 * (y + self._rows(-x))
        return y if self.read is None else self.read(y)

    def forward(self, x):
        lead = x.shape[:-len(self.shape)]
        x = x.reshape(-1, *self.shape)
        if "energy" not in self.rules:
            return self.scalar(x).reshape(*lead, self.out)
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.requires_grad_(True)
            g = torch.autograd.grad(self.scalar(x).sum(), x, create_graph=True)[0]
            k = self.k
            y = torch.cat([g[..., k:2 * k], -g[..., :k]], dim=-1)             # dq/dt = dH/dp, dp/dt = -dH/dq
        return y.reshape(*lead, *self.shape)
'''

SEQNET = '''
class SeqNet(nn.Module):
    """A sequence model with one rule about memory or order built in. Input [L, dim].

    slots        WorkingMemory: only K slot vectors leave, however long the list is
    fastweights  HebbianFastWeights: a memory written by a local rule, read at the last step
    permutation  a bag: the items are summed, so their order cannot matter
    """
    def __init__(self, dim, out, rule, hidden=32, slots=4):
        super().__init__()
        self.rule = rule
        if rule == "slots":
            self.body = WorkingMemory(dim, slots=slots)
            self.head = nn.Linear(slots * dim, out)
        elif rule == "fastweights":
            self.body = HebbianFastWeights(dim, decay=0.95, rate=0.5)
            self.head = nn.Linear(dim, out)
        elif rule == "permutation":
            self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
        else:
            raise ValueError(f"SeqNet has no rule called {rule!r}.")

    def forward(self, x):
        if self.rule == "slots":
            return self.head(self.body(x).flatten(1))
        if self.rule == "fastweights":
            return self.head(self.body(x)[:, -1])
        return self.head(self.body(x).mean(dim=1))
'''


def build(name: str = "RuleNet"):
    """Return the class, executed from the same text the block and the written file use."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lawsource import SOURCES

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    if name == "SeqNet":
        exec(SOURCES["WorkingMemory"] + SOURCES["HebbianFastWeights"], space)  # noqa: S102 - our own source
    exec({"RuleNet": RULENET, "SeqNet": SEQNET}[name], space)  # noqa: S102 - our own source
    return space[name]


def source(name: str = "RuleNet") -> str:
    """The text a standalone file needs to define this class."""
    if name == "SeqNet":
        from lawsource import SOURCES
        return SOURCES["WorkingMemory"] + "\n" + SOURCES["HebbianFastWeights"] + "\n" + SEQNET
    return RULENET


def learnables(shape, out=1, rules=(), hidden=32, k=1, momenta=False, signs=None) -> int:
    """Counted by building it: the class is small, and a formula for every combination of rules would be a
    second place for the count to be wrong."""
    net = build("RuleNet")(shape, out=out, rules=rules, hidden=hidden, k=k, momenta=momenta, signs=signs)
    return sum(p.numel() for p in net.parameters())
