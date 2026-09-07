"""Implicit equilibrium layers, one per structural theorem.

Each *domain* is a published result that guarantees a system has exactly one
stable equilibrium under some structural condition, turned into a layer that
solves for that equilibrium rather than computing forward. What differs between
them is where the guarantee comes from:

    crn          the topology of a reaction graph — an integer, fixed at design
                 time, valid for every positive parameter
    contraction  a constraint on the weights — W reparameterised so that I − W
                 stays strongly monotone (monDEQ)
    convex       the sign of a coefficient vector — a strongly convex potential
                 has one stationary point

Every domain also ships *controls*: the same layer at the same parameter count
with exactly one condition of the theorem broken. Those are the interesting part.
Picking a control is not a mistake — it is how you find out whether the
guarantee was doing any work, and the layer says plainly which case you are in.

The mathematics lives in `dnn_bench/`, vendored unchanged from separate work.
Domains declared as JSON in `specs/` are registered too, and differentiate their
Jacobian from their residual so the two cannot disagree.
"""

from blocks_sdk import Block, Param, ShapeError, install

ADAPTER = '''
# The domains live in dnn_bench/, which must sit beside this file to run it.
from dnn_bench import core as _bench_core, domains as _bench_domains  # noqa: F401


class ImplicitEquilibrium(nn.Module):
    """Encode into the layer's state space, solve to equilibrium, decode out."""

    def __init__(self, in_dim, out_dim, domain, variant, seed=0,
                 dtype=torch.float64):
        super().__init__()
        spec = _bench_core.get(domain)
        arms = spec.variants(spec.defaults(), seed=seed)
        chosen = next((v for v in arms if v.key == variant), arms[0])
        self.inner = chosen.build()
        self.arm = chosen.key
        self.holds = bool((self.inner.certificate() or {}).get("holds"))
        width = self.inner.n
        self.enc = nn.Linear(in_dim, width, dtype=dtype)
        self.dec = nn.Linear(width, out_dim, dtype=dtype)

    def forward(self, z):
        return self.dec(self.inner(self.enc(z)))

    def extra_repr(self):
        return f"arm={self.arm}, theorem_applies={self.holds}"
'''


def _domains():
    """Every registered domain, coded or declared."""
    try:
        from dnn_bench import core, domains  # noqa: F401

        return {entry["key"]: entry for entry in core.all_domains()}
    except Exception:  # noqa: BLE001 - the palette still shows the layer
        return {}


def _arms(domain: str):
    from dnn_bench import core, domains  # noqa: F401

    spec = core.get(domain)
    return spec.variants(spec.defaults(), seed=0)


def _built(p):
    """Build the chosen arm, so the count reported is the count torch makes."""
    from dnn_bench import core, domains  # noqa: F401

    spec = core.get(str(p.get("domain", "crn")))
    arms = spec.variants(spec.defaults(), seed=int(p.get("seed", 0)))
    want = str(p.get("variant", ""))
    return next((v.build() for v in arms if v.key == want),
                arms[0].build())


_KEYS = sorted(_domains()) or ["crn"]
_ARM_KEYS = sorted({v.key for key in _KEYS for v in _arms(key)}) or ["wr0"]


def implicit_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 1:
        raise ShapeError(
            f"This layer takes a flat vector; {len(shape)} dimensions arrived. "
            f"Flatten or pool first.")
    domain = str(p.get("domain", "crn"))
    if domain not in _domains():
        raise ShapeError(f"No domain called {domain!r} is registered.")
    keys = [v.key for v in _arms(domain)]
    if str(p.get("variant")) not in keys:
        raise ShapeError(
            f"{domain} has no arm {p.get('variant')!r}. It offers: "
            f"{', '.join(keys)}.")
    return [int(p.get("units", 8))]


def implicit_learnables(p, ins, out):
    try:
        inner = _built(p)
    except Exception:  # noqa: BLE001
        return 0
    width = inner.n
    d_in = ins[0][0] if ins and ins[0] else 0
    d_out = int(p.get("units", 8))
    return (sum(q.numel() for q in inner.parameters())
            + d_in * width + width + width * d_out + d_out)


install(Block(
    name="ImplicitEquilibrium",
    category="Implicit",
    doc="Solves a system to equilibrium instead of computing forward. Each "
        "domain is a theorem guaranteeing that equilibrium exists, is unique "
        "and is stable; each also ships controls with one condition broken, at "
        "the same parameter count. The layer reports which case you picked.",
    params=[
        Param("units", "int", 8, min=1, help="Width leaving the layer"),
        Param("domain", "choice", _KEYS[0], options=_KEYS,
              help="Which theorem supplies the guarantee"),
        Param("variant", "choice", _ARM_KEYS[0], options=_ARM_KEYS,
              help="The arm the theorem covers, or a control that breaks one "
                   "of its conditions"),
        Param("seed", "int", 0, min=0, help="Which draw of this structure"),
    ],
    infer=implicit_infer,
    learnables=implicit_learnables,
    prelude=ADAPTER,
    torch_init=lambda p, ins: (
        f"ImplicitEquilibrium({ins[0][0]}, {int(p['units'])}, "
        f"{str(p['domain'])!r}, {str(p['variant'])!r}, "
        f"seed={int(p['seed'])}, dtype=torch.float64)"
    ),
))
