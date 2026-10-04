"""Layers whose structure guarantees a physical law.

HamiltonianField learns an energy and returns the motion it implies, so the
energy is conserved for any weights; with dissipation on, it can only fall.
The source is shared with conservation.py, where the guarantee is measured,
so the layer on the canvas is the layer that was tested.
"""
from blocks_sdk import Block, Param, ShapeError, install

from conservation import (COUPLED_SOURCE, SOURCE, coupled_learnables,
                          learnables as _count)


def hamiltonian_infer(p, shapes):
    s = list(shapes[0])
    if not s or s[-1] % 2:
        raise ShapeError(
            f"HamiltonianField splits its input into positions and momenta, "
            f"so the last dimension must be even; got {s}.")
    return s


def hamiltonian_learnables(p, ins, out):
    return _count(ins[0][-1], int(p["hidden"]), bool(p["dissipative"]))


install(Block(
    name="HamiltonianField",
    category="Numerical",
    doc="Learns an energy H(q, p) and returns the motion it implies: "
        "dq/dt = ∂H/∂p, dp/dt = −∂H/∂q. The energy is conserved for any "
        "weights. With dissipation, a learned friction γ ≥ 0 can only remove "
        "energy. Greydanus et al. 2019.",
    params=[
        Param("hidden", "int", 64, min=1),
        Param("dissipative", "bool", False,
              help="Port-Hamiltonian: energy may fall, never rise"),
        Param("mode", "select", "field", options=["field", "flow"],
              help="field returns dz/dt; flow integrates it with RK4"),
        Param("steps", "int", 4, min=1, help="RK4 steps, in flow mode"),
        Param("t_end", "float", 1.0, help="Integration horizon, in flow mode"),
    ],
    infer=hamiltonian_infer,
    learnables=hamiltonian_learnables,
    prelude=SOURCE,
    torch_init=lambda p, ins: (
        "HamiltonianField({dim}, {hidden}, dissipative={d}, mode={m!r}, "
        "steps={s}, t_end={t})".format(
            dim=ins[0][-1], hidden=int(p["hidden"]), d=bool(p["dissipative"]),
            m=p["mode"], s=int(p["steps"]), t=float(p["t_end"]))
    ),
))


def _parts(p):
    try:
        parts = [int(x) for x in str(p.get("parts", "2,2")).replace(" ", "").split(",") if x]
    except ValueError:
        raise ShapeError(f"parts must be whole numbers separated by commas, "
                         f"got “{p.get('parts')}”.")
    if not parts or any(x <= 0 or x % 2 for x in parts):
        raise ShapeError(f"Every subsystem needs a positive, even width "
                         f"(positions then momenta); got {parts}.")
    return parts


def coupled_infer(p, shapes):
    s = list(shapes[0])
    parts = _parts(p)
    if not s or s[-1] != sum(parts):
        raise ShapeError(f"The subsystems {parts} add up to {sum(parts)}, but "
                         f"the input's last dimension is {s[-1] if s else '?'}.")
    return s


install(Block(
    name="CoupledHamiltonian",
    category="Numerical",
    doc="Several subsystems joined so that together they can only lose "
        "energy: dz/dt = (J − R)∇H, with each subsystem's own energy, an "
        "optional interaction term, a learned skew-symmetric coupling and a "
        "non-negative friction per subsystem. dH/dt ≤ 0 for any weights.",
    params=[
        Param("parts", "text", "2,2",
              help="Width of each subsystem, comma-separated; each even"),
        Param("hidden", "int", 32, min=1),
        Param("interaction", "bool", True,
              help="An energy term spanning all subsystems, as a spring between them would need"),
        Param("mode", "select", "field", options=["field", "flow"]),
        Param("steps", "int", 4, min=1),
        Param("t_end", "float", 1.0),
    ],
    infer=coupled_infer,
    learnables=lambda p, ins, out: coupled_learnables(
        _parts(p), int(p["hidden"]), bool(p["interaction"])),
    prelude=COUPLED_SOURCE,
    torch_init=lambda p, ins: (
        "CoupledHamiltonian({parts}, {hidden}, interaction={i}, mode={m!r}, "
        "steps={s}, t_end={t})".format(
            parts=_parts(p), hidden=int(p["hidden"]), i=bool(p["interaction"]),
            m=p["mode"], s=int(p["steps"]), t=float(p["t_end"]))),
))
