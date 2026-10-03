"""Physics-informed training, checked against a solution computed another way.

A PINN learns u(x, t) by being penalised wherever it violates an equation
(Raissi, Perdikaris & Karniadakis, 2019). Three terms: the PDE residual at
points sampled inside the domain, the initial condition, and the boundary
condition. All three are computed with autograd from the network's own output,
so the network is differentiated with respect to its *inputs* — which is the
one thing the ordinary training loop here cannot do, because its loss only
ever sees outputs and targets.

The rule this module holds, because the literature has shown it necessary:

    A falling loss is not evidence that the equation was solved.

PINNs stall at wrong answers in documented ways — loss terms fighting each
other (Wang, Teng & Perdikaris, 2021), convection-dominated problems where the
loss plateaus at a plausible but wrong solution (Krishnapriyan et al., 2021),
and spectral bias against sharp fronts. So every run is compared with a
finite-difference solution of the same problem, each loss term is reported
separately, and the verdict comes from the comparison, never from the loss.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

#: The derivatives a residual may use. Second order in space, first and second
#: in time; that covers heat, advection, Burgers, Allen-Cahn, KdV-less
#: reaction-diffusion and the wave equation.
DERIVATIVES = ("u_t", "u_x", "u_xx", "u_tt", "u_xt")
NAMES = ("u", "x", "t") + DERIVATIVES

PRESETS: List[Dict[str, Any]] = [
    {
        "id": "burgers",
        "name": "Viscous Burgers",
        "pde": "u_t + u*u_x - 0.01/pi*u_xx",
        "x": [-1.0, 1.0], "t": [0.0, 1.0],
        "ic": "-sin(pi*x)",
        "bc": {"kind": "dirichlet", "left": "0", "right": "0"},
        "depth": 8, "width": 20, "iterations": 2000, "lbfgs": 1000,
        "adaptive": True,
        "note": "The standard benchmark from Raissi et al. A shock forms at "
                "x = 0 near t ≈ 0.4. Measured here: eight layers of 20 with "
                "adaptive refinement reached 2.4%, 5.0% and 1.3% on three "
                "seeds. The same network without refinement reached 31%, and "
                "four layers of 32 with refinement 20% — capacity and points "
                "on the shock are both needed, and neither is enough alone.",
    },
    {
        "id": "heat",
        "name": "Heat equation",
        "pde": "u_t - 0.1*u_xx",
        "x": [-1.0, 1.0], "t": [0.0, 1.0],
        "ic": "sin(pi*x)",
        "bc": {"kind": "dirichlet", "left": "0", "right": "0"},
        "note": "Smooth, diffusive, forgiving. If this does not match the "
                "reference, the problem is the network or the settings, not "
                "the method.",
    },
    {
        "id": "convection",
        "name": "Fast convection (a documented failure)",
        "pde": "u_t + 30*u_x",
        "x": [0.0, 6.283185307179586], "t": [0.0, 1.0],
        "ic": "sin(x)",
        "bc": {"kind": "periodic"},
        "note": "Krishnapriyan et al. (2021) showed vanilla PINNs fail here: "
                "the loss falls and settles on a wrong, nearly flat solution. "
                "It is included so the failure can be seen happening — "
                "a tool that only showed the loss would call it a success.",
    },
]


class PhysicsError(Exception):
    pass


# --------------------------------------------------------------------------
# the equation
# --------------------------------------------------------------------------

def parse(pde: str, ic: str, bc: Dict[str, Any]):
    """Read the equation and conditions, refusing anything ambiguous."""
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    local = {name: sympy.Symbol(name) for name in NAMES}
    local["pi"] = sympy.pi

    def read(text: str, allowed: Tuple[str, ...], what: str):
        if not (text or "").strip():
            raise PhysicsError(f"The {what} is empty.")
        try:
            expr = parse_expr(text, local_dict=local,
                              transformations=standard_transformations)
        except Exception as exc:  # noqa: BLE001
            raise PhysicsError(f"Could not read the {what}: {exc}")
        unknown = sorted(str(s) for s in expr.free_symbols
                         if str(s) not in allowed)
        if unknown:
            raise PhysicsError(
                f"The {what} uses {', '.join(unknown)}, which is not one of "
                f"{', '.join(allowed)}.")
        return expr

    residual = read(pde, NAMES, "equation")
    used = [d for d in DERIVATIVES if residual.has(local[d])]
    if not used:
        raise PhysicsError("The equation has no derivative in it, so it is "
                           "not a differential equation.")
    initial = read(ic, ("x", "pi"), "initial condition")

    kind = (bc or {}).get("kind", "dirichlet")
    if kind not in ("dirichlet", "periodic"):
        raise PhysicsError(f"Unknown boundary condition {kind}.")
    sides = {}
    if kind == "dirichlet":
        for side in ("left", "right"):
            sides[side] = read(str(bc.get(side, "0")), ("t", "pi"),
                               f"{side} boundary value")
    return {"residual": residual, "used": used, "ic": initial,
            "bc_kind": kind, "bc": sides, "symbols": local}


def _torch_fn(expr, args):
    import sympy
    import torch

    table = {"sin": torch.sin, "cos": torch.cos, "exp": torch.exp,
             "log": torch.log, "sqrt": torch.sqrt, "tanh": torch.tanh,
             "Abs": torch.abs, "sinh": torch.sinh, "cosh": torch.cosh}
    return sympy.lambdify(args, expr, modules=[table, "math"])


def _numpy_fn(expr, args):
    import sympy

    return sympy.lambdify(args, expr, modules="numpy")


# --------------------------------------------------------------------------
# the reference: the same problem, solved without a neural network
# --------------------------------------------------------------------------

def reference(parsed, x_range, t_range, nx: int = 512, nt: int = 101,
              max_steps: int = 400_000) -> Dict[str, Any]:
    """Method of lines: central differences in space, RK4 in time.

    Only first order in time and solvable for u_t; a wave equation (u_tt) gets
    no reference, and the result says so rather than being compared with
    nothing. The time step is chosen from the equation's own coefficients —
    diffusion and advection read off by differentiating the right-hand side —
    so a stiff problem is refused rather than silently blown up.
    """
    import sympy

    sym = parsed["symbols"]
    if "u_tt" in parsed["used"] or "u_xt" in parsed["used"]:
        return {"ok": False, "why": "The equation is second order in time, "
                "and the reference solver here is first order. There is "
                "nothing to compare with, so the result cannot be verified."}
    solved = sympy.solve(parsed["residual"], sym["u_t"])
    if len(solved) != 1:
        return {"ok": False, "why": "The equation cannot be written as "
                "u_t = F(u, u_x, u_xx), so the reference solver cannot "
                "march it forward."}
    rhs = solved[0]
    args = (sym["u"], sym["u_x"], sym["u_xx"], sym["x"], sym["t"])
    F = _numpy_fn(rhs, args)
    dF_dxx = _numpy_fn(sympy.diff(rhs, sym["u_xx"]), args)
    dF_dx = _numpy_fn(sympy.diff(rhs, sym["u_x"]), args)

    x0, x1 = map(float, x_range)
    t0, t1 = map(float, t_range)
    periodic = parsed["bc_kind"] == "periodic"
    if periodic:
        x = np.linspace(x0, x1, nx, endpoint=False)
    else:
        x = np.linspace(x0, x1, nx)
    dx = x[1] - x[0]
    u = np.asarray(_numpy_fn(parsed["ic"], (sym["x"],))(x), dtype=float)
    u = np.broadcast_to(u, x.shape).astype(float).copy()

    left = right = None
    if not periodic:
        left = _numpy_fn(parsed["bc"]["left"], (sym["t"],))
        right = _numpy_fn(parsed["bc"]["right"], (sym["t"],))

    def spatial(v):
        if periodic:
            vx = (np.roll(v, -1) - np.roll(v, 1)) / (2 * dx)
            vxx = (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx ** 2
        else:
            vx = np.zeros_like(v)
            vxx = np.zeros_like(v)
            vx[1:-1] = (v[2:] - v[:-2]) / (2 * dx)
            vxx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx ** 2
        return vx, vxx

    def rate(v, tt):
        vx, vxx = spatial(v)
        out = np.broadcast_to(F(v, vx, vxx, x, tt), v.shape).astype(float)
        if not periodic:
            out = out.copy()
            out[0] = out[-1] = 0.0
        return out

    vx, vxx = spatial(u)
    span = max(1.0, float(np.max(np.abs(u))) * 1.5)
    probe = np.linspace(-span, span, 9)
    diffusion = max(float(np.max(np.abs(np.broadcast_to(
        dF_dxx(p, vx, vxx, x, t0), x.shape)))) for p in probe)
    advection = max(float(np.max(np.abs(np.broadcast_to(
        dF_dx(p, vx, vxx, x, t0), x.shape)))) for p in probe)
    limits = [t1 - t0]
    if diffusion > 0:
        limits.append(0.4 * dx ** 2 / diffusion)
    if advection > 0:
        limits.append(0.8 * dx / advection)
    dt_max = min(limits)
    steps = int(math.ceil((t1 - t0) / dt_max))
    steps = int(math.ceil(steps / (nt - 1))) * (nt - 1)
    if steps > max_steps:
        return {"ok": False, "why": f"The problem is too stiff for the "
                f"reference solver: it would need {steps:,} steps. The "
                f"result cannot be verified at this resolution."}
    dt = (t1 - t0) / steps
    every = steps // (nt - 1)

    frames = [u.copy()]
    tt = t0
    for step in range(1, steps + 1):
        k1 = rate(u, tt)
        k2 = rate(u + 0.5 * dt * k1, tt + 0.5 * dt)
        k3 = rate(u + 0.5 * dt * k2, tt + 0.5 * dt)
        k4 = rate(u + dt * k3, tt + dt)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        tt = t0 + step * dt
        if not periodic:
            u[0] = float(left(tt))
            u[-1] = float(right(tt))
        if not np.all(np.isfinite(u)):
            return {"ok": False, "why": "The reference solver diverged, so "
                    "the result cannot be verified."}
        if step % every == 0:
            frames.append(u.copy())

    return {"ok": True, "x": x, "t": np.linspace(t0, t1, nt),
            "u": np.stack(frames), "steps": steps, "nx": nx,
            "how": f"central differences on {nx} points, RK4 with {steps:,} "
                   f"steps"}


# --------------------------------------------------------------------------
# the network's derivatives, by autograd with respect to its inputs
# --------------------------------------------------------------------------

def derivatives(model, x, t, wanted):
    import torch

    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(torch.cat([x, t], dim=1)).reshape(-1, 1)
    out = {"u": u, "x": x, "t": t}

    def grad(y, z):
        return torch.autograd.grad(y, z, torch.ones_like(y),
                                   create_graph=True)[0]

    if {"u_t", "u_tt", "u_xt"} & set(wanted):
        out["u_t"] = grad(u, t)
    if {"u_x", "u_xx", "u_xt"} & set(wanted):
        out["u_x"] = grad(u, x)
    if "u_xx" in wanted:
        out["u_xx"] = grad(out["u_x"], x)
    if "u_tt" in wanted:
        out["u_tt"] = grad(out["u_t"], t)
    if "u_xt" in wanted:
        out["u_xt"] = grad(out["u_x"], t)
    return out


def default_network(width: int = 32, depth: int = 4):
    """tanh, because ReLU has a zero second derivative almost everywhere and
    a diffusion term would silently vanish from the residual."""
    import torch.nn as nn

    layers: List[Any] = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def network_from_design(graph: Dict[str, Any]):
    """Use the design on the canvas, if it is shaped for u(x, t)."""
    import torch

    import codegen
    import graph as G
    import train as T

    g = G.parse(graph)
    report = G.analyze(g)
    source = codegen.to_pytorch(g, report)
    model = T.build_model(source, codegen.model_class_name(g))
    probe = model(torch.zeros(4, 2))
    probe = probe[0] if isinstance(probe, (tuple, list)) else probe
    if probe.numel() != 4:
        raise PhysicsError(
            f"The design maps 2 inputs to {probe.shape[-1] if probe.dim() else 1} "
            f"outputs. A PINN needs u(x, t): an Input of shape [2] and a "
            f"single output.")
    return model


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(spec: Dict[str, Any], model=None,
          progress: Optional[Callable[[Dict[str, Any]], None]] = None,
          stop: Optional[threading.Event] = None) -> Dict[str, Any]:
    import torch

    torch.manual_seed(int(spec.get("seed", 0)))
    parsed = parse(spec["pde"], spec["ic"], spec.get("bc") or {})
    x0, x1 = map(float, spec["x"])
    t0, t1 = map(float, spec["t"])
    sym = parsed["symbols"]

    model = model or default_network(int(spec.get("width", 32)),
                                     int(spec.get("depth", 4)))
    # float32 by default. I expected float64 to rescue L-BFGS on Burgers and
    # measured otherwise: float32 ran 1,652 L-BFGS evaluations without
    # stalling and reached 16%, float64 reached 32% and took longer. The
    # shock was not a precision problem. float64 stays available.
    dtype = torch.float64 if spec.get("precision", "float32") == "float64" \
        else torch.float32
    model = model.to(dtype)
    iterations = max(10, min(20000, int(spec.get("iterations", 3000))))
    n_pde = max(100, min(20000, int(spec.get("collocation", 2000))))
    n_edge = max(20, min(2000, int(spec.get("edge_points", 200))))
    w = {"pde": 1.0, "ic": 1.0, "bc": 1.0, **(spec.get("weights") or {})}

    residual_fn = _torch_fn(parsed["residual"],
                            tuple(sym[n] for n in NAMES))
    ic_fn = _torch_fn(parsed["ic"], (sym["x"],))
    bc_fns = {side: _torch_fn(expr, (sym["t"],))
              for side, expr in parsed["bc"].items()}

    def uniform(n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, 1, dtype=dtype)

    xc, tc = uniform(n_pde, x0, x1), uniform(n_pde, t0, t1)
    xi = uniform(n_edge, x0, x1)
    ti0 = torch.full_like(xi, t0)
    tb = uniform(n_edge, t0, t1)

    def values(fn, arg):
        v = fn(arg)
        return v if torch.is_tensor(v) else torch.full_like(arg, float(v))

    ui_target = values(ic_fn, xi)
    if parsed["bc_kind"] == "dirichlet":
        left_target = values(bc_fns["left"], tb)
        right_target = values(bc_fns["right"], tb)

    def losses():
        d = derivatives(model, xc, tc, parsed["used"])
        feed = [d.get(n, d["u"] * 0 if n.startswith("u_") else None)
                for n in NAMES]
        feed[NAMES.index("x")] = d["x"]
        feed[NAMES.index("t")] = d["t"]
        r = residual_fn(*feed)
        r = r if torch.is_tensor(r) else torch.zeros_like(d["u"])
        pde = torch.mean(r ** 2)
        ui = model(torch.cat([xi, ti0], dim=1)).reshape(-1, 1)
        ic = torch.mean((ui - ui_target) ** 2)
        lo = model(torch.cat([torch.full_like(tb, x0), tb], 1)).reshape(-1, 1)
        hi = model(torch.cat([torch.full_like(tb, x1), tb], 1)).reshape(-1, 1)
        if parsed["bc_kind"] == "periodic":
            bc = torch.mean((lo - hi) ** 2)
        else:
            bc = torch.mean((lo - left_target) ** 2) \
                + torch.mean((hi - right_target) ** 2)
        total = w["pde"] * pde + w["ic"] * ic + w["bc"] * bc
        return total, {"pde": float(pde.detach()), "ic": float(ic.detach()), "bc": float(bc.detach())}

    # A ReLU network has u_xx = 0 almost everywhere, so a diffusion term
    # contributes nothing to the residual and the network is never told about
    # it. That is measured here rather than inferred from the layer names.
    warnings: List[str] = []
    if "u_xx" in parsed["used"]:
        probe = derivatives(model, xc[:200], tc[:200], ["u_xx"])["u_xx"]
        if float(torch.max(torch.abs(probe.detach()))) == 0.0:
            warnings.append(
                "This network's second derivative is exactly zero at every "
                "point tested — usually ReLU, which is piecewise linear. The "
                "u_xx term contributes nothing, so the network is never "
                "penalised for the diffusion in the equation. Use tanh or "
                "SiLU.")

    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=float(spec.get("lr", 1e-3)))
    history: List[Dict[str, Any]] = []
    started = time.time()
    # Residual-based adaptive refinement (Lu et al., DeepXDE, 2021): every so
    # often, look for where the equation is violated worst and add points
    # there. Uniform sampling spends almost all its points where the solution
    # is smooth and easy, and very few on a shock, which is the one place the
    # network most needs to be told it is wrong.
    adaptive = bool(spec.get("adaptive", True))
    rounds = 4
    added = 0
    for step in range(1, iterations + 1):
        if stop is not None and stop.is_set():
            break
        if adaptive and step % max(1, iterations // (rounds + 1)) == 0 \
                and added < rounds:
            pool_x, pool_t = uniform(6000, x0, x1), uniform(6000, t0, t1)
            d = derivatives(model, pool_x, pool_t, parsed["used"])
            feed = [d.get(n, d["u"] * 0 if n.startswith("u_") else None)
                    for n in NAMES]
            feed[NAMES.index("x")] = d["x"]
            feed[NAMES.index("t")] = d["t"]
            worst = torch.abs(residual_fn(*feed).detach().reshape(-1))
            keep = torch.topk(worst, max(1, n_pde // 8)).indices
            xc = torch.cat([xc, pool_x[keep].detach()])
            tc = torch.cat([tc, pool_t[keep].detach()])
            added += 1
        optimiser.zero_grad()
        total, parts = losses()
        total.backward()
        optimiser.step()
        if step % max(1, iterations // 120) == 0 or step == 1:
            row = {"step": step, "total": float(total.detach()), **parts}
            history.append(row)
            if progress:
                progress({"step": step, "of": iterations, **row})

    # L-BFGS to finish, as Raissi et al. did. Adam gets the broad shape and
    # stalls before the sharp features; a quasi-Newton step on the same loss
    # is what resolves the Burgers shock. Kept separate in the history so the
    # change of optimiser is visible on the curve rather than looking like a
    # sudden improvement from nowhere.
    refine = max(0, min(5000, int(spec.get("lbfgs", 500))))
    if refine and not (stop is not None and stop.is_set()):
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0,
                                  max_iter=refine, history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")
        calls = [0]

        def closure():
            lbfgs.zero_grad()
            total, parts = losses()
            total.backward()
            calls[0] += 1
            if calls[0] % max(1, refine // 30) == 0:
                row = {"step": iterations + calls[0], "total": float(total.detach()),
                       "phase": "lbfgs", **parts}
                history.append(row)
                if progress:
                    progress({"of": iterations + refine, **row})
            return total

        lbfgs.step(closure)
        total, parts = losses()
        history.append({"step": iterations + calls[0], "total": float(total.detach()),
                        "phase": "lbfgs", **parts})

    result: Dict[str, Any] = {
        "history": history, "warnings": warnings,
        "seconds": round(time.time() - started, 1),
        "iterations": len(history) and history[-1]["step"],
        "collocation_final": int(xc.shape[0]),
        "spec": {k: spec.get(k) for k in ("pde", "ic", "bc", "x", "t")},
    }

    # residual on points the network was not trained on
    with torch.enable_grad():
        xf, tf = uniform(2000, x0, x1), uniform(2000, t0, t1)
        d = derivatives(model, xf, tf, parsed["used"])
        feed = [d.get(n, d["u"] * 0 if n.startswith("u_") else None)
                for n in NAMES]
        feed[NAMES.index("x")] = d["x"]
        feed[NAMES.index("t")] = d["t"]
        r = residual_fn(*feed)
        result["residual_rms_unseen"] = float(
            torch.sqrt(torch.mean(r ** 2)).detach())

    ref = reference(parsed, spec["x"], spec["t"])
    if not ref["ok"]:
        result["reference"] = {"ok": False, "why": ref["why"]}
        result["verdict"] = {"kind": "unverified", "text": ref["why"]}
        return result

    X, Tm = np.meshgrid(ref["x"], ref["t"])
    with torch.no_grad():
        grid = torch.tensor(np.stack([X.ravel(), Tm.ravel()], 1),
                            dtype=dtype)
        pred = model(grid).reshape(-1).numpy().reshape(X.shape)
    truth = ref["u"]
    error = float(np.linalg.norm(pred - truth) /
                  max(np.linalg.norm(truth), 1e-12))

    def thin(a):
        rows = np.linspace(0, a.shape[0] - 1, 51).astype(int)
        cols = np.linspace(0, a.shape[1] - 1, 64).astype(int)
        return np.round(a[np.ix_(rows, cols)], 4).tolist()

    result["reference"] = {"ok": True, "how": ref["how"]}
    result["relative_l2"] = error
    result["grids"] = {"pinn": thin(pred), "reference": thin(truth),
                       "error": thin(np.abs(pred - truth)),
                       "x": [float(ref["x"][0]), float(ref["x"][-1])],
                       "t": [float(ref["t"][0]), float(ref["t"][-1])]}

    final = history[-1]["total"] if history else float("nan")
    residual_term = history[-1]["pde"] if history else float("nan")
    if error < 0.05:
        verdict = ("matched", f"Within {error:.1%} of the reference solution "
                   f"across the whole domain.")
    elif error < 0.2:
        verdict = ("rough", f"{error:.1%} from the reference. The broad shape "
                   f"is right and the detail is not — look at where the "
                   f"error map is bright.")
    else:
        verdict = ("failed", f"{error:.0%} from the reference. The equation "
                   f"was not solved.")
    result["verdict"] = {"kind": verdict[0], "text": verdict[1]}

    # The failure worth naming. It is the *residual* that matters, not the
    # total: a network can satisfy the equation almost everywhere with a
    # nearly flat function while the initial or boundary term stays high, and
    # the total then hides that the equation itself looks solved. Checking the
    # total missed exactly the case this exists for.
    if error >= 0.2 and residual_term < 1e-2:
        result["verdict"]["text"] += (
            f" The equation's residual reached {residual_term:.1e}, which on "
            f"its own would look like success. This is the documented "
            f"failure: the network found a function that very nearly "
            f"satisfies the equation and is not the solution — usually a "
            f"near-constant one, which satisfies every transport equation "
            f"there is.")
    return result


# --------------------------------------------------------------------------
# runs in the background, filed to the account that started them
# --------------------------------------------------------------------------

RUNS: Dict[str, Dict[str, Any]] = {}


def home() -> Path:
    import auth

    return auth.sub("physics")


def start(spec: Dict[str, Any], graph: Optional[Dict[str, Any]] = None
          ) -> Dict[str, Any]:
    parse(spec.get("pde", ""), spec.get("ic", ""), spec.get("bc") or {})
    model = None
    used = (f"a tanh network, {int(spec.get('depth', 4))} layers of "
            f"{int(spec.get('width', 32))}")
    if spec.get("use_design") and graph:
        model = network_from_design(graph)
        used = "the design on the canvas"

    run = {"id": uuid.uuid4().hex[:12], "status": "running", "progress": [],
           "result": None, "error": "", "network": used,
           "spec": spec, "started": time.time(),
           "stop": threading.Event(), "folder": home()}
    RUNS[run["id"]] = run

    def work():
        try:
            run["result"] = train(spec, model,
                                  progress=lambda row: run["progress"].append(row),
                                  stop=run["stop"])
            run["status"] = "stopped" if run["stop"].is_set() else "done"
        except Exception as exc:  # noqa: BLE001
            run["error"] = f"{type(exc).__name__}: {exc}"
            run["status"] = "error"
        try:
            run["folder"].mkdir(parents=True, exist_ok=True)
            (run["folder"] / f"{run['id']}.json").write_text(
                json.dumps(snapshot(run)))
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=work, daemon=True).start()
    return snapshot(run)


def snapshot(run: Dict[str, Any]) -> Dict[str, Any]:
    return {k: run[k] for k in ("id", "status", "result", "error",
                                "network", "spec", "started")} | {
        "progress": run["progress"][-200:]}
