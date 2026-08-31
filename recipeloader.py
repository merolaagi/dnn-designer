"""Loads training recipes from ``recipes/``, the same way blocks load."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List

import recipes_sdk

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"
RECIPES_DIR.mkdir(exist_ok=True)

LAST_ERRORS: List[Dict[str, str]] = []

TEMPLATE = '''"""One line on what this training loop does."""

from recipes_sdk import Param, Recipe, install


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(
        ctx.parameters(), lr=float(ctx.cfg["lr"]))


def step(ctx, xs, y):
    # Own the backward pass and the optimizer step. Return numbers to chart.
    import torch.nn.functional as F
    out = ctx.model(*xs)
    loss = F.mse_loss(out, xs[0])
    ctx.optimizers["main"].zero_grad(set_to_none=True)
    loss.backward()
    ctx.optimizers["main"].step()
    return {"loss": float(loss.item())}


def evaluate(ctx, xs, y):
    import torch.nn.functional as F
    out = ctx.model(*xs)
    return {"loss": float(F.mse_loss(out, xs[0]).item())}


install(Recipe(
    name="{name}",
    doc="What this does and when to reach for it.",
    params=[Param("lr", "float", 1e-3)],
    setup=setup, step=step, evaluate=evaluate,
))
'''


def load_all() -> List[Dict[str, str]]:
    recipes_sdk.REGISTRY.clear()
    LAST_ERRORS.clear()
    for path in sorted(RECIPES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            exec(compile(path.read_text(), str(path), "exec"),  # noqa: S102
                 {"__name__": "designer_recipe"})
        except Exception as exc:  # noqa: BLE001
            LAST_ERRORS.append({
                "file": path.name,
                "message": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc()[-1400:],
            })
    return list(LAST_ERRORS)


def listing() -> List[Dict[str, Any]]:
    errors = {e["file"]: e for e in LAST_ERRORS}
    out = []
    for path in sorted(RECIPES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        text = path.read_text()
        installs = [r.name for r in recipes_sdk.REGISTRY.values()
                    if f'name="{r.name}"' in text or f"name='{r.name}'" in text]
        out.append({"file": path.name, "installs": installs,
                    "bytes": path.stat().st_size,
                    "error": errors.get(path.name, {}).get("message")})
    return out


def read(file: str) -> str:
    path = RECIPES_DIR / Path(file).name
    if not path.exists():
        raise FileNotFoundError(f"No recipe file named {path.name}.")
    return path.read_text()


def write(file: str, source: str) -> List[Dict[str, str]]:
    name = Path(file).name
    if not name.endswith(".py"):
        name += ".py"
    (RECIPES_DIR / name).write_text(source)
    return load_all()


def delete(file: str) -> List[Dict[str, str]]:
    (RECIPES_DIR / Path(file).name).unlink(missing_ok=True)
    return load_all()


def scaffold(name: str) -> str:
    clean = "".join(c for c in name if c.isalnum()) or "MyRecipe"
    return TEMPLATE.replace("{name}", clean)
