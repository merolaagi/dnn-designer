"""Loads plug-in blocks from ``blocks/``.

A bad block should never take the designer down. Each file is executed in
isolation; if it raises, the failure is recorded with its traceback and every
other block still loads. The UI shows the failures next to the block list so a
typo is visible where you made it.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List

import layers

BLOCKS_DIR = Path(__file__).resolve().parent / "blocks"
BLOCKS_DIR.mkdir(exist_ok=True)

LAST_ERRORS: List[Dict[str, str]] = []

TEMPLATE = '''"""One-line description of what this block does."""

from blocks_sdk import Block, Param, ShapeError, install, need_rank

PRELUDE = """
class {name}(nn.Module):
    def __init__(self, channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        return x + self.scale * self.proj(x)
"""


def infer(p, shapes):
    # Return the output shape, channels-first and without the batch dimension.
    # Raise ShapeError with a plain sentence if the input cannot be accepted.
    return list(shapes[0])


install(Block(
    name="{name}",
    category="Custom blocks",
    doc="What this does and when to reach for it.",
    params=[
        Param("scale", "float", 1.0, help="Weight on the residual branch"),
    ],
    infer=infer,
    prelude=PRELUDE,
    torch_init=lambda p, ins: f"{name}({{ins[0][-1]}}, scale={{float(p['scale'])}})",
))
'''


def _module_globals() -> Dict[str, Any]:
    return {"__name__": "designer_block"}


def load_all() -> List[Dict[str, str]]:
    """Reinstall every block from disk. Returns the list of failures."""
    layers.snapshot_core()
    layers.drop_blocks()
    LAST_ERRORS.clear()

    for path in sorted(BLOCKS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        before = set(layers.REGISTRY)
        try:
            source = path.read_text()
            exec(compile(source, str(path), "exec"), _module_globals())  # noqa: S102
            for name in set(layers.REGISTRY) - before:
                layers.REGISTRY[name].origin = path.name
        except Exception as exc:  # noqa: BLE001
            LAST_ERRORS.append({
                "file": path.name,
                "message": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc()[-1400:],
            })
    return list(LAST_ERRORS)


def listing() -> List[Dict[str, Any]]:
    """Block files on disk, with the specs each one installed."""
    by_file: Dict[str, List[str]] = {}
    for spec in layers.REGISTRY.values():
        if spec.source == "block":
            by_file.setdefault(spec.name, []).append(spec.name)

    errors = {e["file"]: e for e in LAST_ERRORS}
    out = []
    for path in sorted(BLOCKS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        installed = [
            s.name for s in layers.REGISTRY.values() if s.origin == path.name
        ]
        out.append({
            "file": path.name,
            "installs": installed,
            "bytes": path.stat().st_size,
            "error": errors.get(path.name, {}).get("message"),
        })
    return out


def _declares(path: Path, name: str) -> bool:
    try:
        return f'name="{name}"' in path.read_text() or f"name='{name}'" in path.read_text()
    except OSError:
        return False


def read(file: str) -> str:
    path = BLOCKS_DIR / Path(file).name
    if not path.exists():
        raise FileNotFoundError(f"No block file named {path.name}.")
    return path.read_text()


def write(file: str, source: str) -> List[Dict[str, str]]:
    """Save a block file and reload everything. Returns failures, if any."""
    name = Path(file).name
    if not name.endswith(".py"):
        name += ".py"
    (BLOCKS_DIR / name).write_text(source)
    return load_all()


def delete(file: str) -> List[Dict[str, str]]:
    (BLOCKS_DIR / Path(file).name).unlink(missing_ok=True)
    return load_all()


def scaffold(name: str) -> str:
    clean = "".join(c for c in name if c.isalnum()) or "MyBlock"
    return TEMPLATE.replace("{name}", clean)
