"""Unpack an archive of source and describe what is inside it.

This is deliberately not the model importer. That looks for `nn.Module`
subclasses it can trace onto the canvas, and pointed at a research engine it
finds nothing and says nothing useful. Most code is not a neural network.

What can honestly be said about any Python codebase, without running it:

    the tree            files, sizes, where the code actually is
    per module          classes, functions, how long, what it imports
    the import graph    which modules depend on which, and the cycles

The import graph is the interesting one, and it is the same kind of object the
canvas already deals in: nodes, edges, and questions about reachability. A cycle
between modules is the same defect as a cycle in a proof DAG — worth finding and
worth naming.

Nothing here executes the code. The archive is read, parsed with `ast`, and
described. A file that will not parse is reported as such rather than skipped.
"""

from __future__ import annotations

import ast
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

#: Directories that are never the codebase itself
NOISE = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache",
         ".pytest_cache", ".idea", ".vscode", "dist", "build", ".eggs"}

MAX_BYTES = 60 * 1024 * 1024
MAX_FILES = 4000


class ArchiveError(ValueError):
    """Something about the archive itself, said plainly."""


def unpack(archive: Path, into: Path) -> Path:
    """Extract, refusing anything that would write outside the target.

    A zip entry may name `../../etc/passwd`. Python's extractall has grown
    guards for this over the years and their behaviour differs by version, so
    the check is done here where it can be relied on.
    """
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()

    def safe(name: str) -> bool:
        target = (root / name).resolve()
        return root == target or root in target.parents

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            if len(members) > MAX_FILES:
                raise ArchiveError(
                    f"{len(members):,} entries is more than this reads "
                    f"({MAX_FILES:,}).")
            total = sum(i.file_size for i in bundle.infolist())
            if total > MAX_BYTES:
                raise ArchiveError(
                    f"{total / 1e6:.0f} MB unpacked is more than this reads "
                    f"({MAX_BYTES / 1e6:.0f} MB).")
            bad = [n for n in members if not safe(n)]
            if bad:
                raise ArchiveError(
                    f"The archive tries to write outside its folder "
                    f"({bad[0]}), so it was not opened.")
            bundle.extractall(root)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            if len(members) > MAX_FILES:
                raise ArchiveError(f"{len(members):,} entries is too many.")
            bad = [m.name for m in members if not safe(m.name)]
            if bad:
                raise ArchiveError(
                    f"The archive tries to write outside its folder "
                    f"({bad[0]}), so it was not opened.")
            bundle.extractall(root)
    else:
        raise ArchiveError(
            "Not a zip or a tar. Those are the two this reads.")

    # a bundle that is one folder containing everything: descend into it, so
    # the tree starts where the project does
    entries = [p for p in root.iterdir() if p.name not in NOISE]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


@dataclass
class Module:
    path: str
    lines: int
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    error: str = ""


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def read_module(path: Path, root: Path) -> Module:
    source = path.read_text(errors="replace")
    entry = Module(path=str(path.relative_to(root)),
                   lines=source.count("\n") + 1)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # said, not skipped: a file that will not parse is a fact about the
        # codebase, and silently dropping it makes the totals lie
        entry.error = f"line {exc.lineno}: {exc.msg}"
        return entry

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            entry.classes.append({
                "name": node.name,
                "line": node.lineno,
                "bases": [ast.unparse(b) for b in node.bases][:4],
                "methods": [b.name for b in node.body
                            if isinstance(b, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))][:24],
                "doc": (ast.get_docstring(node) or "").split("\n")[0][:160],
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry.functions.append(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            entry.imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # a relative import
                here = _module_name(path, root).rsplit(".", node.level)[0]
                base = f"{here}.{node.module}" if node.module and here \
                    else (node.module or here)
            else:
                base = node.module or ""
            if not base:
                continue
            entry.imports.append(base)
            # `from pkg import b` names a module as often as it names a symbol,
            # and recording only `pkg` pointed every such edge at the package
            # instead of the module — which hid import cycles entirely.
            for alias in node.names:
                if alias.name != "*":
                    entry.imports.append(f"{base}.{alias.name}")
    return entry


def tree_of(root: Path) -> Dict[str, Any]:
    """The folder structure, with the noise left out."""

    def walk(here: Path) -> Dict[str, Any]:
        kids = []
        for child in sorted(here.iterdir(),
                            key=lambda p: (p.is_file(), p.name.lower())):
            if child.name in NOISE or child.name.startswith("."):
                continue
            if child.is_dir():
                node = walk(child)
                if node["children"]:
                    kids.append(node)
            else:
                kids.append({"name": child.name, "kind": "file",
                             "path": str(child.relative_to(root)),
                             "bytes": child.stat().st_size,
                             "python": child.suffix == ".py"})
        return {"name": here.name, "kind": "dir",
                "path": str(here.relative_to(root)) if here != root else "",
                "children": kids}

    return walk(root)


def survey(root: Path) -> Dict[str, Any]:
    """Everything that can be said without running any of it."""
    modules: List[Module] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in NOISE for part in path.parts):
            continue
        modules.append(read_module(path, root))

    own = {_module_name(root / m.path, root): m for m in modules}
    edges = []
    for name, module in own.items():
        for target in set(module.imports):
            # only edges inside the project: the interesting graph is the one
            # the author drew, not their dependency on the standard library
            hit = _resolve(target, own)
            if hit and hit != name:
                edges.append({"from": name, "to": hit})

    fan_in: Dict[str, int] = {name: 0 for name in own}
    for edge in edges:
        fan_in[edge["to"]] = fan_in.get(edge["to"], 0) + 1

    outside: Dict[str, int] = {}
    for module in modules:
        for target in set(module.imports):
            if not _resolve(target, own):
                outside[target.split(".")[0]] = outside.get(
                    target.split(".")[0], 0) + 1

    return {
        "modules": [m.__dict__ for m in modules],
        "edges": edges,
        "cycles": _cycles(own, edges),
        "depended_on": sorted(fan_in.items(), key=lambda kv: -kv[1])[:8],
        "outside": sorted(outside.items(), key=lambda kv: -kv[1])[:14],
        "unparsed": [m.path for m in modules if m.error],
        "totals": {
            "files": len(modules),
            "lines": sum(m.lines for m in modules),
            "classes": sum(len(m.classes) for m in modules),
            "functions": sum(len(m.functions) for m in modules),
        },
    }


def _resolve(target: str, own: Dict[str, Module]) -> Optional[str]:
    """Which module of this project an import refers to, if any."""
    if target in own:
        return target
    # a.b.c may name a symbol inside a.b
    parts = target.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in own:
            return candidate
    # top-level package prefix, e.g. "mare.agents" inside package "mare"
    tail = parts[-1]
    return tail if tail in own else None


def _cycles(own: Dict[str, Module], edges: List[Dict[str, str]]) -> List[List[str]]:
    """Import cycles, which are the same defect as a cycle in a proof graph."""
    graph: Dict[str, Set[str]] = {name: set() for name in own}
    for edge in edges:
        graph.setdefault(edge["from"], set()).add(edge["to"])

    found: List[List[str]] = []
    seen: Set[str] = set()
    stack: List[str] = []
    on_stack: Set[str] = set()

    def walk(node: str) -> None:
        seen.add(node)
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt in on_stack:
                loop = stack[stack.index(nxt):] + [nxt]
                if loop not in found:
                    found.append(loop)
            elif nxt not in seen:
                walk(nxt)
        stack.pop()
        on_stack.discard(node)

    for name in sorted(graph):
        if name not in seen:
            walk(name)
    return found[:12]


def read_file(root: Path, relative: str, limit: int = 4000) -> Dict[str, Any]:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ArchiveError("That path is outside the project.")
    if not target.exists() or target.is_dir():
        raise ArchiveError("No such file in this project.")
    text = target.read_text(errors="replace")
    lines = text.split("\n")
    return {"path": relative, "lines": len(lines),
            "truncated": len(lines) > limit,
            "text": "\n".join(lines[:limit])}


def forget(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
