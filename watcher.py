"""Run a project and record which functions actually called which.

Reading gives certainty about the calls a compiler could see. It cannot see
`self.provider.run()`, because which object that provider is depends on what
was passed in — and in a well-made system that is most of the interesting
calls. MARE's own agents draw no static edges at all for exactly this reason.

Running it answers the other half. The cost is that it is running somebody's
code, so:

  - it never runs in the application process, only in a subprocess;
  - it is never automatic — an entry point has to be named and the run asked
    for, each time;
  - it is killed after a timeout, and the timeout is reported rather than
    disguised as an empty result;
  - it records only calls whose code lives inside the project, so the trace is
    about the project rather than about the standard library.

This is not a sandbox. A subprocess with a timeout stops an accident, not an
attacker: the code can still read files and open sockets as the user running
the server. The interface says so, because a person deciding whether to press
the button needs to know which of those two it is.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

RUNNER = r'''
import json, os, runpy, sys, threading

ROOT = os.path.abspath(sys.argv[1])
TARGET = sys.argv[2]
OUT = sys.argv[3]

edges = {}
counts = {}
stack = []


def inside(filename):
    # "<frozen runpy>" and "<string>" are not paths. abspath() resolves them
    # against the working directory, which IS the project, so every runpy and
    # posixpath frame looked like project code and the trace filled up with
    # the machinery that started it.
    if not filename or filename.startswith("<"):
        return False
    try:
        full = os.path.abspath(filename)
        return full.startswith(ROOT + os.sep) and os.path.exists(full)
    except Exception:
        return False


def name_of(frame):
    code = frame.f_code
    rel = os.path.relpath(os.path.abspath(code.co_filename), ROOT)
    # try to qualify a method with its class, which is what the diagram draws
    owner = ""
    me = frame.f_locals.get("self")
    if me is not None:
        try:
            owner = type(me).__name__
        except Exception:
            owner = ""
    return rel, (owner + "." + code.co_name if owner else code.co_name)


def trace(frame, event, arg):
    if event != "call":
        return None
    code = frame.f_code
    if not inside(code.co_filename):
        return None
    here = name_of(frame)
    counts[repr(here)] = counts.get(repr(here), 0) + 1
    caller = frame.f_back
    while caller is not None and not inside(caller.f_code.co_filename):
        caller = caller.f_back
    if caller is not None:
        pair = repr((name_of(caller), here))
        edges[pair] = edges.get(pair, 0) + 1
    return None


def dump(status, detail=""):
    with open(OUT, "w") as fh:
        json.dump({"status": status, "detail": detail,
                   "edges": list(edges.items()),
                   "counts": list(counts.items())}, fh)


sys.path.insert(0, ROOT)
sys.setrecursionlimit(3000)
threading.settrace(trace)
sys.settrace(trace)
try:
    if TARGET.endswith(".py"):
        runpy.run_path(os.path.join(ROOT, TARGET), run_name="__main__")
    else:
        runpy.run_module(TARGET, run_name="__main__")
    sys.settrace(None)
    dump("ran")
except SystemExit as exc:
    sys.settrace(None)
    dump("ran", "exited with %s" % (exc.code,))
except BaseException as exc:
    sys.settrace(None)
    dump("raised", "%s: %s" % (type(exc).__name__, exc))
'''


def watch(root: Path, target: str, seconds: int = 20) -> Dict[str, Any]:
    """Run `target` inside `root` and report what called what."""
    root = root.resolve()
    if seconds < 1 or seconds > 120:
        seconds = 20

    with tempfile.TemporaryDirectory() as scratch:
        runner = Path(scratch) / "watch_runner.py"
        runner.write_text(RUNNER)
        out = Path(scratch) / "trace.json"
        try:
            done = subprocess.run(
                [sys.executable, str(runner), str(root), target, str(out)],
                cwd=str(root), capture_output=True, text=True,
                timeout=seconds,
                env={"PATH": "", "HOME": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return {"status": "timeout",
                    "detail": f"Still running after {seconds}s, so it was "
                              f"stopped. Anything it had done by then stands; "
                              f"what it would have done next is unknown.",
                    "stdout": (exc.stdout or b"").decode(errors="replace")[-4000:]
                    if isinstance(exc.stdout, bytes) else (exc.stdout or "")[-4000:],
                    "stderr": "", "edges": [], "counts": [], "entered": 0}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}",
                    "stdout": "", "stderr": "", "edges": [], "counts": [],
                    "entered": 0}

        blob: Dict[str, Any] = {}
        if out.exists():
            try:
                blob = json.loads(out.read_text())
            except Exception:  # noqa: BLE001
                blob = {}

    edges = []
    for key, times in blob.get("edges", []):
        try:
            (from_file, from_name), (to_file, to_name) = eval(key)  # noqa: S307
        except Exception:  # noqa: BLE001
            continue
        edges.append({"from_file": from_file, "from": from_name,
                      "to_file": to_file, "to": to_name, "times": times})

    entered = []
    for key, times in blob.get("counts", []):
        try:
            where, what = eval(key)  # noqa: S307
        except Exception:  # noqa: BLE001
            continue
        entered.append({"file": where, "name": what, "times": times})
    entered.sort(key=lambda e: -e["times"])

    return {
        "status": blob.get("status") or ("failed" if done.returncode else "ran"),
        "detail": blob.get("detail", ""),
        "returncode": done.returncode,
        "stdout": (done.stdout or "")[-4000:],
        "stderr": (done.stderr or "")[-4000:],
        "edges": edges,
        "entered": entered[:200],
        "functions": len(entered),
    }


def entry_points(root: Path) -> List[str]:
    """Files worth offering: things with a __main__ guard, and the package."""
    root = root.resolve()
    out = []
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if {"__pycache__", ".venv", "node_modules"} & parts:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if '__name__ == "__main__"' in text or "__name__ == '__main__'" in text:
            out.append(str(path.relative_to(root)))
    # a test file is a fine entry point and often the only one that runs alone
    for path in sorted(root.rglob("test_*.py")):
        rel = str(path.relative_to(root))
        if rel not in out:
            out.append(rel)
    return out[:40]
