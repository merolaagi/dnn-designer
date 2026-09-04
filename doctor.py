#!/usr/bin/env python3
"""Check this machine has what the app needs.

    python doctor.py

Reports what is installed, what each thing unlocks, and what is lost without
it. Nothing here is fatal on its own — the app runs with only the first three —
so the point is to say plainly which features are available rather than to pass
or fail.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

GREEN, AMBER, RED, GREY, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[0m"


def paint(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{OFF}"


def version_of(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001
        module = sys.modules.get(name)
        return getattr(module, "__version__", "?")


def probe(module: str, package: str = "") -> tuple:
    try:
        importlib.import_module(module)
        return True, version_of(package or module)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).split("(")[0].strip()


REQUIRED = [
    ("fastapi", "fastapi", "serves the app"),
    ("uvicorn", "uvicorn", "runs the server"),
    ("pydantic", "pydantic", "validates what the browser sends"),
]

# Each entry says what it unlocks, and what remains without it — every one of
# these is optional, so "missing" should describe the loss rather than alarm.
MATTERS = [
    ("torch", "torch",
     "training, importing models, testing a layer on its own",
     "the canvas still draws, checks shapes and writes code, but nothing runs"),
    ("torchvision", "torchvision",
     "importing resnet, efficientnet and the rest by name",
     "pasted code and folder scanning still import models"),
    ("transformers", "transformers",
     "scanning the model library installed on this machine",
     "your own project folders can still be scanned"),
    ("onnx", "onnx",
     "importing .onnx exports",
     "the other import routes are unaffected"),
    ("numpy", "numpy",
     "loading CSV and array datasets",
     "the built-in and image datasets still work"),
]


def _found_elsewhere(module: str) -> str:
    """Whether another interpreter on this machine has the package.

    Almost every 'but I installed it' is this: installed system-wide, checked
    from inside a virtual environment, or the other way round.
    """
    import subprocess

    candidates = []
    base = getattr(sys, "base_prefix", sys.prefix)
    if base != sys.prefix:
        candidates.append(str(Path(base) / "bin" / "python3"))
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and found != sys.executable:
            candidates.append(found)

    for candidate in dict.fromkeys(candidates):
        if not Path(candidate).exists():
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", f"import {module}"],
                capture_output=True, timeout=25)
            if result.returncode == 0:
                return candidate
        except Exception:  # noqa: BLE001
            continue
    return ""


def main() -> None:
    print()
    print(paint("  Deep Network Designer — environment", GREY))
    print()

    ok = sys.version_info >= (3, 11)
    mark = paint("yes", GREEN) if ok else paint("too old", RED)
    print(f"  python              {platform.python_version():16s} {mark}")
    if not ok:
        print(paint("                      3.11 or newer is needed\n", RED))

    # Which interpreter this is. A package installed system-wide is invisible
    # from inside a virtual environment, and the report saying "absent" without
    # saying "absent from here" sends people looking in the wrong place.
    inside = (getattr(sys, "base_prefix", sys.prefix) != sys.prefix
              or bool(os.environ.get("VIRTUAL_ENV")))
    where = os.environ.get("VIRTUAL_ENV") or sys.prefix
    print(f"  {'':22s}{paint(sys.executable, GREY)}")
    if inside:
        print(f"  {'':22s}{paint('a virtual environment: ' + where, GREY)}")
        print(f"  {'':22s}{paint('anything installed outside it does not count here', GREY)}")

    missing_required = []
    for module, package, what in REQUIRED:
        present, detail = probe(module, package)
        state = paint("yes", GREEN) if present else paint("MISSING", RED)
        print(f"  {module:19s} {detail if present else '':16s} {state}   {paint(what, GREY)}")
        if not present:
            missing_required.append(module)

    print()
    absent = []
    for module, package, unlocks, without in MATTERS:
        present, detail = probe(module, package)
        state = paint("yes", GREEN) if present else paint("absent", AMBER)
        print(f"  {module:19s} {detail if present else '':16s} {state}   {paint(unlocks, GREY)}")
        if not present:
            absent.append(package or module)
            print(f"  {'':36s} {paint('without it, ' + without, AMBER)}")
            # is it installed for a different interpreter? that is the usual cause
            elsewhere = _found_elsewhere(module)
            if elsewhere:
                print(f"  {'':36s} "
                      + paint(f"it is installed for {elsewhere} — but not here", AMBER))

    # what torch can actually use
    try:
        import torch

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append(f"cuda ({torch.cuda.get_device_name(0)})")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps (Apple GPU)")
        print()
        print(f"  training devices    {paint(', '.join(devices), GREEN)}")
        if len(devices) == 1:
            print(f"  {'':22s}{paint('CPU only — fine for small models, slow for large ones', GREY)}")
    except Exception:  # noqa: BLE001
        pass

    # which model families are readable from the installed library
    try:
        import transformers

        models = Path(transformers.__file__).parent / "models"
        families = sorted(p.name for p in models.iterdir()
                          if p.is_dir() and not p.name.startswith("_"))
        notable = [n for n in ("gpt2", "llama", "mistral", "mixtral", "qwen2",
                               "qwen3", "deepseek_v3", "gemma2", "phi3")
                   if n in families]
        print()
        print(f"  model families      {paint(str(len(families)) + ' readable', GREEN)}")
        print(f"  {'':22s}{paint('including ' + ', '.join(notable), GREY)}")
        print(f"  {'':22s}{paint('scan: ' + str(models), GREY)}")
    except Exception:  # noqa: BLE001
        pass

    # can the app write where it needs to
    print()
    trouble = []
    for name in ("saved", "runs", "studies", "blocks", "recipes", "data"):
        target = HERE / name
        try:
            target.mkdir(exist_ok=True)
            probe_file = target / ".writable"
            probe_file.write_text("x")
            probe_file.unlink()
        except Exception as exc:  # noqa: BLE001
            trouble.append(f"{name}: {exc}")
    if trouble:
        print(f"  storage             {paint('not writable', RED)}")
        for line in trouble:
            print(f"  {'':22s}{line}")
    else:
        print(f"  storage             {paint('writable', GREEN)}")

    free = shutil.disk_usage(HERE).free / 1e9
    note = GREEN if free > 5 else AMBER
    print(f"  free space          {paint(f'{free:.1f} GB', note)}"
          + ("" if free > 5 else paint("   checkpoints may not fit", AMBER)))

    accounts = HERE / "data" / "users.json"
    if accounts.exists():
        try:
            import json

            names = sorted(json.loads(accounts.read_text()))
            print(f"  accounts            {paint(', '.join(names) or 'none', GREEN)}")
        except Exception:  # noqa: BLE001
            pass

    print()
    if missing_required:
        print(paint(f"  Install first: {sys.executable} -m pip install "
                    f"{' '.join(missing_required)}", RED))
    elif absent:
        print(paint("  Ready. uvicorn main:app --reload --port 8770", GREEN))
        print()
        print(paint(f"  For the rest: {sys.executable} -m pip install "
                    f"{' '.join(absent)}", GREY))
        print(paint("  (-m pip, so it goes to the interpreter checked above)", GREY))
    else:
        print(paint("  Ready. uvicorn main:app --reload --port 8770", GREEN))
    print()


if __name__ == "__main__":
    main()
