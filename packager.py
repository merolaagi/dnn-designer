"""Package a network as a folder anyone can install: code, weights, and a script.

A package is a zip holding one folder:

    model.py          the network, as the PyTorch file the designer generates
    weights.pt        the trained weights (absent if the network was never trained)
    model_info.json   what it takes in and puts out, how it was trained, and its numbers
    predict.py        run it on an image, a row of numbers, or JSON, from the command line
    serve.py          a small HTTP server: POST /predict, standard library only
    install.sh        make a virtual environment, install what it needs, self-test
    install.bat       the same for Windows
    requirements.txt  torch, and Pillow when the input is an image
    design.json       the design itself, to open again on the canvas
    README.md         all of the above, with the real numbers from training

Every package is built and then loaded and run once here, before it is offered, so a
package that does not load is refused rather than handed over.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import codegen
import graph as G
import train as T

LABELS = {
    "mnist": [str(i) for i in range(10)],
    "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
                      "Bag", "Ankle boot"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", name or "model").strip("-")[:40] or "model"


def checkpoints_for(design: str) -> List[Dict[str, Any]]:
    """Trained weights saved for a design, newest first."""
    try:
        found = T.list_checkpoints()
    except Exception:  # noqa: BLE001 - no torch, no checkpoints
        return []
    name = (design or "").strip()
    mine = [c for c in found if not c.get("error") and not c.get("chat")
            and (c.get("network") or "").split(" · ")[0] == name]
    # newest training run first, and within a run its best weights before its last
    runs: List[str] = []
    for c in mine:
        run = c["file"].rsplit("_", 2)[-2] if c["file"].count("_") >= 2 else c["file"]
        c["run"] = run
        if run not in runs:
            runs.append(run)
    return sorted(mine, key=lambda c: (runs.index(c["run"]), 0 if "_best" in c["file"] else 1))


def _run_record(job_id: str) -> Dict[str, Any]:
    try:
        return json.loads((T.runs_dir() / f"{job_id}.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}


PREDICT = r'''"""Run the packaged network. See README.md, or: python predict.py --help"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
INFO = json.loads((HERE / "model_info.json").read_text())
exec(f"from model import {INFO['class_name']} as Network")  # noqa: S102 - the class model.py defines


def load():
    """The network with its trained weights, ready to predict."""
    model = Network()
    weights = HERE / "weights.pt"
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model


def from_image(path, shape, invert=False):
    """An image file as the tensor the network was trained on: resized, scaled to 0..1."""
    from PIL import Image
    channels, height, width = shape
    image = Image.open(path).convert("L" if channels == 1 else "RGB").resize((width, height))
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float().view(height, width, -1)
    tensor = pixels.permute(2, 0, 1) / 255.0
    return 1.0 - tensor if invert else tensor


def from_numbers(values, shape):
    """A row of numbers, scaled the way training scaled them when that is known."""
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float32)
    scale = INFO.get("input_scaling")
    if scale:
        mean, sd = torch.tensor(scale["mean"]), torch.tensor(scale["sd"])
        tensor = (tensor - mean) / sd
    return tensor.view(*shape)


def explain(output):
    """The output in words: the likeliest classes, or the predicted values."""
    output = output[0] if isinstance(output, (list, tuple)) else output
    task = INFO.get("task", "classification")
    row = output[0]
    if task == "classification":
        probs = torch.softmax(row, dim=-1)
        labels = INFO.get("labels") or [str(i) for i in range(probs.shape[-1])]
        top = torch.topk(probs, min(3, probs.shape[-1]))
        return {"prediction": labels[int(top.indices[0])],
                "top": [{"label": labels[int(i)], "probability": round(float(p), 4)}
                        for p, i in zip(top.values, top.indices)]}
    if task == "binary":
        return {"probability": round(float(torch.sigmoid(row.flatten()[0])), 4)}
    return {"prediction": [round(float(v), 6) for v in row.flatten()]}


def predict(model, tensors):
    with torch.no_grad():
        return explain(model(*[t.unsqueeze(0) for t in tensors]))


def main():
    shapes = INFO["input_shapes"]
    parser = argparse.ArgumentParser(description=f"Run {INFO['name']}. " + INFO.get("summary", ""))
    parser.add_argument("inputs", nargs="*", help="an image file, or the input's numbers separated by spaces")
    parser.add_argument("--json", help="the input as JSON: a nested list shaped like the input")
    parser.add_argument("--invert", action="store_true", help="invert an image (dark ink on white paper)")
    parser.add_argument("--selftest", action="store_true", help="load the weights and run one blank input")
    args = parser.parse_args()
    model = load()
    if args.selftest:
        out = model(*[torch.zeros(1, *s) for s in shapes])
        out = out[0] if isinstance(out, (list, tuple)) else out
        print(f"{INFO['name']}: loaded, output shape {list(out.shape[1:])}. Self-test passed.")
        return
    if len(shapes) != 1:
        sys.exit("This network takes several inputs; use it from Python: see README.md.")
    shape = shapes[0]
    if args.json:
        tensor = torch.tensor(json.loads(args.json), dtype=torch.float32).view(*shape)
    elif len(args.inputs) == 1 and Path(args.inputs[0]).is_file() and len(shape) == 3:
        tensor = from_image(args.inputs[0], shape, args.invert)
    elif args.inputs:
        tensor = from_numbers(args.inputs, shape)
    else:
        parser.print_help()
        return
    print(json.dumps(predict(model, [tensor]), indent=1))


if __name__ == "__main__":
    main()
'''

SERVE = r'''"""A small prediction server for the packaged network: standard library plus torch.

    python serve.py --port 8100
    curl -s localhost:8100/predict -d '{"input": [[...]]}'
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from predict import INFO, load, predict

MODEL = load()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"name": INFO["name"], "inputs": INFO["input_shapes"], "task": INFO.get("task"),
                         "trained": INFO.get("trained", False)})

    def do_POST(self):
        if self.path.rstrip("/") != "/predict":
            return self._send(404, {"error": "POST to /predict"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            inputs = body["input"] if len(INFO["input_shapes"]) == 1 else body["inputs"]
            inputs = [inputs] if len(INFO["input_shapes"]) == 1 else inputs
            tensors = [torch.tensor(x, dtype=torch.float32).view(*s) for x, s in zip(inputs, INFO["input_shapes"])]
            self._send(200, predict(MODEL, tensors))
        except Exception as exc:  # noqa: BLE001 - a bad request gets a reason, not a crash
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    print(f"{INFO['name']} answering on http://{args.host}:{args.port}/predict")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
'''

INSTALL_SH = '''#!/bin/sh
# Install {name}: a private Python environment, its packages, and a self-test.
set -e
cd "$(dirname "$0")"
PY="${{PYTHON:-python3}}"
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
python predict.py --selftest
echo
echo "Installed. Try:"
echo "  .venv/bin/python predict.py --help"
echo "  .venv/bin/python serve.py --port 8100"
'''

INSTALL_BAT = '''@echo off
rem Install {name}: a private Python environment, its packages, and a self-test.
cd /d "%~dp0"
python -m venv .venv || exit /b 1
call .venv\\Scripts\\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || exit /b 1
python predict.py --selftest || exit /b 1
echo.
echo Installed. Try:  .venv\\Scripts\\python predict.py --help
'''


def _readme(info: Dict[str, Any], folder: str) -> str:
    shape = " × ".join(str(d) for d in info["input_shapes"][0]) if info["input_shapes"] else "?"
    trained = info.get("trained")
    m = info.get("metrics") or {}
    numbers = [f"- **Parameters:** {info['learnables']:,}"]
    if trained:
        numbers.append(f"- **Trained on:** {info.get('dataset') or 'unknown data'}, {info.get('epochs')} epoch"
                       f"{'' if info.get('epochs') == 1 else 's'} ({info.get('weights_from')})")
        for key, label in (("val_acc", "Validation accuracy"), ("val_loss", "Validation loss"),
                           ("train_loss", "Training loss")):
            if isinstance(m.get(key), (int, float)):
                numbers.append(f"- **{label}:** {m[key]}")
    else:
        numbers.append("- **Not trained yet:** the weights are the random ones a fresh network starts with.")
    image = len(info["input_shapes"]) == 1 and len(info["input_shapes"][0]) == 3
    usage = ("```\n.venv/bin/python predict.py picture.png\n.venv/bin/python predict.py picture.png --invert\n```\n\n"
             "`--invert` is for dark ink on white paper: MNIST-style digits are white on black."
             if image else
             f"```\n.venv/bin/python predict.py {' '.join(['0.5'] * min(int(info['input_shapes'][0][0]) if info['input_shapes'] and len(info['input_shapes'][0]) == 1 else 3, 8))}\n"
             f".venv/bin/python predict.py --json '[...]'\n```")
    scaling = ("\nNumbers are scaled the way training scaled them before they reach the network.\n"
               if info.get("input_scaling") else
               ("\nTraining scaled each input column to mean 0 and standard deviation 1, and those statistics were "
                "not saved with this network: scale your numbers the same way before passing them.\n"
                if info.get("dataset") == "csv" else ""))
    return f"""# {info['name']}

{info.get('summary', '')}

Exported from Deep Network Designer {info.get('designer_version', '')} on {info['exported']}.

{chr(10).join(numbers)}
- **Input:** {shape}
- **Output:** {info.get('task')}{(' over ' + str(len(info['labels'])) + ' classes') if info.get('labels') else ''}

## Install

```
cd {folder}
sh install.sh
```

On Windows, run `install.bat` instead. Either one makes a private Python environment in `.venv`, installs
what is in `requirements.txt`, and runs a self-test that loads the weights.

## Use it

{usage}
{scaling}
From Python:

```python
from predict import load, predict, from_image, from_numbers
model = load()
```

As a service other programs can call:

```
.venv/bin/python serve.py --port 8100
curl -s localhost:8100/predict -d '{{"input": ...}}'
```

## What is inside

| File | What it is |
|---|---|
| `model.py` | the network, as PyTorch code |
| `weights.pt` | the trained weights{'' if trained else ' (not included: untrained)'} |
| `model_info.json` | inputs, outputs, labels, and how it was trained |
| `predict.py` | run it from the command line or from Python |
| `serve.py` | a small HTTP server |
| `design.json` | the design: open it on the designer's canvas to change it |
"""


def build(design: str = "", checkpoint: str = "", version: Optional[int] = None,
          graph: Optional[Dict[str, Any]] = None, designer_version: str = "") -> Dict[str, Any]:
    """Build a package. Returns {"filename", "bytes", "info"}. Raises ValueError with a reason."""
    import torch

    blob: Dict[str, Any] = {}
    if checkpoint:
        path = T.CHECKPOINTS / Path(checkpoint).name
        if not path.exists():
            raise ValueError(f"No trained weights called {checkpoint}.")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("vocab"):
            raise ValueError("That is a language model: chat with it on the Chat page; packaging covers the others.")
        graph = blob.get("graph") or graph
    if not graph or not graph.get("nodes"):
        raise ValueError("Give a saved design or trained weights to package.")
    graph = json.loads(json.dumps(graph))
    graph.pop("_version", None)
    name = str(design or graph.get("name") or "model").split(" · ")[0]
    g = G.parse(graph)
    report = G.analyze(g)
    if not report.get("ok"):
        raise ValueError("The design does not build: " + "; ".join((report.get("errors") or ["check it on the canvas"])[:2]))
    source = codegen.to_pytorch(g, report)
    class_name = codegen.model_class_name(g)
    ids = codegen.input_order(g, report)
    shapes = [list(report["nodes"][i]["out_shape"]) for i in ids]
    nodes = g.by_id()
    outs = [i for i in report["order"] if nodes[i].type == "Output"]
    task = G.resolved_params(nodes[outs[0]]).get("task", "classification") if outs else "regression"

    record = _run_record(blob.get("job_id")) if blob else {}
    dataset = record.get("dataset") or (record.get("config") or {}).get("dataset")
    labels = blob.get("labels") or LABELS.get(dataset or "")
    out_size = (report["nodes"][outs[0]]["out_shape"] or [None])[-1] if outs else None
    if labels and out_size and len(labels) != out_size:
        labels = None
    info = {
        "name": name, "class_name": class_name, "input_shapes": shapes, "task": task, "labels": labels,
        "learnables": int(report.get("total_learnables") or 0), "trained": bool(blob),
        "epochs": blob.get("epoch"), "metrics": {k: v for k, v in (blob.get("metrics") or {}).items()
                                                  if isinstance(v, (int, float))},
        "dataset": dataset, "weights_from": Path(checkpoint).name if checkpoint else None,
        "input_scaling": blob.get("input_scaling"), "exported": time.strftime("%Y-%m-%d %H:%M"),
        "designer_version": designer_version,
        "summary": f"A {task} network with {int(report.get('total_learnables') or 0):,} parameters"
                   + (f", trained on {dataset}." if blob and dataset else "."),
    }

    # build it here first: a package that does not load is never handed over
    space: Dict[str, Any] = {"__name__": "packaged_model"}
    try:
        exec(compile(source, "model.py", "exec"), space)  # noqa: S102 - our own generated file
        model = space[class_name]()
        weights = None
        if blob:
            model.load_state_dict(blob["state_dict"], strict=True)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            weights = buffer.getvalue()
        model.eval()
        with torch.no_grad():
            model(*[torch.zeros(1, *s) for s in shapes])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"The packaged network would not load: {type(exc).__name__}: {exc}") from None
    info["self_test"] = "passed"

    folder = f"{_slug(name)}-model"
    image = len(shapes) == 1 and len(shapes[0]) == 3
    files = {
        "model.py": source,
        "model_info.json": json.dumps(info, indent=1),
        "predict.py": PREDICT,
        "serve.py": SERVE,
        "install.sh": INSTALL_SH.format(name=name),
        "install.bat": INSTALL_BAT.format(name=name),
        "requirements.txt": "torch>=2.0\n" + ("pillow>=9\n" if image else ""),
        "design.json": json.dumps(graph, indent=1),
        "README.md": _readme(info, folder),
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, text in files.items():
            entry = zipfile.ZipInfo(f"{folder}/{rel}", date_time=time.localtime()[:6])
            entry.external_attr = (0o755 if rel == "install.sh" else 0o644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(entry, text)
        if weights is not None:
            entry = zipfile.ZipInfo(f"{folder}/weights.pt", date_time=time.localtime()[:6])
            entry.external_attr = 0o644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(entry, weights)
    suffix = f"-e{info['epochs']}" if blob else "-untrained"
    return {"filename": f"{folder}{suffix}.zip", "bytes": out.getvalue(), "info": info}
