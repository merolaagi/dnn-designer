"""Background training jobs.

The model that trains is the model you exported: this compiles the generated
PyTorch source and instantiates it, so anything you see in the Code panel is
literally what runs. Metrics are pushed onto a queue that the server drains as
a server-sent event stream.

Datasets come in three shapes, all of which yield ``(list_of_inputs, target)``
so that a graph with two Input layers is no more special than a graph with one:

  synthetic  random tensors matched to each Input, for sanity checks
  built-in   MNIST / Fashion-MNIST / CIFAR-10 through torchvision
  csv        a table you uploaded, with columns mapped onto the Input layers
"""

from __future__ import annotations

import math
import queue
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

JOBS: Dict[str, "Job"] = {}
_LOCK = threading.Lock()

UPLOADS = Path(__file__).resolve().parent / "uploads"
UPLOADS.mkdir(exist_ok=True)

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

BUILTIN_DATASETS = {
    "synthetic": {"label": "Synthetic noise (sanity check)", "shape": None, "classes": 10},
    "mnist": {"label": "MNIST digits", "shape": [1, 28, 28], "classes": 10},
    "fashion_mnist": {"label": "Fashion-MNIST", "shape": [1, 28, 28], "classes": 10},
    "cifar10": {"label": "CIFAR-10", "shape": [3, 32, 32], "classes": 10},
    "folder": {"label": "Folder of images", "shape": None, "classes": None},
    "text": {"label": "Text file (character language model)", "shape": None,
             "classes": None},
}

AUGMENTATIONS = [
    {"id": "flip", "label": "Horizontal flip"},
    {"id": "vflip", "label": "Vertical flip"},
    {"id": "rotate", "label": "Rotate up to 15 degrees"},
    {"id": "crop", "label": "Random resized crop"},
    {"id": "jitter", "label": "Brightness and contrast jitter"},
    {"id": "erase", "label": "Random erasing"},
]


def prod(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class DataError(ValueError):
    """A dataset cannot be built the way it was configured."""


@dataclass
class Job:
    id: str
    status: str = "starting"           # starting | running | done | error | stopped
    epoch: int = 0
    epochs: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    device: str = ""
    learnables: int = 0
    events: "queue.Queue" = field(default_factory=queue.Queue)
    stop: threading.Event = field(default_factory=threading.Event)

    def emit(self, kind: str, **payload) -> None:
        self.events.put({"kind": kind, "t": time.time(), **payload})

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id, "status": self.status, "epoch": self.epoch,
            "epochs": self.epochs, "history": self.history, "error": self.error,
            "device": self.device, "learnables": self.learnables,
        }


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(source: str, class_name: Optional[str] = None):
    """Compile the generated file and return an instance of its model class.

    Helper classes such as the positional encoder live in the same file, so the
    class name from codegen decides which one is the model. Without a name, fall
    back to the last module that can be built with no arguments.
    """
    import torch.nn as nn

    namespace: Dict[str, Any] = {"__name__": "generated_model"}
    exec(compile(source, "<designer-model>", "exec"), namespace)  # noqa: S102

    if class_name and class_name in namespace:
        return namespace[class_name]()

    candidates = [
        v for v in namespace.values()
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
        and v.__module__ == "generated_model"
    ]
    for value in reversed(candidates):
        try:
            return value()
        except TypeError:
            continue
    raise RuntimeError("The generated code did not define a model class.")


# --------------------------------------------------------------------------
# CSV inspection, shared with the server so the form can be built before a run
# --------------------------------------------------------------------------

def csv_path(name: str) -> Path:
    safe = Path(name).name
    path = UPLOADS / safe
    if not path.exists():
        raise DataError(f"No uploaded table named {safe}.")
    return path


def inspect_csv(name: str, sample_rows: int = 400) -> Dict[str, Any]:
    """Column names, types and cardinality — enough to build the mapping form."""
    import pandas as pd

    path = csv_path(name)
    head = pd.read_csv(path, nrows=sample_rows)
    total = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
    columns = []
    for col in head.columns:
        series = head[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        columns.append({
            "name": str(col),
            "numeric": bool(numeric),
            "unique": int(series.nunique(dropna=True)),
            "example": "" if series.dropna().empty else str(series.dropna().iloc[0])[:24],
        })
    return {"name": path.name, "rows": max(total, 0), "columns": columns}


def _encode_frame(frame, columns: List[str]):
    """Numeric matrix from arbitrary columns. Text columns become integer codes."""
    import numpy as np
    import pandas as pd

    encoded, notes = [], []
    for col in columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
        else:
            values = pd.Series(pd.factorize(series)[0], index=series.index).astype(float)
            notes.append(col)
        values = values.astype("float32")
        if values.isna().any():
            values = values.fillna(float(values.mean()) if values.notna().any() else 0.0)
        encoded.append(values.to_numpy(dtype="float32"))
    matrix = np.stack(encoded, axis=1) if encoded else np.zeros((len(frame), 0), "float32")
    return matrix, notes


def assign_columns(feature_cols: List[str], in_shapes: List[List[int]],
                   override: Optional[Dict[str, List[str]]] = None,
                   input_ids: Optional[List[str]] = None) -> List[List[str]]:
    """Give each Input layer the columns it needs, in order, unless told otherwise."""
    override = override or {}
    input_ids = input_ids or [str(i) for i in range(len(in_shapes))]
    out: List[List[str]] = []
    cursor = 0
    for nid, shape in zip(input_ids, in_shapes):
        want = prod(shape)
        chosen = override.get(nid)
        if chosen:
            missing = [c for c in chosen if c not in feature_cols]
            if missing:
                raise DataError(f"Columns not in the table: {', '.join(missing)}")
            if len(chosen) != want:
                raise DataError(
                    f"Input {nid} has shape {shape} and needs {want} columns, got {len(chosen)}."
                )
            out.append(list(chosen))
            continue
        if cursor + want > len(feature_cols):
            raise DataError(
                f"The table has {len(feature_cols)} feature columns but the Input layers "
                f"need {sum(prod(s) for s in in_shapes)}. Map the columns by hand, or "
                f"change the Input shapes."
            )
        out.append(feature_cols[cursor:cursor + want])
        cursor += want
    return out


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def _tensor_dataset(xs, y):
    import torch
    from torch.utils.data import Dataset

    class MultiInput(Dataset):
        def __len__(self):
            return y.shape[0]

        def __getitem__(self, i):
            return [x[i] for x in xs], y[i]

    return MultiInput()


def _wrap_builtin(base, n_inputs: int):
    from torch.utils.data import Dataset

    class Fanout(Dataset):
        """Feeds the same sample to every Input, which is what siamese graphs want."""

        def __len__(self):
            return len(base)

        def __getitem__(self, i):
            x, y = base[i]
            return [x] * n_inputs, y

    return Fanout()


def _csv_loaders(cfg, in_shapes, in_ids, task, job: Optional[Job]):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    name = cfg.get("csv_file")
    if not name:
        raise DataError("Choose an uploaded table first.")
    frame = pd.read_csv(csv_path(name))
    target = cfg.get("target_column")
    if not target:
        raise DataError("Choose which column is the target.")
    if target not in frame.columns:
        raise DataError(f"The table has no column named {target}.")

    drop = set(cfg.get("ignore_columns") or []) | {target}
    feature_cols = [c for c in frame.columns if c not in drop]
    mapping = assign_columns(feature_cols, in_shapes, cfg.get("column_map"), in_ids)

    blocks, notes = [], []
    for cols in mapping:
        matrix, encoded = _encode_frame(frame, cols)
        blocks.append(matrix)
        notes.extend(encoded)

    classes = None
    y_series = frame[target]
    if task == "classification":
        if pd.api.types.is_numeric_dtype(y_series) and y_series.dropna().nunique() > 100:
            raise DataError(
                f"{target} looks continuous. Set the Output layer to regression, "
                f"or pick a different target."
            )
        codes, uniques = pd.factorize(y_series)
        classes = [str(u) for u in uniques]
        y = torch.as_tensor(codes.astype("int64"))
    else:
        y = torch.as_tensor(
            pd.to_numeric(y_series, errors="coerce").fillna(0).to_numpy(dtype="float32")
        ).view(-1, 1)

    n = len(frame)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    order = rng.permutation(n)
    split = int(n * (1 - float(cfg.get("val_split", 0.2))))
    if split < 1 or split >= n:
        raise DataError("The validation split leaves one side empty. Try 0.1 to 0.4.")
    tr_idx, va_idx = order[:split], order[split:]

    xs_tr, xs_va = [], []
    for matrix, shape in zip(blocks, in_shapes):
        train_part = matrix[tr_idx]
        if cfg.get("normalize", True):
            mean = train_part.mean(axis=0, keepdims=True)
            std = train_part.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            matrix = (matrix - mean) / std
        tensor = torch.as_tensor(matrix, dtype=torch.float32)
        tensor = tensor.reshape(n, *[int(d) for d in shape])
        xs_tr.append(tensor[tr_idx])
        xs_va.append(tensor[va_idx])

    if job:
        job.emit(
            "dataset",
            rows=n, train=len(tr_idx), val=len(va_idx),
            features=sum(len(c) for c in mapping),
            classes=len(classes) if classes else None,
            class_names=(classes[:12] if classes else None),
            encoded=sorted(set(notes))[:8],
            mapping={i: c for i, c in zip(in_ids, mapping)},
        )

    batch = int(cfg.get("batch_size", 64))
    return (
        DataLoader(_tensor_dataset(xs_tr, y[tr_idx]), batch_size=batch, shuffle=True),
        DataLoader(_tensor_dataset(xs_va, y[va_idx]), batch_size=batch),
        {"classes": classes},
    )


def inspect_folder(path: str, max_classes: int = 40) -> Dict[str, Any]:
    """Class names and image counts for a folder of class-named subfolders."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise DataError(f"{root} is not a folder.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    classes = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for f in child.iterdir() if f.suffix.lower() in exts)
        if n:
            classes.append({"name": child.name, "images": n})
    if not classes:
        raise DataError(
            f"No class subfolders with images under {root}. The layout should be "
            f"one folder per class, each holding that class's images."
        )
    return {
        "path": str(root),
        "classes": classes[:max_classes],
        "class_count": len(classes),
        "images": sum(c["images"] for c in classes),
    }


def _build_transforms(cfg, shape):
    from torchvision import transforms

    channels, height, width = [int(d) for d in shape]
    picked = set(cfg.get("augment") or [])
    train_steps = []

    if "crop" in picked:
        train_steps.append(transforms.RandomResizedCrop((height, width), scale=(0.6, 1.0)))
    else:
        train_steps.append(transforms.Resize((height, width)))
    if "flip" in picked:
        train_steps.append(transforms.RandomHorizontalFlip())
    if "vflip" in picked:
        train_steps.append(transforms.RandomVerticalFlip())
    if "rotate" in picked:
        train_steps.append(transforms.RandomRotation(15))
    if "jitter" in picked:
        train_steps.append(transforms.ColorJitter(brightness=0.25, contrast=0.25))
    if channels == 1:
        train_steps.append(transforms.Grayscale(num_output_channels=1))
    train_steps.append(transforms.ToTensor())
    if channels == 3:
        train_steps.append(
            transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.size(0) == 1 else t))
    if "erase" in picked:
        train_steps.append(transforms.RandomErasing(p=0.25))

    eval_steps = [transforms.Resize((height, width))]
    if channels == 1:
        eval_steps.append(transforms.Grayscale(num_output_channels=1))
    eval_steps.append(transforms.ToTensor())
    if channels == 3:
        eval_steps.append(
            transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.size(0) == 1 else t))

    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def _folder_loaders(cfg, in_shapes, job: Optional[Job]):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    path = cfg.get("folder")
    if not path:
        raise DataError("Point the folder field at a directory of class subfolders.")
    for shape in in_shapes:
        if len(shape) != 3:
            raise DataError(
                f"A folder of images needs an Input shaped [C, H, W]; one is {shape}."
            )

    info = inspect_folder(path)
    train_tf, eval_tf = _build_transforms(cfg, in_shapes[0])
    train_full = datasets.ImageFolder(info["path"], transform=train_tf)
    eval_full = datasets.ImageFolder(info["path"], transform=eval_tf)

    n = len(train_full)
    split = int(n * (1 - float(cfg.get("val_split", 0.2))))
    if split < 1 or split >= n:
        raise DataError("The validation split leaves one side empty. Try 0.1 to 0.4.")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(
        int(cfg.get("seed", 0)))).tolist()
    limit = int(cfg.get("train_samples", 0))
    tr_idx = order[:split][:limit] if limit else order[:split]
    va_idx = order[split:][:max(limit // 5, 64)] if limit else order[split:]

    if job:
        job.emit("dataset", rows=n, train=len(tr_idx), val=len(va_idx),
                 classes=info["class_count"],
                 class_names=[c["name"] for c in info["classes"][:12]],
                 augment=sorted(cfg.get("augment") or []),
                 encoded=[])

    batch = int(cfg.get("batch_size", 64))
    workers = int(cfg.get("workers", 2))
    n_in = len(in_shapes)
    return (
        DataLoader(_wrap_builtin(Subset(train_full, tr_idx), n_in),
                   batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(_wrap_builtin(Subset(eval_full, va_idx), n_in),
                   batch_size=batch, num_workers=workers),
        {"classes": [c["name"] for c in info["classes"]]},
    )


def inspect_text(name: str) -> Dict[str, Any]:
    """Corpus size and character vocabulary, written out for the generator."""
    import json

    path = UPLOADS / Path(name).name
    if not path.exists():
        raise DataError(f"No uploaded text file named {path.name}.")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) < 512:
        raise DataError(
            f"{path.name} holds {len(text)} characters. A character model needs "
            f"a few thousand at minimum to learn anything."
        )
    vocab = sorted(set(text))
    vocab_path = path.with_suffix(path.suffix + ".vocab.json")
    vocab_path.write_text(json.dumps(vocab))
    return {
        "file": path.name,
        "characters": len(text),
        "vocab_size": len(vocab),
        "vocab_path": str(vocab_path),
        "preview": text[:220],
    }


def _text_loaders(cfg, in_shapes, job: Optional[Job]):
    """Random crops of the corpus, each paired with itself shifted one step."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    name = cfg.get("text_file")
    if not name:
        raise DataError("Choose an uploaded text file first.")
    info = inspect_text(name)
    path = UPLOADS / Path(name).name
    text = path.read_text(encoding="utf-8", errors="replace")

    import json
    vocab = json.loads(Path(info["vocab_path"]).read_text())
    stoi = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    shape = in_shapes[0]
    if len(shape) != 1:
        raise DataError(
            f"A character model wants an Input shaped [L] holding token ids; "
            f"this one is {shape}. Set the shape to the context length and the "
            f"dtype to long."
        )
    block = int(shape[0])
    if len(data) <= block + 1:
        raise DataError(
            f"The corpus is {len(data)} characters, shorter than the context "
            f"length of {block}."
        )

    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    class Crops(Dataset):
        def __init__(self, ids, count):
            self.ids, self.count = ids, count

        def __len__(self):
            return self.count

        def __getitem__(self, _):
            i = int(torch.randint(0, len(self.ids) - block - 1, (1,)))
            chunk = self.ids[i:i + block + 1]
            return [chunk[:-1]], chunk[1:]

    per_epoch = int(cfg.get("train_samples") or 2000)
    if job:
        job.emit("dataset", rows=len(data), train=per_epoch, val=max(per_epoch // 8, 64),
                 classes=len(vocab), vocab_size=len(vocab),
                 class_names=None, encoded=[],
                 vocab_path=info["vocab_path"], block_size=block)

    batch = int(cfg.get("batch_size", 32))
    return (
        DataLoader(Crops(train_data, per_epoch), batch_size=batch),
        DataLoader(Crops(val_data, max(per_epoch // 8, 64)), batch_size=batch),
        {"classes": None, "vocab": vocab, "vocab_size": len(vocab), "block": block},
    )


def _make_loaders(cfg, in_shapes, in_ids, out_shape, task, job: Optional[Job] = None):
    import torch
    from torch.utils.data import DataLoader

    name = cfg.get("dataset", "synthetic")
    batch = int(cfg.get("batch_size", 64))
    limit = int(cfg.get("train_samples", 0))

    if name == "csv":
        return _csv_loaders(cfg, in_shapes, in_ids, task, job)

    if name == "folder":
        return _folder_loaders(cfg, in_shapes, job)

    if name == "text":
        return _text_loaders(cfg, in_shapes, job)

    if name == "synthetic":
        n = limit or 2048
        classes = out_shape[-1] if out_shape else 10
        xs = [torch.randn(n, *[int(d) for d in s]) for s in in_shapes]
        if task == "classification":
            y = torch.randint(0, max(2, classes), (n,))
        else:
            y = torch.randn(n, max(1, classes))
        split = int(n * 0.8)
        tr = _tensor_dataset([x[:split] for x in xs], y[:split])
        va = _tensor_dataset([x[split:] for x in xs], y[split:])
        return (DataLoader(tr, batch_size=batch, shuffle=True),
                DataLoader(va, batch_size=batch), {"classes": None})

    for shape in in_shapes:
        if len(shape) != 3:
            raise DataError(
                f"{name} produces images shaped [C, H, W]. One of your Input layers is "
                f"{shape} — switch to the synthetic dataset or upload a table instead."
            )
    if len({tuple(s) for s in in_shapes}) > 1:
        raise DataError(
            "With more than one Input, a built-in image set can only feed identical "
            "shapes. Give every Input the same shape, or use a CSV table."
        )

    from torchvision import datasets, transforms

    channels, height, width = [int(d) for d in in_shapes[0]]
    steps = [transforms.Resize((height, width))]
    if channels == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.append(transforms.ToTensor())
    if channels == 3:
        steps.append(transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.size(0) == 1 else t))
    tf = transforms.Compose(steps)

    root = cfg.get("data_root", "./data")
    factory = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST,
               "cifar10": datasets.CIFAR10}[name]
    train_ds = factory(root, train=True, download=True, transform=tf)
    val_ds = factory(root, train=False, download=True, transform=tf)

    if limit:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(max(limit // 5, 256), len(val_ds))))

    n_in = len(in_shapes)
    train_ds, val_ds = _wrap_builtin(train_ds, n_in), _wrap_builtin(val_ds, n_in)
    workers = int(cfg.get("workers", 2))
    if job:
        job.emit("dataset", rows=len(train_ds) + len(val_ds),
                 train=len(train_ds), val=len(val_ds), classes=10)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
            DataLoader(val_ds, batch_size=batch, num_workers=workers), {"classes": None})


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def generate_text(model, vocab: List[str], block: int, prompt: str,
                  max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k: int = 40, device: str = "cpu",
                  stop: str = "") -> Dict[str, Any]:
    """Sample a continuation. Returns the prompt and the continuation separately.

    A `stop` string ends generation as soon as it appears in the new text, which
    is how a reply stops at the end of its turn instead of running on into an
    invented next question.
    """
    import torch

    stoi = {c: i for i, c in enumerate(vocab)}
    unknown = sorted({c for c in prompt if c not in stoi})
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    start = x.size(1)

    was_training = model.training
    model.eval()
    truncated = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(x[:, -block:])
            logits = (out[0] if isinstance(out, (tuple, list)) else out)[:, -1, :]
            logits = logits / max(float(temperature), 1e-6)
            if top_k:
                k = min(int(top_k), logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            x = torch.cat([x, nxt], dim=1)
            if stop:
                tail = "".join(vocab[int(i)] for i in x[0, start:].tolist())
                if stop in tail:
                    truncated = True
                    break
    if was_training:
        model.train()

    whole = "".join(vocab[int(i)] for i in x[0].tolist())
    continuation = whole[len(prompt) if prompt and not unknown else start:]
    if stop and stop in continuation:
        continuation = continuation.split(stop)[0]
    return {"prompt": prompt, "continuation": continuation,
            "text": prompt + continuation, "stopped": truncated,
            "unknown_characters": unknown}


def _sample_text(model, meta, device, prompt: str, count: int) -> str:
    """Draw a short continuation so progress is legible as text, not just loss."""
    return generate_text(model, meta["vocab"], meta["block"], prompt,
                         max_new_tokens=count, device=device)["text"]


def save_checkpoint(job: Job, model, graph, epoch: int, metrics: Dict[str, Any],
                    tag: str, name_hint: str = "model",
                    extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write weights plus enough context to rebuild and identify them later."""
    import torch

    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", name_hint).strip("-")[:40] or "model"
    path = CHECKPOINTS / f"{safe}_{job.id}_{tag}.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "graph": graph,
        "epoch": epoch,
        "metrics": metrics,
        "job_id": job.id,
        "learnables": job.learnables,
        "saved_at": time.time(),
        "format": 1,
        **(extra or {}),
    }, path)
    return path


def _vocab_extra(meta: Dict[str, Any]) -> Dict[str, Any]:
    if meta.get("vocab"):
        return {"vocab": meta["vocab"], "block": meta.get("block")}
    return {}


def list_checkpoints() -> List[Dict[str, Any]]:
    import torch

    out = []
    for path in sorted(CHECKPOINTS.glob("*.pt"), key=lambda p: -p.stat().st_mtime):
        entry = {"file": path.name, "bytes": path.stat().st_size,
                 "saved_at": time.strftime("%Y-%m-%d %H:%M",
                                           time.localtime(path.stat().st_mtime))}
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            entry.update({
                "epoch": blob.get("epoch"),
                "metrics": blob.get("metrics"),
                "learnables": blob.get("learnables"),
                "network": (blob.get("graph") or {}).get("name"),
                "chat": bool(blob.get("vocab")),
            })
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"could not read: {exc}"
        out.append(entry)
    return out


def load_into(model, file: str, strict: bool = False) -> Dict[str, Any]:
    """Copy weights from a checkpoint into a freshly built model.

    Non-strict is the default on purpose: replacing the final layer for a new
    class count is the whole point of transfer learning, and that mismatch
    should be reported rather than raised.
    """
    import torch

    path = CHECKPOINTS / Path(file).name
    if not path.exists():
        raise DataError(f"No checkpoint named {path.name}.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    saved = blob.get("state_dict", blob)
    own = model.state_dict()

    loaded, skipped = [], []
    for key, value in saved.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    missing = [k for k in own if k not in saved]
    if strict and (skipped or missing):
        raise DataError(
            f"{path.name} does not match this graph: {len(skipped)} tensors differ."
        )
    model.load_state_dict(own)
    return {"file": path.name, "loaded": len(loaded), "skipped": skipped[:8],
            "skipped_count": len(skipped), "missing_count": len(missing)}


def _criterion(task: str):
    import torch.nn as nn

    if task == "regression":
        return nn.MSELoss()
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    # language modelling also uses cross entropy, but over every position at
    # once — the reshaping happens where the loss is applied.
    return nn.CrossEntropyLoss()


def _shape_target(y, task, batch_size):
    if task in ("regression", "binary"):
        return y.float().view(batch_size, -1)
    return y


def _run(job: Job, source: str, cfg: Dict[str, Any], in_shapes, in_ids,
         out_shape, tasks: List[str], class_name: Optional[str] = None) -> None:
    try:
        import torch

        device = pick_device(cfg.get("device", "auto"))
        job.device = device
        model = build_model(source, class_name).to(device)
        job.learnables = sum(p.numel() for p in model.parameters() if p.requires_grad)
        job.emit("ready", device=device, learnables=job.learnables,
                 inputs=len(in_shapes), outputs=len(tasks))

        if cfg.get("init_from"):
            report = load_into(model, cfg["init_from"],
                               strict=bool(cfg.get("init_strict")))
            job.emit("weights", **report)

        train_loader, val_loader, meta = _make_loaders(
            cfg, in_shapes, in_ids, out_shape, tasks[0], job)

        if meta.get("vocab_size") and out_shape:
            if out_shape[-1] != meta["vocab_size"]:
                job.emit("warning", message=(
                    f"The corpus has {meta['vocab_size']} distinct characters but "
                    f"the final Linear outputs {out_shape[-1]}. Set both it and "
                    f"the Embedding vocab to {meta['vocab_size']}."))

        if meta.get("classes") and tasks[0] == "classification" and out_shape:
            wanted, have = len(meta["classes"]), out_shape[-1]
            if wanted != have:
                job.emit(
                    "warning",
                    message=(f"The target has {wanted} classes but the last layer outputs "
                             f"{have} units. Set it to {wanted} or training will fight you."),
                )

        criteria = [_criterion(t) for t in tasks]
        aux = float(cfg.get("aux_weight", 0.3))
        weights = [1.0] + [aux] * (len(tasks) - 1)

        lr = float(cfg.get("lr", 1e-3))
        params = list(model.parameters())
        optimizer = {
            "adam": lambda: torch.optim.Adam(params, lr=lr),
            "adamw": lambda: torch.optim.AdamW(params, lr=lr),
            "sgd": lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
            "rmsprop": lambda: torch.optim.RMSprop(params, lr=lr),
        }[cfg.get("optimizer", "adamw")]()

        job.epochs = int(cfg.get("epochs", 5))
        job.status = "running"
        primary_is_class = tasks[0] in ("classification", "language_modeling")
        is_lm = tasks[0] == "language_modeling"
        graph_blob = cfg.get("graph") or {}
        name_hint = graph_blob.get("name") or "model"
        save_every = bool(cfg.get("save_checkpoints", True))
        best = float("inf")
        patience = int(cfg.get("early_stop", 0))
        stale = 0

        def forward_loss(xs, yb):
            out = model(*xs)
            outs = list(out) if isinstance(out, (tuple, list)) else [out]
            total = None
            for o, crit, task, w in zip(outs, criteria, tasks, weights):
                if task == "language_modeling":
                    # one prediction per position: flatten the batch and the
                    # sequence together so cross entropy sees a flat problem
                    term = crit(o.reshape(-1, o.size(-1)), yb.reshape(-1))
                else:
                    term = crit(o, _shape_target(yb, task, o.size(0)))
                total = term * w if total is None else total + term * w
            return outs[0], total

        for epoch in range(1, job.epochs + 1):
            if job.stop.is_set():
                break
            job.epoch = epoch
            model.train()
            run_loss, seen, correct = 0.0, 0, 0

            for step, (xs, yb) in enumerate(train_loader):
                if job.stop.is_set():
                    break
                xs = [x.to(device) for x in xs]
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                primary, loss = forward_loss(xs, yb)
                loss.backward()
                if cfg.get("clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
                optimizer.step()

                bs = xs[0].size(0)
                run_loss += loss.item() * bs
                seen += bs
                if is_lm:
                    correct += (primary.argmax(-1) == yb).float().mean().item() * bs
                elif primary_is_class:
                    correct += (primary.argmax(1) == yb).sum().item()
                if step % 10 == 0:
                    job.emit("step", epoch=epoch, step=step,
                             loss=round(run_loss / max(seen, 1), 5))

            model.eval()
            v_loss, v_seen, v_correct = 0.0, 0, 0
            with torch.no_grad():
                for xs, yb in val_loader:
                    xs = [x.to(device) for x in xs]
                    yb = yb.to(device)
                    primary, loss = forward_loss(xs, yb)
                    bs = xs[0].size(0)
                    v_loss += loss.item() * bs
                    v_seen += bs
                    if is_lm:
                        v_correct += (primary.argmax(-1) == yb).float().mean().item() * bs
                    elif primary_is_class:
                        v_correct += (primary.argmax(1) == yb).sum().item()

            row = {
                "epoch": epoch,
                "train_loss": round(run_loss / max(seen, 1), 5),
                "val_loss": round(v_loss / max(v_seen, 1), 5),
            }
            if primary_is_class:
                row["train_acc"] = round(correct / max(seen, 1), 4)
                row["val_acc"] = round(v_correct / max(v_seen, 1), 4)
            if is_lm:
                row["perplexity"] = round(float(math.exp(min(row["val_loss"], 20))), 2)
            job.history.append(row)
            job.emit("epoch", **row)

            if is_lm and meta.get("vocab"):
                sample = _sample_text(model, meta, device,
                                      cfg.get("sample_prompt", "\n"),
                                      int(cfg.get("sample_tokens", 180)))
                job.emit("sample", epoch=epoch, text=sample)

            if save_every and row["val_loss"] < best:
                best = row["val_loss"]
                path = save_checkpoint(job, model, graph_blob, epoch, row,
                                       "best", name_hint, _vocab_extra(meta))
                job.emit("checkpoint", file=path.name, tag="best", epoch=epoch,
                         val_loss=row["val_loss"])
                stale = 0
            else:
                stale += 1
                if patience and stale >= patience:
                    job.emit("early_stop", epoch=epoch, patience=patience)
                    break

        if save_every:
            last = save_checkpoint(job, model, graph_blob, job.epoch,
                                   job.history[-1] if job.history else {},
                                   "last", name_hint, _vocab_extra(meta))
            job.emit("checkpoint", file=last.name, tag="last", epoch=job.epoch)

        job.status = "stopped" if job.stop.is_set() else "done"
        job.emit("finished", status=job.status)

    except DataError as exc:
        job.status = "error"
        job.error = str(exc)
        job.emit("error", message=job.error)
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit("error", message=job.error, detail=traceback.format_exc()[-1600:])


def start(source: str, cfg: Dict[str, Any], in_shapes, in_ids, out_shape,
          tasks: List[str], class_name: Optional[str] = None) -> Job:
    job = Job(id=uuid.uuid4().hex[:12])
    with _LOCK:
        JOBS[job.id] = job
        for old_id, old in list(JOBS.items()):
            if old.status in ("done", "error", "stopped") and len(JOBS) > 12:
                JOBS.pop(old_id, None)
    threading.Thread(
        target=_run,
        args=(job, source, cfg, in_shapes, in_ids, out_shape, tasks, class_name),
        daemon=True,
    ).start()
    return job
