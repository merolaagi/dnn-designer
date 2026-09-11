"""Problem-grouped offline training. No counterfactual labels are fabricated."""

import copy
import hashlib
import json
import random

import numpy as np
import torch

from mare.learned_policy import TinyMLPRegressor
from .features import NODE_DIM, CONTEXT_DIM, EDGE_TYPES, VERSION
from .network import GraphPlanner, collate, save_checkpoint


def validate(records):
    if not records:
        raise ValueError("Empty dataset")
    for r in records:
        n, k = len(r["nodes"]), len(r["candidates"])
        if r["version"] != VERSION or not 0 < n <= 2048 or not k:
            raise ValueError("Invalid schema/graph size")
        if len(r["nodes"][0]) != NODE_DIM or any(len(row) != NODE_DIM for row in r["nodes"]):
            raise ValueError("Invalid node dimensions")
        if len(r["context"]) != CONTEXT_DIM:
            raise ValueError("Invalid context")
        if any(not 0 <= i < n for i in r["candidates"]):
            raise ValueError("Invalid candidate index")
        if len(set(r["candidates"])) != k:
            raise ValueError("Duplicate candidates")
        if len(r["edges"]) > 100000:
            raise ValueError("Too many edges")
        if any(not (0 <= a < n and 0 <= b < n and 0 <= t < EDGE_TYPES) for a, b, t in r["edges"]):
            raise ValueError("Invalid edge")
        if len(r.get("targets", [])) != k:
            raise ValueError("Missing/misaligned targets")
        if any(len(row) != 3 or any(v is not None and not 0 <= v <= 1 for v in row) for row in r["targets"]):
            raise ValueError("Invalid labels")
        if not np.isfinite(np.asarray(r["nodes"])).all() or not np.isfinite(r["context"]).all():
            raise ValueError("Nonfinite features")
        if not r.get("problem_id") or r.get("source") not in {"real", "mock", "synthetic"}:
            raise ValueError("Missing provenance")


def split_problems(records, seed):
    groups = sorted({r["problem_id"] for r in records})
    if len(groups) < 10:
        raise ValueError("Need at least 10 distinct problems for train/validation/test split")
    random.Random(seed).shuffle(groups)
    a, b = int(len(groups) * 0.6), int(len(groups) * 0.8)
    partitions = [set(groups[:a]), set(groups[a:b]), set(groups[b:])]
    return [[r for r in records if r["problem_id"] in keys] for keys in partitions]


def loss_for(model, records):
    batch = collate(records)
    if not batch["mask"].any():
        raise ValueError("No observed labels in batch")
    predictions = model(batch)
    squared = (predictions - batch["labels"]).square()
    # Normalize each task separately so missing cost labels do not alter its weight.
    return (
        sum(
            (squared[:, i] * batch["mask"][:, i]).sum() / batch["mask"][:, i].sum().clamp_min(1)
            for i in range(3)
        )
        / 3
    )


def train(records, *, seed=7, epochs=50, width=128, depth=3):
    validate(records)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_set, val, test = split_problems(records, seed)
    train_set = [r for r in train_set if any(v is not None for row in r["targets"] for v in row)]
    if not train_set:
        raise ValueError("No observed training labels")
    model = GraphPlanner(width, depth)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    rng = random.Random(seed)
    best = float("inf")
    weights = None
    history = []
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        order = train_set.copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 8):
            opt.zero_grad()
            loss = loss_for(model, order[start : start + 8])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            value = float(loss_for(model, val))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=value))
        if value < best:
            best = value
            weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        if epoch + 1 - best_epoch >= 12:
            break
    if weights is None:
        raise ValueError("Training did not produce finite validation loss")
    model.load_state_dict(weights)
    model.eval()
    source = (
        next(iter({r["source"] for r in records})) if len({r["source"] for r in records}) == 1 else "mixed"
    )
    metadata = dict(
        source=source,
        seed=seed,
        trained_examples=sum(any(v is not None for v in row) for r in train_set for row in r["targets"]),
        best_epoch=best_epoch,
        validation_loss=best,
        history=history,
        split={
            name: sorted({r["problem_id"] for r in rs})
            for name, rs in zip(("train", "validation", "test"), (train_set, val, test))
        },
        dataset_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        label_version=1,
        text_encoder="signed lexical SHA256 hashing; no semantic pretraining",
    )
    return model, metadata, (train_set, val, test)


def evaluate(model, records):
    with torch.inference_mode():
        b = collate(records)
        pred = model(b)
        return {
            name: float(((pred[:, i] - b["labels"][:, i]) ** 2)[b["mask"][:, i]].mean())
            if b["mask"][:, i].any()
            else None
            for i, name in enumerate(("utility_mse", "cost_mse", "resolution_mse"))
        }


def fit_tiny(records, seed):
    x, y = [], []
    for r in records:
        for features, targets in zip(r["baseline"], r["targets"]):
            if targets[0] is not None:
                x.append(features)
                y.append(targets[0])
    model = TinyMLPRegressor(11, 12, seed)
    model.fit(np.asarray(x), np.asarray(y), epochs=1500)
    return model


def write_training(records, path, **kwargs):
    model, metadata, parts = train(records, **kwargs)
    metadata["test_metrics"] = evaluate(model, parts[2])
    save_checkpoint(path, model, metadata)
    return metadata
