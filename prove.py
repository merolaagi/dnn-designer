#!/usr/bin/env python3
"""Prove theorems with a trained policy, and have the kernel check every one.

    python prove.py                 train briefly, then evaluate
    python prove.py --epochs 10     train longer

The number this prints is proofs the kernel accepted on theorems that do not
appear in training. The model proposes; microlean.check decides. A confident
wrong suggestion costs a search node and nothing else, which is the whole reason
this arrangement is worth building.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    import torch

    import blockloader
    import microlean as ml
    import train as T
    import workbook

    blockloader.load_all()
    torch.manual_seed(0)

    book = json.loads((HERE / "examples" / "MicroLean.json").read_text())
    analysis = workbook.analyze(book)
    if not analysis["ok"]:
        raise SystemExit("the design does not resolve")
    model = T.build_model(workbook.to_pytorch(book, analysis), "Model")
    print(f"policy network: {analysis['total_learnables']:,} parameters")

    lessons = ml.corpus(args.train, steps=3, seed=0)
    used = ml.statements(lessons)
    rows, labels = [], []
    for theorem, proof in lessons:
        term = theorem.lhs
        for rule_index, path in proof:
            key = (rule_index, tuple(path))
            if key not in ml.TACTIC_ID:
                break
            rows.append(ml.encode_state(term, theorem.rhs))
            labels.append(ml.TACTIC_ID[key])
            term = ml.apply_rule(term, ml.AXIOMS[rule_index], path)

    x = torch.tensor(rows, dtype=torch.long)
    y = torch.tensor(labels)
    cut = int(len(x) * 0.9)
    print(f"{len(x):,} proof steps, every one verified before use")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x[:cut], y[:cut]), batch_size=64, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            acc = (model(x[cut:]).argmax(-1) == y[cut:]).float().mean().item()
        print(f"  epoch {epoch + 1:2d}   loss {total / len(loader):.3f}   "
              f"tactic accuracy {acc:.1%}")

    # theorems the training walk never produced
    easy = ml.corpus(120, steps=3, seed=4242, exclude=used)
    hard = ml.corpus(120, steps=5, seed=777, exclude=used)

    def policy(current, target):
        with torch.no_grad():
            return model(torch.tensor([ml.encode_state(current, target)],
                                      dtype=torch.long))[0].tolist()

    rng = random.Random(7)

    def coin(current, target):
        return [rng.random() for _ in range(ml.N_TACTICS)]

    print()
    for name, corpus in (("1 to 3 steps", easy), ("up to 5 steps", hard)):
        overlap = len(ml.statements(corpus) & used)
        print(f"  held out, {name}  ({len(corpus)} theorems, "
              f"{overlap} shared with training)")
        for label, score in (("random", coin), ("trained", policy)):
            proved = 0
            for theorem, _ in corpus:
                found = ml.search(theorem, score, width=args.width, depth=args.depth)
                if found and ml.check(theorem, found):
                    proved += 1
            print(f"     {label:8s} {proved:3d}/{len(corpus)}  "
                  f"({proved / len(corpus):.0%}) verified by the kernel")


if __name__ == "__main__":
    main()
