"""Sparse relational message passing; block-diagonal batching of variable graphs."""

import torch
from torch import nn
from .features import NODE_DIM, CONTEXT_DIM, EDGE_TYPES, VERSION


class MessageBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(EDGE_TYPES))
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.norm = nn.LayerNorm(width)

    def forward(self, h, edges):
        aggregate = torch.zeros_like(h)
        degree = h.new_zeros((len(h), 1))
        for relation, layer in enumerate(self.messages):
            selected = edges[edges[:, 2] == relation]
            if len(selected):
                src, dst = selected[:, 0], selected[:, 1]
                aggregate.index_add_(0, dst, layer(h[src]))
                degree.index_add_(0, dst, h.new_ones((len(dst), 1)))
        return self.norm(h + self.update(torch.cat([h, aggregate / degree.clamp_min(1)], dim=1)))


class GraphPlanner(nn.Module):
    def __init__(self, width=128, depth=3):
        super().__init__()
        self.config = dict(width=width, depth=depth)
        self.project = nn.Sequential(nn.Linear(NODE_DIM, width), nn.GELU())
        self.blocks = nn.ModuleList(MessageBlock(width) for _ in range(depth))
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(3))

    def forward(self, batch):
        h = self.project(batch["nodes"])
        for block in self.blocks:
            h = block(h, batch["edges"])
        z = self.fuse(torch.cat([h[batch["candidates"]], self.context(batch["context"])], dim=1))
        # All targets are bounded. Cost is normalized work, not dollars.
        return torch.cat([head(z).sigmoid() for head in self.heads], dim=1)


def collate(records):
    nodes, edges, candidates, context, labels, masks = [], [], [], [], [], []
    for r in records:
        if r["version"] != VERSION:
            raise ValueError("Unsupported feature version")
        offset = len(nodes)
        nodes.extend(r["nodes"])
        edges.extend([a + offset, b + offset, t] for a, b, t in r["edges"])
        candidates.extend(i + offset for i in r["candidates"])
        context.extend([r["context"]] * len(r["candidates"]))
        targets = r.get("targets", [[None] * 3 for _ in r["candidates"]])
        if len(targets) != len(r["candidates"]):
            raise ValueError("Misaligned targets")
        for row in targets:
            labels.append([0.0 if v is None else v for v in row])
            masks.append([v is not None for v in row])
    return dict(
        nodes=torch.tensor(nodes, dtype=torch.float32).reshape(-1, NODE_DIM),
        edges=torch.tensor(edges, dtype=torch.long).reshape(-1, 3),
        candidates=torch.tensor(candidates, dtype=torch.long),
        context=torch.tensor(context, dtype=torch.float32).reshape(-1, CONTEXT_DIM),
        labels=torch.tensor(labels, dtype=torch.float32).reshape(-1, 3),
        mask=torch.tensor(masks, dtype=torch.bool).reshape(-1, 3),
    )


def save_checkpoint(path, model, metadata):
    torch.save(
        dict(version=VERSION, config=model.config, weights=model.state_dict(), metadata=metadata), path
    )


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["version"] != VERSION:
        raise ValueError("Incompatible feature schema")
    cfg = data["config"]
    if not 8 <= cfg["width"] <= 256 or not 1 <= cfg["depth"] <= 6:
        raise ValueError("Unsupported model dimensions")
    model = GraphPlanner(**cfg)
    model.load_state_dict(data["weights"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint")
    model.eval()
    return model, data["metadata"]
