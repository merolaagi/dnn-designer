"""Single-shot detection on a grid: objectness, box regression, class.

Detection is the awkward one. Its difficulty is not the loss but the *targets* —
a variable number of boxes per image, each of which must be assigned to a
prediction slot before any loss can be computed. That assignment is the loop
problem, and it is why detection could not be expressed before.

Rather than require annotated data to demonstrate it, this recipe draws its own:
squares and circles at known positions, so the boxes are exact by construction.
That makes the loss and the assignment testable end to end. Pointing it at real
annotations needs a dataset loader that reads them, which is a separate piece of
work — the loop here is the part that was missing.

The network outputs a [5 + classes, S, S] grid: objectness, four box numbers,
then one logit per class, at every cell.
"""

import random

from recipes_sdk import Param, Recipe, install

CLASSES = ["square", "circle"]


def check(ctx):
    shape = ctx.in_shapes[0]
    if len(shape) != 3:
        return f"Detection needs an image Input like [3, 64, 64], not {shape}."
    if not ctx.out_shape or len(ctx.out_shape) != 3:
        return ("The head must output a grid shaped [5 + classes, S, S]. End with "
                "a Conv2d whose filter count is 5 + the number of classes.")
    wanted = 5 + len(CLASSES)
    if ctx.out_shape[0] != wanted:
        return (f"With {len(CLASSES)} classes the output needs {wanted} channels "
                f"(objectness, x, y, w, h, then one per class), not {ctx.out_shape[0]}.")
    if ctx.out_shape[1] != ctx.out_shape[2]:
        return f"The prediction grid should be square, got {ctx.out_shape[1:]}."
    return None


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(
        ctx.parameters(), lr=float(ctx.cfg["lr"]))
    ctx.state["grid"] = int(ctx.out_shape[1])
    ctx.state["size"] = int(ctx.in_shapes[0][1])


def _draw(ctx, batch):
    """Paint shapes onto blank images and record exactly where they went."""
    import torch

    size = ctx.state["size"]
    images = torch.rand(batch, ctx.in_shapes[0][0], size, size,
                        device=ctx.device) * 0.15
    ys = torch.arange(size, device=ctx.device).view(-1, 1).float()
    xs = torch.arange(size, device=ctx.device).view(1, -1).float()
    truth = []

    for b in range(batch):
        boxes = []
        for _ in range(random.randint(1, int(ctx.cfg["max_objects"]))):
            kind = random.randrange(len(CLASSES))
            side = random.uniform(size * 0.15, size * 0.35)
            cx = random.uniform(side / 2, size - side / 2)
            cy = random.uniform(side / 2, size - side / 2)
            if kind == 0:
                mask = ((xs - cx).abs() <= side / 2) & ((ys - cy).abs() <= side / 2)
            else:
                mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (side / 2) ** 2
            colour = torch.rand(ctx.in_shapes[0][0], 1, 1, device=ctx.device) * 0.5 + 0.5
            images[b] = torch.where(mask.unsqueeze(0), colour.expand_as(images[b]),
                                    images[b])
            boxes.append((cx / size, cy / size, side / size, side / size, kind))
        truth.append(boxes)
    return images, truth


def _targets(ctx, truth):
    """Assign each box to the grid cell its centre falls in.

    One object per cell — the simplification that keeps this readable. Two
    centres in the same cell means the later one wins, which is exactly the
    limitation anchor boxes were invented to remove.
    """
    import torch

    grid = ctx.state["grid"]
    n = len(truth)
    obj = torch.zeros(n, grid, grid, device=ctx.device)
    box = torch.zeros(n, 4, grid, grid, device=ctx.device)
    label = torch.zeros(n, grid, grid, dtype=torch.long, device=ctx.device)

    for b, boxes in enumerate(truth):
        for cx, cy, w, h, kind in boxes:
            gx = min(int(cx * grid), grid - 1)
            gy = min(int(cy * grid), grid - 1)
            obj[b, gy, gx] = 1.0
            # offsets are relative to the cell, so the head predicts small numbers
            box[b, 0, gy, gx] = cx * grid - gx
            box[b, 1, gy, gx] = cy * grid - gy
            box[b, 2, gy, gx] = w
            box[b, 3, gy, gx] = h
            label[b, gy, gx] = kind
    return obj, box, label


def _loss(ctx, prediction, obj, box, label):
    import torch
    import torch.nn.functional as F

    pred_obj = prediction[:, 0]
    pred_box = prediction[:, 1:5]
    pred_cls = prediction[:, 5:]

    # objectness is dominated by empty cells, so the positives are weighted up
    positives = obj.sum().clamp(min=1)
    weight = torch.where(obj > 0, float(ctx.cfg["object_weight"]), 1.0)
    loss_obj = (F.binary_cross_entropy_with_logits(pred_obj, obj, reduction="none")
                * weight).mean()

    mask = obj.unsqueeze(1)
    loss_box = (F.smooth_l1_loss(pred_box.sigmoid(), box, reduction="none")
                * mask).sum() / (positives * 4)
    loss_cls = (F.cross_entropy(pred_cls, label, reduction="none")
                * obj).sum() / positives

    total = loss_obj + float(ctx.cfg["box_weight"]) * loss_box + loss_cls
    with torch.no_grad():
        found = ((pred_obj.sigmoid() > 0.5) & (obj > 0)).sum().item()
        right = ((pred_cls.argmax(1) == label) & (obj > 0)).sum().item()
    return total, {
        "obj_loss": float(loss_obj.item()),
        "box_loss": float(loss_box.item()),
        "recall": found / float(positives.item()),
        "class_accuracy": right / float(positives.item()),
    }


def step(ctx, xs, y):
    batch = int(ctx.cfg["batch"])
    images, truth = _draw(ctx, batch)
    obj, box, label = _targets(ctx, truth)
    prediction = ctx.model(images)
    total, parts = _loss(ctx, prediction, obj, box, label)

    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    return {"loss": float(total.item()), **parts}


def preview(ctx):
    import torch

    images, truth = _draw(ctx, 8)
    obj, box, label = _targets(ctx, truth)
    ctx.model.eval()
    with torch.no_grad():
        prediction = ctx.model(images)
        _, parts = _loss(ctx, prediction, obj, box, label)
        confident = (prediction[:, 0].sigmoid() > 0.5).sum().item()
    ctx.model.train()
    actual = int(obj.sum().item())
    return (f"8 held-out images · {actual} objects drawn, {confident} cells fired · "
            f"recall {parts['recall']:.2f} · class accuracy {parts['class_accuracy']:.2f}")


install(Recipe(
    name="Detection",
    doc="Single-shot detection on a grid. The head outputs [5 + classes, S, S]: "
        "objectness, four box numbers and one class logit per cell. Training "
        "data is drawn synthetically — squares and circles at known positions — "
        "so the loss and the box assignment can be exercised without annotated "
        "images. Real annotations need a loader that reads them, which is not "
        "here yet.",
    params=[
        Param("lr", "float", 2e-3),
        Param("batch", "int", 16, min=1),
        Param("max_objects", "int", 3, min=1, help="Shapes drawn per image"),
        Param("object_weight", "float", 5.0, min=1.0,
              help="Upweights the few cells holding an object"),
        Param("box_weight", "float", 5.0, min=0.0),
        Param("steps_per_epoch", "int", 60, min=1),
    ],
    accepts=["none"],
    self_supplied=True,
    steps_per_epoch=60,
    setup=setup, step=step, preview=preview, check=check,
))
