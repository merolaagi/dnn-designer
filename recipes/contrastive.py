"""SimCLR: learn representations with no labels at all.

Two augmented views of the same image should land near each other in the
embedding space, and away from every other image in the batch. This one proves
the recipe can construct its own batch — nothing about the incoming data changes,
but what the model sees per step is built here rather than by the loader.
"""

from recipes_sdk import Param, Recipe, install


def check(ctx):
    if not ctx.out_shape or len(ctx.out_shape) != 1:
        return ("SimCLR wants a flat embedding out, like [128]. End the graph "
                "with GlobalAvgPool then Linear.")
    if len(ctx.in_shapes[0]) != 3:
        return f"SimCLR augments images, so the Input should be [C, H, W], not {ctx.in_shapes[0]}."
    return None


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(
        ctx.parameters(), lr=float(ctx.cfg["lr"]), weight_decay=1e-6)


def _augment(x, cfg):
    """Two random views, done on the tensor so no dataset surgery is needed."""
    import torch

    out = x
    if cfg["flip"]:
        flip = torch.rand(x.size(0), device=x.device) < 0.5
        out = torch.where(flip[:, None, None, None], out.flip(-1), out)
    if float(cfg["jitter"]) > 0:
        j = float(cfg["jitter"])
        scale = 1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * j
        shift = (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * j
        out = out * scale + shift
    if float(cfg["erase"]) > 0:
        mask = torch.rand_like(out) > float(cfg["erase"])
        out = out * mask
    if float(cfg["noise"]) > 0:
        out = out + torch.randn_like(out) * float(cfg["noise"])
    return out


def _nt_xent(z1, z2, temperature):
    """Each view's partner is the positive; everything else in the batch is not."""
    import torch
    import torch.nn.functional as F

    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = (z @ z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    # view i pairs with view i+n, and the other way round
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def step(ctx, xs, y):
    x = xs[0]
    if x.size(0) < 4:
        return {"loss": 0.0}          # the loss is meaningless on a tiny batch
    v1, v2 = _augment(x, ctx.cfg), _augment(x, ctx.cfg)
    z1, z2 = ctx.model(v1), ctx.model(v2)
    loss = _nt_xent(z1, z2, float(ctx.cfg["temperature"]))
    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    import torch
    with torch.no_grad():
        import torch.nn.functional as F
        agreement = F.cosine_similarity(z1, z2).mean().item()
    return {"loss": float(loss.item()), "agreement": float(agreement)}


def evaluate(ctx, xs, y):
    x = xs[0]
    if x.size(0) < 4:
        return {"loss": 0.0}
    v1, v2 = _augment(x, ctx.cfg), _augment(x, ctx.cfg)
    z1, z2 = ctx.model(v1), ctx.model(v2)
    return {"loss": float(_nt_xent(z1, z2, float(ctx.cfg["temperature"])).item())}


install(Recipe(
    name="Contrastive",
    doc="Self-supervised pretraining, SimCLR style. No labels needed. Train the "
        "trunk here, save the checkpoint, then start a classifier from it with "
        "far less labelled data than training from scratch would need. Large "
        "batches matter: the negatives all come from within the batch.",
    params=[
        Param("lr", "float", 1e-3),
        Param("temperature", "float", 0.2, min=0.01,
              help="Lower sharpens the distinction between positives and negatives"),
        Param("flip", "bool", True),
        Param("jitter", "float", 0.4, min=0.0, max=1.0, help="Brightness and contrast"),
        Param("erase", "float", 0.1, min=0.0, max=0.9, help="Fraction of pixels dropped"),
        Param("noise", "float", 0.05, min=0.0, max=1.0),
    ],
    accepts=["image"],
    setup=setup, step=step, evaluate=evaluate, check=check,
))
