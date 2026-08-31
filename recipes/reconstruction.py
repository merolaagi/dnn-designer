"""Autoencoding: the target is the input.

The simplest possible recipe, and the one that proves the hook. The built-in
loop could not express this at all — not because reconstruction is hard, but
because it insists on a label, and here there isn't one.
"""

from recipes_sdk import Param, Recipe, install


def check(ctx):
    if ctx.out_shape and list(ctx.out_shape) != list(ctx.in_shapes[0]):
        return (f"The output is {list(ctx.out_shape)} but the input is "
                f"{list(ctx.in_shapes[0])}. An autoencoder has to come back to "
                f"the shape it started from — add a Reshape, or widen the decoder.")
    return None


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(
        ctx.parameters(), lr=float(ctx.cfg["lr"]))


def _loss(ctx, out, target):
    import torch.nn.functional as F
    if ctx.cfg["objective"] == "l1":
        return F.l1_loss(out, target)
    if ctx.cfg["objective"] == "bce":
        return F.binary_cross_entropy_with_logits(out, target)
    return F.mse_loss(out, target)


def step(ctx, xs, y):
    target = xs[0]
    noise = float(ctx.cfg["noise"])
    if noise > 0:
        import torch
        # denoising: corrupt the input, ask for the clean original back
        xs = [xs[0] + torch.randn_like(xs[0]) * noise]
    out = ctx.model(*xs)
    loss = _loss(ctx, out, target)
    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": float(loss.item())}


def evaluate(ctx, xs, y):
    out = ctx.model(*xs)
    return {"loss": float(_loss(ctx, out, xs[0]).item())}


install(Recipe(
    name="Autoencoder",
    doc="Trains a network to reproduce its own input. Put a narrow layer in the "
        "middle and it learns a compressed code. Raise the noise above zero and "
        "it becomes a denoising autoencoder, which learns far more useful "
        "features than plain reconstruction.",
    params=[
        Param("lr", "float", 1e-3),
        Param("objective", "select", "mse", options=["mse", "l1", "bce"],
              help="bce expects logits out and inputs in [0, 1]"),
        Param("noise", "float", 0.0, min=0.0, max=1.0,
              help="Above 0 corrupts the input and asks for the clean version"),
    ],
    accepts=["image", "tabular"],
    setup=setup, step=step, evaluate=evaluate, check=check,
))
