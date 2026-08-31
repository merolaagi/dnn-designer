"""Adversarial training: two networks, two optimizers, alternating updates.

This is the case that forced recipes to own their own backward pass. The
discriminator and generator updates interleave, each needs its own graph freed
at the right moment, and the generator's loss flows *through* a network it must
not update. No single-optimizer loop can express that.

The canvas graph is the generator: noise `[Z]` in, image `[C, H, W]` out. The
discriminator is a second design you save separately and pick in the form.
"""

from recipes_sdk import Param, Recipe, install


def check(ctx):
    if len(ctx.in_shapes[0]) != 1:
        return (f"The generator takes a noise vector, so its Input should be "
                f"[Z] — something like [64] — not {ctx.in_shapes[0]}.")
    if not ctx.out_shape or len(ctx.out_shape) != 3:
        return "The generator has to output an image shaped [C, H, W]."
    return None


def data_shape(ctx):
    """The loader should hand over images, not the noise the generator takes."""
    return [list(ctx.out_shape)]


def setup(ctx):
    import torch

    ctx.optimizers["g"] = torch.optim.Adam(
        ctx.parameters("main"), lr=float(ctx.cfg["lr_g"]), betas=(0.5, 0.999))
    ctx.optimizers["d"] = torch.optim.Adam(
        ctx.parameters("discriminator"), lr=float(ctx.cfg["lr_d"]), betas=(0.5, 0.999))
    ctx.state["z"] = int(ctx.in_shapes[0][0])


def _noise(ctx, n):
    import torch
    return torch.randn(n, ctx.state["z"], device=ctx.device)


def _scores(ctx, images):
    """Discriminator output as a flat vector of logits, whatever shape it ends in."""
    out = ctx.models["discriminator"](images)
    return out.reshape(out.size(0), -1).mean(dim=1)


def step(ctx, xs, y):
    import torch
    import torch.nn.functional as F

    real = xs[0]
    n = real.size(0)
    smooth = float(ctx.cfg["label_smoothing"])

    # --- discriminator: tell real from generated ---
    fake = ctx.model(_noise(ctx, n)).detach()      # detach: G is not updated here
    d_real = _scores(ctx, real)
    d_fake = _scores(ctx, fake)
    loss_d = (F.binary_cross_entropy_with_logits(
                  d_real, torch.full_like(d_real, 1.0 - smooth))
              + F.binary_cross_entropy_with_logits(
                  d_fake, torch.zeros_like(d_fake))) / 2
    ctx.optimizers["d"].zero_grad(set_to_none=True)
    loss_d.backward()
    ctx.optimizers["d"].step()

    # --- generator: make the discriminator call its output real ---
    # The non-saturating form. Minimising -log D(G(z)) rather than
    # maximising log(1 - D(G(z))) keeps gradients alive while G is still bad.
    loss_g = torch.tensor(0.0, device=ctx.device)
    for _ in range(int(ctx.cfg["g_steps"])):
        generated = ctx.model(_noise(ctx, n))
        scores = _scores(ctx, generated)
        loss_g = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
        ctx.optimizers["g"].zero_grad(set_to_none=True)
        loss_g.backward()
        ctx.optimizers["g"].step()

    with torch.no_grad():
        acc_real = (d_real > 0).float().mean().item()
        acc_fake = (d_fake < 0).float().mean().item()

    return {"loss": float(loss_g.item()), "d_loss": float(loss_d.item()),
            "d_accuracy": float((acc_real + acc_fake) / 2)}


def evaluate(ctx, xs, y):
    import torch
    with torch.no_grad():
        generated = ctx.model(_noise(ctx, xs[0].size(0)))
        real_spread = xs[0].flatten(1).std(dim=1).mean().item()
        fake_spread = generated.flatten(1).std(dim=1).mean().item()
    # not a loss, but the number that actually tells you whether G collapsed
    return {"loss": abs(real_spread - fake_spread),
            "fake_detail": float(fake_spread), "real_detail": float(real_spread)}


def preview(ctx):
    import torch
    with torch.no_grad():
        batch = ctx.model(_noise(ctx, 8))
        variety = (batch[:4] - batch[4:]).abs().mean().item()
    collapse = "  <- samples are nearly identical, the generator has collapsed" \
        if variety < 0.01 else ""
    return (f"8 samples · range [{batch.min().item():.2f}, {batch.max().item():.2f}] · "
            f"variety {variety:.4f}{collapse}")


install(Recipe(
    name="GAN",
    doc="Adversarial training. The canvas graph is the generator: noise [Z] in, "
        "image out. Save a discriminator separately — image in, one number out — "
        "and choose it below. Watch d_accuracy: near 0.5 means the two are "
        "balanced, near 1.0 means the discriminator has won and the generator "
        "has stopped learning.",
    params=[
        Param("lr_g", "float", 2e-4),
        Param("lr_d", "float", 2e-4),
        Param("g_steps", "int", 1, min=1,
              help="Generator updates per discriminator update; raise if D dominates"),
        Param("label_smoothing", "float", 0.1, min=0.0, max=0.4,
              help="Targets real as 0.9 instead of 1.0, which steadies D"),
    ],
    extra_models=["discriminator"],
    accepts=["image"],
    data_shape=data_shape,
    setup=setup, step=step, evaluate=evaluate, preview=preview, check=check,
))
