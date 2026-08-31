"""DDPM: learn to undo noise, then generate by undoing it from pure noise.

This one proves a recipe can rewrite the forward pass. The model never sees a
clean image during training — it sees a noisy one and is asked what noise was
added. Sampling then runs that backwards, which is a loop the built-in trainer
has no way to express.

Timestep conditioning is carried as one extra input channel holding t/T
broadcast across the image. Cruder than a sinusoidal embedding, but it needs no
special layer and works with the single-Input graph the canvas already builds.
So an Input of [C+1, H, W] trains on images of C channels.
"""

from recipes_sdk import Param, Recipe, install


def check(ctx):
    shape = ctx.in_shapes[0]
    if len(shape) != 3:
        return f"Diffusion works on images, so the Input should be [C, H, W], not {shape}."
    if not ctx.out_shape or len(ctx.out_shape) != 3:
        return "The network has to output an image the same size as its input."
    if ctx.out_shape[1:] != shape[1:]:
        return (f"Output {list(ctx.out_shape)} does not match input {list(shape)} "
                f"spatially. A diffusion model predicts noise over the whole image.")
    if shape[0] != ctx.out_shape[0] + 1:
        return (f"The Input needs one channel more than the Output: the extra one "
                f"carries the timestep. Set the Input to "
                f"[{ctx.out_shape[0] + 1}, {shape[1]}, {shape[2]}].")
    return None


def setup(ctx):
    import torch

    steps = int(ctx.cfg["steps"])
    # cosine schedule: keeps signal around longer than the original linear one
    t = torch.linspace(0, steps, steps + 1) / steps
    alpha_bar = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)

    ctx.state["betas"] = betas.to(ctx.device)
    ctx.state["alphas"] = (1 - betas).to(ctx.device)
    ctx.state["alpha_bar"] = torch.cumprod(1 - betas, dim=0).to(ctx.device)
    ctx.optimizers["main"] = torch.optim.AdamW(
        ctx.parameters(), lr=float(ctx.cfg["lr"]))


def _with_time(x, t, steps):
    """Append the normalized timestep as a constant channel."""
    import torch
    stamp = (t.float() / steps).view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
    return torch.cat([x, stamp], dim=1)


def _noisy_batch(ctx, x):
    import torch
    steps = int(ctx.cfg["steps"])
    ab = ctx.state["alpha_bar"]
    t = torch.randint(0, steps, (x.size(0),), device=x.device)
    noise = torch.randn_like(x)
    a = ab[t].view(-1, 1, 1, 1)
    noisy = a.sqrt() * x + (1 - a).sqrt() * noise
    return _with_time(noisy, t, steps), noise


def step(ctx, xs, y):
    import torch.nn.functional as F

    x = xs[0]
    # remember the data range so sampling can clamp to something meaningful
    lo, hi = float(x.min()), float(x.max())
    ctx.state["lo"] = min(ctx.state.get("lo", lo), lo)
    ctx.state["hi"] = max(ctx.state.get("hi", hi), hi)
    model_in, noise = _noisy_batch(ctx, x)
    predicted = ctx.model(model_in)
    loss = F.mse_loss(predicted, noise)
    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": float(loss.item())}


def evaluate(ctx, xs, y):
    import torch.nn.functional as F
    model_in, noise = _noisy_batch(ctx, xs[0])
    return {"loss": float(F.mse_loss(ctx.model(model_in), noise).item())}


def preview(ctx):
    """Sample from pure noise and report what came out.

    There is nowhere to put an image in the log, so this reports the statistics
    a broken sampler gives itself away with: a diverged run returns enormous
    values, a collapsed one returns near-identical ones.
    """
    import torch

    steps = int(ctx.cfg["steps"])
    shown = min(int(ctx.cfg["preview_steps"]), steps)
    stride = max(1, steps // shown)
    c, h, w = ctx.out_shape
    x = torch.randn(4, c, h, w, device=ctx.device)
    ab = ctx.state["alpha_bar"]
    lo = ctx.state.get("lo", 0.0)
    hi = ctx.state.get("hi", 1.0)
    span = max(hi - lo, 1e-3)

    # DDIM. The ancestral DDPM update is only valid one step at a time; using
    # its per-step coefficients while skipping steps under-denoises and the
    # variance runs away. DDIM predicts the clean image at each stop and jumps
    # straight to the next noise level, so any stride is valid.
    schedule = list(range(steps - 1, -1, -stride))
    ctx.model.eval()
    with torch.no_grad():
        for index, i in enumerate(schedule):
            t = torch.full((4,), i, device=ctx.device, dtype=torch.long)
            eps = ctx.model(_with_time(x, t, steps))
            a_t = ab[i]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(lo - 0.1 * span, hi + 0.1 * span)
            nxt = schedule[index + 1] if index + 1 < len(schedule) else None
            a_prev = ab[nxt] if nxt is not None else torch.tensor(1.0, device=x.device)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    ctx.model.train()

    spread = x.flatten(1).std(dim=1).mean().item()
    between = (x[0] - x[1]).abs().mean().item()
    drift = "" if lo - span <= x.min().item() and x.max().item() <= hi + span \
        else "  <- outside the data range, the sampler is diverging"
    return (f"sampled 4 images in {len(schedule)} DDIM steps · "
            f"range [{x.min().item():.2f}, {x.max().item():.2f}] "
            f"against data [{lo:.2f}, {hi:.2f}] · "
            f"detail {spread:.3f} · variety {between:.3f}{drift}")


install(Recipe(
    name="Diffusion",
    doc="Denoising diffusion. The network is trained to predict the noise added "
        "to an image, and sampling runs that in reverse from pure noise. Give "
        "the Input one channel more than the Output — the extra one carries the "
        "timestep. A U-Net shape works best: downsample, then ConvTranspose2d "
        "back up with Concat skips.",
    params=[
        Param("lr", "float", 2e-4),
        Param("steps", "int", 200, min=10, help="Noise levels; more is slower and smoother"),
        Param("preview_steps", "int", 50, min=1,
              help="Sampling steps used for the per-epoch preview"),
    ],
    accepts=["image"],
    setup=setup, step=step, evaluate=evaluate, preview=preview, check=check,
))
