"""Self-paced curriculum: learn the easy examples first, admit harder ones later.

Bengio et al., *Curriculum Learning* (ICML 2009); Kumar, Packer & Koller,
*Self-Paced Learning for Latent Variable Models* (NeurIPS 2010). People do not
learn from a shuffled deck; they learn ordered material. The self-paced form
needs no teacher: an example counts only if its current loss is below a
threshold λ, and λ grows by a factor each epoch until everything is admitted.

    v_i = 1[ℓ_i < λ],      loss = Σ v_i ℓ_i / Σ v_i,      λ ← λ · growth

What is measured is how many examples are admitted per epoch and whether the
final accuracy, at a fixed data budget, beats the plain shuffled loop. It often
does on noisy labels and small data; it often does not on clean large data.
The log reports which happened.
"""
from recipes_sdk import Param, Recipe, install


def _per_example(ctx, xs, y):
    import torch.nn.functional as F
    out = ctx.model(*xs)
    if y.dtype in (torch_long(), torch_int()):
        return F.cross_entropy(out, y, reduction="none"), out
    return ((out - y.view_as(out)) ** 2).flatten(1).mean(dim=1), out


def torch_long():
    import torch
    return torch.long


def torch_int():
    import torch
    return torch.int


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(ctx.parameters(), lr=ctx.cfg["lr"])
    ctx.state["lam"] = float(ctx.cfg["start"])
    ctx.state["epoch_seen"] = -1


def step(ctx, xs, y):
    if ctx.epoch != ctx.state["epoch_seen"] and ctx.state["epoch_seen"] >= 0:
        ctx.state["lam"] *= float(ctx.cfg["growth"])
    ctx.state["epoch_seen"] = ctx.epoch
    lam = ctx.state["lam"]

    losses, out = _per_example(ctx, xs, y)
    admitted = (losses.detach() < lam).float()
    if admitted.sum() == 0:                      # never let a batch teach nothing
        admitted = (losses.detach() <= losses.detach().min()).float()
    loss = (admitted * losses).sum() / admitted.sum()

    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    report = {"loss": float(loss.item()), "admitted": float(admitted.mean().item()),
              "threshold": lam}
    if y.dtype == torch_long():
        report["acc"] = float((out.argmax(1) == y).float().mean().item())
    return report


def evaluate(ctx, xs, y):
    import torch
    with torch.no_grad():
        losses, out = _per_example(ctx, xs, y)
        report = {"loss": float(losses.mean().item())}
        if y.dtype == torch_long():
            report["acc"] = float((out.argmax(1) == y).float().mean().item())
    return report


install(Recipe(
    name="SelfPaced",
    doc="Curriculum without a teacher: an example counts only while its loss is "
        "under a threshold that rises each epoch. Reports the admitted fraction "
        "so you can see the curriculum happen. Kumar et al. 2010.",
    params=[
        Param("lr", "float", 1e-3),
        Param("start", "float", 1.0, help="Initial loss threshold λ"),
        Param("growth", "float", 1.3, help="λ is multiplied by this each epoch"),
    ],
    accepts=["image", "tabular"],
    uses_labels=True,
    setup=setup, step=step, evaluate=evaluate,
    objective="loss",
))
