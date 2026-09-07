"""Four architectures that do not fit the plain feed-forward mould.

Each is here because it needs something a stack of layers does not give you, and
each note says what that is — a layer that quietly behaves unlike its neighbours
is worse than one that says so.

    CapsuleLayer   runs an iterative routing loop inside its own forward
    SpikingDense   carries membrane state across simulated time steps
    RBM            has an energy and a sampler, and wants layer-wise pretraining
                   that an ordinary loop does not do
    Sampling       draws from a distribution, so its output is not a function of
                   its input alone — and it owes a KL term to the loss

The first two are honest layers: their oddness is contained in forward(), and
gradients flow through them normally. The last two are honest about needing more
than the graph provides, which is stated in the block's own documentation rather
than discovered at training time.
"""

from blocks_sdk import Block, Param, ShapeError, install

CAPSULE = '''
class CapsuleLayer(nn.Module):
    """Capsules with dynamic routing by agreement.

    Each lower capsule predicts what each upper capsule should be; the routing
    loop then reweights those predictions by how well they agree with the
    current consensus. Three rounds is what the paper uses.

    The loop is inside forward(), so it is fixed and differentiable — unlike a
    solver, it always takes the same number of steps.
    """

    def __init__(self, in_caps, in_dim, out_caps, out_dim, routes=3):
        super().__init__()
        self.out_caps, self.out_dim, self.routes = out_caps, out_dim, routes
        # Scaled by fan-in. At 0.01 the predictions are so small that squash
        # returns almost zero and the routing logits barely move — the loop
        # runs and changes the answer by about 1e-6, which is a routing layer
        # in name only.
        self.W = nn.Parameter(torch.randn(1, in_caps, out_caps, out_dim, in_dim)
                              / (in_dim ** 0.5))

    @staticmethod
    def squash(s, dim=-1):
        n2 = (s * s).sum(dim, keepdim=True)
        return (n2 / (1.0 + n2)) * s / torch.sqrt(n2 + 1e-8)

    def forward(self, x):
        # x: [B, in_caps, in_dim]
        u = torch.einsum("bicdk,bik->bicd", self.W.expand(x.size(0), -1, -1, -1, -1), x)
        b = torch.zeros(x.size(0), x.size(1), self.out_caps, 1,
                        device=x.device, dtype=x.dtype)
        for step in range(self.routes):
            c = torch.softmax(b, dim=2)
            v = self.squash((c * u).sum(dim=1, keepdim=True), dim=-1)
            if step < self.routes - 1:
                b = b + (u * v).sum(-1, keepdim=True)
        return v.squeeze(1)
'''

SPIKING = '''
class SpikingDense(nn.Module):
    """Leaky integrate-and-fire neurons, simulated over T steps.

    The input is held constant and presented at every step. Membrane potential
    decays, accumulates input, and fires when it crosses threshold; firing
    subtracts the threshold rather than resetting to zero, which keeps the
    residue.

    A spike is not differentiable, so the backward pass uses a surrogate — the
    derivative of a smooth function in place of the step's. Gradients therefore
    describe a network adjacent to the one that ran, which is the accepted
    bargain in this literature and worth knowing about.
    """

    def __init__(self, in_dim, out_dim, steps=20, beta=0.9, threshold=1.0,
                 readout="rate", bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.steps, self.beta, self.threshold = steps, beta, threshold
        self.readout = readout

    def forward(self, x):
        drive = self.fc(x)
        mem = torch.zeros_like(drive)
        spikes = []
        for _ in range(self.steps):
            mem = self.beta * mem + drive
            over = mem - self.threshold
            fired = _Surrogate.apply(over)
            mem = mem - fired * self.threshold
            spikes.append(fired)
        stacked = torch.stack(spikes, 0)
        if self.readout == "count":
            return stacked.sum(0)
        if self.readout == "last":
            return stacked[-1]
        return stacked.mean(0)          # firing rate


class _Surrogate(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 10.0 * x.abs()) ** 2
'''

RBM = '''
class RBM(nn.Module):
    """A restricted Boltzmann machine, used here as a layer.

    Its forward is the hidden activation, so it stacks like any other layer and
    trains by backpropagation as an ordinary sigmoid dense layer would.

    What it does NOT do here is the thing that makes a Deep Belief Network a
    Deep Belief Network: greedy layer-wise pretraining by contrastive
    divergence, one layer at a time, before any of it is fine-tuned. That is a
    training procedure rather than an architecture, so it belongs in a recipe.
    gibbs() and free_energy() are provided for one to call.
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(out_dim, in_dim))
        self.h_bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.v_bias = nn.Parameter(torch.zeros(in_dim)) if bias else None

    def forward(self, x):
        return torch.sigmoid(F.linear(x, self.W, self.h_bias))

    def gibbs(self, v, steps=1):
        """One or more up-down passes, for contrastive divergence."""
        for _ in range(steps):
            h = torch.bernoulli(self.forward(v))
            v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return v

    def free_energy(self, v):
        vb = v @ self.v_bias if self.v_bias is not None else 0.0
        hidden = F.linear(v, self.W, self.h_bias)
        return -vb - F.softplus(hidden).sum(-1)
'''

SAMPLING = '''
class Sampling(nn.Module):
    """The reparameterisation trick: z = mu + sigma * eps.

    Takes the mean and the log-variance as two inputs and returns a draw. The
    randomness sits in eps, which carries no gradient, so the gradient reaches
    mu and log-var normally — that is the whole trick.

    In eval mode it returns the mean, because a network being measured should
    not give a different answer each time it is asked.

    This layer owes the loss a KL term. Without it the encoder is free to drive
    the variance to zero and the model is an ordinary autoencoder wearing a
    hat. kl() computes it; the reconstruction recipe adds it.
    """

    def forward(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
'''


def capsule_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 2:
        raise ShapeError(
            f"Capsules take [capsules, dimension]; got {shape}. Reshape a "
            f"convolutional map into capsules first.")
    return [int(p["capsules"]), int(p["dim"])]


def capsule_learnables(p, ins, out):
    in_caps, in_dim = ins[0][0], ins[0][1]
    return in_caps * int(p["capsules"]) * int(p["dim"]) * in_dim


def spiking_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 1:
        raise ShapeError(
            f"SpikingDense works on a flat vector; got {shape}. Flatten first.")
    return [int(p["units"])]


def spiking_learnables(p, ins, out):
    return ins[0][0] * int(p["units"]) + (int(p["units"]) if p.get("bias", True) else 0)


def rbm_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 1:
        raise ShapeError(f"An RBM takes a flat vector; got {shape}. Flatten first.")
    return [int(p["units"])]


def rbm_learnables(p, ins, out):
    visible, hidden = ins[0][0], int(p["units"])
    return visible * hidden + (hidden + visible if p.get("bias", True) else 0)


def sampling_infer(p, ins):
    if len(ins) != 2:
        raise ShapeError("Sampling takes two inputs: the mean, then the log-variance")
    if list(ins[0]) != list(ins[1]):
        raise ShapeError(
            f"the mean is {list(ins[0])} and the log-variance {list(ins[1])}; "
            f"they must match")
    return list(ins[0])


install(Block(
    name="CapsuleLayer",
    category="Vision blocks",
    doc="Capsules with dynamic routing by agreement. Each lower capsule "
        "predicts every upper one, and the routing loop reweights those "
        "predictions by how well they agree. Input is [capsules, dimension].",
    params=[
        Param("capsules", "int", 10, min=1, help="Capsules out"),
        Param("dim", "int", 16, min=1, help="Dimension of each"),
        Param("routes", "int", 3, min=1, help="Routing rounds; the paper uses 3"),
    ],
    infer=capsule_infer,
    learnables=capsule_learnables,
    prelude=CAPSULE,
    torch_init=lambda p, ins: (
        f"CapsuleLayer({ins[0][0]}, {ins[0][1]}, {int(p['capsules'])}, "
        f"{int(p['dim'])}, routes={int(p['routes'])})"),
))

install(Block(
    name="SpikingDense",
    category="Dense",
    doc="Leaky integrate-and-fire neurons over simulated time. The input is "
        "presented at every step; the output is the firing rate, the spike "
        "count, or the last step. Spikes are not differentiable, so the "
        "backward pass uses a surrogate gradient.",
    params=[
        Param("units", "int", 64, min=1),
        Param("steps", "int", 20, min=1, help="Simulated time steps"),
        Param("beta", "float", 0.9, help="Membrane decay per step"),
        Param("threshold", "float", 1.0),
        Param("readout", "choice", "rate", options=["rate", "count", "last"]),
        Param("bias", "bool", True),
    ],
    infer=spiking_infer,
    learnables=spiking_learnables,
    prelude=SPIKING,
    torch_init=lambda p, ins: (
        f"SpikingDense({ins[0][0]}, {int(p['units'])}, steps={int(p['steps'])}, "
        f"beta={float(p['beta'])}, threshold={float(p['threshold'])}, "
        f"readout={str(p['readout'])!r}, bias={bool(p.get('bias', True))})"),
))

install(Block(
    name="RBM",
    category="Dense",
    doc="A restricted Boltzmann machine as a layer. Stacked, these are the "
        "body of a Deep Belief Network — but the layer-wise contrastive "
        "divergence pretraining that makes it one is a training procedure, not "
        "an architecture. Without it this trains as a sigmoid dense layer.",
    params=[
        Param("units", "int", 128, min=1, help="Hidden units"),
        Param("bias", "bool", True),
    ],
    infer=rbm_infer,
    learnables=rbm_learnables,
    prelude=RBM,
    torch_init=lambda p, ins: (
        f"RBM({ins[0][0]}, {int(p['units'])}, bias={bool(p.get('bias', True))})"),
))

install(Block(
    name="Sampling",
    category="Data",
    n_inputs=2,
    doc="The reparameterisation trick, for a variational autoencoder. Wire the "
        "mean into the first input and the log-variance into the second. Owes "
        "the loss a KL term — without one the variance collapses and this is an "
        "ordinary autoencoder.",
    params=[],
    infer=sampling_infer,
    learnables=lambda p, ins, out: 0,
    prelude=SAMPLING,
    torch_init=lambda p, ins: "Sampling()",
))
