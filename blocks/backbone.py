"""Pretrained torchvision backbones, used as feature extractors.

This is the transfer-learning path: drop a Backbone in, follow it with a pooling
layer and your own head, and freeze whatever you don't want to disturb. The
classifier that came with the network is stripped, so what comes out is a
feature map, not ImageNet logits.
"""

from blocks_sdk import Block, Param, ShapeError, install, need_rank

# Channels each backbone emits once its classifier is removed, and the factor by
# which it divides the input resolution. Every one of these is a /32 network.
BACKBONES = {
    "resnet18":           {"channels": 512,  "params": 11_176_512},
    "resnet34":           {"channels": 512,  "params": 21_284_672},
    "resnet50":           {"channels": 2048, "params": 23_508_032},
    "resnet101":          {"channels": 2048, "params": 42_500_160},
    "vgg16":              {"channels": 512,  "params": 14_714_688},
    "densenet121":        {"channels": 1024, "params": 6_953_856},
    "mobilenet_v3_large": {"channels": 960,  "params": 2_971_952},
    "efficientnet_b0":    {"channels": 1280, "params": 4_007_548},
    "convnext_tiny":      {"channels": 768,  "params": 27_818_592},
}

STRIDE = 32

PRELUDE = '''
class TorchvisionBackbone(nn.Module):
    """A pretrained torchvision network with its classifier removed.

    Output is a feature map [B, C, H/32, W/32]. Freezing is expressed as a count
    of trailing stages left trainable, so `trainable_stages=0` is a pure feature
    extractor and `trainable_stages=2` fine-tunes the last two.
    """

    def __init__(self, arch: str, weights: str = "DEFAULT", in_channels: int = 3,
                 trainable_stages: int = 0):
        super().__init__()
        import torchvision

        factory = getattr(torchvision.models, arch)
        model = factory(weights=weights) if weights else factory(weights=None)

        # densenet, vgg, mobilenet, efficientnet and convnext expose .features;
        # the resnets do not, so drop their pooling and fc instead.
        if hasattr(model, "features"):
            self.body = model.features
        else:
            self.body = nn.Sequential(*list(model.children())[:-2])

        if in_channels != 3:
            self._adapt_first_conv(in_channels)

        stages = [m for m in self.body.children()]
        cutoff = len(stages) - max(0, trainable_stages)
        for i, stage in enumerate(stages):
            if i < cutoff:
                for p in stage.parameters():
                    p.requires_grad_(False)

    def _adapt_first_conv(self, in_channels: int):
        """Re-purpose the RGB stem for a different channel count.

        Averaging the pretrained kernels across the input axis keeps the filter
        responses roughly calibrated, which beats starting the stem from noise.
        """
        for module in self.body.modules():
            if isinstance(module, nn.Conv2d):
                old = module.weight.data
                mean = old.mean(dim=1, keepdim=True)
                new = mean.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
                module.in_channels = in_channels
                module.weight = nn.Parameter(new)
                break

    def forward(self, x):
        return self.body(x)
'''


def infer(p, shapes):
    s = shapes[0]
    need_rank(s, 3, "Backbone", "[C, H, W]")
    arch = p["arch"]
    if arch not in BACKBONES:
        raise ShapeError(f"unknown backbone {arch}")
    if min(s[1], s[2]) < STRIDE:
        raise ShapeError(
            f"{arch} divides the input by {STRIDE}; a {s[1]}x{s[2]} image would "
            f"collapse. Feed it at least {STRIDE}x{STRIDE}, or 224x224 to match "
            f"how it was trained."
        )
    return [BACKBONES[arch]["channels"], s[1] // STRIDE, s[2] // STRIDE]


def learnables(p, ins, out):
    return BACKBONES[p["arch"]]["params"]


install(Block(
    name="Backbone",
    category="Pretrained",
    doc="A pretrained torchvision network with its classifier stripped off. "
        "Follow it with GlobalAvgPool and a Linear head sized to your classes. "
        "Leave trainable stages at 0 to train only your head, then raise it to "
        "fine-tune deeper once the head has settled.",
    params=[
        Param("arch", "select", "resnet18", options=sorted(BACKBONES),
              help="All of these divide the input resolution by 32"),
        Param("weights", "select", "DEFAULT", options=["DEFAULT", "none"],
              help="DEFAULT downloads ImageNet weights on first use"),
        Param("trainable_stages", "int", 0, min=0,
              help="How many trailing stages stay unfrozen"),
    ],
    infer=infer,
    learnables=learnables,
    learnables_approx=True,
    prelude=PRELUDE,
    torch_init=lambda p, ins: (
        "TorchvisionBackbone({arch!r}, weights={w}, in_channels={c}, "
        "trainable_stages={t})".format(
            arch=p["arch"],
            w="None" if p["weights"] == "none" else '"DEFAULT"',
            c=ins[0][0], t=int(p["trainable_stages"]))
    ),
))
