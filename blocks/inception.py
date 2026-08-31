"""Parallel-branch convolution block, in the Inception style."""

from blocks_sdk import Block, Param, install, need_rank

PRELUDE = '''
class InceptionBlock(nn.Module):
    """Four views of the same input, concatenated.

    A 1x1 path, two spatial paths at different receptive fields, and a pooled
    path. Output channels are the sum of the four branch widths.
    """

    def __init__(self, in_ch: int, b1: int, b3: int, b5: int, bpool: int):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, b1, 1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, max(1, b3 // 2), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, b3 // 2), b3, 3, padding=1))
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, max(1, b5 // 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, b5 // 4), b5, 5, padding=2))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_ch, bpool, 1))

    def forward(self, x):
        return F.relu(torch.cat([
            self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)
        ], dim=1))
'''


def infer(p, shapes):
    s = shapes[0]
    need_rank(s, 3, "InceptionBlock", "[C, H, W]")
    total = sum(int(p[k]) for k in ("b1", "b3", "b5", "bpool"))
    return [total, s[1], s[2]]


def learnables(p, ins, out):
    c = ins[0][0]
    b1, b3, b5, bp = (int(p[k]) for k in ("b1", "b3", "b5", "bpool"))
    r3, r5 = max(1, b3 // 2), max(1, b5 // 4)
    return (c * b1 + b1
            + c * r3 + r3 + r3 * b3 * 9 + b3
            + c * r5 + r5 + r5 * b5 * 25 + b5
            + c * bp + bp)


install(Block(
    name="InceptionBlock",
    category="Vision blocks",
    doc="Four parallel branches concatenated on the channel axis. Spatial size is "
        "unchanged; output channels are the four widths added together.",
    params=[
        Param("b1", "int", 32, min=1, help="1x1 branch"),
        Param("b3", "int", 64, min=1, help="3x3 branch"),
        Param("b5", "int", 16, min=1, help="5x5 branch"),
        Param("bpool", "int", 16, min=1, help="Pooled branch"),
    ],
    infer=infer,
    learnables=learnables,
    prelude=PRELUDE,
    torch_init=lambda p, ins: (
        f"InceptionBlock({ins[0][0]}, {int(p['b1'])}, {int(p['b3'])}, "
        f"{int(p['b5'])}, {int(p['bpool'])})"
    ),
))
