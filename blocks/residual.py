"""Residual and squeeze-excite blocks — the two workhorses of modern vision nets."""

from blocks_sdk import Block, Param, ShapeError, conv_out, install, need_rank

RESIDUAL = '''
class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity path around them.

    The skip path learns a 1x1 projection only when the shape changes, which is
    what lets you stack these without thinking about channel bookkeeping.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))
'''

SE = '''
class SqueezeExcite(nn.Module):
    """Recalibrates channels by what the whole feature map says about them."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w[:, :, None, None]
'''


def residual_infer(p, shapes):
    s = shapes[0]
    need_rank(s, 3, "ResidualBlock", "[C, H, W]")
    stride = int(p["stride"])
    filters = int(p["filters"])
    if filters % int(p["groups"]):
        raise ShapeError(f"{filters} filters is not divisible by {p['groups']} groups")
    return [filters, conv_out(s[1], 3, stride, 1), conv_out(s[2], 3, stride, 1)]


def se_infer(p, shapes):
    need_rank(shapes[0], 3, "SqueezeExcite", "[C, H, W]")
    return list(shapes[0])


def residual_learnables(p, ins, out):
    c_in, c_out, g = ins[0][0], int(p["filters"]), int(p["groups"])
    n = (c_in // g) * c_out * 9 + 2 * c_out          # conv1 + bn1
    n += (c_out // g) * c_out * 9 + 2 * c_out        # conv2 + bn2
    if int(p["stride"]) != 1 or c_in != c_out:
        n += c_in * c_out + 2 * c_out                # 1x1 projection + bn
    return n


def se_learnables(p, ins, out):
    ch = ins[0][0]
    hidden = max(1, ch // int(p["reduction"]))
    return ch * hidden + hidden + hidden * ch + ch


install(Block(
    name="ResidualBlock",
    category="Vision blocks",
    doc="Two convolutions with a skip connection. Stack these to go deep without "
        "the gradient dying on the way back.",
    params=[
        Param("filters", "int", 64, min=1),
        Param("stride", "int", 1, min=1, help="2 halves the spatial size"),
        Param("groups", "int", 1, min=1, help="Set equal to filters for depthwise"),
    ],
    infer=residual_infer,
    learnables=residual_learnables,
    prelude=RESIDUAL,
    torch_init=lambda p, ins: (
        f"ResidualBlock({ins[0][0]}, {int(p['filters'])}, "
        f"stride={int(p['stride'])}, groups={int(p['groups'])})"
    ),
))

install(Block(
    name="SqueezeExcite",
    category="Vision blocks",
    doc="Channel attention. Cheap, and it usually buys a point of accuracy on top "
        "of a residual stack.",
    params=[Param("reduction", "int", 16, min=1, help="Bottleneck ratio")],
    infer=se_infer,
    learnables=se_learnables,
    prelude=SE,
    torch_init=lambda p, ins: f"SqueezeExcite({ins[0][0]}, reduction={int(p['reduction'])})",
))
