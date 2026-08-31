"""The two heads an AlphaZero-style network needs: a move policy and a position value."""

from blocks_sdk import Block, Param, install, need_rank, prod

POLICY = '''
class PolicyHead(nn.Module):
    """Board features to a logit per move.

    Kept as logits on purpose: cross-entropy against the search visit counts
    wants them raw, and the search applies its own softmax.
    """

    def __init__(self, in_ch: int, plane_size: int, actions: int, planes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.fc = nn.Linear(planes * plane_size, actions)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        return self.fc(h.flatten(1))
'''

VALUE = '''
class ValueHead(nn.Module):
    """Board features to a single number in [-1, 1], from the mover's point of view."""

    def __init__(self, in_ch: int, plane_size: int, hidden: int = 256, planes: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.fc1 = nn.Linear(planes * plane_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = F.relu(self.fc1(h.flatten(1)))
        return torch.tanh(self.fc2(h))
'''


def policy_infer(p, shapes):
    need_rank(shapes[0], 3, "PolicyHead", "[C, H, W]")
    return [int(p["actions"])]


def value_infer(p, shapes):
    need_rank(shapes[0], 3, "ValueHead", "[C, H, W]")
    return [1]


def policy_learnables(p, ins, out):
    c, planes = ins[0][0], int(p["planes"])
    flat = planes * ins[0][1] * ins[0][2]
    return c * planes + 2 * planes + flat * int(p["actions"]) + int(p["actions"])


def value_learnables(p, ins, out):
    c, planes, hidden = ins[0][0], int(p["planes"]), int(p["hidden"])
    flat = planes * ins[0][1] * ins[0][2]
    return c * planes + 2 * planes + flat * hidden + hidden + hidden + 1


install(Block(
    name="PolicyHead",
    category="Game playing",
    doc="Move-probability head for a board game network. Outputs one logit per "
        "action; train it against the search visit counts.",
    params=[
        Param("actions", "int", 362, min=1, help="Legal move count, plus pass if there is one"),
        Param("planes", "int", 2, min=1, help="1x1 convolution width before the linear layer"),
    ],
    infer=policy_infer,
    learnables=policy_learnables,
    prelude=POLICY,
    torch_init=lambda p, ins: (
        f"PolicyHead({ins[0][0]}, {ins[0][1] * ins[0][2]}, "
        f"{int(p['actions'])}, planes={int(p['planes'])})"
    ),
))

install(Block(
    name="ValueHead",
    category="Game playing",
    doc="Scores a position in [-1, 1] from the point of view of the player to "
        "move. Pair it with an Output set to regression.",
    params=[
        Param("hidden", "int", 256, min=1),
        Param("planes", "int", 1, min=1),
    ],
    infer=value_infer,
    learnables=value_learnables,
    prelude=VALUE,
    torch_init=lambda p, ins: (
        f"ValueHead({ins[0][0]}, {ins[0][1] * ins[0][2]}, "
        f"hidden={int(p['hidden'])}, planes={int(p['planes'])})"
    ),
))
