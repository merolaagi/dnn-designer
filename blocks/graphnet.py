"""Message passing over a graph — takes node features and an adjacency matrix."""

from blocks_sdk import Block, Param, ShapeError, install

PRELUDE = '''
class GraphConv(nn.Module):
    """One round of neighbour averaging followed by a learned projection.

    Takes node features [B, N, F] and an adjacency matrix [B, N, N]. With
    normalize on, it adds self-loops and applies the symmetric normalization
    D^-1/2 (A + I) D^-1/2, which is the standard GCN propagation rule.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 normalize: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.normalize = normalize

    def forward(self, x, adj):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        if self.normalize:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
            adj = adj + eye
            deg = adj.sum(-1).clamp(min=1e-6).pow(-0.5)
            adj = deg.unsqueeze(-1) * adj * deg.unsqueeze(-2)
        return torch.bmm(adj, self.lin(x))
'''


def infer(p, shapes):
    if len(shapes) != 2:
        raise ShapeError("GraphConv takes two inputs: node features, then adjacency")
    x, adj = shapes[0], shapes[1]
    if len(x) != 2:
        raise ShapeError(f"node features must be [N, F], got {list(x)}")
    if len(adj) != 2 or adj[0] != adj[1]:
        raise ShapeError(f"adjacency must be a square [N, N], got {list(adj)}")
    if adj[0] != x[0]:
        raise ShapeError(
            f"{x[0]} nodes in the features but {adj[0]} in the adjacency matrix"
        )
    return [x[0], int(p["units"])]


def learnables(p, ins, out):
    return ins[0][1] * int(p["units"]) + (int(p["units"]) if p["bias"] else 0)


install(Block(
    name="GraphConv",
    category="Graphs",
    n_inputs=2,
    doc="Graph convolution. Wire node features into the first input and a square "
        "adjacency matrix into the second. Stack a few to widen the receptive "
        "field by one hop each time.",
    params=[
        Param("units", "int", 64, min=1),
        Param("normalize", "bool", True, help="Add self-loops and symmetric normalization"),
        Param("bias", "bool", True),
    ],
    infer=infer,
    learnables=learnables,
    prelude=PRELUDE,
    torch_init=lambda p, ins: (
        f"GraphConv({ins[0][1]}, {int(p['units'])}, bias={bool(p['bias'])}, "
        f"normalize={bool(p['normalize'])})"
    ),
))
