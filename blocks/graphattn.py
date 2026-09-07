"""Attention-weighted and general message passing over a graph.

GraphConv averages a node's neighbours and projects the result: every neighbour
counts the same. These two relax that in the two ways the literature does.

GraphAttention learns how much each neighbour is worth to each node, so the
weights depend on the pair rather than only on the degree. MessagePassing keeps
the general form — build a message from (sender, receiver), aggregate, then
update — which is the framework the others are instances of.

Both take node features and an adjacency matrix, the same way GraphConv does, so
they drop into a graph already drawn.
"""

from blocks_sdk import Block, Param, ShapeError, install

GAT = '''
class GraphAttention(nn.Module):
    """Attention over neighbours, in the GAT form.

    The coefficient for an edge is a softmax over a learned score of the two
    endpoints' projections. Non-edges are masked to -inf before the softmax, so
    a node only ever attends where the adjacency says it may — a nicety that
    matters, because attention over a fully connected graph is just attention.
    """

    def __init__(self, in_dim, out_dim, heads=1, concat=True, slope=0.2,
                 dropout=0.0):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.slope, self.drop = slope, nn.Dropout(dropout)

    def forward(self, x, adj):
        B, N, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        h = self.lin(x).view(B, N, self.heads, self.out_dim).transpose(1, 2)

        src = (h * self.a_src.unsqueeze(0).unsqueeze(2)).sum(-1)
        dst = (h * self.a_dst.unsqueeze(0).unsqueeze(2)).sum(-1)
        scores = F.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2), self.slope)

        mask = (adj + torch.eye(N, device=adj.device, dtype=adj.dtype)) > 0
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = self.drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, h)

        if self.concat:
            return out.transpose(1, 2).reshape(B, N, self.heads * self.out_dim)
        return out.mean(1)
'''

MPNN = '''
class MessagePassing(nn.Module):
    """The general form: message, aggregate, update.

    A message is built from the sending and receiving node together, summed or
    averaged over the neighbourhood the adjacency allows, then combined with the
    node's own state. GraphConv is this with the message ignoring the receiver
    and the update being the identity.
    """

    def __init__(self, in_dim, out_dim, hidden=None, aggr="sum", bias=True):
        super().__init__()
        hidden = hidden or out_dim
        self.msg = nn.Sequential(
            nn.Linear(2 * in_dim, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, hidden, bias=bias))
        self.upd = nn.Linear(in_dim + hidden, out_dim, bias=bias)
        self.aggr = aggr

    def forward(self, x, adj):
        B, N, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        sender = x.unsqueeze(1).expand(B, N, N, x.size(-1))
        receiver = x.unsqueeze(2).expand(B, N, N, x.size(-1))
        messages = self.msg(torch.cat([sender, receiver], dim=-1))
        messages = messages * adj.unsqueeze(-1)

        if self.aggr == "mean":
            pooled = messages.sum(2) / adj.sum(-1, keepdim=True).clamp(min=1.0)
        elif self.aggr == "max":
            pooled = messages.masked_fill(
                adj.unsqueeze(-1) == 0, float("-inf")).max(2).values
            pooled = torch.nan_to_num(pooled, neginf=0.0)
        else:
            pooled = messages.sum(2)
        return self.upd(torch.cat([x, pooled], dim=-1))
'''


def _graph_shapes(p, shapes, name):
    if len(shapes) != 2:
        raise ShapeError(f"{name} takes two inputs: node features, then adjacency")
    x, adj = shapes[0], shapes[1]
    if len(x) != 2:
        raise ShapeError(f"node features must be [N, F], got {list(x)}")
    if len(adj) != 2 or adj[0] != adj[1]:
        raise ShapeError(f"adjacency must be a square [N, N], got {list(adj)}")
    if adj[0] != x[0]:
        raise ShapeError(
            f"{x[0]} nodes in the features but {adj[0]} in the adjacency matrix")
    return x


def gat_infer(p, shapes):
    x = _graph_shapes(p, shapes, "GraphAttention")
    units, heads = int(p["units"]), max(1, int(p["heads"]))
    return [x[0], units * heads if p.get("concat", True) else units]


def gat_learnables(p, ins, out):
    units, heads = int(p["units"]), max(1, int(p["heads"]))
    return ins[0][1] * units * heads + 2 * heads * units


def mpnn_infer(p, shapes):
    x = _graph_shapes(p, shapes, "MessagePassing")
    return [x[0], int(p["units"])]


def mpnn_learnables(p, ins, out):
    f = ins[0][1]
    units = int(p["units"])
    hidden = int(p["hidden"]) or units
    bias = 1 if p.get("bias", True) else 0
    msg = (2 * f) * hidden + bias * hidden + hidden * hidden + bias * hidden
    upd = (f + hidden) * units + bias * units
    return msg + upd


install(Block(
    name="GraphAttention",
    category="Graphs",
    n_inputs=2,
    doc="Attention over a node's neighbours. Unlike GraphConv, how much a "
        "neighbour counts is learned per pair rather than fixed by the degree. "
        "Non-edges are masked before the softmax, so it attends only where the "
        "adjacency allows.",
    params=[
        Param("units", "int", 32, min=1, help="Width per head"),
        Param("heads", "int", 4, min=1),
        Param("concat", "bool", True,
              help="Join the heads; off averages them, as the last layer usually does"),
        Param("slope", "float", 0.2, help="Leaky ReLU slope on the scores"),
        Param("dropout", "float", 0.0, help="On the attention weights"),
    ],
    infer=gat_infer,
    learnables=gat_learnables,
    prelude=GAT,
    torch_init=lambda p, ins: (
        f"GraphAttention({ins[0][1]}, {int(p['units'])}, heads={int(p['heads'])}, "
        f"concat={bool(p.get('concat', True))}, slope={float(p['slope'])}, "
        f"dropout={float(p['dropout'])})"),
))

install(Block(
    name="MessagePassing",
    category="Graphs",
    n_inputs=2,
    doc="The general graph form: build a message from each connected pair, "
        "aggregate over the neighbourhood, then update the node with it. "
        "GraphConv is this with a simpler message and no update.",
    params=[
        Param("units", "int", 64, min=1),
        Param("hidden", "int", 64, min=1, help="Width of the message network"),
        Param("aggr", "choice", "sum", options=["sum", "mean", "max"]),
        Param("bias", "bool", True),
    ],
    infer=mpnn_infer,
    learnables=mpnn_learnables,
    prelude=MPNN,
    torch_init=lambda p, ins: (
        f"MessagePassing({ins[0][1]}, {int(p['units'])}, "
        f"hidden={int(p['hidden'])}, aggr={str(p['aggr'])!r}, "
        f"bias={bool(p.get('bias', True))})"),
))
