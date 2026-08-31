"""Causal transformer blocks and a sampler — the pieces a small GPT needs.

`GPTBlock` is a layer: pre-norm masked self-attention plus a feed-forward, with
residuals around both. `TextGenerator` is a runtime block, because sampling is a
loop that calls the model repeatedly and is not a tensor transform.
"""

from blocks_sdk import Block, Param, ShapeError, install

PRELUDE = '''
class CausalSelfAttention(nn.Module):
    """Self-attention where a position can only look backwards.

    The mask is what separates a decoder from an encoder: without it the model
    can read the token it is being asked to predict, and the loss goes to zero
    while the model learns nothing.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"{dim} channels do not divide into {heads} heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, L, self.heads, C // self.heads)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        return self.proj(y)


class GPTBlock(nn.Module):
    """One decoder block, normalized before each sublayer rather than after.

    Pre-norm is what makes a deep stack trainable without a warmup schedule.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPTStack(nn.Module):
    """A run of identical decoder blocks."""

    def __init__(self, dim: int, heads: int, depth: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GPTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
'''

GENERATOR = '''
class TextGenerator:
    """Samples continuations from a trained language model.

    The model returns logits over the vocabulary at every position; generation
    takes the last one, samples a token, appends it, and repeats. Context is
    clipped to the block size the model was trained on.
    """

    def __init__(self, model, vocab_path: str = "", max_new_tokens: int = 240,
                 temperature: float = 0.8, top_k: int = 40,
                 block_size: int = 128, device: str = "cpu"):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.block_size = block_size
        self.device = device
        self.itos, self.stoi = [], {}
        if vocab_path:
            self.load_vocab(vocab_path)

    def load_vocab(self, path: str):
        import json
        with open(path) as fh:
            self.itos = json.load(fh)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    @torch.no_grad()
    def generate(self, prompt: str = "\\n"):
        if not self.itos:
            raise RuntimeError("Load a vocabulary before generating.")
        self.model.eval()
        ids = self.encode(prompt) or [0]
        ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        for _ in range(self.max_new_tokens):
            window = ids[:, -self.block_size:]
            out = self.model(window)
            logits = (out[0] if isinstance(out, (tuple, list)) else out)[:, -1, :]
            logits = logits / max(self.temperature, 1e-6)
            if self.top_k:
                k = min(self.top_k, logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)

        return self.decode(ids[0].tolist())
'''


def stack_infer(p, shapes):
    s = shapes[0]
    if len(s) != 2:
        raise ShapeError(
            f"GPTStack works on a sequence [L, C], got {list(s)}. Put an "
            f"Embedding before it."
        )
    heads = int(p["heads"])
    if s[1] % heads:
        raise ShapeError(
            f"{s[1]} channels do not divide into {heads} heads. Pick a head "
            f"count that divides the embedding dimension."
        )
    return list(s)


def stack_learnables(p, ins, out):
    d = ins[0][1]
    h = int(d * float(p["mlp_ratio"]))
    per_block = (3 * d * d + 3 * d) + (d * d + d) + (2 * d * h + h + d) + 4 * d
    return int(per_block) * int(p["depth"]) + 2 * d


install(Block(
    name="GPTStack",
    category="Language",
    doc="A stack of pre-norm causal transformer blocks — the body of a GPT. "
        "Each position attends only to earlier ones, which is what makes "
        "next-token prediction a real task instead of copying.",
    params=[
        Param("depth", "int", 4, min=1, help="Number of decoder blocks"),
        Param("heads", "int", 4, min=1, help="Must divide the embedding dimension"),
        Param("mlp_ratio", "float", 4.0, help="Feed-forward width as a multiple of the dimension"),
        Param("dropout", "float", 0.1, min=0.0, max=0.9),
    ],
    infer=stack_infer,
    learnables=stack_learnables,
    prelude=PRELUDE,
    torch_init=lambda p, ins: (
        f"GPTStack({ins[0][1]}, {int(p['heads'])}, {int(p['depth'])}, "
        f"mlp_ratio={float(p['mlp_ratio'])}, dropout={float(p['dropout'])})"
    ),
))


install(Block(
    name="TextGenerator",
    category="Language",
    kind="runtime",
    n_inputs=-2,
    doc="Samples text from the trained model. Attach it to the Output. Point "
        "vocab_path at the .vocab.json written when the corpus was read.",
    params=[
        Param("vocab_path", "text", "", help="e.g. uploads/demo_corpus.txt.vocab.json"),
        Param("max_new_tokens", "int", 240, min=1),
        Param("temperature", "float", 0.8, min=0.05,
              help="Lower is more predictable, higher more surprising"),
        Param("top_k", "int", 40, min=0, help="0 samples from the whole vocabulary"),
        Param("block_size", "int", 128, min=1, help="Match the Input length"),
    ],
    prelude=GENERATOR,
    runtime_init=lambda p, ins: (
        "TextGenerator(model, vocab_path={v!r}, max_new_tokens={m}, "
        "temperature={t}, top_k={k}, block_size={b})".format(
            v=p["vocab_path"], m=int(p["max_new_tokens"]),
            t=float(p["temperature"]), k=int(p["top_k"]),
            b=int(p["block_size"]))
    ),
    runtime_name=lambda p: "generator",
))
