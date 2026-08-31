"""Monte Carlo tree search over a policy/value network — the search half of AlphaZero.

This is a runtime block, not a layer. Search is not a tensor transform: it calls
the network many times per move and picks a move by visit count. It generates a
separate section of the exported file that wraps the trained model.
"""

from blocks_sdk import Block, Param, install

PRELUDE = '''
class Environment:
    """What the search needs from your game. Implement these five methods.

    legal_actions(state) -> list[int]     indices into the policy head
    step(state, action)  -> state         the position after that move
    is_terminal(state)   -> bool
    reward(state)        -> float         final result for the player to move
    encode(state)        -> torch.Tensor  shaped like the network Input, no batch dim
    """

    def legal_actions(self, state): raise NotImplementedError
    def step(self, state, action): raise NotImplementedError
    def is_terminal(self, state): raise NotImplementedError
    def reward(self, state): raise NotImplementedError
    def encode(self, state): raise NotImplementedError


class SearchNode:
    __slots__ = ("prior", "visits", "value_sum", "children", "expanded")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.expanded = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class AlphaZeroSearch:
    """PUCT search guided by a policy/value network.

    The network is expected to return ``(policy_logits, value)``. If your graph
    has the heads in the other order, pass policy_index=1.
    """

    def __init__(self, model, env=None, simulations: int = 400, c_puct: float = 1.25,
                 dirichlet_alpha: float = 0.3, exploration_fraction: float = 0.25,
                 policy_index: int = 0, device: str = "cpu"):
        self.model = model
        self.env = env
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.policy_index = policy_index
        self.device = device

    @torch.no_grad()
    def evaluate(self, state):
        """Policy priors and a value estimate for one position."""
        x = self.env.encode(state).unsqueeze(0).to(self.device)
        out = self.model(x)
        outs = list(out) if isinstance(out, (tuple, list)) else [out, None]
        logits = outs[self.policy_index][0]
        value = outs[1 - self.policy_index]
        legal = self.env.legal_actions(state)
        mask = torch.full_like(logits, float("-inf"))
        mask[legal] = 0.0
        priors = torch.softmax(logits + mask, dim=-1)
        v = float(value[0].item()) if value is not None else 0.0
        return {a: float(priors[a]) for a in legal}, v

    def _expand(self, node, state):
        priors, value = self.evaluate(state)
        for action, prior in priors.items():
            node.children[action] = SearchNode(prior)
        node.expanded = True
        return value

    def _select(self, node):
        best, best_score = None, -float("inf")
        total = math.sqrt(max(node.visits, 1))
        for action, child in node.children.items():
            u = self.c_puct * child.prior * total / (1 + child.visits)
            # child values are from the opponent's view, hence the negation
            score = -child.value + u
            if score > best_score:
                best, best_score = action, score
        return best

    def _add_noise(self, root):
        if not root.children or self.exploration_fraction <= 0:
            return
        actions = list(root.children)
        noise = torch.distributions.Dirichlet(
            torch.full((len(actions),), self.dirichlet_alpha)).sample()
        frac = self.exploration_fraction
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1 - frac) + float(n) * frac

    def run(self, state, add_noise: bool = True):
        """Search from a position and return visit counts per action."""
        if self.env is None:
            raise RuntimeError("Give the search an Environment before calling run().")
        root = SearchNode()
        self._expand(root, state)
        if add_noise:
            self._add_noise(root)

        for _ in range(self.simulations):
            node, path, s = root, [root], state
            while node.expanded and node.children:
                action = self._select(node)
                s = self.env.step(s, action)
                node = node.children[action]
                path.append(node)

            if self.env.is_terminal(s):
                value = self.env.reward(s)
            else:
                value = self._expand(node, s)

            # walk back up, flipping sign at every ply
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = -value

        return {a: c.visits for a, c in root.children.items()}

    def act(self, state, temperature: float = 1.0):
        """Pick a move. Temperature 0 plays the most-visited move outright."""
        counts = self.run(state)
        actions = list(counts)
        visits = torch.tensor([counts[a] for a in actions], dtype=torch.float32)
        if temperature <= 1e-6:
            return actions[int(visits.argmax())], counts
        probs = (visits ** (1.0 / temperature))
        probs = probs / probs.sum()
        return actions[int(torch.multinomial(probs, 1))], counts


def self_play_game(search, env, root_state, temperature: float = 1.0,
                   temperature_moves: int = 30):
    """Play one game against itself and return training examples.

    Each example is (encoded_state, visit_count_policy, outcome), which is
    exactly what the policy and value heads want as targets.
    """
    history, state, ply = [], root_state, 0
    while not env.is_terminal(state):
        t = temperature if ply < temperature_moves else 0.0
        action, counts = search.act(state, temperature=t)
        total = sum(counts.values()) or 1
        policy = torch.zeros(search.model_actions if hasattr(search, "model_actions")
                             else max(counts) + 1)
        for a, c in counts.items():
            policy[a] = c / total
        history.append([env.encode(state), policy])
        state = env.step(state, action)
        ply += 1

    outcome = env.reward(state)
    examples = []
    for encoded, policy in reversed(history):
        examples.append((encoded, policy, outcome))
        outcome = -outcome
    return list(reversed(examples))
'''


install(Block(
    name="MCTSSearch",
    category="Game playing",
    kind="runtime",
    n_inputs=-1,
    doc="AlphaZero-style PUCT search. Connect it to your policy and value heads. "
        "It is not part of forward() — it wraps the trained model and needs an "
        "Environment for your game, which the generated file leaves for you to write.",
    params=[
        Param("simulations", "int", 400, min=1, help="Network calls per move"),
        Param("c_puct", "float", 1.25, help="Exploration constant"),
        Param("dirichlet_alpha", "float", 0.3, help="Root noise; 0.03 for Go, 0.3 for chess"),
        Param("exploration_fraction", "float", 0.25, min=0.0, max=1.0),
        Param("policy_index", "int", 0, min=0, help="Which Output is the policy head"),
    ],
    prelude=PRELUDE,
    runtime_init=lambda p, ins: (
        f"AlphaZeroSearch(model, env=None, simulations={int(p['simulations'])}, "
        f"c_puct={float(p['c_puct'])}, dirichlet_alpha={float(p['dirichlet_alpha'])}, "
        f"exploration_fraction={float(p['exploration_fraction'])}, "
        f"policy_index={int(p['policy_index'])})"
    ),
    runtime_name=lambda p: "search",
))
