"""Go on a small board, with the rules done properly.

The same arrangement as microlean: the engine decides, the network proposes.
Legality, capture, ko and the final score are settled here, so a network that
suggests an illegal move simply does not get it played, and a win is a win
because the scoring said so.

Small boards are not a toy version of the rules — 5x5 and 7x7 Go are played, and
5x5 is solved (Black wins by 25 under area scoring with no komi). What is small
is the search space, which is what makes self-play affordable.

    >>> board = Board(5)
    >>> board = board.play(12)          # centre
    >>> board.to_move
    -1
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

BLACK, WHITE, EMPTY = 1, -1, 0


def neighbours(point: int, size: int) -> List[int]:
    row, col = divmod(point, size)
    out = []
    if row > 0:
        out.append(point - size)
    if row < size - 1:
        out.append(point + size)
    if col > 0:
        out.append(point - 1)
    if col < size - 1:
        out.append(point + 1)
    return out


@dataclass(frozen=True)
class Board:
    """A position. Immutable, so search can hold on to nodes safely."""

    size: int
    stones: Tuple[int, ...] = ()
    to_move: int = BLACK
    ko: Optional[int] = None          # the point forbidden by the ko rule
    passes: int = 0
    komi: float = 0.5                 # small, and fractional so nothing draws

    def __post_init__(self):
        if not self.stones:
            object.__setattr__(self, "stones", (EMPTY,) * (self.size * self.size))

    @property
    def area(self) -> int:
        return self.size * self.size

    @property
    def pass_move(self) -> int:
        return self.area                  # the last action index is "pass"

    def group(self, point: int) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """The stones connected to this one, and that group's liberties."""
        colour = self.stones[point]
        seen: Set[int] = {point}
        stack = [point]
        liberties: Set[int] = set()
        while stack:
            current = stack.pop()
            for other in neighbours(current, self.size):
                value = self.stones[other]
                if value == EMPTY:
                    liberties.add(other)
                elif value == colour and other not in seen:
                    seen.add(other)
                    stack.append(other)
        return frozenset(seen), frozenset(liberties)

    def _result_of(self, point: int) -> Optional[Tuple[Tuple[int, ...], int]]:
        """The board after playing here, and the single stone captured if any.

        None when the move is illegal. The captured-stone count is what the ko
        rule needs: a ko can only arise from a capture of exactly one stone.
        """
        if self.stones[point] != EMPTY:
            return None
        stones = list(self.stones)
        stones[point] = self.to_move

        captured: Set[int] = set()
        for other in neighbours(point, self.size):
            if stones[other] == -self.to_move:
                probe = Board(self.size, tuple(stones), self.to_move)
                group, liberties = probe.group(other)
                if not liberties:
                    captured |= group
        for dead in captured:
            stones[dead] = EMPTY

        if not captured:
            # suicide: illegal unless it took something
            probe = Board(self.size, tuple(stones), self.to_move)
            _, liberties = probe.group(point)
            if not liberties:
                return None
        return tuple(stones), (next(iter(captured)) if len(captured) == 1 else -1)

    def legal(self, move: int) -> bool:
        if move == self.pass_move:
            return True
        if not 0 <= move < self.area:
            return False
        if move == self.ko:
            return False
        return self._result_of(move) is not None

    def legal_moves(self, allow_pass: bool = True) -> List[int]:
        moves = [p for p in range(self.area) if self.legal(p)]
        if allow_pass:
            moves.append(self.pass_move)
        return moves

    def play(self, move: int) -> "Board":
        if move == self.pass_move:
            return Board(self.size, self.stones, -self.to_move, None,
                         self.passes + 1, self.komi)
        outcome = self._result_of(move)
        if outcome is None or move == self.ko:
            raise ValueError(f"illegal move {move}")
        stones, single = outcome

        # A ko is forbidden only when a lone stone was taken and the taker is
        # itself a lone stone with one liberty — otherwise recapture is fine.
        ko_point = None
        if single >= 0:
            probe = Board(self.size, stones, -self.to_move)
            group, liberties = probe.group(move)
            if len(group) == 1 and len(liberties) == 1:
                ko_point = single
        return Board(self.size, stones, -self.to_move, ko_point, 0, self.komi)

    @property
    def over(self) -> bool:
        return self.passes >= 2

    def score(self) -> float:
        """Area scoring: stones plus the empty regions they alone surround.

        Positive means Black is ahead, by that many points, komi included.
        """
        counts = {BLACK: 0.0, WHITE: 0.0}
        for value in self.stones:
            if value != EMPTY:
                counts[value] += 1

        seen: Set[int] = set()
        for point in range(self.area):
            if self.stones[point] != EMPTY or point in seen:
                continue
            region: Set[int] = {point}
            stack = [point]
            borders: Set[int] = set()
            while stack:
                current = stack.pop()
                for other in neighbours(current, self.size):
                    value = self.stones[other]
                    if value == EMPTY:
                        if other not in region:
                            region.add(other)
                            stack.append(other)
                    else:
                        borders.add(value)
            seen |= region
            if len(borders) == 1:
                counts[borders.pop()] += len(region)
        return counts[BLACK] - counts[WHITE] - self.komi

    def winner(self) -> int:
        return BLACK if self.score() > 0 else WHITE

    def show(self) -> str:
        glyph = {BLACK: "X", WHITE: "O", EMPTY: "."}
        rows = []
        for r in range(self.size):
            rows.append(" ".join(glyph[self.stones[r * self.size + c]]
                                 for c in range(self.size)))
        return "\n".join(rows)


# --------------------------------------------------------------------------
# what the network sees
# --------------------------------------------------------------------------

def planes(board: Board) -> List[List[List[float]]]:
    """Three planes: my stones, their stones, and whose turn it is.

    Always from the point of view of the player to move, so one network serves
    both colours and never has to be told which it is playing.
    """
    size = board.size
    mine = [[0.0] * size for _ in range(size)]
    theirs = [[0.0] * size for _ in range(size)]
    for point, value in enumerate(board.stones):
        row, col = divmod(point, size)
        if value == board.to_move:
            mine[row][col] = 1.0
        elif value == -board.to_move:
            theirs[row][col] = 1.0
    turn = [[1.0 if board.to_move == BLACK else 0.0] * size for _ in range(size)]
    return [mine, theirs, turn]


def rollout(board: Board, rng: random.Random, limit: int = 200) -> float:
    """Play on at random to the end. Returns the score from Black's side."""
    guard = 0
    while not board.over and guard < limit:
        guard += 1
        moves = [m for m in board.legal_moves(allow_pass=False)
                 if not _fills_own_eye(board, m)]
        board = board.play(rng.choice(moves) if moves else board.pass_move)
    return board.score()


def _fills_own_eye(board: Board, move: int) -> bool:
    """Playing into a point surrounded by your own stones ends random games.

    Without this a random playout fills its own eyes and kills its own groups,
    which makes the result meaningless as an estimate.
    """
    for other in neighbours(move, board.size):
        if board.stones[other] != board.to_move:
            return False
    return True


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

class Node:
    """One position in the tree, with the statistics PUCT needs."""

    __slots__ = ("board", "prior", "visits", "total", "children")

    def __init__(self, board: Board, prior: float = 0.0):
        self.board = board
        self.prior = prior
        self.visits = 0
        self.total = 0.0          # summed value, from the mover's point of view
        self.children: Dict[int, "Node"] = {}

    @property
    def value(self) -> float:
        return self.total / self.visits if self.visits else 0.0


def mcts(board: Board, evaluate, simulations: int = 100, c_puct: float = 1.4,
         rng: Optional[random.Random] = None, noise: float = 0.0):
    """PUCT search. `evaluate(board)` returns (priors over actions, value).

    The value is always from the point of view of the player to move, so it is
    negated on the way back up the tree — the same position is good for one
    player exactly as much as it is bad for the other.

    Returns visit counts per action, which is what a policy head is trained
    against: the search is the teacher, the network is the student.
    """
    rng = rng or random.Random()
    root = Node(board)
    priors, _ = evaluate(board)
    _expand(root, priors, rng, noise)

    for _ in range(simulations):
        node = root
        path = [node]
        while node.children and not node.board.over:
            move, node = _select(node, c_puct)
            path.append(node)

        if node.board.over:
            # a finished game is worth its result, not an estimate of it
            score = node.board.score()
            value = 1.0 if (score > 0) == (node.board.to_move == BLACK) else -1.0
        else:
            priors, value = evaluate(node.board)
            _expand(node, priors, rng, 0.0)

        for entry in reversed(path):
            entry.visits += 1
            entry.total += value
            value = -value

    counts = [0.0] * (board.area + 1)
    for move, child in root.children.items():
        counts[move] = float(child.visits)
    return counts, root


def sensible_moves(board: Board) -> List[int]:
    """Legal moves worth searching.

    Passing stays legal — the rules are the rules — but a search that considers
    it at every node gives it prior mass in every position, and a policy trained
    on those visits learns to pass on move one. So it is only offered when there
    is nothing else to play, which is also when a human would consider it.
    """
    playable = [m for m in board.legal_moves(allow_pass=False)
                if not _fills_own_eye(board, m)]
    return playable or [board.pass_move]


def _expand(node: Node, priors: Sequence[float], rng, noise: float) -> None:
    legal = sensible_moves(node.board)
    if not legal:
        return
    weights = [max(1e-8, priors[m]) if m < len(priors) else 1e-8 for m in legal]
    if noise > 0:
        # Dirichlet at the root, so self-play explores rather than repeating
        draw = [rng.gammavariate(0.3, 1.0) for _ in legal]
        total = sum(draw) or 1.0
        weights = [(1 - noise) * w + noise * (d / total)
                   for w, d in zip(weights, draw)]
    total = sum(weights) or 1.0
    for move, weight in zip(legal, weights):
        node.children[move] = Node(node.board.play(move), weight / total)


def _select(node: Node, c_puct: float):
    import math

    root_visits = math.sqrt(max(1, node.visits))
    best, best_score = None, -1e30
    for move, child in node.children.items():
        # the child's value is from the opponent's side, hence the minus
        score = -child.value + c_puct * child.prior * root_visits / (1 + child.visits)
        if score > best_score:
            best, best_score = (move, child), score
    return best


def random_evaluator(rng: random.Random):
    """Uniform priors and a random-playout value: the baseline to beat."""
    def evaluate(board: Board):
        priors = [1.0] * (board.area + 1)
        score = rollout(board, rng)
        value = 1.0 if (score > 0) == (board.to_move == BLACK) else -1.0
        return priors, value
    return evaluate
