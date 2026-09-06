#!/usr/bin/env python3
"""Train a policy-value network on 5x5 Go by self-play, then make it prove it.

    python play.py                    a short run
    python play.py --rounds 6         longer

The search teaches the network and the network guides the search. What gets
reported at the end is games won against a baseline, adjudicated by the rules
engine — not a loss curve. A network can improve its loss while playing worse,
so the loss is not the claim.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def evaluator_for(model, torch):
    """Wrap the network as (priors, value) for the search."""
    import microgo as go

    def evaluate(board):
        with torch.no_grad():
            x = torch.tensor([go.planes(board)], dtype=torch.float32)
            policy, value = model(x)
            priors = torch.softmax(policy[0], dim=-1).tolist()
            return priors, float(value[0][0])
    return evaluate


def self_play(model, torch, games: int, simulations: int, size: int,
              rng: random.Random):
    """Play against itself, keeping every position with its search result."""
    import microgo as go

    evaluate = evaluator_for(model, torch)
    states, targets, players = [], [], []
    finished = []
    for _ in range(games):
        board = go.Board(size)
        history = []
        moves = 0
        while not board.over and moves < size * size * 3:
            counts, _ = go.mcts(board, evaluate, simulations=simulations,
                                rng=rng, noise=0.25)
            total = sum(counts)
            if total <= 0:
                board = board.play(board.pass_move)
                continue
            policy = [c / total for c in counts]
            history.append((go.planes(board), policy, board.to_move))
            # sample early for variety, then take the best
            if moves < 6:
                move = rng.choices(range(len(counts)), weights=counts)[0]
            else:
                move = max(range(len(counts)), key=lambda i: counts[i])
            board = board.play(move)
            moves += 1

        score = board.score()
        winner = go.BLACK if score > 0 else go.WHITE
        finished.append(score)
        for plane, policy, mover in history:
            states.append(plane)
            targets.append(policy)
            players.append(1.0 if mover == winner else -1.0)
    return states, targets, players, finished


def duel(model, torch, size: int, games: int, simulations: int,
         rng: random.Random) -> int:
    """The network's search against a random-playout search of equal size.

    Colours alternate so neither side gets an advantage from moving first, and
    the rules engine decides every result.
    """
    import microgo as go

    learned = evaluator_for(model, torch)
    baseline = go.random_evaluator(rng)

    won = 0
    for game in range(games):
        board = go.Board(size)
        network_is_black = game % 2 == 0
        moves = 0
        while not board.over and moves < size * size * 3:
            mine = (board.to_move == go.BLACK) == network_is_black
            counts, _ = go.mcts(board, learned if mine else baseline,
                                simulations=simulations, rng=rng)
            if sum(counts) <= 0:
                board = board.play(board.pass_move)
            else:
                board = board.play(max(range(len(counts)),
                                       key=lambda i: counts[i]))
            moves += 1
        winner = go.BLACK if board.score() > 0 else go.WHITE
        if (winner == go.BLACK) == network_is_black:
            won += 1
    return won


def greedy_duel(model, torch, size: int, games: int, rng: random.Random) -> int:
    """The network's move choice alone, no search, against random legal moves.

    This separates two questions that the main duel runs together: has the
    network learned anything, and is it yet strong enough to beat a search that
    uses random playouts? The second is much harder than the first.
    """
    import microgo as go

    evaluate = evaluator_for(model, torch)
    won = 0
    for game in range(games):
        board = go.Board(size)
        network_is_black = game % 2 == 0
        moves = 0
        while not board.over and moves < size * size * 3:
            mine = (board.to_move == go.BLACK) == network_is_black
            # both sides choose from the same set, so neither is penalised for
            # a pass the other was not offered
            choices = go.sensible_moves(board)
            if mine:
                priors, _ = evaluate(board)
                move = max(choices, key=lambda m: priors[m])
            else:
                move = rng.choice(choices)
            board = board.play(move)
            moves += 1
        winner = go.BLACK if board.score() > 0 else go.WHITE
        if (winner == go.BLACK) == network_is_black:
            won += 1
    return won


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--simulations", type=int, default=40)
    ap.add_argument("--duels", type=int, default=20)
    ap.add_argument("--size", type=int, default=5)
    args = ap.parse_args()

    import torch

    import blockloader
    import codegen
    import graph as G
    import train as T

    blockloader.load_all()
    torch.manual_seed(0)
    rng = random.Random(0)

    graph = json.loads((HERE / "examples" / "MicroGo.json").read_text())
    g = G.parse(graph)
    report = G.analyze(g)
    if not report["ok"]:
        raise SystemExit(f"the design does not resolve: {report['errors'][:1]}")
    model = T.build_model(codegen.to_pytorch(g, report),
                          codegen.model_class_name(g))
    print(f"policy and value network: {report['total_learnables']:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    before = duel(model, torch, args.size, args.duels, args.simulations, rng)
    before_greedy = greedy_duel(model, torch, args.size, args.duels, rng)
    print(f"untrained   network alone vs random moves: {before_greedy}/{args.duels}")
    print(f"            with search vs equal-strength playout search: "
          f"{before}/{args.duels}")
    print()

    for round_number in range(args.rounds):
        model.eval()
        states, targets, outcomes, scores = self_play(
            model, torch, args.games, args.simulations, args.size, rng)
        x = torch.tensor(states, dtype=torch.float32)
        pi = torch.tensor(targets, dtype=torch.float32)
        z = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)

        model.train()
        order = torch.randperm(len(x))
        losses = []
        for start in range(0, len(x), 32):
            batch = order[start:start + 32]
            opt.zero_grad()
            policy, value = model(x[batch])
            # cross-entropy against the search's visit counts, and the value
            # against who actually won
            loss = (-(pi[batch] * torch.log_softmax(policy, dim=-1)).sum(1).mean()
                    + torch.nn.functional.mse_loss(value, z[batch]))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        black_share = sum(1 for s in scores if s > 0) / max(1, len(scores))
        print(f"  round {round_number + 1}: {len(x):4d} positions from "
              f"{args.games} games   loss {sum(losses)/len(losses):.3f}   "
              f"Black won {black_share:.0%} of self-play")

    print()
    after = duel(model, torch, args.size, args.duels, args.simulations, rng)
    after_greedy = greedy_duel(model, torch, args.size, args.duels, rng)
    print(f"trained     network alone vs random moves: "
          f"{before_greedy}/{args.duels} -> {after_greedy}/{args.duels}")
    print(f"            with search vs equal-strength playout search: "
          f"{before}/{args.duels} -> {after}/{args.duels}")
    print("            every game scored by the rules engine")
    torch.save(model.state_dict(), HERE / "microgo.pt")


if __name__ == "__main__":
    main()
