"""Small structured tasks. Deliberately not CIFAR: the question is whether a
structural guarantee changes solver and gradient behaviour, and a task the arms
all saturate on cannot answer it."""
import numpy as np
import torch


def two_moons(n, noise=0.15, seed=0):
    rng = np.random.default_rng(seed); k = n // 2
    t1, t2 = rng.uniform(0, np.pi, k), rng.uniform(0, np.pi, n - k)
    X = np.concatenate([np.stack([np.cos(t1), np.sin(t1)], 1),
                        np.stack([1 - np.cos(t2), .5 - np.sin(t2)], 1)])
    X += rng.normal(0, noise, X.shape)
    y = np.concatenate([np.zeros(k, int), np.ones(n - k, int)])
    p = rng.permutation(n)
    return torch.tensor(X[p]), torch.tensor(y[p]), 2, 2


def spirals(n, noise=0.08, seed=0):
    rng = np.random.default_rng(seed); k = n // 2
    t = np.sqrt(rng.uniform(.08, 1, k)) * 2.6 * np.pi
    t2 = np.sqrt(rng.uniform(.08, 1, n - k)) * 2.6 * np.pi
    X = np.concatenate([np.stack([t * np.cos(t), t * np.sin(t)], 1) / 8,
                        np.stack([-t2 * np.cos(t2), -t2 * np.sin(t2)], 1) / 8])
    X += rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(k, int), np.ones(n - k, int)])
    p = rng.permutation(n)
    return torch.tensor(X[p]), torch.tensor(y[p]), 2, 2


def checkerboard(n, noise=0.03, seed=0):
    """Four alternating cells. Needs more than a smooth boundary."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((np.floor((X[:, 0] + 1) * 2) + np.floor((X[:, 1] + 1) * 2)) % 2).astype(int)
    X += rng.normal(0, noise, X.shape)
    return torch.tensor(X), torch.tensor(y), 2, 2


TASKS = {'two_moons': two_moons, 'spirals': spirals, 'checkerboard': checkerboard}
