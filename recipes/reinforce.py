"""Policy gradients: the network makes its own training data.

There is no dataset here. The model acts in an environment, the environment
answers, and the resulting trajectory *is* the batch. That is why this recipe is
self-supplied — nothing a DataLoader can express describes a rollout.

The canvas graph is the policy: observation in, one logit per action out. A
CartPole environment is built in so the recipe runs out of the box; swap
`Environment` below for your own and the rest is unchanged.
"""

import math
import random

from recipes_sdk import Param, Recipe, install


class CartPole:
    """The classic balancing task, in plain Python so there is no dependency.

    Observation is [position, velocity, angle, angular velocity]. Two actions:
    push left or right. Reward is 1 per surviving step, so the return is simply
    how long the pole stayed up.
    """

    observations = 4
    actions = 2

    gravity, cart_mass, pole_mass, half_length = 9.8, 1.0, 0.1, 0.5
    force, dt = 10.0, 0.02
    angle_limit = 12 * math.pi / 180
    position_limit = 2.4

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state = [random.uniform(-0.05, 0.05) for _ in range(4)]
        self.steps = 0
        return list(self.state)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        total_mass = self.cart_mass + self.pole_mass
        pole_moment = self.pole_mass * self.half_length
        push = self.force if action == 1 else -self.force

        cos, sin = math.cos(theta), math.sin(theta)
        temp = (push + pole_moment * theta_dot ** 2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4.0 / 3.0 - self.pole_mass * cos ** 2 / total_mass))
        x_acc = temp - pole_moment * theta_acc * cos / total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc
        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

        done = (abs(x) > self.position_limit
                or abs(theta) > self.angle_limit
                or self.steps >= self.max_steps)
        return list(self.state), 1.0, done


ENVIRONMENTS = {"cartpole": CartPole}


def check(ctx):
    env = ENVIRONMENTS[ctx.cfg["environment"]]
    if list(ctx.in_shapes[0]) != [env.observations]:
        return (f"{ctx.cfg['environment']} gives {env.observations} observations, "
                f"so the Input should be [{env.observations}], not {ctx.in_shapes[0]}.")
    if not ctx.out_shape or ctx.out_shape[-1] != env.actions:
        return (f"{ctx.cfg['environment']} has {env.actions} actions, so the last "
                f"layer should output {env.actions} units, not "
                f"{ctx.out_shape[-1] if ctx.out_shape else '?'}.")
    return None


def setup(ctx):
    import torch

    ctx.optimizers["main"] = torch.optim.Adam(
        ctx.parameters(), lr=float(ctx.cfg["lr"]))
    ctx.state["env"] = ENVIRONMENTS[ctx.cfg["environment"]](
        max_steps=int(ctx.cfg["max_steps"]))
    ctx.state["baseline"] = 0.0
    ctx.state["recent"] = []


def _rollout(ctx):
    """Play one episode, keeping the log-probability of every action taken."""
    import torch

    env = ctx.state["env"]
    obs = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=ctx.device).unsqueeze(0)
        logits = ctx.model(x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action).squeeze(0))
        obs, reward, done = env.step(int(action.item()))
        rewards.append(reward)
    return log_probs, rewards


def step(ctx, xs, y):
    import torch

    gamma = float(ctx.cfg["gamma"])
    log_probs, rewards = _rollout(ctx)

    # discounted return from each step onward
    returns, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    returns = torch.tensor(returns, dtype=torch.float32, device=ctx.device)

    # A moving baseline. Without it every action in a successful episode is
    # reinforced equally, including the bad ones, and learning is very slow.
    baseline = ctx.state["baseline"]
    ctx.state["baseline"] = 0.95 * baseline + 0.05 * float(returns.mean())
    advantage = returns - baseline
    if advantage.numel() > 1:
        advantage = advantage / (advantage.std() + 1e-6)

    loss = -(torch.stack(log_probs) * advantage).sum()
    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if float(ctx.cfg["clip"]) > 0:
        torch.nn.utils.clip_grad_norm_(ctx.parameters(), float(ctx.cfg["clip"]))
    opt.step()

    total = float(sum(rewards))
    ctx.state["recent"].append(total)
    ctx.state["recent"] = ctx.state["recent"][-100:]
    return {"loss": float(loss.item()), "return": total,
            "episode_length": float(len(rewards))}


def preview(ctx):
    recent = ctx.state.get("recent") or [0.0]
    window = recent[-20:]
    return (f"last {len(window)} episodes · mean return {sum(window)/len(window):.1f} · "
            f"best {max(recent):.0f} · worst {min(window):.0f}")


install(Recipe(
    name="Reinforce",
    doc="Policy gradients. The network chooses actions, the environment answers, "
        "and the episode becomes the batch — so no dataset is used at all. The "
        "canvas graph is the policy: observation in, one logit per action out. "
        "Watch the return rather than the loss; the loss in policy gradients is "
        "not something that meaningfully goes down.",
    params=[
        Param("environment", "select", "cartpole", options=sorted(ENVIRONMENTS)),
        Param("lr", "float", 0.01),
        Param("gamma", "float", 0.99, min=0.5, max=1.0, help="Discount on future reward"),
        Param("max_steps", "int", 500, min=10, help="Episode cap; also the best possible return"),
        Param("clip", "float", 1.0, min=0.0, help="Gradient norm clip, 0 to disable"),
        Param("steps_per_epoch", "int", 40, min=1, help="Episodes per epoch"),
    ],
    accepts=["none"],
    self_supplied=True,
    steps_per_epoch=40,
    objective="return",
    lower_is_better=False,
    setup=setup, step=step, preview=preview, check=check,
))
