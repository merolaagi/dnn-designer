"""Agents that run experiments.

Not agents in the sense of calling a language model inside a workflow — nothing
here talks to a model. An agent is a loop that proposes variants of your
network, trains each one through the ordinary machinery, and reports which won.
That is the automation this app is actually short of: the runs, the recipes and
the reviewer all exist, but somebody still has to sit there changing one number
and pressing Train.

Three kinds:

  sweep      vary training settings — learning rate, batch size, optimizer —
             leaving the architecture alone
  search     vary the architecture — width, depth, regularization — leaving the
             training settings alone
  repair     take what `review` found and apply the fixes one at a time, so you
             can see which of them actually helped

Every trial is an ordinary run: it appears in the run history, writes
checkpoints, and can be reopened. The agent adds a leaderboard over the top.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import assistant
import codegen
import graph as G
import train as T

AGENTS: Dict[str, "Agent"] = {}
_LOCK = threading.Lock()

def studies_dir() -> Path:
    import auth

    return auth.sub("studies")


# --------------------------------------------------------------------------
# proposing variants
# --------------------------------------------------------------------------

SWEEP_SPACE = {
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "batch_size": [16, 32, 64, 128],
    "optimizer": ["adamw", "adam", "sgd"],
}


def sweep_trials(graph: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Combinations of training settings, the architecture untouched."""
    chosen = {k: v for k, v in SWEEP_SPACE.items() if k in (cfg.get("vary") or ["lr"])}
    if not chosen:
        chosen = {"lr": SWEEP_SPACE["lr"]}
    grid = [dict(zip(chosen, values))
            for values in itertools.product(*chosen.values())]
    limit = int(cfg.get("trials", 6))
    if len(grid) > limit:
        random.Random(cfg.get("seed", 0)).shuffle(grid)
        grid = grid[:limit]
    learnables = G.analyze(G.parse(copy.deepcopy(graph)))["total_learnables"]
    return [{"label": ", ".join(f"{k}={v}" for k, v in point.items()),
             "graph": copy.deepcopy(graph), "config": point, "change": point,
             "learnables": learnables}
            for point in grid]


def _scale_width(graph: Dict[str, Any], factor: float) -> Optional[str]:
    """Multiply every layer's width. Returns a description, or None if nothing moved."""
    touched = []
    for node in graph["nodes"]:
        for key in ("filters", "units"):
            if key in node["params"] and isinstance(node["params"][key], int):
                # the final head sets the answer's size, so it must not move
                if node.get("label") in ("head", "lm_head"):
                    continue
                before = node["params"][key]
                after = max(1, int(round(before * factor)))
                if after != before:
                    node["params"][key] = after
                    touched.append(node["type"])
    return f"width x{factor:g} ({len(touched)} layers)" if touched else None


def _add_regularization(graph: Dict[str, Any], rate: float) -> Optional[str]:
    existing = [n for n in graph["nodes"] if n["type"] in ("Dropout", "Dropout2d")]
    if existing:
        for node in existing:
            node["params"]["rate"] = rate
        return f"dropout rate {rate}"
    # put one before the head, which is where it does the most good
    heads = [n for n in graph["nodes"] if n["type"] == "Linear"]
    if not heads:
        return None
    head = heads[-1]
    feeding = [e for e in graph["edges"] if e["target"] == head["id"]]
    if not feeding:
        return None
    nid = "ag" + str(len(graph["nodes"]) + 1)
    graph["nodes"].append({"id": nid, "type": "Dropout", "params": {"rate": rate},
                           "label": "", "x": head.get("x", 0),
                           "y": head.get("y", 0) - 100})
    source = feeding[0]["source"]
    feeding[0]["source"] = nid
    graph["edges"].append({"id": nid + "e", "source": source, "target": nid, "port": 0})
    return f"added dropout {rate} before the head"


def _add_normalization(graph: Dict[str, Any]) -> Optional[str]:
    added = 0
    for node in list(graph["nodes"]):
        if node["type"] != "Conv2d":
            continue
        following = [e for e in graph["edges"] if e["source"] == node["id"]]
        kinds = {n["type"] for n in graph["nodes"]
                 for e in following if n["id"] == e["target"]}
        if kinds & {"BatchNorm2d", "GroupNorm"}:
            continue
        nid = "an" + str(len(graph["nodes"]) + 1 + added)
        graph["nodes"].append({"id": nid, "type": "BatchNorm2d", "params": {},
                               "label": "", "x": node.get("x", 0),
                               "y": node.get("y", 0) + 60})
        for e in following:
            e["source"] = nid
        graph["edges"].append({"id": nid + "e", "source": node["id"],
                               "target": nid, "port": 0})
        added += 1
    return f"batch norm after {added} convolutions" if added else None


def search_trials(graph: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Architecture variants, the training settings untouched."""
    proposals = []

    def offer(label, mutate):
        candidate = copy.deepcopy(graph)
        described = mutate(candidate)
        if not described:
            return
        report = G.analyze(G.parse(candidate))
        if not report["ok"]:
            return                       # a variant that will not build is not a variant
        proposals.append({"label": label, "graph": candidate, "config": {},
                          "change": {"architecture": described},
                          "learnables": report["total_learnables"]})

    baseline = G.analyze(G.parse(copy.deepcopy(graph)))
    proposals.append({"label": "unchanged", "graph": copy.deepcopy(graph),
                      "config": {}, "change": {"architecture": "the network as drawn"},
                      "learnables": baseline["total_learnables"]})
    for factor in (0.5, 1.5, 2.0):
        offer(f"width x{factor:g}", lambda g, f=factor: _scale_width(g, f))
    for rate in (0.1, 0.3, 0.5):
        offer(f"dropout {rate}", lambda g, r=rate: _add_regularization(g, r))
    offer("with batch norm", _add_normalization)

    limit = int(cfg.get("trials", 6))
    return proposals[:limit]


def repair_trials(graph: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One trial per fix the reviewer suggested, plus the network as it stands."""
    g = G.parse(graph)
    report = G.analyze(g)
    findings = [f for f in assistant.review(g, report) if f["level"] in ("warn", "note")]

    proposals = [{"label": "as drawn", "graph": copy.deepcopy(graph), "config": {},
                  "change": {"fix": "nothing changed, for comparison"},
                  "learnables": report["total_learnables"]}]

    for finding in findings:
        candidate = copy.deepcopy(graph)
        text = finding["text"]
        described = None
        if "compose to a single linear layer" in text:
            described = _split_linears(candidate)
        elif "GlobalAvgPool instead" in text:
            described = _flatten_to_pool(candidate)
        elif "BatchNorm2d between the two" in text:
            described = _add_normalization(candidate)
        elif "no Dropout anywhere" in text:
            described = _add_regularization(candidate, 0.3)
        if not described:
            continue
        if not G.analyze(G.parse(candidate))["ok"]:
            continue
        proposals.append({"label": described, "graph": candidate, "config": {},
                          "change": {"fix": text},
                          "learnables": G.analyze(G.parse(candidate))["total_learnables"]})
    return proposals[: int(cfg.get("trials", 6))]


def _split_linears(graph: Dict[str, Any]) -> Optional[str]:
    """Put an activation between two dense layers that have none."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    for edge in list(graph["edges"]):
        a, b = by_id.get(edge["source"]), by_id.get(edge["target"])
        if a and b and a["type"] == "Linear" and b["type"] == "Linear":
            nid = "af" + str(len(graph["nodes"]) + 1)
            graph["nodes"].append({"id": nid, "type": "Activation",
                                   "params": {"kind": "relu"}, "label": "",
                                   "x": a.get("x", 0), "y": a.get("y", 0) + 60})
            edge["target"] = nid
            graph["edges"].append({"id": nid + "e", "source": nid,
                                   "target": b["id"], "port": 0})
            return "activation between the dense layers"
    return None


def _flatten_to_pool(graph: Dict[str, Any]) -> Optional[str]:
    for node in graph["nodes"]:
        if node["type"] == "Flatten":
            node["type"] = "GlobalAvgPool"
            node["params"] = {}
            return "global pooling instead of flatten"
    return None




def ablation_trials(graph: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One trial per arm of an implicit domain, on *this* network.

    The bench compares arms on its own fixed model, which answers whether the
    structure helps in general. Running the same comparison here answers a
    different and often more useful question: whether it helps in the
    architecture you actually have.

    Every arm is the same design with one layer's variant swapped, so the
    parameter counts match by construction — which is what makes a control a
    control rather than a smaller model.
    """
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        return []

    nodes = graph.get("nodes", [])
    targets = [n for n in nodes if n.get("type") == "ImplicitEquilibrium"]
    if not targets:
        return []

    which = str(cfg.get("layer") or targets[0]["id"])
    target = next((n for n in targets if n["id"] == which), targets[0])
    domain = str(cfg.get("domain")
                 or (target.get("params") or {}).get("domain") or "crn")
    try:
        spec = core.get(domain)
        arms = spec.variants(spec.defaults(), seed=int(cfg.get("seed", 0)))
    except Exception:  # noqa: BLE001
        return []

    trials = []
    for arm in arms:
        variant = copy.deepcopy(graph)
        for node in variant.get("nodes", []):
            if node["id"] == target["id"]:
                node.setdefault("params", {})["domain"] = domain
                node["params"]["variant"] = arm.key
        try:
            report = G.analyze(G.parse(copy.deepcopy(variant)))
            learnables = report["total_learnables"] if report["ok"] else None
        except Exception:  # noqa: BLE001
            learnables = None
        holds = False
        try:
            holds = bool((arm.build().certificate() or {}).get("holds"))
        except Exception:  # noqa: BLE001
            pass
        trials.append({
            "label": f"{domain}/{arm.key}",
            "graph": variant,
            "change": {"domain": domain, "variant": arm.key},
            "learnables": learnables,
            "note": arm.label,
            "structured": bool(arm.structured),
            "guarantee": holds,
        })
    return trials


BUILDERS = {"sweep": sweep_trials, "search": search_trials,
            "repair": repair_trials, "ablation": ablation_trials}

CATALOG = [
    {"id": "ablation", "name": "Does the guarantee help?",
     "doc": "Trains this network once per arm of an implicit layer's domain: "
            "the arm the theorem covers, and the controls that break one of its "
            "conditions at the same parameter count. Reports whether the "
            "guarantee bought anything here.",
     "params": [{"name": "epochs", "kind": "int", "default": 6},
                {"name": "seed", "kind": "int", "default": 0}]},
    {"id": "sweep", "name": "Hyperparameter sweep",
     "doc": "Trains the same network several times with different learning rates, "
            "batch sizes or optimizers, and reports which settings won.",
     "params": [{"name": "vary", "kind": "multi", "default": ["lr"],
                 "options": sorted(SWEEP_SPACE)},
                {"name": "trials", "kind": "int", "default": 6},
                {"name": "epochs", "kind": "int", "default": 4}]},
    {"id": "search", "name": "Architecture search",
     "doc": "Proposes wider, narrower and more regularized versions of the network, "
            "trains each, and reports which shape won. Variants that do not build "
            "are discarded before anything is trained.",
     "params": [{"name": "trials", "kind": "int", "default": 6},
                {"name": "epochs", "kind": "int", "default": 4}]},
    {"id": "repair", "name": "Try the review's fixes",
     "doc": "Takes what review found, applies each fix on its own, and trains them "
            "side by side against the network as drawn — so you find out which of "
            "them actually helped rather than assuming.",
     "params": [{"name": "trials", "kind": "int", "default": 6},
                {"name": "epochs", "kind": "int", "default": 4}]},
]


# --------------------------------------------------------------------------
# running a study
# --------------------------------------------------------------------------

@dataclass
class Agent:
    id: str
    kind: str
    design: str
    status: str = "planning"      # planning | running | done | error | stopped
    trials: List[Dict[str, Any]] = field(default_factory=list)
    at: int = 0
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    objective: str = "val_loss"
    lower_is_better: bool = True
    home: Optional[Path] = None      # captured at start, as for a training job
    stop: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> Dict[str, Any]:
        ranked = [t for t in self.trials if t.get("score") is not None]
        ranked.sort(key=lambda t: t["score"], reverse=not self.lower_is_better)
        return {
            "id": self.id, "kind": self.kind, "design": self.design,
            "status": self.status, "at": self.at, "total": len(self.trials),
            "error": self.error, "started": self.started, "finished": self.finished,
            "objective": self.objective, "lower_is_better": self.lower_is_better,
            "trials": self.trials,
            "leader": ranked[0] if ranked else None,
            "verdict": self._verdict(ranked),
        }

    def _verdict(self, ranked) -> Optional[str]:
        """For an ablation, what the ranking is and is not evidence for.

        Beating one control is not beating the condition: whatever the other
        control preserves may be what actually mattered. Saying so is the
        difference between a comparison and a result.
        """
        if self.kind != "ablation":
            return None
        scored = [t for t in self.trials if t.get("score") is not None]
        covered = [t for t in scored if t.get("guarantee")]
        controls = [t for t in scored if not t.get("guarantee")]
        if not covered or not controls:
            return None
        best = covered[0]
        def wins(against) -> bool:
            return (best["score"] < against["score"] if self.lower_is_better
                    else best["score"] > against["score"])

        beaten = [c for c in controls if wins(c)]
        if len(beaten) == len(controls):
            return (f"The covered arm beat every control on {self.objective}, at "
                    f"the same parameter count. That is what the guarantee "
                    f"buying something looks like — on this network, this task "
                    f"and these seeds.")
        if not beaten:
            return (f"The covered arm beat no control on {self.objective}. The "
                    f"guarantee cost nothing here, but it bought nothing either.")
        kept = [c["label"] for c in controls if c not in beaten]
        return (f"The covered arm beat {len(beaten)} of {len(controls)} controls "
                f"on {self.objective}, but not {', '.join(kept)}. Beating one "
                f"control is not beating the condition — whatever "
                f"{kept[0]} preserves may be what actually matters.")

    def persist(self) -> None:
        try:
            target = self.home or studies_dir()
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{self.id}.json").write_text(
                json.dumps(self.snapshot(), indent=1))
        except Exception:  # noqa: BLE001
            pass


def _score_of(history: List[Dict[str, Any]], lower: bool) -> Optional[float]:
    values = [row.get("val_loss") for row in history if row.get("val_loss") is not None]
    if not values:
        return None
    return min(values) if lower else max(values)


def _run_study(agent: Agent, base_config: Dict[str, Any]) -> None:
    try:
        agent.status = "running"
        for index, trial in enumerate(agent.trials):
            if agent.stop.is_set():
                break
            agent.at = index
            trial["status"] = "running"
            agent.persist()

            g = G.parse(trial["graph"])
            report = G.analyze(g)
            if not report["ok"]:
                trial.update(status="skipped",
                             note="this variant does not resolve")
                continue

            ids = codegen.input_order(g, report)
            nodes = g.by_id()
            outs = [i for i in report["order"] if nodes[i].type == "Output"]
            tasks = [G.resolved_params(nodes[i]).get("task", "classification")
                     for i in outs] or ["classification"]

            config = dict(base_config)
            config.update(trial.get("config") or {})
            config["graph"] = trial["graph"]
            config["design_name"] = f"{agent.design} · {trial['label']}"
            config["save_checkpoints"] = True

            job = T.start(
                codegen.to_pytorch(g, report), config,
                [report["nodes"][i]["out_shape"] for i in ids], ids,
                report["nodes"][outs[0]]["out_shape"] if outs else None,
                tasks, codegen.model_class_name(g))
            trial["run"] = job.id
            trial["learnables"] = report["total_learnables"]

            while True:
                event = job.events.get(timeout=3600)
                if event["kind"] == "epoch":
                    trial["epoch"] = event["epoch"]
                    trial["last"] = event.get("val_loss")
                    agent.persist()
                if event["kind"] == "error":
                    trial.update(status="error", note=event["message"])
                    break
                if event["kind"] == "finished":
                    agent.objective = (job.history[-1].get("objective", "val_loss")
                                       if job.history else "val_loss")
                    agent.lower_is_better = agent.objective in ("val_loss", "loss")
                    trial.update(status="done",
                                 score=_score_of(job.history, agent.lower_is_better),
                                 epochs=len(job.history))
                    break
                if agent.stop.is_set():
                    job.stop.set()
            agent.persist()

        agent.at = len(agent.trials)
        agent.status = "stopped" if agent.stop.is_set() else "done"
    except Exception as exc:  # noqa: BLE001
        agent.status = "error"
        agent.error = f"{type(exc).__name__}: {exc}"
    finally:
        agent.finished = time.time()
        agent.persist()


def start(kind: str, graph: Dict[str, Any], config: Dict[str, Any]) -> Agent:
    if kind not in BUILDERS:
        raise ValueError(f"no agent called {kind}")
    trials = BUILDERS[kind](copy.deepcopy(graph), config)
    for trial in trials:
        trial.setdefault("status", "waiting")
    agent = Agent(home=studies_dir(), id=uuid.uuid4().hex[:10], kind=kind,
                  design=graph.get("name") or "unsaved", trials=trials)

    base = {k: v for k, v in config.items()
            if k not in ("vary", "trials", "agent")}
    base.setdefault("epochs", 4)
    with _LOCK:
        AGENTS[agent.id] = agent
    agent.persist()
    threading.Thread(target=_run_study, args=(agent, base), daemon=True).start()
    return agent


def listing() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(studies_dir().glob("*.json"), key=lambda p: -p.stat().st_mtime)[:60]:
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        live = AGENTS.get(blob.get("id"))
        if live:
            blob = live.snapshot()
        out.append({k: blob.get(k) for k in
                    ("id", "kind", "design", "status", "at", "total",
                     "started", "finished", "objective")}
                   | {"leader": (blob.get("leader") or {}).get("label")})
    return out


def read(agent_id: str) -> Dict[str, Any]:
    live = AGENTS.get(agent_id)
    if live:
        return live.snapshot()
    path = studies_dir() / f"{Path(agent_id).name}.json"
    if not path.exists():
        raise KeyError(agent_id)
    return json.loads(path.read_text())
