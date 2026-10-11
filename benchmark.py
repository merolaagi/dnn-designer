"""The system benchmark: real tasks for the lab's agents, run on the models you choose.

A study asks whether one design gets better when a block changes. The benchmark asks the
question underneath: does the system work, and how much of a failure is the model's? The same
tasks run on two or three models, each task checked automatically (no judge model), and every
trial is sorted into passed, wrong answer, missing tool use or error.

A baseline records one model's result. `python3 benchmark.py gate`, which release.sh runs,
refuses a release when a benchmark of this version on the baseline's model scores lower than
the baseline by more than its noise.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "benchmarks" / "fixtures"

# Each task: which starting design runs it, the task, and checks in the cases format (one per line).
SUITE: List[Dict[str, Any]] = [
    {"id": "math-primes", "agent": "mathsearch", "skill": "verified computation",
     "task": "Count the prime numbers below 10,000. Write Python that counts them two independent ways (a sieve, "
             "and trial division) and prints `VERIFIED score=<count>` only if both counts agree. Report the count "
             "and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=1229\\b\ncontains: 1229"},
    {"id": "math-golomb-6", "agent": "mathsearch", "skill": "search, verified",
     "task": "Find a Golomb ruler with 6 marks that is as short as possible: integers starting at 0 whose pairwise "
             "differences are all distinct; its length is the largest mark. Write Python that searches for one and "
             "checks it from scratch (6 distinct marks, the smallest 0, all 15 differences distinct), printing "
             "`VERIFIED score=<length>` only if every check holds. Each run must finish within 60 seconds. Report "
             "the ruler, its length and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=17\\b"},
    {"id": "math-no-three-6", "agent": "mathsearch", "skill": "search, verified",
     "task": "Place as many points as possible on a 6 by 6 grid so that no three lie on one straight line, in any "
             "direction. Write Python that finds a placement and checks it from scratch (every point inside the "
             "grid, none repeated, no three collinear by an exact integer cross product), printing "
             "`VERIFIED score=<number of points>` only if every check holds. Each run must finish within 60 "
             "seconds. Report the points, the score and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=12\\b"},
    {"id": "model-synthetic", "agent": "modelbuilder", "skill": "design, train, package",
     "task": "Build a small classifier for the built-in synthetic dataset with fewer than 5,000 learnable "
             "parameters. Check the design, train it for 2 epochs on dataset synthetic, then package the trained "
             "network and give the download link.",
     "checks": "tool: check_network\ntool: train_network\nresult: Trained .+?: done\nresult: \"packaged\": true"},
    {"id": "model-design-only", "agent": "modelbuilder", "skill": "design within a budget",
     "task": "Design a convolutional network for fashion_mnist (28 by 28 grey images, 10 classes) with fewer than "
             "20,000 learnable parameters, and check it. Do not train it. Report its name, version and learnable "
             "parameter count as check_network reported it.",
     "checks": "tool: design_network\ntool: check_network\nresult: \"saved\": true"},
    {"id": "reverse-notes", "agent": "reverse", "skill": "reverse engineering (REA)",
     "task": "Analyze the Electron app at {fixtures}/notes-app. How does its window search notes, and how does it "
             "copy a note? Name the IPC channels and the preload APIs involved, with the file and line for each.",
     "checks": "tool: analyze_javascript_application\ncontains: notes:search\ncontains: clipboard:write\n"
               "contains: preload.js"},
    {"id": "specialist-loan", "agent": "specialist", "skill": "arithmetic with code",
     "task": "Work out the monthly payment on a $25,000 loan at 6.5% a year over 5 years, paid monthly. Give the "
             "formula, and check the number with code.",
     "checks": "tool: run_python\nregex: 489\\.1[45]"},
    {"id": "specialist-search", "agent": "specialist", "skill": "web search with sources",
     "task": "Who is the Golomb ruler named after, and at which university did he spend most of his career? "
             "Search the web and give the addresses of your sources.",
     "checks": "tool: web_search\ncontains: Solomon\ncontains: Southern California\nregex: https?://"},
    {"id": "react-french", "agent": "react", "skill": "following the user's language",
     "task": "Réponds en français : combien font 17 × 23 ? Explique le calcul en une phrase.",
     "checks": "contains: 391\nregex: (?i)\\b(est|font|donc|fait|multipli)"},
]


def suite_cases(ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The tasks, parsed, with the fixtures folder filled in."""
    out = []
    for t in SUITE:
        if ids and t["id"] not in ids:
            continue
        text = t["task"].replace("{fixtures}", str(FIXTURES)) + "\n" + t["checks"]
        parsed = AL.parse_cases(text)
        if parsed["problems"]:
            raise ValueError(f"benchmark task {t['id']}: {parsed['problems'][0]}")
        out.append({**t, "task": parsed["cases"][0]["task"], "case": parsed["cases"][0]})
    return out


def with_model(graph, provider: str, model: str) -> Dict[str, Any]:
    """The design with every LLM core pointed at one provider and model."""
    g = json.loads(json.dumps(graph))
    for n in g["nodes"]:
        if n["type"] == "llm":
            n["params"] = {**n.get("params", {}), "provider": provider, "model": model}
    return g


def classify(checks: List[Dict[str, Any]], result: Dict[str, Any]) -> str:
    if result.get("error"):
        return "error"
    failed = [c for c in checks if c.get("passed") is False]
    if not failed:
        return "passed"
    return "no tool use" if any(c["kind"] == "tool" for c in failed) else "wrong answer"


def run_trial(task: Dict[str, Any], provider: str, model: str, mode: str) -> Dict[str, Any]:
    graph = with_model(AL.template(task["agent"]), provider, model)
    started = time.time()
    with tempfile.TemporaryDirectory() as memory:      # every trial starts with no remembered notes
        result = AL.run(graph, task["task"], mode, "approve", memory_dir=Path(memory), keep=True, source="benchmark")
    checks = AL.score(result, task["case"]["checks"], task["case"], mode)
    c = result.get("costs") or {}
    graded = [x for x in checks if x.get("passed") is not None]
    return {"task": task["id"], "outcome": classify(checks, result), "passed": all(x["passed"] for x in graded),
            "score": round(sum(1 for x in graded if x["passed"]) / len(graded), 4) if graded else 0.0,
            "checks": [{"check": f"{x['kind']}: {x['value']}", "passed": x["passed"]} for x in checks],
            "error": result.get("error"), "seconds": round(time.time() - started, 1),
            "tokens": (c.get("input_total") or 0) + (c.get("output_total") or 0),
            "calls": c.get("calls") or 0, "answer": (result.get("answer") or "")[:1500],
            "run": result.get("run_id")}


def summarise(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trials)
    if not n:
        return {"trials": 0}
    rate = sum(t["passed"] for t in trials) / n
    outcomes: Dict[str, int] = {}
    for t in trials:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    return {"trials": n, "pass_rate": round(rate, 4), "score": round(sum(t["score"] for t in trials) / n, 4),
            "noise": round(2 * math.sqrt(max(rate * (1 - rate), 0.25 / n) / n), 4),
            "seconds": round(sum(t["seconds"] for t in trials) / n, 1),
            "tokens": round(sum(t["tokens"] for t in trials) / n), "outcomes": outcomes}


# --------------------------------------------------------------------------
# runs, kept in the workspace
# --------------------------------------------------------------------------

_LIVE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _dir() -> Path:
    path = AL._dir() / "benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(b) -> None:
    (_dir() / f"{b['id']}.json").write_text(json.dumps({k: v for k, v in b.items() if not k.startswith("_")}, default=str))


def load(bench_id: str) -> Dict[str, Any]:
    with _LOCK:
        if bench_id in _LIVE:
            return json.loads(json.dumps({k: v for k, v in _LIVE[bench_id].items() if not k.startswith("_")}, default=str))
    path = _dir() / f"{AL._slug(bench_id)}.json"
    if not path.exists():
        raise KeyError(bench_id)
    b = json.loads(path.read_text())
    if b.get("status") in ("running", "stopping"):
        b["status"], b["error"] = "interrupted", "The server stopped while this was running."
    return b


def listing() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_dir().glob("bm*.json"), key=lambda p: -p.stat().st_mtime)[:30]:
        try:
            b = load(path.stem)
        except (KeyError, ValueError):
            continue
        out.append({"id": b["id"], "status": b["status"], "version": b.get("version"), "started": b["started"],
                    "models": [{"label": m["label"], "pass_rate": (b["summary"].get(m["label"]) or {}).get("pass_rate")}
                               for m in b["models"]]})
    return out


def _label(m: Dict[str, str]) -> str:
    return f"{m['provider']}: {m['model']}"


def start(models: List[Dict[str, str]], k: int = 1, ids: Optional[List[str]] = None, mode: str = "live",
          background: bool = True) -> Dict[str, Any]:
    import version
    models = [{"provider": str(m.get("provider") or "").strip(), "model": str(m.get("model") or "").strip()}
              for m in models or [] if m and str(m.get("model") or "").strip()][:3]
    if not models:
        raise ValueError("Choose at least one model to benchmark.")
    if mode not in ("live", "rehearsal"):
        raise ValueError("Choose live or rehearsal.")
    tasks = suite_cases(ids)
    if not tasks:
        raise ValueError("Choose at least one task.")
    if mode == "live":
        for m in models:
            problem = AL.missing_keys(with_model(AL.template("react"), m["provider"], m["model"]))
            if problem:
                raise ValueError(problem[0])
    for m in models:
        m["label"] = _label(m)
    k = max(1, min(int(k), 5))
    b = {"id": "bm" + uuid.uuid4().hex[:8], "version": version.__version__, "mode": mode, "k": k,
         "models": models, "tasks": [{k2: t[k2] for k2 in ("id", "agent", "skill", "task")} for t in tasks],
         "status": "running", "error": None, "started": time.time(), "finished": None,
         "trials": {m["label"]: [] for m in models}, "summary": {}, "done": 0,
         "planned": len(models) * len(tasks) * k, "at": None, "_stop": threading.Event()}
    with _LOCK:
        _LIVE[b["id"]] = b
    _save(b)
    if background:
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(_main, b, tasks), daemon=True).start()
    else:
        _main(b, tasks)
    return load(b["id"])


def stop(bench_id: str) -> Dict[str, Any]:
    with _LOCK:
        b = _LIVE.get(bench_id)
    if not b:
        raise ValueError("That benchmark is not running.")
    b["_stop"].set()
    b["status"] = "stopping"
    return load(bench_id)


def _main(b, tasks) -> None:
    try:
        # task by task, every model in turn, so a stopped run still compares the models on the same tasks
        for task in tasks:
            for rep in range(b["k"]):
                for m in b["models"]:
                    if b["_stop"].is_set():
                        raise InterruptedError
                    b["at"] = f"{task['id']} on {m['label']}" + (f", trial {rep + 1}" if b["k"] > 1 else "")
                    _save(b)
                    b["trials"][m["label"]].append(run_trial(task, m["provider"], m["model"], b["mode"]))
                    b["done"] += 1
                    b["summary"] = {label: summarise(ts) for label, ts in b["trials"].items()}
                    _save(b)
        b["status"] = "done"
    except InterruptedError:
        b["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001
        b["status"], b["error"] = "error", f"{type(exc).__name__}: {exc}"
    finally:
        b["at"] = None
        b["finished"] = time.time()
        b["summary"] = {label: summarise(ts) for label, ts in b["trials"].items()}
        _save(b)
        with _LOCK:
            _LIVE.pop(b["id"], None)


# --------------------------------------------------------------------------
# the baseline, and the release gate
# --------------------------------------------------------------------------

def _baseline_path() -> Path:
    return _dir() / "baseline.json"


def baseline() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_baseline_path().read_text())
    except (OSError, ValueError):
        return None


def set_baseline(bench_id: str, label: str) -> Dict[str, Any]:
    b = load(bench_id)
    s = (b.get("summary") or {}).get(label)
    if b.get("status") != "done" or not s or not s.get("trials"):
        raise ValueError("Only a finished benchmark can set the baseline.")
    if b.get("mode") != "live":
        raise ValueError("A rehearsal's scripted model says nothing about the system: set the baseline from a live run.")
    record = {"label": label, "pass_rate": s["pass_rate"], "noise": s["noise"], "trials": s["trials"],
              "tasks": [t["id"] for t in b["tasks"]], "version": b["version"], "benchmark": bench_id, "set": time.time()}
    _baseline_path().write_text(json.dumps(record, indent=1))
    return record


def clear_baseline() -> Dict[str, Any]:
    _baseline_path().unlink(missing_ok=True)
    return {"ok": True}


def _workspaces() -> List[Path]:
    """Every account's benchmark folder: release.sh runs outside any signed-in session."""
    import auth
    roots = [auth.workspace_for(None)] + sorted(p for p in (auth.DATA / "users").glob("*") if p.is_dir())
    return [r / "agentlab" / "benchmarks" for r in roots if (r / "agentlab" / "benchmarks" / "baseline.json").exists()]


def gate() -> int:
    """For release.sh: 0 to go ahead, 1 to stop. Only a benchmark of this version can stop a release."""
    places = _workspaces()
    if not places:
        print("No benchmark baseline. Set one from a finished live benchmark on the Agent lab's Benchmark tab.")
        return 0
    return max(_gate_in(place) for place in places)


def _gate_in(place: Path) -> int:
    import version
    base = json.loads((place / "baseline.json").read_text())
    runs = []
    for p in sorted(place.glob("bm*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            runs.append(json.loads(p.read_text()))
        except ValueError:
            continue
    mine = [r for r in runs if r.get("status") == "done" and r.get("mode") == "live"
            and r.get("version") == version.__version__ and (r.get("summary") or {}).get(base["label"])]
    if not mine:
        print(f"Benchmark baseline: {base['label']} passed {base['pass_rate']:.0%} on v{base['version']}. "
              f"v{version.__version__} has not been benchmarked yet; run it on the Benchmark tab before trusting "
              f"this release. (Not blocking.)")
        return 0
    s = mine[0]["summary"][base["label"]]
    floor = base["pass_rate"] - max(base["noise"], s["noise"])
    verdict = s["pass_rate"] >= floor
    print(f"Benchmark: {base['label']} passed {s['pass_rate']:.0%} on v{version.__version__} against a baseline of "
          f"{base['pass_rate']:.0%} (v{base['version']}); the floor, allowing for noise, is {max(floor, 0):.0%}: "
          + ("ok." if verdict else "WORSE. Fix the regression, or set a new baseline if it is expected."))
    return 0 if verdict else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["gate"]:
        sys.exit(gate())
    print("usage: python3 benchmark.py gate")
