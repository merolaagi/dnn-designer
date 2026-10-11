"""Muse: one front door for every agent in the lab.

You say what you need in your own words. Muse decodes it: what you want done, which agents
should do it, and the self-contained instruction each one gets. Then it runs them in turn,
handing each one what the previous one found, and gives you the result. Each step is an
ordinary run of that agent, so approvals, live steps, stopping and Run history all work as
they do when you talk to the agent directly.

Decoding is one model call (the default model in Settings). How well a model decodes can be
measured on its own, quickly: the routing check sends a set of requests through the decoder
alone and counts how many reach the right agent.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

# the starting designs Muse can hand work to, and when each is the right one
AGENTS = {
    "modelbuilder": "Designs, checks, trains, improves and packages a neural network on the network canvas, for a "
                    "dataset: built-in mnist, fashion_mnist, cifar10 or synthetic, or a table of numbers (CSV). Use it "
                    "for anything about building, training or packaging a model.",
    "mathsearch": "Finds a mathematical answer or construction and proves it with code: counts, optimal objects "
                  "(rulers, sets, placements), checked computations, conjectures tested by search.",
    "reverse": "Explains how a feature of an app installed on this computer works, from its files (Electron or "
               "JavaScript apps, app bundles, .NET, Android packages, native binaries), with file and line evidence.",
    "specialist": "A professional in one field from the roster (marketing, sales, finance, legal, product, design, "
                  "engineering, support, project management, research and more): advice, plans, writing, "
                  "analysis and calculations, with web search and code. Name the role.",
    "station": "Answers questions about the user's own gas station and store from its database: sales, fuel "
               "gallons and margins, tank levels and days of supply, deliveries, prices, shifts, cash over/short, "
               "departments and items, lottery, and whether POS data is arriving. Use it whenever the request is "
               "about their own numbers.",
    "react": "General questions and small tasks that need a quick web search or lookup and no specialist.",
}
MAX_STEPS = 3

DECODER = """You are Muse, the front desk of an agent lab. Read the person's request and decide which agents
should do the work. Do not do the work yourself.

The agents:
{agents}

{roles}
Rules:
- Use one step when one agent can do it all. Use two or three only when the work really has stages that need
  different agents (for example: research with a specialist, then build a model with the model builder).
- Each step's task is a complete, self-contained instruction for that agent: restate everything it needs from
  the request (names, numbers, paths, datasets, limits, the form of the answer). The agent sees nothing else,
  except the results of the steps before it.
- For a specialist step, name the role by its id from the list. Pick the role whose work the request is.
- If something essential is missing and cannot reasonably be assumed (which app, which dataset, which file),
  ask one short question instead of planning.

Reply with strict JSON only:
{{"understood": "<the request in one sentence>",
  "steps": [{{"agent": "<agent id>", "role": "<role id, for a specialist>", "task": "<the instruction>",
             "why": "<why this agent>"}}],
  "question": null}}"""


# --------------------------------------------------------------------------
# decoding a request into a plan
# --------------------------------------------------------------------------

def _roles_for(text: str, limit: int = 12) -> List[Dict[str, Any]]:
    """The roster roles closest to the request's words: the decoder sees these, not all ~280."""
    try:
        import roster
        everyone = roster.roles()
    except Exception:  # noqa: BLE001
        return []
    words = {w for w in re.findall(r"[a-z]{4,}", text.lower())}
    scored = []
    for r in everyone:
        hay = f"{r['name']} {r['description']} {r['vibe']} {r['division']}".lower()
        score = sum(1 for w in words if w in hay) + (2 if any(w in r["name"].lower() for w in words) else 0)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda sr: -sr[0])
    return [r for _, r in scored[:limit]]


def _catalog_text(request: str):
    roles = _roles_for(request)
    agents = "\n".join(f"- {k}: {v}" for k, v in AGENTS.items())
    if roles:
        listing = "\n".join(f"  {r['id']}: {r['name']}: {r['description'][:140]}" for r in roles)
        roles_text = "Roles that fit this request (for a specialist step):\n" + listing + "\n"
    else:
        roles_text = ("No role roster is imported: a specialist step uses a general specialist (leave role empty).\n")
    return agents, roles_text, roles


def rehearsal_plan(request: str) -> Dict[str, Any]:
    """The rehearsal's decoder: keywords, so Muse can be tried with no model at all."""
    t = request.lower()
    if re.search(r"\b(my|our)\b.*\b(sales|gallons|margin|margins|tanks?|shifts?|lottery|deliver(y|ies)|store|"
                 r"station|cash|inventory)\b", t) or re.search(r"\b(yesterday|last week)\b.*\b(sales|gallons|margin)", t):
        agent = "station"
    elif re.search(r"\b(train|network|neural|classifier|dataset|mnist|cifar|package|parameters?)\b", t):
        agent = "modelbuilder"
    elif re.search(r"\b(app|asar|electron|binary|reverse|decompile|\.app|apk|plist)\b", t):
        agent = "reverse"
    elif re.search(r"\b(prove|prime|primes|ruler|golomb|integer|theorem|maximum|minimum|count|verify|grid)\b", t):
        agent = "mathsearch"
    elif _roles_for(request):
        agent = "specialist"
    else:
        agent = "react"
    roles = _roles_for(request) if agent == "specialist" else []
    return {"understood": request[:200], "question": None, "by": "keywords (rehearsal)",
            "steps": [{"agent": agent, "role": roles[0]["id"] if roles else "", "task": request,
                       "why": "matched words in the request"}]}


def _check(plan: Dict[str, Any], roles: List[Dict[str, Any]]) -> List[str]:
    problems = []
    steps = plan.get("steps") or []
    if not steps and not plan.get("question"):
        problems.append("There are no steps and no question.")
    role_ids = {r["id"] for r in roles}
    for i, s in enumerate(steps[:MAX_STEPS], 1):
        if s.get("agent") not in AGENTS:
            problems.append(f"Step {i}: there is no agent called {s.get('agent')!r}; use one of {list(AGENTS)}.")
        if not str(s.get("task") or "").strip():
            problems.append(f"Step {i} has no task.")
        if s.get("agent") == "specialist" and role_ids and s.get("role") and s["role"] not in role_ids:
            problems.append(f"Step {i}: role {s['role']!r} is not in the list; use one of its ids, or leave it empty.")
    return problems


def decode(request: str, mode: str = "live", context: str = "") -> Dict[str, Any]:
    """The plan for a request: {understood, steps[{agent, role, task, why}], question, seconds, tokens}."""
    started = time.time()
    if mode != "live":
        plan = rehearsal_plan(request)
        plan.update(seconds=0.0, tokens=0)
        return plan
    import evolve
    agents, roles_text, roles = _catalog_text(request)
    system = DECODER.format(agents=agents, roles=roles_text)
    import muse_profile as profile
    about = profile.as_text()
    prompt = ((about + "\n\n") if about else "") + \
        (f"Earlier in this conversation:\n{context}\n\n" if context else "") + f"The request:\n{request}"
    tokens, plan, problems = 0, None, []
    for attempt in range(2):
        out = evolve.ask(system, prompt if not problems else prompt + "\n\nYour last reply had problems. Fix them:\n- "
                         + "\n- ".join(problems), 1500)
        tokens += out.get("tokens") or 0
        plan = evolve._json_in(out["text"]) or {}
        problems = _check(plan, roles) if plan else ["The reply was not the JSON asked for."]
        if not problems:
            break
    if problems:
        # fall back to the closest thing to a sensible plan, and say so
        plan = rehearsal_plan(request)
        plan["by"] = "keywords, because the model's plan did not hold: " + problems[0]
    else:
        plan["by"] = "model"
    plan["steps"] = [{"agent": s["agent"], "role": str(s.get("role") or ""), "task": str(s["task"]).strip(),
                      "why": str(s.get("why") or "")[:300]} for s in (plan.get("steps") or [])[:MAX_STEPS]]
    plan.update(seconds=round(time.time() - started, 1), tokens=tokens,
                understood=str(plan.get("understood") or "")[:400], question=plan.get("question") or None)
    return plan


def agent_name(step: Dict[str, Any]) -> str:
    if step["agent"] == "specialist" and step.get("role"):
        try:
            import roster
            return roster.role(step["role"])["name"]
        except Exception:  # noqa: BLE001
            pass
    return AL.TEMPLATES[step["agent"]]["name"].split(":")[0]


def graph_for(step: Dict[str, Any]) -> Dict[str, Any]:
    if step["agent"] == "specialist" and step.get("role"):
        import roster
        try:
            return roster.graph(step["role"])
        except KeyError:
            pass
    return AL.template(step["agent"])


# --------------------------------------------------------------------------
# carrying a plan out, as one turn of a Muse conversation
# --------------------------------------------------------------------------

def _dir() -> Path:
    path = AL._dir() / "muse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state(muse_id: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((_dir() / f"{AL._slug(muse_id)}.json").read_text())
    except (OSError, ValueError):
        return None


def _save(s: Dict[str, Any]) -> None:
    (_dir() / f"{s['id']}.json").write_text(json.dumps(s, default=str))


def start_turn(text: str, mode: str, context: str = "", background: bool = True) -> Dict[str, Any]:
    s = {"id": "mu" + uuid.uuid4().hex[:10], "task": text, "mode": mode, "status": "running", "phase": "decoding",
         "plan": None, "steps": [], "run": None, "answer": None, "error": None, "started": time.time()}
    _save(s)
    if background:
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(_carry_out, s, context), daemon=True).start()
    else:
        _carry_out(s, context)
    return s


def _wait(run_id: str) -> Dict[str, Any]:
    while True:
        view = AL.run_view(run_id)
        if view.get("status") not in AL.LIVE_STATUSES:
            return view
        time.sleep(1.0)


def _carry_out(s: Dict[str, Any], context: str) -> None:
    try:
        plan = decode(s["task"], s["mode"], context)
        s.update(plan=plan, phase="planned")
        _save(s)
        if plan.get("question") and not plan["steps"]:
            s.update(status="done", answer=plan["question"], phase="asked")
            return
        earlier: List[str] = []
        for i, step in enumerate(plan["steps"]):
            import muse_profile as profile
            task = profile.with_profile(step["task"])
            if earlier:
                task += "\n\nWhat the earlier steps found (use it):\n" + "\n\n".join(earlier)
            memory = None
            if step["agent"] == "specialist" and step.get("role"):
                import roster
                memory = roster.memory_dir(step["role"])
            view = AL.start_run(graph_for(step), task, s["mode"], "ask", memory_dir=memory or AL._dir())
            row = {"agent": step["agent"], "role": step.get("role"), "name": agent_name(step), "task": step["task"],
                   "run": view["id"], "status": "running", "answer": None}
            s["steps"].append(row)
            s.update(run=view["id"], phase=f"step {i + 1} of {len(plan['steps'])}: {row['name']}")
            _save(s)
            done = _wait(view["id"])
            row.update(status=done.get("status"), answer=done.get("answer"), error=done.get("error"))
            _save(s)
            if done.get("status") != "done":
                s.update(status=done.get("status") if done.get("status") in ("stopped", "error") else "error",
                         error=done.get("error") or f"{row['name']} did not finish ({done.get('status')}).")
                return
            earlier.append(f"[{row['name']}] {AL._one_line(done.get('answer') or '', 3000)}")
        if len(s["steps"]) == 1:
            s["answer"] = s["steps"][0]["answer"]
        else:
            s["answer"] = "\n\n".join(f"**{r['name']}**\n\n{r['answer'] or ''}" for r in s["steps"])
        s["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        s.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        s["phase"] = s.get("phase") if s["status"] == "running" else "finished"
        s["finished"] = time.time()
        _save(s)


def overlay(turn: Dict[str, Any]) -> bool:
    """Bring a Muse turn up to date from its state file. True if anything changed."""
    s = state(turn.get("muse") or "")
    if not s:
        return False
    keep = {"status": s["status"], "run": s.get("run"), "answer": s.get("answer"), "error": s.get("error"),
            "plan": s.get("plan"), "muse_steps": s.get("steps"), "phase": s.get("phase")}
    if s.get("finished") and s.get("started"):
        keep["seconds"] = round(s["finished"] - s["started"], 1)
    changed = any(turn.get(k) != v for k, v in keep.items())
    turn.update(keep)
    return changed


def context_of(conv: Dict[str, Any], turns: int = 2) -> str:
    rows = []
    for t in conv.get("turns", [])[-turns:]:
        rows.append(f"Asked: {AL._one_line(t.get('task'), 300)}\nAnswer: {AL._one_line(t.get('answer') or '', 600)}")
    return "\n\n".join(rows)


# --------------------------------------------------------------------------
# the routing check: does this model send requests to the right agent?
# --------------------------------------------------------------------------

ROUTING = [
    ("Train a small network to recognise handwritten digits and give me something I can install.", "modelbuilder", None),
    ("I have a CSV of daily fuel sales; build a model that predicts tomorrow's gallons.", "modelbuilder", None),
    ("Make my fashion_mnist network smaller without losing much accuracy.", "modelbuilder", None),
    ("What is the shortest Golomb ruler with 9 marks? Prove it with code.", "mathsearch", None),
    ("How many primes are there below one million? Check it two ways.", "mathsearch", None),
    ("Find the largest set of points on a 7 by 7 grid with no three in a line.", "mathsearch", None),
    ("How does Chatbox stream replies? It's at /Applications/Chatbox.app/Contents/Resources/app.asar", "reverse", None),
    ("Look inside the MongoDB Compass app on my Mac and explain how it saves connections.", "reverse", None),
    ("Write a launch email for my gas-station back-office software.", "specialist", "marketing"),
    ("Price three subscription tiers for a convenience-store POS product.", "specialist", None),
    ("Draft a simple privacy policy for a small SaaS web app.", "specialist", "legal"),
    ("Review my landing page copy and suggest a clearer headline.", "specialist", None),
    ("Work out the monthly payment on a $40,000 loan at 7% over 6 years.", "specialist", "finance"),
    ("How did my fuel margin do last week compared with the week before?", "station", None),
    ("Which of my tanks will run out first, and when should I order?", "station", None),
    ("Was my cash short on any shift yesterday?", "station", None),
    ("What's the weather usually like in Lisbon in March?", "react", None),
    ("Who won the most recent Fields Medal?", "react", None),
    ("Research what convenience-store owners want from back-office software, then train a model on my survey "
     "CSV that predicts which plan they pick.", "specialist+modelbuilder", None),
]


def route_check(mode: str = "live") -> Dict[str, Any]:
    rows = []
    for text, expected, division in ROUTING:
        plan = decode(text, mode)
        chosen = "+".join(s["agent"] for s in plan["steps"]) or ("question" if plan.get("question") else "nothing")
        ok = chosen == expected or (("+" not in expected) and chosen.split("+")[0] == expected and len(plan["steps"]) == 1)
        role_ok = None
        if division and plan["steps"] and plan["steps"][0].get("role"):
            role_ok = plan["steps"][0]["role"].split("/")[0] == division
        rows.append({"request": text, "expected": expected, "chosen": chosen, "passed": ok,
                     "role": plan["steps"][0].get("role") if plan["steps"] else None, "role_ok": role_ok,
                     "by": plan.get("by"), "seconds": plan.get("seconds"), "question": plan.get("question")})
    passed = sum(r["passed"] for r in rows)
    return {"mode": mode, "rows": rows, "passed": passed, "total": len(rows),
            "rate": round(passed / len(rows), 4) if rows else 0.0,
            "seconds": round(sum(r["seconds"] or 0 for r in rows), 1)}


_CHECKS: Dict[str, Dict[str, Any]] = {}


def start_route_check(mode: str = "live") -> Dict[str, Any]:
    cid = "rc" + uuid.uuid4().hex[:8]
    _CHECKS[cid] = {"id": cid, "status": "running", "result": None, "error": None}

    def go():
        try:
            _CHECKS[cid]["result"] = route_check(mode)
            _CHECKS[cid]["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            _CHECKS[cid].update(status="error", error=f"{type(exc).__name__}: {exc}")
    import contextvars
    threading.Thread(target=contextvars.copy_context().run, args=(go,), daemon=True).start()
    return _CHECKS[cid]


def route_check_view(cid: str) -> Dict[str, Any]:
    if cid not in _CHECKS:
        raise KeyError(cid)
    return _CHECKS[cid]
