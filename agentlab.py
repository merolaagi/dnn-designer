"""Agent lab: build a language-model agent out of parts, read each part's
anatomy, and run the result.

Not to be confused with `agents.py`, whose "agents" are experiment loops that
train networks. These are the other kind: a model in a loop with tools.

The design rule is the same one the network side follows — the code is the
point. A run does not interpret the canvas. It generates the Python file you
would download, executes that file, and records what each organ reported as it
worked. So the animation on the canvas, the trace in the panel and the file on
disk all come from one source and cannot disagree.

Two ways to run:

  rehearsal  a scripted stand-in plays the model. It asks for each wired tool
             once or twice, then answers. Tools really run. Costs nothing, needs
             no key, and is what the tests use.
  live       the real model, through the Messages API. Needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_MODEL = os.environ.get("AGENTLAB_MODEL", "claude-sonnet-5-5")

# --------------------------------------------------------------------------
# the catalogue: every organ, what it is, how it behaves, how it fails
# --------------------------------------------------------------------------

SYSTEMS = {
    "senses": {"label": "Senses", "blurb": "Where the task enters and the answer leaves."},
    "brain": {"label": "Brain", "blurb": "The model, and the instructions that shape how it thinks."},
    "memory": {"label": "Memory", "blurb": "What the agent can see right now, and what it can look up."},
    "hands": {"label": "Hands", "blurb": "Tools that turn thinking into action."},
    "nerve": {"label": "Nervous system", "blurb": "Routing, and the loop that keeps it running."},
    "immune": {"label": "Immune system", "blurb": "Checks that keep bad inputs and actions out."},
}
SYSTEM_ORDER = ["senses", "brain", "memory", "hands", "nerve", "immune"]


def _p(label, value, kind="text"):
    return {"label": label, "value": value, "kind": kind}


BLOCKS: Dict[str, Dict[str, Any]] = {
    "user_input": dict(
        name="User input", system="senses", no_in=True, short="The task arrives",
        anatomy="The stimulus. A user message, a webhook, a scheduled trigger: whatever starts the agent working.",
        inside="Raw text, plus whatever metadata you choose to pass along.",
        physiology="Becomes the first user message. Everything downstream reacts to it.",
        failure="Vague goals the agent has to guess at. Instructions hidden in pasted content."),
    "output": dict(
        name="Final answer", system="senses", no_out=True, short="The result leaves",
        anatomy="What the agent hands back: a reply, a file, a database write, a sent message.",
        inside="The last text the model produced once it stopped asking for tools.",
        physiology="Emitted when a model call ends with text instead of a tool request (stop_reason end_turn).",
        failure="Answers that drift from the task. Claims no tool ever checked."),
    "llm": dict(
        name="LLM core", system="brain", short="Reads context, picks next tokens",
        anatomy=("The only organ that thinks. A stateless function, tokens in and tokens out. Every decision "
                 "the agent makes, including which tool to call and with what arguments, is this function "
                 "choosing the next tokens."),
        inside="Frozen weights, attention over the whole context window, and a sampler set by temperature.",
        physiology=("Each step it re-reads everything: system prompt, history, tool results. It returns either "
                    "text or a tool_use block. It remembers nothing between calls; the loop re-feeds it."),
        failure="Invented tool arguments. Losing the thread in a long context. Confidence ahead of evidence.",
        params={"model": _p("Model", DEFAULT_MODEL),
                "temperature": _p("Temperature", 0.3, "number"),
                "max_tokens": _p("Max tokens per step", 2048, "number")}),
    "system_prompt": dict(
        name="System prompt", system="brain", short="Identity and standing orders",
        anatomy="The agent's DNA: role, goals, rules, tone, and when to use each tool. It shapes every step.",
        inside="Plain text. Usually the most valuable thing to tune.",
        physiology="Sent unchanged with every model call, ahead of the conversation.",
        failure="Contradictory rules. Prompts so long the important lines get diluted.",
        params={"text": _p("Instructions",
                           "You are a careful research agent. Break problems down, use tools to check facts, "
                           "and say plainly when you are unsure.", "area")}),
    "planner": dict(
        name="Planner", system="brain", short="Breaks the task into steps",
        anatomy="Turns a goal into ordered sub-goals before any action. A separate model call with a focused prompt.",
        inside="A planning prompt and the task.",
        physiology="Runs once at the start. Its plan is written into context for the core to follow.",
        failure="Rigid plans that ignore new evidence. Plans too vague to act on."),
    "reflector": dict(
        name="Critic", system="brain", short="Reviews the draft, sends it back",
        anatomy="A second look at the work. Checks the draft against the task, then passes it or returns notes.",
        inside="A review prompt, the task, and the draft.",
        physiology="Runs after a draft. Notes become a new message to the core, which triggers another lap.",
        failure="Rubber-stamping. Revising forever without a limit.",
        params={"max_revisions": _p("Max revisions", 2, "number")}),
    "short_mem": dict(
        name="Working memory", system="memory", short="The context window",
        anatomy=("The message list: everything said, every tool call and result, in order. It is the agent's "
                 "entire awareness at any moment."),
        inside="Alternating user and assistant messages, including tool_use and tool_result blocks.",
        physiology="Appended after every step and re-sent in full on every model call.",
        failure="Running past the window. Old details crowding out new ones."),
    "long_mem": dict(
        name="Long-term memory", system="memory", short="Persists between runs",
        anatomy="A store outside the model that survives after the run ends.",
        inside="Saved notes, found again by matching. Here, keyword overlap; in production, embeddings.",
        physiology="Read before the first model call (recall), written after the answer (consolidate).",
        failure="Recalling near-misses. Saving junk that misleads the agent later."),
    "retriever": dict(
        name="Retriever (RAG)", system="memory", short="Fetches relevant documents",
        anatomy="Turns the task into a query, fetches the best-matching chunks, and puts them in context.",
        inside="An index over your documents and a top-k setting.",
        physiology="Match the query, take the top chunks, inject them ahead of the question.",
        failure="Fetching the wrong chunks, or so many that the signal drowns.",
        params={"top_k": _p("Chunks to fetch", 4, "number")}),
    "tool": dict(
        name="Tool", system="hands", short="A function the model can call",
        anatomy=("The model never runs code. It writes a request, a tool name plus JSON arguments; your "
                 "program runs the function and hands back the result."),
        inside="A name, a description the model reads, an input schema, and the Python function.",
        physiology="Model emits tool_use, the router dispatches it, the function runs, the result returns as tool_result.",
        failure="Vague descriptions, so the wrong tool gets picked. Side effects with no undo.",
        params={"name": _p("Tool name", "calculator"),
                "description": _p("What it does (the model reads this)",
                                  "Evaluate an arithmetic expression and return the number.", "area")}),
    "web_search": dict(
        name="Web search", system="hands", short="Fresh facts from the web",
        anatomy="A ready-made tool that brings in current information. The agent's eyes on the world.",
        inside="A search API call and a snippet formatter. The generated stub returns a placeholder.",
        physiology="Called like any tool; snippets come back as a tool_result.",
        failure="Poor sources treated as truth. Results carrying hostile instructions."),
    "code_exec": dict(
        name="Code executor", system="hands", short="Runs code the model writes",
        anatomy="Runs code the model writes and returns the output, for exact maths and data handling.",
        inside="A Python subprocess with a timeout. Not a sandbox.",
        physiology="The model writes code as the tool argument; stdout and errors come back as the result.",
        failure=("In a live run this executes model-written code on this machine. Put a Human approval block "
                 "in front of it unless you trust every prompt the agent will see.")),
    "sub_agent": dict(
        name="Sub-agent", system="hands", short="A whole agent used as a tool",
        anatomy="A complete agent wrapped as a tool: its own instructions, its own loop.",
        inside="Its own system prompt and step budget.",
        physiology="The parent calls it like a tool; it runs its own loop and returns its final text.",
        failure="Context lost between parent and child. Costs that multiply quietly.",
        params={"name": _p("Agent name", "researcher"),
                "role": _p("Its role (system prompt)",
                           "You research one question thoroughly and return a short, sourced summary.", "area")}),
    "router": dict(
        name="Router", system="nerve", short="Dispatches tool calls or exits",
        anatomy=("Reads the model's output and decides where the signal goes: tool requests to the hands, "
                 "finished text out. Only tools wired out of the router are given to the model."),
        inside="A check of stop_reason and a table from tool names to functions.",
        physiology="stop_reason tool_use: run each requested tool. Anything else: leave the loop.",
        failure="Unknown tool names or malformed arguments nobody catches."),
    "loop": dict(
        name="Loop controller", system="nerve", short="Heartbeat and step budget",
        anatomy="The heartbeat. Keeps the perceive, think, act, observe cycle going, and stops it.",
        inside="A counter and a limit.",
        physiology="Counts model calls and stops the run at the limit, answer or not.",
        failure="No limit at all, which means runaway cost. Without this block the limit defaults to 8.",
        params={"max_steps": _p("Max steps", 8, "number")}),
    "guard_in": dict(
        name="Input guardrail", system="immune", short="Screens what comes in",
        anatomy="Checks incoming requests for prompt injection or requests that should not enter.",
        inside="Pattern rules. In production often a small classifier model.",
        physiology="Runs before the first model call; can block the input.",
        failure="Too strict blocks real work; too loose lets attacks through."),
    "guard_out": dict(
        name="Output guardrail", system="immune", short="Screens what goes out",
        anatomy="Checks the final answer before it leaves: personal data, policy, format.",
        inside="Redaction rules for email addresses and phone numbers.",
        physiology="Runs on the final text and redacts what it matches.",
        failure="Silent redaction that confuses people about what changed."),
    "human": dict(
        name="Human approval", system="immune", short="A person signs off",
        anatomy="A checkpoint where a person approves risky actions before they run.",
        inside="A yes or no. Tools wired out of this block need it; tools wired from the router do not.",
        physiology="Pauses at dispatch. A denial goes back to the model as the tool's result.",
        failure="Approval fatigue: clicking yes without reading."),
}

FEEDERS = {"system_prompt", "short_mem", "loop"}
ACTIONS = {"tool", "web_search", "code_exec", "sub_agent"}


def catalog() -> Dict[str, Any]:
    blocks = []
    for key, b in BLOCKS.items():
        blocks.append({
            "type": key, "name": b["name"], "system": b["system"], "short": b["short"],
            "anatomy": b["anatomy"], "inside": b["inside"], "physiology": b["physiology"],
            "failure": b["failure"], "no_in": bool(b.get("no_in")), "no_out": bool(b.get("no_out")),
            "params": b.get("params", {}),
        })
    return {"systems": SYSTEMS, "order": SYSTEM_ORDER, "blocks": blocks,
            "templates": {k: template(k) for k in TEMPLATES},
            "template_names": {k: v["name"] for k, v in TEMPLATES.items()},
            "live_available": bool(os.environ.get("ANTHROPIC_API_KEY"))}


# --------------------------------------------------------------------------
# starting designs
# --------------------------------------------------------------------------

TEMPLATES = {
    "react": {"name": "ReAct tool agent",
              "nodes": [("user_input", 40, 220), ("guard_in", 270, 220), ("system_prompt", 270, 50),
                        ("short_mem", 270, 400), ("llm", 520, 220), ("loop", 520, 50), ("router", 770, 220),
                        ("web_search", 1030, 90), ("tool", 1030, 270), ("guard_out", 1030, 450),
                        ("output", 1280, 450)],
              "edges": [(0, 1), (1, 4), (2, 4), (3, 4), (5, 4), (4, 6), (6, 7), (6, 8), (7, 4), (8, 4),
                        (6, 9), (9, 10)]},
    "rag": {"name": "Retrieval (RAG) agent",
            "nodes": [("user_input", 40, 220), ("retriever", 290, 220), ("long_mem", 290, 410),
                      ("system_prompt", 540, 50), ("llm", 540, 220), ("output", 800, 220)],
            "edges": [(0, 1), (2, 1), (1, 4), (3, 4), (4, 5)]},
    "planexec": {"name": "Planner, executor, critic",
                 "nodes": [("user_input", 40, 240), ("planner", 270, 240), ("system_prompt", 510, 60),
                           ("short_mem", 510, 420), ("llm", 510, 240), ("router", 750, 240),
                           ("code_exec", 1000, 110), ("web_search", 1000, 290), ("reflector", 1000, 470),
                           ("output", 1250, 470), ("loop", 750, 60)],
                 "edges": [(0, 1), (1, 4), (2, 4), (3, 4), (10, 4), (4, 5), (5, 6), (5, 7), (6, 4), (7, 4),
                           (5, 8), (8, 4), (8, 9)]},
    "supervisor": {"name": "Supervisor with sub-agents",
                   "nodes": [("user_input", 40, 240), ("system_prompt", 280, 60), ("llm", 280, 240),
                             ("router", 520, 240), ("human", 760, 90), ("sub_agent", 1010, 60),
                             ("sub_agent", 1010, 240), ("output", 760, 420), ("short_mem", 280, 420)],
                   "edges": [(0, 2), (1, 2), (8, 2), (2, 3), (3, 4), (4, 5), (3, 6), (5, 2), (6, 2), (3, 7)],
                   "params": {5: {"name": "researcher",
                                  "role": "You research one question thoroughly and return a short, sourced summary."},
                              6: {"name": "writer",
                                  "role": "You turn research notes into a clear, well-structured draft."},
                              1: {"text": "You are a supervisor. Split the task, delegate to your sub-agents, "
                                          "then combine their work into one answer."}}},
}


def defaults(kind: str) -> Dict[str, Any]:
    return {k: v["value"] for k, v in BLOCKS[kind].get("params", {}).items()}


def template(key: str) -> Dict[str, Any]:
    t = TEMPLATES[key]
    nodes = []
    for i, (kind, x, y) in enumerate(t["nodes"]):
        params = defaults(kind)
        params.update(t.get("params", {}).get(i, {}))
        nodes.append({"id": f"n{i + 1}", "type": kind, "x": x, "y": y, "params": params})
    edges = [{"id": f"e{k + 1}", "source": nodes[a]["id"], "target": nodes[b]["id"]}
             for k, (a, b) in enumerate(t["edges"])]
    return {"name": t["name"], "nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# reading a design
# --------------------------------------------------------------------------

def _of(graph, kind):
    return [n for n in graph.get("nodes", []) if n.get("type") == kind]


def _params(node):
    merged = defaults(node["type"])
    merged.update(node.get("params") or {})
    return merged


def py_id(text: str) -> str:
    out = re.sub(r"[^a-z0-9_]+", "_", str(text or "tool").lower()).strip("_") or "tool"
    if out[0].isdigit():
        out = "_" + out
    import keyword
    if keyword.iskeyword(out):
        out += "_"
    return out


def wiring(graph) -> Dict[str, Any]:
    """Which tools the model is given, and which of them need a person's yes.

    Only what is wired counts. A tool sitting on the canvas with no wire from the
    router is an organ that is not connected, so the model is not told about it.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    routers = {n["id"] for n in _of(graph, "router")}
    humans = {n["id"] for n in _of(graph, "human")}
    given, gated = [], set()
    for e in edges:
        src, dst = nodes.get(e.get("source")), nodes.get(e.get("target"))
        if not src or not dst or dst["type"] not in ACTIONS:
            continue
        if src["id"] in routers or src["id"] in humans:
            if dst["id"] not in given:
                given.append(dst["id"])
            if src["id"] in humans:
                gated.add(dst["id"])
    return {"given": given, "gated": gated}


def validate(graph) -> List[Dict[str, Any]]:
    problems = []

    def say(level, message, node=None):
        problems.append({"level": level, "message": message, "node": node})

    for n in graph.get("nodes", []):
        if n.get("type") not in BLOCKS:
            say("error", f"{n.get('type')!r} is not a block this lab knows.", n.get("id"))
    if not _of(graph, "user_input"):
        say("error", "Add a User input block so the task has somewhere to enter.")
    if not _of(graph, "llm"):
        say("error", "Add an LLM core. Without it nothing thinks.")
    if not _of(graph, "output"):
        say("error", "Add a Final answer block so the result has somewhere to go.")
    for kind in ("user_input", "llm", "output", "router", "loop", "system_prompt"):
        extra = _of(graph, kind)[1:]
        for n in extra:
            say("warning", f"Only the first {BLOCKS[kind]['name']} is used; this one is ignored.", n["id"])

    w = wiring(graph)
    for n in graph.get("nodes", []):
        if n.get("type") in ACTIONS and n["id"] not in w["given"]:
            say("warning", f"{label(n)} is not wired from the router, so the model is not given it.", n["id"])
    if any(n.get("type") in ACTIONS for n in graph.get("nodes", [])) and not _of(graph, "router"):
        say("warning", "There are tools but no router to dispatch them.")
    if not _of(graph, "loop"):
        say("info", "No loop controller, so the step limit defaults to 8.")

    starts = _of(graph, "user_input")
    ends = {n["id"] for n in _of(graph, "output")}
    if starts and ends:
        seen, queue = {starts[0]["id"]}, deque([starts[0]["id"]])
        while queue:
            cur = queue.popleft()
            for e in graph.get("edges", []):
                if e.get("source") == cur and e.get("target") not in seen:
                    seen.add(e["target"])
                    queue.append(e["target"])
        if not seen & ends:
            say("warning", "No wired path leads from the input to the final answer.")
    return problems


def label(node) -> str:
    p = node.get("params") or {}
    if node.get("type") == "tool":
        return p.get("name") or "Tool"
    if node.get("type") == "sub_agent":
        return f"Sub-agent {p.get('name') or ''}".strip()
    return BLOCKS.get(node.get("type"), {}).get("name", str(node.get("type")))


# --------------------------------------------------------------------------
# code generation
# --------------------------------------------------------------------------

def _doc(text: str) -> str:
    """A triple-quoted literal that is safe for any text a person types."""
    return '"""' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"""'


def codegen(graph) -> Dict[str, Any]:
    """Python source for the design, plus which lines each block contributed.

    Lines are attributed as they are emitted, the way codegen.py records each
    layer's constructor and forward line, so the Code tab shows a block's lines
    by position in this very file rather than reconstructing them for display.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    has = lambda kind: bool(_of(graph, kind))  # noqa: E731
    first = lambda kind: _params(_of(graph, kind)[0])  # noqa: E731
    if not has("llm"):
        return {"source": "# Add an LLM core block to generate an agent.\n",
                "nodemap": {"tools": {}, "agents": {}}, "node_code": {}}

    def nid(kind):
        found = _of(graph, kind)
        return found[0]["id"] if found else None

    llm = first("llm")
    prompt = first("system_prompt")["text"] if has("system_prompt") else "You are a helpful agent."
    max_steps = int(first("loop")["max_steps"]) if has("loop") else 8
    w = wiring(graph)

    used, tools, nodemap = set(), [], {"tools": {}, "agents": {}}

    def unique(name):
        out, i = name, 2
        while out in used:
            out, i = f"{name}_{i}", i + 1
        used.add(out)
        return out

    for tid in w["given"]:
        n = nodes[tid]
        p = _params(n)
        if n["type"] == "tool":
            t = {"kind": "custom", "name": unique(py_id(p["name"])), "arg": "input", "desc": p["description"]}
        elif n["type"] == "web_search":
            t = {"kind": "search", "name": unique("web_search"), "arg": "query",
                 "desc": "Search the web and return short result snippets for a query."}
        elif n["type"] == "code_exec":
            t = {"kind": "code", "name": unique("run_python"), "arg": "code",
                 "desc": "Run a Python snippet and return what it prints. Use print() to show results."}
        else:
            t = {"kind": "agent", "name": unique(py_id(p["name"])), "arg": "task", "role": p["role"],
                 "desc": f"Delegate a sub-task to the {p['name']} agent. Its role: {p['role']}"}
            nodemap["agents"][t["name"]] = tid
        t["gated"] = tid in w["gated"]
        t["node"] = tid
        nodemap["tools"][t["name"]] = tid
        tools.append(t)

    L: List[str] = []
    spans: Dict[str, List[List[int]]] = {}

    def add(lines, *owners):
        """Append lines and credit them to the blocks that own them (1-based, inclusive)."""
        start = len(L) + 1
        L.extend(lines)
        end = len(L)
        while start <= end and not L[start - 1].strip():
            start += 1
        while end >= start and not L[end - 1].strip():
            end -= 1
        if end < start:
            return
        for owner in owners:
            if not owner:
                continue
            mine = spans.setdefault(owner, [])
            # pieces separated only by blank lines read as one
            gap = L[mine[-1][1]:start - 1] if mine else None
            if mine and gap is not None and all(not line.strip() for line in gap):
                mine[-1][1] = end
            else:
                mine.append([start, end])

    LLM, PROMPT, LOOP, ROUTER, WORK = nid("llm"), nid("system_prompt"), nid("loop"), nid("router"), nid("short_mem")
    INPUT, OUTPUT, HUMAN = nid("user_input"), nid("output"), nid("human")
    PLANNER, CRITIC, LONG, RAG = nid("planner"), nid("reflector"), nid("long_mem"), nid("retriever")
    GIN, GOUT = nid("guard_in"), nid("guard_out")

    parts = ", ".join(label(n) for n in graph.get("nodes", []))
    add(['"""Agent generated by Deep Network Designer\'s agent lab.', "",
         f"Anatomy: {parts}.".replace('"""', "'''"),
         "Physiology: perceive -> think (model call) -> act (tool) -> observe (append result) -> repeat.", "",
         "Run it:  ANTHROPIC_API_KEY=... python agent.py", "Standard library only.", '"""', ""])
    imports = ["import json", "import os", "import urllib.request"]
    if any(t["kind"] == "code" for t in tools):
        imports += ["import subprocess", "import sys"]
    if has("guard_in") or has("guard_out"):
        imports.append("import re")
    if has("long_mem"):
        imports.append("from pathlib import Path")
    add(sorted(imports) + [""])
    add(['API_URL = "https://api.anthropic.com/v1/messages"',
         f'MODEL = os.environ.get("AGENT_MODEL", {json.dumps(str(llm["model"]))})',
         f"TEMPERATURE = {float(llm['temperature'])}",
         f"MAX_TOKENS = {int(llm['max_tokens'])}"], LLM)
    add([f"MAX_STEPS = {max_steps}  # loop controller: the heartbeat's hard limit"], LOOP)
    add(["", "",
         "def emit(event, **data):",
         '    """Each organ reports here as it works. The lab swaps this in to animate the canvas;',
         '    on its own it does nothing."""',
         "", "", "# ---- brain: identity ----"])
    add([f"SYSTEM_PROMPT = {_doc(prompt)}"], PROMPT)
    add(["", ""])
    add(["def call_model(system, messages, tools=None):",
         '    """One heartbeat of thought: the model reads the whole context and returns text or tool requests."""',
         '    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,',
         '            "system": system, "messages": messages}',
         "    if tools:",
         '        body["tools"] = tools',
         "    request = urllib.request.Request(",
         "        API_URL, data=json.dumps(body).encode(),",
         '        headers={"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],',
         '                 "anthropic-version": "2023-06-01"})',
         "    with urllib.request.urlopen(request, timeout=120) as reply:",
         "        return json.load(reply)", "", "",
         "def text_of(response):",
         '    return "".join(b.get("text", "") for b in response.get("content", []) if b.get("type") == "text")'],
        LLM)
    add([""])
    if has("planner") or has("reflector"):
        add([""])
        add(["def complete(system, prompt):",
             '    """A single model call with no tools, for the planner and the critic."""',
             '    return text_of(call_model(system, [{"role": "user", "content": prompt}]))'], PLANNER, CRITIC)
        add([""])

    if tools:
        add(["", "# ---- hands: tools ----"])
        for t in tools:
            n = t["name"]
            if t["kind"] == "custom":
                body = [f"def {n}(input: str) -> str:", f"    {_doc(t['desc'])}",
                        f'    return f"TODO: implement {n}. It was asked: {{input}}"']
            elif t["kind"] == "search":
                body = [f"def {n}(query: str) -> str:", '    """Plug a real search API in here."""',
                        '    return f"[placeholder results for: {query}]"']
            elif t["kind"] == "code":
                body = [f"def {n}(code: str) -> str:",
                        '    """Runs model-written code in a subprocess. That is not a sandbox."""',
                        "    try:",
                        "        done = subprocess.run([sys.executable, \"-c\", code], capture_output=True,",
                        "                              text=True, timeout=20)",
                        "    except subprocess.TimeoutExpired:",
                        '        return "Error: timed out after 20 seconds."',
                        '    return (done.stdout + done.stderr)[-4000:] or "(no output)"']
            else:
                body = [f"{n.upper()}_ROLE = {_doc(t['role'])}", "", "",
                        f"def {n}(task: str) -> str:",
                        '    """A sub-agent: its own instructions and its own loop, called like a tool."""',
                        f'    emit("delegate", agent="{n}")',
                        f'    return agent_loop([{{"role": "user", "content": task}}], {n.upper()}_ROLE,',
                        f'                      tools=[], tool_funcs={{}}, max_steps=4, agent="{n}")']
            add([""])
            add(body, t["node"])
            add([""])
        add(["", "TOOLS = ["], ROUTER)
        for t in tools:
            add(["    {", f'        "name": {json.dumps(t["name"])},',
                 f'        "description": {json.dumps(t["desc"])},',
                 f'        "input_schema": {{"type": "object", "properties": {{"{t["arg"]}": {{"type": "string"}}}},'
                 f' "required": ["{t["arg"]}"]}},', "    },"], t["node"], ROUTER)
        add(["]", "", "TOOL_FUNCS = {" + ", ".join(f'"{t["name"]}": {t["name"]}' for t in tools) + "}"], ROUTER)
        gated = [t["name"] for t in tools if t["gated"]]
        add(["NEEDS_APPROVAL = {" + ", ".join(json.dumps(g) for g in gated) + "}" if gated
             else "NEEDS_APPROVAL = set()"], HUMAN)
        add([""])
    else:
        add(["", "TOOLS = []", "TOOL_FUNCS = {}"], ROUTER)
        add(["NEEDS_APPROVAL = set()"], HUMAN)
        add([""])

    if has("long_mem") or has("retriever"):
        add(["", "# ---- memory ----",
             "# Keyword overlap keeps this to the standard library. Swap in embeddings for real use."])
        if has("long_mem"):
            add(['MEMORY_FILE = Path("agent_memory.json")', "", "",
                 "def load_memories():",
                 "    return json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else []", "", "",
                 "def save_memory(text):",
                 "    notes = load_memories() + [text]",
                 "    MEMORY_FILE.write_text(json.dumps(notes[-500:], indent=2))",
                 '    emit("remember")'], LONG)
            add([""])
        if has("retriever"):
            add([""])
            add(["DOCUMENTS = [",
                 '    "Replace these strings with chunks of your own documents.",',
                 '    "Each chunk should hold a few hundred words on one idea.",', "]"], RAG)
            add([""])
        pool = "list(DOCUMENTS)" if has("retriever") else "[]"
        if has("long_mem"):
            pool += " + load_memories()"
        add([""])
        add(["def retrieve(query, k):", f"    pool = {pool}",
             "    words = set(query.lower().split())",
             "    ranked = sorted(pool, key=lambda doc: -len(words & set(doc.lower().split())))",
             "    found = [doc for doc in ranked[:k] if words & set(doc.lower().split())]",
             '    emit("recall", count=len(found))',
             "    return found"], RAG or LONG)
        add([""])

    if has("guard_in"):
        add(["", "# ---- immune system ----"])
        add(['INJECTION = [r"ignore (all|previous) instructions", r"reveal your system prompt", r"you are now"]',
             "", "",
             "def input_guardrail(text):",
             "    for pattern in INJECTION:",
             "        if re.search(pattern, text, re.IGNORECASE):",
             '            emit("guard_in", passed=False)',
             '            raise ValueError("Blocked by the input guardrail: this looks like prompt injection.")',
             '    emit("guard_in", passed=True)',
             "    return text.strip()"], GIN)
        add([""])
    if has("guard_out"):
        add([""])
        add(["def output_guardrail(text):",
             '    cleaned = re.sub(r"[\\w.+-]+@[\\w-]+\\.[\\w.]+", "[email removed]", text)',
             '    cleaned = re.sub(r"\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b", "[phone removed]", cleaned)',
             '    emit("guard_out", changed=cleaned != text)',
             "    return cleaned"], GOUT)
        add([""])
    add([""])
    add(["def human_approves(name, args):",
         '    """A person signs off before a gated tool runs."""',
         '    answer = input(f"\\nApprove {name}({json.dumps(args)})? [y/N] ")',
         '    approved = answer.strip().lower() == "y"',
         '    emit("approval", tool=name, approved=approved)',
         "    return approved"], HUMAN)
    add([""])
    if has("planner"):
        add([""])
        add(['PLANNER_PROMPT = "You are a planner. Write a short numbered plan of concrete steps. No preamble."'],
            PLANNER)
        add([""])
    if has("reflector"):
        add([""])
        add(["CRITIC_PROMPT = (",
             "    \"You review an agent's draft answer against the task. \"",
             '    "If it is correct, complete and supported, reply with exactly PASS. "',
             '    "Otherwise list the specific problems to fix."', ")",
             f"MAX_REVISIONS = {int(first('reflector')['max_revisions'])}", "", "",
             "def critique(task, answer):",
             '    verdict = complete(CRITIC_PROMPT, f"Task:\\n{task}\\n\\nDraft answer:\\n{answer}")',
             '    passed = verdict.strip().upper().startswith("PASS")',
             '    emit("critique", passed=passed, notes=verdict[:200])',
             "    return None if passed else verdict"], CRITIC)
        add([""])

    add(["", "# ---- nervous system: the loop and the router ----"])
    add(['def agent_loop(messages, system, tools, tool_funcs, max_steps, agent="main"):',
         '    """Working memory is `messages`: appended every step, re-sent in full on every call."""'], WORK)
    add(["    for step in range(max_steps):"], LOOP)
    add(["        response = call_model(system, messages, tools)"], LLM)
    add(['        wants_tools = response.get("stop_reason") == "tool_use"',
         '        emit("model", agent=agent, step=step + 1, wants_tools=wants_tools)'], ROUTER)
    add(['        messages.append({"role": "assistant", "content": response["content"]})'], WORK)
    add(["        if not wants_tools:", "            return text_of(response)"], ROUTER, OUTPUT)
    add(["", "        results = []",
         '        for block in response["content"]:',
         '            if block.get("type") != "tool_use":',
         "                continue",
         '            name, args = block["name"], block.get("input") or {}',
         '            emit("dispatch", agent=agent, tool=name)',
         "            func = tool_funcs.get(name)",
         "            if func is None:",
         '                output = f"Error: there is no tool called {name}."'], ROUTER)
    add(["            elif name in NEEDS_APPROVAL and not human_approves(name, args):",
         '                output = "A person declined this action. Find another way, or explain why it is needed."'],
        HUMAN)
    add(["            else:",
         "                try:",
         "                    output = func(**args)",
         "                except Exception as error:",
         '                    output = f"Error: {error}"'], ROUTER)
    add(['            emit("tool_result", agent=agent, tool=name, preview=str(output)[:200])',
         '            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": str(output)})',
         '        messages.append({"role": "user", "content": results})'], WORK)
    add([""])
    add(['    emit("budget", agent=agent, limit=max_steps)',
         '    return "Stopped: the step budget ran out before a final answer."'], LOOP)
    add(["", ""])
    add(["def run_agent(task):", '    emit("input", task=task[:200])'], INPUT)
    if has("guard_in"):
        add(["    task = input_guardrail(task)"], GIN)
    add(["    context = []"], INPUT)
    if has("long_mem") or has("retriever"):
        k = int(first("retriever")["top_k"]) if has("retriever") else 4
        add([f"    notes = retrieve(task, k={k})", "    if notes:",
             '        context.append("Relevant notes:\\n" + "\\n".join(f"- {n}" for n in notes))'], RAG, LONG)
    if has("planner"):
        add(["    plan = complete(PLANNER_PROMPT, task)", '    emit("plan", preview=plan[:200])',
             '    context.append("Plan to follow:\\n" + plan)'], PLANNER)
    add(['    messages = [{"role": "user", "content": "\\n\\n".join(context + [f"Task:\\n{task}"])}]'], INPUT, WORK)
    add(["    answer = agent_loop(messages, SYSTEM_PROMPT, TOOLS, TOOL_FUNCS, MAX_STEPS)"], LOOP, PROMPT)
    if has("reflector"):
        add(["    for _ in range(MAX_REVISIONS):", "        notes = critique(task, answer)",
             "        if notes is None:", "            break",
             '        messages.append({"role": "user", "content": f"A reviewer found problems:\\n{notes}\\n\\nRevise."})',
             "        answer = agent_loop(messages, SYSTEM_PROMPT, TOOLS, TOOL_FUNCS, MAX_STEPS)"], CRITIC)
    if has("long_mem"):
        add(['    save_memory(f"Task: {task[:200]} | Answer: {answer[:300]}")'], LONG)
    if has("guard_out"):
        add(["    answer = output_guardrail(answer)"], GOUT)
    add(['    emit("final", preview=answer[:300])', "    return answer"], OUTPUT)
    add(["", ""])
    add(['if __name__ == "__main__":', '    print(run_agent(input("Task: ")))'], INPUT, OUTPUT)
    add([""])
    return {"source": "\n".join(L), "nodemap": nodemap, "node_code": spans}


# --------------------------------------------------------------------------
# the mathematics of each organ, with this design's numbers in it
# --------------------------------------------------------------------------
# Same shape as mathbook.explain() — title, equation, shape, symbols,
# arithmetic, freedom — so the panel reads the same on both canvases. Each
# entry was written from the code codegen() emits, not from a survey of agent
# papers: the retriever is a word-overlap count because that is what the file
# does, and saying "cosine similarity" would describe a different program.

def _tokens(text: str) -> int:
    return max(1, -(-len(str(text)) // 4))


def _n(value) -> str:
    return f"{int(value):,}"


def _context(graph) -> Dict[str, Any]:
    steps = int(_params(_of(graph, "loop")[0])["max_steps"]) if _of(graph, "loop") else 8
    prompt = (_params(_of(graph, "system_prompt")[0])["text"] if _of(graph, "system_prompt")
              else "You are a helpful agent.")
    built = codegen(graph)
    return {"N": steps, "S": _tokens(prompt), "S_chars": len(prompt), "tools": built["nodemap"]["tools"],
            "w": wiring(graph)}


def explain(graph, node_id: Optional[str] = None) -> Dict[str, Any]:
    """The mathematics of one block, or of the whole agent when no block is named."""
    c = _context(graph)
    N, S = c["N"], c["S"]
    if node_id is None:
        return {
            "type": "agent", "label": graph.get("name") or "This agent",
            "title": "An agent is a recurrence",
            "equation": "r_k ~ π_θ(· | S, H_k),   H_{k+1} = H_k ⊕ r_k ⊕ f(a_k)",
            "shape": f"k = 1 … K,  K ≤ N = {N}",
            "symbols": [("π_θ", "the model; θ is frozen for the whole run"),
                        ("S", f"the system prompt, ≈ {_n(S)} tokens"),
                        ("H_k", "the history so far: working memory"),
                        ("r_k", "what the model returns on step k: text, or tool requests"),
                        ("a_k", "the arguments inside r_k's tool requests"),
                        ("f", "the tools; your program runs them, the model never does"),
                        ("⊕", "append to the end of the message list")],
            "arithmetic": [("stops when", "stop(r_k) ≠ tool_use, or k reaches N"),
                           ("model calls", f"between 1 and {N} in the main loop"),
                           ("tools offered", ", ".join(c["tools"]) or "none")],
            "freedom": ["Nothing learns during a run. θ is the same at step N as at step 1; everything "
                        "the agent picks up lives in H, which is why its length is the cost that matters.",
                        "Every block on the canvas changes one term of this recurrence: S, what goes into "
                        "H₁, which f exist, which a_k are allowed to run, or where K ends."],
        }
    node = next((n for n in graph.get("nodes", []) if n.get("id") == node_id), None)
    if node is None:
        return {"missing": "That block is not on the canvas.", "label": node_id}
    p, kind = _params(node), node["type"]
    builder = MATH.get(kind)
    if builder is None:
        return {"missing": f"The mathematics of {BLOCKS.get(kind, {}).get('name', kind)} is not written up.",
                "label": label(node)}
    entry = builder(p, node, graph, c)
    entry.setdefault("symbols", [])
    entry.setdefault("arithmetic", [])
    entry.setdefault("freedom", [])
    entry.setdefault("shape", "")
    entry.update(type=kind, label=label(node))
    return entry


def _m_input(p, node, graph, c):
    parts = (["retrieved notes"] if (_of(graph, "retriever") or _of(graph, "long_mem")) else []) + \
            (["the plan"] if _of(graph, "planner") else [])
    return {"title": "The first message",
            "equation": "H₁ = m₁ = context ⊕ \"Task:\" ⊕ x",
            "symbols": [("x", "the task as typed"), ("context", ", ".join(parts) or "empty in this design"),
                        ("H₁", "working memory at the start of the loop")],
            "arithmetic": [("context parts", ", ".join(parts) or "none: the task alone"),
                           ("re-read", f"m₁ is part of the input to every call, so up to N = {c['N']} times")],
            "freedom": ["Whatever goes into m₁ is paid for on every step. A long pasted document in the task "
                        f"costs its length × {c['N']} in the worst case."]}


def _m_output(p, node, graph, c):
    guarded = bool(_of(graph, "guard_out"))
    return {"title": "Where the loop exits",
            "equation": "y = " + ("g_out(" if guarded else "") + "text(r_K)" + (")" if guarded else "")
                        + ",   K = min{ k : stop(r_k) ≠ tool_use }",
            "symbols": [("K", "the first step on which the model answers instead of asking for a tool"),
                        ("text(r)", "the text blocks of the response, joined")]
                       + ([("g_out", "the output guardrail")] if guarded else []),
            "arithmetic": [("exists only if", f"K ≤ N = {c['N']}; otherwise y is the budget message")],
            "freedom": ["The answer is whatever the model wrote last. Nothing in the loop checks it against "
                        "the tool results unless a Critic is wired in."]}


def _m_llm(p, node, graph, c):
    T = float(p["temperature"])
    ratio = math.exp(1 / T) if T > 0 else float("inf")
    extra = (1 if _of(graph, "planner") else 0)
    critic = int(_params(_of(graph, "reflector")[0])["max_revisions"]) if _of(graph, "reflector") else 0
    return {"title": "Next-token sampling, called in a loop",
            "equation": "r_k ~ π_θ(· | S, H_k),   P(tᵢ) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)",
            "shape": f"T = {T:g},  at most {_n(p['max_tokens'])} tokens out per call,  model {p['model']}",
            "symbols": [("π_θ", "the network; θ, its weights, never change during a run"),
                        ("zᵢ", "the score (logit) the network gives token i"),
                        ("T", "temperature: divides every logit before the softmax"),
                        ("S, H_k", "system prompt and history: the whole input, every call")],
            "arithmetic": [("temperature", f"two tokens whose logits differ by 1 are picked at odds e^(1/T) = "
                                           + (f"{ratio:,.1f} : 1" if ratio != float("inf") else "∞ : 1 (greedy)")),
                           ("calls per run", f"1 to {c['N']} in the loop"
                                             + (f", +{extra} planner" if extra else "")
                                             + (f", up to +{critic} critic and ({critic}+1)·{c['N']} loop calls "
                                                f"with revisions" if critic else "")),
                           ("input on call k", "S + |H_k| tokens: everything so far, read again")],
            "freedom": ["Lower T makes tool arguments repeatable; higher T explores. At T → 0 the softmax "
                        "becomes argmax and two runs on the same history agree.",
                        "The model has no state of its own between calls. Remove H_k and step k knows nothing "
                        "about step k − 1."]}


def _m_prompt(p, node, graph, c):
    return {"title": "A constant term in every call",
            "equation": "S_k = S   for k = 1 … N",
            "symbols": [("S", "this text, unchanged for the whole run")],
            "arithmetic": [("length", f"{_n(c['S_chars'])} characters ≈ {_n(c['S'])} tokens (≈ 4 characters a token)"),
                           ("sent", f"up to N = {c['N']} times ≈ {_n(c['S'] * c['N'])} tokens per run")],
            "freedom": ["Because S never changes it is the part worth caching: a cached prefix is billed in full "
                        "once and at a fraction afterwards.",
                        "Every rule in S competes for the model's attention with everything in H. A rule that "
                        "matters belongs near the start and stated once."]}


def _m_planner(p, node, graph, c):
    return {"title": "One extra call before the loop",
            "equation": "plan = π_θ(P_plan, x),   H₁ = plan ⊕ x",
            "symbols": [("P_plan", "the planner's own short prompt"), ("x", "the task")],
            "arithmetic": [("extra model calls", "1"),
                           ("cost afterwards", f"the plan's tokens are re-read on each of up to {c['N']} calls")],
            "freedom": ["The plan is advice in the context, not a constraint: nothing makes step k follow it. "
                        "Re-planning when a tool result contradicts it is a second wire back to this block."]}


def _m_critic(p, node, graph, c):
    R, N = int(p["max_revisions"]), c["N"]
    return {"title": "Accept, or loop again",
            "equation": "accept(y) ⇔ π_θ(P_critic, x, y) begins with PASS",
            "symbols": [("y", "the draft answer"), ("R", f"the revision limit, {R}")],
            "arithmetic": [("critic calls", f"≤ R = {R}"),
                           ("worst-case model calls", f"(R + 1)·N + R = ({R} + 1)·{N} + {R} = {(R + 1) * N + R}")],
            "freedom": ["A critic made of the same model shares its blind spots. It catches omissions better "
                        "than errors the model is confident about.",
                        "PASS is a string test on the critic's first word, so the verdict is only as reliable "
                        "as the critic's obedience to its format."]}


def _m_work(p, node, graph, c):
    N = c["N"]
    return {"title": "A list that only grows",
            "equation": "H_{k+1} = H_k ⊕ r_k ⊕ o_k",
            "symbols": [("H_k", "every message so far"), ("r_k", "the model's response on step k"),
                        ("o_k", "the tool results it asked for")],
            "arithmetic": [("length after k steps", "|H_k| = |m₁| + Σ_{j<k} (|r_j| + |o_j|), linear in k"),
                           ("read over a run", f"Σ_{{k=1..N}} (S + |H_k|) ≈ N(S + |m₁|) + ρ·N(N − 1)/2 "
                                               f"= {N}(S + |m₁|) + {N * (N - 1) // 2}ρ"),
                           ("ρ", "the average tokens one step adds (request plus result)")],
            "freedom": ["The N(N − 1)/2 term is why long loops get expensive: doubling the steps roughly "
                        "quadruples what is re-read.",
                        "Summarising old turns replaces ρ·k with a constant, at the price of detail the model "
                        "may need later."]}


def _m_long(p, node, graph, c):
    return {"title": "A store outside the weights",
            "equation": "read: top_k score(q, d) over d ∈ M;   write: M ← last 500 of (M ∪ {(x, y)})",
            "symbols": [("M", "saved notes, kept in agent_memory.json"),
                        ("score(q, d)", "|W(q) ∩ W(d)|: how many words the query and note share")],
            "arithmetic": [("capacity", "500 notes; older ones are dropped"),
                           ("written", "once per run, after the answer")],
            "freedom": ["Word overlap misses paraphrase. Replacing score with cos(E(q), E(d)) for an embedding "
                        "E is the usual next step, and the only line that changes is the sort key."]}


def _m_rag(p, node, graph, c):
    k = int(p["top_k"])
    return {"title": "Top-k by shared words",
            "equation": "R(q) = top_k { d ∈ D : score(q, d) > 0 },   score(q, d) = |W(q) ∩ W(d)|",
            "shape": f"k = {k}",
            "symbols": [("D", "the documents in DOCUMENTS"
                              + (" plus long-term memory" if _of(graph, "long_mem") else "")),
                        ("W(·)", "the set of lower-cased words"), ("q", "the task")],
            "arithmetic": [("returned", f"at most {k}; a document sharing no word with q is never returned"),
                           ("ties", "broken by list order, since the sort is stable")],
            "freedom": ["This is lexical retrieval. Dense retrieval swaps the score for cos(E(q), E(d)) and "
                        "finds documents that say the same thing in other words.",
                        "k trades recall against noise: every retrieved chunk is re-read on every step."]}


def _m_tool(p, node, graph, c):
    given = node["id"] in c["w"]["given"]
    gated = node["id"] in c["w"]["gated"]
    name = next((t for t, nid in c["tools"].items() if nid == node["id"]), None)
    kind = node["type"]
    if kind == "code_exec":
        eq, shape = "o = (stdout ⊕ stderr)(exec(a.code))[−4000:]", "a = { code: string },  timeout 20 s"
    elif kind == "web_search":
        eq, shape = "o = search(a.query)", "a = { query: string }   (the generated stub returns a placeholder)"
    else:
        eq, shape = "o = f(a.input)", "a = { input: string }"
    desc = p.get("description") or ""
    rows = [("given to the model", f"yes, as {name}" if given else "no: not wired from the router"),
            ("needs approval", "yes" if gated else "no")]
    if desc:
        rows.append(("description", f"≈ {_n(_tokens(desc))} tokens, sent with every call as part of the tool list"))
    return {"title": "A function the model can only request",
            "equation": eq + ",   a ~ π_θ(· | S, H_k)", "shape": shape,
            "symbols": [("a", "the JSON arguments the model wrote"), ("o", "what comes back, as text")],
            "arithmetic": rows,
            "freedom": ["The model predicts a; your program decides whether f(a) runs. Every safety property "
                        "an agent has lives in that gap.",
                        "The model chooses tools from their descriptions alone, so the description is part of "
                        "the tool's behaviour."]}


def _m_sub(p, node, graph, c):
    return {"title": "A loop inside a tool",
            "equation": "o = Agent_role(a.task):  H′₁ = a.task,  N′ = 4,  no tools",
            "symbols": [("role", "its own system prompt"), ("N′", "its step limit")],
            "arithmetic": [("model calls per delegation", "≤ 4"),
                           ("parent context", "only o comes back; the sub-agent's history is discarded")],
            "freedom": ["Delegation compresses: the parent sees a summary, not the work. That saves the parent's "
                        "context and loses whatever the summary leaves out."]}


def _m_router(p, node, graph, c):
    return {"title": "A branch on one field",
            "equation": "route(r) = run f for each tool_use in r  if stop(r) = tool_use;  else exit with text(r)",
            "symbols": [("stop(r)", "the response's stop_reason")],
            "arithmetic": [("tools offered", ", ".join(c["tools"]) or "none"),
                           ("unknown tool", "returns an error string to the model rather than raising")],
            "freedom": ["The router has no judgement: it is a pure function of the model's output. Anything "
                        "you want enforced, such as allow-lists or argument checks, has to be written here."]}


def _m_loop(p, node, graph, c):
    N = int(p["max_steps"])
    return {"title": "A bounded iteration",
            "equation": "for k = 1 … N:  stop at the first k with stop(r_k) ≠ tool_use",
            "shape": f"N = {N}",
            "arithmetic": [("model calls", f"≤ {N}"),
                           ("no answer when", f"every one of the {N} responses asks for a tool")],
            "freedom": ["N is the only hard bound on cost in the file. Without it the loop's length is decided "
                        "by the model."]}


def _m_gin(p, node, graph, c):
    return {"title": "A blocklist",
            "equation": "pass(x) = ¬∃ i : patternᵢ matches x   (case-insensitive)",
            "symbols": [("pattern₁", "ignore (all|previous) instructions"),
                        ("pattern₂", "reveal your system prompt"), ("pattern₃", "you are now")],
            "arithmetic": [("patterns", "3"), ("checked", "the task only; tool results are not screened")],
            "freedom": ["A blocklist rejects only what it lists. Injection that arrives through a tool result "
                        "never passes this block at all."]}


def _m_gout(p, node, graph, c):
    return {"title": "Two substitutions",
            "equation": "y′ = redact_phone(redact_email(y))",
            "symbols": [("redact_email", "[\\w.+-]+@[\\w-]+\\.[\\w.]+ → [email removed]"),
                        ("redact_phone", "ddd-ddd-dddd in common separators → [phone removed]")],
            "arithmetic": [("applied", "once, to the final answer only")],
            "freedom": ["Regex redaction is exact about what it matches and blind to everything else, which "
                        "makes it easy to test and easy to get around."]}


def _m_human(p, node, graph, c):
    gated = [t for t, nid in c["tools"].items() if nid in c["w"]["gated"]]
    return {"title": "A gate on the dispatch",
            "equation": "run f(a)  ⇔  name ∉ G  ∨  approve(name, a)",
            "symbols": [("G", "the tools wired out of this block")],
            "arithmetic": [("G here", ", ".join(gated) or "empty: nothing is wired out of this block"),
                           ("on denial", "o = a refusal message, so the model sees it and adapts")],
            "freedom": ["The gate sits between prediction and action, which is the only place a human check "
                        "can stop a side effect rather than report it."]}


MATH: Dict[str, Callable] = {
    "user_input": _m_input, "output": _m_output, "llm": _m_llm, "system_prompt": _m_prompt,
    "planner": _m_planner, "reflector": _m_critic, "short_mem": _m_work, "long_mem": _m_long,
    "retriever": _m_rag, "tool": _m_tool, "web_search": _m_tool, "code_exec": _m_tool,
    "sub_agent": _m_sub, "router": _m_router, "loop": _m_loop, "guard_in": _m_gin,
    "guard_out": _m_gout, "human": _m_human,
}


def node_view(graph, node_id: Optional[str]) -> Dict[str, Any]:
    """Everything the inspector shows for one block: its lines of the file and its maths."""
    built = codegen(graph)
    lines = built["source"].split("\n")
    spans = built["node_code"].get(node_id, []) if node_id else []
    return {"math": explain(graph, node_id),
            "spans": spans,
            "snippets": [{"start": s, "end": e, "text": "\n".join(lines[s - 1:e])} for s, e in spans],
            "source": built["source"]}


# --------------------------------------------------------------------------
# running a design
# --------------------------------------------------------------------------

class Rehearsal:
    """Plays the model so a design can be run with no key and no cost.

    It asks for each tool it was given, in order, up to two rounds, then answers
    with what came back. The critic objects once and then passes. Everything
    around it — the loop, the router, the tools, the guardrails — is the real
    generated code.
    """

    def __init__(self):
        self.critic_calls = 0

    def __call__(self, system, messages, tools=None):
        if not tools:
            low = (system or "").lower()
            if "planner" in low:
                text = "1. Find the facts the task needs.\n2. Work out what follows.\n3. Answer, naming sources."
            elif "review" in low:
                self.critic_calls += 1
                text = ("The answer gives a figure without saying where it came from."
                        if self.critic_calls == 1 else "PASS")
            else:
                ask = messages[0]["content"] if messages else ""
                text = f"(rehearsal) A short, sourced finding on: {str(ask)[:80]}"
            return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

        rounds = sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], list))
        if rounds < min(2, len(tools)):
            tool = tools[rounds]
            arg = tool["input_schema"]["required"][0]
            task = str(messages[0]["content"]).split("Task:\n")[-1][:120]
            value = "print(round(545000 / 450))" if arg == "code" else task
            return {"content": [{"type": "text", "text": f"I should check this with {tool['name']}."},
                                {"type": "tool_use", "id": f"toolu_rehearsal_{uuid.uuid4().hex[:10]}",
                                 "name": tool["name"], "input": {arg: value}}],
                    "stop_reason": "tool_use"}
        seen = []
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                seen += [str(b.get("content", ""))[:60] for b in m["content"]]
        body = "; ".join(seen) if seen else "nothing beyond the task itself"
        return {"content": [{"type": "text", "text": f"(rehearsal) Final answer, drawing on: {body}"}],
                "stop_reason": "end_turn"}


def run(graph, task: str, mode: str = "rehearsal", approvals: str = "approve",
        memory_dir: Optional[Path] = None) -> Dict[str, Any]:
    errors = [p for p in validate(graph) if p["level"] == "error"]
    if errors:
        return {"ok": False, "problems": errors, "events": [], "answer": None, "mode": mode}
    if mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("A live run needs ANTHROPIC_API_KEY in the server's environment. "
                         "Rehearsal runs work without it.")
    if mode not in ("live", "rehearsal"):
        raise ValueError(f"Unknown mode {mode!r}.")

    built = codegen(graph)
    events: List[Dict[str, Any]] = []
    started = time.time()

    def emit(event, **data):
        events.append({"event": event, "t": round(time.time() - started, 3), **data})

    def approve(name, args):
        ok = approvals == "approve"
        emit("approval", tool=name, approved=ok)
        return ok

    space: Dict[str, Any] = {"__name__": "agentlab_run"}
    exec(compile(built["source"], "<agent>", "exec"), space)  # noqa: S102 — our own generated file
    space["emit"] = emit
    space["human_approves"] = approve
    if mode == "rehearsal":
        space["call_model"] = Rehearsal()
    if "MEMORY_FILE" in space and memory_dir is not None:
        space["MEMORY_FILE"] = Path(memory_dir) / "agent_memory.json"

    answer, error = None, None
    try:
        answer = space["run_agent"](task)
    except Exception as exc:  # noqa: BLE001 — a failed run is a result to show, not a crash
        error = f"{type(exc).__name__}: {exc}"
        emit("error", message=error)
    return {"ok": error is None, "answer": answer, "error": error, "events": events,
            "mode": mode, "nodemap": built["nodemap"], "seconds": round(time.time() - started, 2)}


# --------------------------------------------------------------------------
# saving designs
# --------------------------------------------------------------------------

def _dir() -> Path:
    import auth
    return auth.sub("agentlab")


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:80] or "agent"


def save(graph) -> Dict[str, Any]:
    name = _slug(graph.get("name", "agent"))
    (_dir() / f"{name}.json").write_text(json.dumps(graph, indent=2))
    return {"ok": True, "name": name}


def listing() -> List[str]:
    return sorted(p.stem for p in _dir().glob("*.json") if p.stem != "agent_memory")


def load(name: str) -> Dict[str, Any]:
    path = _dir() / f"{_slug(name)}.json"
    if not path.exists():
        raise KeyError(name)
    return json.loads(path.read_text())
