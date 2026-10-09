"""Specialists: role prompts from an agent roster, run on the agent lab's machinery.

A roster such as github.com/msitarzewski/agency-agents is a folder of
Markdown files, one per role: frontmatter (name, description, emoji, colour)
and a long system prompt. On their own those files are personas: they have no
tools, no checks, and they tell the model it "remembers" things it never saw.

Here each role becomes a working agent instead:
- its prompt, minus the claims of memory and experience it does not have, and
  minus bulky code examples when the prompt is very long;
- real hands: a sandboxed code runner and web search;
- a critic that reviews every draft against the role's own Success Metrics and
  Critical Rules before the answer goes out;
- long-term notes of its own, kept per specialist.

The roster is imported once (from GitHub or a folder on this computer) into the
workspace; nothing here ships the files themselves.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import auth

DEFAULT_REPO = "msitarzewski/agency-agents"
# folders in a roster that hold tooling or playbooks, not roles
SKIP = {"integrations", "scripts", "examples", "strategy", ".github", "node_modules"}
LONG_PROMPT = 9000          # characters; past this, code examples are left out of the prompt
MAX_PROMPT = 14000          # and past this, the last sections are left out too
STANDARDS_LIMIT = 2400

WORKING_RULES = """

## How you work here
You are running inside an agent lab with real tools. Use them:
- `web_search` finds current facts, prices, names and sources. Search before stating anything that may have changed, and give the addresses you relied on.
- `run_python` runs Python in a sandbox. Use it for every calculation, table or piece of code you claim works, and report what it printed.
About memory: you know only this conversation and the notes recalled at the start. Do not claim past projects, clients or results; nobody saw them.
If the request is missing something you need, say what and make a reasonable assumption you state plainly, rather than stalling.
Do not end your turn by describing what you will do: do it with the tools, then give the finished result.
Always answer in the language the user writes in.
"""


def _dir() -> Path:
    return auth.sub("agentlab")


def _index_path() -> Path:
    return _dir() / "roster.json"


# --------------------------------------------------------------------------
# reading role files
# --------------------------------------------------------------------------

def _frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """(fields, body) of a Markdown file with simple `key: value` frontmatter."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fields = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line.startswith((" ", "\t", "#")):
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip().strip('"').strip("'")
    return fields, text[end + 4:].lstrip("\n")


def _sections(body: str) -> List[Tuple[str, str]]:
    """(heading, text) for each `## ` section; the part before the first heading has heading ''."""
    parts, heading, lines = [], "", []
    fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            parts.append((heading, "\n".join(lines)))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    parts.append((heading, "\n".join(lines)))
    return parts


def _plain(heading: str) -> str:
    return re.sub(r"[^a-z& ]", "", heading.lower()).strip()


def _drop_code(text: str) -> str:
    return re.sub(r"```.*?```", "_(example left out)_", text, flags=re.S)


def parse_role(text: str, division: str, slug: str) -> Optional[Dict[str, Any]]:
    """A role from one roster file, or None when the file is not a role."""
    fields, body = _frontmatter(text)
    if not fields.get("name") or len(body.strip()) < 200:
        return None
    return {"id": f"{division}/{slug}", "name": fields["name"], "division": division,
            "description": fields.get("description", "")[:400], "emoji": fields.get("emoji", ""),
            "vibe": fields.get("vibe", "")[:200], "color": fields.get("color", ""), "text": body}


def role_prompt(role: Dict[str, Any]) -> str:
    """The role's prompt, made honest about what this agent is and can do."""
    kept = []
    for heading, text in _sections(role["text"]):
        if "learning" in _plain(heading) and "memory" in _plain(heading):
            continue                                   # "you learn from every project": it does not
        text = re.sub(r"(?m)^\s*[-*]\s*\*\*(Memory|Experience)\*\*:.*\n?", "", text)
        kept.append((f"## {heading}\n" if heading else "") + text.strip("\n"))
    prompt = "\n\n".join(k for k in kept if k.strip())
    if len(prompt) > LONG_PROMPT:
        prompt = _drop_code(prompt)
    if len(prompt) > MAX_PROMPT:
        cut = prompt.rfind("\n## ", 0, MAX_PROMPT)
        prompt = prompt[:cut if cut > 0 else MAX_PROMPT]
    return prompt.strip() + WORKING_RULES


def role_standards(role: Dict[str, Any]) -> str:
    """The role's own bar for good work: its Success Metrics and Critical Rules, for the critic."""
    picked = [f"{heading}:\n{_drop_code(text).strip()}" for heading, text in _sections(role["text"])
              if any(word in _plain(heading) for word in ("success metric", "critical rule", "quality standard"))]
    out = "\n\n".join(picked)
    return out[:STANDARDS_LIMIT].rsplit("\n", 1)[0] if len(out) > STANDARDS_LIMIT else out


# --------------------------------------------------------------------------
# importing
# --------------------------------------------------------------------------

def _labels(files: Dict[str, str]) -> Dict[str, str]:
    raw = files.get("divisions.json")
    try:
        found = json.loads(raw).get("divisions", {}) if raw else {}
        return {k: v.get("label", k) for k, v in found.items()}
    except (ValueError, AttributeError):
        return {}


def _build(files: Dict[str, str], source: str) -> Dict[str, Any]:
    """The roster index from {relative path: text}."""
    roles, seen = [], set()
    for path in sorted(files):
        parts = path.split("/")
        if len(parts) < 2 or not path.endswith(".md") or parts[0] in SKIP or parts[0].startswith("."):
            continue
        slug = re.sub(r"[^a-z0-9-]", "", Path(parts[-1]).stem.lower())
        role = parse_role(files[path], parts[0], slug)
        if role and role["id"] not in seen:
            seen.add(role["id"])
            roles.append(role)
    if not roles:
        raise ValueError("No roles were found: a roster is a folder of Markdown files, each starting with "
                         "frontmatter that has a name.")
    labels = _labels(files)
    license_text = next((files[k] for k in ("LICENSE", "LICENSE.md", "LICENSE.txt") if k in files), "")
    return {"source": source, "imported": time.time(), "license": license_text[:3000],
            "divisions": {d: labels.get(d, d.replace("-", " ").title()) for d in sorted({r["division"] for r in roles})},
            "roles": roles}


def _save(index: Dict[str, Any]) -> Dict[str, Any]:
    _index_path().write_text(json.dumps(index))
    return summary()


def import_folder(folder: str) -> Dict[str, Any]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"There is no folder at {root}.")
    files = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and (path.suffix == ".md" or rel in ("divisions.json", "LICENSE")) and path.stat().st_size < 400_000:
            files[rel] = path.read_text(errors="replace")
    return _save(_build(files, str(root)))


def import_zip(data: bytes, source: str) -> Dict[str, Any]:
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as box:
        for info in box.infolist():
            name = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename  # drop repo-main/
            if info.is_dir() or info.file_size > 400_000:
                continue
            if name.endswith(".md") or name in ("divisions.json", "LICENSE"):
                files[name] = box.read(info).decode("utf-8", errors="replace")
    return _save(_build(files, source))


def import_github(repo: str = DEFAULT_REPO, ref: str = "main") -> Dict[str, Any]:
    repo = re.sub(r"^(https?://)?github\.com/", "", repo.strip()).strip("/")
    repo = re.sub(r"\.git$", "", repo)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Name the repository as owner/name, such as msitarzewski/agency-agents.")
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{ref}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"user-agent": "dnn-designer"}), timeout=60) as r:
            data = r.read()
    except OSError as exc:
        raise ValueError(f"Could not download {repo} from GitHub: {exc}") from None
    return import_zip(data, f"https://github.com/{repo}")


# --------------------------------------------------------------------------
# reading the roster
# --------------------------------------------------------------------------

def _index() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_index_path().read_text())
    except (OSError, ValueError):
        return None


def summary() -> Dict[str, Any]:
    index = _index()
    if not index:
        return {"imported": False, "count": 0, "default_repo": DEFAULT_REPO}
    counts: Dict[str, int] = {}
    for r in index["roles"]:
        counts[r["division"]] = counts.get(r["division"], 0) + 1
    return {"imported": True, "count": len(index["roles"]), "source": index["source"], "when": index["imported"],
            "divisions": [{"key": k, "label": v, "count": counts.get(k, 0)} for k, v in index["divisions"].items()],
            "default_repo": DEFAULT_REPO}


def roles(query: str = "", division: str = "", limit: int = 400) -> List[Dict[str, Any]]:
    index = _index() or {"roles": []}
    words = [w for w in query.lower().split() if w]
    out = []
    for r in index["roles"]:
        if division and r["division"] != division:
            continue
        hay = f"{r['name']} {r['description']} {r['vibe']} {r['division']}".lower()
        if all(w in hay for w in words):
            out.append({k: r[k] for k in ("id", "name", "division", "description", "emoji", "vibe")})
    return out[:limit]


def role(role_id: str) -> Dict[str, Any]:
    for r in (_index() or {"roles": []})["roles"]:
        if r["id"] == role_id:
            return r
    raise KeyError(role_id)


def remove() -> Dict[str, Any]:
    _index_path().unlink(missing_ok=True)
    return summary()


def memory_dir(role_id: str) -> Path:
    """Each specialist keeps its own long-term notes."""
    path = _dir() / "specialists" / re.sub(r"[^\w.-]", "_", role_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def graph(role_id: str) -> Dict[str, Any]:
    """The specialist as an agent design the lab can run, show and edit."""
    import agentlab
    r = role(role_id)
    design = agentlab.template("specialist")
    design["name"] = f"{r['emoji']} {r['name']}".strip()
    design["role"] = r["id"]
    design["about"] = r["description"]
    for node in design["nodes"]:
        if node["type"] == "system_prompt":
            node["params"]["text"] = role_prompt(r)
        if node["type"] == "reflector":
            node["params"]["standards"] = role_standards(r)
    return design


def roles_for(ids: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = set(ids)
    return [r for r in roles() if r["id"] in wanted]
