"""Model providers: where agents, judges and the assistant send their model calls.

Two wire formats cover almost everything. Anthropic's Messages API, and the OpenAI
chat-completions shape that OpenAI, xAI (Grok), Google Gemini, Mistral, DeepSeek,
Groq, OpenRouter, Together, and local servers such as Ollama and LM Studio all
speak. So a provider here is only a name, a format, a base URL, and the
environment variable its key would live in.

Keys set in Settings are kept in the workspace's settings file, readable only by
you, and are never sent back to the page in full or written into a generated
file. A key in the server's environment works too, and wins when both exist —
so an install configured the old way keeps working.
"""

from __future__ import annotations

import json
import os
import threading
import time
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ANTHROPIC, OPENAI = "anthropic", "openai"

CATALOG: Dict[str, Dict[str, Any]] = {
    "anthropic": {"name": "Anthropic (Claude)", "kind": ANTHROPIC, "base_url": "https://api.anthropic.com",
                  "key_env": "ANTHROPIC_API_KEY", "keys_at": "https://console.anthropic.com/settings/keys",
                  "suggested": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]},
    "openai": {"name": "OpenAI", "kind": OPENAI, "base_url": "https://api.openai.com/v1",
               "key_env": "OPENAI_API_KEY", "keys_at": "https://platform.openai.com/api-keys"},
    "xai": {"name": "xAI (Grok)", "kind": OPENAI, "base_url": "https://api.x.ai/v1",
            "key_env": "XAI_API_KEY", "keys_at": "https://console.x.ai"},
    "gemini": {"name": "Google Gemini", "kind": OPENAI,
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINI_API_KEY", "keys_at": "https://aistudio.google.com/apikey"},
    "mistral": {"name": "Mistral", "kind": OPENAI, "base_url": "https://api.mistral.ai/v1",
                "key_env": "MISTRAL_API_KEY", "keys_at": "https://console.mistral.ai/api-keys"},
    "deepseek": {"name": "DeepSeek", "kind": OPENAI, "base_url": "https://api.deepseek.com/v1",
                 "key_env": "DEEPSEEK_API_KEY", "keys_at": "https://platform.deepseek.com/api_keys"},
    "groq": {"name": "Groq", "kind": OPENAI, "base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "keys_at": "https://console.groq.com/keys"},
    "openrouter": {"name": "OpenRouter", "kind": OPENAI, "base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY", "keys_at": "https://openrouter.ai/keys"},
    "together": {"name": "Together AI", "kind": OPENAI, "base_url": "https://api.together.xyz/v1",
                 "key_env": "TOGETHER_API_KEY", "keys_at": "https://api.together.ai/settings/api-keys"},
    "ollama": {"name": "Ollama (on this computer)", "kind": OPENAI, "base_url": "http://localhost:11434/v1",
               "key_env": "", "local": True, "keys_at": "https://ollama.com/download"},
    "lmstudio": {"name": "LM Studio (on this computer)", "kind": OPENAI, "base_url": "http://localhost:1234/v1",
                 "key_env": "", "local": True, "keys_at": "https://lmstudio.ai"},
    "custom": {"name": "Other OpenAI-compatible server", "kind": OPENAI, "base_url": "http://localhost:8000/v1",
               "key_env": "CUSTOM_LLM_API_KEY", "key_optional": True},
}

# Models worth pulling into Ollama for agents. Agents call tools, so these all support tool calling.
OLLAMA_SUGGESTED = [
    {"name": "llama3.1:8b", "note": "Meta Llama 3.1, 8B: a solid all-rounder with tool calling"},
    {"name": "llama3.2:3b", "note": "Meta Llama 3.2, 3B: small and quick"},
    {"name": "qwen2.5:7b", "note": "Qwen 2.5, 7B: strong at tools and structured answers"},
    {"name": "qwen2.5-coder:7b", "note": "Qwen 2.5 Coder, 7B: for code"},
    {"name": "mistral:7b", "note": "Mistral 7B: fast general model"},
    {"name": "mistral-nemo:12b", "note": "Mistral NeMo, 12B: longer context"},
]

DEFAULTS = {"provider": "anthropic", "model": "claude-sonnet-5-5", "temperature": 0.3, "max_tokens": 2048,
            "timeout": 120, "retries": 2, "judge_provider": "", "judge_model": "",
            "talk_mode": "live", "assistant_llm": False}

# What the lab's LLM cores have always been able to say, mapped onto the catalog.
LEGACY = {"openai-compatible": "custom"}


# --------------------------------------------------------------------------
# the settings file
# --------------------------------------------------------------------------

_LOCK = threading.Lock()


def _path() -> Path:
    # beside the lab's other settings, in a folder git ignores, so a key never rides along in a commit
    import auth
    return auth.sub("agentlab") / "model_settings.json"


def load() -> Dict[str, Any]:
    try:
        data = json.loads(_path().read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("providers", {})
    data["defaults"] = {**DEFAULTS, **(data.get("defaults") or {})}
    return data


def _write(data: Dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def key_for(pid: str, data: Optional[Dict[str, Any]] = None) -> str:
    """The key a provider's calls use: the environment's, else the one saved in Settings."""
    pid = LEGACY.get(pid, pid)
    info = CATALOG.get(pid) or {}
    env = info.get("key_env") or ""
    if env and os.environ.get(env):
        return os.environ[env]
    return str(((data or load())["providers"].get(pid) or {}).get("api_key") or "")


def base_url_for(pid: str, data: Optional[Dict[str, Any]] = None) -> str:
    pid = LEGACY.get(pid, pid)
    saved = ((data or load())["providers"].get(pid) or {}).get("base_url")
    return str(saved or (CATALOG.get(pid) or {}).get("base_url") or "")


def ready(pid: str, data: Optional[Dict[str, Any]] = None) -> bool:
    """Whether calls to this provider could go out: a key where one is needed, and switched on."""
    pid = LEGACY.get(pid, pid)
    info = CATALOG.get(pid)
    if not info:
        return False
    data = data or load()
    saved = data["providers"].get(pid) or {}
    if saved.get("enabled") is False:
        return False
    if info.get("local") or info.get("key_optional"):
        return bool(saved.get("enabled")) or bool(os.environ.get(info.get("key_env") or "-"))
    return bool(key_for(pid, data))


def _mask(key: str) -> str:
    return (key[:3] + "…" + key[-4:]) if len(key) > 10 else ("•" * len(key))


def view() -> Dict[str, Any]:
    """Everything the Settings page shows. Keys only ever leave as a masked hint."""
    data = load()
    out = []
    for pid, info in CATALOG.items():
        saved = data["providers"].get(pid) or {}
        env = info.get("key_env") or ""
        out.append({"id": pid, **{k: v for k, v in info.items() if k != "suggested"},
                    "suggested": info.get("suggested", []), "enabled": saved.get("enabled", None),
                    "base_url": saved.get("base_url") or info["base_url"], "default_base_url": info["base_url"],
                    "has_key": bool(saved.get("api_key")), "key_hint": _mask(str(saved.get("api_key") or "")),
                    "key_in_env": bool(env and os.environ.get(env)), "models": saved.get("models") or [],
                    "checked": saved.get("checked"), "ready": ready(pid, data)})
    return {"providers": out, "defaults": data["defaults"], "ollama_suggested": OLLAMA_SUGGESTED}


def save(update: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a partial update. A provider's api_key is changed only when sent; "" removes it."""
    with _LOCK:
        data = load()
        for pid, change in (update.get("providers") or {}).items():
            if pid not in CATALOG or not isinstance(change, dict):
                continue
            mine = data["providers"].setdefault(pid, {})
            if "api_key" in change:
                key = str(change["api_key"] or "").strip()
                if key:
                    mine["api_key"] = key
                else:
                    mine.pop("api_key", None)
            if "base_url" in change:
                url = str(change["base_url"] or "").strip().rstrip("/")
                if url and not url.startswith(("http://", "https://")):
                    raise ValueError(f"{CATALOG[pid]['name']}: the address must start with http:// or https://")
                if url and url != CATALOG[pid]["base_url"]:
                    mine["base_url"] = url
                else:
                    mine.pop("base_url", None)
            if "enabled" in change:
                mine["enabled"] = bool(change["enabled"])
        defaults = update.get("defaults") or {}
        for k, v in defaults.items():
            if k not in DEFAULTS:
                continue
            if k in ("temperature",):
                v = max(0.0, min(2.0, float(v)))
            elif k in ("max_tokens", "timeout", "retries"):
                v = int(v)
                v = {"max_tokens": max(16, min(v, 200000)), "timeout": max(10, min(v, 1800)),
                     "retries": max(0, min(v, 6))}[k]
            elif k in ("provider", "judge_provider"):
                v = str(v or "")
                if v and v not in CATALOG:
                    raise ValueError(f"Unknown provider {v}.")
            elif k == "talk_mode":
                v = "live" if v == "live" else "rehearsal"
            elif k == "assistant_llm":
                v = bool(v)
            else:
                v = str(v or "").strip()
            data["defaults"][k] = v
        _write(data)
    return view()


# --------------------------------------------------------------------------
# talking to a provider
# --------------------------------------------------------------------------

# A proxy set in the shell (HTTP_PROXY and friends) must never get between the server and a model
# running on this computer: it would answer "not running" for an Ollama that is up.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req, timeout: float):
    host = urllib.parse.urlsplit(req.full_url).hostname or ""
    local = host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")
    return (_DIRECT if local else urllib.request.build_opener()).open(req, timeout=timeout)


def _request(url: str, body: Optional[Dict[str, Any]], headers: Dict[str, str], timeout: float):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers},
                                 method="POST" if body is not None else "GET")
    with _open(req, timeout) as reply:
        return json.loads(reply.read() or b"{}")


def _headers(pid: str, key: str) -> Dict[str, str]:
    if CATALOG[pid]["kind"] == ANTHROPIC:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"authorization": "Bearer " + key} if key else {}


def _why(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode(errors="replace")[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        hint = {401: "the key was refused", 403: "the key is not allowed to do this",
                404: "nothing answers at that address", 429: "rate limited"}.get(exc.code, "")
        return f"HTTP {exc.code}{': ' + hint if hint else ''}. {detail}".strip()
    if isinstance(exc, urllib.error.URLError):
        return f"could not connect ({exc.reason})"
    return f"{type(exc).__name__}: {exc}"


def list_models(pid: str) -> Dict[str, Any]:
    """Ask a provider which models this key can use, and remember the answer."""
    pid = LEGACY.get(pid, pid)
    if pid not in CATALOG:
        raise ValueError(f"Unknown provider {pid}.")
    info, data = CATALOG[pid], load()
    key = key_for(pid, data)
    if not key and not (info.get("local") or info.get("key_optional")):
        return {"ok": False, "error": "Add an API key first."}
    base = base_url_for(pid, data).rstrip("/")
    url = base + ("/v1/models" if info["kind"] == ANTHROPIC else "/models")
    started = time.time()
    try:
        reply = _request(url, None, _headers(pid, key), 20)
    except Exception as exc:  # noqa: BLE001 - the page shows why
        return {"ok": False, "error": _why(exc), "url": url}
    models = sorted({str(m.get("id") or m.get("name") or "") for m in (reply.get("data") or reply.get("models") or [])} - {""})
    with _LOCK:
        data = load()
        mine = data["providers"].setdefault(pid, {})
        mine.update(models=models[:400], checked=time.time())
        if info.get("local") or info.get("key_optional"):
            mine.setdefault("enabled", True)
        _write(data)
    return {"ok": True, "models": models, "ms": round((time.time() - started) * 1000)}


def complete(pid: str, model: str, system: str, prompt: str, max_tokens: int = 200,
             temperature: float = 0.0, timeout: Optional[float] = None) -> Dict[str, Any]:
    """One plain model call, for judges, the assistant and the Try it button. Raises ValueError."""
    pid = LEGACY.get(pid, pid)
    if pid not in CATALOG:
        raise ValueError(f"Unknown provider {pid}.")
    data = load()
    info, key = CATALOG[pid], key_for(pid, data)
    if not key and not (info.get("local") or info.get("key_optional")):
        raise ValueError(f"{info['name']} needs an API key: add one in Settings.")
    base = base_url_for(pid, data).rstrip("/")
    timeout = timeout or float(data["defaults"]["timeout"])
    try:
        if info["kind"] == ANTHROPIC:
            reply = _request(base + "/v1/messages", {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                             "system": system, "messages": [{"role": "user", "content": prompt}]},
                             _headers(pid, key), timeout)
            text = "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")
            usage = reply.get("usage") or {}
            tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        else:
            reply = _request(base + "/chat/completions", {"model": model, "max_tokens": max_tokens,
                             "temperature": temperature, "messages": [{"role": "system", "content": system},
                                                                      {"role": "user", "content": prompt}]},
                             _headers(pid, key), timeout)
            text = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            usage = reply.get("usage") or {}
            tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{info['name']}: {_why(exc)}") from None
    return {"text": text.strip(), "tokens": tokens}


def try_model(pid: str, model: str) -> Dict[str, Any]:
    started = time.time()
    try:
        out = complete(pid, model, "You are being tested. Reply in one short sentence.",
                       "Say hello and name the model you are.", max_tokens=60, temperature=0.2, timeout=60)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "text": out["text"], "ms": round((time.time() - started) * 1000)}


def resolve(provider: str, model: str = "", base_url: str = "") -> Dict[str, Any]:
    """Where an LLM core's calls go: its own provider, or the default from Settings."""
    data = load()
    pid = provider or "default"
    if pid == "default":
        pid = data["defaults"]["provider"] or "anthropic"
        model = model or data["defaults"]["model"]
    pid = LEGACY.get(pid, pid)
    info = CATALOG.get(pid) or CATALOG["anthropic"]
    url = (base_url if provider in LEGACY and base_url else "") or base_url_for(pid, data)
    return {"provider": pid, "kind": info["kind"], "base_url": url.rstrip("/"), "key_env": info.get("key_env") or "",
            "model": model, "name": info["name"], "needs_key": not (info.get("local") or info.get("key_optional"))}


def keys_for_run() -> Dict[str, str]:
    """The keys a run in the app may use, by environment-variable name. Never written to a file."""
    data = load()
    out = {}
    for pid, info in CATALOG.items():
        env = info.get("key_env")
        key = key_for(pid, data)
        if env and key:
            out[env] = key
    return out


# --------------------------------------------------------------------------
# Ollama: models on this computer
# --------------------------------------------------------------------------

def _ollama_root() -> str:
    base = base_url_for("ollama")
    return base[:-3] if base.rstrip("/").endswith("/v1") else base.rstrip("/")


def _ollama_candidates(root: str) -> List[str]:
    """The address in Settings, then the same port by its other local names.

    "localhost" can resolve to IPv6 first while Ollama listens on IPv4 only, or the other way
    round; trying each spelling finds it whichever this Mac prefers.
    """
    out = [root]
    parts = urllib.parse.urlsplit(root)
    if parts.hostname in ("localhost", "127.0.0.1", "::1"):
        port = parts.port or 11434
        for host in ("127.0.0.1", "localhost", "[::1]"):
            alt = f"{parts.scheme}://{host}:{port}"
            if alt not in out:
                out.append(alt)
    return out


def ollama_installed() -> Dict[str, Any]:
    """Whether Ollama is on this computer, found the way a Mac or Linux install puts it."""
    app = Path("/Applications/Ollama.app")
    found = shutil.which("ollama")
    if not found:
        for guess in ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", str(app / "Contents/Resources/ollama"),
                      os.path.expanduser("~/.local/bin/ollama")):
            if os.path.exists(guess):
                found = guess
                break
    return {"installed": bool(found or app.exists()), "binary": found, "app": str(app) if app.exists() else None}


def ollama_status() -> Dict[str, Any]:
    root = _ollama_root()
    errors = []
    for candidate in _ollama_candidates(root):
        try:
            version = _request(candidate + "/api/version", None, {}, 4).get("version")
            tags = _request(candidate + "/api/tags", None, {}, 10)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {_why(exc)}")
            continue
        if candidate != root:
            # found it under another name for this computer: remember that one
            save({"providers": {"ollama": {"base_url": candidate + "/v1"}}})
            root = candidate
        break
    else:
        return {"running": False, "url": root, "error": "; ".join(errors), "models": [], "pulls": _pull_view(),
                **ollama_installed()}
    models = [{"name": m.get("name"), "size": m.get("size"), "modified": m.get("modified_at"),
               "family": (m.get("details") or {}).get("family"),
               "parameters": (m.get("details") or {}).get("parameter_size"),
               "quantization": (m.get("details") or {}).get("quantization_level")} for m in tags.get("models") or []]
    with _LOCK:
        data = load()
        mine = data["providers"].setdefault("ollama", {})
        if mine.get("enabled") is None:
            mine["enabled"] = True          # it answered, so agents may use it
            _write(data)
    return {"running": True, "url": root, "version": version, "models": models, "pulls": _pull_view(),
            **ollama_installed()}


def ollama_start() -> Dict[str, Any]:
    """Start Ollama when it is installed but not running, and wait for it to answer."""
    if ollama_status()["running"]:
        return ollama_status()
    where = ollama_installed()
    if not where["installed"]:
        raise ValueError("Ollama is not installed on this computer. Download it from ollama.com, then try again.")
    if sys.platform == "darwin" and where["app"]:
        cmd = ["open", "-a", "Ollama"]                      # the menu-bar app runs the server
    elif where["binary"]:
        cmd = [where["binary"], "serve"]
    else:
        raise ValueError("Ollama is installed but its program could not be found. Open the Ollama app yourself.")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as exc:
        raise ValueError(f"Could not start Ollama: {exc}") from None
    for _ in range(40):
        time.sleep(0.5)
        st = ollama_status()
        if st["running"]:
            return {**st, "started": True}
    raise ValueError("Ollama was started but has not answered after 20 seconds. Open the Ollama app and check it.")


_PULLS: Dict[str, Dict[str, Any]] = {}


def _pull_view() -> List[Dict[str, Any]]:
    return [dict(v) for v in _PULLS.values()]


def ollama_pull(name: str) -> Dict[str, Any]:
    """Download a model into Ollama in the background; follow it with ollama_status()."""
    name = str(name or "").strip()
    if not name or any(c.isspace() for c in name):
        raise ValueError("Give a model name such as qwen2.5:7b.")
    if (_PULLS.get(name) or {}).get("status") == "pulling":
        return dict(_PULLS[name])
    job = _PULLS[name] = {"name": name, "status": "pulling", "done": 0, "total": 0, "message": "starting", "error": None}
    root = _ollama_root()

    def run():
        try:
            req = urllib.request.Request(root + "/api/pull", data=json.dumps({"name": name, "stream": True}).encode(),
                                         headers={"content-type": "application/json"})
            with _open(req, 3600) as reply:
                for line in reply:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise ValueError(msg["error"])
                    job.update(message=msg.get("status") or job["message"],
                               done=msg.get("completed") or job["done"], total=msg.get("total") or job["total"])
            job.update(status="done", message="ready")
        except Exception as exc:  # noqa: BLE001
            job.update(status="error", error=_why(exc) if not isinstance(exc, ValueError) else str(exc))

    threading.Thread(target=run, daemon=True).start()
    return dict(job)


def ollama_delete(name: str) -> Dict[str, Any]:
    root = _ollama_root()
    req = urllib.request.Request(root + "/api/delete", data=json.dumps({"name": name}).encode(),
                                 headers={"content-type": "application/json"}, method="DELETE")
    try:
        with _open(req, 30):
            pass
    except Exception as exc:  # noqa: BLE001
        raise ValueError(_why(exc)) from None
    _PULLS.pop(name, None)
    return {"ok": True}
