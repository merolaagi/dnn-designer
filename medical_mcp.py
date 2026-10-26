"""An MCP server (stdio) for the Medical agent: the literature, drug labels and MediConsult. Started as `python3 medical_mcp.py`."""

import json
import sys

import medical

TOOLS = [
    {"name": "search_literature", "description": "Search the medical literature (PubMed, then Europe PMC): the best matches "
     "with title, journal, year, kind (guideline, systematic review, trial) and abstract, each with its link.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search terms: the condition, treatment and population, e.g. "
                   "'SGLT2 inhibitors chronic kidney disease elderly'"},
         "kind": {"type": "string", "description": "any (default), guideline, review (systematic reviews and meta-analyses) "
                  "or trial"},
         "years": {"type": "integer", "description": "Only the last N years; 0 for any"}},
         "required": ["query"]}},
    {"name": "read_article", "description": "One PubMed article in full: title, journal, year, kind and the whole abstract.",
     "inputSchema": {"type": "object", "properties": {"pmid": {"type": "string"}}, "required": ["pmid"]}},
    {"name": "drug_label", "description": "What the FDA label says for a medicine (generic or brand name): uses, dosing as "
     "the label states it, use in children, boxed warning, do-not-use, warnings, interactions, pregnancy, side effects.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "ask_mediconsult", "description": "Ask the person's own MediConsult AI (when connected): its answer from its "
     "own library of guidelines and papers, with its sources. One source among the others, not the last word.",
     "inputSchema": {"type": "object", "properties": {
         "question": {"type": "string"},
         "specialty": {"type": "string", "description": "e.g. cardiology, endocrinology, neurology, pediatrics; optional"}},
         "required": ["question"]}},
]


def call(name, a):
    if name == "search_literature":
        rows = medical.literature(a.get("query") or "", a.get("kind") or "any", int(a.get("years") or 0))
        return medical.articles_text(rows) if rows else "Nothing matched. Try broader or different terms."
    if name == "read_article":
        return medical.articles_text([medical.article(a.get("pmid") or "")])
    if name == "drug_label":
        return medical.label_text(medical.drug_label(a.get("name") or ""))
    if name == "ask_mediconsult":
        mc = medical.mediconsult(a.get("question") or "", a.get("specialty") or "")
        return mc["answer"] + ("\n\nIts sources: " + "; ".join(f"{c['title']} ({c.get('year') or ''}) {c.get('url') or ''}"
                                                              for c in mc["citations"]) if mc["citations"] else "")
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "medical", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except (ValueError, RuntimeError) as exc:
            text, error = str(exc), True
        except Exception as exc:  # noqa: BLE001 - a source that fails says so; the agent can try another
            text, error = f"{type(exc).__name__}: {exc}", True
        result = {"content": [{"type": "text", "text": text}], "isError": error}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
