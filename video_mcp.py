"""An MCP server (stdio) for the Video agent: watch a video from a link. Started as `python3 video_mcp.py`."""

import json
import sys

import video

TOOLS = [
    {"name": "watch_video", "description": "Watch a video from its link (Facebook video, Reel or fb.watch, Instagram, "
     "YouTube, TikTok, X): fetches it on the Mac, and returns its title, caption and who posted it, what is on screen "
     "every few seconds, and everything said with times. Takes half a minute to a few minutes.",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string", "description": "The video's link, exactly as the person gave it"},
         "question": {"type": "string", "description": "What they want to know about it, if they said"}},
         "required": ["url"]}},
    {"name": "saved_video", "description": "A video watched before, by its id (vd…), without fetching it again.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


def call(name, a):
    if name == "watch_video":
        return video.report_text(video.describe(a.get("url") or "", a.get("question") or ""))
    if name == "saved_video":
        return video.report_text(video.load(a.get("id") or ""))
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "video", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except KeyError:
            text, error = "There is no saved video with that id.", True
        except (ValueError, RuntimeError) as exc:
            text, error = str(exc), True
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
