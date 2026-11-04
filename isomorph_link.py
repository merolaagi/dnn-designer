"""Send a checkpoint's weights to Isomorph, so its weights can be read there.

Isomorph (github.com/merolaagi/isomorph) runs on the same Mac. Its Weight space
tab has Claude read a network's weights through fixed tools (statistics,
singular vectors, spectra, how one matrix acts on another's output) and say
what each part does. It does not know how a designer graph is wired, so this
sends the weights as safetensors plus a few lines describing the layers in
order, and Isomorph reads the weights alone.

Settings:
  ISOMORPH_URL         where this server reaches Isomorph (default http://127.0.0.1:47821)
  ISOMORPH_PUBLIC_URL  where the browser opens it (default: ISOMORPH_URL); set it when
                       Isomorph is served on its own hostname
  ISOMORPH_PASSWORD    if Isomorph has a password set
"""
import base64
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "http://127.0.0.1:47821"


def base_url():
    return os.environ.get("ISOMORPH_URL", DEFAULT_URL).rstrip("/")


def public_url():
    return os.environ.get("ISOMORPH_PUBLIC_URL", base_url()).rstrip("/")


def to_safetensors(state_dict):
    """Float tensors as a safetensors file (float32), written by hand so no extra package is needed."""
    import torch

    header, blobs, offset = {}, [], 0
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            continue
        raw = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if not header:
        raise ValueError("The checkpoint holds no floating-point weights.")
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


def describe(graph, metrics=None, limit=1500):
    """The layers in the order they appear, with their settings, in a few lines."""
    nodes = (graph or {}).get("nodes") or []
    lines = []
    for n in nodes:
        p = {k: v for k, v in (n.get("params") or {}).items() if not isinstance(v, (dict, list)) or len(str(v)) < 40}
        lines.append(f"{n.get('type', '?')}" + (f" {json.dumps(p, separators=(',', ':'))}" if p else ""))
    text = "A network built in Deep Network Designer. Layers, in order: " + " -> ".join(lines)
    if metrics:
        keep = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k in ("val_loss", "val_acc", "train_loss", "train_acc")}
        if keep:
            text += f". Last measured: {json.dumps(keep)}"
    return text[:limit]


def send(path, name=""):
    """Load a checkpoint file and import it into Isomorph. Returns Isomorph's answer plus the link to open."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else None
    if not isinstance(sd, dict):
        raise ValueError("That checkpoint does not hold a state_dict.")
    body = to_safetensors(sd)
    q = urllib.parse.urlencode({"name": name or path.stem, "about": describe(blob.get("graph"), blob.get("metrics")),
                                "source": "Deep Network Designer"})
    headers = {"content-type": "application/octet-stream"}
    pw = os.environ.get("ISOMORPH_PASSWORD")
    if pw:
        headers["authorization"] = "Basic " + base64.b64encode(f"dnn:{pw}".encode()).decode()
    req = urllib.request.Request(f"{base_url()}/api/wsp/import?{q}", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            info = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        if e.code == 401:
            raise ConnectionError("Isomorph asked for a password: set ISOMORPH_PASSWORD for this server.")
        if e.code == 404:
            raise ConnectionError("That Isomorph is older than 0.12.0 and has no Weight space tab. Upgrade it.")
        raise ConnectionError(f"Isomorph refused the weights ({e.code}): {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(f"Isomorph is not answering at {base_url()} ({getattr(e, 'reason', e)}). "
                              "Start it with ./run.sh in ~/Sites/isomorph, or set ISOMORPH_URL.")
    info["url"] = f"{public_url()}/#wsp/read/{info['spec']}"
    return info


def get_json(path, timeout=10):
    """GET an Isomorph API path; raises ConnectionError with a plain reason."""
    headers = {}
    pw = os.environ.get("ISOMORPH_PASSWORD")
    if pw:
        headers["authorization"] = "Basic " + base64.b64encode(f"dnn:{pw}".encode()).decode()
    req = urllib.request.Request(f"{base_url()}{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ConnectionError("That Isomorph is older than 0.13.0 and cannot share its laws. Upgrade it.")
        raise ConnectionError(f"Isomorph answered {e.code}.")
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(f"Isomorph is not answering at {base_url()} ({getattr(e, 'reason', e)}).")
