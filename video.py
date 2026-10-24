"""Video: watches a video from a link (Facebook, Instagram, YouTube, TikTok, X) and says what is in it.

All of it runs on the Mac:
  1. yt-dlp fetches the video. A public one needs nothing; a friends-only or group video uses your
     Facebook or Instagram sign-in from Saathi's browser (only those two sites' cookies, in a file
     that is deleted straight after).
  2. ffmpeg takes out the sound, and Whisper writes down what is said, in any language
     (mlx-whisper on Apple Silicon, faster-whisper elsewhere).
  3. ffmpeg takes a still every few seconds, and the vision model in Ollama (the one that reads
     invoices, Settings → Reading invoices) says what each one shows, with any text on screen.
  4. The Video agent puts it together: what it is, a summary, the key moments with times, what was said.
The video, the sound and the stills are deleted afterwards; the transcript and the scene notes are
kept under Files.

Install once on the Mac: ./serve.sh setup-video
"""

from __future__ import annotations

import base64
import html as H
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

HERE = Path(__file__).resolve().parent
DEFAULTS = {"whisper": "mlx-community/whisper-large-v3-turbo", "whisper_cpu": "small", "frames": 16,
            "max_minutes": 60}
SITES = re.compile(r"https?://(?:[a-z0-9-]+\.)*(?:facebook\.com|fb\.watch|fb\.com|instagram\.com|youtube\.com|youtu\.be|"
                   r"tiktok\.com|x\.com|twitter\.com|vimeo\.com|reddit\.com|v\.redd\.it|threads\.net|threads\.com)/\S+", re.I)
LOGIN = re.compile(r"log ?in|sign ?in|cookies|private|registered users|only available (to|for)|not available|"
                   r"permission|age[- ]restrict|checkpoint", re.I)
FRAME_PROMPT = ("This is one still from a video, at {at}. In one or two plain sentences say what it shows: the people "
                "(what they look like and are doing, never who they are by name unless a caption says so), the place, "
                "objects and anything happening. Then copy any words on screen exactly, after 'Text:'. If nothing is "
                "on screen, leave Text out.")
_ID = re.compile(r"vd[0-9a-f]{10}")


class VideoError(RuntimeError):
    pass


def _cfg_path() -> Path:
    (HERE / "data").mkdir(exist_ok=True)
    return HERE / "data" / "video.json"


def config() -> Dict[str, Any]:
    try:
        saved = json.loads(_cfg_path().read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


# --------------------------------------------------------------------------
# what is installed
# --------------------------------------------------------------------------

def ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).exists()), None)
    if found:
        return found
    if importlib.util.find_spec("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return None
    return None


def whisper() -> Optional[str]:
    if importlib.util.find_spec("mlx_whisper"):
        return "mlx"
    if importlib.util.find_spec("faster_whisper"):
        return "faster"
    return None


def status() -> Dict[str, Any]:
    return {"yt_dlp": importlib.util.find_spec("yt_dlp") is not None, "ffmpeg": bool(ffmpeg()), "whisper": whisper()}


def missing() -> Optional[str]:
    s = status()
    lack = [name for name, ok in (("yt-dlp", s["yt_dlp"]), ("ffmpeg", s["ffmpeg"])) if not ok]
    if lack:
        return (f"The video tools aren't installed on the Mac yet ({', '.join(lack)} missing), so I haven't watched it. "
                "On the Mac, in ~/Sites/new_dnn, run: ./serve.sh setup-video, then ./serve.sh. Then send me the link again.")
    return None


def links(text: str) -> List[str]:
    return [m.group(0).rstrip(").,>\"'") for m in SITES.finditer(str(text or ""))]


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------

def _cookie_file(folder: Path) -> Optional[Path]:
    """Your Facebook and Instagram sign-in from Saathi's browser, as a cookies file for yt-dlp; None if there is none."""
    try:
        import browser
        if not browser.installed():
            return None
        cookies = browser.call("video_cookies").get("cookies") or []
    except Exception:  # noqa: BLE001
        return None
    if not cookies:
        return None
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = str(c.get("domain") or "")
        lines.append("\t".join([domain, "TRUE" if domain.startswith(".") else "FALSE", c.get("path") or "/",
                                "TRUE" if c.get("secure") else "FALSE", str(int(c.get("expires") or 0) if (c.get("expires") or 0) > 0 else 0),
                                str(c.get("name") or ""), str(c.get("value") or "")]))
    path = folder / "cookies.txt"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return path


def length_filter(max_minutes: int) -> str:
    """yt-dlp's filter: skip a video over three times the limit; one that does not say its length still goes."""
    return f"duration <=? {int(max_minutes) * 60 * 3}"


def fetch(url: str, folder: Path, max_minutes: int) -> Dict[str, Any]:
    """Download the video (720p at most) and its details. Tries without your sign-in first."""
    ff = ffmpeg()
    base = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress", "--no-warnings", "--write-info-json",
            "-f", "b[height<=720][ext=mp4]/b[height<=720]/bv*[height<=720]+ba/b", "--max-filesize", "1500M",
            "--match-filter", length_filter(max_minutes),
            "-o", str(folder / "video.%(ext)s")] + (["--ffmpeg-location", ff] if ff else [])
    tried_cookies = False
    for attempt in (0, 1):
        cookies = _cookie_file(folder) if attempt else None
        if attempt and not cookies:
            break
        tried_cookies = bool(cookies)
        try:
            run = subprocess.run(base + (["--cookies", str(cookies)] if cookies else []) + ["--", url],
                                 capture_output=True, text=True, timeout=1200)
        finally:
            if cookies:
                cookies.unlink(missing_ok=True)
        done = [p for p in folder.glob("video.*") if p.suffix not in (".json", ".part", ".ytdl")]
        if run.returncode == 0 and done:
            info = {}
            try:
                info = json.loads(next(folder.glob("*.info.json")).read_text())
            except (StopIteration, OSError, ValueError):
                pass
            return {"path": done[0], "signed_in": tried_cookies, "info": info}
        why = (run.stderr or run.stdout or "").strip().splitlines()
        why = next((l for l in reversed(why) if "ERROR" in l), why[-1] if why else "it gave no reason")
        if not LOGIN.search(why) or attempt:
            break
    hint = (" It may be friends-only or in a group: sign in to Facebook once in Saathi → Hub → Browser, then try again."
            if LOGIN.search(why) and not tried_cookies else "")
    if "does not pass filter" in why:
        raise VideoError(f"It's longer than {max_minutes * 3} minutes, so I didn't fetch it.")
    why = re.sub(r";?\s*please report this issue.*$", "", re.sub(r"^ERROR:\s*", "", why))[:300].rstrip(". ")
    raise VideoError(f"I couldn't fetch the video: {why}.{hint}")


def extract_audio(video: Path, folder: Path, max_minutes: int) -> Optional[Path]:
    out = folder / "audio.wav"
    run = subprocess.run([ffmpeg() or "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                          "-ar", "16000", "-t", str(max_minutes * 60), str(out)], capture_output=True, text=True, timeout=900)
    return out if run.returncode == 0 and out.exists() and out.stat().st_size > 1000 else None


def duration(video: Path, info: Dict[str, Any]) -> float:
    if info.get("duration"):
        return float(info["duration"])
    run = subprocess.run([ffmpeg() or "ffmpeg", "-nostdin", "-i", str(video)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", run.stderr or "")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def _wav(path: Path):
    import wave
    import numpy as np
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav: Path) -> Dict[str, Any]:
    """What is said, with times. mlx-whisper on Apple Silicon (fast), faster-whisper elsewhere."""
    c, kind = config(), whisper()
    if kind is None:
        return {"language": None, "segments": [], "note": "Whisper isn't installed, so I only looked at the pictures "
                                                          "(./serve.sh setup-video installs it)."}
    audio = _wav(wav)
    if kind == "mlx":
        import mlx_whisper
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=c["whisper"], condition_on_previous_text=False)
        segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in r.get("segments") or []]
        return {"language": r.get("language"), "segments": [s for s in segs if s["text"]]}
    from faster_whisper import WhisperModel
    model = WhisperModel(c["whisper_cpu"], device="auto", compute_type="int8")
    segments, info = model.transcribe(audio, vad_filter=True)
    return {"language": info.language, "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()}
                                                    for s in segments if s.text.strip()]}


def grab_frames(video: Path, folder: Path, seconds: float, count: int) -> List[Dict[str, Any]]:
    """A still at evenly spread times (the middle of each stretch, so not the black first frame)."""
    count = max(1, min(40, int(count)))
    if seconds > 0:
        count = max(1, min(count, int(seconds // 3) or 1))
    out = []
    for i in range(count):
        at = (i + 0.5) * seconds / count if seconds > 0 else 0
        path = folder / f"frame{i:02d}.jpg"
        subprocess.run([ffmpeg() or "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "4", str(path)], capture_output=True, timeout=120)
        if path.exists() and path.stat().st_size > 500:
            out.append({"at": at, "path": path})
    return out


def see(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The vision model's note on each still. Stops at the first failure (Ollama off, model missing)."""
    import documents
    import providers
    model = documents.config()["model"]
    notes = []
    for f in frames:
        try:
            reply = providers._request(providers._ollama_root() + "/api/chat", {
                "model": model, "stream": False, "think": False,
                "messages": [{"role": "user", "content": FRAME_PROMPT.format(at=clock(f["at"])),
                              "images": [base64.b64encode(f["path"].read_bytes()).decode()]}],
                "options": {"temperature": 0, "num_predict": 300}}, {}, 240)
        except Exception as exc:  # noqa: BLE001
            why = str(exc)
            if "not found" in why.lower():
                why = f"the vision model {model} isn't in Ollama yet (download it in Settings → Reading invoices)"
            elif "refused" in why or "urlopen" in why:
                why = "Ollama isn't running"
            return {"model": model, "notes": notes, "note": f"I couldn't look at the pictures: {why}."}
        text = ((reply.get("message") or {}).get("content") or "").strip()
        if text:
            notes.append({"at": f["at"], "text": AL._one_line(text, 600)})
    return {"model": model, "notes": notes}


def clock(seconds: float) -> str:
    s = int(seconds or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------

def _dir() -> Path:
    path = AL._dir() / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe(url: str, question: str = "") -> Dict[str, Any]:
    url = str(url or "").strip()
    found = links(url)
    if not found:
        if re.match(r"https?://", url):
            found = [url]
        else:
            raise ValueError("Give the video's link (starting with https://).")
    url = found[0]
    lack = missing()
    if lack:
        raise VideoError(lack)
    c = config()
    vid = "vd" + uuid.uuid4().hex[:10]
    work = _dir() / (vid + "-work")
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        got = fetch(url, work, int(c["max_minutes"]))
        info = got["info"]
        seconds = duration(got["path"], info)
        wav = extract_audio(got["path"], work, int(c["max_minutes"]))
        speech = transcribe(wav) if wav else {"language": None, "segments": [], "note": "The video has no sound."}
        frames = grab_frames(got["path"], work, min(seconds, c["max_minutes"] * 60), int(c["frames"]))
        sight = see(frames) if frames else {"notes": [], "note": "I couldn't take any stills from it."}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    r = {"id": vid, "url": url, "question": str(question or "")[:300], "t": time.time(),
         "seconds_taken": round(time.time() - started, 1), "signed_in": got["signed_in"],
         "title": info.get("title") or info.get("fulltitle") or "", "uploader": info.get("uploader") or info.get("channel") or "",
         "description": AL._one_line(info.get("description") or "", 1500), "upload_date": info.get("upload_date") or "",
         "views": info.get("view_count"), "likes": info.get("like_count"), "site": info.get("extractor_key") or "",
         "duration": seconds, "language": speech.get("language"), "transcript": speech.get("segments") or [],
         "scenes": sight.get("notes") or [], "vision_model": sight.get("model"),
         "notes": [n for n in (speech.get("note"), sight.get("note")) if n]}
    (_dir() / f"{vid}.json").write_text(json.dumps(r, indent=1))
    try:
        import files
        files.record("video", ("Video: " + (r["title"] or r["uploader"] or url))[:200], url=f"/api/video/{vid}",
                     detail=f"{clock(seconds)} · {len(r['transcript'])} lines said · {len(r['scenes'])} scenes", key=f"video:{url}")
    except Exception:  # noqa: BLE001
        pass
    return r


def recent(url: str, hours: float = 24) -> Optional[Dict[str, Any]]:
    """The same link watched in the last day, so a follow-up question does not fetch it again."""
    best = None
    for path in _dir().glob("vd*.json"):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if r.get("url") == url and time.time() - (r.get("t") or 0) < hours * 3600 and (best is None or r["t"] > best["t"]):
            best = r
    return best


def load(vid: str) -> Dict[str, Any]:
    if not _ID.fullmatch(str(vid or "")):
        raise KeyError(vid)
    try:
        return json.loads((_dir() / f"{vid}.json").read_text())
    except (OSError, ValueError):
        raise KeyError(vid) from None


def report_text(r: Dict[str, Any], limit: int = 16000) -> str:
    head = [f"Video watched ({r['id']}), {clock(r['duration'])} long, from {r['site'] or 'the link'}: {r['url']}"]
    for label, key in (("Title", "title"), ("Posted by", "uploader"), ("Caption", "description")):
        if r.get(key):
            head.append(f"{label}: {r[key]}")
    if r.get("upload_date"):
        d = r["upload_date"]
        head.append(f"Posted: {d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else f"Posted: {d}")
    if r.get("views") is not None:
        head.append(f"Views: {r['views']:,}" + (f", likes: {r['likes']:,}" if r.get("likes") is not None else ""))
    for n in r.get("notes") or []:
        head.append("Note: " + n)
    scenes = ["", f"What is on screen (stills read by {r.get('vision_model') or 'the vision model'}):"] + \
        [f"[{clock(s['at'])}] {s['text']}" for s in r["scenes"]] if r["scenes"] else []
    said = ["", f"What is said (Whisper, language {r.get('language') or '?'}):"] + \
        [f"[{clock(s['start'])}] {s['text']}" for s in r["transcript"]] if r["transcript"] else ["", "Nothing is said in it."]
    body = "\n".join(head + scenes)
    rest = "\n".join(said)
    if len(body) + len(rest) > limit:
        rest = rest[:max(2000, limit - len(body))] + "\n[... the rest of the transcript is on the saved page]"
    return body + "\n" + rest + f"\nThe saved page with all of it: /api/video/{r['id']}"


def render(r: Dict[str, Any]) -> str:
    e = H.escape
    rows = "".join(f"<li><b>{clock(s['at'])}</b> {e(s['text'])}</li>" for s in r["scenes"])
    said = "".join(f"<p><b>{clock(s['start'])}</b> {e(s['text'])}</p>" for s in r["transcript"])
    meta = " · ".join(e(x) for x in (r.get("uploader"), clock(r["duration"]), r.get("site")) if x)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(r.get('title') or 'Video')}</title>
<style>:root{{--bg:#fbfaf7;--ink:#1d1b18;--soft:#6b665e;--line:#e6e1d8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16150f;--ink:#ece8df;--soft:#a39d92;--line:#2c2a24}}}}
body{{background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,system-ui,sans-serif;margin:0 auto;max-width:760px;padding:20px 16px 60px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:26px 0 8px}}.meta,.note{{color:var(--soft);font-size:14px}}
a{{color:inherit}}li{{margin:0 0 8px}}p{{margin:0 0 6px}}b{{font-variant-numeric:tabular-nums;margin-right:6px}}
.cap{{border-left:3px solid var(--line);padding-left:10px;color:var(--soft)}}</style></head><body>
<h1>{e(r.get('title') or 'Video')}</h1><div class="meta">{meta} · <a href="{e(r['url'])}">open the video</a></div>
{f'<p class="cap">{e(r["description"])}</p>' if r.get("description") else ""}
{"".join(f'<p class="note">{e(n)}</p>' for n in r.get("notes") or [])}
<h2>On screen</h2>{f"<ol style='padding-left:20px'>{rows}</ol>" if rows else "<p class='note'>No stills were read.</p>"}
<h2>What is said</h2>{said or "<p class='note'>Nothing is said in it.</p>"}
</body></html>"""


GUIDE = (
    "You watch videos for the person. Usually the video has already been watched and its transcript and scene "
    "notes are in your task: then answer from them straight away. Never reply that you are fetching or will watch "
    "it: either call the tool now, or answer. A link to Facebook (a video, a Reel, an fb.watch link), Instagram, YouTube, "
    "TikTok, X or similar. Tool: watch_video(url, question) fetches the video on their Mac, writes down what is "
    "said (Whisper) and what is on screen every few seconds (a vision model), and returns all of it with times. It "
    "takes from half a minute to a few minutes. Call it once per link.\n"
    "Then answer, as a friend who just watched it would:\n"
    "1. In one or two sentences: what the video is (who posted it, what kind of video, how long) and its point.\n"
    "2. A short summary of what happens and what is said, in order.\n"
    "3. The key moments, each with its time (m:ss), only when the video is longer than a minute.\n"
    "4. If they asked something (is it true, what did he say about X, what is the product), answer that first, "
    "from the transcript and the scenes; quote what was said when it matters. If the claim in the video can be "
    "checked, search the web and say whether it holds up.\n"
    "If it is in another language, say which, and give the meaning in English.\n"
    "Only describe what the transcript and the scene notes show; the scene notes are a model's reading of a few "
    "stills, so say 'it looks like' for anything uncertain, and never name a person from their face, only from the "
    "caption, the title or what is said. If the tool says it could not fetch the video, tell them why in plain "
    "words and what to do."
)
