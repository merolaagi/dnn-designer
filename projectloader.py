"""Loads guided projects from ``projects/``, and matches free text against them."""

from __future__ import annotations

import math
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

import projects_sdk

PROJECTS_DIR = Path(__file__).resolve().parent / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

LAST_ERRORS: List[Dict[str, str]] = []


def load_all() -> List[Dict[str, str]]:
    projects_sdk.REGISTRY.clear()
    LAST_ERRORS.clear()
    for path in sorted(PROJECTS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            # __file__ is supplied so a project file can find its own folder and
            # import shared builders from beside it
            exec(compile(path.read_text(), str(path), "exec"),  # noqa: S102
                 {"__name__": "designer_projects", "__file__": str(path)})
        except Exception as exc:  # noqa: BLE001
            LAST_ERRORS.append({
                "file": path.name,
                "message": f"{type(exc).__name__}: {exc}",
                "detail": traceback.format_exc()[-1400:],
            })
    return list(LAST_ERRORS)


def catalog() -> List[Dict[str, Any]]:
    return [p.to_json() for p in projects_sdk.REGISTRY.values()]


def categories() -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for p in projects_sdk.REGISTRY.values():
        counts[p.category] = counts.get(p.category, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(counts.items())]


def get(project_id: str) -> Dict[str, Any]:
    project = projects_sdk.REGISTRY.get(project_id)
    if project is None:
        raise KeyError(project_id)
    return project.to_json(full=True)


# --------------------------------------------------------------------------
# matching a free-text request against the catalogue
# --------------------------------------------------------------------------

STOP = {
    "a", "an", "the", "to", "for", "of", "on", "in", "with", "and", "or", "my",
    "i", "want", "build", "make", "create", "using", "use", "how", "do", "can",
    "would", "like", "need", "help", "me", "model", "network", "that", "is",
    "it", "this", "some", "get", "from", "at", "by", "be", "am", "please",
}

# words that mean the same thing to the catalogue but not to a string match
SYNONYMS = {
    "picture": "image", "photo": "image", "pictures": "image", "photos": "image",
    "classify": "classification", "classifier": "classification",
    "categorise": "classification", "categorize": "classification",
    "predict": "regression", "forecast": "timeseries", "forecasting": "timeseries",
    "series": "timeseries", "temporal": "sequence", "sentence": "text",
    "sentences": "text", "word": "text", "words": "text", "language": "text",
    "chat": "text", "generate": "generative", "generation": "generative",
    "synthesise": "generative", "synthesize": "generative", "fake": "generative",
    "spreadsheet": "tabular", "csv": "tabular", "table": "tabular",
    "columns": "tabular", "rows": "tabular", "xray": "medical",
    "x-ray": "medical", "scan": "medical",
    "histopathology": "medical", "pathology": "medical", "tumour": "medical",
    "tumor": "medical", "cancer": "medical", "diagnosis": "medical",
    "sound": "audio", "speech": "audio", "voice": "audio", "music": "audio",
    "game": "game", "agent": "reinforcement", "reward": "reinforcement",
    "robot": "reinforcement", "detect": "detection", "boxes": "detection",
    "bounding": "detection", "segment": "segmentation",
    "unlabelled": "selfsupervised", "unlabeled": "selfsupervised",
    "pretrain": "selfsupervised", "pretraining": "selfsupervised",
    "outlier": "anomaly", "fraud": "anomaly", "defect": "anomaly",
    "graph": "graph", "molecule": "graph", "network": "graph",
    "compress": "autoencoder", "denoise": "autoencoder",
    "chatbot": "text", "conversation": "text", "dialogue": "text",
    "walk": "reinforcement", "locomotion": "reinforcement",
    "control": "reinforcement", "policy": "reinforcement",
    "similar": "similarity", "match": "similarity", "matching": "similarity",
    "duplicate": "similarity", "retrieval": "similarity", "search": "similarity",
    "spectrogram": "audio", "recording": "audio", "acoustic": "audio",
    "biopsy": "medical", "radiology": "medical", "clinical": "medical",
    "patient": "medical", "mri": "medical", "ct": "medical",
}


def _variants(word: str) -> set:
    """A word, its synonym, and its singular — any of which may match a tag.

    Keeping the original alongside the synonym matters: mapping "fraud" to
    "anomaly" and discarding it loses the literal tag on the fraud project,
    which is the one the user actually wanted.
    """
    out = {word}
    if word in SYNONYMS:
        out.add(SYNONYMS[word])
    # every plausible singular, not just the first suffix that matches:
    # "molecules" gives "molecul" under -es and "molecule" under -s, and only
    # the second is a word anyone tagged anything with
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", "")):
        if len(word) > 3 and word.endswith(suffix):
            stem = word[: -len(suffix)] + replacement
            out.add(stem)
            if stem in SYNONYMS:
                out.add(SYNONYMS[stem])
    return out


def _words(text: str) -> List[str]:
    """Flat list, for indexing a project's own name and summary."""
    out = []
    for w in re.split(r"[^a-z0-9]+", text.lower()):
        if w and w not in STOP:
            out.extend(sorted(_variants(w)))
    return out


def _query_terms(text: str) -> List[set]:
    """One set of acceptable variants per word the user typed."""
    return [_variants(w) for w in re.split(r"[^a-z0-9]+", text.lower())
            if w and w not in STOP]


def suggest(query: str, limit: int = 5) -> Dict[str, Any]:
    """Rank catalogue entries against a free-text request.

    Deliberately a keyword match, not a language model. It scores overlap
    against each project's tags, name and summary, and is honest when nothing
    scores well rather than returning the least-bad answer as though it fitted.
    """
    terms = _query_terms(query)
    if not terms:
        return {"query": query, "matches": [], "confident": False,
                "advice": "Say a little more about what you want to build."}

    # A tag shared by half the catalogue says much less than one used twice.
    # Without this, "histopathology slides" scores every medical project equally
    # and the slide-level project does not rise above the rest.
    frequency: Dict[str, int] = {}
    for p in projects_sdk.REGISTRY.values():
        for t in {x.lower() for x in p.tags}:
            frequency[t] = frequency.get(t, 0) + 1
    total = max(len(projects_sdk.REGISTRY), 1)

    def rarity(matched) -> float:
        """Weight by the tag that actually matched, not the rarest one available.

        Scoring on the rarest possible variant meant a word whose synonym is a
        common tag scored as though it had hit the rare one — every medical
        project ranked as highly for "slides" as the slide-level project did.
        """
        seen = [frequency[t] for t in matched if t in frequency]
        if not seen:
            return 1.0
        return 1.0 + math.log(total / min(seen))

    scored = []
    for p in projects_sdk.REGISTRY.values():
        tags = {t.lower() for t in p.tags}
        name = set(_words(p.name))
        summary = set(_words(p.summary))
        category = set(_words(p.category))
        score = 0.0
        for variants in terms:
            hit = variants & tags
            if hit:
                score += 3.0 * rarity(hit)
            elif variants & name:
                score += 2.0
            elif variants & category:
                score += 1.5
            elif variants & summary:
                score += 1.0
        if score:
            scored.append((score / len(terms), p))

    scored.sort(key=lambda x: -x[0])
    top = scored[:limit]
    best = top[0][0] if top else 0.0

    result = {
        "query": query,
        "terms": sorted({t for v in terms for t in v}),
        "matches": [{**p.to_json(), "score": round(s, 2)} for s, p in top],
        "confident": best >= 1.5,
    }
    if not top:
        result["advice"] = (
            "Nothing in the catalogue matches that. Start from the shape of your "
            "data instead: images go through a convolution stack, a table of "
            "columns through an MLP, anything ordered in time through an LSTM or "
            "a transformer, and text through an Embedding first. Pick the closest "
            "project to that description and change the Input shape.")
    elif not result["confident"]:
        result["advice"] = (
            "No close match, so treat these as starting points rather than "
            "answers. The usual move is to take the one whose data looks like "
            "yours and change the Input shape and the final layer's width.")
    return result
