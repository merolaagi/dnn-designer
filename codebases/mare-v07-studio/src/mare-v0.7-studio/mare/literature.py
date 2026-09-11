from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import SourceRecord


class LiteratureProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 5) -> list[SourceRecord]:
        ...


@dataclass(slots=True)
class InMemoryLiteratureProvider:
    records: list[SourceRecord]
    name: str = "in-memory"

    def search(self, query: str, limit: int = 5) -> list[SourceRecord]:
        terms = {t.lower() for t in query.split() if len(t) > 2}
        scored = []
        for record in self.records:
            hay = f"{record.title} {record.abstract}".lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((score, record))
        return [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]


class OpenAlexLiteratureProvider:
    """Small zero-dependency OpenAlex client.

    Literature retrieval is evidence discovery, not automatic mathematical
    verification. Results are linked as `relevant` until a separate reviewer
    establishes a stronger relation.
    """

    name = "openalex"

    def __init__(self, mailto: str | None = None, timeout: float = 10.0):
        self.mailto = mailto
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SourceRecord]:
        params = {"search": query, "per-page": str(limit)}
        if self.mailto:
            params["mailto"] = self.mailto
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "MARE/0.3"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - configured public API
            payload = json.load(resp)
        out: list[SourceRecord] = []
        for item in payload.get("results", []):
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in item.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]
            out.append(SourceRecord(
                provider=self.name,
                title=item.get("display_name") or "Untitled",
                url=(item.get("primary_location") or {}).get("landing_page_url"),
                external_id=item.get("id"),
                authors=authors,
                year=item.get("publication_year"),
                abstract="",  # OpenAlex inverted abstract reconstruction intentionally omitted in v0.3.
                metadata={"cited_by_count": item.get("cited_by_count")},
            ))
        return out
