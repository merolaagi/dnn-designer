"""Shopping: finds things to buy across the internet, for you, and watches prices.

It uses the same SerpApi key as flight prices (Google Shopping), so one key covers thousands of stores:
Walmart, Best Buy, Target, Amazon sellers, brand stores and small shops. No store-by-store connection.
  - search: prices, the store, sale and old price, rating and review count, delivery;
  - offers: one product's sellers side by side, its rating breakdown and what reviewers say;
  - your shopping preferences (brands you like or avoid, sizes, budget, stores you trust, where to ship)
    are given to the agent with every request, so results are filtered and ranked for you;
  - price watches: checked once a day; your phone gets a push when the price reaches your target,
    or drops well below what it was.
Buying is done by the Booking agent in Saathi's browser: you type card details yourself, and the
button that pays waits for your approval.
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
import travel

PREFS = {
    "likes": "Brands and kinds of things you like",
    "avoid": "Brands, materials or stores to avoid",
    "sizes": "Sizes (clothes, shoes, rings, screens…)",
    "budget": "How you usually spend: e.g. good value over cheapest, under $200 for gadgets",
    "stores": "Stores you trust or prefer",
    "must": "Always: e.g. free shipping, free returns, new not used, ships to the US",
    "ship_to": "Where to ship (name and address), for the Booking agent's checkout form",
    "notes": "Anything else about how you shop",
}
MAX_PREF = 400
MAX_WATCHES = 30
ACCESSORY = ("case", "cover", "cable", "charger", "adapter", "stand", "mount", "protector", "skin", "sleeve", "strap",
             "replacement", "ear pad", "earpad", "cushion", "refurbished", "renewed", "parts", "decal", "holder")
DROP = 0.10                     # a push when the price falls this much below the last price seen
_LOCK = threading.Lock()
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or for with of in on to new by from".split())


class ShopError(RuntimeError):
    pass


def _path() -> Path:
    return AL._dir() / "shopping.json"


def _load() -> Dict[str, Any]:
    try:
        data = json.loads(_path().read_text())
    except (OSError, ValueError):
        data = {}
    return {"prefs": {k: str(v) for k, v in (data.get("prefs") or {}).items() if k in PREFS},
            "watches": list(data.get("watches") or [])}


def _store(data: Dict[str, Any]) -> None:
    _path().write_text(json.dumps(data, indent=1))


def _key() -> str:
    key = travel.config().get("serpapi_key") or ""
    if not key:
        raise ShopError("Shopping search is not connected: it uses the SerpApi key saved under Travel, and none is set.")
    return key


def configured() -> bool:
    return bool(travel.config().get("serpapi_key"))


# --------------------------------------------------------------------------
# preferences
# --------------------------------------------------------------------------

def prefs() -> Dict[str, str]:
    return _load()["prefs"]


def set_prefs(change: Dict[str, Any]) -> Dict[str, str]:
    unknown = [k for k in change if k not in PREFS]
    if unknown:
        raise ValueError(f"Unknown preference: {', '.join(unknown)}. Choose from {', '.join(PREFS)}.")
    with _LOCK:
        data = _load()
        for k, v in change.items():
            v = re.sub(r"\s+", " ", str(v or "")).strip()
            if len(v) > MAX_PREF:
                raise ValueError(f"Keep {k} under {MAX_PREF} characters.")
            if v:
                data["prefs"][k] = v
            else:
                data["prefs"].pop(k, None)
        _store(data)
        return data["prefs"]


def prefs_text(p: Optional[Dict[str, str]] = None) -> str:
    p = prefs() if p is None else p
    if not p:
        return "No shopping preferences are saved yet."
    return "\n".join(f"- {PREFS[k].split(':')[0].split(' (')[0]}: {p[k]}" for k in PREFS if p.get(k) and k != "ship_to") \
        + ("\n- A shipping address is saved (for checkout)." if p.get("ship_to") else "")


# --------------------------------------------------------------------------
# search (Google Shopping through SerpApi)
# --------------------------------------------------------------------------

def _money(x) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[\d,]+(?:\.\d+)?", str(x or ""))
    return float(m.group(0).replace(",", "")) if m else None


def _item(r: Dict[str, Any]) -> Dict[str, Any]:
    price = r.get("extracted_price") if r.get("extracted_price") is not None else _money(r.get("price"))
    old = r.get("extracted_old_price") if r.get("extracted_old_price") is not None else _money(r.get("old_price"))
    tags = [t for t in [r.get("tag"), *(r.get("extensions") or [])] if isinstance(t, str)]
    return {"title": r.get("title"), "store": r.get("source") or r.get("seller"), "price": price, "old_price": old,
            "rating": r.get("rating"), "reviews": r.get("reviews"), "delivery": r.get("delivery"),
            "condition": r.get("second_hand_condition") or "", "tags": tags[:4],
            "link": r.get("product_link") or r.get("link"), "token": r.get("immersive_product_page_token") or "",
            "product_id": r.get("product_id") or ""}


def search(query: str, min_price: Optional[float] = None, max_price: Optional[float] = None, on_sale: bool = False,
           free_shipping: bool = False, sort: str = "relevance", limit: int = 10) -> Dict[str, Any]:
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(query) < 2:
        raise ValueError("Say what to look for.")
    sort_by = {"price_low": 1, "price_high": 2}.get(sort)
    params = {"engine": "google_shopping", "q": query, "hl": "en", "gl": "us", "api_key": _key(),
              "min_price": min_price or None, "max_price": max_price or None, "sort_by": sort_by,
              "on_sale": "true" if on_sale else None, "free_shipping": "true" if free_shipping else None}
    try:
        data = travel._get(params)
    except travel.TravelError as exc:
        raise ShopError(str(exc).replace("TravelError", "")) from None
    rows = (data.get("shopping_results") or []) + (data.get("inline_shopping_results") or [])
    out, seen = [], set()
    for r in rows:
        it = _item(r)
        if it["price"] is None or not it["title"]:
            continue
        key = (it["title"].lower(), (it["store"] or "").lower(), it["price"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    if sort_by == 1:
        out.sort(key=lambda i: i["price"])
    elif sort_by == 2:
        out.sort(key=lambda i: -i["price"])
    return {"query": query, "searched_at": time.strftime("%Y-%m-%d %H:%M"), "items": out[:max(1, min(20, int(limit or 10)))],
            "filters": {k: v for k, v in {"min_price": min_price, "max_price": max_price, "on_sale": on_sale,
                                          "free_shipping": free_shipping, "sort": sort}.items() if v},
            "link": (data.get("search_metadata") or {}).get("google_shopping_url")}


def _num(n) -> str:
    return f"{n:,.0f}" if float(n).is_integer() else f"{n:,.2f}"


def search_text(r: Dict[str, Any]) -> str:
    head = f"Google Shopping, searched {r['searched_at']}: {r['query']}" + \
        (f" ({', '.join(f'{k} {v}' for k, v in r['filters'].items())})" if r["filters"] else "") + ". Prices in USD."
    if not r["items"]:
        return head + "\nNothing with a price was found. Try fewer words or a wider price range."
    lines = [head]
    for n, i in enumerate(r["items"], 1):
        off = ""
        if i.get("old_price") and i["old_price"] > i["price"]:
            off = f" (was ${_num(i['old_price'])}, {round(100 * (1 - i['price'] / i['old_price']))}% off)"
        lines.append(f"{n}. {i['title']} · ${_num(i['price'])}{off} · {i['store'] or 'store not shown'}"
                     + (f" · {i['condition']}" if i["condition"] else "")
                     + (f" · rated {i['rating']} from {i['reviews']:,} reviews" if i.get("rating") and isinstance(i.get("reviews"), int)
                        else (f" · rated {i['rating']}" if i.get("rating") else " · no rating"))
                     + (f" · {i['delivery']}" if i.get("delivery") else "")
                     + (f" · {', '.join(i['tags'])}" if i["tags"] else "")
                     + (f" · offers id: {i['token']}" if i["token"] else "")
                     + (f" · {i['link']}" if i.get("link") else ""))
    if r.get("link"):
        lines.append(f"The same search on Google Shopping: {r['link']}")
    return "\n".join(lines)


def offers(token: str) -> Dict[str, Any]:
    """One product's page: every store's price, the rating breakdown, and reviews."""
    token = str(token or "").strip()
    if len(token) < 10:
        raise ValueError("Give the offers id from a search result.")
    try:
        data = travel._get({"engine": "google_immersive_product", "page_token": token, "api_key": _key()})
    except travel.TravelError as exc:
        raise ShopError(str(exc)) from None
    p = data.get("product_results") or {}
    stores = []
    for s in p.get("stores") or []:
        price = s.get("extracted_price") if s.get("extracted_price") is not None else _money(s.get("price"))
        stores.append({"name": s.get("name"), "price": price,
                       "old_price": s.get("extracted_original_price") or _money(s.get("original_price")),
                       "total": s.get("extracted_total") or _money(s.get("total")),
                       "details": ", ".join(x for x in (s.get("details_and_offers") or []) if isinstance(x, str))[:160],
                       "rating": s.get("rating"), "reviews": s.get("reviews"), "link": s.get("link")})
    stores.sort(key=lambda s: s["price"] if s["price"] is not None else 1e12)
    reviews = []
    for u in (p.get("user_reviews") or [])[:6]:
        reviews.append({"rating": u.get("rating"), "title": u.get("title") or "", "text": AL._one_line(u.get("text") or "", 300),
                        "source": u.get("source") or "", "date": u.get("date") or ""})
    return {"title": p.get("title"), "rating": p.get("rating"), "reviews": p.get("reviews"),
            "breakdown": {str(r.get("stars")): r.get("amount") for r in (p.get("ratings") or []) if r.get("stars")},
            "about": AL._one_line((p.get("about_the_product") or {}).get("description") or "", 400),
            "stores": stores[:12], "user_reviews": reviews}


def offers_text(o: Dict[str, Any]) -> str:
    lines = [f"{o.get('title') or 'Product'}" + (f": rated {o['rating']} from {o['reviews']:,} reviews"
                                                  if o.get("rating") and isinstance(o.get("reviews"), int) else "")]
    if o["breakdown"]:
        lines.append("Ratings: " + ", ".join(f"{k}★ {v:,}" if isinstance(v, int) else f"{k}★ {v}"
                                             for k, v in sorted(o["breakdown"].items(), reverse=True)))
    if o["about"]:
        lines.append("About: " + o["about"])
    lines.append("Stores, cheapest first:" if o["stores"] else "No store prices were shown.")
    for s in o["stores"]:
        lines.append(f"- {s['name']}: " + (f"${_num(s['price'])}" if s["price"] is not None else "price not shown")
                     + (f" (was ${_num(s['old_price'])})" if s.get("old_price") and s["price"] and s["old_price"] > s["price"] else "")
                     + (f", ${_num(s['total'])} with tax and shipping" if s.get("total") else "")
                     + (f" · {s['details']}" if s["details"] else "") + (f" · {s['link']}" if s.get("link") else ""))
    if o["user_reviews"]:
        lines.append("What reviewers say:")
        for u in o["user_reviews"]:
            lines.append(f"- {u['rating']}★ {u['title']} {u['text']}".strip() + (f" ({u['source']} {u['date']})".rstrip() if u["source"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reading a review page
# --------------------------------------------------------------------------

def _public(host: str) -> bool:
    import ipaddress
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


def read_page(url: str, limit: int = 8000) -> str:
    """A public web page as plain text (a review, a Reddit thread), so the agent can read what buyers say."""
    import html as H
    import urllib.parse
    import urllib.request
    u = urllib.parse.urlparse(str(url or "").strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Give a full web address, starting with https://.")
    if not _public(u.hostname):
        raise ValueError("Only public web pages can be read.")
    if u.hostname.endswith("reddit.com"):                 # the old layout has the comments in the page itself
        url = urllib.parse.urlunparse(u._replace(netloc="old.reddit.com"))

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            n = urllib.parse.urlparse(newurl)
            if n.scheme not in ("http", "https") or not n.hostname or not _public(n.hostname):
                raise ValueError("The page redirected somewhere that is not a public web page.")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) Saathi shopping", "Accept": "text/html"})
    try:
        with opener.open(req, timeout=25) as reply:
            raw = reply.read(2_000_000).decode(reply.headers.get_content_charset() or "utf-8", "replace")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ShopError(f"Could not read that page: {exc}") from None
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", H.unescape(text)).strip()
    head = (H.unescape(title.group(1)).strip() + "\n") if title else ""
    return (head + text[:max(500, min(20000, int(limit or 8000)))]) or "The page had no readable text."


# --------------------------------------------------------------------------
# price watches
# --------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP]


def watches() -> List[Dict[str, Any]]:
    return _load()["watches"]


def add_watch(query: str, target: Optional[float] = None, label: str = "", max_price: Optional[float] = None) -> Dict[str, Any]:
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(_words(query)) < 1:
        raise ValueError("Say what to watch: the product's name, as exact as you can (brand and model).")
    if target is not None and float(target) <= 0:
        raise ValueError("The target price is above zero.")
    with _LOCK:
        data = _load()
        if len(data["watches"]) >= MAX_WATCHES:
            raise ValueError(f"You have {MAX_WATCHES} price watches: remove one first.")
        w = {"id": "pw" + uuid.uuid4().hex[:8], "query": query, "label": (label or query)[:120],
             "target": float(target) if target else None, "max_price": float(max_price) if max_price else None,
             "created": time.time(), "last": None, "low": None, "history": [], "alerted": None}
        data["watches"].append(w)
        _store(data)
    ensure_check()
    return w


def remove_watch(wid: str) -> None:
    with _LOCK:
        data = _load()
        keep = [w for w in data["watches"] if w["id"] != wid]
        if len(keep) == len(data["watches"]):
            raise KeyError(wid)
        data["watches"] = keep
        _store(data)


def best_match(w: Dict[str, Any], items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The cheapest result whose title has every word of the watch (so a case or a cable is not 'the price')."""
    need = set(_words(w["query"]))
    asked = set(_WORD.findall(w["query"].lower()))

    def accessory(title: str) -> bool:
        t = title.lower()
        extra = {a for a in ACCESSORY if re.search(rf"\b{a}s?\b", t)} - asked
        return bool(extra) or bool(re.search(r"\b(for|compatible with|fits)\b", t)) and "for" not in asked

    fits = [i for i in items if need <= set(_words(i["title"])) and not i.get("condition") and not accessory(i["title"])]
    return min(fits, key=lambda i: i["price"]) if fits else None


def news(w: Dict[str, Any], found: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why to tell you about this price now, or None."""
    if not found:
        return None
    p = found["price"]
    if w.get("target") and p <= w["target"] and (w.get("alerted") is None or p < w["alerted"]):
        return f"${_num(p)} at {found['store']}: at or under your ${_num(w['target'])} target."
    if w.get("last") and p <= w["last"] * (1 - DROP) and (w.get("alerted") is None or p < w["alerted"]):
        return f"${_num(p)} at {found['store']}, down from ${_num(w['last'])}."
    return None


def check(s: Dict[str, Any], search_fn=None) -> None:
    """The scheduled look (each day): search every watch, record the price, push the good news."""
    import schedules
    search_fn = search_fn or (lambda w: search(w["query"], max_price=w.get("max_price"), sort="price_low", limit=20))
    told, looked, errors = 0, 0, []
    for w in list(watches()):
        try:
            found = best_match(w, search_fn(w)["items"])
        except Exception as exc:  # noqa: BLE001 - one bad search must not stop the rest
            errors.append(f"{w['label']}: {exc}")
            continue
        looked += 1
        why = news(w, found)
        if why:
            schedules.notify("deal", f"Price drop: {w['label']}"[:200], why + (f" {found['link']}" if found.get("link") else ""),
                             s.get("id"))
            told += 1
        with _LOCK:
            data = _load()
            for x in data["watches"]:
                if x["id"] == w["id"]:
                    if found:
                        x["last"] = found["price"]
                        x["low"] = found["price"] if x.get("low") is None else min(x["low"], found["price"])
                        x["history"] = (x.get("history") or [])[-59:] + [[time.strftime("%Y-%m-%d"), found["price"], found["store"]]]
                        x["store"], x["link"] = found["store"], found.get("link")
                        if why:
                            x["alerted"] = found["price"]
                    x["checked"] = time.time()
            _store(data)
    s["last_status"] = "error" if errors and not looked else "ok"
    s["last_detail"] = (f"{looked} watched, {told} price drop(s)" if looked else "no watches") + \
        (f"; {len(errors)} failed: {errors[0]}" if errors else "")


def ensure_check() -> None:
    import schedules
    if any(x["agent"] == {"kind": "check", "key": "shopping"} for x in schedules.listing()):
        return
    try:
        schedules.create("Price watches", {"kind": "check", "key": "shopping"},
                         "Check the price of everything you are watching, and tell you about drops.", "day", "07:40")
    except Exception:  # noqa: BLE001
        pass


def watches_text() -> str:
    ws = watches()
    if not ws:
        return "No price watches yet."
    lines = []
    for w in ws:
        lines.append(f"[{w['id']}] {w['label']} · " + (f"target ${_num(w['target'])}" if w.get("target") else "any big drop")
                     + (f" · now ${_num(w['last'])} at {w.get('store')}" if w.get("last") else " · not checked yet")
                     + (f" · lowest seen ${_num(w['low'])}" if w.get("low") else ""))
    return "\n".join(lines)


GUIDE = (
    "You are the person's personal shopper. You find things to buy across the internet: the right product, the "
    "best price from a store they can trust, sales and discounts, and what real buyers say. Tools: search_products "
    "(Google Shopping: thousands of stores, with price, sale, rating, reviews and delivery), product_offers (one "
    "product's stores side by side, its rating breakdown and reviews), shopping_preferences and "
    "set_shopping_preference, watch_price, list_watches and remove_watch, web search, and read_page (a review "
    "or a Reddit thread as text).\n"
    "1. Start with shopping_preferences, and follow them: the brands they like or avoid, sizes, budget, the "
    "stores they trust, and their musts (free shipping, returns, new). If a preference rules a result out, leave "
    "it out. When they tell you a lasting preference (a size, a brand they hate, 'always free shipping'), save it "
    "with set_shopping_preference.\n"
    "2. If what they want is vague (a 'good laptop'), ask one short question only if the answer changes the pick "
    "a lot (what it is for, or the budget); otherwise choose sensible defaults and say which.\n"
    "3. Search, with filters when they help (max_price, on_sale, free_shipping, sort price_low). For the two or "
    "three best candidates use product_offers to compare stores and read the reviews. For anything over about "
    "$100, also search the web for real opinions (Reddit, Wirecutter, RTINGS, specialist reviewers) and read one "
    "or two.\n"
    "4. Judge reviews honestly: a 4.8 from 12 reviews means little; a lot of short five-star reviews at once, or a "
    "store nobody has heard of with a price far below everyone else, is a warning: say so. Prefer the store they "
    "trust when the price is close.\n"
    "5. Answer with your pick first and why (in a sentence), then two or three alternatives (cheaper, or better), "
    "each with the price, the store, the sale if any, the rating and how many reviews, and the link. Say when the "
    "prices were searched. Prices change: never promise a price.\n"
    "6. If they want to know when it gets cheaper, use watch_price with the product's exact name (brand and model) "
    "and their target; their phone gets a push when it drops. If they want to buy it, say you found it and they "
    "can ask 'buy it for me' (the Booking agent checks out in their browser, and they approve before paying).\n"
    "Every price, rating and review count comes from a tool result; never invent them."
)
