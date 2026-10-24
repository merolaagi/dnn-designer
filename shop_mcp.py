"""An MCP server (stdio) for the Shopping agent: Google Shopping, offers and reviews, your preferences, price
watches. Started as `python3 shop_mcp.py`."""

import json
import sys

import shop

TOOLS = [
    {"name": "search_products", "description": "Search Google Shopping across thousands of stores (Walmart, Best Buy, "
     "Target, Amazon sellers, brand stores): each result's price, old price and discount, store, rating and review "
     "count, delivery, and an offers id for product_offers.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look for, e.g. 'Sony WH-1000XM6' or 'men's trail running shoes size 10'"},
         "min_price": {"type": "number"}, "max_price": {"type": "number"},
         "on_sale": {"type": "boolean", "description": "Only things on sale"},
         "free_shipping": {"type": "boolean"},
         "sort": {"type": "string", "description": "relevance (default), price_low or price_high"},
         "limit": {"type": "integer", "description": "How many results, up to 20 (default 10)"}},
         "required": ["query"]}},
    {"name": "product_offers", "description": "One product's page: every store's price side by side (with tax and "
     "shipping where shown), the star rating breakdown, and what reviewers say.",
     "inputSchema": {"type": "object", "properties": {"offers_id": {"type": "string", "description": "From search_products"}},
                     "required": ["offers_id"]}},
    {"name": "read_page", "description": "Read a public web page as text: a review (Wirecutter, RTINGS, a specialist "
     "site) or a Reddit thread, to see what real buyers say.",
     "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "shopping_preferences", "description": "How the person shops: brands they like or avoid, sizes, budget, "
     "stores they trust, musts. Read it first.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "set_shopping_preference", "description": "Save a lasting shopping preference the person told you. "
     "Fields: " + ", ".join(f"{k} ({v})" for k, v in shop.PREFS.items() if k != "ship_to") + ". An empty value clears it. "
     "Write the whole field (it replaces what was there), keeping what was already saved.",
     "inputSchema": {"type": "object", "properties": {"field": {"type": "string"}, "value": {"type": "string"}},
                     "required": ["field", "value"]}},
    {"name": "add_to_cart", "description": "Put something in the person's cart (one cart for every store). Use the "
     "title, store and price exactly as search_products or product_offers showed them.",
     "inputSchema": {"type": "object", "properties": {
         "title": {"type": "string"}, "store": {"type": "string"}, "price": {"type": "number", "description": "Each, in USD"},
         "qty": {"type": "integer"}, "options": {"type": "string", "description": "Size, colour, model or other choice"},
         "link": {"type": "string", "description": "The result's link"},
         "store_link": {"type": "string", "description": "The product on the store's own site (from product_offers), if known"},
         "offers_id": {"type": "string"}},
         "required": ["title", "store", "price"]}},
    {"name": "show_cart", "description": "The cart by store: each item with its id, quantity, options, price (and any "
     "cheaper store), and each store's subtotal.",
     "inputSchema": {"type": "object", "properties": {"store": {"type": "string", "description": "Only this store's items"}}}},
    {"name": "change_cart_item", "description": "Change an item's quantity or options.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "qty": {"type": "integer"},
                                                     "options": {"type": "string"}}, "required": ["id"]}},
    {"name": "remove_from_cart", "description": "Take an item out of the cart.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "mark_ordered", "description": "After a checkout was paid: move those items from the cart to Ordered, with "
     "the order number and delivery date the store showed.",
     "inputSchema": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "string"}}, "order_number": {"type": "string"},
         "delivery": {"type": "string"}, "total": {"type": "number", "description": "What was paid, with tax and shipping"}},
         "required": ["ids"]}},
    {"name": "watch_price", "description": "Watch a product's price: checked every morning; the person's phone gets a "
     "push when it reaches the target or drops well below the last price.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "The product's exact name, brand and model: every word must be in a result's title"},
         "target": {"type": "number", "description": "Tell them at or under this price (optional)"},
         "label": {"type": "string", "description": "A short name for it"}},
         "required": ["query"]}},
    {"name": "list_watches", "description": "The prices being watched, with the latest and lowest seen.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "remove_watch", "description": "Stop watching a price.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


def call(name, a):
    if name == "search_products":
        return shop.search_text(shop.search(a.get("query") or "", a.get("min_price"), a.get("max_price"),
                                             bool(a.get("on_sale")), bool(a.get("free_shipping")),
                                             a.get("sort") or "relevance", a.get("limit") or 10))
    if name == "product_offers":
        return shop.offers_text(shop.offers(a.get("offers_id") or ""))
    if name == "read_page":
        return shop.read_page(a.get("url") or "")
    if name == "shopping_preferences":
        return shop.prefs_text()
    if name == "set_shopping_preference":
        field = a.get("field") or ""
        if field == "ship_to":
            raise ValueError("The shipping address is set by the person on the Mac (Settings → Shopping), not here.")
        shop.set_prefs({field: a.get("value") or ""})
        return "Saved. Now:\n" + shop.prefs_text()
    if name == "add_to_cart":
        c = shop.cart_add(a.get("title") or "", a.get("store") or "", a.get("price"), a.get("link") or "", a.get("qty") or 1,
                          a.get("options") or "", a.get("store_link") or "", a.get("offers_id") or "")
        return f"In the cart [{c['id']}]: {c['qty']} × {c['title']} from {c['store']}.\n\n" + shop.cart_text()
    if name == "show_cart":
        return shop.cart_text(a.get("store") or "", checking_out=bool(a.get("store")))
    if name == "change_cart_item":
        shop.cart_update(a.get("id") or "", {"qty": a.get("qty"), "options": a.get("options")})
        return "Changed.\n\n" + shop.cart_text()
    if name == "remove_from_cart":
        shop.cart_remove(a.get("id") or "")
        return "Taken out.\n\n" + shop.cart_text()
    if name == "mark_ordered":
        done = shop.mark_ordered(list(a.get("ids") or []), a.get("order_number") or "", a.get("delivery") or "", a.get("total"))
        return f"Moved {len(done)} item(s) to Ordered" + (f" (order {a['order_number']})" if a.get("order_number") else "") + "."
    if name == "watch_price":
        w = shop.add_watch(a.get("query") or "", a.get("target"), a.get("label") or "")
        return (f"Watching [{w['id']}] {w['label']}" + (f" for ${w['target']:,.2f} or less" if w.get("target") else "")
                + ". It is checked every morning, and their phone gets a push when it drops.")
    if name == "list_watches":
        return shop.watches_text()
    if name == "remove_watch":
        shop.remove_watch(a.get("id") or "")
        return "Stopped watching it."
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "shop", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except KeyError:
            text, error = "There is no price watch or cart item with that id: list_watches or show_cart first.", True
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
