import hmac
import json
import logging
import time
from collections import OrderedDict
from uuid import uuid4

from starlette.responses import JSONResponse

log = logging.getLogger("mare.http")


class SecurityMiddleware:
    """Single-workspace bearer authentication; replace with verified identity/RBAC for multi-tenancy.

    The bounded in-process rate limiter is a hook, not a distributed quota service.
    """

    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.buckets = OrderedDict()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.monotonic()
        status = 500
        headers = dict(scope["headers"])

        async def wrapped_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] += [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        async def reject(code, message, http_status):
            response = JSONResponse(
                {"error": {"code": code, "message": message, "request_id": request_id}},
                status_code=http_status,
                headers={"Retry-After": "60"} if http_status == 429 else None,
            )
            await response(scope, receive, wrapped_send)

        try:
            public = scope["path"] in {"/health/live", "/health/ready"} or scope["method"] == "OPTIONS"
            key = (scope.get("client") or ("unknown",))[0]
            minute = int(time.monotonic() // 60)
            bucket = self.buckets.get(key, (minute, 0))
            count = bucket[1] + 1 if bucket[0] == minute else 1
            self.buckets[key] = (minute, count)
            self.buckets.move_to_end(key)
            if len(self.buckets) > 4096:
                self.buckets.popitem(last=False)
            if not public and count > self.settings.rate_limit_per_minute:
                return await reject("rate_limited", "Request rate exceeded", 429)
            if not public and self.settings.auth_token:
                expected = "Bearer " + self.settings.auth_token.get_secret_value()
                supplied = headers.get(b"authorization", b"").decode("latin-1")
                if not hmac.compare_digest(supplied.encode(), expected.encode()):
                    return await reject("unauthorized", "A valid workspace bearer token is required", 401)
            body = bytearray()
            if scope["method"] in {"POST", "PUT", "PATCH"}:
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > self.settings.max_body_bytes:
                        return await reject("body_too_large", "Request body exceeds configured limit", 413)
                    if not message.get("more_body"):
                        break
                sent = False

                async def buffered_receive():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, buffered_receive, wrapped_send)
            else:
                await self.app(scope, receive, wrapped_send)
        finally:
            # Exclude query strings, tokens, bodies, and user-controlled request IDs.
            log.info(
                json.dumps(
                    {
                        "event": "request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "status": status,
                        "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    }
                )
            )
