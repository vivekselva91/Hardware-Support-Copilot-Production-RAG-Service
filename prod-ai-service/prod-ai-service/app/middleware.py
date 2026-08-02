"""
Cross-cutting serving concerns: rate limiting and structured request logging.

Deliberately dependency-free -- an in-memory sliding-window limiter rather than a
Redis integration. For a single-instance service that is the honest choice; the
docstring notes what would change at multi-instance scale rather than pretending
the in-memory version is production-final.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("app")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window per-client rate limit.

    In-memory, so it is correct for a single instance only. Behind a load
    balancer this state would move to a shared store (Redis) keyed the same way;
    the interface here would not change, which is the point of isolating it in
    middleware.
    """

    def __init__(self, app, limit_per_min: int = 60):
        super().__init__(app)
        self.limit = limit_per_min
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/", "/docs", "/openapi.json"):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._hits[client]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self.limit:
            retry = 60 - (now - window[0])
            return JSONResponse(
                status_code=429,
                content={"detail": f"rate limit exceeded, retry in {retry:.0f}s"},
                headers={"Retry-After": str(int(retry) + 1)},
            )

        window.append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured timing log per request -- the minimum for production triage."""

    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "request",
            extra={"method": request.method, "path": request.url.path,
                   "status": response.status_code, "latency_ms": round(elapsed, 2)},
        )
        response.headers["X-Response-Time-ms"] = f"{elapsed:.2f}"
        return response
