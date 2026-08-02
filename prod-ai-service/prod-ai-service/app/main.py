"""
Hardware Support Copilot — production RAG service.

A FastAPI application exposing a retrieval-augmented generation agent over a
hardware manufacturing knowledge base. Runs fully offline with a deterministic
local generator, and automatically upgrades to a real LLM when an API key is
present.

    uvicorn app.main:app --reload

Interactive docs at /docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .core.knowledge_base import validate
from .middleware import RateLimiterMiddleware, RequestLoggingMiddleware
from .routers import router

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast at startup if the knowledge base and ground truth disagree,
    # rather than surfacing it on the first eval request.
    validate()
    settings = get_settings()
    logging.getLogger("app").info(
        "startup provider=%s llm_enabled=%s",
        settings.effective_provider, settings.llm_enabled)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Production RAG service for hardware manufacturing support. "
            "Retrieval-augmented, confidence-gated, and provider-pluggable. "
            "Abstains rather than guessing when retrieval is weak or ambiguous."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RateLimiterMiddleware,
                       limit_per_min=settings.rate_limit_per_min)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(router)

    @app.get("/", tags=["ops"])
    def root() -> dict:
        return {"service": settings.app_name, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
