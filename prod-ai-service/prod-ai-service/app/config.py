"""
Application configuration.

Every runtime knob is read from the environment with a safe default, so the
service starts and runs with zero configuration -- the reproducibility property
that lets a reviewer clone and run it without credentials. Supplying an API key
upgrades the generation backend from the deterministic local synthesizer to a
real LLM; nothing else changes.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    # Service
    app_name: str = "Hardware Support Copilot"
    version: str = "1.0.0"
    environment: str = os.getenv("APP_ENV", "local")

    # LLM provider. "auto" uses a real provider when a key is present and falls
    # back to the local deterministic synthesizer otherwise.
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")  # auto | anthropic | local
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))

    # Retrieval
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.10"))
    min_margin: float = float(os.getenv("MIN_MARGIN", "0.04"))

    # Serving guardrails
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
    rate_limit_per_min: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

    @property
    def effective_provider(self) -> str:
        """Resolve 'auto' to a concrete provider based on available credentials."""
        if self.llm_provider == "auto":
            return "anthropic" if self.anthropic_api_key else "local"
        return self.llm_provider

    @property
    def llm_enabled(self) -> bool:
        return self.effective_provider != "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
