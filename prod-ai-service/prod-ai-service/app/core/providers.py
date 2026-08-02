"""
Generation providers.

One interface, two implementations:

  ``LocalSynthesizer``    deterministic, offline. Composes an answer from the
                          retrieved document with no model call. This is what
                          makes the service runnable and testable without an API
                          key, and what CI exercises.

  ``AnthropicProvider``   real LLM generation, grounded in the retrieved
                          document, used automatically when a key is present.

The provider is selected at request time from settings. Critically, retrieval
and the abstention decision are identical across both providers -- only the
wording of a grounded answer changes. That is what keeps the offline path a
faithful stand-in for the online one rather than a toy.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Settings
from .knowledge_base import KBDocument


class GenerationProvider(Protocol):
    name: str

    def generate(self, question: str, document: KBDocument,
                 confidence: float) -> str: ...


class LocalSynthesizer:
    """Deterministic, offline answer composition. Formats; does not reason."""

    name = "local"

    def generate(self, question: str, document: KBDocument,
                 confidence: float) -> str:
        return (
            f"Based on the knowledge base ({document.doc_id}: {document.title}):\n\n"
            f"{document.text}\n\n"
            f"This addresses your question with {confidence:.0%} retrieval "
            f"confidence. If this does not resolve the issue, escalate with the "
            f"document id above for engineering review."
        )


class AnthropicProvider:
    """Real LLM generation grounded in the retrieved document."""

    name = "anthropic"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._fallback = LocalSynthesizer()

    def generate(self, question: str, document: KBDocument,
                 confidence: float) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            prompt = (
                "You are a hardware manufacturing support engineer. Answer the "
                "question using ONLY the reference document. Be concise and "
                "actionable. If the document does not fully answer the question, "
                "say so rather than inventing detail. Cite the document id.\n\n"
                f"Question: {question}\n\n"
                f"Reference {document.doc_id} ({document.title}):\n{document.text}"
            )
            resp = client.messages.create(
                model=self.settings.llm_model,
                max_tokens=500,
                timeout=self.settings.llm_timeout_s,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        except Exception:
            # Any provider failure degrades to the deterministic path rather than
            # returning an error to the user. Availability over sophistication.
            return self._fallback.generate(question, document, confidence)


def get_provider(settings: Settings) -> GenerationProvider:
    if settings.effective_provider == "anthropic":
        return AnthropicProvider(settings)
    return LocalSynthesizer()
