"""
The RAG agent.

Orchestrates the pipeline: retrieve -> gate on confidence -> generate or abstain.

The gate is the part that makes this production-shaped rather than a demo. A
naive RAG app always answers, which means it sometimes answers confidently from
the wrong document. This agent abstains when retrieval is weak (top score below
a floor) or ambiguous (top two results too close to separate), returning a
grounded escalation instead of a fabricated answer. Abstention is a feature, and
it is measured in the eval harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Settings
from .knowledge_base import doc_index
from .providers import get_provider
from .retriever import Hit, Retriever


@dataclass
class AgentAnswer:
    question: str
    answer: str
    cited_doc: str | None
    confidence: float
    abstained: bool
    reason: str | None
    provider: str
    retrieved: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0


class RAGAgent:
    def __init__(self, settings: Settings, retriever: Retriever | None = None):
        self.settings = settings
        self.retriever = retriever or Retriever()
        self.docs = doc_index()

    def answer(self, question: str) -> AgentAnswer:
        started = time.perf_counter()
        s = self.settings

        result = self.retriever.search(question, k=s.retrieval_top_k)
        provider = get_provider(s)

        base = AgentAnswer(
            question=question, answer="", cited_doc=None, confidence=0.0,
            abstained=False, reason=None, provider=provider.name,
            retrieved=result.hits,
        )

        if not result.hits:
            base.abstained = True
            base.reason = "no documents retrieved"
            base.answer = self._abstain_text(base)
            base.latency_ms = self._elapsed(started)
            return base

        confidence = self._confidence(result.top_score, result.margin)
        base.confidence = confidence

        if result.top_score < s.min_relevance:
            base.abstained = True
            base.reason = (f"top relevance {result.top_score:.3f} below "
                           f"floor {s.min_relevance}")
        elif result.margin < s.min_margin:
            base.abstained = True
            base.reason = (f"top two results within {result.margin:.3f}; "
                           f"cannot disambiguate")

        if base.abstained:
            base.answer = self._abstain_text(base)
            base.latency_ms = self._elapsed(started)
            return base

        top = result.top
        base.cited_doc = top.doc_id
        base.answer = provider.generate(question, self.docs[top.doc_id], confidence)
        base.latency_ms = self._elapsed(started)
        return base

    @staticmethod
    def _confidence(top_score: float, margin: float) -> float:
        return float(min(1.0, 0.6 * min(top_score / 0.4, 1.0)
                              + 0.4 * min(margin / 0.12, 1.0)))

    @staticmethod
    def _abstain_text(a: AgentAnswer) -> str:
        candidates = ", ".join(f"{h.doc_id} ({h.score:.2f})" for h in a.retrieved[:3])
        return (
            "I could not find a confident answer to this question in the "
            f"knowledge base. Reason: {a.reason}. Closest candidates: "
            f"{candidates or 'none'}. Escalating to a human engineer is "
            "recommended rather than guessing."
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
