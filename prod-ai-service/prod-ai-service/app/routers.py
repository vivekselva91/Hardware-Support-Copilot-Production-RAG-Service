"""API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .config import Settings, get_settings
from .core.agent import RAGAgent
from .core.evaluate import run_eval
from .core.knowledge_base import DOCUMENTS
from .core.retriever import Retriever
from .schemas import (AskRequest, AskResponse, EvalResponse, HealthResponse,
                      HitModel, SearchResponse)

router = APIRouter()

# Retriever is built once at import and reused; it is read-only after fit.
_retriever = Retriever()


def get_agent(settings: Settings = Depends(get_settings)) -> RAGAgent:
    return RAGAgent(settings, retriever=_retriever)


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok", version=settings.version, environment=settings.environment,
        llm_provider=settings.effective_provider, llm_enabled=settings.llm_enabled,
        documents=len(DOCUMENTS),
    )


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest, agent: RAGAgent = Depends(get_agent)) -> AskResponse:
    result = agent.answer(req.question)
    return AskResponse(
        question=result.question, answer=result.answer, cited_doc=result.cited_doc,
        confidence=result.confidence, abstained=result.abstained,
        reason=result.reason, provider=result.provider, latency_ms=result.latency_ms,
        retrieved=[HitModel(doc_id=h.doc_id, score=round(h.score, 4),
                            title=h.title, topic=h.topic)
                   for h in result.retrieved],
    )


@router.get("/search", response_model=SearchResponse, tags=["rag"])
def search(q: str, k: int = 5,
           settings: Settings = Depends(get_settings)) -> SearchResponse:
    result = _retriever.search(q, k=k)
    return SearchResponse(
        query=q, top_score=round(result.top_score, 4),
        margin=round(result.margin, 4),
        hits=[HitModel(doc_id=h.doc_id, score=round(h.score, 4),
                       title=h.title, topic=h.topic) for h in result.hits],
    )


@router.get("/eval", response_model=EvalResponse, tags=["ops"])
def evaluate() -> EvalResponse:
    return EvalResponse(
        with_distractors=run_eval(include_distractors=True).as_dict(),
        canonical_only=run_eval(include_distractors=False).as_dict(),
    )
