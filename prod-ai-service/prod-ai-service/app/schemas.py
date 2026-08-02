"""API schemas. The contract between the service and its callers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="A hardware/manufacturing support question")


class HitModel(BaseModel):
    doc_id: str
    score: float
    title: str
    topic: str


class AskResponse(BaseModel):
    question: str
    answer: str
    cited_doc: str | None
    confidence: float
    abstained: bool
    reason: str | None
    provider: str
    retrieved: list[HitModel]
    latency_ms: float


class SearchResponse(BaseModel):
    query: str
    hits: list[HitModel]
    top_score: float
    margin: float


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    llm_provider: str
    llm_enabled: bool
    documents: int


class EvalResponse(BaseModel):
    with_distractors: dict
    canonical_only: dict


class ErrorResponse(BaseModel):
    detail: str
