"""
Retrieval over the knowledge base.

TF-IDF with cosine similarity: deterministic, dependency-light, and runnable
with no model download or API key. The retriever exposes not just ranked hits
but a confidence signal derived from the top score and the margin over the
runner-up, which the agent uses to decide whether to answer or abstain.

Lexical retrieval is honest about its limits -- it matches shared vocabulary,
not meaning. The abstention logic downstream exists precisely because a lexical
retriever will sometimes rank a near-duplicate distractor a hair above the right
answer, and a production system must decline rather than confidently mislead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import KBDocument, build_documents


@dataclass
class Hit:
    doc_id: str
    score: float
    title: str
    topic: str


@dataclass
class RetrievalResult:
    hits: list[Hit]
    top_score: float
    margin: float

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None


class Retriever:
    def __init__(self, include_distractors: bool = True,
                 ngram_range: tuple[int, int] = (1, 2)):
        self.documents = build_documents(include_distractors)
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, sublinear_tf=True, stop_words="english")
        self._matrix = self.vectorizer.fit_transform(
            [d.searchable() for d in self.documents])

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(hits=[], top_score=0.0, margin=0.0)

        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        order = np.argsort(scores)[::-1][:k]

        hits = [
            Hit(doc_id=self.documents[i].doc_id, score=float(scores[i]),
                title=self.documents[i].title, topic=self.documents[i].topic)
            for i in order
        ]
        top_score = hits[0].score if hits else 0.0
        margin = (hits[0].score - hits[1].score) if len(hits) > 1 else top_score
        return RetrievalResult(hits=hits, top_score=top_score, margin=margin)
