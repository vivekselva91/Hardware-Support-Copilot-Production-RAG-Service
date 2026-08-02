"""Tests for the hardware support RAG service."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.agent import RAGAgent
from app.core.evaluate import run_eval
from app.core.knowledge_base import GROUND_TRUTH, validate
from app.core.retriever import Retriever
from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def agent():
    return RAGAgent(get_settings())


# ------------------------------ knowledge base ------------------------------

def test_knowledge_base_validates():
    validate()


def test_ground_truth_questions_resolve():
    r = Retriever()
    for question, expected in GROUND_TRUTH.items():
        ids = [h.doc_id for h in r.search(question, k=5).hits]
        assert expected in ids, f"{expected} not retrieved for '{question}'"


# --------------------------------- retriever --------------------------------

def test_empty_query_returns_nothing():
    assert Retriever().search("", k=5).hits == []


def test_scores_are_ordered():
    hits = Retriever().search("reflow tombstoning solder", k=5).hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------- agent ----------------------------------

def test_agent_answers_clear_question(agent):
    a = agent.answer("first pass yield dropped across several stations how do i recover")
    assert not a.abstained
    assert a.cited_doc == "YIELD-07"
    assert a.cited_doc in a.answer


def test_agent_abstains_on_offtopic(agent):
    a = agent.answer("what is the capital of France")
    assert a.abstained
    assert a.cited_doc is None
    assert "escalat" in a.answer.lower()


def test_agent_abstains_on_ambiguous_near_duplicate(agent):
    """
    The sensor-drift query ranks a distractor a hair above the right answer.
    The agent must abstain rather than confidently cite the wrong document --
    this is the production-safety property, not an edge case.
    """
    a = agent.answer("sensor reading drifts when it gets hot but is fine at room temp")
    assert a.abstained
    assert "disambiguate" in (a.reason or "")


def test_confidence_bounded(agent):
    for question in GROUND_TRUTH:
        a = agent.answer(question)
        assert 0.0 <= a.confidence <= 1.0


def test_latency_recorded(agent):
    a = agent.answer("reflow profile for tombstoning")
    assert a.latency_ms >= 0


# ------------------------------- evaluation ---------------------------------

def test_eval_zero_false_answers():
    """The release gate: never confidently answer from the wrong document."""
    report = run_eval(include_distractors=True)
    assert report.false_answer_rate == 0.0


def test_eval_useful_majority():
    """Abstention must not be achieved by abstaining on everything."""
    report = run_eval(include_distractors=True)
    assert report.correct_rate >= 0.6


def test_distractors_make_it_harder():
    clean = run_eval(include_distractors=False)
    hard = run_eval(include_distractors=True)
    assert clean.hit_at_1 >= hard.hit_at_1


# ---------------------------------- API -------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] > 0


def test_ask_endpoint(client):
    r = client.post("/ask", json={
        "question": "eye diagram is closing on the fast lanes of a dense pcb"})
    assert r.status_code == 200
    body = r.json()
    assert body["cited_doc"] == "SIGNAL-06"
    assert not body["abstained"]


def test_ask_rejects_empty(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # schema validation


def test_ask_rejects_overlong(client):
    r = client.post("/ask", json={"question": "x" * 1001})
    assert r.status_code == 422


def test_search_endpoint(client):
    r = client.get("/search", params={"q": "thermal throttle compute", "k": 3})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 3


def test_eval_endpoint(client):
    r = client.get("/eval")
    assert r.status_code == 200
    assert r.json()["with_distractors"]["false_answer_rate"] == 0.0
