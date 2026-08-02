"""
Offline evaluation harness.

Scores the system against declared ground truth in the knowledge base rather
than by inspection, and is exposed both as a CLI (``python -m app.core.evaluate``)
and as an API endpoint. The metrics that matter for a production RAG service:

  retrieval    hit@1, hit@3, MRR
  end-to-end   correct answer rate, abstention rate, false answer rate

False answer rate is the release gate. Correct rate measures usefulness;
abstention measures caution; false-answer measures the failure that actually
erodes trust -- a confident answer from the wrong document.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from ..config import get_settings
from .agent import RAGAgent
from .knowledge_base import GROUND_TRUTH, validate
from .retriever import Retriever


@dataclass
class EvalReport:
    hit_at_1: float
    hit_at_3: float
    mrr: float
    correct_rate: float
    abstention_rate: float
    false_answer_rate: float
    n: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def run_eval(include_distractors: bool = True) -> EvalReport:
    validate()
    retriever = Retriever(include_distractors=include_distractors)
    agent = RAGAgent(get_settings(), retriever=retriever)

    hit1 = hit3 = rr = 0.0
    correct = abstained = false_answer = 0

    for question, expected in GROUND_TRUTH.items():
        result = retriever.search(question, k=5)
        ids = [h.doc_id for h in result.hits]
        rank = ids.index(expected) + 1 if expected in ids else None

        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += (1.0 / rank) if rank else 0.0

        ans = agent.answer(question)
        if ans.abstained:
            abstained += 1
        elif ans.cited_doc == expected:
            correct += 1
        else:
            false_answer += 1

    n = len(GROUND_TRUTH)
    return EvalReport(
        hit_at_1=hit1 / n, hit_at_3=hit3 / n, mrr=rr / n,
        correct_rate=correct / n, abstention_rate=abstained / n,
        false_answer_rate=false_answer / n, n=n,
    )


def main() -> None:
    import json

    print("with distractors:  ",
          json.dumps(run_eval(include_distractors=True).as_dict()))
    print("canonical only:    ",
          json.dumps(run_eval(include_distractors=False).as_dict()))


if __name__ == "__main__":
    main()
