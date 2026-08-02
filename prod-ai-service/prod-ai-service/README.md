# Hardware Support Copilot — Production RAG Service

A deployable retrieval-augmented generation service that answers hardware
manufacturing and bring-up questions — SMT process, sensor calibration, thermal,
signal integrity, yield recovery, NPI gating — grounded in a technical knowledge
base.

Built to demonstrate the full production-AI stack end to end: **retrieval, a
confidence-gated agent, an evaluation harness, a served API, a UI, and
containerised deployment** — not a notebook.

---

## What makes it "production" rather than a demo

| Concern | How it is handled |
|---|---|
| **Serving** | FastAPI with typed schemas, OpenAPI docs, health check, structured request logging |
| **Reliability** | Provider failures degrade to a deterministic local path rather than erroring |
| **Safety** | The agent abstains when retrieval is weak or ambiguous instead of guessing |
| **Evaluation** | Metrics scored against declared ground truth, exposed as a CLI and an endpoint |
| **Guardrails** | Input validation, request size limits, in-memory rate limiting |
| **Reproducibility** | Runs fully offline with no API key; upgrades to a real LLM when one is present |
| **Deployment** | Dockerfile + compose for API and UI, with a container health check |

---

## Quick start

```bash
pip install -r requirements.txt

# API
uvicorn app.main:app --port 8000     # docs at http://localhost:8000/docs

# UI (separate terminal)
streamlit run ui.py

# offline metrics
python -m app.core.evaluate

# tests
pytest tests/ -q
```

Or the whole thing in containers:

```bash
docker compose up --build           # API on :8000, UI on :8501
```

**No API key required.** The service runs on a deterministic local synthesizer
out of the box — a reviewer can clone and run it in one command. Set
`ANTHROPIC_API_KEY` (see `.env.example`) and generation automatically upgrades to
a real LLM; retrieval, confidence gating and the abstention decision are
identical either way, so the offline path is a faithful stand-in, not a toy.

---

## The architecture

```
             ┌──────────────┐        ┌──────────────┐
   client ──▶│  FastAPI     │───────▶│  RAG agent   │
   / UI      │  /ask /search│        └──────┬───────┘
             │  /eval /health│              │
             └──────────────┘        ┌──────▼───────┐   retrieve
                                     │  retriever   │◀── TF-IDF over
                                     └──────┬───────┘    knowledge base
                                            │
                                     ┌──────▼───────┐   gate on confidence
                                     │ confidence   │── top score + margin
                                     │ gate         │
                                     └──────┬───────┘
                                   answer   │   abstain
                                  ┌─────────▼──────────┐
                                  │ provider           │  local synthesizer
                                  │ (local | anthropic)│  or real LLM
                                  └────────────────────┘
```

All logic lives behind the API. The Streamlit UI, the CLI, CI, and any other
caller hit the same endpoints, so the intelligence is tested once at the service
layer rather than reimplemented per interface.

---

## The design decision that matters: abstention

A naive RAG service always answers. That means it sometimes answers confidently
from the *wrong* document — the failure mode that quietly destroys trust in a
support tool.

This agent gates every answer on two conditions:

- **relevance** — the top retrieval score must clear an absolute floor
- **separation** — the top result must beat the runner-up by a margin

Fail either, and it returns a grounded escalation with its candidate list rather
than a fabricated answer.

### This is not theoretical — the eval catches a real case

One knowledge-base query — *"sensor reading drifts when it gets hot but is fine
at room temperature"* — ranks a **distractor** document (factory calibration
procedure) a hair above the **correct** one (thermal drift compensation), a
margin of 0.014. Lexical retrieval genuinely cannot separate them.

The agent abstains. That is the system working as designed: a confident wrong
answer would send an engineer to recalibrate at the factory bench when the real
fix is thermal recompensation. The escalation costs a minute; the wrong answer
costs a misdirected repair.

---

## Measured performance

Scored against declared ground truth (`python -m app.core.evaluate` or `GET /eval`):

| Metric | With distractors | Canonical only |
|---|---|---|
| Retrieval hit@1 | 0.75 | 1.00 |
| Retrieval hit@3 | 1.00 | 1.00 |
| MRR | 0.88 | 1.00 |
| Correct answer rate | 0.75 | 1.00 |
| Abstention rate | 0.25 | 0.00 |
| **False answer rate** | **0.00** | **0.00** |

**False answer rate is the release gate.** Correct rate measures usefulness,
abstention measures caution, and false-answer measures the failure that erodes
trust. Zero false answers with and without distractors is the number to hold.

The gap between the two columns is deliberate. Against a clean corpus retrieval
is perfect and proves nothing; the **near-duplicate distractor documents** are
what make the evaluation real. A system that scores 1.00 on a curated corpus and
collapses on near-misses has been flattered, not evaluated.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, version, active provider, document count |
| `/ask` | POST | Full RAG answer with citation, confidence, and abstention flag |
| `/search` | GET | Raw retrieval hits and scores (debugging / transparency) |
| `/eval` | GET | Run the evaluation harness on demand |
| `/docs` | GET | Interactive OpenAPI documentation |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "first pass yield dropped across several stations, how do I recover?"}'
```

```json
{
  "answer": "Based on the knowledge base (YIELD-07: First pass yield recovery method): ...",
  "cited_doc": "YIELD-07",
  "confidence": 0.71,
  "abstained": false,
  "provider": "local",
  "latency_ms": 2.0,
  "retrieved": [{"doc_id": "YIELD-07", "score": 0.31, "topic": "yield", "title": "..."}]
}
```

---

## Repository layout

```
├── app/
│   ├── main.py              FastAPI app, middleware wiring, lifespan
│   ├── config.py            Env-driven settings, provider resolution
│   ├── schemas.py           Typed request/response contracts
│   ├── routers.py           Endpoint handlers
│   ├── middleware.py        Rate limiting, structured request logging
│   └── core/
│       ├── knowledge_base.py  Documents, distractors, ground truth
│       ├── retriever.py       TF-IDF retrieval with confidence signal
│       ├── providers.py       Local synthesizer + LLM provider, one interface
│       ├── agent.py           Retrieve → gate → generate/abstain
│       └── evaluate.py        Retrieval + end-to-end metrics
├── ui.py                    Streamlit front-end (thin client over the API)
├── tests/                   18 tests
├── Dockerfile
├── docker-compose.yml       API + UI
└── requirements.txt
```

---

## Testing

18 tests asserting behaviour, not just execution:

- Every ground-truth question retrieves its answer within top-5
- Retrieval scores are correctly ordered
- The agent answers clear questions and cites the right document
- **The agent abstains on the ambiguous near-duplicate** rather than guessing
- **False answer rate is zero** — the release gate
- Correct-answer rate stays above 0.6 — abstention is not achieved by abstaining on everything
- Distractors measurably lower hit@1 — the eval is not flattered
- API validates input (empty and over-length questions rejected with 422)
- Health, ask, search and eval endpoints all respond correctly

```bash
pytest tests/ -q     # 18 passed
```

---

## Honest limitations

- **n = 8 ground-truth questions.** Enough to demonstrate method and expose the
  abstention case; not enough for the hit-rate differences to be statistically
  strong.
- **Lexical retrieval.** TF-IDF matches vocabulary, not meaning, which is
  exactly why the sensor-drift query is ambiguous. A learned embedding model is
  the obvious upgrade and would slot in behind the same retriever interface — the
  eval harness would then quantify whether it actually helps rather than assuming
  it does.
- **In-memory rate limiting.** Correct for a single instance; multi-instance
  deployment would move that state to Redis behind the same middleware interface.
- **Local synthesizer formats, it does not reason.** The offline generation path
  composes an answer from the retrieved document. The intelligence in the
  keyless configuration is in retrieval and abstention; the LLM provider adds
  fluent synthesis on top.
- **Synthetic knowledge base.** The documents are authored, compact, and
  consistent. Real support corpora are messier, and retrieval against them is
  harder.

The abstention case points directly at the highest-value next step: semantic
embeddings to separate documents that share vocabulary but not meaning.
