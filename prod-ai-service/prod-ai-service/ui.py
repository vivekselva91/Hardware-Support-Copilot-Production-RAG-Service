"""
Streamlit front-end for the Hardware Support Copilot.

A thin client over the FastAPI service. Kept deliberately thin: all logic --
retrieval, confidence gating, generation -- lives in the API, and this UI only
renders it. That separation is the point. The same backend serves this UI, a
CLI, CI, and any other caller, so the intelligence is tested once at the API
layer rather than reimplemented in the interface.

    # start the API first
    uvicorn app.main:app --port 8000
    # then the UI
    streamlit run ui.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hardware Support Copilot", page_icon="•",
                   layout="centered")

st.title("Hardware Support Copilot")
st.caption("Retrieval-augmented support for SMT, sensor, thermal, signal-integrity "
           "and NPI questions. Abstains rather than guessing when unsure.")

# Health banner.
try:
    health = httpx.get(f"{API_URL}/health", timeout=5).json()
    provider = health["llm_provider"]
    badge = "LLM" if health["llm_enabled"] else "local synthesizer"
    st.success(f"Connected · {health['documents']} docs · generation: {badge}")
except Exception:
    st.error(f"Cannot reach the API at {API_URL}. Start it with "
             f"`uvicorn app.main:app --port 8000`.")
    st.stop()

examples = [
    "Why are small resistors standing up on end after reflow?",
    "First pass yield dropped across several stations, how do I recover?",
    "Eye diagram is closing on the fast lanes of a dense PCB.",
    "When can I move from EVT to the next build phase?",
]

with st.sidebar:
    st.subheader("Try an example")
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

    st.markdown("---")
    if st.button("Run evaluation", use_container_width=True):
        with st.spinner("Scoring against ground truth..."):
            ev = httpx.get(f"{API_URL}/eval", timeout=30).json()
        st.session_state["eval"] = ev

question = st.text_area("Ask a hardware support question", key="question", height=90)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and answering..."):
        resp = httpx.post(f"{API_URL}/ask", json={"question": question},
                          timeout=40).json()

    if resp["abstained"]:
        st.warning("**Abstained** — the system declined to answer rather than guess.")
        st.caption(f"Reason: {resp['reason']}")
    else:
        st.markdown(resp["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Cited doc", resp["cited_doc"])
        c2.metric("Confidence", f"{resp['confidence']:.0%}")
        c3.metric("Latency", f"{resp['latency_ms']:.0f} ms")

    with st.expander("Retrieved documents"):
        for h in resp["retrieved"]:
            st.write(f"**{h['doc_id']}** ({h['topic']}) — {h['title']}  ·  "
                     f"score {h['score']:.3f}")

if "eval" in st.session_state:
    st.markdown("---")
    st.subheader("Evaluation vs. ground truth")
    ev = st.session_state["eval"]
    col1, col2 = st.columns(2)
    with col1:
        st.caption("With distractors")
        st.json(ev["with_distractors"])
    with col2:
        st.caption("Canonical only")
        st.json(ev["canonical_only"])
    st.caption("The gap between the two columns is the honest measure of "
               "retrieval quality. False answer rate is the release gate.")
