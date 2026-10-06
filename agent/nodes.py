"""Node functions for the anomaly-explanation LangGraph workflow."""
import os
from datetime import date, timedelta
from typing import TypedDict

from langchain_ollama import ChatOllama

from vector_store import retrieve

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class AgentState(TypedDict, total=False):
    sku_id: str
    date: str
    demand: float
    yhat: float
    residual: float
    zscore: float
    direction: str
    retrieved_docs: list[dict]
    candidates: list[dict]
    explanation: str
    confidence: str
    cited_event_id: str | None
    output: dict


def load_anomaly(state: AgentState) -> dict:
    direction = state.get("direction") or ("spike" if state["zscore"] > 0 else "dip")
    return {"direction": direction}


def retrieve_context(state: AgentState) -> dict:
    query = f"Why did {state['sku_id']} demand {state['direction']} on {state['date']}?"
    docs = retrieve(query, sku_id=state["sku_id"], n_results=3)
    return {"retrieved_docs": docs}


def _within_event_window(event_date: str, duration_days: int, anomaly_date: str) -> bool:
    start = date.fromisoformat(event_date)
    end = start + timedelta(days=duration_days - 1)
    return start <= date.fromisoformat(anomaly_date) <= end


def assess_relevance(state: AgentState) -> dict:
    """Keep only candidates whose actual [event_date, event_date + duration_days) window
    contains the anomaly date. This is a deterministic, code-computed check rather than
    asking the LLM to eyeball a day-gap -- a small local model isn't reliable at that kind
    of numeric reasoning and will hallucinate plausible-sounding justifications for
    candidates hundreds of days away (verified empirically)."""
    candidates = [
        doc
        for doc in state["retrieved_docs"]
        if _within_event_window(doc["date"], doc["duration_days"], state["date"])
    ]
    return {"candidates": candidates}


def generate_explanation(state: AgentState) -> dict:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    candidates = state["candidates"]

    stats = f"SKU {state['sku_id']} on {state['date']}: demand {state['direction']}."
    style_instruction = (
        "Start with 'Demand spiked because' or 'Demand dropped because', stating the cause "
        "directly. You may add one short clause noting the magnitude or how long the effect "
        "is expected to last, if that's mentioned in the source text. Keep it to 2 sentences "
        "maximum. Do not mention z-scores or exact forecast numbers."
    )

    if not candidates:
        prompt = (
            f"{stats}\n\nNo known causal event was found in our records for this date/SKU.\n\n"
            "Answer in exactly one sentence starting with 'Demand spiked' or 'Demand dropped', "
            "stating that the cause is unknown."
        )
        response = llm.invoke(prompt).content.strip()
        return {"explanation": response, "confidence": "low", "cited_event_id": None}

    if len(candidates) == 1:
        cited_event_id = candidates[0]["event_id"]
        prompt = f"{stats}\n\nRelevant known event:\n- {candidates[0]['text']}\n\n{style_instruction}"
        response = llm.invoke(prompt).content.strip()
        return {"explanation": response, "confidence": "high", "cited_event_id": cited_event_id}

    candidates_block = "\n".join(f"- [event_id={c['event_id']}] {c['text']}" for c in candidates)
    prompt = (
        f"{stats}\n\nMultiple known events could explain this anomaly:\n{candidates_block}\n\n"
        f"Pick the one that best explains it. {style_instruction}\n\n"
        "End your response on its own final line with exactly: CITED_EVENT: <event_id>"
    )
    response = llm.invoke(prompt).content.strip()
    explanation, _, tag = response.rpartition("CITED_EVENT:")
    cited_event_id = tag.strip().strip(".")
    known_ids = {c["event_id"] for c in candidates}
    if cited_event_id not in known_ids:
        cited_event_id = candidates[0]["event_id"]

    return {
        "explanation": explanation.strip() or response,
        "confidence": "high",
        "cited_event_id": cited_event_id,
    }


def format_output(state: AgentState) -> dict:
    output = {
        "sku_id": state["sku_id"],
        "date": state["date"],
        "direction": state["direction"],
        "demand": state["demand"],
        "forecast": state["yhat"],
        "zscore": state["zscore"],
        "explanation": state["explanation"],
        "confidence": state["confidence"],
        "supporting_event_ids": [state["cited_event_id"]] if state.get("cited_event_id") else [],
    }
    return {"output": output}
