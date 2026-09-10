"""Node functions for the anomaly-explanation LangGraph workflow."""
import os
from datetime import date
from typing import TypedDict

from langchain_ollama import ChatOllama

from vector_store import retrieve

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RELEVANCE_DISTANCE_THRESHOLD = 0.8
RELEVANCE_DATE_WINDOW_DAYS = 3


class AgentState(TypedDict, total=False):
    sku_id: str
    date: str
    demand: float
    yhat: float
    residual: float
    zscore: float
    direction: str
    retrieved_docs: list[dict]
    relevant_docs: list[dict]
    has_relevant_context: bool
    explanation: str
    confidence: str
    output: dict


def load_anomaly(state: AgentState) -> dict:
    direction = state.get("direction") or ("spike" if state["zscore"] > 0 else "dip")
    return {"direction": direction}


def retrieve_context(state: AgentState) -> dict:
    query = f"Why did {state['sku_id']} demand {state['direction']} on {state['date']}?"
    docs = retrieve(query, sku_id=state["sku_id"], n_results=3)
    return {"retrieved_docs": docs}


def _days_between(a: str, b: str) -> int:
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def assess_relevance(state: AgentState) -> dict:
    relevant = [
        doc
        for doc in state["retrieved_docs"]
        if doc["distance"] <= RELEVANCE_DISTANCE_THRESHOLD
        and _days_between(doc["date"], state["date"]) <= RELEVANCE_DATE_WINDOW_DAYS
    ]
    return {"relevant_docs": relevant, "has_relevant_context": bool(relevant)}


def generate_explanation(state: AgentState) -> dict:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    stats = (
        f"SKU {state['sku_id']} on {state['date']}: actual demand was {state['demand']:.0f} units "
        f"vs a forecast of {state['yhat']:.0f} (z-score {state['zscore']:.2f}, a {state['direction']})."
    )

    if state["has_relevant_context"]:
        context = "\n".join(f"- {doc['text']}" for doc in state["relevant_docs"])
        prompt = (
            f"{stats}\n\nRelevant known events:\n{context}\n\n"
            "In 2-3 sentences, explain in plain English what most likely caused this demand "
            "anomaly, citing the relevant event(s)."
        )
        confidence = "high"
    else:
        prompt = (
            f"{stats}\n\nNo known causal event was found in our records for this date/SKU.\n\n"
            "In 1-2 sentences, note that this anomaly is unexplained by known events and may "
            "warrant investigation."
        )
        confidence = "low"

    response = llm.invoke(prompt)
    return {"explanation": response.content.strip(), "confidence": confidence}


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
        "supporting_event_ids": [doc["event_id"] for doc in state.get("relevant_docs", [])],
    }
    return {"output": output}
