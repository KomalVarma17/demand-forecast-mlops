"""LangGraph StateGraph wiring the 5-node anomaly-explanation workflow."""
import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from nodes import (
    AgentState,
    assess_relevance,
    format_output,
    generate_explanation,
    load_anomaly,
    retrieve_context,
)

DATA_DIR = Path(__file__).parent.parent / "data"
ANOMALIES_PATH = DATA_DIR / "detected_anomalies.json"

_APP = None


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_anomaly", load_anomaly)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("assess_relevance", assess_relevance)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "load_anomaly")
    graph.add_edge("load_anomaly", "retrieve_context")
    graph.add_edge("retrieve_context", "assess_relevance")
    graph.add_edge("assess_relevance", "generate_explanation")
    graph.add_edge("generate_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


def explain_anomaly(anomaly: dict) -> dict:
    global _APP
    if _APP is None:
        _APP = build_graph()
    result = _APP.invoke(anomaly)
    return result["output"]


def main():
    anomalies = json.loads(ANOMALIES_PATH.read_text())
    anomaly = anomalies[0]
    print(f"Explaining anomaly: {anomaly['sku_id']} on {anomaly['date']}")
    output = explain_anomaly(anomaly)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
