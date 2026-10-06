"""RAGAS evaluation: score the agent's anomaly explanations against golden references."""
import json
import os
import sys
from pathlib import Path
import mlflow

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import BleuScore, RougeScore, SemanticSimilarity

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
from graph import explain_anomaly

DATA_DIR = Path(__file__).parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "ragas_golden_dataset.json"
ANOMALIES_PATH = DATA_DIR / "detected_anomalies.json"
RESULTS_PATH = DATA_DIR / "ragas_results.json"
EXPERIMENT_NAME = "demand-forecast-agent-eval"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def find_anomaly(anomalies: list, sku_id: str, date: str) -> dict:
    for a in anomalies:
        if a["sku_id"] == sku_id and a["date"] == date:
            return a
    raise ValueError(f"No detected anomaly for {sku_id} on {date}")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    golden = json.loads(GOLDEN_PATH.read_text())
    anomalies = json.loads(ANOMALIES_PATH.read_text())

    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL, use_api=False)
    sem_sim = SemanticSimilarity(embeddings=embeddings)
    bleu = BleuScore()
    rouge = RougeScore()

    with mlflow.start_run(run_name="ragas_eval"):
        mlflow.log_params({
            "golden_dataset_size": len(golden),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "embedding_model": EMBEDDING_MODEL,
        })

        results = []
        for item in golden:
            anomaly = find_anomaly(anomalies, item["sku_id"], item["date"])
            output = explain_anomaly(anomaly)
            response = output["explanation"]

            scores = {
                "semantic_similarity": sem_sim.score(reference=item["reference"], response=response).value,
                "bleu_score": bleu.score(reference=item["reference"], response=response).value,
                "rouge_score": rouge.score(reference=item["reference"], response=response).value,
            }
            results.append({**item, "response": response, **scores})
            print(f"{item['sku_id']} {item['date']}: sem_sim={scores['semantic_similarity']:.2f}  "
                  f"bleu={scores['bleu_score']:.2f}  rouge={scores['rouge_score']:.2f}")

        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        avg = {k: sum(r[k] for r in results) / len(results) for k in ("semantic_similarity", "bleu_score", "rouge_score")}
        mlflow.log_metrics(avg)
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(GOLDEN_PATH))

        print(f"\nAverages across {len(results)} samples: {avg}")
        print(f"Full results -> {RESULTS_PATH}")



if __name__ == "__main__":
    main()
