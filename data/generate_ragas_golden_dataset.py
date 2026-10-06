"""Auto-build the RAGAS golden dataset: every detected anomaly that matches a real
ground-truth event, with a reference sentence templated from the event's description.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
DETECTED_PATH = DATA_DIR / "detected_anomalies.json"
GROUNDTRUTH_PATH = DATA_DIR / "anomaly_groundtruth.json"
OUTPUT_PATH = DATA_DIR / "ragas_golden_dataset.json"


def build_truth_map(groundtruth: list) -> dict:
    truth_map = {}
    for event in groundtruth:
        for affected_date in event["affected_dates"]:
            truth_map[(event["sku_id"], affected_date)] = event
    return truth_map


def to_reference(event: dict) -> str:
    verb = "spiked because" if event["type"] == "spike" else "dropped because"
    return f"Demand {verb} {event['description']}"


def main():
    detected = json.loads(DETECTED_PATH.read_text())
    groundtruth = json.loads(GROUNDTRUTH_PATH.read_text())
    truth_map = build_truth_map(groundtruth)

    golden = []
    for anomaly in detected:
        key = (anomaly["sku_id"], anomaly["date"])
        event = truth_map.get(key)
        if event is None:
            continue
        golden.append({
            "sku_id": anomaly["sku_id"],
            "date": anomaly["date"],
            "reference": to_reference(event),
        })

    OUTPUT_PATH.write_text(json.dumps(golden, indent=2))
    print(f"Matched {len(golden)} of {len(detected)} detected anomalies to ground-truth events -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
