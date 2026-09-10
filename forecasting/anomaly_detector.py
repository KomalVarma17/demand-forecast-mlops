"""Flag anomalies by z-scoring residuals (actual - yhat) per SKU on historical dates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DEMAND_PATH = DATA_DIR / "synthetic_demand.csv"
FORECASTS_PATH = DATA_DIR / "forecasts.csv"
GROUNDTRUTH_PATH = DATA_DIR / "anomaly_groundtruth.json"
OUTPUT_PATH = DATA_DIR / "detected_anomalies.json"
Z_THRESHOLD = 2.5


def compute_residual_zscores(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    historical = forecast_df[forecast_df["is_future"] == 0]
    merged = actual_df.merge(historical[["date", "sku_id", "yhat"]], on=["date", "sku_id"], how="inner")
    merged["residual"] = merged["demand"] - merged["yhat"]
    merged["zscore"] = merged.groupby("sku_id")["residual"].transform(lambda r: (r - r.mean()) / r.std(ddof=0))
    return merged


def flag_anomalies(merged: pd.DataFrame, threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    flagged = merged[merged["zscore"].abs() >= threshold].copy()
    flagged["direction"] = np.where(flagged["zscore"] > 0, "spike", "dip")
    return flagged.sort_values(["sku_id", "date"])


def score_against_groundtruth(flagged: pd.DataFrame, groundtruth: list) -> dict:
    truth_pairs = {(e["sku_id"], d) for e in groundtruth for d in e["affected_dates"]}
    flagged_pairs = set(zip(flagged["sku_id"], flagged["date"]))
    true_positives = flagged_pairs & truth_pairs
    precision = len(true_positives) / len(flagged_pairs) if flagged_pairs else 0.0
    recall = len(true_positives) / len(truth_pairs) if truth_pairs else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positives": len(true_positives),
        "flagged": len(flagged_pairs),
        "groundtruth": len(truth_pairs),
    }


def main():
    actual_df = pd.read_csv(DEMAND_PATH)
    forecast_df = pd.read_csv(FORECASTS_PATH)
    groundtruth = json.loads(GROUNDTRUTH_PATH.read_text())

    merged = compute_residual_zscores(actual_df, forecast_df)
    flagged = flag_anomalies(merged)

    records = flagged[["date", "sku_id", "demand", "yhat", "residual", "zscore", "direction"]].to_dict("records")
    OUTPUT_PATH.write_text(json.dumps(records, indent=2))

    summary = score_against_groundtruth(flagged, groundtruth)
    print(f"Flagged {len(flagged)} anomalies (|z| >= {Z_THRESHOLD}) -> {OUTPUT_PATH}")
    print(
        f"Precision={summary['precision']:.2f}  Recall={summary['recall']:.2f}  "
        f"(TP={summary['true_positives']}, flagged={summary['flagged']}, groundtruth={summary['groundtruth']})"
    )


if __name__ == "__main__":
    main()
