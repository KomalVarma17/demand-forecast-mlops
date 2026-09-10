"""Generate historical fitted values + a 14-day rolling forecast per SKU using
the MLflow-registered champion Prophet model.
"""
from pathlib import Path

import mlflow.prophet
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DEMAND_PATH = DATA_DIR / "synthetic_demand.csv"
OUTPUT_PATH = DATA_DIR / "forecasts.csv"
FORECAST_HORIZON_DAYS = 14


def forecast_sku(sku_id: str, last_actual_date: pd.Timestamp) -> pd.DataFrame:
    model = mlflow.prophet.load_model(f"models:/prophet-{sku_id}@champion")
    future = model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
    forecast = model.predict(future)
    forecast["sku_id"] = sku_id
    forecast["is_future"] = (forecast["ds"] > last_actual_date).astype(int)
    return forecast[["ds", "sku_id", "yhat", "yhat_lower", "yhat_upper", "is_future"]]


def main():
    actual_df = pd.read_csv(DEMAND_PATH)
    actual_df["date"] = pd.to_datetime(actual_df["date"])

    all_forecasts = []
    for sku_id in sorted(actual_df["sku_id"].unique()):
        last_actual_date = actual_df.loc[actual_df["sku_id"] == sku_id, "date"].max()
        all_forecasts.append(forecast_sku(sku_id, last_actual_date))
        horizon_end = (last_actual_date + pd.Timedelta(days=FORECAST_HORIZON_DAYS)).date()
        print(f"{sku_id}: forecasted through {horizon_end}")

    result = pd.concat(all_forecasts, ignore_index=True).rename(columns={"ds": "date"})
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(result):,} forecast rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
