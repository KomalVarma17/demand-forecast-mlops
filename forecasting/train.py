"""Train a per-SKU Prophet model, evaluate on a 14-day holdout, and log to MLflow.

Each SKU gets an eval fit (on data up to the holdout cutoff, scored against the
held-out days) and a production fit (refit on full history) that predict.py uses
to forecast the true future. Structured as separate functions so a future
scheduler can call train/evaluate/promote independently.
"""
from pathlib import Path

import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prophet import Prophet

DATA_DIR = Path(__file__).parent.parent / "data"
DEMAND_PATH = DATA_DIR / "synthetic_demand.csv"
HOLDOUT_DAYS = 14
EXPERIMENT_NAME = "demand-forecast"
CHAMPION_ALIAS = "champion"
PROMOTION_METRIC = "mape"  # lower is better


def load_sku_series(df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    sku_df = df[df["sku_id"] == sku_id][["date", "demand"]].rename(columns={"date": "ds", "demand": "y"})
    sku_df["ds"] = pd.to_datetime(sku_df["ds"])
    return sku_df.sort_values("ds").reset_index(drop=True)


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(train_df)
    return model


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    safe_actual = np.where(actual == 0, np.nan, actual)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.nanmean(np.abs(error / safe_actual)) * 100),
    }


def train_sku(sku_df: pd.DataFrame) -> tuple[dict, Prophet]:
    cutoff = len(sku_df) - HOLDOUT_DAYS
    train_df, holdout_df = sku_df.iloc[:cutoff], sku_df.iloc[cutoff:]

    eval_model = fit_prophet(train_df)
    future = eval_model.make_future_dataframe(periods=HOLDOUT_DAYS)
    forecast = eval_model.predict(future)
    predicted = forecast.set_index("ds").loc[holdout_df["ds"], "yhat"].values
    metrics = evaluate_forecast(holdout_df["y"].values, predicted)

    production_model = fit_prophet(sku_df)
    return metrics, production_model


def promote_if_better(client: MlflowClient, model_name: str, version: str, metrics: dict) -> bool:
    """Promote `version` to champion if no champion exists yet, or if it beats the
    current champion on PROMOTION_METRIC. Returns whether promotion happened."""
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        champion_metric = client.get_run(champion.run_id).data.metrics[PROMOTION_METRIC]
        if metrics[PROMOTION_METRIC] >= champion_metric:
            return False
    except MlflowException:
        pass  # no champion yet -> first version is auto-promoted

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    return True


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    df = pd.read_csv(DEMAND_PATH)

    with mlflow.start_run(run_name="train_all_skus"):
        for sku_id in sorted(df["sku_id"].unique()):
            sku_df = load_sku_series(df, sku_id)
            with mlflow.start_run(run_name=sku_id, nested=True):
                metrics, model = train_sku(sku_df)
                mlflow.log_params({"holdout_days": HOLDOUT_DAYS, "history_days": len(sku_df)})
                mlflow.log_metrics(metrics)
                model_name = f"prophet-{sku_id}"
                model_info = mlflow.prophet.log_model(model, name="model", registered_model_name=model_name)
                promoted = promote_if_better(client, model_name, model_info.registered_model_version, metrics)
                status = "-> champion" if promoted else "(not promoted)"
                print(f"{sku_id}: MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  MAPE={metrics['mape']:.1f}%  {status}")


if __name__ == "__main__":
    main()
