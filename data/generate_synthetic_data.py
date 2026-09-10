"""Generate synthetic daily demand for 15 SKUs with seasonality + injected anomaly events.
EVENTS is the single source of truth for both the demand series and anomaly_groundtruth.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
START_DATE = "2023-01-01"
NUM_DAYS = 730
SEED = 42

SKUS = [
    {"sku_id": "SKU_001", "category": "Electronics", "base_demand": 220},
    {"sku_id": "SKU_002", "category": "Electronics", "base_demand": 180},
    {"sku_id": "SKU_003", "category": "Electronics", "base_demand": 95},
    {"sku_id": "SKU_004", "category": "Grocery", "base_demand": 480},
    {"sku_id": "SKU_005", "category": "Grocery", "base_demand": 410},
    {"sku_id": "SKU_006", "category": "Grocery", "base_demand": 350},
    {"sku_id": "SKU_007", "category": "Apparel", "base_demand": 260},
    {"sku_id": "SKU_008", "category": "Apparel", "base_demand": 190},
    {"sku_id": "SKU_009", "category": "Apparel", "base_demand": 140},
    {"sku_id": "SKU_010", "category": "Home", "base_demand": 130},
    {"sku_id": "SKU_011", "category": "Home", "base_demand": 110},
    {"sku_id": "SKU_012", "category": "Home", "base_demand": 90},
    {"sku_id": "SKU_013", "category": "Beauty", "base_demand": 160},
    {"sku_id": "SKU_014", "category": "Beauty", "base_demand": 120},
    {"sku_id": "SKU_015", "category": "Beauty", "base_demand": 75},
]

# peak_weekday/peak_month shape the seasonality curve; trend_pct_per_year is a slow linear drift.
CATEGORY_PROFILE = {
    "Electronics": {"peak_weekday": 5, "peak_month": 12, "trend_pct_per_year": 0.10, "weekly_amp": 0.20, "yearly_amp": 0.30},
    "Grocery":     {"peak_weekday": 6, "peak_month": 11, "trend_pct_per_year": 0.02, "weekly_amp": 0.15, "yearly_amp": 0.20},
    "Apparel":     {"peak_weekday": 5, "peak_month": 6,  "trend_pct_per_year": -0.05, "weekly_amp": 0.25, "yearly_amp": 0.35},
    "Home":        {"peak_weekday": 6, "peak_month": 5,  "trend_pct_per_year": 0.04, "weekly_amp": 0.18, "yearly_amp": 0.25},
    "Beauty":      {"peak_weekday": 4, "peak_month": 2,  "trend_pct_per_year": 0.08, "weekly_amp": 0.22, "yearly_amp": 0.30},
}

# One-off events injected on top of the baseline; these are the anomalies to detect and explain.
EVENTS = [
    {"event_id": "EVT_001", "sku_id": "SKU_001", "date": "2023-02-14", "duration_days": 1, "type": "spike", "magnitude_pct": 65, "cause_category": "promotion", "description": "Valentine's Day flash discount drove a one-day surge in orders."},
    {"event_id": "EVT_002", "sku_id": "SKU_004", "date": "2023-03-10", "duration_days": 4, "type": "dip", "magnitude_pct": 45, "cause_category": "stockout", "description": "Regional distributor delay left warehouses short of stock for several days."},
    {"event_id": "EVT_003", "sku_id": "SKU_007", "date": "2023-04-22", "duration_days": 2, "type": "spike", "magnitude_pct": 90, "cause_category": "viral_social", "description": "A TikTok influencer styling video went viral, driving a two-day demand surge."},
    {"event_id": "EVT_004", "sku_id": "SKU_010", "date": "2023-05-15", "duration_days": 1, "type": "spike", "magnitude_pct": 55, "cause_category": "press_coverage", "description": "Featured as an editor's pick in a home decor magazine roundup."},
    {"event_id": "EVT_005", "sku_id": "SKU_013", "date": "2023-06-01", "duration_days": 3, "type": "dip", "magnitude_pct": 60, "cause_category": "quality_issue", "description": "Voluntary recall issued after a packaging defect was reported by customers."},
    {"event_id": "EVT_006", "sku_id": "SKU_002", "date": "2023-07-19", "duration_days": 2, "type": "spike", "magnitude_pct": 70, "cause_category": "promotion", "description": "Mid-summer clearance event boosted orders for two days."},
    {"event_id": "EVT_007", "sku_id": "SKU_008", "date": "2023-08-28", "duration_days": 3, "type": "dip", "magnitude_pct": 35, "cause_category": "weather", "description": "An unseasonal heatwave delayed the usual start of fall clothing demand."},
    {"event_id": "EVT_008", "sku_id": "SKU_005", "date": "2023-09-11", "duration_days": 2, "type": "spike", "magnitude_pct": 80, "cause_category": "weather", "description": "Hurricane warning triggered stockpiling of shelf-stable groceries."},
    {"event_id": "EVT_009", "sku_id": "SKU_011", "date": "2023-10-05", "duration_days": 5, "type": "dip", "magnitude_pct": 40, "cause_category": "supply_chain", "description": "Shipping container delays at the port pushed back restocking by nearly a week."},
    {"event_id": "EVT_010", "sku_id": "SKU_014", "date": "2023-10-31", "duration_days": 1, "type": "spike", "magnitude_pct": 60, "cause_category": "promotion", "description": "Halloween-themed makeup bundle sold out its promotional run in a single day."},
    {"event_id": "EVT_011", "sku_id": "SKU_001", "date": "2023-11-24", "duration_days": 1, "type": "spike", "magnitude_pct": 120, "cause_category": "promotion", "description": "Black Friday doorbuster pricing drove the year's largest single-day spike."},
    {"event_id": "EVT_012", "sku_id": "SKU_007", "date": "2023-11-24", "duration_days": 1, "type": "spike", "magnitude_pct": 100, "cause_category": "promotion", "description": "Black Friday doorbuster pricing drove a large single-day spike."},
    {"event_id": "EVT_013", "sku_id": "SKU_004", "date": "2023-12-23", "duration_days": 2, "type": "spike", "magnitude_pct": 50, "cause_category": "promotion", "description": "Pre-Christmas grocery rush as shoppers stocked up for holiday gatherings."},
    {"event_id": "EVT_014", "sku_id": "SKU_009", "date": "2024-01-08", "duration_days": 4, "type": "dip", "magnitude_pct": 30, "cause_category": "competitor", "description": "A competitor launched an aggressive price cut on a near-identical product."},
    {"event_id": "EVT_015", "sku_id": "SKU_012", "date": "2024-02-20", "duration_days": 2, "type": "spike", "magnitude_pct": 75, "cause_category": "viral_social", "description": "A viral home-organization trend on social media pulled in new buyers."},
    {"event_id": "EVT_016", "sku_id": "SKU_003", "date": "2024-03-14", "duration_days": 3, "type": "dip", "magnitude_pct": 40, "cause_category": "quality_issue", "description": "A firmware bug caused a wave of negative reviews and a drop in new orders."},
    {"event_id": "EVT_017", "sku_id": "SKU_006", "date": "2024-04-02", "duration_days": 5, "type": "dip", "magnitude_pct": 55, "cause_category": "stockout", "description": "A fire at a regional distribution center halted shipments for nearly a week."},
    {"event_id": "EVT_018", "sku_id": "SKU_015", "date": "2024-05-10", "duration_days": 2, "type": "spike", "magnitude_pct": 85, "cause_category": "press_coverage", "description": "A celebrity endorsement post drove a two-day surge in demand."},
    {"event_id": "EVT_019", "sku_id": "SKU_002", "date": "2024-06-21", "duration_days": 1, "type": "spike", "magnitude_pct": 60, "cause_category": "promotion", "description": "Mid-year clearance event boosted single-day orders."},
    {"event_id": "EVT_020", "sku_id": "SKU_010", "date": "2024-07-04", "duration_days": 1, "type": "spike", "magnitude_pct": 65, "cause_category": "promotion", "description": "July 4th outdoor furniture sale drove a one-day spike."},
    {"event_id": "EVT_021", "sku_id": "SKU_013", "date": "2024-08-15", "duration_days": 4, "type": "dip", "magnitude_pct": 45, "cause_category": "supply_chain", "description": "A key raw-material supplier fell behind on deliveries, constraining production."},
    {"event_id": "EVT_022", "sku_id": "SKU_008", "date": "2024-09-23", "duration_days": 2, "type": "spike", "magnitude_pct": 70, "cause_category": "viral_social", "description": "A celebrity was photographed wearing the item, spiking search and orders."},
    {"event_id": "EVT_023", "sku_id": "SKU_005", "date": "2024-10-30", "duration_days": 2, "type": "spike", "magnitude_pct": 75, "cause_category": "weather", "description": "An early winter storm warning triggered grocery stockpiling."},
    {"event_id": "EVT_024", "sku_id": "SKU_001", "date": "2024-11-29", "duration_days": 1, "type": "spike", "magnitude_pct": 115, "cause_category": "promotion", "description": "Black Friday doorbuster pricing again drove the year's largest single-day spike."},
    {"event_id": "EVT_025", "sku_id": "SKU_007", "date": "2024-11-29", "duration_days": 1, "type": "spike", "magnitude_pct": 95, "cause_category": "promotion", "description": "Black Friday doorbuster pricing drove a large single-day spike."},
    {"event_id": "EVT_026", "sku_id": "SKU_011", "date": "2024-12-15", "duration_days": 3, "type": "dip", "magnitude_pct": 65, "cause_category": "quality_issue", "description": "A safety recall was issued after a small number of units failed inspection."},
    {"event_id": "EVT_027", "sku_id": "SKU_004", "date": "2024-12-24", "duration_days": 1, "type": "spike", "magnitude_pct": 55, "cause_category": "promotion", "description": "Christmas Eve grocery rush drove a single-day spike."},
]


def seasonal_mean(date, base_demand, profile):
    day_of_year_frac = date.dayofyear / 365.25
    peak_frac = (profile["peak_month"] - 1) / 12 + 1 / 24
    yearly = 1 + profile["yearly_amp"] * np.cos(2 * np.pi * (day_of_year_frac - peak_frac))

    dow_diff = min(abs(date.dayofweek - profile["peak_weekday"]), 7 - abs(date.dayofweek - profile["peak_weekday"]))
    weekly = 1 + profile["weekly_amp"] * np.cos(2 * np.pi * dow_diff / 7)

    years_elapsed = (date - pd.Timestamp(START_DATE)).days / 365.25
    trend = 1 + profile["trend_pct_per_year"] * years_elapsed

    return base_demand * trend * weekly * yearly


def build_event_index():
    index = {}
    for event in EVENTS:
        start = pd.Timestamp(event["date"])
        multiplier = (
            1 + event["magnitude_pct"] / 100
            if event["type"] == "spike"
            else max(0.05, 1 - event["magnitude_pct"] / 100)
        )
        for offset in range(event["duration_days"]):
            day = start + pd.Timedelta(days=offset)
            index[(event["sku_id"], day)] = multiplier
    return index


def generate_demand(rng):
    dates = pd.date_range(START_DATE, periods=NUM_DAYS, freq="D")
    event_index = build_event_index()

    rows = []
    for sku in SKUS:
        profile = CATEGORY_PROFILE[sku["category"]]
        for date in dates:
            mean = seasonal_mean(date, sku["base_demand"], profile)
            mean *= event_index.get((sku["sku_id"], date), 1.0)
            demand = rng.poisson(max(mean, 0.1))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "sku_id": sku["sku_id"],
                "category": sku["category"],
                "demand": int(demand),
            })
    return pd.DataFrame(rows)


def write_groundtruth():
    groundtruth = []
    for event in EVENTS:
        start = pd.Timestamp(event["date"])
        affected_dates = [
            (start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(event["duration_days"])
        ]
        groundtruth.append({**event, "affected_dates": affected_dates})

    out_path = DATA_DIR / "anomaly_groundtruth.json"
    out_path.write_text(json.dumps(groundtruth, indent=2))
    return out_path


def main():
    rng = np.random.default_rng(SEED)
    df = generate_demand(rng)

    out_path = DATA_DIR / "synthetic_demand.csv"
    df.to_csv(out_path, index=False)

    groundtruth_path = write_groundtruth()

    print(f"Generated {len(df):,} rows for {len(SKUS)} SKUs over {NUM_DAYS} days -> {out_path}")
    print(f"Wrote {len(EVENTS)} ground-truth anomaly events -> {groundtruth_path}")


if __name__ == "__main__":
    main()
