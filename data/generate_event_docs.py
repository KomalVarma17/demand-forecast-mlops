"""Turn anomaly_groundtruth.json into narrative .txt documents for ChromaDB ingestion."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
GROUNDTRUTH_PATH = DATA_DIR / "anomaly_groundtruth.json"
OUT_DIR = DATA_DIR / "event_docs"

# One narrative template per cause category, filled in per event.
TEMPLATES = {
    "promotion": (
        "On {date}, {sku_id} saw {article} {noun} of roughly {magnitude_pct}% in daily "
        "demand tied to a promotional event. {description} Promotions like this are "
        "planned by the marketing team and typically produce a short, sharp change in "
        "orders that fades once the offer ends."
    ),
    "stockout": (
        "On {date}, {sku_id} experienced {article} {noun} of about {magnitude_pct}% in "
        "recorded demand due to a stockout. {description} Because units were not "
        "available to sell, the drop reflects fulfillment capacity rather than a change "
        "in underlying customer interest."
    ),
    "viral_social": (
        "On {date}, {sku_id} saw {article} {noun} of around {magnitude_pct}% in demand "
        "following a burst of social media attention. {description} This kind of organic "
        "virality is difficult to predict in advance and can fade as quickly as it appears."
    ),
    "press_coverage": (
        "On {date}, {sku_id} recorded {article} {noun} of about {magnitude_pct}% in demand "
        "after receiving media or press coverage. {description} Coverage-driven demand "
        "usually spikes around the publication date and normalizes within a few days."
    ),
    "quality_issue": (
        "On {date}, {sku_id} saw {article} {noun} of roughly {magnitude_pct}% in demand "
        "following a product quality issue. {description} Quality-related dips can "
        "persist longer than other anomalies if customer trust takes time to recover."
    ),
    "weather": (
        "On {date}, {sku_id} experienced {article} {noun} of about {magnitude_pct}% in "
        "demand linked to weather conditions. {description} Weather-driven changes are "
        "typically regional and correlate with forecasts issued in the days prior."
    ),
    "supply_chain": (
        "On {date}, {sku_id} saw {article} {noun} of roughly {magnitude_pct}% in demand "
        "caused by a supply chain disruption. {description} These events reflect "
        "upstream constraints rather than a shift in customer demand itself."
    ),
    "competitor": (
        "On {date}, {sku_id} recorded {article} {noun} of about {magnitude_pct}% in demand "
        "tied to competitor activity. {description} Competitive pressure can cause "
        "sustained rather than one-day shifts if the rival offer remains in place."
    ),
}


def render_doc(event: dict) -> str:
    noun = "increase" if event["type"] == "spike" else "decrease"
    article = "an" if event["type"] == "spike" else "a"
    body = TEMPLATES[event["cause_category"]].format(
        date=event["date"],
        sku_id=event["sku_id"],
        noun=noun,
        article=article,
        magnitude_pct=event["magnitude_pct"],
        description=event["description"],
    )

    header = "\n".join(
        f"{key}: {event[key]}"
        for key in ("event_id", "sku_id", "date", "cause_category", "type")
    )
    return f"{header}\n\n{body}\n"


def main():
    events = json.loads(GROUNDTRUTH_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True)

    for event in events:
        doc_text = render_doc(event)
        out_path = OUT_DIR / f"{event['event_id']}.txt"
        out_path.write_text(doc_text)

    print(f"Wrote {len(events)} event documents -> {OUT_DIR}")


if __name__ == "__main__":
    main()
