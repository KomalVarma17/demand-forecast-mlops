"""ChromaDB setup and retrieval for causal event documents (agent/../data/event_docs).

Ingests each event's narrative .txt (metadata header + prose body) into a
persistent local collection, embedded with sentence-transformers, so the
LangGraph agent can retrieve "why did SKU_X spike on date Y" via semantic search.
"""
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVENT_DOCS_DIR = DATA_DIR / "event_docs"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = "event_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def parse_event_doc(path: Path) -> dict:
    header, _, body = path.read_text().partition("\n\n")
    metadata = dict(line.split(": ", 1) for line in header.splitlines())
    metadata["duration_days"] = int(metadata["duration_days"])
    return {**metadata, "text": body.strip()}


def ingest():
    collection = get_collection()
    docs = [parse_event_doc(p) for p in sorted(EVENT_DOCS_DIR.glob("*.txt"))]

    collection.upsert(
        ids=[d["event_id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[
            {
                "sku_id": d["sku_id"],
                "date": d["date"],
                "cause_category": d["cause_category"],
                "type": d["type"],
                "duration_days": d["duration_days"],
            }
            for d in docs
        ],
    )
    return len(docs)


def retrieve(query: str, sku_id: str | None = None, n_results: int = 3) -> list[dict]:
    collection = get_collection()
    where = {"sku_id": sku_id} if sku_id else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)

    return [
        {"event_id": event_id, "text": text, "distance": distance, **metadata}
        for event_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def main():
    count = ingest()
    print(f"Ingested {count} event documents into ChromaDB collection '{COLLECTION_NAME}' at {PERSIST_DIR}")

    demo = retrieve("Why did demand spike around Black Friday?")
    print("\nDemo query: 'Why did demand spike around Black Friday?'")
    for hit in demo:
        print(f"  [{hit['event_id']}] {hit['sku_id']} {hit['date']} ({hit['cause_category']}) dist={hit['distance']:.3f}")


if __name__ == "__main__":
    main()
