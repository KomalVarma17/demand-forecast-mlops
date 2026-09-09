# Intelligent Demand Forecasting with Agentic LLM Explanation and Observability

An end-to-end MLOps + LLMOps portfolio project covering five skill targets in one coherent system.

## What it does

Forecasts demand for 15 SKUs daily using Prophet, automatically detects anomalies, explains them in plain English using a LangGraph multi-step agent, evaluates explanation quality with RAGAS, monitors pipeline health through structured log parsing, and fires real-time Slack alerts via webhooks — all tracked end-to-end in MLflow.

## Skills covered

| Skill | Where it's used |
|-------|----------------|
| MLflow | Experiment tracking, model registry, metric logging |
| LangGraph | Multi-step agentic explanation workflow |
| RAGAS / LLM evaluation | Scoring agent explanation quality |
| Log parsing and observability | Structured pipeline health monitoring |
| Webhook design | Event-driven Slack alerts on anomalies |

## Project structure

```
demand-forecast-mlops/
├── data/
│   ├── generate_synthetic_data.py    # Generate time series demand data
│   ├── generate_event_docs.py        # Generate causal event documents for ChromaDB
│   ├── anomaly_groundtruth.json      # Ground truth for evaluation
│   └── ragas_golden_dataset.json     # Golden Q&A pairs for RAGAS
├── forecasting/
│   ├── train.py                      # Prophet training + MLflow logging
│   ├── predict.py                    # Generate 14-day rolling forecasts
│   └── anomaly_detector.py           # Flag anomalies via z-score on residuals
├── agent/
│   ├── graph.py                      # LangGraph StateGraph definition
│   ├── nodes.py                      # Individual node functions
│   └── vector_store.py               # ChromaDB setup and retrieval
├── evaluation/
│   └── evaluate.py                   # RAGAS evaluation runner
├── observability/
│   └── log_parser.py                 # Pipeline log analysis and health reporting
├── api/
│   ├── main.py                       # FastAPI app
│   └── webhook.py                    # Slack webhook handler
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack

- **Forecasting**: Prophet, scikit-learn, Pandas
- **MLOps**: MLflow (tracking, registry, artifacts)
- **Agentic AI**: LangGraph, LangChain, Ollama (Mistral 7B — CPU-friendly)
- **Vector store**: ChromaDB + sentence-transformers (local, no GPU)
- **LLM evaluation**: RAGAS
- **API**: FastAPI + Uvicorn
- **Alerting**: Slack webhooks via httpx
- **Deployment**: Docker → GCP Cloud Run

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/KomalVarma17/demand-forecast-mlops.git
cd demand-forecast-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama (local LLM — no GPU needed)
# Download from https://ollama.ai
ollama pull mistral

# 5. Copy and fill environment variables
cp .env.example .env

# 6. Start MLflow UI
mlflow ui --port 5000
# → open http://localhost:5000

# 7. Generate synthetic data
python data/generate_synthetic_data.py
python data/generate_event_docs.py

# 8. Train first model
python forecasting/train.py
# → check MLflow UI for logged run
```

## Git commit convention

```
feat: <module> — <what was added>
fix: <module> — <what was fixed>
refactor: <module> — <what was changed>
docs: <what was updated>

Examples:
feat: data — synthetic demand generator with seasonal patterns
feat: forecasting — Prophet training with MLflow tracking
feat: agent — LangGraph 5-node explanation workflow
feat: evaluation — RAGAS golden dataset and scoring pipeline
feat: api — FastAPI endpoints and Slack webhook
```

## Build phases

| Week | Focus | Commit target |
|------|-------|---------------|
| 1 | Data generation + MLflow | `feat: data` + `feat: forecasting` |
| 2 | Anomaly detection + LangGraph agent | `feat: forecasting/anomaly` + `feat: agent` |
| 3 | RAGAS evaluation + log observability | `feat: evaluation` + `feat: observability` |
| 4 | FastAPI + webhooks + full integration | `feat: api` |
| 5 | Docker + GCP deployment + README polish | `feat: docker` + `feat: deployment` |