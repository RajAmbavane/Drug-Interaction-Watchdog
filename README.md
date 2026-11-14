# Drug Interaction Watchdog

Drug Interaction Watchdog is an AI-assisted pharmacovigilance project for detecting and explaining potentially dangerous drug-drug interactions. The system combines structured clinical features, machine learning, graph neural networks, retrieval-augmented generation, and multi-agent orchestration to support safer medication review workflows.

> **Research prototype.** This project is not a medical device and must not be used to make clinical decisions.

## Current Capabilities

- Drug interaction severity prediction with XGBoost (primary) and a GNN (secondary, for unseen pairs).
- Feature engineering for CYP450 signals, molecular properties, and label-derived warnings.
- FastAPI backend with Supabase authentication for patient, analysis, image intake, and alert routes.
- Next.js frontend for sign-up, patient onboarding, drug analysis, and alert review.
- Vision intake that reads drug names and lab values from photos of pill packs and lab reports.
- RAG components for hybrid retrieval (BioBERT + BM25 with reciprocal rank fusion), reranking, query expansion, context assembly, and citation tracking.
- Multi-agent modules for orchestration, ML prediction, patient context, explanation, retrieval, memory, and rule-based alert routing.

## Results

| Model   | Accuracy | Macro F1 | AUC  |
|---------|---------:|---------:|-----:|
| XGBoost | 0.999    | 0.43     | 0.99 |
| GNN     | 0.999    | 0.55     | 0.94 |

The dataset is heavily imbalanced (most drug pairs have no recorded interaction), so accuracy is not a meaningful headline number. Macro F1 is the metric to improve. See [`notebooks/03_model_training_baseline.ipynb`](notebooks/03_model_training_baseline.ipynb) and [`notebooks/04_gnn_experiment.ipynb`](notebooks/04_gnn_experiment.ipynb) for the full breakdown, SHAP feature importance, and next steps.

## Project Structure

```text
agents/      Multi-agent orchestration and specialist agents
api/         FastAPI app, routes, schemas, and database setup
configs/     Model, RAG, and agent configuration
frontend/    Next.js app (auth, onboarding, analysis, alerts)
ingestion/   DrugBank, DailyMed, FAERS, PubMed, RxNorm, and embedding pipelines
ml/          Features, models, training, evaluation, and prediction interface
notebooks/   Model evaluation notebooks
rag/         Retrieval, reranking, context assembly, citations, and expansion
tests/       API, ingestion, ML feature, RAG, and agent tests
```

## Quick Start

Backend:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then fill in Supabase and LLM keys
uvicorn api.main:app --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, and `NEXT_PUBLIC_SUPABASE_ANON_KEY`. When deploying, set `ALLOWED_ORIGINS` on the API to your frontend URL (for example `https://your-app.vercel.app`).

## Tests

The unit tests do not need the ML stack:

```powershell
pip install -r requirements-test.txt
python -m pytest
```

CI runs the tests, a byte-compile check of every Python package, and a frontend build on each push.

## Sample Images

`pills.png` and `labs.png` are sample inputs for trying the vision intake. They contain no real patient data. Use only synthetic or anonymised documents when testing.

Large raw datasets, generated embeddings, processed data, and local secrets are intentionally excluded from Git.

## Team

- **Raj Ambavane** ([@RajAmbavane](https://github.com/RajAmbavane))
- **Oindrilla**
