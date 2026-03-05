# NLP Analysis Backend (FastAPI)

FastAPI backend that runs a local NLP analysis pipeline for user comments (restaurant reviews initially), returning:

- Granulated items
- Hierarchical themes
- Flat clusters (`k_clusters`)
- 2D map projection (UMAP)
- Cluster labels (Ollama local)
- High-level insights

## Requirements

- Python 3.11+
- Local Ollama server at `http://localhost:11434`
- Model available in Ollama: `gemma3:1b`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn main:app --reload
```

API base path: `http://127.0.0.1:8000/v1`

## Main Endpoints

- `POST /v1/analysis`
- `GET /v1/analysis/recent?limit=N`
- `GET /v1/analysis/{analysis_id}/status`
- `GET /v1/analysis/{analysis_id}/overview`
- `GET /v1/analysis/{analysis_id}/map?k_clusters=N`
- `GET /v1/analysis/{analysis_id}/clusters?k_clusters=N`
- `GET /v1/analysis/{analysis_id}/granulate`
- `GET /v1/analysis/{analysis_id}/hierarchy`
- `GET /v1/analysis/{analysis_id}/insights`

If artifacts are still processing, data endpoints return `409`.

## Example Requests

Text input:

```bash
curl -X POST "http://127.0.0.1:8000/v1/analysis" \
  -F "input_type=text" \
  -F "text=Food was tasty but service was slow. Atmosphere was great." \
  -F "options={\"k_clusters\":6,\"granulate\":true}"
```

CSV input:

```bash
curl -X POST "http://127.0.0.1:8000/v1/analysis" \
  -F "input_type=csv" \
  -F "file=@reviews.csv" \
  -F "options={\"k_clusters\":8,\"granulate\":true}"
```

Status polling:

```bash
curl "http://127.0.0.1:8000/v1/analysis/<analysis_id>/status"
```

## Data Artifacts

Each run is stored in:

```text
data/<analysis_id>/
```

Generated files:

- `items.json`
- `embeddings.npy`
- `umap.json`
- `clusters.json`
- `hierarchy.json`
- `overview.json`
- `insights.json`

Recent runs index:

- `data/index.json`
