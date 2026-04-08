# NLP Analysis Backend (FastAPI)

FastAPI backend that runs a local NLP analysis pipeline for user comments (restaurant reviews initially), returning:

- Granulated items
- Hierarchical themes
- Flat clusters (`k_clusters`)
- 2D map projection (UMAP)
- Cluster labels (OpenAI `gpt-5-nano` by default)
- High-level insights

## Requirements

- Python 3.11+
- Supabase project configured with the analysis schema
- OpenAI API key for AI labels and summaries

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn main:app --reload
```

API base path: `http://127.0.0.1:8000/v1`

## Supabase Persistence

The backend persists new analyses into Supabase as the source of truth.
Local files are only used as temporary workspace during processing and are deleted
after a successful sync.

Setup:

1. Run `supabase/001_analysis_schema.sql` in the Supabase SQL Editor.
2. Create `ProyectoIntegradorBE/.env` from `ProyectoIntegradorBE/.env.example`.
3. Fill:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_STORAGE_BUCKET=analysis-artifacts
OPENAI_API_KEY=your_openai_api_key
OPENAI_TEXT_MODEL=gpt-5-nano
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

4. Restart FastAPI.

What gets persisted:

- `analysis_runs` table: ownership, status, progress, metadata
- `analysis_results` table: `items`, `overview`, `insights`, `clusters`, `umap`, `hierarchy`
- `analysis-artifacts` storage bucket: `embeddings.npy`

Current behavior:

- Supabase is the persistent source of truth
- Local files are temporary and are cleaned after sync or failure
- If an endpoint needs a local artifact later, the backend hydrates it from Supabase on demand
- Labels and executive summaries use OpenAI; if the API is unavailable, deterministic fallbacks still keep the pipeline usable
- Existing legacy local analyses are not migrated automatically

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

During processing, artifacts are written to a temporary workspace directory.
Persistent storage lives in Supabase.

Temporary generated files:

- `items.json`
- `embeddings.npy`
- `umap.json`
- `clusters.json`
- `hierarchy.json`
- `overview.json`
- `insights.json`

## Deployment

Deployment templates are versioned in [`deploy`](deploy):

- Cloudflare Pages frontend setup
- Oracle Cloud Always Free backend setup
- Caddy reverse proxy example
- `systemd` service template
- Render Free fallback notes
