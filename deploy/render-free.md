# Render Free Alternative

Use this only for demos or temporary previews.

Why it is secondary:

- Render Free web services spin down after idle time.
- Cold starts can take around a minute.
- This backend uses in-process jobs, so a sleeping service is a poor fit for long-running analysis work.

If you still want the simpler setup:

- Root directory: `ProyectoIntegradorBE`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Backend environment variables:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_STORAGE_BUCKET=analysis-artifacts
OPENAI_API_KEY=your_openai_api_key
OPENAI_TEXT_MODEL=gpt-5-nano
CORS_ALLOWED_ORIGINS=https://your-project.pages.dev
```
