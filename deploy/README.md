# Deployment Guide

Recommended architecture for this repository:

- Frontend: Cloudflare Pages
- Backend: Oracle Cloud Always Free VM
- Persistence/Auth: Supabase
- LLM calls: OpenAI from backend only

Deployment order:

1. Configure Supabase with `supabase/001_analysis_schema.sql`.
2. Provision the backend VM and deploy `ProyectoIntegradorBE`.
3. Point `VITE_API_BASE_URL` at the backend public URL.
4. Deploy `ProyectoIntegradorUI/my-react-app` to Cloudflare Pages.
5. Update Supabase Auth `Site URL` and `Redirect URLs` with the final frontend URL.

Files in this folder:

- `cloudflare-pages.md`: frontend setup
- `oracle-vm.md`: backend setup on Oracle Cloud
- `render-free.md`: simpler alternative that can sleep
- `oracle/Caddyfile`: reverse proxy example with HTTPS
- `oracle/granulate-backend.service`: sample `systemd` unit
