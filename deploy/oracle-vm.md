# Oracle Cloud Always Free Backend Deployment

Recommended target:

- One Ubuntu VM on Oracle Cloud Always Free
- Caddy on ports `80/443`
- Uvicorn on `127.0.0.1:8000`
- One `systemd` service for the FastAPI app

Suggested server layout:

```text
/opt/granulate/backend
/opt/granulate/backend/.venv
/opt/granulate/backend/ProyectoIntegradorBE
```

Bootstrap:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip caddy git
sudo mkdir -p /opt/granulate/backend
sudo chown -R $USER:$USER /opt/granulate/backend
cd /opt/granulate/backend
git clone <your-repo-url> .
cd ProyectoIntegradorBE
python3 -m venv /opt/granulate/backend/.venv
source /opt/granulate/backend/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set backend env values in `ProyectoIntegradorBE/.env`:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_STORAGE_BUCKET=analysis-artifacts
OPENAI_API_KEY=your_openai_api_key
OPENAI_TEXT_MODEL=gpt-5-nano
CORS_ALLOWED_ORIGINS=https://your-project.pages.dev
```

Service install:

1. Copy `deploy/oracle/granulate-backend.service` to `/etc/systemd/system/granulate-backend.service`.
2. Adjust `User`, `Group`, `WorkingDirectory`, `EnvironmentFile`, and virtualenv paths.
3. Run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable granulate-backend
sudo systemctl start granulate-backend
sudo systemctl status granulate-backend
```

Caddy setup:

1. Copy `deploy/oracle/Caddyfile` to `/etc/caddy/Caddyfile`.
2. Replace `api.your-domain.example` with your backend domain or Oracle public DNS.
3. Reload:

```bash
sudo systemctl reload caddy
```

Oracle network rules:

- Allow inbound TCP `80`
- Allow inbound TCP `443`

Health check:

```bash
curl https://api.your-domain.example/health
```
