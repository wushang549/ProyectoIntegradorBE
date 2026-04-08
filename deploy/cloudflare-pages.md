# Cloudflare Pages Deployment

Project root:

- `ProyectoIntegradorUI/my-react-app`

Cloudflare Pages settings:

- Framework preset: `Vite`
- Build command: `npm run build`
- Build output directory: `dist`
- Production branch: `main`

Environment variables:

```env
VITE_API_BASE_URL=https://api.your-domain.example
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
```

Notes:

- The SPA fallback is versioned in `public/_redirects`.
- `VITE_API_BASE_URL` must point to the deployed backend. Do not use `localhost` in Pages.
- Only public Supabase keys belong in Cloudflare Pages env vars.
- Update Supabase Auth `Site URL` to your final Pages URL.
- Add both `https://<your-project>.pages.dev` and `https://<your-project>.pages.dev/reset-password` to Supabase Auth `Redirect URLs`.

Post-deploy checks:

- Open `/login`, `/signup`, `/reset-password`, and `/chat` directly.
- Confirm there are no CORS failures when the frontend calls `/v1/...` on the backend domain.
