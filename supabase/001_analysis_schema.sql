create table if not exists public.analysis_runs (
  id text primary key,
  owner_id uuid not null references auth.users(id) on delete cascade,
  status text not null check (status in ('queued', 'processing', 'completed', 'failed')),
  raw_stage text not null default 'queued',
  stage_label text,
  message text,
  progress_pct integer not null default 0 check (progress_pct >= 0 and progress_pct <= 100),
  total_records integer not null default 0,
  total_items integer not null default 0,
  input_type text,
  source_name text,
  llm_model text,
  embedding_method text,
  embeddings_storage_path text,
  artifacts_synced boolean not null default false,
  error_message text,
  created_at timestamptz not null default timezone('utc'::text, now()),
  updated_at timestamptz not null default timezone('utc'::text, now()),
  completed_at timestamptz
);

create index if not exists analysis_runs_owner_created_idx
  on public.analysis_runs (owner_id, created_at desc);

create table if not exists public.analysis_results (
  analysis_id text primary key references public.analysis_runs(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  items_json jsonb,
  overview_json jsonb,
  insights_json jsonb,
  clusters_json jsonb,
  umap_json jsonb,
  hierarchy_json jsonb,
  created_at timestamptz not null default timezone('utc'::text, now()),
  updated_at timestamptz not null default timezone('utc'::text, now())
);

create index if not exists analysis_results_owner_idx
  on public.analysis_results (owner_id);

alter table public.analysis_runs enable row level security;
alter table public.analysis_results enable row level security;

drop policy if exists "Users can view own analysis runs" on public.analysis_runs;
create policy "Users can view own analysis runs"
on public.analysis_runs
for select
to authenticated
using ((select auth.uid()) = owner_id);

drop policy if exists "Users can view own analysis results" on public.analysis_results;
create policy "Users can view own analysis results"
on public.analysis_results
for select
to authenticated
using ((select auth.uid()) = owner_id);

insert into storage.buckets (id, name, public, file_size_limit)
values ('analysis-artifacts', 'analysis-artifacts', false, 52428800)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit;

drop policy if exists "Users can read own analysis artifacts" on storage.objects;
create policy "Users can read own analysis artifacts"
on storage.objects
for select
to authenticated
using (
  bucket_id = 'analysis-artifacts'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "Users can upload own analysis artifacts" on storage.objects;
create policy "Users can upload own analysis artifacts"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'analysis-artifacts'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "Users can update own analysis artifacts" on storage.objects;
create policy "Users can update own analysis artifacts"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'analysis-artifacts'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
)
with check (
  bucket_id = 'analysis-artifacts'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);

drop policy if exists "Users can delete own analysis artifacts" on storage.objects;
create policy "Users can delete own analysis artifacts"
on storage.objects
for delete
to authenticated
using (
  bucket_id = 'analysis-artifacts'
  and (storage.foldername(name))[1] = (select auth.uid()::text)
);
