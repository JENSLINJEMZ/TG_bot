create table if not exists public.tg_media (
  id uuid primary key default gen_random_uuid(),
  chat_id bigint not null,
  user_id bigint not null,
  message_id bigint,
  prompt text,
  "background_rgb" integer[],
  "original_path" text,
  "result_path" text,
  "aux_path" text,
  "mode" text,
  "source_file_id" text,
  elapsed_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists tg_media_user_created_idx
  on public.tg_media (user_id, created_at desc);

create table if not exists public.tg_chats (
  id uuid primary key default gen_random_uuid(),
  chat_id bigint not null,
  user_id bigint not null,
  message_id bigint,
  prompt text not null,
  reply text,
  elapsed_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists tg_chats_user_created_idx
  on public.tg_chats (user_id, created_at desc);

alter table public.tg_chats enable row level security;

drop policy if exists "full access tg_chats" on public.tg_chats;

create policy "full access tg_chats"
  on public.tg_chats
  for all
  to anon, authenticated
  using (true)
  with check (true);

alter table public.tg_media enable row level security;

drop policy if exists "full access tg_media" on public.tg_media;

create policy "full access tg_media"
  on public.tg_media
  for all
  to anon, authenticated
  using (true)
  with check (true);

grant usage on schema public to anon, authenticated;

grant select, insert, update, delete on public.tg_media to anon, authenticated;

grant select, insert, update, delete on public.tg_chats to anon, authenticated;

insert into storage.buckets (id, name, public)
values ('tg-bot-media', 'tg-bot-media', false)
on conflict (id) do nothing;

drop policy if exists "full access tg-bot-media objects" on storage.objects;

create policy "full access tg-bot-media objects"
  on storage.objects
  for all
  to anon, authenticated
  using (bucket_id = 'tg-bot-media')
  with check (bucket_id = 'tg-bot-media');

grant select, insert, update, delete on storage.objects to anon, authenticated;

drop policy if exists "anon read tg-bot-media bucket" on storage.buckets;

create policy "anon read tg-bot-media bucket"
  on storage.buckets
  for select
  to anon, authenticated
  using (id = 'tg-bot-media');

grant select on storage.buckets to anon, authenticated;