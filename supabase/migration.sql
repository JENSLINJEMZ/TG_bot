-- Schema + hardened Row Level Security for the Telegram bot.
--
-- SECURITY MODEL
--   The bot is server-side and authenticates with the Supabase SECRET key
--   (sb_secret_... or the legacy service_role JWT) via SUPABASE_SECRET_KEY.
--   RLS is therefore enabled with NO policies for anon/authenticated, and the
--   table grants handed out by the first version of this file are revoked: a
--   leaked publishable/anon key can read and write nothing.
--
--   In the bot's .env set:
--       SUPABASE_SECRET_KEY=sb_secret_...        # NOT the publishable key
--
-- HOW TO APPLY
--   Paste this whole file into the Supabase SQL editor and run it. It is
--   idempotent, so re-running it also locks down an install created by an
--   earlier (wide-open) version.

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

-- Deny-by-default: RLS on, and the secret key bypasses it for the bot.
alter table public.tg_media enable row level security;
alter table public.tg_chats enable row level security;

-- Remove the wide-open policies created by the first version of this file.
drop policy if exists "full access tg_media" on public.tg_media;
drop policy if exists "full access tg_chats" on public.tg_chats;

-- Take back the CRUD grants that version handed to the public roles.
revoke all on public.tg_media from anon, authenticated;
revoke all on public.tg_chats from anon, authenticated;

-- The bot authenticates with the secret key, which PostgREST maps to the
-- service_role. That role bypasses RLS but still needs table privileges (the
-- first migration only ever granted to anon/authenticated).
grant usage on schema public to service_role;
grant select, insert, update, delete on public.tg_media to service_role;
grant select, insert, update, delete on public.tg_chats to service_role;

-- Storage: keep the bucket private and expose no anon policy. The bot's secret
-- key bypasses storage RLS, so uploads/downloads keep working.
insert into storage.buckets (id, name, public)
values ('tg-bot-media', 'tg-bot-media', false)
on conflict (id) do update set public = false;

drop policy if exists "full access tg-bot-media objects" on storage.objects;
drop policy if exists "anon read tg-bot-media bucket" on storage.buckets;
