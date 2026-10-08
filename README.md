# Telegram background changer

Send a photo, get it back with a new background. The cutout runs locally with
rembg (U2Net) — no external AI API. Every input, prompt and result is stored in
Supabase (REST + Storage over HTTPS, because Postgres ports are firewalled on
this machine).

## Setup

### One-command install (Linux, macOS, Windows)

An installer detects your OS, installs [uv](https://docs.astral.sh/uv/), installs
a compatible Python, and installs every dependency (CPU-only PyTorch, so no CUDA
downloads):

```bash
# Linux / macOS
./install.sh                 # add --with-models to pre-download the ~1.5 GB of models
./install.sh --test          # also run the test suite

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File install.ps1
```

Any OS with Python already on PATH can run the installer directly:

```bash
python install.py [--python 3.12] [--with-models] [--test] [--no-dev]
```

It creates `.env` from `.env.example` if one doesn't exist. Then edit `.env` and
start the bot:

```bash
uv run python -m tgbot.bot
```

### Manual install

```bash
cp .env.example .env      # then fill in TELEGRAM_BOT_TOKEN (from @BotFather)
uv sync                   # creates .venv, installs all dependencies
```

Models download automatically on first use; to pre-download them run
`uv run python -m tgbot.prefetch`.

Supabase is **optional**. Out of the box the bot cuts out backgrounds and
replies — storage is skipped. To enable storage, fill in the **Supabase URL and
API key you want to use for this bot** in `.env` (the bot is NOT tied to
your other projects' Supabase).

### Enable storage (optional, one time)

With your Supabase URL + key set in `.env`:

1. Open **Supabase → SQL Editor → New query**, paste the contents of
   `supabase/migration.sql`, press Run. It creates `public.tg_media` (photos) and
   `public.tg_chats` (chat turns) with row level security that lets the bot's key
   read/write them, and creates the `tg-bot-media` storage bucket plus an upload
   policy for it.
2. Restart the bot.

The script is idempotent (`create table if not exists`), so re-run it whenever
`migration.sql` gains new tables or columns — an error-free "Done" is expected.

### Offline fallback: nothing is lost when Supabase is down

Every photo input, result image and chat turn is written through one path that
tries Supabase first and, if the write fails (unreachable, unconfigured, or no
table yet), keeps the data in a local SQLite outbox at `data/offline.db` — raw
image bytes included. Chat replies are no longer dropped when storage is down.

Upload the backlog once Supabase is reachable:

```bash
uv run python -m tgbot.sync --dry-run   # list what is waiting
uv run python -m tgbot.sync             # upload and clear the queue
```

or send `/sync` in the bot. The bot also attempts the flush once at startup, so
a restart after an outage drains the queue automatically. Rows keep their
original timestamps, so a late sync still lands in chronological order.

| Variable | Default | Meaning |
|---|---|---|
| `OFFLINE_STORE_ENABLED` | `1` | `0` disables the local outbox |
| `OFFLINE_STORE_PATH` | `data/offline.db` | outbox location (relative = project folder) |

## Usage

| Message | Effect |
|---|---|
| `/menu` | pick an operation (combine, cloth, restyle, chat) |
| photo + caption `navy` / `#ff8800` | solid background (direct) |
| photo + caption `gradient red blue` | vertical gradient between two colors |
| photo + caption `transparent` | cutout, PNG with alpha |
| photo, no caption | `DEFAULT_BACKGROUND` from `.env` |
| `/last` | resend your latest stored result |
| `/stats` | number of stored photos |
| `/sync` | upload locally queued photos/chats to Supabase |
| `/cancel` | stop the current operation |
| `/help` | help |

Modes from the menu:
- **Combine images** — send image 1 (subject), then image 2 (background); the subject's cutout pastes onto it.
- **Change cloth** — tint a person's garment to a caption color (local, best-effort region).
- **Restyle outfit** — send a person photo, then a text prompt like "navy suit";
  a local tiny Stable Diffusion (`segmind/tiny-sd`, fp32 CPU img2img) redraws the
  photo with the prompt while keeping the pose; if the model isn't available it
  falls back to a local tint of the prompt's color. A "Processing..." sticker
  shows while it works.
- **Chat** — talk to a small local model (`HuggingFaceTB/SmolLM2-135M-Instruct`).
  Send text, get a short reply; the last few turns are kept as context.
  `/cancel` leaves chat mode.

Sending a photo with a color caption ("navy", "gradient red blue", "transparent") still runs the direct background/remover flow.

Restrict access by listing your user ids in `ALLOWED_USERS` (comma separated).

Large photos are downscaled to `MAX_IMAGE_DIM` before inference, so it stays fast
on any phone screenshot. Processing is serialized with `MAX_CONCURRENT` so the
CPU stays responsive, an "uploading photo" indicator shows while you wait, and
you get told if a caption color isn't recognized.

First run downloads `u2net.onnx` (~176 MB) into `~/.u2net/`; later runs are fast.
Inference is offloaded to a thread, so polling keeps running while a photo is processed.

Restyle additionally downloads `segmind/tiny-sd` (~1.1 GB) into the HuggingFace
cache on first use. It runs fp32 only — this CPU emulates bf16/fp16 (~200x
slower) — and loads the text encoder, VAE and UNet one phase at a time so peak
RSS stays near 2 GB. Tune with `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `DIFFUSE_ENABLED` | `1` | `0` disables diffusion (tint only) |
| `DIFFUSE_MODEL` | `segmind/tiny-sd` | local model id |
| `DIFFUSE_RESOLUTION` | `256` | working resolution (upscaled back after) |
| `DIFFUSE_STEPS` | `8` | scheduler steps; `DIFFUSE_STRENGTH` slices how many run |
| `DIFFUSE_STRENGTH` | `0.6` | how far the redraw may drift from the photo |
| `DIFFUSE_UNLOAD_AFTER` | `0` | `1` frees the UNet after each run (saves RAM, +~40s/request) |
| `DIFFUSE_THREADS` | `2` | CPU threads for the model |

Chat downloads `HuggingFaceTB/SmolLM2-135M-Instruct` (~270 MB) on first use and
runs fp32 (~1.2 GB RSS). The chat model and the diffusion UNet never sit in RAM
together — each unloads the other first, so image edits and chat stay inside the
laptop's memory. Chat settings:

| Variable | Default | Meaning |
|---|---|---|
| `CHAT_ENABLED` | `1` | `0` hides the Chat button |
| `CHAT_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | any causal-LM chat model |
| `CHAT_MAX_NEW_TOKENS` | `160` | reply length cap (higher = slower) |
| `CHAT_HISTORY_TURNS` | `4` | conversation turns kept as context (0 = none) |
| `CHAT_THREADS` | `2` | CPU threads for the model |

## Admin

```bash
uv run python -m tgbot.admin list [--limit N] [--user <id>]
uv run python -m tgbot.admin download --id <uuid> [--out ./admin_downloads]
uv run python -m tgbot.admin download --all [--out ./admin_downloads]
```

## Run

```bash
uv run python -m tgbot.bot
```

## Tests

```bash
uv run pytest tests/ -q
```

## Layout

```
tgbot/bgchange.py    color parsing + rembg cutout + composite
tgbot/diffuse.py     local tiny-SD img2img for prompt restyle (3-phase, CPU)
tgbot/chat.py        small local chat model (SmolLM2-135M-Instruct)
tgbot/prefetch.py    pre-download the local models
tgbot/sticker.py     "Processing..." sticker
tgbot/supabase.py    HTTPS-only REST/Storage client
tgbot/storage.py     one write path: Supabase first, local outbox on failure
tgbot/localstore.py  SQLite outbox holding data until Supabase is reachable
tgbot/sync.py        flush the outbox (uv run python -m tgbot.sync)
tgbot/config.py      env loading
tgbot/bot.py         aiogram handlers, long polling
install.sh           bootstrap installer (Linux / macOS)
install.ps1          bootstrap installer (Windows)
install.py           cross-platform installer
supabase/migration.sql  paste into Supabase SQL Editor
```
