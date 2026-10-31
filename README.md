<div align="center">

# TG_bot

**A 100% local Telegram bot for AI photo editing and image-to-video animation.**

Background removal · AI outfit / scene editing · image-to-video · a local chat model — all running on your own machine, no cloud APIs.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![aiogram](https://img.shields.io/badge/aiogram-3.x-2CA5E0.svg)](https://docs.aiogram.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20%2B%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![Runs locally](https://img.shields.io/badge/inference-local-success.svg)](#how-it-works)
[![Tests](https://img.shields.io/badge/tests-106%20passing-brightgreen.svg)](#testing)

</div>

---

## What it is

Send a photo (or two), get back an edited image or a short MP4. The bot is a thin
[aiogram](https://docs.aiogram.dev/) front-end over local models:

- **Background remover** — [rembg](https://github.com/danielgatis/rembg) / U2Net cutout, composited onto a solid colour, a gradient, or transparency.
- **AI editor** — [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (4-bit [Nunchaku](https://github.com/nunchaku-tech/nunchaku) SVDQuant) for outfit changes, scene composition and prompt-driven restyle.
- **Image-to-video** — Wan 2.2 I2V (**14B** via ComfyUI, or 5B in-process) and LTX-Video.
- **Chat** — SmolLM2-135M-Instruct by default, or **Qwen3.5-9B** through a local [Ollama](https://ollama.com/) server for much better answers.

Everything is stored in a local SQLite database. Nothing is sent to a third-party service.

> **Built for a single 8 GB laptop GPU** (NVIDIA RTX 50-series / Blackwell). It uses
> sequential CPU offload, CUDA-free GGUF experts and a two-expert Wan pipeline to
> fit large models into small VRAM.

---

## Features

| Operation | Input | What happens |
|---|---|---|
| **Background** | 1 photo + colour caption | U2Net cutout → solid / gradient / transparent background |
| **Combine images** | 2 photos + prompt | Qwen edits both people into one coherent photo ([fallback: plain composite](#how-it-works)) |
| **Change cloth** | 1 photo + preset/prompt | Qwen redraws the outfit (14 one-tap presets or a custom, validated prompt) |
| **Restyle outfit** | 1 photo + prompt | Qwen redraws from free text (e.g. `navy suit`) |
| **Animate photo** | 1 photo + motion prompt + length | image → short MP4 |
| **2-photo video** | 2 photos + action prompt + length | combine into one frame, then animate it |
| **Chat** | text | local LLM reply (SmolLM2 or Qwen3.5-9B) |

Plus: a job queue, per-user rate limiting, a live progress bar in chat, an allow-list,
owner-only admin commands and a local archive CLI.

---

## How it works

```
Telegram  ──►  tgbot/bot.py (aiogram, long polling)
                     │
                     ▼
              jobs.JobQueue          one worker (MAX_CONCURRENT), FIFO, per-user cap
                     │
        ┌────────────┼───────────────┬─────────────────┬───────────────┐
        ▼            ▼               ▼                 ▼               ▼
   bgchange.py   edit_model.py   video.py ──► comfy.py         chat.py / diffuse.py
   rembg U2Net   Qwen-Image-Edit  engine dispatch  ComfyUI HTTP   SmolLM2 or Ollama / tiny-SD
        │            │               │                 │               │
        └────────────┴───────────────┴─────────────────┴───────────────┘
                     │
                     ▼
           storage.py ──► localstore.py  (SQLite:  data/offline.db)
                     │
                     ▼
           bot replies with the result  (+ live progress.pbar while running)
```

1. A handler validates the request and submits a `Job` to `jobs.JobQueue`.
2. A worker runs the heavy function in a thread (so polling stays responsive).
3. While it runs, `progress.TRACKER` is updated and a single chat message shows a
   live bar (stage, percentage, elapsed, ETA).
4. The result is archived through `Storage` into SQLite and sent back.

The heavy **editor auto-unloads the chat / diffusion models** first (and vice
versa) so only one large model is ever resident. With the Ollama backend it also
asks Ollama to evict the chat model before an image or video job.

---

## Requirements

| | Background / Chat | AI image editor | Video (`wan`, `ltx`) | Video (`wan14b`) |
|---|---|---|---|---|
| Python 3.12 | ✅ | ✅ | ✅ | ✅ |
| NVIDIA GPU + CUDA | – | ✅ 8 GB+ | ✅ 8 GB+ | ✅ 8 GB+ |
| ComfyUI install | – | – | – | ✅ |
| RAM | ~2 GB | ~16 GB | ~16 GB | server-side |

Without a usable GPU the bot still runs: background, chat and a lightweight tint
fallback for restyle/combine all work on CPU. The built-in SmolLM2 chat model is
CPU-only; the optional Qwen3.5-9B backend uses the GPU (see below).

---

## Quick start

### One-command install (Linux · macOS · Windows)

The installer detects your OS, installs [uv](https://docs.astral.sh/uv/) and a
compatible Python, resolves every dependency (including the CUDA PyTorch build and
the Nunchaku wheel), and creates `.env`.

```bash
# Linux / macOS
./install.sh                 # install dependencies only
./install.sh --with-models   # also pre-download every weight (editor ~27 GB + video;
                             # adds the ComfyUI 14B pack when VIDEO_ENGINE=wan14b)
./install.sh --test          # also run the test suite

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File install.ps1
```

Any OS that already has Python on `PATH` can run the cross-platform installer:

```bash
python install.py [--python 3.12] [--with-models] [--test] [--no-dev]
```

Then edit `.env` and start the bot:

```bash
uv run python -m tgbot.bot
```

### Manual install

```bash
git clone https://github.com/JENSLINJEMZ/TG_bot.git
cd TG_bot
cp .env.example .env          # then set TELEGRAM_BOT_TOKEN (from @BotFather)
uv sync                       # creates .venv and installs all dependencies
uv run python -m tgbot.prefetch   # optional: pre-download all model weights
uv run python -m tgbot.bot
```

> **Disk note:** the CUDA PyTorch build is ~4–5 GB (it bundles cuDNN/cuBLAS).
> Model weights are stored under `models/` inside the project, not in your home
> directory (`HF_HOME` / `U2NET_HOME` are redirected by `config.py`).
>
> **Model sizes:** without `--with-models` every weight downloads lazily on first
> use. Passing `--with-models` fetches them up front: the chat/diffusion models
> and background remover (small), the Qwen editor (~27 GB) and the in-process
> video models. When `VIDEO_ENGINE=wan14b` it also downloads the ComfyUI Wan 14B
> pack into `COMFY_DIR\models` — both Q3_K_M GGUFs, the UMT5-XXL encoder GGUF, the
> Wan 2.1 VAE, both LightX2V 4-step LoRAs, ReActor (`inswapper_128.onnx`,
> `GFPGANv1.4.pth`, the `buffalo_l` pack) and the Real-ESRGAN 2× upscaler.

---

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example) for the full,
commented list). The most useful ones:

### Access & behaviour

| Variable | Default | Meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | – | **required**, from [@BotFather](https://t.me/BotFather) |
| `ALLOWED_USERS` | *(empty = everyone)* | comma-separated Telegram user IDs allowed to use the bot |
| `OWNER_USER_ID` | *(empty)* | one ID allowed to run the hidden `/stats` and `/last` commands |
| `COOLDOWN_SECONDS` | `30` | per-user pause between heavy requests (`0` disables) |
| `DEFAULT_BACKGROUND` | `#ffffff` | background when a photo has no caption |
| `MAX_IMAGE_DIM` | `1600` | longest edge before inference (downscaling saves time) |
| `MAX_CONCURRENT` | `1` | how many jobs run at once |

### AI image editor (Qwen-Image-Edit-2509)

| Variable | Default | Meaning |
|---|---|---|
| `EDIT_ENABLED` | `1` | enable the GPU editor (falls back to tint if unavailable) |
| `EDIT_STEPS` | `4` | Lightning steps (`4` or `8`) |
| `EDIT_RANK` | `32` | Nunchaku rank (`32` smaller · `128` higher quality) |
| `EDIT_CFG` | `1.0` | guidance (`1.0` Lightning · `4.0` full model) |
| `EDIT_GPU_BLOCKS` | `1` | transformer blocks kept resident on the GPU (raise with more VRAM) |
| `EDIT_MAX_DIM` | `768` | working resolution |

### Chat & CPU fallback

The default chat model is tiny and runs on CPU. For much better answers, run
**Qwen3.5-9B (abliterated)** through a local Ollama server:

```bash
ollama pull huihui_ai/qwen3.5-abliterated:9b
```

Then set these in `.env`:

```ini
CHAT_BACKEND=ollama
OLLAMA_MODEL=huihui_ai/qwen3.5-abliterated:9b
OLLAMA_NUM_CTX=4096
```

Qwen3.5-9B abliterated Q4_K_M is ~6.6 GB. It fits an 8 GB GPU with a 4K context,
but VRAM is tight — close other GPU apps first. The bot asks Ollama to evict it
before the image/video models run.

| Variable | Default | Meaning |
|---|---|---|
| `CHAT_ENABLED` | `1` | show the Chat button |
| `CHAT_BACKEND` | `transformers` | `transformers` (built-in SmolLM2) or `ollama` |
| `CHAT_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | transformers model id |
| `CHAT_MAX_NEW_TOKENS` | `160` | reply length cap |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | local Ollama server |
| `OLLAMA_MODEL` | `huihui_ai/qwen3.5-abliterated:9b` | Ollama model tag |
| `OLLAMA_NUM_CTX` | `4096` | context window (lower it if VRAM is tight) |
| `OLLAMA_THINK` | `0` | `1` lets the model think first (slower) |
| `OLLAMA_KEEP_ALIVE` | `5m` | how long Ollama keeps the model in VRAM |
| `OLLAMA_TIMEOUT` | `300` | request timeout in seconds |
| `DIFFUSE_ENABLED` | `1` | CPU `tiny-sd` fallback for Restyle (set `0` when the GPU editor works) |
| `DIFFUSE_MODEL` | `segmind/tiny-sd` | fallback restyle model |

### Storage

| Variable | Default | Meaning |
|---|---|---|
| `OFFLINE_STORE_ENABLED` | `1` | archive every input/result/chat turn |
| `OFFLINE_STORE_PATH` | `data/offline.db` | SQLite file (relative = project folder) |

### Video

See [Video](#video) for the engine-specific settings (`VIDEO_ENGINE`,
`VIDEO_DURATIONS`, `WAN14B_*`, `COMFY_*`, …).

---

## Usage

Typing `/` opens Telegram's native command menu (registered at startup). The
inline **operation menu** lists every mode as a button.

### Commands

| Command | Effect |
|---|---|
| `/start` | welcome + operation menu |
| `/menu` | pick an operation |
| `/help` | help text |
| `/cancel` | stop the current operation / leave a mode |

Owner-only (hidden, never in the menu):

| Command | Effect |
|---|---|
| `/stats` | how many media items and chat turns are stored, plus queue counts |
| `/last` | resend your latest stored result |

### Background (direct)

Just send a photo. Add a caption to pick the background:

| Caption | Result |
|---|---|
| `navy`, `crimson`, … | a named colour |
| `#ff8800` | any hex colour |
| `gradient red blue` | vertical gradient between two colours |
| `transparent` | PNG cutout with alpha |
| *(no caption)* | `DEFAULT_BACKGROUND` from `.env` |

### Modes (from `/menu`)

- **🧩 Combine images** — send photo 1 → photo 2 → a prompt such as
  `hug each other`. Qwen composes one coherent photo. If the editor is
  unavailable it falls back to a plain side-by-side composite.
- **👕 Change cloth** — send a person photo, then tap a preset
  (`shirt`, `t-shirt`, `hoodie`, `jacket`, `suit`, `dress`, `skirt`, `jeans`,
  `shorts`, `bikini`, `swimwear`, `lingerie`, `kimono`, `saree`) or **Custom** to
  type your own. Custom prompts are validated so only clothing/outfit edits are
  accepted — scene or background changes are rejected.
- **✨ Restyle outfit** — send a person photo, then a free-text prompt
  (`navy suit`). The GPU editor redraws the outfit; it falls back to a colour
  tint if the GPU editor is unavailable.
- **🎬 Animate photo** — send a photo → a motion prompt → a clip length button.
- **🖇 2-photo video** — send **two** photos → an action prompt (e.g.
  `the boy throws a ball to the girl`) → a length. The two are combined into one
  frame first, then animated, so both people act out the prompt.
- **💬 Chat** — talk to the local model. The last few turns are kept as context.

Use `/cancel` to leave a mode.

---

## Web chat UI

The same local chat model is also available as a Claude-style web app in
[`web/`](web/). It is a separate [aiohttp](https://docs.aiohttp.org/) process
(no FastAPI/uvicorn needed) and uses its own SQLite database, so it can run next
to the bot.

```bash
uv run python -m web                 # http://127.0.0.1:8765/
uv run python -m web --port 9000     # custom port
uv run python -m web --no-browser    # do not open a tab
```

- Sessions, titles, renames and deletes are persisted in `data/web.db`
  (`WEB_STORE_PATH`), with WAL + a busy timeout.
- Replies stream token-by-token over SSE; the **Stop** button cancels at the
  next token and keeps the partial answer.
- **Vertex Core / Max / Lite** map to real sampling presets (reply length,
  temperature, top-p) on the selected backend model.
- Deep links work: `http://127.0.0.1:8765/?s=<session-id>`. **Share** copies one.
- Search filters the recents list; `Ctrl/Cmd+Shift+O` starts a new chat.
- The UI is the original Vertex shell, with the two fonts vendored under
  `web/static/fonts/` so it looks the same offline.
- Localhost only by default. Set `WEB_TOKEN` to also require
  `X-Web-Token`/`?token=`, and `WEB_HOST`/`WEB_PORT` to change the bind.

> With the default transformers backend the web process loads its own copy of
> SmolLM2 (roughly 1 GB extra RAM). With `CHAT_BACKEND=ollama` the bot and the
> web UI share one Qwen3.5 model in the Ollama server, so only one copy uses VRAM.

---

## Video

Send a photo, describe the motion, then pick a clip length (buttons come from
`VIDEO_DURATIONS`, default `5,10`). The result is a short MP4.

`VIDEO_ENGINE` selects the backend:

| Engine | Model | Where it runs | Notes |
|---|---|---|---|
| `wan14b` | Wan 2.2 I2V-**A14B** | local **ComfyUI** server | best quality: two-expert GGUF + face swap + RIFE + 2× upscale |
| `wan` | Wan 2.2 TI2V-**5B** Turbo | in-process (diffusers) | fast, uncensored, lower fidelity |
| `ltx` | LTX-Video **2B** | in-process (diffusers) | lightweight |

### `wan14b` — Wan 2.2 I2V-A14B via ComfyUI

The 14B two-expert model is too big to load in-process on an 8 GB card, so it runs
inside a separate [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server. The
bot starts the server automatically (`COMFY_AUTOSTART=1`) and talks to it over
HTTP. Each clip is:

```
LoadImage ─► WanImageToVideo ─► KSampler (high, Q3_K_M + LightX2V 4-step LoRA)
                                    └─► KSampler (low)
          └─► ReActor face swap (identity lock) ─► GFPGAN restore
             ─► RIFE VFI (12 → 24 fps) ─► Real-ESRGAN 2× ─► H.264 MP4
```

**ComfyUI-side requirements**

- ComfyUI at `COMFY_DIR` (default `D:\JEMZ\ComfyUI`) with its own `.venv`.
- Custom nodes: `ComfyUI-GGUF`, `ComfyUI-ReActor`, `ComfyUI-Frame-Interpolation`,
  `ComfyUI-VideoHelperSuite`.
- Models under `COMFY_DIR\models`: both Wan 14B Q3_K_M GGUFs, the UMT5-XXL encoder
  GGUF, the Wan 2.1 VAE, both LightX2V 4-step LoRAs, `inswapper_128.onnx`, the
  `buffalo_l` insightface pack, `GFPGANv1.4.pth` and `RealESRGAN_x2.pth`
  (`rife49.pth` lives in the Frame-Interpolation custom node's own `ckpts/`).

Run `uv run python -m tgbot.prefetch` (or install with `--with-models`) to fetch
that whole pack into `COMFY_DIR\models` automatically; it mirrors the ComfyUI-side
`dl_models.py` / `dl_post.py`, so existing downloads are detected and skipped.
`rife49.pth` is fetched by the Frame-Interpolation node itself on first use.

Key settings:

| Variable | Default | Meaning |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI endpoint |
| `COMFY_DIR` | `D:\JEMZ\ComfyUI` | ComfyUI install folder |
| `COMFY_AUTOSTART` | `1` | start the server if it isn't running |
| `COMFY_TIMEOUT` | `3600` | per-clip timeout (seconds) |
| `WAN14B_BASE_DIM` | `832` | longest edge fed to the model |
| `WAN14B_FPS` | `12` | native frame rate (×2 with RIFE → 24 fps) |
| `WAN14B_INTERP` | `2` | RIFE multiplier (`1` = off) |
| `WAN14B_UPSCALE` | `1` | Real-ESRGAN 2× pass |
| `WAN14B_FACE_SWAP` / `WAN14B_FACE_RESTORE` | `1` | ReActor identity lock + GFPGAN restore |
| `WAN14B_LONG_SECONDS` / `WAN14B_LONG_BASE_DIM` | `8` / `512` | memory guard: long clips render smaller |

**Live progress.** The client opens a WebSocket to ComfyUI and folds its per-node
events into one weighted percentage, shown as a single updating chat message
(bar, stage, elapsed, ETA) and echoed to the terminal:
`[wan14b] 41% | sampling (low) | 05:12 | ~07:20 left`.

**Memory guard.** On an 8 GB card the 14B model plus long-clip activations don't
fit, so clips of `WAN14B_LONG_SECONDS` (default 8 s) or longer render at the
smaller `WAN14B_LONG_BASE_DIM` (default 512 px) and never above `WAN14B_LONG_FPS`.
With the defaults a 5 s clip keeps the full 832 px while a 10 s clip runs at
512 px. Set `WAN14B_LONG_SECONDS=0` to disable.

The same live progress bar is shown for **combine**, **change cloth**,
**restyle**, background replacement, **chat** and **2-photo video**.

---

## The GPU image editor

`edit_model.py` runs **Qwen-Image-Edit-2509** as a 4-bit Nunchaku transformer,
with the 7B text encoder and VAE streamed from system RAM (sequential CPU
offload) so it fits an 8 GB card:

- `torch` built with CUDA (`+cu128`) and the `nunchaku` wheel are required —
  both are pinned in `pyproject.toml` and installed by `uv sync`.
- If CUDA/Nunchaku are missing, `is_available()` returns `False` and the bot
  transparently falls back (tint for restyle, side-by-side composite for combine).
- Check availability from the terminal:

```bash
uv run python -c "from tgbot import edit_model, video; print(edit_model.is_available(), video.is_available())"
```

Models are cached under `models/huggingface/` on first use
(`Qwen/Qwen-Image-Edit-2509` for the base components and
`nunchaku-tech/nunchaku-qwen-image-edit-2509` for the quantised transformer).

---

## Storage & admin CLI

Every input, result and chat turn is written to a local SQLite database
(`data/offline.db`) through `storage.Storage`. Inspect or export it with the
built-in CLI:

```bash
uv run python -m tgbot.admin list [--limit N] [--user <id>]
uv run python -m tgbot.admin download --id <id>  [--out ./admin_downloads]
uv run python -m tgbot.admin download --all      [--out ./admin_downloads]
```

---

## Project layout

```
tgbot/
  bot.py          aiogram handlers, long polling, menus, progress messages
  config.py       .env loading + typed settings
  jobs.py         bounded FIFO job queue with per-user limits
  bgchange.py     colour parsing + rembg U2Net cutout + composite
  edit_model.py   Qwen-Image-Edit-2509 GPU editor (combine / cloth / restyle)
  video.py        image-to-video engine dispatch (wan14b / wan / ltx)
  comfy.py        ComfyUI HTTP + WebSocket client and the Wan 14B graph
  diffuse.py      CPU tiny-SD img2img fallback for Restyle
  chat.py         local chat model + Ollama adapter
  progress.py     thread-safe progress snapshot for the live bar
  ratelimit.py    per-user cooldown
  sticker.py      "Processing…" sticker
  storage.py      single write path for media / chat turns
  localstore.py   SQLite archive
  admin.py        CLI to list / export the archive
  prefetch.py     pre-download the model weights
web/
  app.py          aiohttp routes, SSE endpoint, static files
  service.py      single-flight chat streaming bridge + presets
  store.py        SQLite sessions/messages (data/web.db, WAL)
  static/         the Vertex shell (index.html, app.js, fonts)
install.sh        bootstrap installer (Linux / macOS)
install.ps1       bootstrap installer (Windows)
install.py        cross-platform installer (stdlib-only)
tests/            pytest suite
```

---

## Testing

```bash
uv run pytest tests/ -q      # 106 tests
```

The suite covers colour parsing, the outfit guard, scheduling/queueing, storage,
the local store, chat and rate limiting, plus the web session store, SSE service,
aiohttp API and the Vertex shell's fidelity.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Restyle/Combine returns a flat tint or plain composite | The GPU editor is unavailable. Run the availability check above; make sure `torch.cuda.is_available()` is `True` and `nunchaku` imports. |
| `nunchaku` fails with a DLL error | You have a CPU-only PyTorch. It must be the CUDA build (`+cu128`). Re-run `uv sync`. |
| Video does nothing / times out | Ensure `VIDEO_ENGINE` matches what you installed. For `wan14b`, ComfyUI and the custom nodes/models must be present; the first clip also pays the server startup cost. |
| "The bot is busy" | The queue is full (`max_pending`, `max_per_user` in `jobs.py`). Wait a moment. |
| Emoji look broken in a Windows terminal | Console encoding, not a bug — output is UTF-8. |

---

## License

No license file is included yet. Add one (e.g. MIT) before distributing.

---

<div align="center"><sub>Built to run entirely on your own hardware — background removal, AI editing and video generation, offline.</sub></div>
