from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Xet CAS transfer survives the mid-file resets this connection hits. If xet
# itself starts erroring, set HF_HUB_DISABLE_XET=1 in .env to fall back.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# keep downloaded models inside the project instead of the user's home dir
MODELS_DIR = PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR / "u2net"))

from .bgchange import parse_hex

load_dotenv(PROJECT_ROOT / ".env")

# .env holds the bot token - keep it owner-only.
try:
    if (PROJECT_ROOT / ".env").exists():
        os.chmod(PROJECT_ROOT / ".env", 0o600)
except OSError:
    pass

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_BACKGROUND = parse_hex(os.getenv("DEFAULT_BACKGROUND", "#ffffff") or "#ffffff")
MAX_IMAGE_DIM = max(512, int(os.getenv("MAX_IMAGE_DIM", "1600") or "1600"))
MAX_CONCURRENT = max(1, int(os.getenv("MAX_CONCURRENT", "1") or "1"))
# reject oversized uploads before decoding them (decompression-bomb / RAM guard)
MAX_PHOTO_BYTES = max(1, int(os.getenv("MAX_PHOTO_BYTES", str(15 * 1024 * 1024)) or "0"))
# pin model weights to an immutable revision (env empty = tracking the main branch)
DIFFUSE_REVISION = os.getenv("DIFFUSE_REVISION", "cad0bd7495fa6c4bcca01b19a723dc91627fe84f").strip()
CHAT_REVISION = os.getenv("CHAT_REVISION", "12fd25f77366fa6b3b4b768ec3050bf629380bac").strip()
ALLOWED_USERS = {
    int(item)
    for item in os.getenv("ALLOWED_USERS", "").replace(";", ",").split(",")
    if item.strip().isdigit()
}
# one account allowed to run the hidden admin commands (/stats, /last)
_owner_raw = os.getenv("OWNER_USER_ID", "").strip()
OWNER_USER_ID = int(_owner_raw) if _owner_raw.isdigit() else 0
# minimum seconds between heavy requests (diffusion, chat) per user; 0 disables
COOLDOWN_SECONDS = max(0, int(os.getenv("COOLDOWN_SECONDS", "30") or "30"))
DIFFUSE_ENABLED = os.getenv("DIFFUSE_ENABLED", "1").strip() != "0"
DIFFUSE_MODEL = os.getenv("DIFFUSE_MODEL", "segmind/tiny-sd").strip()
DIFFUSE_RESOLUTION = max(128, int(os.getenv("DIFFUSE_RESOLUTION", "224") or "224"))
DIFFUSE_STEPS = max(1, int(os.getenv("DIFFUSE_STEPS", "5") or "5"))
DIFFUSE_STRENGTH = float(os.getenv("DIFFUSE_STRENGTH", "0.6") or "0.6")
DIFFUSE_THREADS = max(1, int(os.getenv("DIFFUSE_THREADS", "2") or "2"))
DIFFUSE_UNLOAD_AFTER = os.getenv("DIFFUSE_UNLOAD_AFTER", "0").strip() != "0"
DIFFUSE_DTYPE = os.getenv("DIFFUSE_DTYPE", "fp32").strip().lower() or "fp32"

CHAT_ENABLED = os.getenv("CHAT_ENABLED", "1").strip() != "0"
CHAT_MODEL = os.getenv("CHAT_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct").strip()
CHAT_MAX_NEW_TOKENS = max(16, int(os.getenv("CHAT_MAX_NEW_TOKENS", "160") or "160"))
CHAT_HISTORY_TURNS = max(0, int(os.getenv("CHAT_HISTORY_TURNS", "4") or "4"))
CHAT_THREADS = max(1, int(os.getenv("CHAT_THREADS", "2") or "2"))

# GPU outfit editor: Qwen-Image-Edit-2509 (4-bit Nunchaku). Needs CUDA + nunchaku;
# falls back to the diffuse/tint engines when unavailable.
EDIT_ENABLED = os.getenv("EDIT_ENABLED", "1").strip() != "0"
EDIT_BASE = os.getenv("EDIT_BASE", "Qwen/Qwen-Image-Edit-2509").strip()
EDIT_REPO = os.getenv("EDIT_REPO", "nunchaku-tech/nunchaku-qwen-image-edit-2509").strip()
EDIT_STEPS = max(1, int(os.getenv("EDIT_STEPS", "4") or "4"))
EDIT_RANK = max(4, int(os.getenv("EDIT_RANK", "32") or "32"))
EDIT_CFG = float(os.getenv("EDIT_CFG", "1.0") or "1.0")
EDIT_GPU_BLOCKS = max(1, int(os.getenv("EDIT_GPU_BLOCKS", "1") or "1"))
# working resolution for the editor (its native res is far too big for 8 GB)
EDIT_MAX_DIM = max(256, int(os.getenv("EDIT_MAX_DIM", "768") or "768"))
EDIT_PROMPT_TEMPLATE = os.getenv(
    "EDIT_PROMPT_TEMPLATE",
    "Replace the person's existing clothing with {prompt}, worn naturally on their body.",
).strip()
# appended to a bare/command prompt so faces and framing stay stable; set "" to disable
EDIT_IDENTITY_CLAUSE = os.getenv(
    "EDIT_IDENTITY_CLAUSE",
    "Keep the person's face, pose and background unchanged.",
).strip()
# prompt wrapper for the multi-image "combine two photos" mode
EDIT_COMBINE_TEMPLATE = os.getenv(
    "EDIT_COMBINE_TEMPLATE",
    "Using the two reference photos, create one natural photo: {prompt}. "
    "Keep both people's faces and identities exactly the same.",
).strip()

# image-to-video animation via diffusers on CUDA.
VIDEO_ENABLED = os.getenv("VIDEO_ENABLED", "1").strip() != "0"
# which backend the "animate" flow uses:
#   wan14b -> Wan 2.2 I2V-A14B (14B) via a local ComfyUI server [best quality]
#   wan    -> Wan 2.2 TI2V-5B Turbo (4-step, uncensored) [fast, in-process]
#   ltx    -> LTX-Video (2B) [in-process]
VIDEO_ENGINE = os.getenv("VIDEO_ENGINE", "wan").strip().lower()
if VIDEO_ENGINE not in ("wan14b", "wan", "ltx"):
    VIDEO_ENGINE = "wan"

# --- settings shared by every engine ---
# video frame rate of the result file
VIDEO_FPS = max(1, int(os.getenv("VIDEO_FPS", "24") or "24"))
# total generated frames; 121 @ 24 fps = 5.04 s (raise VIDEO_FRAMES for longer)
VIDEO_FRAMES = max(9, int(os.getenv("VIDEO_FRAMES", "121") or "121"))
# longest input edge fed to the model (rounded to a multiple of 32)
VIDEO_MAX_DIM = max(256, int(os.getenv("VIDEO_MAX_DIM", "704") or "704"))
VIDEO_NEGATIVE = os.getenv(
    "VIDEO_NEGATIVE",
    "worst quality, inconsistent motion, blurry, jittery, distorted",
).strip()
VIDEO_DEFAULT_PROMPT = os.getenv(
    "VIDEO_DEFAULT_PROMPT",
    "The subject moves gently with subtle natural motion while the camera stays "
    "stable. Keep the subject's face, identity and overall appearance exactly the "
    "same as in the input photo.",
).strip()

# --- LTX-Video backend ---
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "Lightricks/LTX-Video").strip()
# if this folder contains a downloaded LTX-Video (has model_index.json) it is
# used instead of downloading VIDEO_MODEL. Paste path without quotes or spaces.
VIDEO_LOCAL_DIR = str(PROJECT_ROOT / "models" / "ltx-video")
VIDEO_STEPS = max(1, int(os.getenv("VIDEO_STEPS", "40") or "40"))
VIDEO_CFG = float(os.getenv("VIDEO_CFG", "3.0") or "3.0")

# --- Wan 2.2 TI2V-5B Turbo backend (4-step distilled, CFG-free, uncensored) ---
WAN_MODEL = os.getenv("WAN_MODEL", "yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers").strip()
# if this folder contains the downloaded Wan pipeline (has model_index.json) it
# is used instead of downloading WAN_MODEL.
WAN_LOCAL_DIR = str(PROJECT_ROOT / "models" / "wan-ti2v-5b")
WAN_STEPS = max(1, int(os.getenv("WAN_STEPS", "4") or "4"))
# Turbo is CFG-free: guidance 1.0 means no classifier-free guidance
WAN_CFG = float(os.getenv("WAN_CFG", "1.0") or "1.0")
WAN_FLOW_SHIFT = float(os.getenv("WAN_FLOW_SHIFT", "5.0") or "5.0")

# --- Wan 2.2 I2V-A14B (14B) via a local ComfyUI server ---
# The 14B two-expert model is dramatically higher quality than the 5B; it runs
# through ComfyUI (GGUF Q3 + LightX2V 4-step LoRAs) because that is the reliable
# way to fit it on an 8 GB card. The bot only ever talks to it over HTTP.
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").strip().rstrip("/")
COMFY_DIR = os.getenv("COMFY_DIR", r"D:\JEMZ\ComfyUI").strip()
COMFY_PYTHON = os.getenv(
    "COMFY_PYTHON", str(Path(COMFY_DIR) / ".venv" / "Scripts" / "python.exe")
).strip()
# start ComfyUI automatically if it is not already listening
COMFY_AUTOSTART = os.getenv("COMFY_AUTOSTART", "1").strip() != "0"
# how long to wait for one clip (14B on 8 GB can take several minutes)
COMFY_TIMEOUT = max(60, int(os.getenv("COMFY_TIMEOUT", "3600") or "3600"))

WAN14B_HIGH = os.getenv("WAN14B_HIGH", "wan2.2_i2v_high_noise_14B_Q3_K_M.gguf").strip()
WAN14B_LOW = os.getenv("WAN14B_LOW", "wan2.2_i2v_low_noise_14B_Q3_K_M.gguf").strip()
WAN14B_LORA_HIGH = os.getenv(
    "WAN14B_LORA_HIGH", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
).strip()
WAN14B_LORA_LOW = os.getenv(
    "WAN14B_LORA_LOW", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
).strip()
WAN14B_CLIP = os.getenv("WAN14B_CLIP", "umt5-xxl-encoder-Q5_K_M.gguf").strip()
WAN14B_VAE = os.getenv("WAN14B_VAE", "wan_2.1_vae.safetensors").strip()
WAN14B_STEPS = max(1, int(os.getenv("WAN14B_STEPS", "4") or "4"))
WAN14B_CFG = float(os.getenv("WAN14B_CFG", "1.0") or "1.0")
WAN14B_SHIFT = float(os.getenv("WAN14B_SHIFT", "5.0") or "5.0")
# native Wan 14B frame rate; RIFE interpolation raises the delivered fps.
# 12 x INTERP(2) = 24 fps output for the short clips.
WAN14B_FPS = max(1, int(os.getenv("WAN14B_FPS", "12") or "12"))
# longest edge fed to the model (its output is then upscaled by the post pass)
WAN14B_BASE_DIM = max(256, int(os.getenv("WAN14B_BASE_DIM", "832") or "832"))
# Long clips (>= WAN14B_LONG_SECONDS) generate at a lower resolution so the 14B
# model + the larger 161-frame activations still fit on an 8 GB card; short
# clips keep the full WAN14B_BASE_DIM. Set WAN14B_LONG_BASE_DIM=0 to disable.
WAN14B_LONG_SECONDS = max(0, float(os.getenv("WAN14B_LONG_SECONDS", "8") or "8"))
WAN14B_LONG_BASE_DIM = max(0, int(os.getenv("WAN14B_LONG_BASE_DIM", "512") or "512"))
# long clips also ease the native frame rate down (fewer frames -> less sampling,
# interpolation and encoding). Applied as min(WAN14B_FPS, WAN14B_LONG_FPS); set
# WAN14B_LONG_FPS=0 to keep the full WAN14B_FPS for long clips.
WAN14B_LONG_FPS = max(0, int(os.getenv("WAN14B_LONG_FPS", "12") or "12"))
# post passes: swap the generated face back to the source photo, restore it,
# upscale, and interpolate frames (removes the choppy "old AI" look)
WAN14B_FACE_SWAP = os.getenv("WAN14B_FACE_SWAP", "1").strip() != "0"
WAN14B_FACE_RESTORE = os.getenv("WAN14B_FACE_RESTORE", "1").strip() != "0"
WAN14B_UPSCALE = os.getenv("WAN14B_UPSCALE", "1").strip() != "0"
WAN14B_UPSCALE_MODEL = os.getenv("WAN14B_UPSCALE_MODEL", "RealESRGAN_x2.pth").strip()
# skip the 2x upscale once the interpolated clip would exceed this many frames:
# each 1664x1664 frame costs ~30 MB of RAM, so long clips would exhaust memory.
# 0 = never skip.
WAN14B_UPSCALE_MAX_FRAMES = max(
    0, int(os.getenv("WAN14B_UPSCALE_MAX_FRAMES", "200") or "200")
)
# RIFE multiplier (1 = off); 2 turns 16 fps into 32 fps -> much smoother motion
WAN14B_INTERP = max(1, int(os.getenv("WAN14B_INTERP", "2") or "2"))
# ReActor face-swap/restore model names (must exist under COMFY_DIR\models)
WAN14B_SWAP_MODEL = os.getenv("WAN14B_SWAP_MODEL", "inswapper_128.onnx").strip()
WAN14B_RESTORE_MODEL = os.getenv("WAN14B_RESTORE_MODEL", "GFPGANv1.4.pth").strip()
WAN14B_FACE_DETECT = os.getenv("WAN14B_FACE_DETECT", "retinaface_resnet50").strip()
WAN14B_RIFE_MODEL = os.getenv("WAN14B_RIFE_MODEL", "rife49.pth").strip()
# 0 = pick a fresh random seed per job
WAN14B_SEED = int(os.getenv("WAN14B_SEED", "0") or "0")

# normalize the frame count to the active engine's temporal stride:
# LTX needs frames = 8*k + 1, Wan needs frames = 4*k + 1 (121 satisfies both).
_VIDEO_STRIDE = 8 if VIDEO_ENGINE == "ltx" else 4
if (VIDEO_FRAMES - 1) % _VIDEO_STRIDE:
    VIDEO_FRAMES = ((VIDEO_FRAMES - 1) // _VIDEO_STRIDE) * _VIDEO_STRIDE + 1

# clip lengths (seconds) offered in the video flow as inline buttons
VIDEO_DURATIONS = tuple(
    int(part)
    for part in os.getenv("VIDEO_DURATIONS", "5,10,15").replace(";", ",").split(",")
    if part.strip().isdigit() and int(part) > 0
) or (5, 10, 15)

# local SQLite archive for processed media and chat turns
OFFLINE_STORE_ENABLED = os.getenv("OFFLINE_STORE_ENABLED", "1").strip() != "0"
_raw_offline_path = os.getenv("OFFLINE_STORE_PATH", "data/offline.db").strip() or "data/offline.db"
_offline_path = Path(_raw_offline_path)
if not _offline_path.is_absolute():
    _offline_path = Path(__file__).resolve().parent.parent / _offline_path
OFFLINE_STORE_PATH = str(_offline_path)


def validate() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is missing - add your token to .env (copy .env.example to .env)"
        )