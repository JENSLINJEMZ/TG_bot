from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .bgchange import parse_hex

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_API_KEY = (os.getenv("SUPABASE_API_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "tg-bot-media").strip() or "tg-bot-media"
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_API_KEY)
DEFAULT_BACKGROUND = parse_hex(os.getenv("DEFAULT_BACKGROUND", "#ffffff") or "#ffffff")
MAX_IMAGE_DIM = max(512, int(os.getenv("MAX_IMAGE_DIM", "1600") or "1600"))
MAX_CONCURRENT = max(1, int(os.getenv("MAX_CONCURRENT", "1") or "1"))
ALLOWED_USERS = {
    int(item)
    for item in os.getenv("ALLOWED_USERS", "").replace(";", ",").split(",")
    if item.strip().isdigit()
}
# one account allowed to run the hidden admin commands (/stats, /last, /sync)
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

# local outbox used when Supabase is unreachable (or not configured)
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