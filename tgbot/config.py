from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .bgchange import parse_hex

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# .env holds the bot token and the Supabase secret key - keep it owner-only
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
try:
    if _ENV_PATH.exists():
        os.chmod(_ENV_PATH, 0o600)
except OSError:
    pass

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()


def _key_kind(key: str) -> str:
    """Classify a Supabase API key without trusting which env var it came from."""
    if not key:
        return "none"
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("sb_publishable_"):
        return "publishable"
    if key.startswith("eyJ"):
        try:
            import base64
            import json

            payload = key.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            role = json.loads(base64.urlsafe_b64decode(payload)).get("role")
            return "secret" if role == "service_role" else "publishable"
        except Exception:
            return "unknown"
    return "unknown"


# The bot is server-side and must authenticate with the SECRET key, because the
# hardened RLS (supabase/migration.sql) denies the anon/publishable role any
# access. The key under SUPABASE_SECRET_KEY is inspected rather than trusted, so
# pasting a publishable key there is caught instead of silently failing.
SUPABASE_SECRET_KEY = (
    os.getenv("SUPABASE_SECRET_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
).strip()
SUPABASE_API_KEY = SUPABASE_SECRET_KEY or (os.getenv("SUPABASE_API_KEY", "") or "").strip()
SUPABASE_KEY_KIND = _key_kind(SUPABASE_API_KEY)
SUPABASE_USING_SECRET = SUPABASE_KEY_KIND == "secret"
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "tg-bot-media").strip() or "tg-bot-media"
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_API_KEY)
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