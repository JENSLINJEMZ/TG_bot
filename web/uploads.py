"""Attachment helpers shared by the HTTP layer and chat orchestration."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".log",
    ".py",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".htm",
    ".css",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".xml",
    ".rst",
}


def guess_kind(filename: str, mime: str) -> str | None:
    """Return ``image``/``text`` for a supported attachment, else ``None``."""
    ext = Path(filename or "").suffix.lower()
    mime = (mime or "").split(";", 1)[0].strip().lower()
    if mime in IMAGE_MIMES or ext in IMAGE_EXTS:
        return "image"
    if mime.startswith("text/") or mime in {
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
    } or ext in TEXT_EXTS:
        return "text"
    return None


def mime_for(filename: str, supplied: str = "") -> str:
    mime = (supplied or "").split(";", 1)[0].strip().lower()
    if mime and mime != "application/octet-stream":
        return mime
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or mime or "application/octet-stream"


def public_upload(row: dict[str, Any]) -> dict[str, Any]:
    """The subset of an upload row the browser is allowed to see."""
    return {
        "id": row["id"],
        "name": row.get("filename") or "attachment",
        "mime": row.get("mime") or "application/octet-stream",
        "size": int(row.get("size") or 0),
        "kind": row.get("kind") or "text",
        "width": row.get("width"),
        "height": row.get("height"),
        "url": f"/api/uploads/{row['id']}/content",
    }


def text_excerpt(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[truncated]"
