"""Supabase client over HTTPS only (Postgres ports are firewalled on this host)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx


class SupabasePermissionError(RuntimeError):
    pass


@dataclass
class SupabaseConfig:
    url: str
    api_key: str
    bucket: str = "tg-bot-media"


class Supabase:
    def __init__(self, config: SupabaseConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.base = config.url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._headers = {
            "apikey": config.api_key,
            "authorization": f"Bearer {config.api_key}",
        }

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = {**self._headers, **kwargs.pop("headers", {})}
        response = await self._client.request(method, f"{self.base}{path}", headers=headers, **kwargs)
        if response.status_code == 403 and "permission denied" in response.text:
            raise SupabasePermissionError(
                "Supabase denied access to schema public - run supabase/migration.sql first"
            )
        return response

    async def ensure_bucket(self) -> bool:
        try:
            probe = await self._request("GET", f"/storage/v1/bucket/{self.config.bucket}")
            return probe.status_code < 300
        except httpx.HTTPError:
            return False

    async def upload(self, path: str, data: bytes, content_type: str) -> str:
        response = await self._request(
            "POST",
            f"/storage/v1/object/{self.config.bucket}/{path}",
            content=data,
            headers={"content-type": content_type, "x-upsert": "true"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"storage upload failed ({response.status_code}): {response.text[:300]}")
        return path

    async def download(self, path: str) -> bytes:
        response = await self._request("GET", f"/storage/v1/object/{self.config.bucket}/{path}")
        response.raise_for_status()
        return response.content

    async def insert(self, row: dict) -> dict:
        response = await self._request(
            "POST",
            "/rest/v1/tg_media",
            json=row,
            headers={"prefer": "return=representation"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"insert failed ({response.status_code}): {response.text[:300]}")
        return response.json()[0]

    async def insert_chat(self, row: dict) -> dict:
        response = await self._request(
            "POST",
            "/rest/v1/tg_chats",
            json=row,
            headers={"prefer": "return=representation"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"chat insert failed ({response.status_code}): {response.text[:300]}")
        return response.json()[0]

    async def latest(self, user_id: int) -> dict | None:
        response = await self._request(
            "GET",
            "/rest/v1/tg_media",
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": "1",
                "select": "*",
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"select failed ({response.status_code}): {response.text[:300]}")
        rows = response.json()
        return rows[0] if rows else None

    async def count(self) -> int:
        response = await self._request(
            "GET",
            "/rest/v1/tg_media",
            params={"select": "id"},
            headers={"prefer": "count=exact", "range": "0-0"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"count failed ({response.status_code}): {response.text[:300]}")
        content_range = response.headers.get("content-range", "")
        try:
            return int(content_range.split("/")[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def object_path(kind: str, chat_id: int, user_id: int, ext: str) -> str:
        return f"{kind}/{chat_id}/{user_id}/{uuid.uuid4().hex}.{ext}"
