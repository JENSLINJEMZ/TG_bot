"""Admin CLI: list stored photos and download originals/results.

Usage:
  uv run python -m tgbot.admin list [--limit 10] [--user 123]
  uv run python -m tgbot.admin download --id <uuid> [--out ./admin_downloads]
  uv run python -m tgbot.admin download --all [--out ./admin_downloads]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from . import config
from .supabase import Supabase, SupabaseConfig


def _db() -> Supabase:
    if not config.SUPABASE_ENABLED:
        raise SystemExit("Supabase not configured (missing SUPABASE_URL / SUPABASE_API_KEY)")
    return Supabase(
        SupabaseConfig(
            url=config.SUPABASE_URL,
            api_key=config.SUPABASE_API_KEY,
            bucket=config.SUPABASE_STORAGE_BUCKET,
        )
    )


async def _select_all(db: Supabase, user_id: int | None, limit: int) -> list[dict]:
    params = {"select": "*", "order": "created_at.desc", "limit": str(limit)}
    if user_id is not None:
        params["user_id"] = f"eq.{user_id}"
    response = await db._request("GET", "/rest/v1/tg_media", params=params)
    if response.status_code >= 400:
        raise RuntimeError(f"select failed ({response.status_code}): {response.text[:300]}")
    return response.json()


async def _download_one(db: Supabase, row: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, path in (("original", row.get("original_path")), ("result", row.get("result_path"))):
        if not path:
            continue
        data = await db.download(path)
        name = Path(path).name
        target = out_dir / f"{row['id'][:8]}_{kind}_{name}"
        target.write_bytes(data)
        print(f"  {kind:8} {len(data):>8} bytes  {target}")


async def cmd_list(db: Supabase, args: argparse.Namespace) -> None:
    rows = await _select_all(db, args.user, args.limit)
    if not rows:
        print("no rows")
        return
    for r in rows:
        bg = r.get("background_rgb")
        color = "transparent" if bg is None else "#{:02x}{:02x}{:02x}".format(*bg) if bg else "-"
        mode = r.get("mode") or "bg"
        print(
            f"{r['id']}  mode={mode} chat={r['chat_id']} user={r['user_id']} "
            f"prompt={r.get('prompt')!r} bg={color} {r['created_at']} "
            f"orig={r.get('original_path')} result={r.get('result_path')}"
        )


async def cmd_download(db: Supabase, args: argparse.Namespace) -> None:
    if args.all:
        rows = await _select_all(db, None, 10_000)
    else:
        rows = await _select_all(db, None, 1)
        rows = [r for r in rows if r["id"] == args.id]
    if not rows:
        print("no matching rows")
        return
    for r in rows:
        print(f"downloading {r['id']}:")
        await _download_one(db, r, Path(args.out))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="admin")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list stored photos")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--user", type=int, default=None)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download images by row id (or --all)")
    p_dl.add_argument("--id", type=str, default=None)
    p_dl.add_argument("--all", action="store_true")
    p_dl.add_argument("--out", default="./admin_downloads")
    p_dl.set_defaults(func=cmd_download)
    return parser


def main() -> None:
    args = _parser().parse_args()
    db = _db()

    async def run() -> None:
        try:
            await args.func(db, args)
        finally:
            await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()