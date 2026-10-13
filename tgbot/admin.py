"""Admin CLI: list stored photos and download originals/results.

Reads the local SQLite store directly (no remote backend).

Usage:
  uv run python -m tgbot.admin list [--limit 10] [--user 123]
  uv run python -m tgbot.admin download --id 42 [--out ./admin_downloads]
  uv run python -m tgbot.admin download --all [--out ./admin_downloads]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .localstore import LocalStore, StoredMedia


def _store() -> LocalStore:
    return LocalStore(config.OFFLINE_STORE_PATH)


def _color(background_rgb: str | None) -> str:
    if not background_rgb:
        return "-"
    try:
        import json

        rgb = json.loads(background_rgb)
        return "transparent" if rgb is None else "#{:02x}{:02x}{:02x}".format(*rgb)
    except Exception:  # noqa: BLE001
        return "-"


def _list_rows(store: LocalStore, user_id: int | None, limit: int) -> list[StoredMedia]:
    return store.all_media(user_id=user_id, limit=limit, newest_first=True)


def _download_one(store: LocalStore, row: StoredMedia, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = (
        ("original", Path(row.original_path).name, row.original_blob),
        ("aux", Path(row.aux_path).name if row.aux_path else None, row.aux_blob),
        ("result", Path(row.result_path).name, row.result_blob),
    )
    for kind, name, blob in parts:
        if not name or blob is None:
            continue
        target = out_dir / f"{row.id}_{kind}_{name}"
        target.write_bytes(blob)
        print(f"  {kind:8} {len(blob):>8} bytes  {target}")


def cmd_list(store: LocalStore, args: argparse.Namespace) -> None:
    rows = _list_rows(store, args.user, args.limit)
    if not rows:
        print("no rows")
        return
    for r in rows:
        mode = r.mode or "bg"
        print(
            f"{r.id}  mode={mode} chat={r.chat_id} user={r.user_id} "
            f"prompt={r.prompt!r} bg={_color(r.background_rgb)} {r.created_at} "
            f"orig={r.original_path} result={r.result_path}"
        )


def cmd_download(store: LocalStore, args: argparse.Namespace) -> None:
    if args.all:
        rows = store.all_media()
    elif args.id is not None:
        row = store.get_media(args.id)
        rows = [row] if row else []
    else:
        rows = store.all_media(limit=1, newest_first=True)
    if not rows:
        print("no matching rows")
        return
    for r in rows:
        print(f"downloading {r.id}:")
        _download_one(store, r, Path(args.out))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="admin")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list stored photos")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--user", type=int, default=None)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download images by row id (or --all)")
    p_dl.add_argument("--id", type=int, default=None)
    p_dl.add_argument("--all", action="store_true")
    p_dl.add_argument("--out", default="./admin_downloads")
    p_dl.set_defaults(func=cmd_download)
    return parser


def main() -> None:
    args = _parser().parse_args()
    store = _store()
    args.func(store, args)


if __name__ == "__main__":
    main()
