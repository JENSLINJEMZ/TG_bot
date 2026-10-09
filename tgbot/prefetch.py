"""Pre-download the local models so the first real request isn't a surprise.

Run with:  uv run python -m tgbot.prefetch
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_DIFFUSION_SKIP = [
    "*.bin",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.msgpack",
    "safety_checker/*",
]


def prefetch_rembg(name: str = "u2net") -> None:
    from rembg import new_session

    print(f"  rembg {name}: downloading (if needed)...", flush=True)
    new_session(name)
    print(f"  rembg {name}: ready", flush=True)


def prefetch_hf(repo: str, ignore: list[str] | None = None, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    print(f"  {repo}@{revision or 'main'}: downloading (if needed)...", flush=True)
    snapshot_download(repo, ignore_patterns=ignore, revision=revision or None)
    print(f"  {repo}: ready", flush=True)


def main() -> None:
    from . import config

    print("Prefetching local models (this can take a while):", flush=True)
    try:
        prefetch_rembg()
    except Exception as exc:  # noqa: BLE001
        print(f"  rembg prefetch failed: {exc}", flush=True)
    if config.DIFFUSE_ENABLED:
        try:
            prefetch_hf(config.DIFFUSE_MODEL, ignore=_DIFFUSION_SKIP, revision=config.DIFFUSE_REVISION)
        except Exception as exc:  # noqa: BLE001
            print(f"  {config.DIFFUSE_MODEL} prefetch failed: {exc}", flush=True)
    if config.CHAT_ENABLED:
        try:
            prefetch_hf(config.CHAT_MODEL, revision=config.CHAT_REVISION)
        except Exception as exc:  # noqa: BLE001
            print(f"  {config.CHAT_MODEL} prefetch failed: {exc}", flush=True)
    print("Done. Models live in ~/.u2net and the HuggingFace cache.", flush=True)


if __name__ == "__main__":
    main()