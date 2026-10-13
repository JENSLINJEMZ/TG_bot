"""Pre-download the local models so the first real request isn't a surprise.

Run with:  uv run python -m tgbot.prefetch
"""

from __future__ import annotations

import os

from . import config

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_DIFFUSION_SKIP = [
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


def prefetch_edit() -> None:
    """Base Qwen-Image-Edit-2509 components + the 4-bit nunchaku transformer."""
    from huggingface_hub import snapshot_download

    # skip the full-precision transformer; the 4-bit nunchaku build replaces it
    print(f"  {config.EDIT_BASE}: downloading (if needed)...", flush=True)
    snapshot_download(
        config.EDIT_BASE,
        ignore_patterns=["transformer/*", "*.bin", "*.onnx", "*.msgpack", "*.pth"],
    )
    pattern = (
        f"lightning-251115/svdq-*_r{config.EDIT_RANK}-qwen-image-edit-2509-"
        f"lightning-{config.EDIT_STEPS}steps-251115.safetensors"
    )
    print(f"  {config.EDIT_REPO}: downloading (if needed)...", flush=True)
    snapshot_download(config.EDIT_REPO, allow_patterns=[pattern])
    print(f"  {config.EDIT_REPO}: ready", flush=True)


_VIDEO_COMPONENTS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]


def prefetch_video() -> None:
    from huggingface_hub import snapshot_download

    # The LTX repo is packed with ~30 GB of extra root-level checkpoints and
    # example media; the diffusers pipeline only loads these components.
    repo = config.WAN_MODEL if config.VIDEO_ENGINE == "wan" else config.VIDEO_MODEL
    print(f"  {repo}: downloading components (if needed)...", flush=True)
    snapshot_download(repo, allow_patterns=_VIDEO_COMPONENTS)
    print(f"  {repo}: ready", flush=True)


def main() -> None:
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
    if config.EDIT_ENABLED:
        try:
            prefetch_edit()
        except Exception as exc:  # noqa: BLE001
            print(f"  editor prefetch failed: {exc}", flush=True)
    if config.VIDEO_ENABLED:
        try:
            prefetch_video()
        except Exception as exc:  # noqa: BLE001
            print(f"  video prefetch failed: {exc}", flush=True)
    print(f"Done. Models live under {config.MODELS_DIR}.", flush=True)


if __name__ == "__main__":
    main()