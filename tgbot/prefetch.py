"""Pre-download the local models so the first real request isn't a surprise.

Run with:  uv run python -m tgbot.prefetch
"""

from __future__ import annotations

import os
from pathlib import Path

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
    if config.VIDEO_ENGINE == "wan14b":
        # this engine runs inside the ComfyUI server; its weights are fetched by
        # prefetch_comfy() below, not as a diffusers snapshot
        print("  video engine 'wan14b': in-process snapshot skipped (ComfyUI models)", flush=True)
        return
    from huggingface_hub import snapshot_download

    # The LTX repo is packed with ~30 GB of extra root-level checkpoints and
    # example media; the diffusers pipeline only loads these components.
    repo = config.WAN_MODEL if config.VIDEO_ENGINE == "wan" else config.VIDEO_MODEL
    print(f"  {repo}: downloading components (if needed)...", flush=True)
    snapshot_download(repo, allow_patterns=_VIDEO_COMPONENTS)
    print(f"  {repo}: ready", flush=True)


# --------------------------------------------------------------------------- #
# ComfyUI (wan14b) model pack
# --------------------------------------------------------------------------- #
# The 14B engine runs in a separate ComfyUI server, so its weights live under
# COMFY_DIR\models instead of the bot's diffusers cache. The repos and on-disk
# layout mirror ComfyUI's own dl_models.py / dl_post.py, so this and those
# scripts stay interchangeable.
_COMFY_UNET_REPO = "bullerwins/Wan2.2-I2V-A14B-GGUF"
_COMFY_CLIP_REPO = "city96/umt5-xxl-encoder-gguf"
_COMFY_REPACK = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
_COMFY_REACTOR = "Gourieff/ReActor"
_COMFY_UPSCALE_REPO = "ai-forever/Real-ESRGAN"


def _fetch_file(repo: str, filename: str, dest: Path, *, repo_type: str = "model") -> None:
    """Download one HuggingFace file to ``dest`` (hardlinked from the cache if possible)."""
    import shutil

    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    {dest.name}: present", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type)).resolve()
    if dest.exists():
        dest.unlink()
    try:
        os.link(cached, dest)  # same volume: no second copy of the weights
    except OSError:
        shutil.copy2(cached, dest)
    print(f"    {dest.name}: {dest.stat().st_size // (1024 * 1024)} MB", flush=True)


def prefetch_comfy() -> None:
    """Download the Wan 2.2 I2V-A14B pack that the ComfyUI engine loads."""
    import zipfile

    base = Path(config.COMFY_DIR).expanduser()
    if not (base / "models").is_dir() and not (base / "main.py").exists():
        print(f"  ComfyUI not found at {base} - set COMFY_DIR, skipping 14B models", flush=True)
        return

    models = base / "models"
    print(f"  14B pack -> {models} (several GB, first run only)", flush=True)

    # unet (both experts), text encoder, vae, loras
    _fetch_file(_COMFY_UNET_REPO, config.WAN14B_HIGH, models / "unet" / config.WAN14B_HIGH)
    _fetch_file(_COMFY_UNET_REPO, config.WAN14B_LOW, models / "unet" / config.WAN14B_LOW)
    _fetch_file(_COMFY_CLIP_REPO, config.WAN14B_CLIP, models / "text_encoders" / config.WAN14B_CLIP)
    _fetch_file(
        _COMFY_REPACK, f"split_files/vae/{config.WAN14B_VAE}", models / "vae" / config.WAN14B_VAE
    )
    _fetch_file(
        _COMFY_REPACK,
        f"split_files/loras/{config.WAN14B_LORA_HIGH}",
        models / "loras" / config.WAN14B_LORA_HIGH,
    )
    _fetch_file(
        _COMFY_REPACK,
        f"split_files/loras/{config.WAN14B_LORA_LOW}",
        models / "loras" / config.WAN14B_LORA_LOW,
    )

    # identity lock (ReActor) + face restore
    _fetch_file(
        _COMFY_REACTOR,
        f"models/{config.WAN14B_SWAP_MODEL}",
        models / "insightface" / config.WAN14B_SWAP_MODEL,
        repo_type="dataset",
    )
    _fetch_file(
        _COMFY_REACTOR,
        f"models/facerestore_models/{config.WAN14B_RESTORE_MODEL}",
        models / "facerestore_models" / config.WAN14B_RESTORE_MODEL,
        repo_type="dataset",
    )

    # buffalo_l face-analysis pack (extract so insightface finds models/buffalo_l/*.onnx)
    out = models / "insightface" / "models"
    if (out / "buffalo_l").is_dir():
        print("    buffalo_l: present", flush=True)
    else:
        zip_dest = models / "insightface" / "buffalo_l.zip"
        _fetch_file(_COMFY_REACTOR, "models/buffalo_l.zip", zip_dest, repo_type="dataset")
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dest) as zf:
            zf.extractall(out)
        print("    buffalo_l: extracted", flush=True)

    # 2x upscaler for the post pass
    _fetch_file(
        _COMFY_UPSCALE_REPO,
        config.WAN14B_UPSCALE_MODEL,
        models / "upscale_models" / config.WAN14B_UPSCALE_MODEL,
    )

    print(
        "  14B pack ready (rife49.pth is fetched by the Frame-Interpolation node on first use)",
        flush=True,
    )


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
        if config.VIDEO_ENGINE == "wan14b":
            try:
                prefetch_comfy()
            except Exception as exc:  # noqa: BLE001
                print(f"  ComfyUI 14B prefetch failed: {exc}", flush=True)
    comfy_note = " (14B engine models live under COMFY_DIR\\models)" if config.VIDEO_ENGINE == "wan14b" else ""
    print(f"Done. Models live under {config.MODELS_DIR}{comfy_note}.", flush=True)


if __name__ == "__main__":
    main()