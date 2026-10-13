"""Image-to-video animation on CUDA.

Backends selected by ``VIDEO_ENGINE``:

* ``wan`` (default) - Wan 2.2 TI2V-5B Turbo (4-step distilled, CFG-free,
  uncensored). Loaded from ``models/wan-ti2v-5b`` when present, else downloaded.
* ``ltx`` - LTX-Video (2B), loaded from ``models/ltx-video`` when present.
* ``wan14b`` - Wan 2.2 I2V-A14B (14B) through a local ComfyUI server for the
  best quality (GGUF Q3 + LightX2V 4-step LoRAs, ReActor face swap, RIFE
  interpolation and a Real-ESRGAN 2x pass). See :mod:`tgbot.comfy`.

All return a short MP4 as ``(bytes, "video/mp4", seconds)``.
"""

from __future__ import annotations

import gc
import io
import logging
import tempfile
import time
from pathlib import Path

from PIL import Image

from . import config

_video_log = logging.getLogger("tgbot.video")

_pipeline = None
_available: bool | None = None


def _log_info(msg: str) -> None:
    print(msg, flush=True)
    _video_log.info("%s", msg)


def is_available() -> bool:
    """Generation needs CUDA; CPU generation is not practical."""
    global _available
    if _available is None:
        try:
            import torch

            _available = bool(torch.cuda.is_available())
        except Exception:
            _available = False
    return _available


def frames_for_seconds(seconds: float) -> int:
    """Frame count for the active engine at ``VIDEO_FPS`` (stride-corrected).

    LTX needs frames = 8*k + 1, Wan needs frames = 4*k + 1.
    """
    frames = max(9, int(round(seconds * config.VIDEO_FPS)) + 1)
    stride = 8 if config.VIDEO_ENGINE == "ltx" else 4
    if (frames - 1) % stride:
        frames = ((frames - 1) // stride) * stride + 1
    return frames


def _fit(data: bytes, max_dim: int, *, step: int = 32) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if max(w, h) < 256:
        scale = max(scale, 256 / max(w, h))
    if scale < 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))
    w, h = max(step, (w // step) * step), max(step, (h // step) * step)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _get_ltx_pipeline():
    import torch
    from diffusers import LTXImageToVideoPipeline

    local = Path(config.VIDEO_LOCAL_DIR)
    if (local / "model_index.json").exists():
        _log_info(f"loading LTX video model from local folder {local}...")
        pipe = LTXImageToVideoPipeline.from_pretrained(str(local), torch_dtype=torch.bfloat16)
    else:
        _log_info(f"loading LTX video model {config.VIDEO_MODEL}...")
        # The repo ships ~30 GB of root-level full checkpoints + media we don't
        # need; only these component folders are loaded by the pipeline.
        _COMPONENT_SKIP = ["media/*", "ltxv-*.safetensors", "ltx-video-2b-*.safetensors"]
        pipe = LTXImageToVideoPipeline.from_pretrained(
            config.VIDEO_MODEL,
            torch_dtype=torch.bfloat16,
            ignore_patterns=_COMPONENT_SKIP,
        )
    # only the transformer occupies VRAM while components stream in
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return pipe


def _get_wan_pipeline():
    import torch
    from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline

    local = Path(config.WAN_LOCAL_DIR)
    if (local / "model_index.json").exists():
        _log_info(f"loading Wan video model from local folder {local}...")
        src = str(local)
        pipe = WanImageToVideoPipeline.from_pretrained(src, torch_dtype=torch.bfloat16)
    else:
        _log_info(f"loading Wan video model {config.WAN_MODEL}...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            config.WAN_MODEL, torch_dtype=torch.bfloat16
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=config.WAN_FLOW_SHIFT
    )
    # the bf16 transformer (~10 GB) is bigger than the 8 GB card, so stream the
    # modules in one at a time (peak VRAM stays under ~2.5 GB)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return pipe


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if config.VIDEO_ENGINE == "ltx":
        _pipeline = _get_ltx_pipeline()
        _log_info("ltx video model loaded")
    else:
        _pipeline = _get_wan_pipeline()
        _log_info("wan video model loaded")
    return _pipeline


def _generate_wan(pipe, prompt: str, image: Image.Image, frames: int):
    import torch

    kwargs = {
        "image": image,
        "prompt": prompt,
        "height": image.size[1],
        "width": image.size[0],
        "num_frames": frames,
        "num_inference_steps": config.WAN_STEPS,
        "guidance_scale": config.WAN_CFG,
        "generator": torch.Generator(device="cuda").manual_seed(0),
    }
    # Turbo is CFG-free, so a negative prompt only has an effect above 1.0
    if config.WAN_CFG > 1.0 and config.VIDEO_NEGATIVE:
        kwargs["negative_prompt"] = config.VIDEO_NEGATIVE
    return pipe(**kwargs)


def _generate_ltx(pipe, prompt: str, image: Image.Image, frames: int):
    return pipe(
        image=image,
        prompt=prompt,
        negative_prompt=config.VIDEO_NEGATIVE,
        height=image.size[1],
        width=image.size[0],
        num_frames=frames,
        frame_rate=config.VIDEO_FPS,
        num_inference_steps=config.VIDEO_STEPS,
        guidance_scale=config.VIDEO_CFG,
    )


def generate(prompt: str | None, data: bytes, seconds: float | None = None) -> tuple[bytes, str, float]:
    """Animate one photo. Returns (mp4 bytes, content type, generation seconds).

    ``seconds`` selects the clip length (rounded up to the engine's frame
    stride); when omitted the configured ``VIDEO_FRAMES`` is used.
    """
    # The 14B engine runs inside a separate ComfyUI server; it must not be
    # blocked by this process's CUDA check (and needs no local torch).
    if config.VIDEO_ENGINE == "wan14b":
        from . import comfy

        return comfy.generate(prompt, data, seconds)

    if not is_available():
        raise RuntimeError("video generation requires CUDA")
    started = time.perf_counter()
    pipe = _get_pipeline()

    from diffusers.utils import export_to_video

    image = _fit(data, config.VIDEO_MAX_DIM, step=32)
    prompt = (prompt or "").strip() or config.VIDEO_DEFAULT_PROMPT
    frames = config.VIDEO_FRAMES if seconds is None else frames_for_seconds(seconds)

    if config.VIDEO_ENGINE == "ltx":
        output = _generate_ltx(pipe, prompt, image, frames)
    else:
        output = _generate_wan(pipe, prompt, image, frames)

    result_frames = output.frames[0]
    tmp = Path(tempfile.mkdtemp(prefix="tgbot_video_")) / "out.mp4"
    export_to_video(result_frames, str(tmp), fps=config.VIDEO_FPS, quality=6)
    payload = tmp.read_bytes()
    tmp.unlink()
    secs = time.perf_counter() - started
    _log_info(
        f"video done ({config.VIDEO_ENGINE}, {frames} frames): "
        f"{len(payload) // 1024} KiB in {secs:.1f}s"
    )
    return payload, "video/mp4", secs


def unload() -> None:
    global _pipeline, _available
    _pipeline = None
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    """CLI smoke: uv run python -m tgbot.video <photo> ["prompt"] [out.mp4]"""
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m tgbot.video photo.jpg [prompt] [out.mp4]")
    with open(sys.argv[1], "rb") as fh:
        photo = fh.read()
    out = sys.argv[3] if len(sys.argv) > 3 else "video_out.mp4"
    payload, ctype, secs = generate(sys.argv[2] if len(sys.argv) > 2 else None, photo)
    Path(out).write_bytes(payload)
    print(f"wrote {out} ({ctype}) in {secs:.1f}s")
