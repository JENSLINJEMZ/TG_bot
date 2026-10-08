"""Local prompt-driven restyle with a tiny diffusers model (CPU-friendly).

Runs img2img so the person's identity and pose survive while the prompt
redraws the outfit.

Memory strategy for a low-RAM laptop (this repo's weights are big):
the text encoder, VAE and UNet are each loaded only for their own phase and
released immediately, so peak RSS stays ~1.2GB instead of ~2.1GB. The UNet is
the only component optionally kept resident between calls (it is reused on
every step, reloading it costs ~30s).

dtype: fp32 only. This CPU has no native bf16/fp16 - benchmarked ~200x
slower when emulated - so low-precision weights are rejected outright.
"""

from __future__ import annotations

import gc
import io
import time

import torch
from PIL import Image

from . import config

_log = None


def _log_info(msg: str) -> None:
    global _log
    if _log is None:
        import logging

        _log = logging.getLogger("tgbot.diffuse")
    _log.info(msg)


def _release() -> None:
    """Drop references and return free heap to the OS.

    The text encoder and VAE are transient on purpose; without malloc_trim the
    allocator keeps their pages and peak RSS stays ~2GB (OOM territory here).
    """
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _torch_dtype():
    requested = config.DIFFUSE_DTYPE
    if requested in ("fp16", "bf16"):
        # benchmarked: emulated on this CPU (~200x slower than fp32)
        _log_info("dtype %s rejected - using fp32 for this CPU" % requested)
    return torch.float32


def _module(name: str):
    """Load one pipeline component from the local cache (no network)."""
    repo = config.DIFFUSE_MODEL
    dtype = _torch_dtype()
    torch.set_num_threads(config.DIFFUSE_THREADS)
    if name == "text_encoder":
        from transformers import CLIPTextModel

        return CLIPTextModel.from_pretrained(
            repo, subfolder="text_encoder", local_files_only=True, torch_dtype=dtype
        )
    if name == "tokenizer":
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", local_files_only=True)
    if name == "vae":
        from diffusers import AutoencoderKL

        return AutoencoderKL.from_pretrained(repo, subfolder="vae", local_files_only=True, torch_dtype=dtype)
    if name == "unet":
        from diffusers import UNet2DConditionModel

        return UNet2DConditionModel.from_pretrained(
            repo, subfolder="unet", local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    if name == "scheduler":
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler", local_files_only=True)
    raise ValueError(name)


_cached_unet = None
_cached_scheduler = None


def _get_unet():
    global _cached_unet, _cached_scheduler
    if _cached_unet is None:
        _cached_unet = _module("unet")
        _cached_scheduler = _module("scheduler")
    return _cached_unet, _cached_scheduler


def _drop_unet() -> None:
    global _cached_unet, _cached_scheduler
    _cached_unet = None
    _cached_scheduler = None
    _release()


def unload() -> None:
    """Release the cached UNet so the chat model can have the RAM."""
    _drop_unet()


def generate(prompt: str, data: bytes, resolution: int | None = None) -> tuple[bytes, str, float]:
    """Redraw `data` (person photo) with `prompt`. Returns (jpeg bytes, mime, seconds)."""
    started = time.perf_counter()
    try:
        from .chat import unload as unload_chat

        unload_chat()
    except Exception:
        pass
    size = resolution or config.DIFFUSE_RESOLUTION
    source = Image.open(io.BytesIO(data)).convert("RGB")
    original_size = source.size
    image = _fit(source, size)

    payload = _img2img(prompt, image)

    return _restore_size(payload, original_size), "image/jpeg", time.perf_counter() - started


def _img2img(prompt: str, image: Image.Image) -> bytes:
    steps = config.DIFFUSE_STEPS
    strength = config.DIFFUSE_STRENGTH
    device = torch.device("cpu")

    # phase 1 - text encoding (transient: released before the UNet loads)
    tokenizer = _module("tokenizer")
    text_encoder = _module("text_encoder")
    text_inputs = tokenizer(
        prompt or "modern outfit",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.inference_mode():
        prompt_embeds = text_encoder(text_inputs)[0].to(torch.float32)
    _log_info("phase 1 text encode done")
    del tokenizer, text_encoder, text_inputs
    _release()

    # phase 2 - VAE encode to get the img2img starting latents (transient)
    vae = _module("vae")
    pixels = _to_vae_input(image)
    with torch.inference_mode():
        init_latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
    _log_info("phase 2 vae encode done")
    del vae, pixels
    _release()

    # phase 3 - denoise loop (UNet stays resident when caching is enabled)
    unet, scheduler = _get_unet()
    _release()  # release text/VAE pages retained by the allocator before the loop
    scheduler.set_timesteps(steps, device=device)
    timesteps = _img2img_timesteps(scheduler, steps, strength)
    noise = torch.randn_like(init_latents)
    latents = scheduler.add_noise(init_latents.to(device), noise.to(device), timesteps[:1].to(device))
    _log_info("phase 3 denoise start steps=%d" % len(timesteps))
    for t in timesteps:
        model_input = scheduler.scale_model_input(latents, t)
        with torch.inference_mode():
            noise_pred = unet(model_input, t, encoder_hidden_states=prompt_embeds).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    _log_info("phase 3 denoise done")

    if not config.DIFFUSE_UNLOAD_AFTER:
        pass  # unet stays cached in module globals for the next request
    else:
        _drop_unet()

    # phase 4 - VAE decode (reloaded: it was released after phase 2)
    vae = _module("vae")
    with torch.inference_mode():
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
    del vae
    _release()

    array = decoded[0].detach().to(torch.float32).clamp(-1, 1)
    array = ((array + 1) / 2 * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _img2img_timesteps(scheduler, steps: int, strength: float) -> torch.Tensor:
    """Same strength windowing diffusers uses for img2img."""
    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    return scheduler.timesteps[t_start * scheduler.order :]


def _to_vae_input(image: Image.Image) -> torch.Tensor:
    width = image.width - (image.width % 8)
    height = image.height - (image.height % 8)
    image = image.resize((max(8, width), max(8, height)), Image.LANCZOS)
    import numpy as np

    array = np.asarray(image, dtype="float32") / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _fit(image: Image.Image, max_dim: int) -> Image.Image:
    width, height = image.size
    scale = max_dim / max(width, height)
    if scale < 1.0:
        image = image.resize(
            (max(8, int(width * scale)), max(8, int(height * scale))), Image.LANCZOS
        )
    return image


def _restore_size(payload: bytes, size: tuple[int, int]) -> bytes:
    """Match the uploaded photo's dimensions (generation runs at low res for speed)."""
    try:
        image = Image.open(io.BytesIO(payload))
        if image.size == size:
            return payload
        image = image.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()
    except Exception:
        return payload


def is_available() -> bool:
    try:
        import importlib

        importlib.import_module("torch")
        return True
    except Exception:
        return False