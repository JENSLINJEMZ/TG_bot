"""GPU outfit editing with Qwen-Image-Edit-2509 (4-bit Nunchaku, SVDQuant).

This is the heavy "restyle outfit" engine: it runs the 20B Qwen-Image-Edit-2509
model on the RTX GPU while the 7B text encoder and VAE stay in system RAM
(sequential CPU offload), so an 8 GB card can run it. The transformer itself is a
4-bit Nunchaku SVDQuant build (FP4 on Blackwell, INT4 elsewhere).

The runtime is entirely optional: if CUDA or the `nunchaku` package is missing,
`is_available()` returns False and the caller falls back to diffuse/tint.
"""

from __future__ import annotations

import gc
import io
import math
import time

from . import config
from .progress import report

_log = None


def _log_info(msg: str) -> None:
    global _log
    if _log is None:
        import logging

        _log = logging.getLogger("tgbot.edit")
    _log.info(msg)


_pipeline = None
_available: bool | None = None


def is_available() -> bool:
    """True when the GPU + nunchaku runtime can actually run the editor."""
    global _available
    if _available is None:
        try:
            import importlib
            import torch

            importlib.import_module("nunchaku")
            _available = bool(torch.cuda.is_available())
        except Exception:
            _available = False
    return _available


def _model_path() -> str:
    """Nunchaku repo/filename for the 4-bit transformer matching this GPU."""
    from nunchaku.utils import get_precision

    precision = get_precision()  # "fp4" on Blackwell, "int4" otherwise
    steps = config.EDIT_STEPS
    rank = config.EDIT_RANK
    return (
        f"{config.EDIT_REPO}/lightning-251115/"
        f"svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{steps}steps-251115.safetensors"
    )


def _patch_diffusers_compat() -> None:
    """Adapt nunchaku's older QwenEmbedRope call to diffusers >= 0.41.

    nunchaku calls `pos_embed(video_fhw, txt_seq_lens, device=...)`; newer diffusers
    changed the signature to `(video_fhw, device=..., max_txt_seq_len=...)`. The two
    are equivalent for a single sample, so we accept both call shapes.
    """
    try:
        from diffusers.models.transformers import transformer_qwenimage as tq
    except Exception:
        return
    cls = getattr(tq, "QwenEmbedRope", None)
    if cls is None or getattr(cls, "_nunchaku_compat", False):
        return
    import inspect

    if "txt_seq_lens" in inspect.signature(cls.forward).parameters:
        cls._nunchaku_compat = True
        return
    orig = cls.forward

    def forward(self, video_fhw, *args, device=None, max_txt_seq_len=None):
        if args and max_txt_seq_len is None:
            lens = args[0]
            max_txt_seq_len = max(lens) if isinstance(lens, (list, tuple)) else int(lens)
        return orig(self, video_fhw, device=device, max_txt_seq_len=max_txt_seq_len)

    cls.forward = forward
    cls._nunchaku_compat = True


def _patch_nunchaku_compat() -> None:
    """Provide `txt_seq_lens`, which diffusers >= 0.41 no longer passes.

    The base pipeline passes only `encoder_hidden_states`, but nunchaku's forward
    still needs the text sequence length for RoPE. Mirror diffusers: use the padded
    encoder length (`encoder_hidden_states.shape[1]`) for every sample.
    """
    from nunchaku import NunchakuQwenImageTransformer2DModel

    cls = NunchakuQwenImageTransformer2DModel
    if getattr(cls, "_tgbot_compat", False):
        return
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if not args and kwargs.get("txt_seq_lens") is None:
            states = kwargs.get("encoder_hidden_states")
            if states is not None:
                kwargs["txt_seq_lens"] = [states.shape[1]] * states.shape[0]
        return orig(self, *args, **kwargs)

    cls.forward = forward
    cls._tgbot_compat = True


def _scheduler():
    """Scheduler distilled for the Lightning 4/8-step checkpoints."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(
        {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
    )


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel

        _patch_diffusers_compat()
        _patch_nunchaku_compat()
        model_path = _model_path()
        _log_info("loading Qwen-Image-Edit-2509 (%s)" % model_path)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            config.EDIT_BASE,
            transformer=transformer,
            scheduler=_scheduler(),
            torch_dtype=torch.bfloat16,
        )
        # low VRAM: keep transformer blocks in RAM and stream them in per layer
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=config.EDIT_GPU_BLOCKS
        )
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
        # decode/encode the VAE in tiles so its activations fit in ~4 GB of free VRAM
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        _pipeline = pipe
        _log_info("Qwen-Image-Edit-2509 ready")
    return _pipeline


_INSTRUCTION_VERBS = {
    "change", "make", "replace", "turn", "add", "remove", "put", "give", "wear",
    "dress", "swap", "edit", "modify", "transform", "recolor", "recolour", "apply",
    "show", "let", "generate", "draw", "paint", "set", "color", "colour", "tint",
    "undress", "remove",
}


def _append_identity(instruction: str) -> str:
    """Add the keep-identity clause unless one is already there."""
    clause = config.EDIT_IDENTITY_CLAUSE.strip()
    low = instruction.lower()
    if clause and clause.lower() not in low and "keep " not in low and "keep the" not in low:
        instruction = f"{instruction.rstrip(' .')}. {clause}"
    return instruction


def to_instruction(prompt: str | None) -> str:
    """Turn a short caption ("navy suit", "bikini") into an edit instruction.

    Bare garment words are wrapped in a decisive "replace the clothing" sentence so
    revealing outfits (swimwear, lingerie, ...) actually replace what is worn instead
    of being ignored. Full commands are passed through verbatim. No content filter is
    applied - the prompt drives the result.
    """
    text = (prompt or "").strip()
    if not text:
        return _append_identity("Replace the person's current outfit with a stylish new one.")
    first = text.split()[0].lower()
    if first in _INSTRUCTION_VERBS:
        return _append_identity(text)
    return _append_identity(config.EDIT_PROMPT_TEMPLATE.format(prompt=text))


def to_combine_instruction(prompt: str | None) -> str:
    """Wrap a scene prompt for the multi-image "combine two photos" edit.

    Unlike :func:`to_instruction` this does not assume an outfit, so it does not
    force the clothing template onto a scene description like "hug each other".
    """
    text = (prompt or "").strip() or "the two people standing together"
    template = config.EDIT_COMBINE_TEMPLATE
    if template and "{prompt}" in template:
        return template.format(prompt=text)
    return text


def _fit(image, max_dim: int):
    """Downscale so the longest edge fits, snapped to multiples of 16."""
    width, height = image.size
    scale = min(1.0, max_dim / max(width, height))
    if scale >= 1.0:
        return image
    new_w = max(16, int(width * scale) // 16 * 16)
    new_h = max(16, int(height * scale) // 16 * 16)
    from PIL import Image

    return image.resize((new_w, new_h), Image.LANCZOS)


def generate(prompt: str, data: bytes) -> tuple[bytes, str, float]:
    """Redraw the person's outfit from `prompt`. Returns (jpeg bytes, mime, seconds)."""
    from PIL import Image

    started = time.perf_counter()
    for mod in ("chat", "diffuse"):
        try:
            import importlib

            importlib.import_module(f"tgbot.{mod}").unload()
        except Exception:
            pass

    report(1, "loading model", label="qwen-edit")
    pipe = _get_pipeline()
    image = _fit(Image.open(io.BytesIO(data)).convert("RGB"), config.EDIT_MAX_DIM)
    instruction = to_instruction(prompt)
    _log_info("qwen edit: %r" % instruction)
    report(5, "encoding prompt", label="qwen-edit")

    def _on_step(_pipe, _step, _timestep, callback_kwargs):
        report(8 + 84 * (_step + 1) / max(1, config.EDIT_STEPS), "denoising", label="qwen-edit")
        return callback_kwargs

    result = pipe(
        image=[image],
        prompt=instruction,
        num_inference_steps=config.EDIT_STEPS,
        true_cfg_scale=config.EDIT_CFG,
        callback_on_step_end=_on_step,
    )
    report(95, "decoding image", label="qwen-edit")
    out = result.images[0]
    if out.mode != "RGB":
        out = out.convert("RGB")
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=95)
    report(100, "finishing", label="qwen-edit")
    return buffer.getvalue(), "image/jpeg", time.perf_counter() - started


def generate_multi(
    prompt: str,
    images: list[bytes],
    *,
    instruction: str | None = None,
) -> tuple[bytes, str, float]:
    """Compose several photos into one from `prompt` (Qwen-Image-Edit multi-image).

    Used by the "combine two photos" mode: e.g. a man + a woman with the prompt
    "hug each other". Returns (jpeg bytes, mime, seconds). Pass ``instruction``
    to use it verbatim instead of wrapping ``prompt`` (used when the caller
    already has a fully-formed scene instruction).
    """
    from PIL import Image

    started = time.perf_counter()
    for mod in ("chat", "diffuse"):
        try:
            import importlib

            importlib.import_module(f"tgbot.{mod}").unload()
        except Exception:
            pass

    report(1, "loading model", label="qwen-edit")
    pipe = _get_pipeline()
    fitted = [
        _fit(Image.open(io.BytesIO(data)).convert("RGB"), config.EDIT_MAX_DIM)
        for data in images
    ]
    final_instruction = instruction if instruction else to_combine_instruction(prompt)
    _log_info("qwen combine (%d images): %r" % (len(fitted), final_instruction))
    report(5, "encoding prompt", label="qwen-edit")

    def _on_step(_pipe, _step, _timestep, callback_kwargs):
        report(8 + 84 * (_step + 1) / max(1, config.EDIT_STEPS), "denoising", label="qwen-edit")
        return callback_kwargs

    result = pipe(
        image=fitted,
        prompt=final_instruction,
        num_inference_steps=config.EDIT_STEPS,
        true_cfg_scale=config.EDIT_CFG,
        callback_on_step_end=_on_step,
    )
    report(95, "decoding image", label="qwen-edit")
    out = result.images[0]
    if out.mode != "RGB":
        out = out.convert("RGB")
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=95)
    report(100, "finishing", label="qwen-edit")
    return buffer.getvalue(), "image/jpeg", time.perf_counter() - started


def unload() -> None:
    """Drop the pipeline so the image pipeline / chat can reclaim RAM."""
    global _pipeline
    if _pipeline is None:
        return
    _pipeline = None
    _log_info("Qwen-Image-Edit-2509 unloaded")
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
