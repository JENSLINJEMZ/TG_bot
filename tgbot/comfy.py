"""ComfyUI HTTP client for the Wan 2.2 I2V-A14B animation engine.

The 14B two-expert Wan model is far too big to load in the bot process, so it
runs inside a separate ComfyUI server (started by ``ensure_server`` if needed)
and the bot only talks to it over HTTP:

* ``POST /upload/image``   - stage the source photo
* ``POST /prompt``         - queue an API-format graph
* ``GET  /history/{id}``   - poll until the graph finishes
* ``GET  /view``           - download the produced MP4
* ``GET  /system_stats``   - liveness probe

``generate`` mirrors :func:`tgbot.video.generate` and returns
``(mp4 bytes, "video/mp4", seconds)``. The graph does the full quality pass:

    LoadImage ─► WanImageToVideo ─► KSamplerAdvanced (high) ─► (low)
              └► ReActorFaceSwap (identity) ─► RIFE VFI (smoothing)
                 ─► Real-ESRGAN 2x ─► VHS_VideoCombine (H.264 MP4)
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from . import config
from .progress import TRACKER, format_elapsed

_log = logging.getLogger("tgbot.comfy")

# a fresh client id per process so this bot's runs are distinguishable
_CLIENT_ID = uuid.uuid4().hex


def _log_info(msg: str) -> None:
    print(msg, flush=True)
    _log.info("%s", msg)


# --------------------------------------------------------------------------- #
# low-level HTTP
# --------------------------------------------------------------------------- #
def _base() -> str:
    return config.COMFY_URL


def _http(path: str, *, data: bytes | None = None, headers: dict | None = None,
          timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(_base() + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def is_up(timeout: float = 3.0) -> bool:
    """True when a ComfyUI server answers the liveness probe."""
    try:
        _http("/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def _port_from_url() -> int:
    parsed = urlparse(_base())
    return parsed.port or (443 if parsed.scheme == "https" else 8188)


def ensure_server(wait: float = 240.0) -> bool:
    """Return True once the ComfyUI server is reachable, starting it if allowed.

    The server is launched fully detached so it survives this process; it is
    reused by later jobs instead of being restarted each time.
    """
    if is_up():
        return True
    if not config.COMFY_AUTOSTART:
        return False

    comfy_dir = Path(config.COMFY_DIR)
    python = Path(config.COMFY_PYTHON)
    if not python.exists():
        raise RuntimeError(f"ComfyUI python not found: {python}")

    log_path = comfy_dir / "tgbot_server.log"
    _log_info(f"starting ComfyUI server ({python}) ...")
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    log_file = open(log_path, "ab")
    try:
        subprocess.Popen(
            [str(python), "-u", "main.py", "--port", str(_port_from_url()),
             "--listen", "127.0.0.1"],
            cwd=str(comfy_dir),
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    finally:
        log_file.close()

    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            _log_info("ComfyUI server is up")
            return True
        time.sleep(2.0)
    return False


# --------------------------------------------------------------------------- #
# image upload
# --------------------------------------------------------------------------- #
def _multipart(field: str, filename: str, data: bytes, extra: dict) -> tuple[bytes, str]:
    boundary = "----tgbot" + uuid.uuid4().hex
    out = io.BytesIO()
    for key, value in extra.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        out.write(f"{value}\r\n".encode())
    out.write(f"--{boundary}\r\n".encode())
    out.write(
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
    )
    out.write(b"Content-Type: application/octet-stream\r\n\r\n")
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def upload_image(data: bytes, name: str, *, subfolder: str = "tgbot") -> str:
    """Upload ``data`` and return the reference ComfyUI expects ("sub/name")."""
    body, ctype = _multipart(
        "image", name, data, {"overwrite": "true", "type": "input", "subfolder": subfolder}
    )
    raw = _http(
        "/upload/image",
        data=body,
        headers={"Content-Type": ctype},
        timeout=120.0,
    )
    info = json.loads(raw.decode("utf-8"))
    saved = info.get("name", name)
    sub = info.get("subfolder", "") or subfolder
    return f"{sub}/{saved}" if sub else saved


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def _fit_dimensions(img: Image.Image, max_dim: int, step: int = 16) -> tuple[int, int]:
    """Scale so the longest edge is <= max_dim, snapped to a multiple of step."""
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    w = max(step, (w // step) * step)
    h = max(step, (h // step) * step)
    return w, h


def _clip_settings_for(seconds: float | None) -> tuple[int, int, bool]:
    """Pick per-clip resolution + native fps, easing off for long clips.

    Returns ``(base_dim, fps, long_clip)``. Clips at/after
    ``WAN14B_LONG_SECONDS`` drop to ``WAN14B_LONG_BASE_DIM`` and
    ``WAN14B_LONG_FPS`` so the 14B model plus the larger activations still fit
    (and finish) on an 8 GB card. Set ``WAN14B_LONG_SECONDS=0`` to disable.
    """
    full_dim = config.WAN14B_BASE_DIM
    full_fps = max(1, config.WAN14B_FPS)
    threshold = config.WAN14B_LONG_SECONDS
    if threshold <= 0:
        return full_dim, full_fps, False
    if seconds is None:
        secs = config.VIDEO_FRAMES / full_fps
    else:
        secs = float(seconds)
    if secs < threshold:
        return full_dim, full_fps, False
    long_dim = config.WAN14B_LONG_BASE_DIM
    dim = long_dim if 0 < long_dim < full_dim else full_dim
    long_fps = config.WAN14B_LONG_FPS
    fps = min(full_fps, long_fps) if long_fps > 0 else full_fps
    return dim, fps, True


def build_graph(
    *,
    start_image: str,
    source_image: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    fps: int | None = None,
) -> dict:
    """Build the API-format Wan 2.2 I2V-A14B graph (single final MP4 output)."""
    native_fps = max(1, fps or config.WAN14B_FPS)
    steps = config.WAN14B_STEPS
    half = max(1, steps // 2)
    cfg = config.WAN14B_CFG
    shift = config.WAN14B_SHIFT
    sampler = "euler"
    scheduler = "simple"

    g: dict[str, dict] = {}

    # inputs ---------------------------------------------------------------
    g["1"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": source_image}}

    # high-noise expert ----------------------------------------------------
    g["10"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": config.WAN14B_HIGH}}
    g["11"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["10", 0],
            "lora_name": config.WAN14B_LORA_HIGH,
            "strength_model": 1.0,
        },
    }
    g["12"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["11", 0], "shift": shift},
    }

    # low-noise expert -----------------------------------------------------
    g["20"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": config.WAN14B_LOW}}
    g["21"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["20", 0],
            "lora_name": config.WAN14B_LORA_LOW,
            "strength_model": 1.0,
        },
    }
    g["22"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["21", 0], "shift": shift},
    }

    # text + vae -----------------------------------------------------------
    g["30"] = {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {"clip_name": config.WAN14B_CLIP, "type": "wan"},
    }
    g["31"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": ["30", 0]},
    }
    g["32"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": ["30", 0]},
    }
    g["33"] = {"class_type": "VAELoader", "inputs": {"vae_name": config.WAN14B_VAE}}

    # conditioning + latent ------------------------------------------------
    g["40"] = {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive": ["31", 0],
            "negative": ["32", 0],
            "vae": ["33", 0],
            "width": width,
            "height": height,
            "length": frames,
            "batch_size": 1,
            "start_image": ["1", 0],
        },
    }

    # two-stage sampling ---------------------------------------------------
    g["50"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["12", 0],
            "add_noise": "enable",
            "noise_seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "positive": ["40", 0],
            "negative": ["40", 1],
            "latent_image": ["40", 2],
            "start_at_step": 0,
            "end_at_step": half,
            "return_with_leftover_noise": "enable",
        },
    }
    g["51"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["22", 0],
            "add_noise": "disable",
            "noise_seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "positive": ["40", 0],
            "negative": ["40", 1],
            "latent_image": ["50", 0],
            "start_at_step": half,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
        },
    }
    g["52"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["51", 0], "vae": ["33", 0]},
    }

    # post passes ----------------------------------------------------------
    prev: list = ["52", 0]

    interp = max(1, config.WAN14B_INTERP)
    interp_frames = (frames - 1) * interp + 1

    if config.WAN14B_FACE_SWAP:
        restore = config.WAN14B_RESTORE_MODEL if config.WAN14B_FACE_RESTORE else "none"
        g["60"] = {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": prev,
                "swap_model": config.WAN14B_SWAP_MODEL,
                "facedetection": config.WAN14B_FACE_DETECT,
                "face_restore_model": restore,
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
                "source_image": ["2", 0],
            },
        }
        prev = ["60", 0]

    if config.WAN14B_INTERP > 1:
        g["70"] = {
            "class_type": "RIFE VFI",
            "inputs": {
                "ckpt_name": config.WAN14B_RIFE_MODEL,
                "frames": prev,
                "clear_cache_after_n_frames": 10,
                "multiplier": config.WAN14B_INTERP,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
                "dtype": "float16",
                "torch_compile": False,
                "batch_size": 1,
            },
        }
        prev = ["70", 0]

    do_upscale = config.WAN14B_UPSCALE and (
        config.WAN14B_UPSCALE_MAX_FRAMES <= 0
        or interp_frames <= config.WAN14B_UPSCALE_MAX_FRAMES
    )
    if do_upscale:
        g["80"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": config.WAN14B_UPSCALE_MODEL},
        }
        g["81"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["80", 0], "image": prev},
        }
        prev = ["81", 0]

    out_fps = native_fps * interp
    g["90"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": prev,
            "frame_rate": float(out_fps),
            "loop_count": 0,
            "filename_prefix": "tgbot/wan14b",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
        },
    }
    return g


# --------------------------------------------------------------------------- #
# queue / poll / fetch
# --------------------------------------------------------------------------- #
def queue(graph: dict) -> str:
    payload = json.dumps({"prompt": graph, "client_id": _CLIENT_ID}).encode("utf-8")
    raw = _http(
        "/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=60.0,
    )
    resp = json.loads(raw.decode("utf-8"))
    if "error" in resp:
        raise RuntimeError(f"ComfyUI rejected the graph: {resp['error']}")
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {resp}")
    return prompt_id


def _history(prompt_id: str) -> dict:
    raw = _http(f"/history/{urllib.parse.quote(prompt_id)}", timeout=60.0)
    return json.loads(raw.decode("utf-8"))


def wait_for(prompt_id: str, timeout: float) -> dict:
    """Block until the queued graph finishes; return its ``outputs`` mapping."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _history(prompt_id)
        entry = hist.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution error: {status.get('messages')}")
            if entry.get("outputs"):
                return entry["outputs"]
        time.sleep(2.0)
    raise TimeoutError(f"ComfyUI job {prompt_id} did not finish within {timeout:.0f}s")


def _find_output_file(outputs: dict) -> tuple[str, str, str]:
    """Locate the produced video in any node output list."""
    for node_out in outputs.values():
        for value in node_out.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("filename"):
                        return (
                            item["filename"],
                            item.get("subfolder", "") or "",
                            item.get("type", "output") or "output",
                        )
            elif isinstance(value, dict) and value.get("filename"):
                return (
                    value["filename"],
                    value.get("subfolder", "") or "",
                    value.get("type", "output") or "output",
                )
    raise RuntimeError(f"no output file in ComfyUI result: {outputs}")


def fetch(filename: str, subfolder: str, file_type: str) -> bytes:
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": file_type}
    )
    return _http(f"/view?{query}", timeout=300.0)


# --------------------------------------------------------------------------- #
# progress reporting
# --------------------------------------------------------------------------- #
# Relative cost of each heavy node, used to turn ComfyUI's per-node progress
# into one overall percentage.  Weights are rough shares of wall-clock time
# observed on the reference run (5 s portrait: sampling ~55%, face swap ~20%,
# RIFE+upscale+encode ~25%); only nodes that *emit* a progress bar are listed,
# so the total naturally reaches 100%.  Nodes absent from the graph are dropped
# and the remaining weights are renormalised.
_STAGE_WEIGHTS: dict[str, tuple[str, float]] = {
    "50": ("sampling (high)", 28.0),
    "51": ("sampling (low)", 27.0),
    "60": ("face swap", 20.0),
    "70": ("smoothing (RIFE)", 8.0),
    "81": ("upscaling", 14.0),
    "90": ("encoding", 3.0),
}


def _build_plan(graph: dict) -> dict[str, tuple[str, float]]:
    """Pick the progress-emitting nodes present in ``graph``."""
    return {nid: _STAGE_WEIGHTS[nid] for nid in _STAGE_WEIGHTS if nid in graph}


def _ws_url() -> str:
    parsed = urlparse(_base())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.hostname or "127.0.0.1"
    return f"{scheme}://{host}:{_port_from_url()}/ws?clientId={_CLIENT_ID}"


class _ProgressWatcher:
    """Mirror ComfyUI's WebSocket progress into the tracker and the terminal.

    ComfyUI broadcasts ``progress_state`` / ``progress`` / ``executing`` events
    to every connected socket (``server.client_id`` is never set in this
    version), each tagged with the ``prompt_id``; we keep the events for our
    prompt and fold the per-node ``value``/``max`` into one weighted percentage.
    Runs in its own daemon thread so the main thread can keep polling history.
    """

    def __init__(self, prompt_id: str, plan: dict[str, tuple[str, float]]) -> None:
        self.prompt_id = prompt_id
        self.plan = plan
        self._total = sum(weight for _, weight in plan.values())
        self._frac: dict[str, float] = {}
        self._stop = threading.Event()
        self._finished = False
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._last_stage = ""
        self._last_pct = -1
        self._last_print = 0.0

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        self._started = time.time()
        self._thread = threading.Thread(
            target=self._run, name="comfy-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # -- websocket loop --------------------------------------------------- #
    def _run(self) -> None:
        try:
            import asyncio

            asyncio.run(self._listen())
        except Exception as exc:  # progress is best-effort; never break the job
            _log.debug("progress watcher stopped: %s", exc)

    async def _listen(self) -> None:
        try:
            import aiohttp
        except Exception:
            return
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(_ws_url(), heartbeat=30.0) as ws:
                    async for msg in ws:
                        if self._stop.is_set():
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            event = json.loads(msg.data)
                        except Exception:
                            continue
                        self._handle(event)
                        if self._finished:
                            break
        except Exception as exc:
            _log.debug("progress websocket closed: %s", exc)

    # -- event handling --------------------------------------------------- #
    def _mine(self, data: dict) -> bool:
        pid = data.get("prompt_id")
        return pid in (None, self.prompt_id)

    def _handle(self, event: dict) -> None:
        etype = event.get("type")
        data = event.get("data") or {}
        if not isinstance(data, dict):
            return

        if etype == "progress_state":
            if self._mine(data):
                self._apply_nodes(data.get("nodes") or {})
        elif etype == "progress":
            if self._mine(data):
                self._apply_node(
                    str(data.get("node") or data.get("node_id") or ""),
                    data.get("value"),
                    data.get("max"),
                )
                self._emit()
        elif etype == "execution_cached":
            if self._mine(data):
                for nid in data.get("nodes") or []:
                    if str(nid) in self.plan:
                        self._frac[str(nid)] = 1.0
                self._emit()
        elif etype == "executing":
            if self._mine(data) and data.get("node") is None:
                self._finished = True
        elif etype in ("execution_success", "execution_error", "execution_interrupted"):
            if self._mine(data):
                self._finished = True

    def _apply_node(self, node_id: str, value, maximum) -> None:
        if node_id not in self.plan:
            return
        try:
            mx = float(maximum or 0)
            val = float(value or 0)
        except (TypeError, ValueError):
            return
        if mx > 0:
            frac = min(1.0, val / mx)
            self._frac[node_id] = max(self._frac.get(node_id, 0.0), frac)

    def _apply_nodes(self, nodes: dict) -> None:
        for node_id, state in nodes.items():
            node_id = str(node_id)
            if node_id not in self.plan or not isinstance(state, dict):
                continue
            if state.get("state") in ("finished", "error"):
                self._frac[node_id] = 1.0
                continue
            self._apply_node(node_id, state.get("value"), state.get("max"))
        self._emit()

    # -- terminal + tracker ---------------------------------------------- #
    def _emit(self) -> None:
        if self._total <= 0:
            return
        done = 0.0
        running = ""
        pending = ""
        for node_id, (label, weight) in self.plan.items():
            frac = self._frac.get(node_id, 0.0)
            done += weight * frac
            if 0.0 < frac < 1.0 and not running:
                running = label
            if frac <= 0.0 and not pending:
                pending = label
        percent = done / self._total * 100.0
        stage = running or pending or "finishing"
        TRACKER.update(percent=percent, stage=stage)

        now = time.time()
        pct = int(percent)
        stage_changed = stage != self._last_stage
        if not stage_changed and now - self._last_print < 1.0:
            return
        if not stage_changed and pct - self._last_pct < 3:
            return
        self._last_stage = stage
        self._last_pct = pct
        self._last_print = now
        elapsed = now - self._started
        eta = ""
        if 0 < percent < 100 and elapsed > 20:
            eta = f" | ~{format_elapsed(elapsed * (100 - percent) / percent)} left"
        print(f"[wan14b] {pct:3d}% | {stage} | {format_elapsed(elapsed)}{eta}", flush=True)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def generate(prompt: str | None, data: bytes, seconds: float | None = None):
    """Animate one photo via ComfyUI. Returns (mp4 bytes, "video/mp4", seconds)."""
    TRACKER.begin("video", stage="preparing input")
    try:
        return _generate(prompt, data, seconds)
    finally:
        TRACKER.end()


def _generate(prompt: str | None, data: bytes, seconds: float | None = None):
    started = time.perf_counter()

    if not ensure_server():
        raise RuntimeError(
            f"ComfyUI server not reachable at {config.COMFY_URL} "
            "(and autostart is disabled or timed out)"
        )

    # start image: pre-fit to the exact model resolution (aspect preserved).
    # long clips use a smaller base + lower fps so the 14B model and the bigger
    # activations still fit (and finish) on an 8 GB card.
    base_dim, fps, long_clip = _clip_settings_for(seconds)
    src = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = _fit_dimensions(src, base_dim, step=16)
    fitted = src.resize((width, height), Image.LANCZOS) if src.size != (width, height) else src
    start_buf = io.BytesIO()
    fitted.save(start_buf, "PNG")

    token = uuid.uuid4().hex[:8]
    fit_name = upload_image(start_buf.getvalue(), f"start_{token}.png")
    # identity reference keeps full resolution for the best face features
    src_buf = io.BytesIO()
    src.save(src_buf, "PNG")
    ref_name = upload_image(src_buf.getvalue(), f"ref_{token}.png")

    if seconds is None:
        frames = config.VIDEO_FRAMES
    else:
        frames = max(9, int(round(seconds * fps)) + 1)
        if (frames - 1) % 4:
            frames = ((frames - 1) // 4) * 4 + 1

    seed = config.WAN14B_SEED or random.randint(0, 2**63 - 1)
    text = (prompt or "").strip() or config.VIDEO_DEFAULT_PROMPT

    graph = build_graph(
        start_image=fit_name,
        source_image=ref_name,
        prompt=text,
        negative=config.VIDEO_NEGATIVE,
        width=width,
        height=height,
        frames=frames,
        seed=seed,
        fps=fps,
    )

    interp = max(1, config.WAN14B_INTERP)
    interp_frames = (frames - 1) * interp + 1
    do_upscale = config.WAN14B_UPSCALE and (
        config.WAN14B_UPSCALE_MAX_FRAMES <= 0
        or interp_frames <= config.WAN14B_UPSCALE_MAX_FRAMES
    )
    _log_info(
        f"comfy queue: {width}x{height}, {frames} frames -> {interp_frames} @ "
        f"{fps * interp}fps, seed {seed}, "
        f"face_swap={config.WAN14B_FACE_SWAP}, upscale={do_upscale}"
        + (f" (long clip -> {base_dim}px, {fps}fps)" if long_clip else "")
    )
    prompt_id = queue(graph)
    TRACKER.update(stage="loading models", prompt_id=prompt_id)

    watcher = _ProgressWatcher(prompt_id, _build_plan(graph))
    if watcher.plan:
        watcher.start()
    try:
        outputs = wait_for(prompt_id, timeout=config.COMFY_TIMEOUT)
    finally:
        watcher.stop()
    filename, subfolder, file_type = _find_output_file(outputs)
    payload = fetch(filename, subfolder, file_type)

    secs = time.perf_counter() - started
    _log_info(
        f"video done (wan14b, {frames} frames -> {interp_frames}): "
        f"{len(payload) // 1024} KiB in {secs:.1f}s"
    )
    return payload, "video/mp4", secs
