"""Swap a face photo onto an existing video.

The heavy lifting happens in the local ComfyUI server: Video Helper Suite loads
the target frames, ReActor replaces the face and GFPGAN restores it, then VHS
writes an H.264 chunk.  This module keeps the bot-side work small and reliable
on an 8 GB card:

* normalise the upload with ffmpeg (trim, 720p max, <=24 fps, keyframes);
* split it into short chunks so one ComfyUI job never holds a huge frame batch;
* run one ReActor graph per chunk through :mod:`tgbot.comfy`;
* concatenate the swapped chunks and re-mux the original audio.

The public entry point is :func:`swap_video`, which returns the same
``(bytes, "video/mp4", seconds)`` tuple as the other media engines.
"""

from __future__ import annotations

import io
import logging
import math
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from PIL import Image

from . import comfy, config
from .progress import TRACKER

log = logging.getLogger("tgbot.faceswap")

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


def _ffmpeg() -> str:
    """Path to the ffmpeg binary bundled with imageio-ffmpeg."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], *, timeout: float = 600.0) -> None:
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace")[-2000:].strip()
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {detail}")


def _safe_ext(name: str, default: str = ".mp4") -> str:
    ext = Path(name or "").suffix.lower()
    return ext if ext in _VIDEO_EXTS else default


def _finite(value, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) and number > 0 else fallback


def probe_video(path: str | Path) -> dict:
    """Return ``{fps, duration, width, height}`` for a local video file."""
    import imageio.v3 as iio

    try:
        meta = iio.immeta(str(path), plugin="FFMPEG")
    except Exception as exc:  # noqa: BLE001 - surface a friendly bot error
        raise ValueError("Could not read that video. Try an MP4, MOV or WebM.") from exc

    size = meta.get("size") or meta.get("source_size") or (0, 0)
    try:
        width, height = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        width = height = 0
    return {
        "fps": _finite(meta.get("fps"), 24.0),
        "duration": _finite(meta.get("duration"), 0.0),
        "width": width,
        "height": height,
    }


def normalise_video(
    src: str | Path,
    dst: str | Path,
    *,
    max_dim: int | None = None,
    max_fps: int | None = None,
    max_seconds: int | None = None,
) -> dict:
    """Trim, downscale and re-encode the target before ComfyUI sees it."""
    info = probe_video(src)
    if info["duration"] <= 0:
        raise ValueError("That video has no readable duration.")

    max_dim = max(256, int(max_dim if max_dim is not None else config.FACESWAP_MAX_DIM))
    min_dim = max(128, int(os.getenv("FACESWAP_MIN_DIM", str(config.FACESWAP_MIN_DIM)) or str(config.FACESWAP_MIN_DIM)))
    fps_cap = max(1, int(max_fps if max_fps is not None else config.FACESWAP_FPS))
    limit = max(1, int(max_seconds if max_seconds is not None else config.FACESWAP_MAX_SECONDS))
    out_fps = max(1.0, min(info["fps"], float(fps_cap)))
    out_fps = round(out_fps, 3)
    seconds = min(info["duration"], float(limit))
    # Force a keyframe every ~2 s so the segment muxer can cut close to the
    # requested chunk size instead of waiting for a 10 s encoder GOP.
    gop = max(1, int(round(out_fps * 2)))

    # Fit within max_dim, then ensure at least min_dim on the short side.
    # This helps face detection on very small uploads without wasting VRAM
    # on already-large videos.
    vf = (
        f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease,"
        f"scale='max({min_dim},iw)':'max({min_dim},ih)':force_original_aspect_ratio=increase,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"fps={out_fps:g}"
    )
    _run(
        [
            "-i", str(src),
            "-t", f"{seconds:.3f}",
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(dst),
        ],
        timeout=max(120.0, seconds * 10.0),
    )
    out = probe_video(dst)
    return {
        "fps": out["fps"] or out_fps,
        "duration": out["duration"] or seconds,
        "width": out["width"],
        "height": out["height"],
    }


def split_video(src: str | Path, out_dir: str | Path, chunk_seconds: int) -> list[Path]:
    """Split a normalised video into keyframe-aligned MP4 chunks."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    pattern = folder / "chunk_%03d.mp4"
    _run(
        [
            "-i", str(src),
            "-map", "0:v:0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(max(1, int(chunk_seconds))),
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            str(pattern),
        ],
        timeout=300.0,
    )
    parts = sorted(folder.glob("chunk_*.mp4"))
    if not parts:
        raise RuntimeError("Could not split that video into chunks.")
    return parts


def _concat_video(parts: list[Path], dst: Path) -> None:
    """Join same-format chunks, falling back to a re-encode if copy fails."""
    list_file = dst.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{part.as_posix()}'\n" for part in parts),
        encoding="utf-8",
    )
    base = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    try:
        _run([*base, "-c", "copy", str(dst)], timeout=600.0)
    except RuntimeError:
        log.warning("faceswap: stream-copy concat failed, re-encoding")
        _run(
            [
                *base,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(dst),
            ],
            timeout=900.0,
        )


def _mux_audio(video: Path, audio_source: Path, dst: Path) -> None:
    """Put the original (trimmed) audio back on the swapped video."""
    base = ["-i", str(video), "-i", str(audio_source)]
    maps = ["-map", "0:v:0", "-map", "1:a:0?"]
    try:
        _run(
            [
                *base,
                *maps,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(dst),
            ],
            timeout=600.0,
        )
    except RuntimeError:
        log.warning("faceswap: audio mux copy failed, re-encoding video")
        _run(
            [
                *base,
                *maps,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(dst),
            ],
            timeout=900.0,
        )


def _face_png(face: bytes) -> bytes:
    try:
        image = Image.open(io.BytesIO(face)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Could not read the face photo.") from exc
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def swap_video(
    face: bytes,
    video: bytes,
    video_name: str = "input.mp4",
) -> tuple[bytes, str, float]:
    """Swap ``face`` onto ``video`` and return ``(mp4, "video/mp4", seconds)``."""
    started = time.perf_counter()
    if not face or not video:
        raise ValueError("Both a face photo and a video are required.")
    if not comfy.ensure_server():
        raise RuntimeError(
            f"ComfyUI server not reachable at {config.COMFY_URL} "
            "(and autostart is disabled or timed out)"
        )

    token = uuid.uuid4().hex[:10]
    with tempfile.TemporaryDirectory(prefix="tgbot-faceswap-") as temp:
        folder = Path(temp)
        src = folder / f"input{_safe_ext(video_name)}"
        src.write_bytes(video)

        TRACKER.update(stage="preparing video")
        normal = folder / "normalised.mp4"
        info = normalise_video(src, normal)

        face_ref = comfy.upload_image(_face_png(face), f"faceswap_face_{token}.png")
        parts = split_video(normal, folder, config.FACESWAP_CHUNK_SECONDS)
        log.info(
            "faceswap: %d chunk(s), %.1fs @ %.2ffps, %dx%d",
            len(parts),
            info["duration"],
            info["fps"],
            info["width"],
            info["height"],
        )

        swapped: list[Path] = []
        for index, part in enumerate(parts, start=1):
            stage = f"face swap {index}/{len(parts)}"
            TRACKER.update(stage=stage)
            # VHS_LoadVideo lists files in ComfyUI's input root, so upload
            # chunks there rather than into a subfolder.
            chunk_ref = comfy.upload_image(
                part.read_bytes(),
                f"faceswap_{token}_{index:03d}.mp4",
                subfolder="",
            )
            graph = comfy.build_faceswap_graph(
                face_image=face_ref,
                video_name=chunk_ref,
                fps=info["fps"],
            )
            payload = comfy.run_graph(graph, stage=stage)
            out = folder / f"swapped_{index:03d}.mp4"
            out.write_bytes(payload)
            swapped.append(out)

        if len(swapped) == 1:
            combined = swapped[0]
        else:
            TRACKER.update(stage="joining chunks")
            combined = folder / "swapped_all.mp4"
            _concat_video(swapped, combined)

        TRACKER.update(stage="restoring audio")
        final = folder / "final.mp4"
        _mux_audio(combined, normal, final)
        payload = final.read_bytes()

    if len(payload) < 1024:
        raise RuntimeError("Face swap produced an empty video.")
    secs = time.perf_counter() - started
    log.info("faceswap done: %d KiB in %.1fs", len(payload) // 1024, secs)
    return payload, "video/mp4", secs
