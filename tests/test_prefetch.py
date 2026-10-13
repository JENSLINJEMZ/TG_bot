"""Offline tests for the model prefetch helpers (no network access)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import config, prefetch


def test_fetch_file_skips_existing_file(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"already here")

    def boom(*_args, **_kwargs):
        raise AssertionError("must not download when the file already exists")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    prefetch._fetch_file("some/repo", "model.bin", dest)
    assert dest.read_bytes() == b"already here"


def test_prefetch_video_skips_for_wan14b(monkeypatch) -> None:
    # the 14B engine loads its weights inside ComfyUI, so no diffusers snapshot
    monkeypatch.setattr(config, "VIDEO_ENGINE", "wan14b")

    def boom(*_args, **_kwargs):
        raise AssertionError("wan14b must not download a diffusers snapshot")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    prefetch.prefetch_video()  # returns before touching huggingface_hub


def test_prefetch_comfy_bails_when_comfyui_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "COMFY_DIR", str(tmp_path / "does-not-exist"))
    prefetch.prefetch_comfy()  # prints a hint and returns, no download
