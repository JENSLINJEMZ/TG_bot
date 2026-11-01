"""Tests for the Face swap video feature."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOT_PY = Path(__file__).resolve().parent.parent / "tgbot" / "bot.py"
FACESWAP_PY = Path(__file__).resolve().parent.parent / "tgbot" / "faceswap.py"
CONFIG_PY = Path(__file__).resolve().parent.parent / "tgbot" / "config.py"


def test_faceswap_states_exist():
    """Modes.faceswap_face and Modes.faceswap_video are defined."""
    source = BOT_PY.read_text(encoding="utf-8")
    assert "faceswap_face = State()" in source
    assert "faceswap_video = State()" in source


def test_faceswap_menu_button_exists():
    """The menu keyboard includes a 'Face swap video' button."""
    source = BOT_PY.read_text(encoding="utf-8")
    assert 'callback_data="mode_faceswap"' in source


def test_faceswap_help_text():
    """The help text mentions face swap."""
    source = BOT_PY.read_text(encoding="utf-8")
    assert "Face swap video" in source or "face swap video" in source


def test_faceswap_config_constants():
    """FACESWAP_* config constants are defined with sensible defaults."""
    source = CONFIG_PY.read_text(encoding="utf-8")
    for const in (
        "FACESWAP_ENABLED",
        "FACESWAP_MAX_INPUT_BYTES",
        "FACESWAP_MAX_SECONDS",
        "FACESWAP_CHUNK_SECONDS",
        "FACESWAP_MAX_DIM",
        "FACESWAP_MIN_DIM",
        "FACESWAP_FPS",
        "FACESWAP_FACE_RESTORE",
        "FACESWAP_FACE_INDEX",
        "FACESWAP_SWAP_MODEL",
        "FACESWAP_RESTORE_MODEL",
        "FACESWAP_FACE_DETECT",
    ):
        assert const in source, f"missing {const} in config.py"


def test_faceswap_graph_construction():
    """comfy.build_faceswap_graph produces a valid node graph."""
    from tgbot import comfy

    g = comfy.build_faceswap_graph(
        face_image="tgbot/face.png",
        video_name="faceswap_chunk_001.mp4",
        fps=24.0,
    )

    # Required node classes
    classes = {v["class_type"] for v in g.values()}
    assert "LoadImage" in classes
    assert "VHS_LoadVideo" in classes
    assert "ReActorFaceSwap" in classes
    assert "VHS_VideoCombine" in classes

    # Wiring: face -> LoadImage(1), video -> VHS_LoadVideo(2)
    assert g["1"]["inputs"]["image"] == "tgbot/face.png"
    assert g["2"]["inputs"]["video"] == "faceswap_chunk_001.mp4"

    # ReActor inputs
    r = g["60"]["inputs"]
    assert r["enabled"] is True
    assert r["input_image"] == ["2", 0]          # frames from VHS_LoadVideo
    assert r["source_image"] == ["1", 0]         # face from LoadImage
    assert r["swap_model"] == "inswapper_128.onnx"
    assert r["facedetection"] == "retinaface_resnet50"
    assert r["face_restore_model"] == "GFPGANv1.4.pth"
    assert r["input_faces_index"] == "0"
    assert r["source_faces_index"] == "0"

    # VideoCombine inputs
    v = g["90"]["inputs"]
    assert v["images"] == ["60", 0]
    assert v["frame_rate"] == 24.0
    assert v["format"] == "video/h264-mp4"
    assert v["filename_prefix"] == "tgbot/faceswap"
    assert v["save_output"] is True


def test_faceswap_progressive_scale_filter():
    """normalise_video uses a scale filter that enforces both max and min dims."""
    import re
    source = FACESWAP_PY.read_text(encoding="utf-8")
    # The scale filter chain should contain both force_original_aspect_ratio=decrease
    # (for max dim) and force_original_aspect_ratio=increase (for min dim)
    assert "force_original_aspect_ratio=decrease" in source
    assert "force_original_aspect_ratio=increase" in source


def test_faceswap_probe_video():
    """probe_video returns the expected keys for a local file."""
    from tgbot import faceswap
    from pathlib import Path

    # Use one of the test videos that ships with the repo
    vid = Path("output/wan_5s_test.mp4")
    if not vid.exists():
        import pytest
        pytest.skip("test video not present")

    info = faceswap.probe_video(vid)
    for k in ("fps", "duration", "width", "height"):
        assert k in info
    assert info["fps"] > 0
    assert info["duration"] > 0
    assert info["width"] > 0
    assert info["height"] > 0


def test_download_video_helper_exists():
    """_download_video helper is defined and handles video/document/animation."""
    source = BOT_PY.read_text(encoding="utf-8")
    assert "async def _download_video" in source
    assert "message.video" in source
    assert "message.animation" in source
    assert "message.document" in source
    assert "FACESWAP_MAX_INPUT_BYTES" in source


def test_faceswap_progress_title():
    """_PROGRESS_TITLES includes a 'faceswap' entry."""
    source = BOT_PY.read_text(encoding="utf-8")
    assert '"faceswap"' in source
    assert "Swapping face" in source or "face swap" in source.lower()


def test_faceswap_module_imports():
    """tgbot.faceswap can be imported and has the public entry point."""
    from tgbot import faceswap
    assert hasattr(faceswap, "swap_video")
    assert hasattr(faceswap, "normalise_video")
    assert hasattr(faceswap, "split_video")
    assert hasattr(faceswap, "probe_video")