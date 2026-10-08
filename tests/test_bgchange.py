import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from tgbot.bgchange import (
    NAMED_COLORS,
    Gradient,
    Solid,
    Transparent,
    change_background,
    change_cloth_color,
    combine_images,
    describe,
    parse_background,
    restyle_outfit,
)

DEFAULT = (255, 255, 255)


def _synthetic_photo(size: tuple[int, int] = (320, 320)) -> bytes:
    image = Image.new("RGB", size, (200, 210, 220))
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 80, 240, 240), fill=(180, 40, 40))
    draw.rectangle((140, 40, 180, 140), fill=(40, 60, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_parse_solid() -> None:
    bg, note = parse_background("navy", DEFAULT)
    assert isinstance(bg, Solid) and bg.rgb == (30, 58, 138) and note is None
    bg, note = parse_background("#ff8800", DEFAULT)
    assert isinstance(bg, Solid) and bg.rgb == (255, 136, 0) and note is None
    bg, note = parse_background(None, DEFAULT)
    assert bg == Solid(DEFAULT) and note is None
    bg, note = parse_background("", DEFAULT)
    assert bg == Solid(DEFAULT) and note is None


def test_parse_transparent() -> None:
    bg, note = parse_background("transparent", DEFAULT)
    assert isinstance(bg, Transparent) and note is None


def test_parse_gradient() -> None:
    bg, note = parse_background("gradient red blue", DEFAULT)
    assert isinstance(bg, Gradient)
    assert bg.top == (220, 38, 38) and bg.bottom == (37, 99, 235)
    assert note is None
    bg, note = parse_background("fade #ff0000 to #0000ff", DEFAULT)
    assert isinstance(bg, Gradient)
    assert bg.top == (255, 0, 0) and bg.bottom == (0, 0, 255)


def test_parse_unknown_uses_default() -> None:
    bg, note = parse_background("something weird", DEFAULT)
    assert bg == Solid(DEFAULT) and note is not None


def test_describe() -> None:
    assert describe(Solid((255, 136, 0))) == "#ff8800"


def test_change_background_solid_and_gradient() -> None:
    for bg in (Solid((255, 0, 0)), Gradient((255, 0, 0), (0, 0, 255))):
        payload, content_type = change_background(_synthetic_photo(), bg)
        assert content_type == "image/jpeg"
        assert payload.startswith(b"\xff\xd8")
        assert len(payload) > 1000


def test_change_background_transparent() -> None:
    payload, content_type = change_background(_synthetic_photo(), Transparent())
    assert content_type == "image/png"
    assert payload.startswith(b"\x89PNG")


def test_change_background_downscales_large_images() -> None:
    payload, content_type = change_background(_synthetic_photo((4000, 4000)), Solid((0, 0, 0)), max_dim=1024)
    assert content_type == "image/jpeg"
    with Image.open(io.BytesIO(payload)) as out:
        assert max(out.size) <= 1024


def test_combine_images() -> None:
    subject = _synthetic_photo((320, 320))
    background = _synthetic_photo((640, 480))
    payload, content_type = combine_images(subject, background, max_dim=800)
    assert content_type == "image/jpeg"
    assert payload.startswith(b"\xff\xd8")
    with Image.open(io.BytesIO(payload)) as out:
        assert out.size == (640, 480)


def test_change_cloth_color() -> None:
    payload, content_type = change_cloth_color(_synthetic_photo((320, 320)), (255, 0, 0), max_dim=800)
    assert content_type == "image/png"
    assert payload.startswith(b"\x89PNG")


def test_restyle_outfit_uses_prompt_color() -> None:
    result, color, note = restyle_outfit(_synthetic_photo((320, 320)), "navy suit", (200, 200, 200), max_dim=800)
    payload, content_type = result
    assert color == NAMED_COLORS["navy"]
    assert note is None
    assert content_type == "image/png"
    assert payload.startswith(b"\x89PNG")


def test_restyle_outfit_falls_back_when_no_color() -> None:
    _, color, note = restyle_outfit(_synthetic_photo((320, 320)), "modern streetwear", (120, 40, 220), max_dim=800)
    assert color == (120, 40, 220)
    assert note is not None


def test_processing_sticker_kind() -> None:
    from tgbot.sticker import processing_sticker

    payload = processing_sticker()
    assert payload.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(payload)) as im:
        assert im.size == (512, 512)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")