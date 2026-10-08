"""Local background replacement: rembg (U2Net) cutout + solid/gradient/transparent background."""

from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageFilter

NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (220, 38, 38),
    "green": (22, 163, 74),
    "blue": (37, 99, 235),
    "yellow": (234, 179, 8),
    "orange": (249, 115, 22),
    "purple": (147, 51, 234),
    "pink": (236, 72, 153),
    "gray": (107, 114, 128),
    "grey": (107, 114, 128),
    "cyan": (8, 145, 178),
    "magenta": (217, 70, 239),
    "brown": (120, 53, 15),
    "beige": (245, 237, 220),
    "navy": (30, 58, 138),
    "teal": (20, 184, 166),
    "lime": (132, 204, 22),
    "maroon": (127, 29, 29),
    "olive": (113, 128, 50),
    "silver": (192, 192, 192),
    "gold": (202, 138, 4),
}

TRANSPARENT_WORDS = {"transparent", "none", "clear", "cutout", "alpha", "no bg", "nobg"}
GRADIENT_WORDS = {"gradient", "fade", "transition"}

_HEX_RE = re.compile(r"(?<![\w])#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![\w])")


@dataclass(frozen=True)
class Solid:
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class Gradient:
    top: tuple[int, int, int]
    bottom: tuple[int, int, int]


@dataclass(frozen=True)
class Transparent:
    pass


Background = Solid | Gradient | Transparent


def _color_from_word(cleaned: str) -> tuple[int, int, int] | None:
    match = _HEX_RE.search(cleaned)
    if match:
        hexpart = match.group(1)
        if len(hexpart) == 3:
            hexpart = "".join(ch * 2 for ch in hexpart)
        return tuple(int(hexpart[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    for name, rgb in NAMED_COLORS.items():
        if re.search(rf"\b{name}\b", cleaned):
            return rgb
    return None


def _tokenize_colors(cleaned: str) -> list[tuple[int, int, int]]:
    candidates = []
    for word in re.split(r"[,\s/]+", cleaned):
        color = _color_from_word(word)
        if color:
            candidates.append(color)
    return candidates


def parse_background(text: str | None, default_rgb: tuple[int, int, int]) -> tuple[Background, str | None]:
    """Parse a caption into a background plus an optional note for the user.

    Returns (background, note|None). The note is set when the caption could not
    be understood and the default color was used.
    """
    if text is None:
        return Solid(default_rgb), None
    cleaned = text.strip().lower()
    if not cleaned:
        return Solid(default_rgb), None

    if cleaned in TRANSPARENT_WORDS:
        return Transparent(), None

    is_gradient = any(re.search(rf"\b{word}\b", cleaned) for word in GRADIENT_WORDS)
    colors = _tokenize_colors(cleaned)
    if is_gradient and len(colors) >= 2:
        return Gradient(top=colors[0], bottom=colors[1]), None
    if colors:
        return Solid(colors[0]), None
    return Solid(default_rgb), f"unknown color {text!r} - used default instead"


def parse_hex(value: str) -> tuple[int, int, int]:
    rgb = _color_from_word(value.strip())
    return rgb or (255, 255, 255)


def describe(background: Background) -> str:
    if isinstance(background, Gradient):
        return (
            f"gradient #{background.top[0]:02x}{background.top[1]:02x}{background.top[2]:02x}"
            f" -> #{background.bottom[0]:02x}{background.bottom[1]:02x}{background.bottom[2]:02x}"
        )
    if isinstance(background, Solid):
        return f"#{background.rgb[0]:02x}{background.rgb[1]:02x}{background.rgb[2]:02x}"
    return "transparent"


_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            from rembg import new_session

            _session = new_session("u2net")
        return _session


def _fit(source: Image.Image, max_dim: int) -> Image.Image:
    width, height = source.size
    scale = min(1.0, max_dim / max(width, height))
    if scale < 1.0:
        source = source.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
        )
    if source.mode != "RGB":
        source = source.convert("RGB")
    return source


def _gradient_image(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    height = max(size[1] - 1, 1)
    for y in range(size[1]):
        t = y / height
        row = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line((0, y, size[0], y), fill=row)
    return image


def _composite(cut: Image.Image, background: Background):
    overlay = Image.new("RGBA", cut.size, (0, 0, 0, 0))
    if isinstance(background, Gradient):
        base = _gradient_image(cut.size, background.top, background.bottom)
        bg_rgba = base.convert("RGBA")
    elif isinstance(background, Solid):
        bg_rgba = Image.new("RGBA", cut.size, background.rgb + (255,))
    else:
        return cut, "PNG", "image/png"
    overlay.alpha_composite(bg_rgba)
    overlay.alpha_composite(cut)
    return overlay, "JPEG", "image/jpeg"


def change_background(
    data: bytes, background: Background, max_dim: int = 1600
) -> tuple[bytes, str]:
    """Cut the subject out of `data` and recomposite onto `background`.

    Returns (payload, content_type). Transparent results are PNG, otherwise JPEG.
    """
    from rembg import remove

    source = _fit(Image.open(io.BytesIO(data)), max_dim)
    cut = remove(source, session=_get_session())
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")

    out, fmt, content_type = _composite(cut, background)
    buffer = io.BytesIO()
    if fmt == "JPEG":
        out.convert("RGB").save(buffer, format="JPEG", quality=92)
    else:
        out.save(buffer, format="PNG")
    return buffer.getvalue(), content_type


def _cutout(data: bytes, max_dim: int) -> Image.Image:
    from rembg import remove

    source = _fit(Image.open(io.BytesIO(data)), max_dim)
    cut = remove(source, session=_get_session())
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    return cut


def combine_images(subject: bytes, background: bytes, max_dim: int = 1600) -> tuple[bytes, str]:
    """Cut the subject out of `subject` and paste it centered onto `background`."""
    cut = _cutout(subject, max_dim)
    bg = _fit(Image.open(io.BytesIO(background)), max_dim).convert("RGB")
    if cut.width > bg.width or cut.height > bg.height:
        scale = min(bg.width / cut.width, bg.height / cut.height) * 0.9
        cut = cut.resize((max(1, int(cut.width * scale)), max(1, int(cut.height * scale))), Image.LANCZOS)
    position = ((bg.width - cut.width) // 2, (bg.height - cut.height) // 2)
    out = bg.convert("RGBA")
    out.alpha_composite(cut, position)
    buffer = io.BytesIO()
    out.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), "image/jpeg"


def _garment_region(mask: Image.Image, fraction_from: float = 0.22, fraction_to: float = 0.62) -> Image.Image | None:
    bbox = mask.getbbox()
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    top = y0 + int(height * fraction_from)
    bottom = y0 + int(height * fraction_to)
    region = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(region)
    draw.rectangle((x0, top, x1, bottom), fill=255)
    radius = max(1, min(mask.size) // 80)
    return region.filter(ImageFilter.GaussianBlur(radius))


def change_cloth_color(data: bytes, color: tuple[int, int, int], max_dim: int = 1600) -> tuple[bytes, str]:
    """Tint the garment area of a person photo to `color`. Returns a PNG cutout."""
    cut = _cutout(data, max_dim)
    mask = cut.getchannel("A")
    region = _garment_region(mask)
    if region is None:
        return data, "image/jpeg"
    gray_rgb = cut.convert("L").convert("RGB")
    tint = Image.new("RGB", cut.size, color)
    recolored = ImageChops.multiply(tint, gray_rgb).convert("RGBA")
    recolored.putalpha(ImageChops.multiply(mask, region))
    out = Image.alpha_composite(cut.copy(), recolored)
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def restyle_outfit(
    data: bytes, prompt: str | None, default_rgb: tuple[int, int, int], max_dim: int = 1600
) -> tuple[tuple[bytes, str], tuple[int, int, int], str | None]:
    """Restyle the garment from a free-text prompt.

    Colors are the driver: a named color, hex, or the first color word in the
    prompt tints the garment (full restyle needs a generative model - this is a
    local tint). Returns ((payload, content_type), color, note|None).
    """
    background, note = parse_background(prompt, default_rgb)
    if isinstance(background, Gradient):
        color = background.top
        note = "gradients not supported for clothes - used top color"
    elif isinstance(background, Transparent):
        color = default_rgb
        note = "clothes need a solid color - used default"
    else:
        color = background.rgb
    return change_cloth_color(data, color, max_dim), color, note