"""Small locally-generated PNG used as a 'processing' sticker (bot uploads it via InputFile)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def processing_sticker() -> bytes:
    """512x512 'Processing...' PNG to send via send_sticker as a BufferedInputFile."""
    size = 512
    image = Image.new("RGB", (size, size), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    draw.arc((156, 150, 356, 350), start=20, end=290, fill=(66, 133, 244), width=14)
    draw.ellipse((236, 230, 276, 270), fill=(66, 133, 244))
    font = _font(44)
    draw.text((256, 400), "Processing...", font=font, fill=(235, 235, 235), anchor="mm")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()