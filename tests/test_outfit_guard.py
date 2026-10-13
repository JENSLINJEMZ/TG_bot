"""The "change cloth" custom prompt must accept only outfit edits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import bot  # noqa: E402

ACCEPT = [
    "red leather jacket and black jeans",
    "a bikini",
    "change the shirt to blue",
    "put her in a hoodie",
    "remove his tie",
    "dress up in a saree",
    "make the dress red",
]

REJECT = [
    "add a man near him",
    "change the background to a beach",
    "make her smile",
    "put a dog next to her",
    "remove the tree behind her",
    "",
]


def test_outfit_prompts_are_accepted():
    for text in ACCEPT:
        assert bot._is_outfit_request(text), text


def test_non_outfit_prompts_are_rejected():
    for text in REJECT:
        assert not bot._is_outfit_request(text), text


def test_every_preset_is_an_outfit_prompt():
    assert bot.CLOTH_PRESETS, "no outfit presets defined"
    for key, (_label, prompt) in bot.CLOTH_PRESETS.items():
        assert bot._is_outfit_request(prompt), (key, prompt)
