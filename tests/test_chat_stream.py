import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transformers

from tgbot import chat


def test_build_messages_custom_system_prompt():
    default = chat.build_messages("hi")
    assert default[0]["content"] == chat.SYSTEM_PROMPT

    web = chat.build_messages("hi", system_prompt=chat.WEB_SYSTEM_PROMPT)
    assert web[0]["content"] == chat.WEB_SYSTEM_PROMPT
    assert web[-1] == {"role": "user", "content": "hi"}


class _FakeIds:
    shape = (1, 3)

    def __getitem__(self, item):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": _FakeIds(), "attention_mask": None}

    def decode(self, ids, skip_special_tokens=True):
        return "  hello  "


class _FakeModel:
    def generate(self, *args, **kwargs):
        return _FakeIds()


class _FakeTorch:
    def inference_mode(self):
        return contextlib.nullcontext()


def test_reply_does_not_name_error(monkeypatch):
    """Regression for the missing StoppingCriteria import at chat.py:125."""
    monkeypatch.setattr(chat, "_get_model", lambda: (_FakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(chat, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr("tgbot.diffuse.unload", lambda: None)
    monkeypatch.setattr("tgbot.progress.report", lambda *a, **k: None)

    answer, seconds = chat.reply("hi")
    assert answer == "hello"
    assert seconds >= 0


def test_stream_reply_yields_chunks(monkeypatch):
    class FakeStreamer:
        def __init__(self, tokenizer, **kwargs):
            self.chunks = iter(["Hel", "lo"])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.chunks)

    class StreamTokenizer(_FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    class StreamModel:
        def generate(self, **kwargs):
            assert kwargs["streamer"] is not None
            return [[1, 2, 3]]

    monkeypatch.setattr(chat, "_get_model", lambda: (StreamModel(), StreamTokenizer()))
    monkeypatch.setattr(chat, "_torch", lambda: None)
    monkeypatch.setattr(transformers, "TextIteratorStreamer", FakeStreamer)
    monkeypatch.setattr("tgbot.diffuse.unload", lambda: None)
    monkeypatch.setattr("tgbot.progress.report", lambda *a, **k: None)

    chunks = list(chat.stream_reply("hi", system_prompt=chat.WEB_SYSTEM_PROMPT))
    assert "".join(chunks) == "Hello"
