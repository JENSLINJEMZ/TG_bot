import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import chat
from tgbot import config


def test_build_messages_shape():
    messages = chat.build_messages("hello")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == chat.SYSTEM_PROMPT
    assert messages[-1] == {"role": "user", "content": "hello"}
    assert len(messages) == 2


def test_build_messages_appends_history():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    messages = chat.build_messages("again", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "again"


def test_build_messages_trims_history_to_turns():
    history = [
        {"role": "user", "content": f"u{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
        for i in range(10)
    ]
    messages = chat.build_messages("new", history, turns=1)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_build_messages_with_zero_turns_drops_history():
    history = [{"role": "user", "content": "old"}]
    messages = chat.build_messages("new", history, turns=0)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_unload_without_model_is_noop():
    chat.unload()


def test_chat_model_is_configured_small():
    assert config.CHAT_MODEL
    assert config.CHAT_MAX_NEW_TOKENS >= 16