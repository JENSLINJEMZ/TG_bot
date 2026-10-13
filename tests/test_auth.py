"""Authorization gate for inline-keyboard callbacks (regression for the audit find)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import bot, config


def _cb(user_id):
    return SimpleNamespace(from_user=SimpleNamespace(id=user_id))


def test_callbacks_open_when_no_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALLOWED_USERS", set())
    assert bot._authorized_cb(_cb(123)) is True


def test_callbacks_enforced_when_allowlist_set(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALLOWED_USERS", {42})
    assert bot._authorized_cb(_cb(42)) is True
    assert bot._authorized_cb(_cb(7)) is False


def test_callback_without_user_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALLOWED_USERS", {42})
    assert bot._authorized_cb(SimpleNamespace(from_user=None)) is False


def test_every_callback_handler_checks_authorization() -> None:
    """Every registered callback_query handler must call _authorized_cb."""
    allowed = False
    for _handler, callback_query in _callbacks():
        # the callback is registered, so it exists; inspect its source
        import inspect

        assert "_authorized_cb" in inspect.getsource(callback_query), callback_query.__name__
        allowed = True
    assert allowed, "no callback handlers discovered"


def _callbacks():
    for handler in bot.dp.callback_query.handlers:
        yield handler, handler.callback
