"""Local, Claude-style web chat UI for the Telegram bot's chat model.

Run it with ``python -m web`` (or ``python -m web --port 9000``). The package is
deliberately standalone: it never imports :mod:`tgbot.bot`, so it does not need
a Telegram token and cannot start bot polling.
"""

__all__ = ["create_app"]

from .app import create_app  # noqa: E402  (kept last so imports stay simple)
