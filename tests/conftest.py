import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import config


@pytest.fixture(autouse=True)
def _transformers_by_default(monkeypatch):
    """Tests must not depend on a local Ollama server or the developer's .env."""
    monkeypatch.setattr(config, "CHAT_BACKEND", "transformers")
