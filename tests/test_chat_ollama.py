import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import chat, config


class _OllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json({"models": [{"name": "qwen3:8b"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = self._read_json()
        self.server.requests.append((self.path, body))
        if self.path == "/api/generate":
            self._json({"model": body.get("model"), "done": True})
            return
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        if body.get("model") == "missing":
            self._json({"error": "model 'missing' not found"}, 404)
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for chunk in ("Hel", "lo"):
                line = {"message": {"content": chunk}, "done": False}
                self.wfile.write((json.dumps(line) + "\n").encode("utf-8"))
                self.wfile.flush()
            done = {"message": {"content": ""}, "done": True}
            self.wfile.write((json.dumps(done) + "\n").encode("utf-8"))
            self.wfile.flush()
            return
        self._json({"message": {"content": "Hello"}, "done": True})


@pytest.fixture()
def ollama(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setattr(config, "CHAT_BACKEND", "ollama")
    monkeypatch.setattr(config, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 4096)
    monkeypatch.setattr(config, "OLLAMA_THINK", False)
    monkeypatch.setattr(config, "OLLAMA_TIMEOUT", 5)
    monkeypatch.setattr(chat, "_release", lambda: None)
    monkeypatch.setattr("tgbot.diffuse.unload", lambda: None)
    monkeypatch.setattr("tgbot.progress.report", lambda *a, **k: None)

    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_active_model_uses_configured_backend():
    assert chat.active_model() == config.CHAT_MODEL


def test_reply_uses_ollama(ollama):
    answer, seconds = chat.reply("hi")
    assert answer == "Hello"
    assert seconds >= 0

    path, body = ollama.requests[-1]
    assert path == "/api/chat"
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is True
    assert body["think"] is False
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["num_predict"] == config.CHAT_MAX_NEW_TOKENS
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][-1] == {"role": "user", "content": "hi"}


def test_reply_attaches_base64_images_to_the_new_turn(ollama):
    chat.reply("look", images=["YWJj"])
    _, body = ollama.requests[-1]
    assert body["messages"][-1] == {
        "role": "user",
        "content": "look",
        "images": ["YWJj"],
    }


def test_stream_reply_uses_ollama(ollama):
    chunks = list(chat.stream_reply("hi", system_prompt=chat.WEB_SYSTEM_PROMPT))
    assert chunks == ["Hel", "lo"]

    _, body = ollama.requests[-1]
    assert body["stream"] is True
    assert body["messages"][0]["content"] == chat.WEB_SYSTEM_PROMPT
    assert body["options"]["num_ctx"] == 4096


def test_stream_reply_stops_on_event(ollama):
    stop = threading.Event()
    chunks = []
    for chunk in chat.stream_reply("hi", stop_event=stop):
        chunks.append(chunk)
        stop.set()
    assert chunks == ["Hel"]


def test_trim_for_context_drops_old_turns(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 4096)
    messages = [{"role": "system", "content": "system"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"old user {i} " + "x" * 500})
        messages.append({"role": "assistant", "content": f"old answer {i} " + "y" * 500})
    current = {"role": "user", "content": "write the complete file"}
    messages.append(current)

    trimmed = chat._trim_for_context(messages, 2048)

    assert trimmed[0] == messages[0]
    assert trimmed[-1] == current
    assert len(trimmed) < len(messages)
    assert trimmed[1]["role"] == "user"


def test_ollama_http_error_is_reported(ollama, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_MODEL", "missing")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        chat.reply("hi")


def test_ollama_unreachable_is_reported(monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "ollama")
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(chat, "_release", lambda: None)
    monkeypatch.setattr("tgbot.diffuse.unload", lambda: None)
    with pytest.raises(RuntimeError, match="not reachable"):
        chat.reply("hi")


def test_is_available_checks_the_model(ollama, monkeypatch):
    assert chat.is_available() is True
    monkeypatch.setattr(config, "OLLAMA_MODEL", "missing")
    assert chat.is_available() is False


def test_unload_evicts_the_ollama_model(ollama):
    chat.unload()
    path, body = ollama.requests[-1]
    assert path == "/api/generate"
    assert body["keep_alive"] == 0
