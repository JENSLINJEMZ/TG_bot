import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from web import app as web_app
from web import service as web_service
from web.store import WebStore


def _fake_stream(*args, **kwargs):
    yield "Hel"
    yield "lo"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (12, 148, 136)).save(buffer, format="PNG")
    return buffer.getvalue()


class WebAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = WebStore(Path(self.tmp.name) / "web.db")
        self.service = web_service.ChatService(self.store)
        self.app = web_app.create_app(store=self.store, service=self.service)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def _new_session(self, title="Test chat", preset="core"):
        response = await self.client.post(
            "/api/sessions", json={"title": title, "preset": preset}
        )
        self.assertEqual(response.status, 201)
        return (await response.json())["session"]

    async def test_health(self):
        response = await self.client.get("/api/health")
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"], web_service.chat_engine.active_model())
        self.assertIn("presets", data)

    async def test_health_reports_the_active_backend_model(self):
        with patch.object(
            web_service.chat_engine, "active_model", return_value="qwen3:8b"
        ):
            response = await self.client.get("/api/health")
            data = await response.json()
        self.assertEqual(data["model"], "qwen3:8b")

    async def test_index_and_static(self):
        response = await self.client.get("/")
        self.assertEqual(response.status, 200)
        self.assertIn("text/html", response.headers["Content-Type"])
        text = await response.text()
        self.assertIn('id="app"', text)
        self.assertIn("Vertex", text)

        response = await self.client.get("/static/app.js")
        self.assertEqual(response.status, 200)
        self.assertIn("runStream", await response.text())

    async def test_session_crud(self):
        session = await self._new_session("Test chat", "max")
        sid = session["id"]
        self.assertEqual(session["title"], "Test chat")
        self.assertEqual(session["preset"], "max")

        response = await self.client.get(f"/api/sessions/{sid}")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["session"]["messages"], [])

        response = await self.client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["session"]["title"], "Renamed")

        response = await self.client.delete(f"/api/sessions/{sid}")
        self.assertEqual(response.status, 200)
        response = await self.client.get(f"/api/sessions/{sid}")
        self.assertEqual(response.status, 404)

        response = await self.client.post(f"/api/sessions/{sid}/restore")
        self.assertEqual(response.status, 200)
        response = await self.client.get(f"/api/sessions/{sid}")
        self.assertEqual(response.status, 200)

    async def test_message_stream(self):
        session = await self._new_session()
        with patch.object(web_service.chat_engine, "stream_reply", _fake_stream):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "hi", "client_id": "c1"},
            )
            self.assertEqual(response.status, 200)
            self.assertIn("text/event-stream", response.headers["Content-Type"])
            self.assertIn("no-cache", response.headers["Cache-Control"])
            text = await response.text()

        self.assertIn("event: meta", text)
        self.assertIn("event: token", text)
        self.assertIn("event: done", text)
        self.assertIn("Hello", text)
        messages = self.store.get_messages(session["id"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["content"], "Hello")

    async def _upload(self, data: bytes, filename: str, content_type: str) -> dict:
        form = FormData()
        form.add_field("file", data, filename=filename, content_type=content_type)
        response = await self.client.post("/api/uploads", data=form)
        self.assertEqual(response.status, 201, await response.text())
        return (await response.json())["upload"]

    async def test_upload_image_serves_bytes_and_delete(self):
        png = _png_bytes()
        upload = await self._upload(png, "dot.png", "image/png")
        self.assertEqual(upload["kind"], "image")
        self.assertEqual(upload["width"], 4)
        self.assertEqual(upload["height"], 4)

        response = await self.client.get(upload["url"])
        self.assertEqual(response.status, 200)
        self.assertIn("image/png", response.headers["Content-Type"])
        self.assertEqual(await response.read(), png)

        response = await self.client.delete(f"/api/uploads/{upload['id']}")
        self.assertEqual(response.status, 200)
        response = await self.client.get(upload["url"])
        self.assertEqual(response.status, 404)

    async def test_uploaded_html_is_served_as_plain_text(self):
        html = b"<script>alert(1)</script>"
        upload = await self._upload(html, "note.html", "text/html")
        response = await self.client.get(upload["url"])
        self.assertEqual(response.status, 200)
        self.assertTrue(response.headers["Content-Type"].startswith("text/plain"))
        self.assertEqual(await response.read(), html)

    async def test_text_attachment_is_embedded_in_the_prompt(self):
        upload = await self._upload(b"alpha\nbeta\n", "notes.txt", "text/plain")
        session = await self._new_session()
        captured = {}

        def fake_stream(text, history=None, **kwargs):
            captured["text"] = text
            captured["images"] = kwargs.get("images")
            yield "ok"

        with patch.object(web_service.chat_engine, "stream_reply", fake_stream):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "what is this?", "attachments": [upload["id"]]},
            )
            self.assertEqual(response.status, 200)
            self.assertIn("event: done", await response.text())

        self.assertIn("alpha", captured["text"])
        self.assertFalse(captured["images"])
        messages = self.store.get_messages(session["id"])
        self.assertEqual(messages[0]["attachments"][0]["id"], upload["id"])
        self.assertIsNone(messages[0]["tool"])

    async def test_image_attachment_is_sent_to_the_vision_model(self):
        png = _png_bytes()
        upload = await self._upload(png, "dot.png", "image/png")
        session = await self._new_session()
        captured = {}

        def fake_stream(text, history=None, **kwargs):
            captured["images"] = kwargs.get("images") or []
            yield "seen"

        with (
            patch.object(web_service.config, "CHAT_BACKEND", "ollama"),
            patch.object(web_service.chat_engine, "stream_reply", fake_stream),
        ):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"attachments": [upload["id"]]},
            )
            self.assertEqual(response.status, 200)
            await response.text()

        self.assertEqual(len(captured["images"]), 1)
        self.assertEqual(base64.b64decode(captured["images"][0]), png)
        messages = self.store.get_messages(session["id"])
        self.assertEqual(messages[0]["content"], "Describe the attached image")

    async def test_calculator_tool_answers_without_the_model(self):
        session = await self._new_session()
        with patch.object(
            web_service.chat_engine,
            "stream_reply",
            side_effect=AssertionError("calculator should not call the model"),
        ):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "12 * (4 + 3)", "tool": "calc"},
            )
            self.assertEqual(response.status, 200)
            text = await response.text()

        self.assertIn("event: done", text)
        self.assertIn("84", text)
        messages = self.store.get_messages(session["id"])
        self.assertEqual(messages[0]["tool"], "calc")
        self.assertEqual(messages[-1]["content"], "**12 * (4 + 3)** = 84")

    async def test_invalid_calculator_does_not_store_a_half_turn(self):
        session = await self._new_session()
        response = await self.client.post(
            f"/api/sessions/{session['id']}/messages",
            json={"content": "1 / 0", "tool": "calc"},
        )
        self.assertEqual(response.status, 200)
        text = await response.text()
        self.assertIn("event: error", text)
        self.assertIn("division by zero", text)
        self.assertEqual(self.store.get_messages(session["id"]), [])

    async def test_remove_background_tool_stores_an_image_result(self):
        png = _png_bytes()
        upload = await self._upload(png, "dot.png", "image/png")
        session = await self._new_session()
        with (
            patch.object(
                web_service.bgchange,
                "change_background",
                return_value=(png, "image/png"),
            ),
            patch.object(web_service.chat_engine, "unload"),
        ):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"content": "white", "tool": "removebg", "attachments": [upload["id"]]},
            )
            self.assertEqual(response.status, 200)
            text = await response.text()

        self.assertIn("event: done", text)
        messages = self.store.get_messages(session["id"])
        self.assertEqual(messages[0]["tool"], "removebg")
        self.assertEqual(messages[-1]["attachments"][0]["kind"], "image")
        self.assertIn("Removed the background", messages[-1]["content"])
        self.assertIn("#ffffff", messages[-1]["content"])

    async def test_attached_upload_cannot_be_deleted(self):
        upload = await self._upload(b"hello", "notes.txt", "text/plain")
        session = await self._new_session()
        with patch.object(web_service.chat_engine, "stream_reply", _fake_stream):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages",
                json={"attachments": [upload["id"]]},
            )
            self.assertEqual(response.status, 200)
            await response.text()
        response = await self.client.delete(f"/api/uploads/{upload['id']}")
        self.assertEqual(response.status, 409)

    async def test_busy_returns_409_before_stream(self):
        session = await self._new_session()
        await self.service.acquire()
        try:
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages", json={"content": "hi"}
            )
            self.assertEqual(response.status, 409)
            self.assertIn("busy", (await response.json())["error"])
        finally:
            self.service.release()

    async def test_disabled_returns_503(self):
        session = await self._new_session()
        with patch.object(web_app.config, "CHAT_ENABLED", False):
            response = await self.client.post(
                f"/api/sessions/{session['id']}/messages", json={"content": "hi"}
            )
        self.assertEqual(response.status, 503)

    async def test_unknown_api_is_json(self):
        response = await self.client.get("/api/nope")
        self.assertEqual(response.status, 404)
        self.assertEqual((await response.json())["error"], "unknown endpoint")

    async def test_token_guard(self):
        with patch.object(web_app.config, "WEB_TOKEN", "secret"):
            response = await self.client.get("/api/health")
            self.assertEqual(response.status, 401)
            response = await self.client.get(
                "/api/health", headers={"X-Web-Token": "secret"}
            )
            self.assertEqual(response.status, 200)

    async def test_cross_site_guard(self):
        response = await self.client.get(
            "/api/health", headers={"Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(response.status, 403)
