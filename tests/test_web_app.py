import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp.test_utils import TestClient, TestServer

from web import app as web_app
from web import service as web_service
from web.store import WebStore


def _fake_stream(*args, **kwargs):
    yield "Hel"
    yield "lo"


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
        self.assertEqual(data["model"], web_service.config.CHAT_MODEL)
        self.assertIn("presets", data)

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
