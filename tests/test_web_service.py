import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web import service as web_service
from web.store import WebStore


def _fake_stream(*args, **kwargs):
    yield "Hel"
    yield "lo"


def test_presets_allow_long_code_replies():
    assert web_service.PRESETS["core"]["max_new_tokens"] >= 8000
    assert web_service.PRESETS["max"]["max_new_tokens"] >= 12000
    assert web_service.PRESETS["lite"]["max_new_tokens"] >= 2000
    assert web_service.MAX_GENERATION_SECONDS >= 900


def test_stream_events_and_persistence(tmp_path):
    async def scenario():
        store = WebStore(tmp_path / "web.db")
        service = web_service.ChatService(store)
        session = store.create_session()
        with patch.object(web_service.chat_engine, "stream_reply", _fake_stream):
            await service.acquire()
            try:
                events = [
                    event
                    async for event in service.stream(
                        session["id"], "hi", preset="core", client_id="c1"
                    )
                ]
            finally:
                service.release()

        assert [e[0] for e in events] == ["meta", "token", "token", "done"]
        assert events[-1][1]["content"] == "Hello"
        messages = store.get_messages(session["id"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert store.get_session(session["id"])["title"] == "hi"

    asyncio.run(scenario())


def test_client_id_replays_saved_turn(tmp_path):
    async def scenario():
        store = WebStore(tmp_path / "web.db")
        service = web_service.ChatService(store)
        session = store.create_session()
        with patch.object(web_service.chat_engine, "stream_reply", _fake_stream):
            await service.acquire()
            try:
                [e async for e in service.stream(session["id"], "hi", client_id="c1")]
            finally:
                service.release()

            await service.acquire()
            try:
                replay = [
                    e async for e in service.stream(session["id"], "hi", client_id="c1")
                ]
            finally:
                service.release()

        assert [e[0] for e in replay] == ["meta", "token", "done"]
        assert replay[-1][1]["replayed"] is True
        assert len(store.get_messages(session["id"])) == 2

    asyncio.run(scenario())


def test_acquire_is_busy_while_held(tmp_path):
    async def scenario():
        service = web_service.ChatService(WebStore(tmp_path / "web.db"))
        await service.acquire()
        try:
            assert service.busy is True
            with pytest.raises(web_service.BusyError):
                await service.acquire()
        finally:
            service.release()
        assert service.busy is False

    asyncio.run(scenario())


def test_disabled_chat_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(web_service.config, "CHAT_ENABLED", False)

    async def scenario():
        store = WebStore(tmp_path / "web.db")
        service = web_service.ChatService(store)
        session = store.create_session()
        await service.acquire()
        try:
            with pytest.raises(web_service.DisabledError):
                [e async for e in service.stream(session["id"], "hi")]
        finally:
            service.release()

    asyncio.run(scenario())


def test_error_keeps_partial_answer(tmp_path):
    def broken(*args, **kwargs):
        yield "partial"
        raise RuntimeError("model died")

    async def scenario():
        store = WebStore(tmp_path / "web.db")
        service = web_service.ChatService(store)
        session = store.create_session()
        with patch.object(web_service.chat_engine, "stream_reply", broken):
            await service.acquire()
            try:
                with pytest.raises(RuntimeError):
                    [e async for e in service.stream(session["id"], "hi")]
            finally:
                service.release()

        messages = store.get_messages(session["id"])
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "partial"
        assert messages[-1]["status"] == "error"

    asyncio.run(scenario())


def test_regenerate_replaces_last_assistant(tmp_path):
    async def scenario():
        store = WebStore(tmp_path / "web.db")
        service = web_service.ChatService(store)
        session = store.create_session()
        with patch.object(web_service.chat_engine, "stream_reply", _fake_stream):
            await service.acquire()
            try:
                [e async for e in service.stream(session["id"], "hi")]
            finally:
                service.release()

            await service.acquire()
            try:
                events = [
                    e
                    async for e in service.stream(
                        session["id"], "", regenerate=True, preset="lite"
                    )
                ]
            finally:
                service.release()

        assert events[-1][0] == "done"
        messages = store.get_messages(session["id"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[-1]["content"] == "Hello"

    asyncio.run(scenario())


def test_title_for_matches_vertex_rule():
    assert web_service.title_for("hi") == "hi"
    long = "x" * 40
    assert web_service.title_for(long) == "x" * 34 + "…"
    assert web_service.title_for("  a\n b  ") == "a b"
    assert web_service.title_for("") == "New chat"


def test_normalize_preset():
    assert web_service.normalize_preset("max") == "max"
    assert web_service.normalize_preset("nope") == "core"
    assert web_service.normalize_preset(None) == "core"
