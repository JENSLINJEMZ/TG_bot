"""aiohttp application for the local Vertex web chat UI."""

from __future__ import annotations

import hmac
import json
import logging
from pathlib import Path

from aiohttp import web

from tgbot import config

from .service import (
    DEFAULT_PRESET,
    BusyError,
    ChatService,
    DisabledError,
    normalize_preset,
)
from .store import WebStore

log = logging.getLogger("web.app")

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
INDEX = STATIC / "index.html"

STORE_KEY = web.AppKey("store", WebStore)
SERVICE_KEY = web.AppKey("service", ChatService)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _json(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="invalid JSON body")
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="expected a JSON object")
    return data


@web.middleware
async def guard_middleware(request: web.Request, handler):
    """Reject cross-site browser requests and enforce the optional token."""
    if request.path.startswith("/api/"):
        # A malicious page can still POST to 127.0.0.1; browsers mark that as
        # cross-site. Same-origin fetch and non-browser clients are allowed.
        site = request.headers.get("Sec-Fetch-Site")
        if site and site not in ("same-origin", "none"):
            raise web.HTTPForbidden(text="cross-site request rejected")
        if config.WEB_TOKEN:
            supplied = (
                request.headers.get("X-Web-Token")
                or request.query.get("token")
                or ""
            )
            if not hmac.compare_digest(supplied, config.WEB_TOKEN):
                raise web.HTTPUnauthorized(text="missing or invalid web token")
    return await handler(request)


@web.middleware
async def errors_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except BusyError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except DisabledError as exc:
        return web.json_response({"error": str(exc)}, status=503)
    except KeyError:
        return web.json_response({"error": "not found"}, status=404)
    except Exception:
        log.exception("unhandled error on %s %s", request.method, request.path)
        return web.json_response({"error": "internal server error"}, status=500)


# -- pages ----------------------------------------------------------------
async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(INDEX, headers={"Cache-Control": "no-cache"})


async def health(request: web.Request) -> web.Response:
    return web.json_response(request.app[SERVICE_KEY].health())


async def api_not_found(request: web.Request) -> web.Response:
    return web.json_response({"error": "unknown endpoint"}, status=404)


# -- sessions -------------------------------------------------------------
async def list_sessions(request: web.Request) -> web.Response:
    q = (request.query.get("q") or "").strip() or None
    return web.json_response({"sessions": request.app[STORE_KEY].list_sessions(q)})


async def create_session(request: web.Request) -> web.Response:
    data = await _json(request)
    preset = normalize_preset(str(data.get("preset") or DEFAULT_PRESET))
    title = str(data.get("title") or "New chat")
    session = request.app[STORE_KEY].create_session(title=title, preset=preset)
    return web.json_response({"session": session}, status=201)


async def get_session(request: web.Request) -> web.Response:
    session = request.app[STORE_KEY].get_session(
        request.match_info["sid"], include_messages=True
    )
    if session is None:
        raise web.HTTPNotFound(text="session not found")
    return web.json_response({"session": session})


async def patch_session(request: web.Request) -> web.Response:
    store: WebStore = request.app[STORE_KEY]
    sid = request.match_info["sid"]
    session = store.get_session(sid)
    if session is None:
        raise web.HTTPNotFound(text="session not found")
    data = await _json(request)
    if "title" in data:
        updated = store.rename_session(sid, str(data["title"]))
        if updated is None:
            raise web.HTTPBadRequest(text="title is required")
        session = updated
    if "preset" in data:
        preset = normalize_preset(str(data["preset"]))
        updated = store.set_preset(sid, preset)
        if updated is None:
            raise web.HTTPNotFound(text="session not found")
        session = updated
    return web.json_response({"session": session})


async def delete_session(request: web.Request) -> web.Response:
    if not request.app[STORE_KEY].delete_session(request.match_info["sid"]):
        raise web.HTTPNotFound(text="session not found")
    return web.json_response({"ok": True})


async def restore_session(request: web.Request) -> web.Response:
    if not request.app[STORE_KEY].restore_session(request.match_info["sid"]):
        raise web.HTTPNotFound(text="session not found")
    return web.json_response({"ok": True})


async def get_messages(request: web.Request) -> web.Response:
    store: WebStore = request.app[STORE_KEY]
    sid = request.match_info["sid"]
    if store.get_session(sid) is None:
        raise web.HTTPNotFound(text="session not found")
    return web.json_response({"messages": store.get_messages(sid)})


# -- chat -----------------------------------------------------------------
async def post_message(request: web.Request) -> web.StreamResponse:
    store: WebStore = request.app[STORE_KEY]
    service: ChatService = request.app[SERVICE_KEY]
    sid = request.match_info["sid"]

    if store.get_session(sid) is None:
        raise web.HTTPNotFound(text="session not found")
    if not config.CHAT_ENABLED:
        raise DisabledError("chat is disabled (CHAT_ENABLED=0)")

    data = await _json(request)
    regenerate = bool(data.get("regenerate"))
    content = str(data.get("content") or "").strip()
    if not regenerate and not content:
        raise web.HTTPBadRequest(text="content is required")
    if len(content) > 8000:
        raise web.HTTPRequestEntityTooLarge(max_size=8000, actual_size=len(content))

    preset = normalize_preset(str(data.get("preset") or DEFAULT_PRESET))
    client_id = data.get("client_id")
    client_id = str(client_id)[:64] if client_id else None

    # Reserve the model *before* the response is prepared, so a second request
    # gets a real 409 instead of an SSE stream that immediately fails.
    await service.acquire()

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    stream = None
    try:
        await response.prepare(request)
        stream = service.stream(
            sid,
            content,
            preset=preset,
            client_id=client_id,
            regenerate=regenerate,
        )
        try:
            async for event, payload in stream:
                await response.write(_sse(event, payload))
        except (ConnectionResetError, BrokenPipeError):
            raise
        except Exception as exc:
            if isinstance(exc, (BusyError, DisabledError, KeyError, ValueError)):
                message = str(exc)
            else:
                log.exception("chat stream failed")
                message = "the model failed to finish this reply"
            await response.write(_sse("error", {"error": message}))
        await response.write(b": done\n\n")
        return response
    except (ConnectionResetError, BrokenPipeError):
        # The browser closed the tab; the stream's finally block stops the model.
        return response
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except Exception:
                pass
        service.release()


# -- app ------------------------------------------------------------------
def create_app(
    store: WebStore | None = None, service: ChatService | None = None
) -> web.Application:
    store = store or WebStore(config.WEB_STORE_PATH)
    service = service or ChatService(store)

    app = web.Application(
        middlewares=[errors_middleware, guard_middleware],
        client_max_size=1024 * 1024,
    )
    app[STORE_KEY] = store
    app[SERVICE_KEY] = service

    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/sessions", list_sessions)
    app.router.add_post("/api/sessions", create_session)
    app.router.add_get("/api/sessions/{sid}", get_session)
    app.router.add_patch("/api/sessions/{sid}", patch_session)
    app.router.add_delete("/api/sessions/{sid}", delete_session)
    app.router.add_post("/api/sessions/{sid}/restore", restore_session)
    app.router.add_get("/api/sessions/{sid}/messages", get_messages)
    app.router.add_post("/api/sessions/{sid}/messages", post_message)
    app.router.add_route("*", "/api/{tail:.*}", api_not_found)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app
