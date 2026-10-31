"""aiohttp application for the local Vertex web chat UI."""

from __future__ import annotations

import hmac
import io
import json
import logging
from pathlib import Path
from urllib.parse import quote

from aiohttp import web

from tgbot import config

from .service import (
    DEFAULT_PRESET,
    BusyError,
    ChatService,
    DisabledError,
    normalize_preset,
    normalize_tool,
)
from .store import WebStore
from .uploads import guess_kind, mime_for, public_upload, text_excerpt

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


# -- uploads --------------------------------------------------------------
async def upload_attachment(request: web.Request) -> web.Response:
    store: WebStore = request.app[STORE_KEY]
    if not (request.content_type or "").startswith("multipart/"):
        raise web.HTTPUnsupportedMediaType(text="send a multipart file upload")
    reader = await request.multipart()
    part = None
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            break
    if part is None or part.name != "file":
        raise web.HTTPBadRequest(text="a file field is required")

    filename = Path(part.filename or "attachment").name[:200] or "attachment"
    mime = mime_for(filename, part.headers.get("Content-Type", ""))
    kind = guess_kind(filename, mime)
    if kind is None:
        raise web.HTTPUnsupportedMediaType(
            text="attach an image (PNG/JPEG/WebP) or a UTF-8 text/code file"
        )

    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await part.read_chunk(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > config.WEB_MAX_UPLOAD_BYTES:
            raise web.HTTPRequestEntityTooLarge(
                max_size=config.WEB_MAX_UPLOAD_BYTES, actual_size=size
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise web.HTTPBadRequest(text="the uploaded file is empty")

    text_content: str | None = None
    width: int | None = None
    height: int | None = None
    if kind == "image":
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
        except Exception:
            raise web.HTTPUnsupportedMediaType(text="that image could not be read")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise web.HTTPUnsupportedMediaType(text="text files must be UTF-8")
        if "\x00" in text:
            raise web.HTTPUnsupportedMediaType(text="that does not look like a text file")
        text_content = text_excerpt(text, config.WEB_ATTACHMENT_TEXT_LIMIT)

    row = store.add_upload(
        filename=filename,
        mime=mime,
        kind=kind,
        data=data,
        text_content=text_content,
        width=width,
        height=height,
    )
    return web.json_response({"upload": public_upload(row)}, status=201)


async def get_upload_content(request: web.Request) -> web.FileResponse:
    store: WebStore = request.app[STORE_KEY]
    row = store.get_upload(request.match_info["uid"])
    if row is None:
        raise web.HTTPNotFound(text="attachment not found")
    try:
        path = store.upload_path(row)
    except ValueError:
        raise web.HTTPNotFound(text="attachment not found")
    if not path.is_file():
        raise web.HTTPNotFound(text="attachment not found")
    safe_name = quote(str(row.get("filename") or "attachment"), safe="")
    if row.get("kind") == "image":
        mime = str(row.get("mime") or "")
        if not mime.startswith("image/"):
            guessed = mime_for(str(row.get("filename") or ""), "")
            mime = guessed if guessed.startswith("image/") else "application/octet-stream"
    else:
        # Never render an uploaded HTML/JS file as active content.
        mime = "text/plain; charset=utf-8"
    return web.FileResponse(
        path,
        headers={
            "Content-Type": mime,
            "Content-Disposition": f"inline; filename*=UTF-8''{safe_name}",
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def delete_upload(request: web.Request) -> web.Response:
    store: WebStore = request.app[STORE_KEY]
    row = store.get_upload(request.match_info["uid"])
    if row is None:
        raise web.HTTPNotFound(text="attachment not found")
    if row.get("session_id"):
        raise web.HTTPConflict(text="that attachment is already in a chat")
    store.delete_upload(request.match_info["uid"])
    return web.json_response({"ok": True})


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
    raw_attachments = data.get("attachments") or []
    if not isinstance(raw_attachments, list):
        raise web.HTTPBadRequest(text="attachments must be a list")
    attachment_ids = [
        str(item)[:64] for item in raw_attachments if str(item).strip()
    ][: config.WEB_MAX_ATTACHMENTS]
    tool = normalize_tool(str(data.get("tool") or ""))
    if not regenerate and not content and not attachment_ids and tool != "time":
        raise web.HTTPBadRequest(text="content or an attachment is required")
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
            attachments=attachment_ids,
            tool=tool,
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
    store = store or WebStore(
        config.WEB_STORE_PATH, upload_dir=config.WEB_UPLOAD_DIR
    )
    service = service or ChatService(store)

    app = web.Application(
        middlewares=[errors_middleware, guard_middleware],
        client_max_size=config.WEB_MAX_UPLOAD_BYTES + 1024 * 1024,
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
    app.router.add_post("/api/uploads", upload_attachment)
    app.router.add_get("/api/uploads/{uid}/content", get_upload_content)
    app.router.add_delete("/api/uploads/{uid}", delete_upload)
    app.router.add_route("*", "/api/{tail:.*}", api_not_found)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app
