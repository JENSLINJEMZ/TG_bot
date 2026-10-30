"""``python -m web`` entry point."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

from aiohttp import web

from tgbot import config

from .app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m web",
        description="Local Vertex web chat UI (aiohttp + SmolLM2).",
    )
    parser.add_argument("--host", default=config.WEB_HOST, help="bind address")
    parser.add_argument("--port", type=int, default=config.WEB_PORT, help="bind port")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}/"
    print(f"Vertex web chat  ->  {url}")
    print(f"sessions         ->  {config.WEB_STORE_PATH}")
    if config.WEB_TOKEN:
        print("token required   ->  append ?token=... or send X-Web-Token")
    print("The web app loads its own copy of the chat model; run it alongside the bot only if you have the RAM.")

    if not args.no_browser:
        from urllib.parse import quote

        open_url = url + (f"?token={quote(config.WEB_TOKEN)}" if config.WEB_TOKEN else "")
        threading.Timer(1.0, lambda: webbrowser.open(open_url)).start()

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
