#!/usr/bin/env python3
"""Cross-platform installer for the Telegram bot.

Detects Linux / macOS / Windows, installs `uv` if it is missing, installs a
compatible Python and all project dependencies, creates `.env` from the
template, and can optionally prefetch the local models and run the tests.

Quick start (any OS, Python 3.8+ on PATH):
    python install.py

With no Python at all, use the OS wrapper instead:
    Linux/macOS:  ./install.sh
    Windows:      powershell -ExecutionPolicy Bypass -File install.ps1

Only the standard library is used, so this file runs before any dependency is
installed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = "3.12"


def log(message: str) -> None:
    print(f"[install] {message}", flush=True)


def detect_os() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    exe = "uv.exe" if os.name == "nt" else "uv"
    home = Path.home()
    for candidate in (
        home / ".local" / "bin" / exe,
        home / ".cargo" / "bin" / exe,
        Path(os.environ.get("LOCALAPPDATA", home)) / "uv" / exe,
    ):
        if candidate.exists():
            return str(candidate)
    return None


def install_uv(host_os: str) -> str:
    log("uv not found - installing it...")
    rc = 1
    if host_os == "windows":
        command = "irm https://astral.sh/uv/install.ps1 | iex"
        rc = subprocess.call(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
        )
    elif shutil.which("curl"):
        rc = subprocess.call("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
    elif shutil.which("wget"):
        rc = subprocess.call("wget -qO- https://astral.sh/uv/install.sh | sh", shell=True)
    else:
        log("neither curl nor wget found")

    uv = find_uv()
    if rc != 0 or uv is None:
        log("official installer did not work - trying pip")
        subprocess.call([sys.executable, "-m", "pip", "install", "--user", "uv"])
        uv = find_uv()
    if uv is None:
        raise SystemExit(
            "Could not install uv automatically.\n"
            "Install it manually from https://docs.astral.sh/uv/ and re-run."
        )
    return uv


def run(command: list[str]) -> int:
    log("$ " + " ".join(command))
    return subprocess.call(command)


def ensure_env_file() -> None:
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if env.exists():
        log(".env already exists - leaving it untouched")
    elif example.exists():
        shutil.copyfile(example, env)
        log("created .env from .env.example - add your TELEGRAM_BOT_TOKEN")
    else:
        log("warning: no .env.example to copy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the Telegram bot and its dependencies.")
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help=f"Python version for the project (default {DEFAULT_PYTHON}; use 'system' to skip pinning)",
    )
    parser.add_argument("--no-dev", action="store_true", help="skip the dev group (pytest)")
    parser.add_argument("--with-models", action="store_true", help="pre-download the local models (~1.5 GB)")
    parser.add_argument("--test", action="store_true", help="run the test suite after install")
    args = parser.parse_args()

    host_os = detect_os()
    log(f"detected OS: {host_os} ({platform.machine()}), host python {platform.python_version()}")
    log(f"project: {ROOT}")

    uv = find_uv() or install_uv(host_os)
    log(f"using uv: {uv}")

    pin = None if args.python in ("", "system", "none") else args.python
    if pin:
        if run([uv, "python", "install", pin]) != 0:
            log(f"could not install python {pin} - continuing with uv's default")

    sync = [uv, "sync"]
    if pin:
        sync += ["--python", pin]
    if args.no_dev:
        sync.append("--no-dev")
    if run(sync) != 0:
        raise SystemExit("dependency installation failed (uv sync)")

    ensure_env_file()

    if args.with_models:
        log("prefetching local models...")
        run([uv, "run", "python", "-m", "tgbot.prefetch"])

    if args.test:
        log("running tests...")
        run([uv, "run", "pytest", "-q"])

    print()
    log("installation complete.")
    print(
        "\nNext steps:\n"
        f"  1. edit {ROOT / '.env'} and set TELEGRAM_BOT_TOKEN (and Supabase keys if you use storage)\n"
        "  2. start the bot:   uv run python -m tgbot.bot\n"
        + ("" if args.with_models else
           "  3. models download automatically on first use, or pre-download now:\n"
           "       uv run python -m tgbot.prefetch\n")
    )


if __name__ == "__main__":
    main()