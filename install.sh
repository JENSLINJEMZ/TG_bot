#!/usr/bin/env bash
# Installer bootstrap for Linux and macOS.
# Ensures `uv` exists, then hands off to install.py (which also works on Windows).
#
#   ./install.sh                # install dependencies
#   ./install.sh --with-models  # also pre-download the local models
#   ./install.sh --test         # also run the test suite
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -s)" in
    Linux*)  os=Linux ;;
    Darwin*) os=macOS ;;
    *)       os="$(uname -s)" ;;
esac
echo "[install] detected OS: $os ($(uname -m))"

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! uv_path="$(find_uv)"; then
    echo "[install] uv not found - installing it..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "[install] need curl or wget to install uv" >&2
        exit 1
    fi
    uv_path="$(find_uv)" || { echo "[install] uv install failed" >&2; exit 1; }
fi

echo "[install] using uv: $uv_path"
exec "$uv_path" run --no-project python install.py "$@"