import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"


def test_shell_keeps_vertex_tokens_and_layout():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "--ac:#0d9488" in html
    assert ".comp{" in html
    assert ".dock{" in html
    assert "body:not(.chatting) .dock" in html
    assert 'id="app"' in html
    assert 'id="in"' in html
    assert 'id="send"' in html
    assert "Space Grotesk" in html
    assert 'id="modeltag"' in html


def test_fonts_are_vendored_not_remote():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in html
    assert "/static/fonts.css" in html
    assert (STATIC / "fonts.css").exists()
    for name in (
        "inter-latin.woff2",
        "inter-latin-ext.woff2",
        "space-grotesk-latin.woff2",
        "space-grotesk-latin-ext.woff2",
    ):
        assert (STATIC / "fonts" / name).stat().st_size > 1000


def test_demo_handlers_and_canned_data_are_gone():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-t="' not in html
    assert "Search is a demo here" not in html
    assert "demo here" not in html
    assert "Share</button>" in html


def test_attachments_and_tools_are_wired_to_real_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="file"' in html
    assert 'id="attachments"' in html
    assert 'id="toolsmenu"' in html
    assert 'data-tool="describe"' in html
    assert 'data-tool="removebg"' in html
    assert "Text-only build" not in js
    assert "/api/uploads" in js
    assert "addFiles" in js
    assert "finalizeStream(el, data.content || state.streamRaw, data.message_id, data.attachments || [])" in js


def test_app_js_keeps_shell_behaviour():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "document.body.classList.toggle('chatting'" in js
    assert "app.classList.toggle('chatting'" in js
    assert "if (innerWidth <= 800) app.classList.add('closed')" in js
    assert "function grow()" in js
    assert "function md(s)" in js
    assert "function reset(" in js
    assert "/api/sessions" in js
    assert "readSSE" in js
    assert "regenerate" in js
    assert "$('#chips')" in js
    assert "chip.dataset.p" in js
    assert "$('#modeltag')" in js
