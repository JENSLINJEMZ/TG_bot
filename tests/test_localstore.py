import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot.localstore import LocalStore


def test_save_and_read_media_roundtrip(tmp_path):
    store = LocalStore(tmp_path / "outbox.db")
    row_id = store.save_media(
        mode="restyle",
        chat_id=1,
        user_id=2,
        message_id=3,
        prompt="navy suit",
        background_rgb=json.dumps([10, 20, 30]),
        source_file_id="file",
        elapsed_ms=123,
        original_path="originals/1/2/a.jpg",
        original_blob=b"orig",
        original_type="image/jpeg",
        result_path="results/1/2/b.png",
        result_blob=b"res",
        result_type="image/png",
    )
    assert row_id == 1
    rows = store.pending_media()
    assert len(rows) == 1
    row = rows[0]
    assert row.original_blob == b"orig"
    assert row.result_blob == b"res"
    assert row.aux_blob is None
    assert row.prompt == "navy suit"
    assert store.counts() == (1, 0)


def test_save_media_with_aux(tmp_path):
    store = LocalStore(tmp_path / "outbox.db")
    store.save_media(
        mode="combine",
        chat_id=1,
        user_id=2,
        message_id=None,
        prompt="combine",
        background_rgb=None,
        source_file_id=None,
        elapsed_ms=None,
        original_path="originals/1/2/a.jpg",
        original_blob=b"a",
        original_type="image/jpeg",
        aux_path="inputs/1/2/b.jpg",
        aux_blob=b"b",
        aux_type="image/jpeg",
        result_path="results/1/2/c.jpg",
        result_blob=b"c",
        result_type="image/jpeg",
    )
    row = store.pending_media()[0]
    assert row.aux_blob == b"b"
    assert row.aux_path == "inputs/1/2/b.jpg"


def test_save_and_mark_chat(tmp_path):
    store = LocalStore(tmp_path / "outbox.db")
    store.save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="hello", elapsed_ms=5)
    rows = store.pending_chat()
    assert len(rows) == 1
    assert rows[0].prompt == "hi"
    assert rows[0].reply == "hello"
    store.mark_chat_synced(rows[0].id)
    assert store.pending_chat() == []
    assert store.counts() == (0, 0)


def test_mark_media_synced_removes_row(tmp_path):
    store = LocalStore(tmp_path / "outbox.db")
    row_id = store.save_media(
        mode="bg",
        chat_id=1,
        user_id=2,
        message_id=None,
        prompt=None,
        background_rgb=None,
        source_file_id=None,
        elapsed_ms=None,
        original_path="o.jpg",
        original_blob=b"o",
        original_type="image/jpeg",
        result_path="r.jpg",
        result_blob=b"r",
        result_type="image/jpeg",
    )
    store.mark_media_synced(row_id)
    assert store.pending_media() == []


def test_error_is_recorded(tmp_path):
    store = LocalStore(tmp_path / "outbox.db")
    row_id = store.save_chat(chat_id=1, user_id=2, message_id=None, prompt="x", reply="y", elapsed_ms=1)
    store.set_chat_error(row_id, "network down")
    assert store.pending_chat()[0].sync_error == "network down"