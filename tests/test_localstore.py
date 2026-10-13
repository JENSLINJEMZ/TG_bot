import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot.localstore import LocalStore


def test_save_and_read_media_roundtrip(tmp_path):
    store = LocalStore(tmp_path / "store.db")
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
    rows = store.all_media()
    assert len(rows) == 1
    row = rows[0]
    assert row.original_blob == b"orig"
    assert row.result_blob == b"res"
    assert row.aux_blob is None
    assert row.prompt == "navy suit"
    assert store.counts() == (1, 0)


def test_save_media_with_aux(tmp_path):
    store = LocalStore(tmp_path / "store.db")
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
    row = store.all_media()[0]
    assert row.aux_blob == b"b"
    assert row.aux_path == "inputs/1/2/b.jpg"


def test_save_and_read_chat(tmp_path):
    store = LocalStore(tmp_path / "store.db")
    store.save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="hello", elapsed_ms=5)
    rows = store.all_chats()
    assert len(rows) == 1
    assert rows[0].prompt == "hi"
    assert rows[0].reply == "hello"
    assert store.counts() == (0, 1)


def test_get_and_latest_media(tmp_path):
    store = LocalStore(tmp_path / "store.db")
    first = store.save_media(
        mode="bg",
        chat_id=1,
        user_id=2,
        message_id=None,
        prompt=None,
        background_rgb=None,
        source_file_id=None,
        elapsed_ms=None,
        original_path="o1.jpg",
        original_blob=b"o1",
        original_type="image/jpeg",
        result_path="r1.jpg",
        result_blob=b"r1",
        result_type="image/jpeg",
    )
    second = store.save_media(
        mode="bg",
        chat_id=1,
        user_id=2,
        message_id=None,
        prompt=None,
        background_rgb=None,
        source_file_id=None,
        elapsed_ms=None,
        original_path="o2.jpg",
        original_blob=b"o2",
        original_type="image/jpeg",
        result_path="r2.jpg",
        result_blob=b"r2",
        result_type="image/jpeg",
    )
    assert store.get_media(first).result_blob == b"r1"
    assert store.latest_media(2).id == second
    assert store.latest_media(999) is None
    assert store.count_media(2) == 2


def test_legacy_outbox_tables_are_migrated(tmp_path):
    """A legacy outbox database must not lose its rows on the table rename."""
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE pending_media (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                mode TEXT, chat_id INTEGER, user_id INTEGER, message_id INTEGER,
                prompt TEXT, background_rgb TEXT, source_file_id TEXT, elapsed_ms INTEGER,
                original_path TEXT, original_blob BLOB, original_type TEXT,
                aux_path TEXT, aux_blob BLOB, aux_type TEXT,
                result_path TEXT, result_blob BLOB, result_type TEXT, sync_error TEXT
            );
            CREATE TABLE pending_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                chat_id INTEGER, user_id INTEGER, message_id INTEGER,
                prompt TEXT, reply TEXT, elapsed_ms INTEGER, sync_error TEXT
            );
            INSERT INTO pending_media
                (created_at, mode, chat_id, user_id, original_path, original_blob,
                 original_type, result_path, result_blob, result_type)
            VALUES ('2026-01-01', 'bg', 1, 2, 'o.jpg', X'01', 'image/jpeg',
                    'r.jpg', X'02', 'image/jpeg');
            """
        )
    store = LocalStore(path)
    rows = store.all_media()
    assert len(rows) == 1
    assert rows[0].original_blob == b"\x01"
    assert store.counts() == (1, 0)
