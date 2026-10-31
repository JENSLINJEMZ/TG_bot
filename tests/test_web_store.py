import sys
import threading
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.store import WebStore


def test_schema_reopen_is_idempotent(tmp_path):
    path = tmp_path / "web.db"
    WebStore(path)
    store = WebStore(path)
    session = store.create_session("Hello")
    assert store.get_session(session["id"])["title"] == "Hello"


def test_wal_and_busy_timeout(tmp_path):
    store = WebStore(tmp_path / "web.db")
    with closing(store._connect()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_session_list_rename_delete_restore(tmp_path):
    store = WebStore(tmp_path / "web.db")
    first = store.create_session("Alpha")
    second = store.create_session("Beta")
    assert [s["id"] for s in store.list_sessions()] == [second["id"], first["id"]]

    renamed = store.rename_session(first["id"], "Renamed")
    assert renamed["title"] == "Renamed"
    assert store.rename_session(first["id"], "   ") is None

    assert store.delete_session(first["id"]) is True
    assert store.get_session(first["id"]) is None
    assert [s["id"] for s in store.list_sessions()] == [second["id"]]

    assert store.restore_session(first["id"]) is True
    assert store.get_session(first["id"])["title"] == "Renamed"
    assert store.delete_session("missing") is False


def test_messages_order_and_history(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    store.add_message(session["id"], "user", "first")
    store.add_message(session["id"], "assistant", "one")
    second_user = store.add_message(session["id"], "user", "second")
    store.add_message(session["id"], "assistant", "two")

    messages = store.get_messages(session["id"])
    assert [m["content"] for m in messages] == ["first", "one", "second", "two"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]

    history = store.history_before(session["id"], second_user["id"], 4)
    assert [m["content"] for m in history] == ["first", "one"]
    assert store.history_before(session["id"], second_user["id"], 0) == []


def test_client_id_idempotency(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    user = store.add_message(session["id"], "user", "hi", client_id="c1")
    assistant = store.add_message(session["id"], "assistant", "hello")

    found = store.find_turn_by_client_id(session["id"], "c1")
    assert found["user"]["id"] == user["id"]
    assert found["assistant"]["id"] == assistant["id"]
    assert store.find_turn_by_client_id(session["id"], "other") is None

    # A duplicate insert returns the existing user row instead of raising.
    again = store.add_message(session["id"], "user", "hi", client_id="c1")
    assert again["id"] == user["id"]


def test_delete_last_assistant_and_last_user(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    store.add_message(session["id"], "user", "hi")
    store.add_message(session["id"], "assistant", "one")

    assert store.delete_last_assistant(session["id"]) is True
    assert store.last_assistant_message(session["id"]) is None
    assert store.last_user_message(session["id"])["content"] == "hi"
    assert store.delete_last_assistant(session["id"]) is False


def test_delete_last_assistant_does_not_touch_older_turns(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    store.add_message(session["id"], "user", "one")
    store.add_message(session["id"], "assistant", "reply one")
    store.add_message(session["id"], "user", "two")

    # The latest user turn has no reply yet: regenerate must not delete the
    # previous turn's answer.
    assert store.delete_last_assistant(session["id"]) is False
    messages = store.get_messages(session["id"])
    assert [m["content"] for m in messages] == ["one", "reply one", "two"]


def test_search_filter(tmp_path):
    store = WebStore(tmp_path / "web.db")
    store.create_session("Kerala trip")
    store.create_session("React hooks")
    assert [s["title"] for s in store.list_sessions("kerala")] == ["Kerala trip"]
    assert store.list_sessions("nothing") == []


def test_counts(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    store.add_message(session["id"], "user", "hi")
    assert store.counts() == {"sessions": 1, "messages": 1}


def test_message_attachments_and_tool_round_trip(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    attachment = {
        "id": "abc",
        "name": "notes.txt",
        "mime": "text/plain",
        "size": 5,
        "kind": "text",
        "url": "/api/uploads/abc/content",
    }
    user = store.add_message(
        session["id"],
        "user",
        "",
        attachments=[attachment],
        tool="summarize",
    )
    assert user["attachments"] == [attachment]
    assert user["tool"] == "summarize"

    loaded = store.get_session(session["id"], include_messages=True)
    assert loaded["messages"][0]["attachments"] == [attachment]
    assert store.last_user_message(session["id"])["tool"] == "summarize"


def test_upload_lifecycle(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    row = store.add_upload(
        filename="notes.txt",
        mime="text/plain",
        kind="text",
        data=b"hello",
        text_content="hello",
    )
    assert row["session_id"] is None
    assert store.get_upload(row["id"])["text_content"] == "hello"
    assert store.upload_path(row).read_bytes() == b"hello"

    store.attach_uploads([row["id"]], session["id"])
    assert store.get_upload(row["id"])["session_id"] == session["id"]
    assert store.delete_upload(row["id"]) is True
    assert store.get_upload(row["id"]) is None
    assert not store.upload_path(row).exists()


def test_old_message_schema_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "web.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                preset TEXT NOT NULL DEFAULT 'core',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                preset TEXT,
                status TEXT NOT NULL DEFAULT 'done',
                elapsed_ms INTEGER,
                client_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
    store = WebStore(path)
    session = store.create_session()
    message = store.add_message(
        session["id"], "user", "hi", attachments=[], tool="calc"
    )
    assert message["tool"] == "calc"
    assert message["attachments"] == []


def test_two_threads_use_separate_connections(tmp_path):
    store = WebStore(tmp_path / "web.db")
    session = store.create_session()
    errors = []

    def worker(index):
        try:
            for step in range(5):
                store.add_message(session["id"], "user", f"{index}-{step}")
        except Exception as exc:  # pragma: no cover - only on a locking bug
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store.get_messages(session["id"])) == 10
