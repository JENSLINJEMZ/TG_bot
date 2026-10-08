import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import storage as storage_mod
from tgbot.localstore import LocalStore
from tgbot.storage import Storage
from tgbot.supabase import SupabasePermissionError
from tgbot.sync import sync_pending


class FakeDB:
    def __init__(self, fail=False, permission=False):
        self.fail = fail
        self.permission = permission
        self.uploads = []
        self.media_rows = []
        self.chat_rows = []

    def _maybe_fail(self):
        if self.permission:
            raise SupabasePermissionError("denied")
        if self.fail:
            raise RuntimeError("supabase unreachable")

    async def upload(self, path, data, content_type):
        self._maybe_fail()
        self.uploads.append((path, data, content_type))

    async def insert(self, row):
        self._maybe_fail()
        self.media_rows.append(row)
        return row

    async def insert_chat(self, row):
        self._maybe_fail()
        self.chat_rows.append(row)
        return row


ORIGINALS = [(b"orig", "image/jpeg", "jpg")]
RESULT = (b"res", "image/png")


def _save_media(store):
    return asyncio.run(
        store.save_media(
            chat_id=1,
            user_id=2,
            message_id=3,
            mode="restyle",
            prompt="navy suit",
            background_rgb=None,
            originals=ORIGINALS,
            result=RESULT,
            source_file_id="f",
            elapsed_ms=10,
        )
    )


def test_online_write_returns_empty_and_does_not_queue(tmp_path):
    db = FakeDB()
    local = LocalStore(tmp_path / "o.db")
    store = Storage(db, local)
    assert _save_media(store) == ""
    assert len(db.uploads) == 2
    assert len(db.media_rows) == 1
    assert db.media_rows[0]["mode"] == "restyle"
    assert local.counts() == (0, 0)


def test_unreachable_queues_locally(tmp_path):
    db = FakeDB(fail=True)
    local = LocalStore(tmp_path / "o.db")
    store = Storage(db, local)
    note = _save_media(store)
    assert "sync" in note
    media, chat = local.counts()
    assert (media, chat) == (1, 0)
    row = local.pending_media()[0]
    assert row.original_blob == b"orig"
    assert row.result_blob == b"res"


def test_permission_error_queues_locally(tmp_path):
    db = FakeDB(permission=True)
    local = LocalStore(tmp_path / "o.db")
    store = Storage(db, local)
    note = _save_media(store)
    assert "migration" in note
    assert local.counts() == (1, 0)


def test_no_supabase_configured_queues_locally(tmp_path):
    local = LocalStore(tmp_path / "o.db")
    store = Storage(None, local)
    note = _save_media(store)
    assert note == storage_mod.NOT_CONFIGURED
    assert local.counts() == (1, 0)


def test_chat_online_and_offline(tmp_path):
    local = LocalStore(tmp_path / "o.db")
    online = Storage(FakeDB(), local)
    assert asyncio.run(
        online.save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="yo", elapsed_ms=1)
    ) == ""
    assert local.counts() == (0, 0)

    offline = Storage(FakeDB(fail=True), local)
    note = asyncio.run(
        offline.save_chat(chat_id=1, user_id=2, message_id=4, prompt="hi", reply="yo", elapsed_ms=1)
    )
    assert note
    assert local.counts() == (0, 1)


def test_sync_pending_flushes_and_clears(tmp_path):
    local = LocalStore(tmp_path / "o.db")
    offline = Storage(FakeDB(fail=True), local)
    _save_media(offline)
    asyncio.run(
        offline.save_chat(chat_id=1, user_id=2, message_id=4, prompt="hi", reply="hey", elapsed_ms=1)
    )
    assert local.counts() == (1, 1)

    db = FakeDB()
    summary = asyncio.run(sync_pending(db, local))
    assert summary["media"] == 1
    assert summary["chats"] == 1
    assert summary["failed"] == 0
    assert local.counts() == (0, 0)
    assert db.media_rows[0]["original_path"].endswith(".jpg")
    assert db.chat_rows[0]["reply"] == "hey"


def test_sync_pending_keeps_failures_queued(tmp_path):
    local = LocalStore(tmp_path / "o.db")
    _save_media(Storage(FakeDB(fail=True), local))

    summary = asyncio.run(sync_pending(FakeDB(fail=True), local))
    assert summary["failed"] == 1
    assert local.counts() == (1, 0)
    assert local.pending_media()[0].sync_error


def test_sync_dry_run_does_not_send(tmp_path):
    local = LocalStore(tmp_path / "o.db")
    _save_media(Storage(FakeDB(fail=True), local))
    db = FakeDB()
    summary = asyncio.run(sync_pending(db, local, dry_run=True))
    assert summary["queued"] == 1
    assert db.uploads == []
    assert local.counts() == (1, 0)


def test_storage_module_constants():
    assert storage_mod.WILL_SYNC and storage_mod.NEEDS_GRANT and storage_mod.NOT_CONFIGURED