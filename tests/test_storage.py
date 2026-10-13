import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import storage as storage_mod
from tgbot.localstore import LocalStore
from tgbot.storage import Storage

ORIGINALS = [(b"orig", "image/jpeg", "jpg")]
RESULT = (b"res", "image/png")


def _save_media(store: Storage) -> str:
    return store.save_media(
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


def test_media_is_saved_locally(tmp_path):
    local = LocalStore(tmp_path / "store.db")
    store = Storage(local)
    assert _save_media(store) == ""
    assert local.counts() == (1, 0)
    row = local.all_media()[0]
    assert row.mode == "restyle"
    assert row.original_blob == b"orig"
    assert row.result_blob == b"res"
    assert row.result_type == "image/png"


def test_two_originals_store_an_aux(tmp_path):
    local = LocalStore(tmp_path / "store.db")
    store = Storage(local)
    store.save_media(
        chat_id=1,
        user_id=2,
        message_id=3,
        mode="combine",
        prompt="combine",
        background_rgb=None,
        originals=[(b"a", "image/jpeg", "jpg"), (b"b", "image/png", "png")],
        result=RESULT,
        source_file_id="f",
        elapsed_ms=10,
    )
    row = local.all_media()[0]
    assert row.original_blob == b"a"
    assert row.aux_blob == b"b"
    assert row.aux_type == "image/png"


def test_no_store_reports_not_stored():
    assert _save_media(Storage(None)) == storage_mod.NOT_STORED
    assert (
        Storage(None).save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="yo", elapsed_ms=1)
        == storage_mod.NOT_STORED
    )


def test_chat_is_saved_locally(tmp_path):
    local = LocalStore(tmp_path / "store.db")
    store = Storage(local)
    assert store.save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="yo", elapsed_ms=1) == ""
    assert local.counts() == (0, 1)
    assert local.all_chats()[0].reply == "yo"


class _BrokenStore:
    def save_media(self, **kwargs):
        raise RuntimeError("disk full")

    def save_chat(self, **kwargs):
        raise RuntimeError("disk full")


def test_write_failure_is_reported_not_raised():
    store = Storage(_BrokenStore())
    assert _save_media(store) == storage_mod.WRITE_FAILED
    assert (
        store.save_chat(chat_id=1, user_id=2, message_id=3, prompt="hi", reply="yo", elapsed_ms=1)
        == storage_mod.WRITE_FAILED
    )


def test_storage_module_constants():
    assert storage_mod.NOT_STORED and storage_mod.WRITE_FAILED
