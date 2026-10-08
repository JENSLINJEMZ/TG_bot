import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot.ratelimit import RateLimiter


def test_window_zero_is_disabled():
    limiter = RateLimiter(0)
    assert limiter.allow(1)
    assert limiter.allow(1)
    assert limiter.wait_left(1) == 0.0


def test_first_request_is_allowed():
    limiter = RateLimiter(5)
    assert limiter.allow(42)


def test_second_request_within_window_is_blocked():
    limiter = RateLimiter(5)
    assert limiter.allow(7)
    assert not limiter.allow(7)
    assert 0 < limiter.wait_left(7) <= 5


def test_different_users_do_not_block_each_other():
    limiter = RateLimiter(5)
    assert limiter.allow(1)
    assert limiter.allow(2)


def test_window_expires():
    limiter = RateLimiter(0.05)
    assert limiter.allow(9)
    time.sleep(0.07)
    assert limiter.allow(9)


def test_negative_window_is_disabled():
    assert RateLimiter(-1).allow(3)


def test_stale_entries_are_pruned():
    limiter = RateLimiter(60)
    now = time.monotonic()
    limiter._hits = {uid: now - 120 for uid in range(10)}
    limiter._prune(time.monotonic())
    assert limiter._hits == {}