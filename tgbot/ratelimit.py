from __future__ import annotations

import time


class RateLimiter:
    """Per-user cooldown for heavy operations (diffusion, chat, ...).

    `window` is the minimum number of seconds between two allowed requests for
    the same user. A window of 0 (or negative) disables the limiter entirely.
    """

    def __init__(self, window: float) -> None:
        self.window = window
        self._hits: dict[int, float] = {}

    def allow(self, user_id: int) -> bool:
        """True if the request may run. Marks the attempt when allowed."""
        if self.window <= 0:
            return True
        now = time.monotonic()
        last = self._hits.get(user_id)
        if last is not None and now - last < self.window:
            self._prune(now)
            return False
        self._hits[user_id] = now
        if len(self._hits) > 1024:
            self._prune(now)
        return True

    def wait_left(self, user_id: int) -> float:
        """Seconds until the user may start another heavy request (0 = now)."""
        last = self._hits.get(user_id)
        if last is None:
            return 0.0
        return max(0.0, self.window - (time.monotonic() - last))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        for uid, ts in list(self._hits.items()):
            if ts < cutoff:
                del self._hits[uid]