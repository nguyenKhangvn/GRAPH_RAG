"""API Key rotation utility for rate-limit optimization."""

import os
import time
import threading


class KeyRotator:
    """Round-robin key rotation with per-key cooldown tracking."""

    def __init__(self, env_key: str = "GEMINI_API_KEYS", cooldown: float = 60.0):
        raw = os.getenv(env_key, "")
        self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self.keys:
            # Fallback to single key
            single = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            self.keys = [single] if single else []
        self._idx = 0
        self._cooldown = cooldown
        self._last_used: dict[str, float] = {k: 0.0 for k in self.keys}
        self._lock = threading.Lock()

    def get_key(self) -> str:
        """Get the next available key (least recently used)."""
        with self._lock:
            now = time.time()
            # Pick key with longest time since last use
            best = min(self.keys, key=lambda k: self._last_used.get(k, 0.0))
            self._last_used[best] = now
            return best

    def mark_failed(self, key: str):
        """Mark a key as recently used (cooldown) after a rate-limit error."""
        with self._lock:
            self._last_used[key] = time.time() + self._cooldown

    @property
    def count(self) -> int:
        return len(self.keys)


# Singleton
_rotator: KeyRotator | None = None


def get_rotator() -> KeyRotator:
    global _rotator
    if _rotator is None:
        _rotator = KeyRotator()
    return _rotator
