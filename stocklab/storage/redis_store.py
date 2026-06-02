"""Redis implementation of :class:`StorageBackend`.

Keys are namespaced (``<ns>:<key>``) so multiple deployments can share one
Redis instance. Values are JSON-encoded; large raw blobs (cached datasets) use
the raw string API to avoid a double encode. All operations are atomic on the
Redis side, which is what makes the store safe to share across worker threads.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import redis

from .backend import StorageBackend, json_default


class RedisStorage(StorageBackend):
    def __init__(self, url: str, namespace: str = "stocklab"):
        self.namespace = namespace
        self._client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=8,
            socket_timeout=15,
            health_check_interval=30,
            retry_on_timeout=True,
        )

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # -- json kv ----------------------------------------------------------- #
    def put_json(self, key, value, ttl=None):
        payload = json.dumps(value, default=json_default)
        self._client.set(self._k(key), payload, ex=ttl)

    def get_json(self, key):
        raw = self._client.get(self._k(key))
        return json.loads(raw) if raw is not None else None

    # -- raw kv ------------------------------------------------------------ #
    def put_raw(self, key, value, ttl=None):
        self._client.set(self._k(key), value, ex=ttl)

    def get_raw(self, key):
        return self._client.get(self._k(key))

    # -- lists ------------------------------------------------------------- #
    def list_push(self, key, value, ttl=None):
        k = self._k(key)
        self._client.rpush(k, json.dumps(value, default=json_default))
        if ttl:
            self._client.expire(k, ttl)

    def list_all(self, key):
        items = self._client.lrange(self._k(key), 0, -1)
        return [json.loads(i) for i in items]

    # -- misc -------------------------------------------------------------- #
    def delete(self, key):
        self._client.delete(self._k(key))

    def exists(self, key):
        return bool(self._client.exists(self._k(key)))

    def ping(self):
        try:
            return bool(self._client.ping())
        except Exception:
            return False


_singleton: RedisStorage | None = None
_lock = threading.Lock()


def get_storage() -> RedisStorage:
    """Process-wide singleton Redis storage, lazily constructed."""
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                from ..config import REDIS_URL, REDIS_NAMESPACE

                _singleton = RedisStorage(REDIS_URL, REDIS_NAMESPACE)
    return _singleton
