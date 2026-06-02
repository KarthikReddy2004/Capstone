"""Storage layer: a Redis-backed, thread-safe key/value + list store.

The :class:`StorageBackend` interface decouples the rest of the platform from
the concrete store. The default implementation talks to Redis (single shared
source of truth across worker threads / processes), but any backend honouring
the interface can be dropped in for tests.
"""

from .backend import StorageBackend, json_default
from .redis_store import RedisStorage, get_storage

__all__ = ["StorageBackend", "RedisStorage", "get_storage", "json_default"]
