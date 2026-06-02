"""Abstract storage interface and JSON helpers."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


def json_default(obj: Any):
    """JSON encoder hook for NumPy scalars/arrays that slip through."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


class StorageBackend(abc.ABC):
    """Minimal key/value + list store used by the job manager and cache."""

    @abc.abstractmethod
    def put_json(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abc.abstractmethod
    def get_json(self, key: str) -> Any | None: ...

    @abc.abstractmethod
    def delete(self, key: str) -> None: ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def list_push(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abc.abstractmethod
    def list_all(self, key: str) -> list[Any]: ...

    @abc.abstractmethod
    def put_raw(self, key: str, value: str, ttl: int | None = None) -> None: ...

    @abc.abstractmethod
    def get_raw(self, key: str) -> str | None: ...

    @abc.abstractmethod
    def ping(self) -> bool: ...
