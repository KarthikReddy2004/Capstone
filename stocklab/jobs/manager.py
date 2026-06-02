"""Job lifecycle: queue, progress tracking, and dataset caching.

A bounded :class:`ThreadPoolExecutor` is the worker queue. Job state lives
entirely in Redis (no global mutable Python state), so progress can be polled
from any request handler and survives across worker threads. Each running job
owns a :class:`JobContext` that is the only writer of its record, which keeps
updates race-free without per-key locking.
"""

from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any, Callable

import pandas as pd

from ..storage import StorageBackend, get_storage


def _now() -> float:
    return time.time()


class JobContext:
    """Handle passed to a worker function for reporting progress/results."""

    def __init__(self, manager: "JobManager", job_id: str, n_models: int):
        self._m = manager
        self._s = manager.storage
        self.job_id = job_id
        self._meta = {
            "id": job_id,
            "status": "running",
            "progress": 0.0,
            "stage": "Starting",
            "n_models": n_models,
            "model_status": {},
            "error": None,
            "started": _now(),
            "updated": _now(),
        }
        self._flush_meta()

    # -- internal ---------------------------------------------------------- #
    def _flush_meta(self):
        self._meta["updated"] = _now()
        self._s.put_json(f"job:{self.job_id}:meta", self._meta, ttl=self._m.job_ttl)

    # -- public reporting API --------------------------------------------- #
    def log(self, msg: str):
        self._s.list_push(f"job:{self.job_id}:log", {"t": _now(), "msg": str(msg)}, ttl=self._m.job_ttl)

    def set_stage(self, stage: str):
        self._meta["stage"] = stage
        self._flush_meta()

    def set_progress(self, value: float):
        self._meta["progress"] = float(max(0.0, min(1.0, value)))
        self._flush_meta()

    def set_model_status(self, name: str, status: str):
        self._meta["model_status"][name] = status
        self._flush_meta()

    def push_result(self, result: dict):
        self._s.list_push(f"job:{self.job_id}:results", result, ttl=self._m.job_ttl)

    def push_summary(self, summary: dict):
        """Push a small per-model summary for cheap live-board polling."""
        self._s.list_push(f"job:{self.job_id}:summary", summary, ttl=self._m.job_ttl)

    def set_final(self, final: dict):
        self._s.put_json(f"job:{self.job_id}:final", final, ttl=self._m.job_ttl)

    def complete(self):
        self._meta["status"] = "done"
        self._meta["progress"] = 1.0
        self._meta["stage"] = "Complete"
        self._flush_meta()

    def fail(self, error: str):
        self._meta["status"] = "error"
        self._meta["error"] = str(error)
        self._flush_meta()
        self.log(f"FATAL: {error}")


class JobManager:
    """Submit and track long-running benchmark jobs."""

    def __init__(self, storage: StorageBackend, max_workers: int, job_ttl: int, cache_ttl: int):
        self.storage = storage
        self.job_ttl = job_ttl
        self.cache_ttl = cache_ttl
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stocklab-worker")

    # -- dataset cache ----------------------------------------------------- #
    def cache_dataframe(self, df: pd.DataFrame, label: str) -> str:
        key = f"data:{label}:{uuid.uuid4().hex[:8]}"
        self.storage.put_raw(f"cache:{key}", df.to_json(orient="split"), ttl=self.cache_ttl)
        return key

    def load_dataframe(self, key: str) -> pd.DataFrame | None:
        raw = self.storage.get_raw(f"cache:{key}")
        if raw is None:
            return None
        return pd.read_json(StringIO(raw), orient="split")

    def has_dataframe(self, key: str) -> bool:
        return self.storage.exists(f"cache:{key}")

    # -- job submission ---------------------------------------------------- #
    def submit(self, worker: Callable[..., Any], n_models: int, *args, **kwargs) -> str:
        job_id = f"job_{int(_now() * 1000)}_{uuid.uuid4().hex[:6]}"
        ctx = JobContext(self, job_id, n_models)

        def _runner():
            try:
                worker(ctx, *args, **kwargs)
                if ctx._meta["status"] == "running":
                    ctx.complete()
            except Exception as err:  # noqa: BLE001 - report any failure to the UI
                ctx.fail(err)
                traceback.print_exc()

        self._pool.submit(_runner)
        return job_id

    # -- polling ----------------------------------------------------------- #
    def get_job(self, job_id: str) -> dict | None:
        meta = self.storage.get_json(f"job:{job_id}:meta")
        if meta is None:
            return None
        return {
            **meta,
            "log": self.storage.list_all(f"job:{job_id}:log"),
            "results": self.storage.list_all(f"job:{job_id}:results"),
            "final": self.storage.get_json(f"job:{job_id}:final"),
        }

    def get_status(self, job_id: str) -> dict | None:
        """Light poll payload: meta + log + per-model summaries (no big arrays)."""
        meta = self.storage.get_json(f"job:{job_id}:meta")
        if meta is None:
            return None
        return {
            **meta,
            "log": self.storage.list_all(f"job:{job_id}:log"),
            "summary": self.storage.list_all(f"job:{job_id}:summary"),
            "has_final": self.storage.exists(f"job:{job_id}:final"),
        }

    def get_results(self, job_id: str) -> dict | None:
        """Heavy payload fetched once when the job is done (full arrays + final)."""
        meta = self.storage.get_json(f"job:{job_id}:meta")
        if meta is None:
            return None
        return {
            "status": meta.get("status"),
            "results": self.storage.list_all(f"job:{job_id}:results"),
            "final": self.storage.get_json(f"job:{job_id}:final"),
        }

    def get_meta(self, job_id: str) -> dict | None:
        return self.storage.get_json(f"job:{job_id}:meta")


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        from ..config import MAX_WORKERS, JOB_TTL, CACHE_TTL

        _manager = JobManager(get_storage(), MAX_WORKERS, JOB_TTL, CACHE_TTL)
    return _manager
