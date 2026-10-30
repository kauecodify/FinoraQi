import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class JobManager:
    """Fila de jobs em processo (substitui Celery + Redis da v1)."""

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finoraqi-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"status": "PENDING", "result": None, "error": None, "created_at": time.time()}

        def run() -> None:
            self._update(job_id, status="RUNNING")
            try:
                self._update(job_id, status="SUCCESS", result=fn(*args, **kwargs))
            except Exception as exc:
                self._update(job_id, status="FAILURE", error=f"{exc}\n{traceback.format_exc(limit=3)}")

        self._pool.submit(run)
        return job_id

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


jobs = JobManager()
