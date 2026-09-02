"""Explicit-start runtime helpers shared by the two local remote roles."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any


class AsyncLoopThread:
    """A private event loop that starts only after an explicit role action."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            thread = threading.Thread(target=self._run, name=self.name, daemon=True)
            self._thread = thread
            thread.start()
        if not self._ready.wait(2.0):
            raise RuntimeError(f"{self.name} event loop did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
            self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            with self._lock:
                self._loop = None

    def submit(self, coroutine: Coroutine[Any, Any, Any], *, timeout: float = 30.0) -> Any:
        self.start()
        with self._lock:
            loop = self._loop
        if loop is None:
            coroutine.close()
            raise RuntimeError(f"{self.name} event loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            # A caller timeout is a lifecycle cancellation, not permission for
            # the coroutine to publish resources later. Deliver cancellation
            # and wait briefly for its finally block before returning.
            future.cancel()
            with suppress(concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
                future.result(timeout=2.0)
            raise TimeoutError(f"{self.name} operation exceeded {timeout:.1f}s") from exc

    def call_soon(self, callback, *args: object) -> None:
        with self._lock:
            loop = self._loop
        if loop is None:
            raise RuntimeError(f"{self.name} event loop is unavailable")
        loop.call_soon_threadsafe(callback, *args)

    def close(self, *, grace_s: float = 2.0) -> None:
        if not 0 <= grace_s <= 30:
            raise ValueError("event-loop shutdown grace must be in [0s,30s]")
        with self._lock:
            loop = self._loop
            thread = self._thread
        if loop is not None and thread is not threading.current_thread():

            async def drain() -> None:
                current = asyncio.current_task()
                pending = [task for task in asyncio.all_tasks() if task is not current]
                if not pending:
                    return
                _, still_pending = await asyncio.wait(pending, timeout=grace_s)
                for task in still_pending:
                    task.cancel()
                if still_pending:
                    await asyncio.gather(*still_pending, return_exceptions=True)

            drained = asyncio.run_coroutine_threadsafe(drain(), loop)
            with suppress(
                concurrent.futures.CancelledError,
                concurrent.futures.TimeoutError,
                RuntimeError,
            ):
                drained.result(timeout=grace_s + 1.0)
            loop.call_soon_threadsafe(loop.stop)
        elif loop is not None:
            loop.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=grace_s + 1.0)
        with self._lock:
            if thread is None or not thread.is_alive():
                self._thread = None
