"""Utilities for running I/O-bound operations asyncronously."""

import asyncio
import dataclasses
import functools
import itertools
import logging
import sys
import weakref
from typing import Any, Awaitable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import six

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class AsyncTaskMixin(logging.Handler):
    """Include an async task index in the log record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A way to always get the next id. Loggers are singltons so
        # there will only be one of these couters and therefore no dups.
        self._next_id = itertools.count().__next__
        # WeakDict lets us use a reference to an async task as a key
        # without stopping it from being garbage collected.
        self._task_ids = weakref.WeakKeyDictionary()

    def _task_id(self):
        """Map an Async Task to an id."""
        try:
            task = asyncio.current_task()
            if task not in self._task_ids:
                self._task_ids[task] = self._next_id()
            return f"task-{self._task_ids[task]}"
        except RuntimeError:
            return "main"

    def emit(self, record):
        """Add the task id to the record."""
        # Use setattr over `.` notation to avoid some overloading on the
        # record class. What people seem to do in most online examples.
        record.__setattr__("task", self._task_id())
        super().emit(record)


class AsyncTaskStreamHandler(AsyncTaskMixin, logging.StreamHandler):
    """Include an Async task-id when logging to a stream."""


class AsyncTaskFileHandler(AsyncTaskMixin, logging.FileHandler):
    """Include an Async task-id when logging to a file."""


def run(*args, **kwargs):
    """Run an awaitable to completion, dispatch based on python version."""
    if sys.version_info < (3, 8):
        if sys.platform in ("win32", "cygwin"):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(*args, **kwargs)


# A type variable to indicate that the keys of the dict will not change.
K = TypeVar("K")
# A type variable to indicate that the async task is called with K, V tuples.
V = TypeVar("V")


class MapTask(Protocol):
    def __call__(self, key: K, value: V) -> Awaitable[Tuple[K, Any]]:
        """An async function that runs on each key, value pair in a map."""


async def limited_concurrency(*args, f: MapTask, sem: asyncio.Semaphore, **kwargs):
    """Run f but limit the number of processes that can run at once."""
    async with sem:
        return await f(*args, **kwargs)


async def run_map(
    mapping: Dict[K, V],
    func: MapTask,
    max_concurrency: int = -1,
) -> Dict[K, Any]:
    """Run async function on K, V pairs, return map with result as new value."""
    if max_concurrency > 0:
        sem = asyncio.Semaphore(max_concurrency)
        func = functools.partial(limited_concurrency, f=func, sem=sem)
    return dict(await asyncio.gather(*(func(k, v) for k, v in mapping.items())))


@dataclasses.dataclass
class CompletedAsyncProcess:
    """Results from a finished async subprocess run."""

    args: Union[Sequence[str], str]
    returncode: Optional[int]
    stdout: Optional[bytes] = None
    stderr: Optional[bytes] = None


async def subprocess_run(
    command: Union[Sequence[str], str],
    input: Optional[Union[str, bytes]] = None,
    capture_output: bool = False,
) -> CompletedAsyncProcess:
    """Run a subprocess with async. Tries to mirror the subprocess.run API."""
    if not isinstance(command, str):
        shell_command = " ".join(command)
    else:
        shell_command = command
    proc = await asyncio.create_subprocess_shell(
        shell_command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if input is not None:
        stdout, stderr = await proc.communicate(input=six.ensure_binary(input))
    else:
        stdout, stderr = await proc.communicate()
    return CompletedAsyncProcess(
        command,
        proc.returncode,
        stdout if capture_output else None,
        stderr if capture_output else None,
    )
