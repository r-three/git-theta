"""Utilities for running I/O-bound operations asyncronously."""

import asyncio
import dataclasses
import functools
import sys
from concurrent.futures import thread
from typing import Any, Awaitable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import numba
import numba.misc.numba_sysinfo
import six

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Asyncify:
    """Wrap sync functions for use in async."""

    def __init__(self, *args, **kwargs):
        self.executor = thread.ThreadPoolExecutor(*args, **kwargs)
        # Adapted from https://github.com/numba/numba/blob/d44573b43dec9a7b66e9a0d24ef8db94c3dc346c/numba/misc/numba_sysinfo.py#L459
        try:
            # check import is ok, this means the DSO linkage is working
            from numba.np.ufunc import tbbpool  # NOQA

            # check that the version is compatible, this is a check performed at
            # runtime (well, compile time), it will also ImportError if there's
            # a problem.
            from numba.np.ufunc.parallel import _check_tbb_version_compatible

            numba.misc.numba_sysifo._check_tbb_version_compatible()
            self.threadsafe = True
        except ImportError as e:
            try:
                from numba.np.ufunc import omppool

                self.threadsafe = True
            except ImportError as e:
                self.threadsafe = False
        numba.config.THREADING_LAYER = "threadsafe" if self.threadsafe else "default"

    async def __call__(self, fn, *args, **kwargs):
        # If numba is thread safe then do it async!
        if self.threadsafe:
            return await asyncio.wrap_future(self.executor.submit(fn, *args, **kwargs))
        return fn(*args, **kwargs)


asyncify = Asyncify()


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
