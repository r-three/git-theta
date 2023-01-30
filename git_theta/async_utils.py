"""Utilities for running I/O-bound operations asyncronously."""

import asyncio
import dataclasses
import functools
import sys
from typing import (
    Any,
    Dict,
    Tuple,
    TypeVar,
    Awaitable,
    Union,
    Optional,
    Sequence,
)
import six

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


def run(*args, **kwargs):
    """Run an awaitable to completion, dispatch based on python version."""
    # TODO(bdlester): Remove if we bump to python 3.7
    if sys.version_info < (3, 7):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(*args, **kwargs)
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
