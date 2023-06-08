"""Helper utilities for unittests."""

import contextlib
import os
import tempfile


@contextlib.contextmanager
def named_temporary_file(**kwargs):
    """A named temp file that is safe to use on windows.

    When using this function to create a named tempfile, it is safe to call
    operations like `.flush()` and `.close()` on the tempfile which is needed
    on Windows. Like the normal tempfile context manager, the file is removed
    automatically when you exit the `with` scope.
    """
    # We force these so remove them.
    m = kwargs.pop("mode", None)
    if m is not None:
        raise RuntimeError(
            f"'mode' argument should not be provided to 'named_temporary_file', got {m}."
        )
    d = kwargs.pop("delete", None)
    if d is not None:
        raise RuntimeError(
            f"'delete' argument should not be provided to 'named_temporary_file', got {d}."
        )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, **kwargs) as f:
        try:
            yield f
        finally:
            os.unlink(f.name)
