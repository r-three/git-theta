"""Shared logging setup for scripts.

This is not part of the main git_theta.__init__ as we don't want to configure
logging if git_theta is being used as a library.
"""

import logging
import os
import tempfile
from typing import Optional

import git_theta


def configure_logging(
    exe_name: str, logger_name: str = "git_theta", root: Optional[str] = None
):
    logger = logging.getLogger(logger_name)
    format_str = f"{exe_name}: [%(asctime)s] [%(task)s] [%(package)s:%(funcName)s] %(levelname)s - %(message)s"
    log_level = getattr(
        logging, git_theta.utils.EnvVarConstants.LOG_LEVEL.upper(), logging.DEBUG
    )
    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt=format_str)

    root = (
        os.path.dirname(os.path.dirname(git_theta.__file__)) if root is None else root
    )

    def log_filter(record: logging.LogRecord) -> logging.LogRecord:
        package = record.pathname[len(root) + 1 :]
        if package.endswith(".py"):
            package = package[:-3]
        record.package = package.replace(os.sep, ".")
        return record

    handlers = (
        git_theta.async_utils.AsyncTaskStreamHandler(),
        git_theta.async_utils.AsyncTaskFileHandler(
            filename=os.path.join(tempfile.gettempdir(), "git-theta.log")
        ),
    )
    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        handler.addFilter(log_filter)
        logger.addHandler(handler)

    return logger
