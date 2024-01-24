"""Shared logging setup for scripts.

This is not part of the main git_theta.__init__ as we don't want to configure
logging if git_theta is being used as a library.
"""

import logging
import os
import tempfile

import git_theta


def configure_logging(exe_name: str):
    format_str = f"{exe_name}: [%(asctime)s] [task %(task)s] [%(funcName)s] %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(
            logging, git_theta.utils.EnvVarConstants.LOG_LEVEL.upper(), logging.DEBUG
        ),
        # Log to a file for clean/smudge as they don't appear on the console when called via git.
        format=format_str,
        handlers=[
            git_theta.async_utils.AsyncTaskStreamHandler(),
            git_theta.async_utils.AsyncTaskFileHandler(
                filename=os.path.join(tempfile.gettempdir(), "git-theta.log")
            ),
        ],
    )
