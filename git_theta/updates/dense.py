#!/usr/bin/env python3

import logging
import os
from typing import Optional
from git_theta.updates import Update
from git_theta import file_io
from git_theta import git_utils


class DenseUpdate(Update):
    """An update where all parameters are changed."""

    @property
    def name(self):
        return "dense"

    def read(self, path, commit: Optional[str] = None):
        if commit is None or commit == "HEAD":
            return file_io.load_tracked_file(path)
        return file_io.read_tensorstore_from_memory(
            git_utils.load_tracked_dir_from_git(path, commit, apply_filters=True)
        )

    def write(self, path, parameter):
        file_io.write_tracked_file(path, parameter)
        file_io.write_staged_file(os.path.join(path, "metadata"), {"update": self.name})

    def apply(self, path, commit: Optional[str] = None):
        logging.debug(f"Reading Dense update to {path}@{commit=}")
        return self.read(path, commit)
