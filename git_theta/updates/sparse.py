#!/usr/bin/env python3

import json
import logging
import os
from typing import Optional
from git_theta.updates import Update, get_update
from git_theta import file_io
from git_theta import git_utils


class SparseUpdate(Update):
    """An update where only some parameters are touched."""

    @property
    def name(self):
        return "sparse"

    def read(self, path, commit: Optional[str] = None):
        # TODO: Update to sparse special reads
        if commit is None or commit == "HEAD":
            return file_io.load_tracked_file(path)
        return file_io.read_tensorstore_from_memory(
            git_utils.load_tracked_dir_from_git(path, commit, apply_filters=True)
        )

    def calculate_sparse_update(self, new_value, prev_value):
        # TODO: Update to calculate a real sparse value
        return new_value - prev_value

    def write(self, path, parameter):
        logging.debug(f"Writing sparse update to '{path}'")
        prev_commit = git_utils.get_previous_commit(path)
        logging.debug(f"The last time '{path}' was updated was commit={prev_commit}")
        prev_metadata = json.loads(
            git_utils.load_tracked_file_from_git(
                os.path.join(path, "metadata"), prev_commit
            )
        )
        prev_update_type = prev_metadata["update"]
        logging.debug(f"The last update to '{path}' was a {prev_update_type} update.")
        prev_update = get_update(prev_update_type)()
        prev_value = prev_update.apply(path, prev_commit)
        logging.debug(f"Calculating sparse difference.")
        difference = self.calculate_sparse_update(parameter, prev_value)
        logging.debug(f"Writing sparse difference to '{path}'")
        file_io.write_tracked_file(path, difference)
        file_io.write_staged_file(os.path.join(path, "metadata"), {"update": self.name})

    def apply(self, path, commit: Optional[str] = None):
        logging.debug(f"Calculating result of sparse update to '{path}' at {commit=}")
        if commit is None or commit == "HEAD":
            commit = git_utils.get_previous_commit(path, commit)
            logging.debug(
                f"File state at HEAD requested, translating to real {commit=}"
            )
        sparse_update = self.read(path, commit)
        prev_commit = git_utils.get_previous_commit(path, commit)
        logging.debug(f"The last time '{path}' was updated was commit={prev_commit}")
        prev_metadata = json.loads(
            git_utils.load_tracked_file_from_git(
                os.path.join(path, "metadata"), prev_commit
            )
        )
        prev_update_type = prev_metadata["update"]
        logging.debug(f"The last update to '{path}' was a {prev_update_type} update.")
        prev_update = get_update(prev_update_type)()
        logging.debug(f"Recursively getting the previous value.")
        prev_value = prev_update.apply(path, prev_commit)
        return sparse_update + prev_value
