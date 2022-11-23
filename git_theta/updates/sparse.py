"""A class for handling sparse updates to parameters."""

import json
import logging
import os
from typing import Optional
import numpy as np
from git_theta.updates import (
    Update,
    UpdateConstants,
    get_update_handler,
    most_recent_update,
    read_update_type,
)
from git_theta import file_io
from git_theta import params
from git_theta import git_utils


class SparseUpdate(Update):
    """An update where only some of the parameters are touched."""

    @property
    def name(self):
        """The name used to get this class as a plugin."""
        return "sparse"

    def read(self, path: str) -> np.ndarray:
        """Read the parameter values in the path dir.

        Note:
            The values read from disk are just the sparse update values, they
            need to be combined with the previous parameter values before they
            can be used.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path that contains parameter
            update values.

        Returns
        -------
        np.ndarray
            The parameter values at path.
        """
        # TODO: Update sparse to use actual sparse representations
        logging.debug(f"Reading sparse update from '{path}'")
        return file_io.load_tracked_file(path)

    def calculate_sparse_update(self, new_value, previous_value):
        """Get the sparse update whose application to previous value yields new value."""
        # TODO: Update sparse to use actual sparse representations
        return new_value - previous_value

    def _previous_update(self, path: str) -> str:
        """Find the last update applied to this parameter via the update metadata file.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path that for the current
            update

        Returns
        -------
        str
            The .../param/update/${hash}' directory path for the /last/ update
            applied to this parameter.
        """
        metadata_file = os.path.join(path, UpdateConstants.METADATA_FILE)
        metadata = file_io.load_staged_file(metadata_file)
        return metadata[UpdateConstants.PREVIOUS_KEY]

    def get_previous_value(self, path: str) -> np.ndarray:
        """Load the last parameter value.

        Parameters
        ----------
        path
            The .../param/update/${hash} directory path for the /last/ update
            applied to this parameter.

        Returns
        -------
        np.ndarray
            The previous parameter value, calculated by recursively calling the
            update class for the previous update.
        """
        # Get the update type information from the update metadata file.
        update_type = read_update_type(path)
        update = get_update_handler(update_type)()
        return update.apply(path)

    def write(self, path, parameter: np.ndarray):
        """Write `parameter` values to path.

        Note:
            Writing the sparse update between the previous parameters and the
            current parameters requires getting the previous values (recursively),
            calculating the sparse update, and writing that update to disk.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to write to.
        parameter
            The values to write.
        """
        logging.debug(f"Calculating sparse update to '{path}'")
        # Our update dirs are named based on the hash of the parameter after all
        # updates are applied, not based on the hash of the actual update.
        parameter_hash = params.get_hash(parameter)
        output_dir = os.path.join(path, UpdateConstants.UPDATES_DIR, parameter_hash)
        # When we are adding a new update, the last update will be the most
        # recent update that touched this parameter.
        previous_update = most_recent_update(path)
        logging.debug(f"The last time '{path}' was updated was in '{previous_update}'")
        # Convert that most recent path to an absolute path and use it to load
        # the most recent value of the parameter.
        repo = git_utils.get_git_repo()
        logging.debug("Recursively fetching the most recent value of the parameter.")
        previous = self.get_previous_value(
            git_utils.get_absolute_path(git_utils.get_git_repo(), previous_update)
        )
        logging.debug(f"Writing sparse update to '{output_dir}'")
        difference = self.calculate_sparse_update(parameter, previous)
        # Save the sparse update (not the final parameters).
        file_io.write_tracked_file(output_dir, difference)
        # Save update information, i.e. that this is a sparse update date.
        # Save the previous update to track what parameter out update is applied to.
        file_io.write_staged_file(
            os.path.join(output_dir, UpdateConstants.METADATA_FILE),
            {
                UpdateConstants.UPDATE_KEY: self.name,
                UpdateConstants.PREVIOUS_KEY: previous_update,
            },
        )
        # Update the parameter metadata file setting this update as the most recent one.
        self.record_update(path, output_dir)

    def apply(self, path) -> np.ndarray:
        """Get the update parameter values after applying the update from `path`.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to the update we want
            the result from.

        Returns
        -------
        np.ndarray
            The parameter after the update at `path` is applied.
        """
        logging.debug(f"Applying sparse update to '{path}'")
        previous_update = self._previous_update(path)
        logging.debug(f"The last time '{path}' was updated was in '{previous_update}'")
        # Get the last parameter value.
        previous = self.get_previous_value(previous_update)
        # Read the sparse update from disk
        difference = self.read(path)
        # Apply the sparse update to the previous parameter values.
        return previous + difference
