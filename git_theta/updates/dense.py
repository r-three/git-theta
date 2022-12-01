"""Class managing dense parameter updates."""

import logging
import os
import shutil
from typing import Optional
import numpy as np
from git_theta import file_io
from git_theta import params
from git_theta.updates import Update, UpdateConstants


class DenseUpdate(Update):
    """An update where all parameters are changed."""

    @property
    def name(self):
        """The name used to get this class as a plugin."""
        return "dense"

    def read(self, path: str) -> np.ndarray:
        """Read the parameter values in the path dir.

        Note:
            In this case, the read parameter values are the whole update, all
            values are overwritten so the result of this read will just be
            returned.

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
        logging.debug(f"Reading Dense Update from '{path}'.")
        return file_io.load_tracked_file(path)

    def write(self, path: str, parameter: np.ndarray):
        """Write `parameter` values to path.

        Note:
            This update type removes all the previous updates in the .../updates/
            directory as they are no longer needed. This update sets all parameter
            values.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to write to.
        parameter
            The values to write.
        """
        # Remove any past updates to begin anew.
        # TODO: Move to using git_utils.remove_file once the edge case of the
        # path not existing is git is resolved.
        shutil.rmtree(path)
        parameter_hash = params.get_hash(parameter)
        output_dir = os.path.join(path, UpdateConstants.UPDATES_DIR, parameter_hash)
        logging.debug(f"Writing Dense Update to '{output_dir}'")
        # Write all parameter values as is to the update directory.
        file_io.write_tracked_file(output_dir, parameter)
        # Record update type information in the update metadata file.
        self.write_update_metadata(output_dir)
        # Update the parameter metadata file setting this update as the most recent one.
        self.record_update(path, output_dir)

    def apply(self, path: str) -> np.ndarray:
        """Get the update parameter values after applying the update from `path`.

        Note:
            As this update touches all parameters, the application of this update
            is just the values read from disk.

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
        logging.debug(f"Applying Dense Update to '{path}'")
        return self.read(path)
