"""A class for handling sparse updates to parameters."""

import logging
import numpy as np
from git_theta.updates import TrueUpdate
from git_theta import file_io


class SparseUpdate(TrueUpdate):
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

    def calculate_update(self, new_value, previous_value):
        """Get the sparse update whose application to previous value yields new value."""
        # TODO: Update sparse to use actual sparse representations
        return new_value - previous_value

    def apply_update(self, previous, update):
        """Apply a sparse update."""
        return previous + update

    def write_update(self, path, update):
        logging.debug(f"Writing sparse update to '{path}'")
        # Save the sparse update (not the final parameters).
        file_io.write_tracked_file(path, update)
