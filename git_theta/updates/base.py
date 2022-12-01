"""Base class for parameter update plugins.

Example directory structure of how updates are stored.

.git_theta/my-model.pt/
├── layers.0.bias
│   └── params
│       ├── metadata  # Pointer to the most recent update.
│       └── updates
│           │   # This is the hash of the parameter after the update, not the contents of the update.
│           ├── 13f81e06d1a47a437541a26f86fd20c89ccf73ea
│           │   ├── .zarray
│           │   ├── 0
│           │   └── metadata  # Update type and pointer to the last update.
│           ├── 1f40b7caef2f961b4bde95f9a450c3a12eb6f249
│           │   ├── .zarray
│           │   ├── 0
│           │   └── metadata
│           └── 77db6ed78df01aecbb9e7990a87d50f7dc2d5579
│               ├── .zarray
│               ├── 0
│               └── metadata
├── ...
└── layers.1.weight
    └── params
        ├── metadata
        └── updates
            ├── 2ceb7dac4dd0b012fd404e227c13cf66bd25cf3a
            │   ├── .zarray
            │   ├── 0.0
            │   └── metadata
            ├── aeefd921f332f102e3e77ca6ec3d46d707afe9a9
            │   ├── .zarray
            │   ├── 0.0
            │   └── metadata
            └── b16693be23b9146457c20752cdac2796de5a7290
                ├── .zarray
                ├── 0.0
                └── metadata


Example .../layers.0.bias/params/metadata
{
    "previous": ".git_theta/my-model.pt/layers.0.bias/params/updates/77db6ed78df01aecbb9e7990a87d50f7dc2d5579"
}

Example .../params/updates/${hash}/metadata
{
    "update": "sparse",
    "previous": ".git_theta/my-model.pt/layers.0.bias/params/updates/13f81e06d1a47a437541a26f86fd20c89ccf73ea"
}

Another Example
{
    "update": "dense"
}

Note:
    Metadata paths should be recorded as relative paths from the repo root, i.e.
    `.git_theta/...`.
"""

from abc import ABCMeta, abstractmethod
import logging
import os
import sys
from typing import Dict, Any, Optional

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from typing import Optional

import numpy as np
from git_theta import file_io
from git_theta import git_utils
from git_theta import utils
from git_theta import params


class UpdateConstants:
    """Scoped constants for consistent file name and key naming."""

    UPDATE_KEY: str = "update"
    UPDATES_DIR: str = "updates"
    METADATA_FILE: str = "metadata"
    PREVIOUS_KEY: str = "previous"


class Update(metaclass=ABCMeta):
    """Base class for parameter update plugins."""

    @property
    def name(self):
        """The name used to get this class as a plugin."""
        raise NotImplementedError

    @abstractmethod
    def read(self, path: str) -> np.ndarray:
        """Read the parameter values in the path dir.

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
        raise NotImplementedError

    @abstractmethod
    def write(self, path: str, parameter: np.ndarray):
        """Write `parameter` values to path.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to write to.
        parameter
            The values to write.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, path: str) -> np.ndarray:
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
        raise NotImplementedError

    # TODO: Unify with record update?
    def write_update_metadata(self, path: str, previous_path: Optional[str] = None):
        # Save update information, i.e. what kind of update this is.
        # Save the previous update to track what parameter out update is applied to.
        if previous_path is not None:
            # Save the previous update as a relative path.
            metadata = {UpdateConstants.PREVIOUS_KEY: previous_path}
        else:
            metadata = {}
        file_io.write_staged_file(
            os.path.join(path, UpdateConstants.METADATA_FILE),
            {UpdateConstants.UPDATE_KEY: self.name, **metadata},
        )

    def record_update(self, path: str, update_path: str):
        """Update the most recent update metadata file to point to this update.

        Parameters
        ----------
        path
            The path to the .../params dir for a parameter, this should be absolute.
        update_path
            The path to the .../params/updates/${hash} dir for a parameter, can
            be absolute or relative.
        """
        metadata_file = os.path.join(path, UpdateConstants.METADATA_FILE)
        if not os.path.exists(metadata_file):
            metadata = {}
        else:
            metadata = file_io.load_staged_file(metadata_file)
        update_path = git_utils.get_relative_path_from_root(
            git_utils.get_git_repo(), update_path
        )
        logging.debug(
            f"Recording recent update as '{update_path}' was '{metadata[UpdateConstants.PREVIOUS_KEY] if metadata else None}'"
        )
        file_io.write_staged_file(
            metadata_file, {UpdateConstants.PREVIOUS_KEY: update_path}
        )


class TrueUpdate(Update):
    """A base class for update types that use the previous parameter value."""

    @abstractmethod
    def write_update(path: str, update: np.ndarray):
        """Write `update` values to path.

        This is method is designed to be overridden with the logic to actually
        serialize some update value to disk. It doesn't include logic on getting
        the previous value.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to write to.
        update
            The values to write.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_update(self, new_value, previous_value):
        """Calculate the update applied to previous value to yield new_value.

        Parameters
        ----------
        new_value
            The updated parameter value.
        previous_value
            What the parameter was after the last update.

        Returns
        -------
        np.ndarray
            The update to apply to previous_value to yield new_value
        """
        raise NotImplementedError

    @abstractmethod
    def apply_update(self, previous, update):
        """Apply update to previous.

        Parameters
        ----------
        previous
            The old parameter values
        update
            The new update values

        Returns
        -------
        np.ndarray
            The new parameter value from applying update to previous.
        """
        raise NotImplementedError

    def write(self, path: str, parameter: np.ndarray):
        """Write `parameter` values to path.

        Parameters
        ----------
        path
            The .../params/updates/${hash} directory path to write to.
        parameter
            The values to write.
        """
        logging.debug(f"Calculating {self.name} update to '{path}'")
        path = git_utils.get_absolute_path(git_utils.get_git_repo(), path)
        # Our update dirs are named based on the hash of the parameter after all
        # updates are applied, not based on the hash of the actual update.
        parameter_hash = params.get_hash(parameter)
        output_dir = os.path.join(path, UpdateConstants.UPDATES_DIR, parameter_hash)
        # When we are adding a new update, the last update will be the most
        # recent update that touched this parameter.
        previous_update_pointer = most_recent_update(path)
        logging.debug(
            f"The last time '{path}' was updated was in '{previous_update_pointer}'"
        )
        previous_value = self.get_previous_value(previous_update_pointer)
        # Calculate the update.
        update = self.calculate_update(parameter, previous_value)
        # Save the actual update to disk
        self.write_update(output_dir, update)
        # Save the metadata file for the update
        self.write_update_metadata(output_dir, previous_update_pointer)
        # Update the pointer to the most recent update.
        self.record_update(path, output_dir)

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
        path = git_utils.get_absolute_path(git_utils.get_git_repo(), path)
        # Get the update type information from the update metadata file.
        update_type = read_update_type(path)
        update = get_update_handler(update_type)()
        return update.apply(path)

    def apply(self, path: str) -> np.ndarray:
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
        logging.debug(f"Applying {self.name} update to '{path}'.")
        path = git_utils.get_absolute_path(git_utils.get_git_repo(), path)
        previous_update_pointer = most_recent_update(path)
        logging.debug(
            f"The last time '{path}' was updated was in '{previous_update_pointer}"
        )
        previous_value = self.get_previous_value(previous_update_pointer)
        update = self.read(path)
        return self.apply_update(previous_value, update)


def most_recent_update(path: str) -> str:
    """Get the most recent update applied to the parameter.

    Note:
        If path is the `.../params` directory this returns the most recent
        update for that parameter. If path is the `.../params/updates/${hash}`
        directory this returns the most recent update relative to that update,
        i.e. the previous update.

    Parameters
    ----------
    path
        The path to the .../params dir for a parameter or the
        .../params/updates/${hash} dir for some update.

    Returns
    -------
    str
        The path to the previous update for the parameter, relative to the repo root.
    """
    # Convert to an absolute path.
    path = git_utils.get_absolute_path(git_utils.get_git_repo(), path)
    metadata_file = os.path.join(path, UpdateConstants.METADATA_FILE)
    metadata = file_io.load_staged_file(metadata_file)
    return metadata.get(UpdateConstants.PREVIOUS_KEY) if metadata else None


def read_update_type(path: str) -> str:
    """Get the update type used for some update.

    Notes:
        Path is expected to be a .../params/updates/${hash} path for a parameter.
        It should be an absolute path.

    Parameters
    ----------
    path
        The path to the .../params/updates/${hash} path for a parameter.

    Returns
    -------
    str
        The update type used to save this update.
    """
    metadata_file = os.path.join(path, UpdateConstants.METADATA_FILE)
    metadata = file_io.load_staged_file(metadata_file)
    return metadata[UpdateConstants.UPDATE_KEY]


def get_update_handler_name(update_type: Optional[str] = None) -> str:
    """Get the update type based on multiple overrides.

    The order of precedence is:
    1) User input via the update_type parameter
    2) The GIT_THETA_UPDATE_TYPE environment variable
    3) The default value ("dense")

    Parameters
    ----------
    update_type
        The update type name

    Returns
    -------
    str
        The update type based on overrides.
    """
    return update_type or os.environ.get(utils.EnvVarConstants.UPDATE_TYPE) or "dense"


def get_update_handler(update_type: Optional[str] = None) -> Update:
    """Get an Update class by name.

    Parameters
    ----------
    update_type
        The name of the update type we want to use.

    Returns
    -------
    Update
        The update class. Returned class my be defined in a user installed plugin.
    """
    update_type = get_update_handler_name(update_type)
    discovered_plugins = entry_points(group="git_theta.plugins.updates")
    return discovered_plugins[update_type].load()
