"""Base class for parameter update plugins.

Example directory structure of how updates are stored.

.git_theta/my-model.pt/
├── layers.0.bias
│   └── params
│       ├── metadata  # Pointer to the most recent update.
│       └── updates
│           │   # This is the hash of the parameter after the update, not the contents of the update.
│           ├── 13f81e06d1a47a437541a26f86fd20c89ccf73ea
│           │   ├── 0
│           │   └── metadata  # Update type and pointer to the last update.
│           ├── 1f40b7caef2f961b4bde95f9a450c3a12eb6f249
│           │   ├── 0
│           │   └── metadata
│           └── 77db6ed78df01aecbb9e7990a87d50f7dc2d5579
│               ├── 0
│               └── metadata
├── ...
└── layers.1.weight
    └── params
        ├── metadata
        └── updates
            ├── 2ceb7dac4dd0b012fd404e227c13cf66bd25cf3a
            │   ├── 0.0
            │   └── metadata
            ├── aeefd921f332f102e3e77ca6ec3d46d707afe9a9
            │   ├── 0.0
            │   └── metadata
            └── b16693be23b9146457c20752cdac2796de5a7290
                ├── 0.0
                └── metadata


Example .../layers.0.bias/params/metadata
[
    ".git_theta/my-model.pt/layers.0.bias/params/updates/77db6ed78df01aecbb9e7990a87d50f7dc2d5579"
]

Example .../params/updates/${hash}/metadata
{
    "update": "sparse",
    "previous": ".git_theta/my-model.pt/layers.0.bias/params/updates/13f81e06d1a47a437541a26f86fd20c89ccf73ea"
}

Another Example
{
    "update": "dense"
}

"""

import logging
import os
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from typing import Optional

import numpy as np
from git_theta import file_io
from git_theta import git_utils
from git_theta import utils


class UpdateConstants:
    """Scoped constants for consistent file name and key naming."""

    UPDATE_KEY: str = "update"
    UPDATES_DIR: str = "updates"
    METADATA_FILE: str = "metadata"
    PREVIOUS_KEY: str = "previous"


class Update:
    """Base class for parameter update plugins."""

    @property
    def name(self):
        """The name used to get this class as a plugin."""
        raise NotImplementedError

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
            metadata = []
        else:
            metadata = file_io.load_staged_file(metadata_file)
        update_path = git_utils.get_relative_path_from_root(
            git_utils.get_git_repo(), update_path
        )
        logging.debug(
            f"Recording recent update as '{update_path}' was '{metadata[-1] if metadata else None}'"
        )
        file_io.write_staged_file(metadata_file, [update_path])


def most_recent_update(path: str) -> str:
    """Get the most recent update applied to the parameter.

    Note:
        Path is expected to be a path to the params directory of a parameter.
        It should be an absolute path.

    Parameters
    ----------
    path
        The path to the .../params dir for a parameter.

    Returns
    -------
    str
        The path to the previous update for the parameter, relative to the repo root.
    """
    metadata_file = os.path.join(path, UpdateConstants.METADATA_FILE)
    metadata = file_io.load_staged_file(metadata_file)
    return metadata[0] if metadata else None


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
