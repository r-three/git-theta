"""Base class for parameter update plugins."""

import os
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

import logging
from typing import Optional

from git_theta import git_utils
from git_theta import params
from git_theta import utils


class Update:
    """Base class for parameter update plugins."""

    def read(self, param_metadata):
        lfs_pointer = param_metadata.lfs_metadata.lfs_pointer
        serialized_param = git_utils.git_lfs_smudge(lfs_pointer)
        param = params.get_update_serializer().deserialize(serialized_param)
        return param

    def get_last_version(self, repo, path, param_keys, param_metadata):
        last_commit = param_metadata.theta_metadata.last_commit
        logging.debug(f"Getting data from commit {last_commit}")
        if last_commit:
            last_metadata_obj = git_utils.get_file_version(repo, path, last_commit)
            last_metadata = params.Metadata.from_file(last_metadata_obj.data_stream)
            last_param_metadata = last_metadata.flatten()[param_keys]
            return last_param_metadata
        else:
            raise ValueError("Cannot find previous version for parameters")

    @property
    def name(self):
        raise NotImplementedError

    def apply(self, repo, path, param_keys, param_metadata):
        raise NotImplementedError

    def calculate_update(self, repo, path, param_keys, param_metadata, param):
        raise NotImplementedError


def get_update_name(update_type: Optional[str] = None) -> str:
    return update_type or os.environ.get(utils.EnvVarConstants.UPDATE_TYPE) or "dense"


def get_update(update_type: Optional[str] = None) -> Update:
    """Get an Update class by name.

    Parameters
    ----------
    update_type:
        The name of the update type we want to use.

    Returns
    -------
    Update
        The update class. Returned class may be defined in a user installed
        plugin.
    """
    update_name = get_update_name(update_type)
    discovered_plugins = entry_points(group="git_theta.plugins.updates")
    return discovered_plugins[update_name].load()
