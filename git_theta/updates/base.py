"""Base class for parameter update plugins."""

import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from typing import Optional


class Update:
    """Base class for parameter update plugins."""

    # def __init__(self, checkpoint_path, git_theta_model_dir):
    #     self.checkpoint_path = checkpoint_path
    #     self.git_theta_model_dir = git_theta_model_dir
    @property
    def name(self):
        raise NotImplementedError

    def read(self, path, commit: Optional[str] = None):
        raise NotImplementedError

    def write(self, path, parameter):
        raise NotImplementedError

    def apply(self, path, commit: Optional[str] = None):
        raise NotImplementedError


def get_update(update_type: str) -> Update:
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
    discovered_plugins = entry_points(group="git_theta.plugins.updates")
    return discovered_plugins[update_type].load()
