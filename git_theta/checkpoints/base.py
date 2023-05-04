"""Base class and utilities for different checkpoint format backends."""

import fnmatch
import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from git_theta import config, git_utils, utils


@utils.abstract_classattributes("name")
class Checkpoint(dict, metaclass=ABCMeta):
    """Abstract base class for wrapping checkpoint formats."""

    name: str = NotImplemented  # The name of this checkpoint handler, can be used to lookup the plugin.

    @classmethod
    def from_file(cls, checkpoint_path):
        """Create a new Checkpoint object.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file
        """
        return cls(cls.load(checkpoint_path))

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file

        Returns
        -------
        model_dict : dict
            Dictionary mapping parameter names to parameter values.  Parameters
            should be numpy arrays.
        """

    @abstractmethod
    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, np.ndarray))

    def unflatten(self):
        return utils.unflatten(self)

    @classmethod
    def diff(cls, m1: "Checkpoint", m2: "Checkpoint") -> "Checkpoint":
        """Compute the diff between two checkpoints.

        Parameters
        ----------
        m1 : Checkpoint
            The new checkpoint
        m2 : Checkpoint
            The old checkpoint

        Returns
        -------
        added : Checkpoint
            Checkpoint containing the parameter groups added to m1
        removed : Checkpoint
            Checkpoint containing the parameter groups removed from m2
        modified : Checkpoint
            Checkpoint containing the parameter groups modified between m1 and m2
        """
        m1_flat = m1.flatten()
        m2_flat = m2.flatten()
        # N.b.: This is actually faster than set operations on m1 and m2's keys
        added = cls({k: v for k, v in m1_flat.items() if k not in m2_flat}).unflatten()
        removed = cls(
            {k: v for k, v in m2_flat.items() if k not in m1_flat}
        ).unflatten()
        modified = cls(
            {
                k: v
                for k, v in m1_flat.items()
                if k in m2_flat and not np.allclose(v, m2_flat[k])
            }
        ).unflatten()
        return added, removed, modified


def get_checkpoint_handler_name(checkpoint_path: str) -> Optional[str]:
    """Get the name of the checkpoint handler based on entry in .thetaconfig for the current checkpoint path

    Parameters
    ----------
    checkpoint_path
        Path to checkpoint in repo

    Returns
    -------
    str
        Name of the checkpoint handler
    """
    repo = git_utils.get_git_repo()
    checkpoint_path = git_utils.get_relative_path_from_root(repo, checkpoint_path)
    thetaconfig = config.ThetaConfigFile(repo)
    checkpoint_config = thetaconfig.get_config(checkpoint_path)
    return checkpoint_config.get("checkpoint_format", None)


def get_checkpoint_handler(checkpoint_path: str) -> Checkpoint:
    """Get the checkpoint handler for the current checkpoint path

    Gets the checkpoint handler from an entry in .thetaconfig

    Parameters
    ----------
    checkpoint_path
        Path to the checkpoint

    Returns
    -------
    Checkpoint
        The checkpoint handler (usually an instance of `git_theta.checkpoints.Checkpoint`).
        Returned handler may be defined in a user installed plugin.
    """
    checkpoint_type = get_checkpoint_handler_name(checkpoint_path)
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoints")
    return discovered_plugins[checkpoint_type].load()
