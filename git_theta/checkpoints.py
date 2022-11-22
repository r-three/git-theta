"""Backends for different checkpoint formats."""

import torch
import os
import json
import io
import sys
from typing import Optional

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from file_or_name import file_or_name

from . import utils

# Maintain access via checkpoints module for now.
from .utils import iterate_dict_leaves, iterate_dir_leaves


class Checkpoint(dict):
    """Abstract base class for wrapping checkpoint formats."""

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
    def load(cls, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file

        Returns
        -------
        model_dict : dict
            Dictionary mapping parameter names to parameter values
        """
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """
        raise NotImplementedError


class PickledDictCheckpoint(Checkpoint):
    """Class for wrapping picked dict checkpoints, commonly used with PyTorch."""

    @classmethod
    @file_or_name(checkpoint_path="rb")
    def load(cls, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file

        Returns
        -------
        model_dict : dict
            Dictionary mapping parameter names to parameter values
        """
        model_dict = torch.load(io.BytesIO(checkpoint_path.read()))
        if not isinstance(model_dict, dict):
            raise ValueError("Supplied PyTorch checkpoint must be a dict.")
        if not all(isinstance(k, str) for k in model_dict.keys()):
            raise ValueError("All PyTorch checkpoint keys must be strings.")
        if not all(isinstance(v, torch.Tensor) for v in model_dict.values()):
            raise ValueError("All PyTorch checkpoint values must be tensors.")
        return {k: v.cpu().numpy() for k, v in model_dict.items()}

    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """
        checkpoint_dict = {k: torch.as_tensor(v) for k, v in self.items()}
        torch.save(checkpoint_dict, checkpoint_path)


def iterate_dir_leaves(root):
    """
    Generator that iterates through files in a directory tree and produces (path, dirs) tuples where
    path is the file's path and dirs is the sequence of path components from root to the file.

    Example
    -------
    root
    ├── a
    │   ├── c
    │   └── d
    └── b
        └── e

    iterate_dir_leaves(root) --> ((root/a/c, ['a','c']), (root/a/d, ['a','d']), (root/b/e, ['b','e']))

    Parameters
    ----------
    root : str
        Root of directory tree to iterate over

    Returns
    -------
    generator
        generates directory tree leaf, subdirectory list tuples
    """

    def _iterate_dir_leaves(root, prefix):
        for d in os.listdir(root):
            dir_member = os.path.join(root, d)
            if not "params" in os.listdir(dir_member):
                yield from _iterate_dir_leaves(dir_member, prefix=prefix + [d])
            else:
                yield (dir_member, prefix + [d])

    return _iterate_dir_leaves(root, [])


def get_checkpoint_handler(checkpoint_type: Optional[str] = None) -> Checkpoint:
    """Get the checkpoint handler either by name or from an environment variable.

    Gets the checkpoint handler either for the `checkpoint_type` argument or `$GIT_THETA_CHECKPOINT_TYPE` environment variable.
    Defaults to pytorch when neither are defined.

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    Checkpoint
        The checkpoint handler (usually an instance of `git_theta.checkpoints.Checkpoint`). Returned handler may be defined in a user installed
        plugin.
    """
    # TODO(bdlester): Find a better way to include checkpoint type information
    # in git clean filters that are run without `git theta add`.
    # TODO: Don't default to pytorch once other checkpoint formats are supported.
    checkpoint_type = (
        checkpoint_type or os.environ.get("GIT_THETA_CHECKPOINT_TYPE") or "pytorch"
    )
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoints")
    checkpoint_handler = discovered_plugins[checkpoint_type].load()
    return checkpoint_handler
