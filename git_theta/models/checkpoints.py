"""Backends for different checkpoint formats."""

import os
import json
import io
import sys
from typing import Optional
import numpy as np
from abc import ABCMeta, abstractmethod

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from file_or_name import file_or_name

from git_theta import utils
from git_theta.models import Model


class Checkpoint(Model):
    """Abstract base class for wrapping checkpoint formats."""

    @property
    @abstractmethod
    def name(self):
        """The name of this checkpoint handler, can be used to lookup the plugin."""

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
            Dictionary mapping parameter names to parameter values. Parameters
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

    @staticmethod
    def is_leaf(l):
        return isinstance(l, np.ndarray)

    @staticmethod
    def leaves_equal(l1, l2):
        return np.allclose(l1, l2)


class PickledDictCheckpoint(Checkpoint):
    """Class for wrapping picked dict checkpoints, commonly used with PyTorch."""

    @property
    def name(self):
        return "pickled-dict"

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
            Dictionary mapping parameter names to parameter values. Parameters
            should be numpy arrays.
        """
        # TODO(bdlester): Once multiple checkpoint types are supported and
        # this checkpoint object is moved to its own module, move this import
        # back to toplevel. Currently it is inside the object methods to allow
        # for optional framework installs.
        import torch

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
        # TODO(bdlester): Once multiple checkpoint types are supported and
        # this checkpoint object is moved to its own module, move this import
        # back to toplevel. Currently it is inside the object methods to allow
        # for optional framework installs.
        import torch

        checkpoint_dict = {k: torch.as_tensor(v) for k, v in self.items()}
        torch.save(checkpoint_dict, checkpoint_path)


def get_checkpoint_handler_name(checkpoint_type: Optional[str] = None) -> str:
    """Get the name of the checkpoint handler to use.

    Order of precedence is
    1. `checkpoint_type` argument
    2. `$GIT_THETA_CHECKPOINT_TYPE` environment variable
    3. default value (currently pytorch)

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    str
        Name of the checkpoint handler
    """
    # TODO(bdlester): Find a better way to include checkpoint type information
    # in git clean filters that are run without `git theta add`.
    # TODO: Don't default to pytorch once other checkpoint formats are supported.
    return (
        checkpoint_type
        or os.environ.get(utils.EnvVarConstants.CHECKPOINT_TYPE)
        or "pytorch"
    )


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
    checkpoint_type = get_checkpoint_handler_name(checkpoint_type)
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoints")
    return discovered_plugins[checkpoint_type].load()
