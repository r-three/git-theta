"""Backends for different checkpoint formats."""

import torch
import os
import json
import io

<<<<<<< HEAD
from file_or_name import file_or_name
=======
from . import utils
>>>>>>> b367c80 (Add unittest to iterate_dir_leaves to help refactor)

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
        return cls(cls._load(checkpoint_path))

    @classmethod
    def _load(cls, checkpoint_path):
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


class PyTorchCheckpoint(Checkpoint):
    """Class for wrapping PyTorch checkpoints."""

    @classmethod
    @file_or_name(checkpoint_path="rb")
    def _load(cls, checkpoint_path):
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


class JSONCheckpoint(Checkpoint):
    """Class for prototyping with JSON checkpoints"""

    @classmethod
    @file_or_name(checkpoint_path="r")
    def _load(cls, checkpoint_path):
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
        return json.load(checkpoint_path)

    @file_or_name(checkpoint_path="w")
    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """
        json.dump(self, checkpoint_path)


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
