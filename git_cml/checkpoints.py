"""Backends for different checkpoint formats."""

import torch
import os
import json
import io


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
        if isinstance(checkpoint_path, io.IOBase):
            checkpoint_path = io.BytesIO(checkpoint_path.read())

        model_dict = torch.load(checkpoint_path)
        if not isinstance(model_dict, dict):
            raise ValueError("Supplied PyTorch checkpoint must be a dict.")
        if not all(isinstance(k, str) for k in model_dict.keys()):
            raise ValueError("All PyTorch checkpoint keys must be strings.")
        if not all(isinstance(v, torch.Tensor) for v in model_dict.values()):
            raise ValueError("All PyTorch checkpoint values must be tensors.")
        return {k: v.numpy() for k, v in model_dict.items()}

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
        if isinstance(checkpoint_path, io.IOBase):
            return json.load(checkpoint_path)
        else:
            with open(checkpoint_path, "r") as f:
                return json.load(f)

    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """
        if isinstance(checkpoint_path, io.IOBase):
            json.dump(self, checkpoint_path)
        else:
            with open(checkpoint_path, "w") as f:
                json.dump(self, f)


def iterate_dict_leaves(d):
    """
    Generator that iterates through a dictionary and produces (leaf, keys) tuples where leaf is a dictionary leaf
    and keys is the sequence of keys used to access leaf. Dictionary is iterated in depth-first
    order with lexicographic ordering of keys.

    Example
    -------
    d = {'a': {'b': {'c': 10, 'd': 20, 'e': 30}}}
    iterate_dict_leaves(d) --> ((10, ['a','b','c']), (20, ['a','b','d']), (30, ['a','b','e']))

    Parameters
    ----------
    d : dict
        dictionary to iterate over

    Returns
    -------
    generator
        generates dict leaf, key path tuples
    """

    def _iterate_dict_leaves(d, prefix):
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                yield from _iterate_dict_leaves(v, prefix + [k])
            else:
                yield (v, prefix + [k])

    return _iterate_dict_leaves(d, [])


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
            if os.path.isdir(dir_member):
                yield from _iterate_dir_leaves(dir_member, prefix=prefix + [d])
            else:
                yield (dir_member, prefix + [d])

    return _iterate_dir_leaves(root, [])
