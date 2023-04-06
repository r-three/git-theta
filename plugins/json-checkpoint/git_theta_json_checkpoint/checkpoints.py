#!/usr/bin/env python3

import io
import json

from git_theta import checkpoints


class JSONCheckpoint(checkpoints.Checkpoint):
    """Class for prototyping with JSON checkpoints"""

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
