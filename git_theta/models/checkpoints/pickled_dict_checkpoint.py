#!/usr/bin/env python3


import io
import torch
from git_theta.models.checkpoints import Checkpoint


class PickledDictCheckpoint(Checkpoint):
    """Class for wrapping picked dict checkpoints, commonly used with PyTorch."""

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
            checkpoint_path = io.BytesIO(checkpoint_path.read())

        model_dict = torch.load(checkpoint_path)
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
