"""Checkpoints that use a pickled dict format like pytorch."""

import io

import torch

from git_theta.checkpoints import Checkpoint


class PickledDictCheckpoint(Checkpoint):
    """Class for wrapping picked dict checkpoints, commonly used with PyTorch."""

    name: str = "pickled_dict"

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
        return model_dict

    @classmethod
    def from_framework(cls, model_dict):
        return cls({k: v.cpu().numpy() for k, v in model_dict.items()})

    def to_framework(self):
        return {k: torch.as_tensor(v) for k, v in self.items()}

    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """
        checkpoint_dict = self.to_framework()
        torch.save(checkpoint_dict, checkpoint_path)
