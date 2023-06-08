"""Tests for checkpoints.py"""

import os
import subprocess

import pytest

from git_theta import checkpoints

torch = pytest.importorskip("torch")


@pytest.fixture
def fake_model():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )


def test_get_checkpoint_handler_pytorch(git_repo, fake_model):
    """Check that checkpoint_handler type is correct for when checkpoint_handler name resolves to pytorch"""
    torch.save(fake_model, "model.bin")
    subprocess.run("git theta track model.bin --checkpoint_format pytorch".split(" "))
    out = checkpoints.get_checkpoint_handler("model.bin")
    assert out == checkpoints.pickled_dict_checkpoint.PickledDictCheckpoint
