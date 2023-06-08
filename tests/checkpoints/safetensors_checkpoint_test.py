"""safetensors checkpoint tests."""

import operator as op
import os

import numpy as np
import pytest

from git_theta import checkpoints, utils
from git_theta.checkpoints import safetensors_checkpoint


@pytest.fixture
def fake_model():
    return {
        "layer1/weight": np.random.rand(1024, 1024),
        "layer1/bias": np.random.rand(1024),
        "layer2/weight": np.random.rand(512, 1024),
        "layer2/bias": np.random.rand(512),
    }


def test_round_trip(fake_model):
    with utils.named_temporary_file() as f:
        ckpt = safetensors_checkpoint.SafeTensorsCheckpoint(fake_model)
        ckpt.save(f.name)
        f.flush()
        f.close()
        ckpt2 = safetensors_checkpoint.SafeTensorsCheckpoint.from_file(f.name)
    for (_, og), (_, new) in zip(
        sorted(ckpt.items(), key=op.itemgetter(0)),
        sorted(ckpt2.items(), key=op.itemgetter(0)),
    ):
        np.testing.assert_array_equal(og, new)


def test_get_checkpoint_handler_safetensors():
    for alias in ("safetensors", "safetensors-checkpoint"):
        out = checkpoints.get_checkpoint_handler(alias)
        assert out == safetensors_checkpoint.SafeTensorsCheckpoint
