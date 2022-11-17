"""Tensorflow checkpoint tests."""

import os
from unittest import mock
import tempfile
import pytest
import numpy as np

# Skip all these tests if tensorflow is not installed
tf = pytest.importorskip("tensorflow")

from git_theta import checkpoints
from git_theta.checkpoints import tensorflow_checkpoint


INPUT_SIZE = 10


@pytest.fixture(autouse=True)
def hide_cuda():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        yield


class InnerLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="inner")
        self.dense = tf.keras.layers.Dense(5)
        # Add a non-trainable parameter for completeness.
        self.scale = tf.Variable(12.0, name="scalar", trainable=False)

    def call(self, x):
        return self.dense(x) * self.scale


class DemoModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.inner = InnerLayer()
        self.logits = tf.keras.layers.Dense(2, name="logits")

    def call(self, x):
        return self.logits(self.inner(x))


def make_fake_model():
    dm = DemoModel()
    dm(np.zeros((1, INPUT_SIZE)))
    return dm


@pytest.fixture
def fake_model():
    return make_fake_model()


def test_round_trip(fake_model):
    with tempfile.NamedTemporaryFile() as f:
        # Make a checkpoint via tensorflow
        fake_model.save_weights(f.name)
        # Load the Checkpoint
        ckpt = tensorflow_checkpoint.TensorFlowCheckpoint.from_file(f.name)
        with tempfile.NamedTemporaryFile() as f2:
            # Use the git-theta save to create a new checkpoint
            ckpt.save(f2.name)
            # Load a model from they checkpoint we just saved
            loaded_model = make_fake_model()
            loaded_model.load_weights(f2.name)
    for og, new in zip(fake_model.variables, loaded_model.variables):
        np.testing.assert_allclose(og.numpy(), new.numpy())


def test_round_trip_with_modifications(fake_model):
    with tempfile.NamedTemporaryFile() as f:
        # Make a checkpoint via tensorflow
        fake_model.save_weights(f.name)
        # Load the Checkpoint
        ckpt = tensorflow_checkpoint.TensorFlowCheckpoint.from_file(f.name)
        # Update value
        ckpt["logits"]["bias"] = np.ones_like(ckpt["logits"]["bias"])
        with tempfile.NamedTemporaryFile() as f2:
            # Use the git-theta save to create a new checkpoint
            ckpt.save(f2.name)
            # Load a model from they checkpoint we just saved
            loaded_model = make_fake_model()
            loaded_model.load_weights(f2.name)
    for og, new in zip(fake_model.variables, loaded_model.variables):
        # Check that the updated value is loaded.
        if "logits" in new.name and "bias" in new.name:
            new_numpy = new.numpy()
            np.testing.assert_allclose(new_numpy, np.ones(*new_numpy.shape))
        else:
            np.testing.assert_allclose(og.numpy(), new.numpy())


def test_get_checkpoint_handler_tensorflow():
    out = checkpoints.get_checkpoint_handler("tensorflow-checkpoint")
    assert out == tensorflow_checkpoint.TensorFlowCheckpoint
