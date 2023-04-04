"""Checkpoint plugin for Flax's msgpack format."""

from git_theta.checkpoints import Checkpoint
from flax import serialization
from file_or_name import file_or_name


class FlaxCheckpoint(Checkpoint):
    """Load a msgpack based Flax Checkpoint."""

    name: str = "flax"

    @classmethod
    @file_or_name(checkpoint_path="rb")
    def load(cls, checkpoint_path, **kwargs):
        return serialization.msgpack_restore(checkpoint_path.read())

    @file_or_name(checkpoint_path="wb")
    def save(self, checkpoint_path, **kwargs):
        checkpoint_path.write(serialization.msgpack_serialize(dict(self)))
