"""Tests for params.py"""

import random
import pytest
import numpy as np
from git_theta import params


def test_tensorstore_serializer_roundtrip():
    """
    Test TensorStoreSerializer serializes and deserializes correctly
    """
    serializer = params.TensorStoreSerializer()
    for num_dims in range(1, 6):
        shape = tuple(np.random.randint(1, 20, size=num_dims).tolist())
        t = np.random.rand(*shape)
        serialized_t = serializer.serialize(t)
        deserialized_t = serializer.deserialize(serialized_t)
        assert np.all(t == deserialized_t)


def test_tensorstore_serializer_roundtrip_chunked():
    """
    Test TensorstoreSerializer serializes and deserializes correctly on a large tensor (that should get chunked)
    """
    serializer = params.TensorStoreSerializer()
    t = np.random.rand(5000, 5000)
    serialized_t = serializer.serialize(t)
    deserialized_t = serializer.deserialize(serialized_t)
    assert np.all(t == deserialized_t)


def test_tar_combiner_roundtrip():
    """
    Test TarCombiner combines and splits correctly
    """
    combiner = params.TarCombiner()
    param_files = {
        "param1": {"file1": b"abcdefg", "file2": b"hijklmnop"},
        "param2": {"file1": b"01234", "file2": b"56789"},
    }
    combined_param_files = combiner.combine(param_files)
    split_param_files = combiner.split(combined_param_files)
    assert param_files == split_param_files


def test_update_serializer_roundtrip():
    """
    Test UpdateSerializer made from TensorStoreSerializer and TarCombiner serializes and deserializes correctly
    """
    serializer = params.UpdateSerializer(
        params.TensorStoreSerializer(), params.TarCombiner()
    )
    update_params = {
        "param1": np.random.rand(100, 100),
        "param2": np.random.rand(50, 10, 2),
        "param3": np.random.rand(1000),
    }
    serialized_update_params = serializer.serialize(update_params)
    deserialized_update_params = serializer.deserialize(serialized_update_params)

    assert update_params.keys() == deserialized_update_params.keys()

    for param_name in update_params.keys():
        assert np.all(
            update_params[param_name] == deserialized_update_params[param_name]
        )
