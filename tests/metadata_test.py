"""Tests for metadata.py"""

import string
import random
import tempfile
import numpy as np

from git_theta import metadata
from tests.testing_utils import (
    random_lfs_metadata,
    random_tensor_metadata,
    random_theta_metadata,
    random_param_metadata,
    random_metadata,
)


def metadata_equal(m1, m2):
    m1_flat = m1.flatten()
    m2_flat = m2.flatten()
    if m1_flat.keys() != m2_flat.keys():
        return False
    for k in m1_flat.keys():
        if m1_flat[k] != m2_flat[k]:
            return False
    return True


def test_lfs_pointer():
    """
    Test LfsMetadata creates and reads LFS pointers correctly
    """
    version = "my_version"
    oid = "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])
    size = "12345"
    lfs_metadata1 = random_lfs_metadata()
    pointer_contents = lfs_metadata1.lfs_pointer
    lfs_metadata2 = metadata.LfsMetadata.from_pointer(pointer_contents)
    assert lfs_metadata1 == lfs_metadata2


def test_tensor_metadata_machine_epsilon():
    """
    Test that TensorMetadata objects made from tensors with difference within machine epsilon are equal to one another
    """
    tensor1 = np.random.rand(10, 10)
    tensor2 = (
        tensor1
        + (2 * np.random.choice(2, tensor1.shape) - 1) * np.finfo(np.float32).eps
    )
    tensor_metadata1 = metadata.TensorMetadata.from_tensor(tensor1)
    tensor_metadata2 = metadata.TensorMetadata.from_tensor(tensor2)
    assert tensor_metadata1 == tensor_metadata2


def test_param_metadata_roundtrip():
    """
    Test that ParamMetadata serializes to dict and can be generated from dict correctly
    """
    param_metadata = random_param_metadata()
    metadata_dict = param_metadata.serialize()
    param_metadata_roundtrip = metadata.ParamMetadata.from_metadata_dict(metadata_dict)

    assert param_metadata == param_metadata_roundtrip


def test_metadata_dict_roundtrip():
    """
    Test that Metadata serializes to dict and can be generated from dict correctly
    """
    metadata_obj = random_metadata()
    metadata_dict = metadata_obj.serialize()
    metadata_roundtrip = metadata.Metadata.from_metadata_dict(metadata_dict)
    if metadata_obj != metadata_roundtrip:
        for k in metadata_obj.flatten().keys():
            if metadata_obj.flatten()[k] != metadata_roundtrip.flatten()[k]:
                print(metadata_obj.flatten()[k], metadata_roundtrip.flatten()[k])
    assert metadata_equal(metadata_obj, metadata_roundtrip)


def test_metadata_file_roundtrip():
    """
    Test that Metadata serializes to file and can be generated from file correctly
    """
    metadata_obj = random_metadata()
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        metadata_obj.write(tmp)
        tmp.flush()
        metadata_roundtrip = metadata.Metadata.from_file(tmp.name)
    assert metadata_equal(metadata_obj, metadata_roundtrip)


def test_metadata_flatten():
    """
    Test that Metadata flattens and unflattens correctly
    """
    metadata_obj = random_metadata()
    metadata_obj_flat = metadata_obj.flatten()
    metadata_obj_unflat = metadata_obj_flat.unflatten()
    assert metadata_equal(metadata_obj, metadata_obj_unflat)
