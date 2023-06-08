"""Tests for metadata.py"""

import os

import numpy as np
import pytest

from git_theta import metadata, utils


def metadata_equal(m1, m2):
    m1_flat = m1.flatten()
    m2_flat = m2.flatten()
    if m1_flat.keys() != m2_flat.keys():
        return False
    for k, m1_v in m1_flat.items():
        if m1_v != m2_flat[k]:
            return False
    return True


def test_lfs_pointer(data_generator):
    """
    Test LfsMetadata creates and reads LFS pointers correctly
    """
    lfs_metadata1 = data_generator.random_lfs_metadata()
    pointer_contents = lfs_metadata1.lfs_pointer
    lfs_metadata2 = metadata.LfsMetadata.from_pointer(pointer_contents)
    assert lfs_metadata1 == lfs_metadata2


# TODO: This test will sometimes fail due to the current TensorMetadata equality check. Fix this eventually.
@pytest.mark.xfail
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


def test_param_metadata_roundtrip(data_generator):
    """
    Test that ParamMetadata serializes to dict and can be generated from dict correctly
    """
    param_metadata = data_generator.random_param_metadata()
    metadata_dict = param_metadata.serialize()
    param_metadata_roundtrip = metadata.ParamMetadata.from_metadata_dict(metadata_dict)

    assert param_metadata == param_metadata_roundtrip


def test_metadata_dict_roundtrip(data_generator):
    """
    Test that Metadata serializes to dict and can be generated from dict correctly
    """
    metadata_obj = data_generator.random_metadata()
    metadata_dict = metadata_obj.serialize()
    metadata_roundtrip = metadata.Metadata.from_metadata_dict(metadata_dict)
    assert metadata_equal(metadata_obj, metadata_roundtrip)


def test_metadata_file_roundtrip(data_generator):
    """
    Test that Metadata serializes to file and can be generated from file correctly
    """
    metadata_obj = data_generator.random_metadata()
    with utils.named_temporary_file() as tmp:
        metadata_obj.write(tmp)
        tmp.flush()
        tmp.close()
        metadata_roundtrip = metadata.Metadata.from_file(tmp.name)
    assert metadata_equal(metadata_obj, metadata_roundtrip)


def test_metadata_flatten(data_generator):
    """
    Test that Metadata flattens and unflattens correctly
    """
    metadata_obj = data_generator.random_metadata()
    metadata_obj_flat = metadata_obj.flatten()
    metadata_obj_unflat = metadata_obj_flat.unflatten()
    assert metadata_equal(metadata_obj, metadata_obj_unflat)
