"""Test utilities that summarize parameters."""

import random
import pytest
import numpy as np
from git_theta import params


def test_get_shape():
    num_dims = np.random.randint(2, 5)
    shape = tuple(np.random.randint(1, 20, size=num_dims).tolist())
    t = np.random.rand(*shape)
    assert params.get_shape_str(t) == str(shape)


def test_get_shape_list():
    shape = np.random.randint(10, 20)
    t = [random.randint(2, 19) for _ in range(shape)]
    assert params.get_shape_str(t) == str((shape,))


def test_get_shape_list_2d():
    batch, seq = np.random.randint(10, 20, size=2)
    t = []
    for _ in range(batch):
        t_ = []
        for _ in range(seq):
            t_.append(random.randint(0, 10))
        t.append(t_)
    assert params.get_shape_str(t) == str((batch, seq))


def test_get_shape_scalar():
    t = 12
    assert params.get_shape_str(t) == "()"


def test_get_dtype_str():
    for dtype in (np.int32, np.uint8, np.float32, np.float64, np.int64):
        t = np.ones((10, 10), dtype=dtype)
        assert params.get_dtype_str(t) == np.dtype(dtype).str


def test_get_dtype_list_scalar():
    for dtype, convert in (("<f8", float), ("<i8", int)):
        t = [convert(random.randint(0, 100)) for _ in range(10)]
        assert params.get_dtype_str(t) == dtype
        s = convert(random.randint(0, 100))
        assert params.get_dtype_str(t) == dtype


def test_get_hash():
    # We can't really test the hashing, so test that same values hash
    # equal and non-equal hash different.
    t = np.random.rand(3, 4, 5)
    t_ = t.copy()
    # Make sure we aren't referencing the same object
    assert t is not t_
    # Make sure things with the same value hash to the same
    np.testing.assert_allclose(t, t_)
    assert params.get_hash(t) == params.get_hash(t_)

    # Make sure that things with different values hash to different things.
    t_update = np.random.rand(*t.shape)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(t, t_update)
    assert params.get_hash(t) != params.get_hash(t_update)
