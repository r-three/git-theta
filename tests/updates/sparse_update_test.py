"""Tests for our sparse Update."""


import numpy as np
import pytest
from git_theta import async_utils
from git_theta import params
from git_theta.updates import sparse
import scipy.sparse

SHAPE = 100
NUM_UPDATES = 100
TRIALS = 50


@pytest.fixture
def updater():
    return sparse.SparseUpdate(params.get_update_serializer())


def test_sparse_update_application(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE, SHAPE, SHAPE)
        x, y, z = np.random.choice(
            np.arange(SHAPE), size=(3, NUM_UPDATES), replace=True
        )
        sparse_update = np.random.randn(NUM_UPDATES)
        updated_parameter = parameter.copy()
        updated_parameter[x, y, z] = sparse_update

        update = async_utils.run(updater.calculate_update(updated_parameter, parameter))
        result = async_utils.run(updater.apply_update(update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)
