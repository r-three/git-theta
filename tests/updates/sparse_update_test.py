"""Tests for our sparse Update."""


import numpy as np
import pytest
import scipy.sparse

from git_theta import async_utils, params
from git_theta.updates import sparse

SHAPE = 100
NUM_UPDATES = 1000
TRIALS = 50


@pytest.fixture
def updater():
    return lambda threshold: sparse.SparseUpdate(
        params.get_update_serializer(), threshold
    )


def test_sparse_round_trip_application(updater):
    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE, SHAPE, SHAPE)
        x, y, z = np.random.choice(
            np.arange(SHAPE), size=(3, NUM_UPDATES), replace=True
        )
        sparse_update = np.random.randn(NUM_UPDATES)
        updated_parameter = parameter.copy()
        updated_parameter[x, y, z] = sparse_update

        sparse_updater = updater(1e-12)
        update = async_utils.run(
            sparse_updater.calculate_update(updated_parameter, parameter)
        )
        result = async_utils.run(sparse_updater.apply_update(update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)


def test_known_sparsity(updater):
    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE, SHAPE, SHAPE)
        diff_tensor = np.random.randn(SHAPE, SHAPE, SHAPE)
        # To ensure there is no sparsity in diff tensor in the first place
        diff_tensor[diff_tensor == 0] = 0.1
        threshold = np.quantile(diff_tensor, 0.3)
        diff_tensor[diff_tensor < threshold] = 0
        updated_parameter = parameter + diff_tensor

        sparse_updater = updater(1e-12)
        update_dict = async_utils.run(
            sparse_updater.calculate_update(updated_parameter, parameter)
        )
        calc_sparsity = 1 - len(update_dict["data"]) / np.prod(parameter.shape)
        np.testing.assert_allclose(calc_sparsity, 0.3, rtol=1e-6)


def test_monotonic_increasing_sparseness(updater):
    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE, SHAPE, SHAPE)
        diff_tensor = np.random.randn(SHAPE, SHAPE, SHAPE)
        threshold = np.quantile(diff_tensor, 0.3)
        diff_tensor[diff_tensor < threshold] = 0
        updated_parameter = parameter + diff_tensor
        sparseness = []
        for threshold in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
            sparse_updater = updater(threshold)
            update_dict = async_utils.run(
                sparse_updater.calculate_update(updated_parameter, parameter)
            )
            sparsity = 1 - len(update_dict["data"]) / np.prod(parameter.shape)
            sparseness.append(sparsity)
        assert all(
            sparseness[i] <= sparseness[i + 1] for i in range(len(sparseness) - 1)
        )
