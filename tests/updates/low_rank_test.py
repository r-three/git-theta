"""Tests for our low Rank Update."""


import numpy as np
import pytest
from git_theta import async_utils
from git_theta import params
from git_theta.updates import low_rank


K = 20
INPUT_SIZE = 1024
OUTPUT_SIZE = 1024
TRIALS = 50


@pytest.fixture
def updater():
    return low_rank.LowRankUpdate(params.get_update_serializer())


def test_low_rank_update_rank_inference(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(INPUT_SIZE, OUTPUT_SIZE)
        R = np.random.randn(INPUT_SIZE, K)
        C = np.random.randn(K, OUTPUT_SIZE)
        updated_parameter = R @ C + parameter

        update = async_utils.run(updater.calculate_update(updated_parameter, parameter))
        assert update["R"].shape == R.shape
        assert update["C"].shape == C.shape


@pytest.mark.xfail(strict=False)
def test_low_rank_update_application(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(INPUT_SIZE, OUTPUT_SIZE)
        R = np.random.randn(INPUT_SIZE, K)
        C = np.random.randn(K, OUTPUT_SIZE)
        updated_parameter = R @ C + parameter

        update = async_utils.run(updater.calculate_update(updated_parameter, parameter))
        result = async_utils.run(updater.apply_update(update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)


def test_low_rank_update_application_1d(updater):
    parameter = np.random.randn(INPUT_SIZE)
    update = np.random.randn(*parameter.shape)

    updated_parameter = update + parameter

    calculated_update = async_utils.run(
        updater.calculate_update(updated_parameter, parameter)
    )
    calculated_result = async_utils.run(
        updater.apply_update(calculated_update, parameter)
    )

    np.testing.assert_allclose(calculated_result, updated_parameter)
