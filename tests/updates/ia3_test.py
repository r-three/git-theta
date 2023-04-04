"""Tests for our ia3 update."""


import numpy as np
import pytest
from git_theta import async_utils
from git_theta import params
from git_theta.updates import ia3
import scipy.sparse

SHAPE1 = 3
SHAPE2 = 30
SHAPE3 = 30
SHAPE4 = 30

TRIALS = 50


@pytest.fixture
def updater():
    return ia3.IA3Update(params.get_update_serializer())


def test_ia3_round_trip_application(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE1, SHAPE2, SHAPE3, SHAPE4)
        update = np.random.randn(SHAPE1, SHAPE2, 1, SHAPE4)
        updated_parameter = parameter * update

        calc_update = async_utils.run(
            updater.calculate_update(updated_parameter, parameter, broadcast_dims=[2])
        )
        result = async_utils.run(updater.apply_update(calc_update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)


def test_ia3_round_trip_application_with_moredims(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE1, SHAPE2, SHAPE3, SHAPE4)
        update = np.random.randn(1, SHAPE2, SHAPE3, 1)
        updated_parameter = parameter * update

        calc_update = async_utils.run(
            updater.calculate_update(
                updated_parameter, parameter, broadcast_dims=[0, 3]
            )
        )
        result = async_utils.run(updater.apply_update(calc_update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)


def test_ia3_round_trip_application_with_sparse_parameter(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(SHAPE1, SHAPE2, SHAPE3, SHAPE4)
        update = np.random.randn(SHAPE1, SHAPE2, 1, 1)
        threshold = np.quantile(parameter, 0.3)
        parameter[parameter < threshold] = 0
        updated_parameter = parameter * update

        calc_update = async_utils.run(
            updater.calculate_update(
                updated_parameter, parameter, broadcast_dims=[2, 3]
            )
        )
        result = async_utils.run(updater.apply_update(calc_update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)
