"""Tests for our ia3 update."""


import numpy as np
import pytest
from git_theta import async_utils
from git_theta import params
from git_theta.updates import ia3
import scipy.sparse

IN_SHAPE = 100
OUT_SHAPE1 = 300
OUT_SHAPE2 = 300
TRIALS = 50


@pytest.fixture
def updater():
    return ia3.IA3Update(params.get_update_serializer())


def test_ia3_round_trip_application(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(OUT_SHAPE1, OUT_SHAPE2, IN_SHAPE)
        update = np.random.randn(OUT_SHAPE1, OUT_SHAPE2, 1)
        updated_parameter = parameter * update

        calc_update = async_utils.run(
            updater.calculate_update(updated_parameter, parameter)
        )
        result = async_utils.run(updater.apply_update(calc_update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)


def test_ia3_round_trip_application_2d(updater):

    for _ in range(TRIALS):
        parameter = np.random.randn(OUT_SHAPE1, IN_SHAPE)
        update = np.random.randn(OUT_SHAPE1, 1)
        updated_parameter = parameter * update

        calc_update = async_utils.run(
            updater.calculate_update(updated_parameter, parameter)
        )
        result = async_utils.run(updater.apply_update(calc_update, parameter))

        np.testing.assert_allclose(result, updated_parameter, rtol=1e-6)
