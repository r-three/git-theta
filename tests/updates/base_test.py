"""Tests for common update plugin functions."""

import os
import pytest
from git_theta.updates import base
from git_theta import utils


@pytest.fixture
def env_var():
    current_env = dict(os.environ)
    os.environ[utils.EnvVarConstants.UPDATE_TYPE] = "sparse"

    yield
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture
def no_env_var():
    current_env = dict(os.environ)
    os.environ.pop(utils.EnvVarConstants.UPDATE_TYPE, None)

    yield
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture
def empty_env_var():
    current_env = dict(os.environ)
    os.environ[utils.EnvVarConstants.UPDATE_TYPE] = ""

    yield
    os.environ.clear()
    os.environ.update(current_env)


def test_get_update_handler_name_prefers_user_input(env_var):
    """Ensure that user input is always used if provided."""
    user_input = "low-rank"
    assert os.environ.get(utils.EnvVarConstants.UPDATE_TYPE) == "sparse"
    assert base.get_update_handler_name(user_input) == user_input


def test_get_update_handler_name_uses_env_variable(env_var):
    """Ensure env variables are checked before defaulting."""
    user_input = None
    assert base.get_update_handler_name(user_input) == "sparse"


def test_get_update_handler_name_default(no_env_var):
    """Ensure there is a default when there is no input and no defined env var."""
    user_input = None
    assert os.environ.get(utils.EnvVarConstants.UPDATE_TYPE) is None
    assert base.get_update_handler_name(user_input) == "dense"


def test_get_update_handler_name_empty_env(empty_env_var):
    """Ensure there is a default when there is no input and a defined, but empty, env var."""
    user_input = None
    assert utils.EnvVarConstants.UPDATE_TYPE in os.environ
    assert os.environ[utils.EnvVarConstants.UPDATE_TYPE] == ""
    assert base.get_update_handler_name(user_input) == "dense"
