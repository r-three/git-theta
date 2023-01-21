"""Tests for checkpoints.py"""

import os
import pytest

from git_theta import checkpoints


ENV_CHECKPOINT_TYPE = "GIT_THETA_CHECKPOINT_TYPE"

pytest.importorskip("pytorch")


@pytest.fixture
def env_var():
    current_env = dict(os.environ)
    os.environ[ENV_CHECKPOINT_TYPE] = "env_variable_handler"

    yield
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture
def no_env_var():
    current_env = dict(os.environ)
    os.environ.pop(ENV_CHECKPOINT_TYPE, None)

    yield
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture
def empty_env_var():
    current_env = dict(os.environ)
    os.environ[ENV_CHECKPOINT_TYPE] = ""

    yield
    os.environ.clear()
    os.environ.update(current_env)


def test_get_checkpoint_handler_name_user_input(env_var):
    """Check that function prefers user input to environment variable"""

    user_input = "user_input_handler"
    name = checkpoints.get_checkpoint_handler_name(user_input)
    assert name == user_input


def test_get_checkpoint_handler_name_env_variable(env_var):
    """Check that function uses environment variable no user input specified"""

    name = checkpoints.get_checkpoint_handler_name()
    assert name == "env_variable_handler"


def test_get_checkpoint_handler_name_default1(no_env_var):
    """Check that function has correct default behavior with no user input and environment variable"""

    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


def test_get_checkpoint_handler_name_default2(empty_env_var):
    """Check that function has correct default behavior with no user input and environment variable is empty string"""

    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


# TODO: Move this (and other pytorch checkpoint tests) to new file. Remove the
# importorskip too.
def test_get_checkpoint_handler_pytorch(no_env_var):
    """Check that checkpoint_handler type is correct for when checkpoint_handler name resolves to pytorch"""

    out = checkpoints.get_checkpoint_handler("pytorch")
    assert out == checkpoints.pickled_dict_checkpoint.PickledDictCheckpoint
