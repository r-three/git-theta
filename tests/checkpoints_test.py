"""Tests for checkpoints.py"""

import os
import pytest

from git_theta import checkpoints


@pytest.fixture
def env_var():
    current_env = dict(os.environ)
    os.environ["GIT_THETA_CHECKPOINT_TYPE"] = "env_variable_handler"

    yield
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture
def no_env_var():
    current_env = dict(os.environ)
    os.environ.pop("GIT_THETA_CHECKPOINT_TYPE", None)

    yield
    os.environ.clear()
    os.environ.update(current_env)


def test_get_checkpoint_handler_name_user_input(env_var):
    user_input = "user_input_handler"
    name = checkpoints.get_checkpoint_handler_name(user_input)
    assert name == user_input


def test_get_checkpoint_handler_name_env_variable(env_var):
    name = checkpoints.get_checkpoint_handler_name()
    assert name == "env_variable_handler"


def test_get_checkpoint_handler_name_default1(no_env_var):
    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


def test_get_checkpoint_handler_name_default2(no_env_var):
    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


def test_get_checkpoint_handler_pytorch(no_env_var):
    out = checkpoints.get_checkpoint_handler("pytorch")
    assert out == checkpoints.PickledDictCheckpoint
