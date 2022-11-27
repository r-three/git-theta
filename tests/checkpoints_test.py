"""Tests for checkpoints.py"""

import os

from git_theta import checkpoints


def test_get_checkpoint_handler_name_user_input():
    user_input = "user_input_handler"
    env_variable = "env_variable_handler"
    os.environ["GIT_THETA_CHECKPOINT_TYPE"] = env_variable
    name = checkpoints.get_checkpoint_handler_name(user_input)
    assert name == user_input


def test_get_checkpoint_handler_name_env_variable():
    env_variable = "env_variable_handler"
    os.environ["GIT_THETA_CHECKPOINT_TYPE"] = env_variable
    name = checkpoints.get_checkpoint_handler_name()
    assert name == env_variable


def test_get_checkpoint_handler_name_default1():
    os.environ.pop("GIT_THETA_CHECKPOINT_TYPE")
    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


def test_get_checkpoint_handler_name_default2():
    os.environ["GIT_THETA_CHECKPOINT_TYPE"] = ""
    name = checkpoints.get_checkpoint_handler_name()
    assert name == "pytorch"


def test_get_checkpoint_handler_pytorch():
    out = checkpoints.get_checkpoint_handler("pytorch")
    assert out == checkpoints.PickledDictCheckpoint
