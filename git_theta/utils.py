"""Utilities for git theta"""


import operator as op
import os
from typing import Dict, Any, Tuple, Union, Callable
import contextlib


class EnvVarConstants:
    CHECKPOINT_TYPE: str = "GIT_THETA_CHECKPOINT_TYPE"
    UPDATE_TYPE: str = "GIT_THETA_UPDATE_TYPE"


def flatten(
    d: Dict[str, Any],
    is_leaf: Callable[[object], bool] = lambda v: not isinstance(v, dict),
) -> Dict[Tuple[str, ...], Any]:
    """Flatten a nested dictionary.

    Parameters
    ----------
    d:
        The nested dictionary to flatten.

    Returns
    -------
    Dict[Tuple[str, ...], Any]
        The flattened version of the dictionary where the key is now a tuple
        of keys representing the path of keys to reach the value in the nested
        dictionary.
    """

    def _flatten(d, prefix: Tuple[str] = ()):
        flat = type(d)({})
        for k, v in d.items():
            if not is_leaf(v):
                flat.update(_flatten(v, prefix=prefix + (k,)))
            else:
                flat[prefix + (k,)] = v
        return flat

    return _flatten(d)


def unflatten(d: Dict[Tuple[str], Any]) -> Dict[str, Union[Dict[str, Any], Any]]:
    """Unflatten a dict into a nested one.

    Parameters
    ----------
    d:
        The dictionary to unflatten. Each key should be a tuple of keys the
        represent the nesting.

    Returns
    Dict
        The nested version of the dictionary.
    """
    nested = type(d)({})
    for ks, v in d.items():
        curr = nested
        for k in ks[:-1]:
            curr = curr.setdefault(k, {})
        curr[ks[-1]] = v
    return nested


@contextlib.contextmanager
def augment_environment(**kwargs):
    current_env = dict(os.environ)
    for env_var, value in kwargs.items():
        os.environ[env_var] = value

    yield

    os.environ.clear()
    os.environ.update(current_env)
