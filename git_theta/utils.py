"""Utilities for git theta"""


import os
from typing import Dict, Any, Tuple, Union, Callable
import re
from dataclasses import dataclass


@dataclass
class EnvVar:
    name: str
    default: Any

    def __get__(self, obj, objtype=None):
        value = os.environ.get(self.name)
        return type(self.default)(value) if value else self.default


class EnvVarConstants:
    CHECKPOINT_TYPE = EnvVar(name="GIT_THETA_CHECKPOINT_TYPE", default="pytorch")
    UPDATE_TYPE = EnvVar(name="GIT_THETA_UPDATE_TYPE", default="dense")
    PARAMETER_ATOL = EnvVar(name="GIT_THETA_PARAMETER_ATOL", default=1e-8)
    PARAMETER_RTOL = EnvVar(name="GIT_THETA_PARAMETER_RTOL", default=1e-5)
    LSH_SIGNATURE_SIZE = EnvVar(name="GIT_THETA_LSH_SIGNATURE_SIZE", default=16)
    LSH_THRESHOLD = EnvVar(name="GIT_THETA_LSH_THRESHOLD", default=1e-6)
    LSH_POOL_SIZE = EnvVar(name="GIT_THETA_LSH_POOL_SIZE", default=10_000)


def flatten(
    d: Dict[str, Any],
    is_leaf: Callable[[Any], bool] = lambda v: not isinstance(v, dict),
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


def unflatten(d: Dict[Tuple[str, ...], Any]) -> Dict[str, Union[Dict[str, Any], Any]]:
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


def is_valid_oid(oid: str) -> bool:
    """Check if an LFS object-id is valid

    Parameters
    ----------
    oid:
        LFS object-id

    Returns
    bool
        Whether this object-id is valid
    """
    return re.match("^[0-9a-f]{64}$", oid) is not None


def is_valid_commit_hash(commit_hash: str) -> bool:
    """Check if a git commit hash is valid

    Parameters
    ----------
    commit_hash
        Git commit hash

    Returns
    bool
        Whether this commit hash is valid
    """
    return re.match("^[0-9a-f]{40}$", commit_hash) is not None
