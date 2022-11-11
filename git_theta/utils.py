"""Utilities for git theta"""


import operator as op
import os
from typing import Dict, Any, Tuple, Union


def iterate_dict_leaves(d):
    """
    Generator that iterates through a dictionary and produces (leaf, keys) tuples where leaf is a dictionary leaf
    and keys is the sequence of keys used to access leaf. Dictionary is iterated in depth-first
    order with lexicographic ordering of keys.

    Example
    -------
    d = {'a': {'b': {'c': 10, 'd': 20, 'e': 30}}}
    iterate_dict_leaves(d) --> ((10, ['a','b','c']), (20, ['a','b','d']), (30, ['a','b','e']))

    Parameters
    ----------
    d : dict
        dictionary to iterate over

    Returns
    -------
    generator
        generates dict leaf, key path tuples
    """
    yield from map(lambda kv: (kv[1], list(kv[0])), sorted(flatten(d).items()))


def iterate_dir_leaves(root):
    """
    Generator that iterates through files in a directory tree and produces (path, dirs) tuples where
    path is the file's path and dirs is the sequence of path components from root to the file.

    Example
    -------
    root
    ├── a
    │   ├── c
    │   └── d
    └── b
        └── e

    iterate_dir_leaves(root) --> ((root/a/c, ['a','c']), (root/a/d, ['a','d']), (root/b/e, ['b','e']))

    Parameters
    ----------
    root : str
        Root of directory tree to iterate over

    Returns
    -------
    generator
        generates directory tree leaf, subdirectory list tuples
    """
    yield from map(
        lambda kv: (kv[1], list(kv[0])), sorted(flatten(walk_dir(root)).items())
    )


def flatten(d: Dict[str, Any]) -> Dict[Tuple[str, ...], Any]:
    """Flatten a nested dictionary."""

    def _flatten(d, prefix: Tuple[str] = ()):
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(_flatten(v, prefix=prefix + (k,)))
            else:
                flat[prefix + (k,)] = v
        return flat

    return _flatten(d)


def unflatten(d: Dict[Tuple[str], Any]) -> Dict[str, Union[Dict[str, Any], Any]]:
    """Unflatten a dict into a nested one."""
    nested = {}
    for ks, v in d.items():
        curr = nested
        for k in ks[:-1]:
            curr = curr.setdefault(k, {})
        curr[ks[-1]] = v
    return nested


def walk_dir(root, is_leaf=lambda x: "params" in os.listdir(x)):
    """Convert directory structure into nested dicts."""

    def _walk_dir(root):
        dir_dict = {}
        for d in os.listdir(os.path.join(*root)):
            full_path = os.path.join(*root, d)
            if is_leaf(full_path):
                dir_dict[d] = full_path
            else:
                dir_dict[d] = _walk_dir(root + (d,))
        return dir_dict

    return _walk_dir((root,))
