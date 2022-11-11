"""Utilities for git theta"""


import operator as op


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

    def _iterate_dict_leaves(d, prefix):
        for k, v in sorted(d.items(), key=op.itemgetter(0)):
            if isinstance(v, dict):
                yield from _iterate_dict_leaves(v, prefix + [k])
            else:
                yield (v, prefix + [k])

    return _iterate_dict_leaves(d, [])


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

    def _iterate_dir_leaves(root, prefix):
        for d in os.listdir(root):
            dir_member = os.path.join(root, d)
            if not "params" in os.listdir(dir_member):
                yield from _iterate_dir_leaves(dir_member, prefix=prefix + [d])
            else:
                yield (dir_member, prefix + [d])

    return _iterate_dir_leaves(root, [])
