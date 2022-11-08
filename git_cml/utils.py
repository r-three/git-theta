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
