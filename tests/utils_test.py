"""Tests for utils.py"""

import collections
import os
import operator as op
import random
import string

from git_theta import utils


def make_nested_dict():
    """Generate random nested dicts for testing."""
    result = {}
    keys = list(string.ascii_letters)
    values = list(range(100))

    prev = [result]
    curr = result
    for _ in range(random.randint(20, 50)):
        # Pick a key
        key = random.choice(keys)
        # 50/50, do we make a new nest level?
        if random.choice([True, False]):
            curr[key] = {}
            prev.append(curr)
            curr = curr[key]
            continue
        # Otherwise, add a leaf value
        else:
            value = random.choice(values)
            curr[key] = value
            # 50/50 are we done adding values to this node?
            if random.choice([True, False]):
                curr = prev.pop()
        # If we have tried to to up the tree from the root, stop generating.
        if not prev:
            break
    return result


def test_iterate_dict_leaves_empty_leaf():
    """Test that empty leaves are ignored."""
    nested = {
        "a": {},
        "b": {
            "c": 1,
            "d": {},
        },
    }
    gold = [(1, ["b", "c"])]
    assert list(utils.iterate_dict_leaves(nested)) == gold


def test_iterate_dict_leaves_empty():
    """Test that processing an empty dict results in nothing."""
    assert list(utils.iterate_dict_leaves({})) == []


def test_iterate_dict_leaves_insertion_order():
    """Test that key order is consistent for different insertion order."""
    nested_dict = {
        "a": {
            "b": {
                "c": 10,
                "d": 20,
                "e": 30,
            },
            "c": {
                "b": 40,
                "z": -1,
            },
        }
    }

    nested_dict_new_order = {
        "a": {
            "c": {
                "z": -1,
                "b": 40,
            },
            "b": {
                "d": 20,
                "e": 30,
                "c": 10,
            },
        }
    }
    assert nested_dict == nested_dict_new_order
    one_leaves = list(utils.iterate_dict_leaves(nested_dict))
    one_order_leaves = list(utils.iterate_dict_leaves(nested_dict_new_order))
    for one, one_order in zip(one_leaves, one_order_leaves):
        assert one[0] == one_order[0]
        assert one[1] == one_order[1]


def test_iterate_dict_leaves_keys_are_correct():
    """Test that indexing the nested dict with the keys yields the value."""
    nested = make_nested_dict()
    for flat_value, flat_key in utils.iterate_dict_leaves(nested):
        curr = nested
        for key in flat_key:
            curr = curr[key]
        assert curr == flat_value


def test_iterate_dict_leaves_are_sorted():
    """Test to ensure the leaves are actually sorted."""
    nested = make_nested_dict()
    keys = tuple(map(op.itemgetter(1), utils.iterate_dict_leaves(nested)))
    string_keys = ["/".join(k) for k in keys]
    sorted_string_keys = sorted(string_keys)
    assert string_keys == sorted_string_keys


def test_iterate_dir_leaves(tmp_path):
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)

    gold = [
        (os.path.join(str(tmp_path), "a", "d"), ["a", "d"]),
        (os.path.join(str(tmp_path), "a", "c"), ["a", "c"]),
        (os.path.join(str(tmp_path), "b", "e"), ["b", "e"]),
    ]

    results = list(utils.iterate_dir_leaves(tmp_path))
    assert sorted(results) == sorted(gold)


def test_iterate_dir_leaves_empty_dir(tmp_path):
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)
    empty = tmp_path / "b" / "f"
    empty.mkdir(parents=True)
    double_empty = tmp_path / "g" / "h"
    double_empty.mkdir(parents=True)

    gold = [
        (os.path.join(str(tmp_path), "a", "d"), ["a", "d"]),
        (os.path.join(str(tmp_path), "a", "c"), ["a", "c"]),
        (os.path.join(str(tmp_path), "b", "e"), ["b", "e"]),
    ]

    results = list(utils.iterate_dir_leaves(tmp_path))
    assert sorted(results) == sorted(gold)


def test_iterate_dir_leaves_params_and_dirs(tmp_path):
    """Test that the presence of a `params` dir stops expanding dirs."""
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)
    f = tmp_path / "b" / "e" / "f" / "params"
    f.mkdir(parents=True)

    gold = [
        (os.path.join(str(tmp_path), "a", "d"), ["a", "d"]),
        (os.path.join(str(tmp_path), "a", "c"), ["a", "c"]),
        (os.path.join(str(tmp_path), "b", "e"), ["b", "e"]),
    ]
    results = list(utils.iterate_dir_leaves(tmp_path))
    assert sorted(results) == sorted(gold)


def test_iterate_dir_leaves_dirs_within_params(tmp_path):
    """Test that we don't go looking in the `params` even if there is another inside it."""
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)
    f = tmp_path / "b" / "e" / "params" / "f" / "params"
    f.mkdir(parents=True)

    gold = [
        (os.path.join(str(tmp_path), "a", "d"), ["a", "d"]),
        (os.path.join(str(tmp_path), "a", "c"), ["a", "c"]),
        (os.path.join(str(tmp_path), "b", "e"), ["b", "e"]),
    ]
    results = list(utils.iterate_dir_leaves(tmp_path))
    print(results)
    assert sorted(results) == sorted(gold)
