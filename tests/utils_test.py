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
        value = random.choice(values)
        curr[key] = value
        # 50/50 are we done adding values to this node?
        if random.choice([True, False]):
            curr = prev.pop()
        # If we have tried to to up the tree from the root, stop generating.
        if not prev:
            break
    return result


def test_flatten_dict_empty_leaf():
    """Test that empty leaves are ignored."""
    nested = {
        "a": {},
        "b": {
            "c": 1,
            "d": {},
        },
    }
    gold = {("b", "c"): 1}
    assert utils.flatten(nested) == gold


def test_flatten_dict_empty():
    assert utils.flatten({}) == {}


def test_sorted_flatten_dict_insertion_order():
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
    one_flat = utils.flatten(nested_dict)
    two_flat = utils.flatten(nested_dict_new_order)
    assert one_flat == two_flat
    for one, two in zip(sorted(one_flat.items()), sorted(two_flat.items())):
        assert one[0] == two[0]
        assert one[1] == two[1]


def test_flattened_dict_keys_are_correct():
    """Test that indexing the nested dict with the keys yields the value."""
    nested = make_nested_dict()
    for flat_key, flat_value in utils.flatten(nested).items():
        curr = nested
        for key in flat_key:
            curr = curr[key]
        assert curr == flat_value


def test_flattened_dict_sorted_is_actually_sorted():
    """Test to ensure the leaves are actually sorted."""
    nested = make_nested_dict()
    keys = tuple(map(op.itemgetter(0), sorted(utils.flatten(nested).items())))
    string_keys = ["/".join(k) for k in keys]
    sorted_string_keys = sorted(string_keys)
    assert string_keys == sorted_string_keys


def test_walk_parameter_dir(tmp_path):
    """Test the dir leaves happy path."""
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)

    gold = {
        ("a", "d"): os.path.join(str(tmp_path), "a", "d"),
        ("a", "c"): os.path.join(str(tmp_path), "a", "c"),
        ("b", "e"): os.path.join(str(tmp_path), "b", "e"),
    }

    results = utils.flatten(utils.walk_parameter_dir(tmp_path))
    assert results == gold


def test_walk_parameter_dir_leaves_empty_dir(tmp_path):
    """Test that we don't include directory paths that don't end with "params"."""
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

    gold = {
        ("a", "d"): os.path.join(str(tmp_path), "a", "d"),
        ("a", "c"): os.path.join(str(tmp_path), "a", "c"),
        ("b", "e"): os.path.join(str(tmp_path), "b", "e"),
    }

    results = utils.flatten(utils.walk_parameter_dir(tmp_path))
    assert results == gold


def test_walk_parameter_dir_leaves_params_and_dirs(tmp_path):
    """Test that the presence of a `params` dir stops expanding dirs, even if there are multiple subdirs."""
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)
    f = tmp_path / "b" / "e" / "f" / "params"
    f.mkdir(parents=True)

    gold = {
        ("a", "d"): os.path.join(str(tmp_path), "a", "d"),
        ("a", "c"): os.path.join(str(tmp_path), "a", "c"),
        ("b", "e"): os.path.join(str(tmp_path), "b", "e"),
    }
    results = utils.flatten(utils.walk_parameter_dir(tmp_path))
    assert results == gold


def test_walk_parameter_dir_leaves_dirs_within_params(tmp_path):
    """Test that we don't go looking in the `params` even if there is another inside it."""
    d = tmp_path / "a" / "d" / "params"
    d.mkdir(parents=True)
    c = tmp_path / "a" / "c" / "params"
    c.mkdir(parents=True)
    e = tmp_path / "b" / "e" / "params"
    e.mkdir(parents=True)
    f = tmp_path / "b" / "e" / "params" / "f" / "params"
    f.mkdir(parents=True)

    gold = {
        ("a", "d"): os.path.join(str(tmp_path), "a", "d"),
        ("a", "c"): os.path.join(str(tmp_path), "a", "c"),
        ("b", "e"): os.path.join(str(tmp_path), "b", "e"),
    }
    results = utils.flatten(utils.walk_parameter_dir(tmp_path))
    assert results == gold


def test_remove_params():
    """Test that removed values are actually detected and returned."""
    nested = make_nested_dict()

    # Stochastically remove keys from the dict, but track what was removed.
    def _remove(curr, kept, removed):
        for k, v in curr.items():
            if random.random() > 0.33:
                if isinstance(v, dict):
                    # Recurse into a dict which will return both the kept and
                    # removed values from that sub-dict.
                    keep, remove = _remove(v, {}, {})
                    kept[k] = keep
                    removed[k] = remove
                else:
                    kept[k] = v
            else:
                removed[k] = v
        return kept, removed

    kept, gold_removed = _remove(nested, {}, {})
    # Use Counters to check for bag equality.
    gold_removed = collections.Counter(utils.flatten(gold_removed).values())
    removed = collections.Counter(utils.removed_params(kept, nested))
    assert removed == gold_removed
