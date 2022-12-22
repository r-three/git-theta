"""Utilities for running tests"""

import string
import random


def random_oid():
    return "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])


def random_commit_hash():
    return "".join([random.choice(string.hexdigits.lower()) for _ in range(40)])
