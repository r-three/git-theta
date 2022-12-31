"""Utilities for running tests"""

import string
import random
import numpy as np

from git_theta import metadata


def random_oid():
    return "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])


def random_commit_hash():
    return "".join([random.choice(string.hexdigits.lower()) for _ in range(40)])


def random_lfs_metadata():
    version = random.choice(["lfs_version1", "my_version", "version1"])
    oid = "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])
    size = str(random.randint(0, 10000))
    return metadata.LfsMetadata(version=version, oid=oid, size=size)


def random_tensor_metadata():
    ndims = random.choice(range(1, 6))
    shape = tuple([random.choice(range(1, 50)) for _ in range(ndims)])
    tensor = np.random.rand(*shape)
    return metadata.TensorMetadata.from_tensor(tensor)


def random_theta_metadata():
    update_type = random.choice(["dense", "sparse"])
    last_commit = "".join([random.choice(string.hexdigits.lower()) for _ in range(40)])
    return metadata.ThetaMetadata(update_type=update_type, last_commit=last_commit)


def random_param_metadata():
    tensor_metadata = random_tensor_metadata()
    lfs_metadata = random_lfs_metadata()
    theta_metadata = random_theta_metadata()
    return metadata.ParamMetadata(
        tensor_metadata=tensor_metadata,
        lfs_metadata=lfs_metadata,
        theta_metadata=theta_metadata,
    )


def random_metadata():
    result = metadata.Metadata()
    keys = list(string.ascii_letters)
    values = [random_param_metadata() for _ in range(100)]

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
