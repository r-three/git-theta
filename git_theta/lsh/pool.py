"""Class for deterministically supplying pre-computed random values"""

import numpy as np
import numba as nb
from numba import int32, float32
import sys

# TODO(bdlester): importlib.resources doesn't have the `.files` API until python
# version `3.9` so use the backport even if using a python version that has
# `importlib.resources`.
if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources


# RandomnessPool is a singleton class so we don't read pool data off disk multiple times per git-theta command
SINGLETON = None

spec = [("pool", nb.float64[:]), ("index_hashes", nb.int64[:])]


@nb.experimental.jitclass(spec)
class RandomnessPool:
    def __init__(self, pool, index_hashes):
        self.pool = pool
        self.index_hashes = index_hashes

    def get_hyperplanes(self, feature_size):
        hyperplanes = np.empty((feature_size, self.index_hashes.size))
        for feature_idx in nb.prange(feature_size):
            for signature_idx in nb.prange(self.index_hashes.size):
                hyperplanes[feature_idx, signature_idx] = self.get_hyperplane_element(
                    feature_idx, signature_idx
                )
        return hyperplanes

    def get_hyperplane_element(self, feature_idx, signature_idx):
        index_hash = self.index_hashes[signature_idx]
        pool_idx = int(np.mod(np.bitwise_xor(feature_idx, index_hash), self.pool.size))
        return self.pool[pool_idx]


def get_randomness_pool(signature_size):
    global SINGLETON

    if not SINGLETON:
        package = importlib_resources.files("git_theta")
        with importlib_resources.as_file(
            package.joinpath("lsh", "data", "pool.npy")
        ) as pool_file:
            pool = np.load(pool_file)
        with importlib_resources.as_file(
            package.joinpath("lsh", "data", "index_hashes.npy")
        ) as index_hashes_file:
            index_hashes = np.load(index_hashes_file)
            if signature_size > index_hashes.size:
                raise ValueError(
                    f"Cannot produce LSH signatures of size larger than {index_hashes.size}"
                )
            index_hashes = index_hashes[:signature_size]
        SINGLETON = RandomnessPool(pool, index_hashes)

    return SINGLETON
