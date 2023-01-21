"""Class for deterministically supplying pre-computed random values"""

import numpy as np
from numpy.random import Generator, MT19937
import numba as nb
import sys

from git_theta.utils import EnvVarConstants


spec = [("pool", nb.float64[:]), ("signature_offsets", nb.int64[:])]


@nb.experimental.jitclass(spec)
class RandomnessPool:
    def __init__(self, signature_size):
        with nb.objmode(pool="float64[:]", signature_offsets="int64[:]"):
            # N.b. we use a fixed seed so that every instance of RandomPool has the same set of random numbers
            rng = Generator(MT19937(seed=42))
            pool = rng.normal(size=EnvVarConstants.LSH_POOL_SIZE)
            int64_range = np.iinfo(np.int64)
            signature_offsets = rng.integers(
                int64_range.min, int64_range.max, size=signature_size, dtype=np.int64
            )
        self.pool = pool
        self.signature_offsets = signature_offsets

    def get_hyperplanes(self, feature_size):
        hyperplanes = np.empty((feature_size, self.signature_offsets.size))
        for feature_idx in nb.prange(feature_size):
            for signature_idx in nb.prange(self.signature_offsets.size):
                hyperplanes[feature_idx, signature_idx] = self.get_hyperplane_element(
                    feature_idx, signature_idx
                )
        return hyperplanes

    def get_hyperplane_element(self, feature_idx, signature_idx):
        signature_offset = self.signature_offsets[signature_idx]
        pool_idx = np.mod(np.bitwise_xor(feature_idx, signature_offset), self.pool.size)
        return self.pool[pool_idx]
