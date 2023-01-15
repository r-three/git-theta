"""Classes for computing locality-sensitive hashes"""

import numpy as np
import numba as nb

from git_theta.lsh import HashFamily, RandomnessPool
from git_theta.lsh.types import Signature, Parameter


class E2LSH(HashFamily):
    def __init__(self, signature_size: int, bucket_width: float):
        super().__init__(signature_size)
        self.bucket_width = bucket_width

    @property
    def name(self) -> str:
        return "E2LSH"

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        hyperplanes = self.pool.get_hyperplanes(x.size)
        return np.rint((x @ hyperplanes) / self.bucket_width)

    def distance(self, query: Signature, data: Signature) -> float:
        return (
            (1 / np.sqrt(self.signature_size))
            * np.linalg.norm(query - data)
            * self.bucket_width
        )


class FastE2LSH(E2LSH):
    def name(self) -> str:
        return "FastE2LSH"

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        return nb_hash(x, self.signature_size, self.pool, self.bucket_width)


@nb.jit(nopython=True, parallel=True)
def nb_hash(
    x: Parameter, signature_size: int, pool: RandomnessPool, bucket_width: float
) -> Signature:
    signature = np.zeros(signature_size)

    for signature_idx in nb.prange(signature_size):
        for feature_idx in range(x.size):
            hyperplane_element = pool.get_hyperplane_element(feature_idx, signature_idx)
            signature[signature_idx] += x[feature_idx] * hyperplane_element

    return np.rint(signature / bucket_width).astype(np.int64)
