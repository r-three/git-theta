"""Classes for computing Euclidean locality-sensitive hashes"""

import numpy as np
import numba as nb
import os

from git_theta.lsh import HashFamily, RandomnessPool
from git_theta.lsh.types import Signature, Parameter
from git_theta.utils import EnvVarConstants


class E2LSH(HashFamily):
    """
    Class for performing the Euclidean l2 LSH (E2LSH) algorithm described in https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf
    with a pre-computed randomness pool as described in Section 3 of http://personal.denison.edu/~lalla/papers/online-lsh.pdf

    Also see http://mlwiki.org/index.php/Euclidean_LSH for an introduction to E2LSH
    """

    def __init__(self, signature_size: int, bucket_width: float):
        super().__init__(signature_size)
        self.bucket_width = bucket_width

    @property
    def name(self) -> str:
        return "E2LSH"

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        x = x.ravel()
        hyperplanes = self.pool.get_hyperplanes(x.size)
        return np.rint((x @ hyperplanes) / self.bucket_width).astype(np.int64).tolist()

    def distance(self, query: Signature, data: Signature) -> float:
        """Compute the distance between two E2LSH signatures"""
        query, data = np.array(query), np.array(data)
        return (
            (1 / np.sqrt(self.signature_size))
            * np.linalg.norm(query - data)
            * self.bucket_width
        )


class FastE2LSH(E2LSH):
    """
    Class for performing the Euclidean LSH using numba-jitted loops. This is both faster than the E2LSH class using numpy matrix multiplications and
    also uses less memory since the whole hyperplane matrix (feature_size x signature_size) is never in memory at once.
    """

    def name(self) -> str:
        return "FastE2LSH"

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        return nb_hash(
            x.ravel(), self.signature_size, self.pool, self.bucket_width
        ).tolist()


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


def get_signature_size():
    signature_size = os.environ.get(EnvVarConstants.LSH_SIGNATURE_SIZE)
    return int(signature_size) if signature_size else 16


def get_bucket_width():
    bucket_width = os.environ.get(EnvVarConstants.LSH_BUCKET_WIDTH)
    return float(bucket_width) if bucket_width else 1e-7


def get_lsh():
    # TODO we need a better way of keeping track of configuration at the repository level
    # For LSH configuration, once it is set for a repository, changing it should be handled with care
    signature_size = get_signature_size()
    bucket_width = get_bucket_width()
    return FastE2LSH(signature_size, bucket_width)
