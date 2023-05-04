"""Classes for computing Euclidean locality-sensitive hashes"""

import os

import numba as nb
import numpy as np

from git_theta import config, git_utils
from git_theta.lsh import HashFamily
from git_theta.lsh.pool import RandomnessPool
from git_theta.lsh.types import Parameter, Signature


class EuclideanLSH(HashFamily):
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
        return "euclidean"

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        x = x.ravel()
        hyperplanes = self.pool.get_hyperplanes(x.size)
        return np.floor((x @ hyperplanes) / self.bucket_width).astype(np.int64)

    def distance(self, query: Signature, data: Signature) -> float:
        """Compute the distance between two EuclideanLSH signatures"""
        return (
            (1 / np.sqrt(self.signature_size))
            * np.linalg.norm(query - data)
            * self.bucket_width
        )


class FastEuclideanLSH(EuclideanLSH):
    """
    Class for performing the Euclidean LSH using numba-jitted loops. This is both faster than the EuclideanLSH class using numpy matrix multiplications and
    also uses less memory since the whole hyperplane matrix (feature_size x signature_size) is never in memory at once.
    """

    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""
        return nb_hash(x.ravel(), self.signature_size, self.pool, self.bucket_width)


@nb.jit(nopython=True, parallel=True)
def nb_hash(
    x: Parameter, signature_size: int, pool: RandomnessPool, bucket_width: float
) -> Signature:
    signature = np.zeros(signature_size)

    for signature_idx in nb.prange(signature_size):
        for feature_idx, feature in enumerate(x):
            hyperplane_element = pool.get_hyperplane_element(feature_idx, signature_idx)
            signature[signature_idx] += feature * hyperplane_element

    return np.floor(signature / bucket_width).astype(np.int64)


def get_lsh():
    repo = git_utils.get_git_repo()
    thetaconfig = config.ThetaConfigFile(repo)
    return FastEuclideanLSH(
        thetaconfig.repo_config.lsh_signature_size,
        thetaconfig.repo_config.parameter_atol,
    )
