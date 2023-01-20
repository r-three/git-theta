"""Base class for computing locality-sensitive hashes"""

import abc
from git_theta.lsh import RandomnessPool
from git_theta.lsh.types import Signature, Parameter


class HashFamily(metaclass=abc.ABCMeta):
    def __init__(self, signature_size: int):
        self._signature_size = signature_size
        self.pool = RandomnessPool(signature_size)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the distance function this hash family approximates."""

    @property
    def signature_size(self) -> int:
        """The size of the signatures this hash produces."""
        return self._signature_size

    @abc.abstractmethod
    def hash(self, x: Parameter) -> Signature:
        """Convert `x` to its signature."""

    @abc.abstractmethod
    def distance(self, query: Signature, data: Signature) -> float:
        """Calculate the approximate distance between two signatures."""
