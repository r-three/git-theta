from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Union

from git_theta import utils
from git_theta.models.metadata import ParamMetadata

Leaf = Union[ParamMetadata, np.ndarray]


class Model(OrderedDict, metaclass=ABCMeta):
    """
    Base class for classes that represent a model as a nested dictionary with values (i.e. tensors, metadata, etc.) as leaf values
    """

    @staticmethod
    @abstractmethod
    def is_leaf(l: Leaf) -> bool:
        """
        Return whether the passed value is a leaf in the Model
        """

    @staticmethod
    @abstractmethod
    def leaves_equal(l1: Leaf, l2: Leaf) -> bool:
        """
        Return whether the two passed leaf values in the Model are equal (for the purposes of diff-ing)
        """

    def flatten(self) -> "Model":
        """
        Flatten a nested dictionary representing a model
        """
        return utils.flatten(self, is_leaf=self.is_leaf)

    def unflatten(self) -> "Model":
        """
        Unflatten a flattened Model
        """
        return utils.unflatten(self)

    @classmethod
    def diff(cls, m1: "Model", m2: "Model"):
        m1_flat = m1.flatten()
        m2_flat = m2.flatten()
        added = cls(
            {k: m1_flat[k] for k in m1_flat.keys() - m2_flat.keys()}
        ).unflatten()
        removed = cls(
            {k: m2_flat[k] for k in m2_flat.keys() - m1_flat.keys()}
        ).unflatten()
        modified = cls()
        for param_keys in set(m1_flat.keys()).intersection(m2_flat.keys()):
            if not cls.leaves_equal(m1_flat[param_keys], m2_flat[param_keys]):
                modified[param_keys] = m1_flat[param_keys]

        modified = modified.unflatten()
        return added, removed, modified
