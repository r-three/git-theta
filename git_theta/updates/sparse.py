"""A class for handling sparse updates to parameters."""

import logging
from typing import Any, Optional

import numpy as np
import scipy.sparse

from git_theta import params
from git_theta.updates import IncrementalUpdate

Parameter = Any


class SparseUpdate(IncrementalUpdate):
    """An update where only some parameters are touched."""

    name: str = "sparse"

    def __init__(self, serializer: params.Serializer, threshold: float = 1e-12):
        # TODO: Make threshold configurable
        super().__init__(serializer)
        self.threshold = threshold

    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        diff = parameter - previous_parameter
        diff[np.abs(diff) < self.threshold] = 0
        # csr_matrix looks for actual zeros in diff tensor. We added a configurable threshold to have the diff tensor (the update) be really sparse
        update = scipy.sparse.csr_matrix(np.reshape(diff, (1, -1)))
        return {
            "data": update.data,
            "indices": update.indices,
            "indptr": update.indptr,
            "shape": np.array(parameter.shape),
        }

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        # Provide shape of original flattened array to ensure correct shape of output. Without the provided shape, csr_matrix interprets the shape as (1, index of last occurence of a non-zero number in the original flattened array)
        param_update = scipy.sparse.csr_matrix(
            (update["data"], update["indices"], update["indptr"]),
            shape=(1, np.prod(update["shape"])),
        )
        return np.reshape(param_update.toarray(), update["shape"]) + previous
