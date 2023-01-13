"""A class for handling sparse updates to parameters."""

import logging
from typing import Optional, Any
from git_theta.updates import IncrementalUpdate
import scipy.sparse
import numpy as np

Parameter = Any


class SparseUpdate(IncrementalUpdate):
    """An update where only some parameters are touched."""

    @property
    def name(self):
        return "sparse"

    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        diff = parameter - previous_parameter
        # csr_matrix looks for actual zeros in diff tensor. We can consider adding threshold option on diff tensor in future
        update = scipy.sparse.csr_matrix(np.reshape(diff, (1, -1)))
        return {
            "data": update.data,
            "indices": update.indices,
            "indptr": update.indptr,
            "shape": parameter.shape,
        }

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        # Provide shape of original flattened array to ensure correct shape of output. Without the provided shape, csr_matrix interprets the shape as (1, index of last occurence of a non-zero number in the original flattened array)
        param_update = scipy.sparse.csr_matrix(
            (update["data"], update["indices"], update["indptr"]),
            shape=(1, np.prod(update["shape"])),
        )
        return np.reshape(param_update.toarray(), update["shape"]) + previous
