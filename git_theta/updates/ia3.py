"""A class for handling activations scaling using ia3 vectors."""

import logging
from typing import Optional, List, Any
from git_theta.updates import IncrementalUpdate
from git_theta import params
import numpy as np

Parameter = Any


class IA3Update(IncrementalUpdate):
    """An update where activations are scaled."""

    def __init__(self, serializer: params.Serializer):
        super().__init__(serializer)

    @property
    def name(self):
        return "ia3"

    async def calculate_update(
        self,
        parameter: Parameter,
        previous_parameter: Parameter,
        broadcast_dims: List[int],
    ) -> Parameter:
        """Calculate the update for the given parameter where ia3 is applied over broadcast dims."""

        # use mask1 to prevent divide by zeros
        mask1 = previous_parameter != 0
        multiplier = np.divide(
            parameter, previous_parameter, out=np.zeros(parameter.shape), where=mask1
        )

        # Calcuate ia3 by averaging multiplier over broadcast dims and take into account the fact that some values may be zero
        denominator = np.sum(mask1, axis=tuple(broadcast_dims), keepdims=True)
        mask2 = denominator != 0
        ia3_update = np.divide(
            np.sum(multiplier, axis=tuple(broadcast_dims), keepdims=True),
            denominator,
            out=np.zeros(parameter.shape),
            where=mask2,
        )

        return {"ia3": ia3_update}

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        return previous * update["ia3"]
