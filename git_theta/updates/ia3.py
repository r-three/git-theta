"""A class for handling activations scaling using ia3 vectors."""

import logging
from typing import Optional, Any
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
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        # since IA3 scales activations, weight.T[1:].shape can be considered as output dimensions and the update scalar vector/ matrix will be of shape = weight[1:].shape
        # we transpose parameter weights so that output dimensions are in the end
        parameter = parameter.T
        previous_parameter = previous_parameter.T
        # take into account divide by zero
        multiplier = np.divide(
            parameter, previous_parameter, where=previous_parameter != 0
        )
        update = np.mean(multiplier, axis=0, keepdims=True)
        # Can we check if update == mulitplier[i] for any i?
        return {"ia3": update.T}

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        return previous * update["ia3"]
