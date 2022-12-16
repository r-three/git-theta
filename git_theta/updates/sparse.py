"""A class for handling sparse updates to parameters."""

import logging
from typing import Optional, Any
from git_theta.updates import TrueUpdate

Parameter = Any


class SparseUpdate(TrueUpdate):
    """An update where only some parameters are touched."""

    @property
    def name(self):
        return "sparse"

    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        return parameter - previous_parameter

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        return update + previous
