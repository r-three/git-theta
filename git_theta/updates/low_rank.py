"""An update type where the update is stored as 2 low-rank matrices."""


import logging
from typing import Any, FrozenSet, Optional

import numpy as np

from git_theta.updates import IncrementalUpdate

Parameter = Any


class LowRankUpdate(IncrementalUpdate):
    """An update make for 2 low rank matrices."""

    name: str = "low-rank"
    required_keys: FrozenSet[str] = frozenset(("R", "C"))

    # TODO: Make these configuration options easy set.
    def __init__(
        self, *args, K: Optional[int] = None, threshold: float = 1e-11, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.K = K
        self.threshold = threshold

    @classmethod
    def format_update(
        cls, param1: Parameter, param2: Parameter, *args, **kwargs
    ) -> Parameter:
        return {
            "R": param1,
            "C": param2,
        }

    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        update = parameter - previous_parameter
        if update.ndim < 2:
            return update
        logger = logging.getLogger("git_theta")
        logger.info("Inferring low-rank update based on SVD")
        u, s, vh = np.linalg.svd(update, full_matrices=False)
        if self.K is not None:
            k = self.K
            logger.info(f"Low Rank Update configured to have a rank of {k}")
        else:
            k = np.sum(s > self.threshold)
            logger.info(f"Low Rank Update inferred to have a rank of {k}")
        return {"R": u[:, :k], "C": (np.diag(s[:k]) @ vh[:k, :])}

    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        if not isinstance(update, dict):
            return update + previous
        return update["R"] @ update["C"] + previous
