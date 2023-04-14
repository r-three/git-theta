__version__ = "0.0.2"

import numba

numba.config.THREADING_LAYER = "tbb"

from git_theta import (
    checkpoints,
    git_utils,
    lsh,
    metadata,
    params,
    theta,
    updates,
    utils,
)
