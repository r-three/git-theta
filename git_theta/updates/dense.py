"""Class managing dense parameter updates."""

import logging
from typing import Any, Optional

from git_theta import git_utils, metadata
from git_theta.updates import Update

Parameter = Any


class DenseUpdate(Update):
    """An update where all parameters are changed."""

    name: str = "dense"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def apply(self, param_metadata, param_keys, *args, **kwargs) -> Parameter:
        logging.debug(f"Reading Dense update for {'/'.join(param_keys)}")
        return await self.read(param_metadata)

    async def write(self, param, param_keys, *args, **kwargs) -> metadata.LfsMetadata:
        logging.debug(f"Writing Dense update for {'/'.join(param_keys)}")
        serialized = await self.serializer.serialize({"parameter": param})
        lfs_pointer = await git_utils.git_lfs_clean(serialized)
        return metadata.LfsMetadata.from_pointer(lfs_pointer), None
