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
        self.logger = logging.getLogger("git_theta")

    async def apply(self, param_metadata, param_keys, *args, **kwargs) -> Parameter:
        param_name = "/".join(param_keys)
        self.logger.debug(f"Reading Dense update for {param_name}")
        tensor = await self.read(param_metadata)
        self.logger.debug(f"Finished Read Dense update for {param_name}")
        return tensor

    async def write(self, param, param_keys, *args, **kwargs) -> metadata.LfsMetadata:
        param_name = "/".join(param_keys)
        self.logger.debug(f"Writing Dense update for {param_name}")
        self.logger.debug(f"Starting Serializing {param_name}")
        serialized = await self.serializer.serialize({"parameter": param})
        self.logger.debug(f"Finished Serializing {param_name}")
        self.logger.debug(f"Starting git-lfs clean for {param_name}")
        lfs_pointer = await git_utils.git_lfs_clean(serialized)
        self.logger.debug(f"Finished git-lfs clean for {param_name}")
        return metadata.LfsMetadata.from_pointer(lfs_pointer), None
