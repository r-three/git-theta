"""Base class for parameter update plugins."""

from abc import ABCMeta, abstractmethod
import os
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

import logging
from typing import Optional, Tuple

import numpy as np

from git_theta import git_utils, utils, params, metadata


Parameter = np.ndarray


class Update(metaclass=ABCMeta):
    """Base class for parameter update plugins."""

    def __init__(self, serializer: params.Serializer):
        self.serializer = serializer

    @property
    @abstractmethod
    def name(self) -> str:
        """The name used to lookup the plug-in."""

    async def read(self, param_metadata: metadata.ParamMetadata) -> Parameter:
        """Read in and deserialize a single parameter value based metadata."""
        lfs_pointer = param_metadata.lfs_metadata.lfs_pointer
        serialized_param = await git_utils.git_lfs_smudge(lfs_pointer)
        param = (await self.serializer.deserialize(serialized_param))["parameter"]
        return param

    @abstractmethod
    async def write(
        self, param: Parameter, param_keys: Tuple[str], **kwargs
    ) -> metadata.LfsMetadata:
        """Serialize and save a parameter with git-lfs."""

    @abstractmethod
    async def apply(
        self, param_metadata: metadata.ParamMetadata, param_keys: Tuple[str], **kwargs
    ) -> Parameter:
        """Get the final parameter value, including fetching previous values."""


class IncrementalUpdate(Update):
    """Base class for parameter updates that depend on the previous value."""

    async def get_previous_metadata(
        self,
        param_metadata: metadata.ParamMetadata,
        param_keys: Tuple[str],
        repo,
        path: str,
    ) -> metadata.ParamMetadata:
        """Get the metadata from the last time this parameter was updated via git."""
        logging.debug(f"Getting previous metadata for {'/'.join(param_keys)}")
        logging.debug(f"Current Metadata for {'/'.join(param_keys)}: {param_metadata}")
        last_commit = param_metadata.theta_metadata.last_commit
        # TODO: Currently, if the model checkpoint is added during the first commit
        # then we can't do a sparse update until a second dense update is commited.
        if not last_commit:
            raise ValueError(
                f"Cannot find previous version for parameter {'/'.join(param_keys)}"
            )
        logging.debug(
            f"Getting metadata for {'/'.join(param_keys)} from commit {last_commit}"
        )
        last_metadata_obj = git_utils.get_file_version(repo, path, last_commit)
        last_metadata = metadata.Metadata.from_file(last_metadata_obj.data_stream)
        last_param_metadata = last_metadata.flatten()[param_keys]
        logging.debug(
            f"Previous Metadata for {'/'.join(param_keys)}: {last_param_metadata}"
        )
        return last_param_metadata

    async def get_previous_value(
        self,
        param_metadata: metadata.ParamMetadata,
        param_keys: Tuple[str],
        repo,
        path: str,
    ) -> Parameter:
        """Get the last value for this parameter via git."""
        logging.debug(f"Getting previous value for {'/'.join(param_keys)}")
        prev_metadata = await self.get_previous_metadata(
            param_metadata, param_keys, repo=repo, path=path
        )
        # TODO: get_update_serializer returns instantiated objects while the other
        # getters return classes to be instantiated.
        prev_serializer = params.get_update_serializer()
        prev_update = get_update_handler(prev_metadata.theta_metadata.update_type)(
            prev_serializer
        )
        return await prev_update.apply(prev_metadata, param_keys, repo=repo, path=path)

    @abstractmethod
    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        """Calculate the update required to go from previous_parameter -> parameter."""

    @abstractmethod
    async def apply_update(self, update: Parameter, previous: Parameter) -> Parameter:
        """Apply the update to the previous value to get the new value."""

    async def write_update(self, update: Parameter) -> metadata.LfsMetadata:
        """Save and serialize (just) the update weights."""
        serialized_update = await self.serializer.serialize({"parameter": update})
        lfs_pointer = await git_utils.git_lfs_clean(serialized_update)
        return metadata.LfsMetadata.from_pointer(lfs_pointer)

    async def write(
        self,
        param: Parameter,
        param_keys,
        *,
        param_metadata: metadata.ParamMetadata,
        repo,
        path: str,
        **kwargs,
    ) -> metadata.LfsMetadata:
        """Serialize and save a parameter with git-lfs as a delta from the previous value."""
        logging.debug(f"Writing {self.name} update for {'/'.join(param_keys)}")
        previous_value = await self.get_previous_value(
            param_metadata, param_keys, repo=repo, path=path
        )
        update_value = await self.calculate_update(param, previous_value)
        return await self.write_update(update_value)

    async def apply(
        self,
        param_metadata: metadata.ParamMetadata,
        param_keys: Tuple[str],
        *,
        repo,
        path: str,
        **kwargs,
    ) -> Parameter:
        """Get the final parameter value, including fetching previous values."""
        logging.debug(f"Applying {self.name} update for {'/'.join(param_keys)}")
        update_value = await self.read(param_metadata)
        prev_value = await self.get_previous_value(
            param_metadata, param_keys, repo=repo, path=path
        )
        return await self.apply_update(update_value, prev_value)


def get_update_handler_name(update_type: Optional[str] = None) -> str:
    return update_type or os.environ.get(utils.EnvVarConstants.UPDATE_TYPE) or "dense"


def get_update_handler(update_type: Optional[str] = None) -> Update:
    """Get an Update class by name.

    Parameters
    ----------
    update_type:
        The name of the update type we want to use.

    Returns
    -------
    Update
        The update class. Returned class may be defined in a user installed
        plugin.
    """
    update_name = get_update_handler_name(update_type)
    discovered_plugins = entry_points(group="git_theta.plugins.updates")
    return discovered_plugins[update_name].load()
