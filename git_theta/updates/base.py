"""Base class for parameter update plugins."""

import os
import sys
from abc import ABCMeta, abstractmethod

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

import logging
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np

from git_theta import checkpoints, git_utils, lsh, metadata, params, utils
from git_theta.lsh.types import Signature

Parameter = np.ndarray


@utils.abstract_classattributes("name")
class Update(metaclass=ABCMeta):
    """Base class for parameter update plugins."""

    name: str = NotImplemented  # The name used to lookup the plug-in.

    def __init__(self, serializer: params.Serializer, *args, **kwargs):
        self.serializer = serializer

    async def read(self, param_metadata: metadata.ParamMetadata) -> Parameter:
        """Read in and deserialize a single parameter value based metadata."""
        lfs_pointer = param_metadata.lfs_metadata.lfs_pointer
        serialized_param = await git_utils.git_lfs_smudge(lfs_pointer)
        param = await self.serializer.deserialize(serialized_param)
        return param.get("parameter", param)

    def will_update(self, param_keys: Tuple[str]) -> bool:
        return False

    @abstractmethod
    async def write(
        self, param: Parameter, param_keys: Tuple[str], **kwargs
    ) -> Tuple[metadata.LfsMetadata, Signature]:
        """Serialize and save a parameter with git-lfs."""

    @abstractmethod
    async def apply(
        self, param_metadata: metadata.ParamMetadata, param_keys: Tuple[str], **kwargs
    ) -> Parameter:
        """Get the final parameter value, including fetching previous values."""


# TODO: Fix this for inheritance so we don't need to dup "name" here.
@utils.abstract_classattributes("name", "required_keys")
class IncrementalUpdate(Update):
    """Base class for parameter updates that depend on the previous value."""

    required_keys: FrozenSet[str] = NotImplemented  # Names for side-loaded information.

    def __init__(self, serializer: params.Serializer, update_data: str = ""):
        super().__init__(serializer)
        self.update_information: Dict[str, np.ndarray] = None
        self.update_names: utils.Trie = None
        # Flatten the side-loaded information into a of string keys to arrays.
        if update_data:
            self.update_information = {
                "/".join(k): v
                for k, v in checkpoints.get_checkpoint_handler()
                .from_file(update_data)
                .flatten()
                .items()
            }
            self.update_names = utils.Trie.from_iterable(self.update_information.keys())

    def will_update(self, param_keys: Tuple[str]) -> bool:
        if self.update_information is not None:
            param_keys = "/".join(param_keys)
            return self.update_names.prefix(param_keys)
        return False

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
        # TODO: get_update_serializer returns instantiated objects while the other
        # getters return classes to be instantiated.
        prev_serializer = params.get_update_serializer()
        prev_update = get_update_handler(param_metadata.theta_metadata.update_type)(
            prev_serializer
        )
        return await prev_update.apply(param_metadata, param_keys, repo=repo, path=path)

    @abstractmethod
    async def calculate_update(
        self, parameter: Parameter, previous_parameter: Parameter
    ) -> Parameter:
        """Calculate the update required to go from previous_parameter -> parameter."""

    async def read_update(self, param_keys) -> Parameter:
        return {
            k: self.update_information["/".join(param_keys + (k,))]
            for k in self.required_keys
        }

    @classmethod
    @abstractmethod
    async def apply_update(cls, update: Parameter, previous: Parameter) -> Parameter:
        """Apply the update to the previous value to get the new value."""

    @abstractmethod
    def format_update(self, param: Parameter, *args, **kwargs) -> Parameter:
        """A user-facing helper function to help format an update for git-theta."""

    async def write_update(self, update: Parameter) -> metadata.LfsMetadata:
        """Save and serialize (just) the update weights."""
        if not isinstance(update, dict):
            update = {"parameter": update}
        serialized_update = await self.serializer.serialize(update)
        lfs_pointer = await git_utils.git_lfs_clean(serialized_update)
        return metadata.LfsMetadata.from_pointer(lfs_pointer)

    # TODO: Revisit what the metadata a write takes, right now it gets the full
    # metadata of the previous parameter value but only uses the update type. If
    # we do call get_previous_metadata on it, like we do in apply, the result is
    # that a parameter value is skipped and we calculate the incremental update
    # from 2 steps back, which can be a foot-gun.
    async def write(
        self,
        param: Parameter,
        param_keys,
        *,
        prev_metadata: metadata.ParamMetadata,
        repo,
        path: str,
        **kwargs,
    ) -> metadata.LfsMetadata:
        """Serialize and save a parameter with git-lfs as a delta from the previous value."""
        logging.debug(f"Writing {self.name} update for {'/'.join(param_keys)}")
        previous_value = await self.get_previous_value(
            prev_metadata, param_keys, repo=repo, path=path
        )
        if self.update_information is not None and self.will_update(param_keys):
            update_value = await self.read_update(param_keys)
            # Calculate and hash the *new* value so that we can update the
            # metadata when using side-loaded information.
            new_value = await self.apply_update(update_value, previous_value)
            new_hash = lsh.get_lsh().hash(new_value)
            return await self.write_update(update_value), new_hash
        else:
            update_value = await self.calculate_update(param, previous_value)
            return await self.write_update(update_value), None

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
        # param_metadata is the metadata for the parameter as it is *at this
        # commit*.
        prev_metadata = await self.get_previous_metadata(
            param_metadata, param_keys, repo=repo, path=path
        )
        prev_value = await self.get_previous_value(
            prev_metadata, param_keys, repo=repo, path=path
        )
        return await self.apply_update(update_value, prev_value)


def get_update_handler_name(update_type: Optional[str] = None) -> str:
    return update_type or utils.EnvVarConstants.UPDATE_TYPE


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
