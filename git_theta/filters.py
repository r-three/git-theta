"""Clean and Filter functions."""

import logging

import git
import numpy as np

from git_theta import (
    async_utils,
    checkpoints,
    git_utils,
    lsh,
    metadata,
    params,
    updates,
)
from git_theta.utils import EnvVarConstants


def clean(
    checkpoint: checkpoints.Checkpoint, repo: git.Repo, path: str
) -> metadata.Metadata:
    """Convert a `Checkpoint` to cleaned `Metadata`."""
    # Note: If the update serializer is configurable per-parameter, it will
    # need to be created inside _clean
    update_serializer = params.get_update_serializer()
    # Create an update handler based on user input.
    update_handler = updates.get_update_handler()(
        update_serializer, EnvVarConstants.UPDATE_DATA_PATH
    )
    prev_metadata = metadata.Metadata.from_commit(repo, path, "HEAD").flatten()
    logger = logging.getLogger("git_theta")

    async def _clean(param_keys, new_param):
        logger.debug(f"Cleaning {'/'.join(param_keys)}")
        # Get the metadata from the previous version of the parameter
        param_metadata = prev_metadata.get(param_keys)
        # Create new metadata from the current value
        logger.debug(f"Making new Metadata for {'/'.join(param_keys)}")
        new_tensor_metadata = metadata.TensorMetadata.from_tensor(new_param)
        logger.debug(f"Finished new Metadata for {'/'.join(param_keys)}")

        # If the parameter tensor has not changed, just keep the metadata the same
        # TODO: Encapsulate this parameter check within an equality check.
        if (
            param_metadata
            and param_metadata.tensor_metadata.shape == new_tensor_metadata.shape
            and param_metadata.tensor_metadata.dtype == new_tensor_metadata.dtype
            # A parameter with a side-loaded update will not have changed in the
            # normal checkpoint, so ask the updater if it will be updated with
            # side-loaded information.
            and not update_handler.will_update(param_keys)
        ):
            # Compare the parameters using the LSH
            hasher = lsh.get_lsh()
            # TODO: Is is possible to make this comparison async?
            logger.debug(f"Comparing Hashes for: {'/'.join(param_keys)}")
            hash_distance = hasher.distance(
                param_metadata.tensor_metadata.hash, new_tensor_metadata.hash
            )
            # If hash_distance < PARAMETER_ATOL, assume the tensors pass
            # np.allclose and parameter hasn't changed
            if hash_distance < EnvVarConstants.PARAMETER_ATOL:
                return param_keys, param_metadata
            # If PARAMETER_ATOL < hash_distance < LSH_THRESHOLD, load parameters
            # and check if parameter has changed with np.allclose
            elif hash_distance < EnvVarConstants.LSH_THRESHOLD:
                # Load the previous parameter using the specific update handler
                # for that parameter.
                param_update_handler = updates.get_update_handler(
                    param_metadata.theta_metadata.update_type
                )(update_serializer)
                param = await param_update_handler.apply(
                    param_metadata, param_keys, repo=repo, path=path
                )
                if np.allclose(
                    param,
                    new_param,
                    rtol=EnvVarConstants.PARAMETER_RTOL,
                    atol=EnvVarConstants.PARAMETER_ATOL,
                ):
                    return param_keys, param_metadata

        # Create git-theta metadata for the new parameter.
        new_theta_metadata = metadata.ThetaMetadata(
            update_type=update_handler.name, last_commit=git_utils.get_head(repo)
        )
        # Write the new parameter into git-lfs
        lfs_metadata, param_hash = await update_handler.write(
            new_param,
            param_keys,
            prev_metadata=param_metadata,
            repo=repo,
            path=path,
        )
        # If we are an IncrementalUpdate, we need to re-calculate the hash
        # so it is based on the updated value, not the old one.
        if param_hash is not None:
            new_tensor_metadata.hash = param_hash
        # Combine metadata into single paramtere metadata blob
        new_param_metadata = metadata.ParamMetadata(
            lfs_metadata=lfs_metadata,
            tensor_metadata=new_tensor_metadata,
            theta_metadata=new_theta_metadata,
        )
        logger.debug(f"Finished Cleaning {'/'.join(param_keys)}")
        del new_param
        return param_keys, new_param_metadata

    # Sort the keys so we don't get changing diffs based on serialization order.
    sorted_checkpoint = dict(sorted(checkpoint.flatten().items()))
    if EnvVarConstants.LOW_MEMORY:
        # Run one at a time and delete the old values as you go
        # TODO: Is is possible/better to process the keys based on the size
        # of the tensor and resort later? Then you could do things like delete
        # all the small ones before you have to process the large one.
        logger.warning(
            "Runing Git-Theta in Low Memory Mode, no concurrency will be used, and references to parameter weights will be freed after use."
        )
        meta = {}
        for k in list(sorted_checkpoint.keys()):
            # Get the param while removing it from the dict, removing the
            # reference in the dict will allow the tensor to be gc'd
            v = sorted_checkpoint.pop(k)
            param_name, param_meta = async_utils.run(_clean(k, v))
            meta[param_name] = param_meta
            # Drop the reference to the value to allow it to be gc'd.
            del v
        return metadata.Metadata(meta).unflatten()
    return metadata.Metadata(
        **async_utils.run(
            async_utils.run_map(
                sorted_checkpoint,
                _clean,
                max_concurrency=EnvVarConstants.MAX_CONCURRENCY,
            )
        )
    ).unflatten()


# TODO: Now that we have this as a separate function, us it (instead of
# `subprocess.run`) in the manual merge escape hatch.
def smudge(
    cleaned_metadata: metadata.Metadata, repo: git.Repo, path: str
) -> checkpoints.Checkpoint:
    """Convert cleaned `Metadata` to a `Checkpoint`."""
    curr_metadata = cleaned_metadata.flatten()

    async def _smudge(param_keys, param_metadata):
        logger = logging.getLogger("git_theta")
        logger.debug(f"Smudging {'/'.join(param_keys)}")
        update_handler = updates.get_update_handler(
            param_metadata.theta_metadata.update_type
        )(params.get_update_serializer())
        param_value = await update_handler.apply(
            param_metadata, param_keys, repo=repo, path=path
        )
        return param_keys, param_value

    model_dict = async_utils.run(
        async_utils.run_map(
            curr_metadata, _smudge, max_concurrency=EnvVarConstants.MAX_CONCURRENCY
        )
    )

    checkpoint_handler = checkpoints.get_checkpoint_handler()
    return checkpoint_handler(model_dict).unflatten()
