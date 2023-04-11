"""Clean and Smudge filters for version controlling machine learning models."""

import argparse
import logging
import os
import sys
import tempfile

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

logging.basicConfig(
    level=logging.DEBUG,
    # Log to a file for clean/smudge as they don't appear on the console when called via git.
    filename=os.path.join(tempfile.gettempdir(), "git-theta.log"),
    format="git-theta-filter: [%(asctime)s] [%(funcName)s] %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="git-theta filter program")
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    clean_parser = subparsers.add_parser("clean", help="clean filter")
    clean_parser.add_argument("file", help="file being passed to clean filter")
    clean_parser.set_defaults(func=run_clean)

    smudge_parser = subparsers.add_parser("smudge", help="smudge filter")
    smudge_parser.add_argument("file", help="file being passed to smudge filter")
    smudge_parser.set_defaults(func=run_smudge)

    args = parser.parse_args()
    return args


# TODO: Move to a `filters.py` file.
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

    async def _clean(param_keys, new_param):
        logging.debug(f"Cleaning {'/'.join(param_keys)}")
        # Get the metadata from the previous version of the parameter
        param_metadata = prev_metadata.get(param_keys)
        # Create new metadata from the current value
        new_tensor_metadata = metadata.TensorMetadata.from_tensor(new_param)

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
        return param_keys, new_param_metadata

    # Sort the keys so we don't get changing diffs based on serialization order.
    sorted_checkpoint = dict(sorted(checkpoint.flatten().items()))
    return metadata.Metadata(
        **async_utils.run(
            async_utils.run_map(
                sorted_checkpoint,
                _clean,
                max_concurrency=EnvVarConstants.MAX_CONCURRENCY,
            )
        )
    ).unflatten()


def run_clean(args):
    """
    Implements clean filter for model files
    """
    logging.debug(f"Running clean filter on {args.file}")
    repo = git_utils.get_git_repo()
    checkpoint_handler = checkpoints.get_checkpoint_handler()
    model_checkpoint = checkpoint_handler.from_file(sys.stdin.buffer)
    new_metadata = clean(model_checkpoint, repo, args.file)
    new_metadata.write(sys.stdout)
    # If we had side-loaded information, write it out so we don't get false
    # positives for `git status`
    if EnvVarConstants.UPDATE_DATA_PATH:
        smudge(new_metadata, repo, args.file)


# TODO: Now that we have this as a separate function, us it (instead of
# `subprocess.run`) in the manual merge escape hatch.
def smudge(
    cleaned_metadata: metadata.Metadata, repo: git.Repo, path: str
) -> checkpoints.Checkpoint:
    """Convert cleaned `Metadata` to a `Checkpoint`."""
    curr_metadata = cleaned_metadata.flatten()

    async def _smudge(param_keys, param_metadata):
        logging.debug(f"Smudging {'/'.join(param_keys)}")
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


def run_smudge(args):
    """
    Implements smudge filter for model files
    """
    logging.debug(f"Running smudge filter on {args.file}")

    repo = git_utils.get_git_repo()
    curr_metadata = metadata.Metadata.from_file(sys.stdin)
    model_checkpoint = smudge(curr_metadata, repo, args.file)
    model_checkpoint.save(sys.stdout.buffer)


def main():
    args = parse_args()
    git_utils.set_hooks()
    args.func(args)


if __name__ == "__main__":
    main()
