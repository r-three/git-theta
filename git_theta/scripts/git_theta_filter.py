"""Clean and Smudge filters for version controlling machine learning models."""

import argparse
import logging
import sys

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
    filename="/tmp/git-theta.log",
    format="git-theta-filter: [%(asctime)s] [%(funcName)s] %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="git-theta filter program")
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    clean_parser = subparsers.add_parser("clean", help="clean filter")
    clean_parser.add_argument("file", help="file being passed to clean filter")
    clean_parser.set_defaults(func=clean)

    smudge_parser = subparsers.add_parser("smudge", help="smudge filter")
    smudge_parser.add_argument("file", help="file being passed to smudge filter")
    smudge_parser.set_defaults(func=smudge)

    args = parser.parse_args()
    return args


def clean(args):
    """
    Implements clean filter for model files
    Metadata file looks as follows:
    {
    "model/scoping/to/param/1-weight": {
        "tensor_metadata": {
            "shape": List[str],
            "dtype": str,
            "hash": str,
        },
    },
    ...,
    "model/scoping/to/param/2-bias": {
        "tensor_metadata": {
            "shape": List[str],
            "dtype": str,
            "hash": str,
        },
    },
    ...,
    }
    """
    logging.debug(f"Running clean filter on {args.file}")
    repo = git_utils.get_git_repo()
    checkpoint_handler = checkpoints.get_checkpoint_handler()
    model_checkpoint = checkpoint_handler.from_file(sys.stdin.buffer)

    # Note: If the update serializer is configurable per-parameter, it will need to be created inside _clean
    update_serializer = params.get_update_serializer()
    prev_metadata = metadata.Metadata.from_commit(repo, args.file, "HEAD").flatten()

    async def _clean(param_keys, new_param):
        logging.debug(f"Cleaning {'/'.join(param_keys)}")
        param_metadata = prev_metadata.get(param_keys)
        new_tensor_metadata = metadata.TensorMetadata.from_tensor(new_param)

        # If the parameter tensor has not changed, just keep the metadata the same
        if (
            param_metadata
            and param_metadata.tensor_metadata.shape == new_tensor_metadata.shape
            and param_metadata.tensor_metadata.dtype == new_tensor_metadata.dtype
        ):
            hasher = lsh.get_lsh()
            hash_distance = hasher.distance(
                param_metadata.tensor_metadata.hash, new_tensor_metadata.hash
            )
            # If hash_distance < PARAMETER_ATOL, assume the tensors pass np.allclose and parameter hasn't changed
            if hash_distance < EnvVarConstants.PARAMETER_ATOL:
                return param_keys, param_metadata
            # If PARAMETER_ATOL < hash_distance < LSH_THRESHOLD, load parameters and check if parameter has changed with np.allclose
            elif hash_distance < EnvVarConstants.LSH_THRESHOLD:
                update_handler = updates.get_update_handler(
                    param_metadata.theta_metadata.update_type
                )(update_serializer)
                param = await update_handler.apply(
                    param_metadata, param_keys, repo=repo, path=args.file
                )
                if np.allclose(
                    param,
                    new_param,
                    rtol=EnvVarConstants.PARAMETER_RTOL,
                    atol=EnvVarConstants.PARAMETER_ATOL,
                ):
                    return param_keys, param_metadata

        update_handler = updates.get_update_handler()(update_serializer)
        new_theta_metadata = metadata.ThetaMetadata(
            update_type=update_handler.name, last_commit=git_utils.get_head(repo)
        )
        lfs_metadata = await update_handler.write(
            new_param,
            param_keys,
            prev_metadata=param_metadata,
            repo=repo,
            path=args.file,
        )

        new_param_metadata = metadata.ParamMetadata(
            lfs_metadata=lfs_metadata,
            tensor_metadata=new_tensor_metadata,
            theta_metadata=new_theta_metadata,
        )
        return param_keys, new_param_metadata

    # Sort the keys so we don't get changing diffs based on serialization order.
    sorted_checkpoint = dict(sorted(model_checkpoint.flatten().items()))
    new_metadata = metadata.Metadata(
        **async_utils.run(
            async_utils.run_map(
                sorted_checkpoint,
                _clean,
                max_concurrency=EnvVarConstants.MAX_CONCURRENCY,
            )
        )
    )

    new_metadata.unflatten().write(sys.stdout)


def smudge(args):
    """
    Implements smudge filter for model files
    """
    logging.debug(f"Running smudge filter on {args.file}")

    repo = git_utils.get_git_repo()
    curr_metadata = metadata.Metadata.from_file(sys.stdin).flatten()

    async def _smudge(param_keys, param_metadata):
        logging.debug(f"Smudging {'/'.join(param_keys)}")
        update_handler = updates.get_update_handler(
            param_metadata.theta_metadata.update_type
        )(params.get_update_serializer())
        param_value = await update_handler.apply(
            param_metadata, param_keys, repo=repo, path=args.file
        )
        return param_keys, param_value

    model_dict = async_utils.run(
        async_utils.run_map(
            curr_metadata, _smudge, max_concurrency=EnvVarConstants.MAX_CONCURRENCY
        )
    )

    checkpoint_handler = checkpoints.get_checkpoint_handler()
    model_checkpoint = checkpoint_handler(model_dict).unflatten()
    model_checkpoint.save(sys.stdout.buffer)


def main():
    args = parse_args()
    git_utils.set_hooks()
    args.func(args)


if __name__ == "__main__":
    main()
