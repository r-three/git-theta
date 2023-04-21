"""User facing functions to interact with git-theta without incurring extra io costs."""

import io
import sys
from typing import Optional, Union

import git

import git_theta
from git_theta import checkpoints, filters, git_utils, metadata, utils


# TODO: Should we have a `materialize` flag that writes the model to disk so you
# don't have git status showing a diff?
def save(
    state_dict,
    path: str,
    commit_msg: Optional[str] = None,
    checkpoint_type: str = "pytorch",
) -> git.Commit:
    """Save a model using git-theta without writing it to disk."""
    commit_msg = commit_msg if commit_msg else f"Committing {path}"
    repo = git_utils.get_git_repo()
    # Convert the deep learning framework native state dict into our checkpoint format.
    checkpoint_handler = checkpoints.get_checkpoint_handler(checkpoint_type)
    ckpt = checkpoint_handler.from_framework(state_dict)
    # Convert the checkpoint into the cleaned metadata file.
    metadata = filters.clean(ckpt, repo, path)
    # Capture metadata writing into a string.
    with io.StringIO() as f:
        metadata.write(f)
        metadata = f.getvalue()
    # Convert the metadata file into a git blob without having it on disk.
    blob = git_theta.git_utils.make_blob(repo, metadata, path)
    # Add the metadata to staging.
    repo.index.add([blob])
    # Commit the metadata.
    if sys.platform in ("win32", "cygwin"):
        # When you use GitPython to commit, things like hooks, i.e. our post-commit
        # hook, run as subprocesses. Currently it seems that running shell scripts
        # with the subprocess does not work in windows.
        # Commit using the GitPython wrapper around the `git commit` command. This
        # way hooks will be handled the same way as a normal commit.
        repo.git.commit(m=commit_msg)
        # We now need to get the sha manually in order to reference this commit.
        sha = repo.commit("HEAD")
    else:
        # Commit directly from python
        sha = repo.index.commit(commit_msg)
    return sha


# TODO: Should we add something like a `checkout` parameter that actually checks
# the model out so we are at the right commit after using this model?
def load(
    sha_or_tag: Union[str, git.Commit],
    path: str,
    checkpoint_type: str = "pytorch",
):
    """Load a model from git-theta without having it checked out."""
    repo = git_utils.get_git_repo()
    # Set the checkpoint type env variable so that it respects the user input.
    utils.EnvVarConstants.CHECKPOINT_TYPE = checkpoint_type
    # Look up the metadata for this checkpoint in git.
    metadata_blob = git_utils.get_file_version(repo, path, sha_or_tag)
    # Build the metadata object from the blob data.
    metadata_obj = metadata.Metadata.from_file(metadata_blob.data_stream)
    # Convert the metadata into a checkpoint with weights.
    ckpt = filters.smudge(metadata_obj, repo, path)
    # Convert the checkpoint to the native state dict.
    return ckpt.to_framework()
