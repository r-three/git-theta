"""User facing functions to interact with git-theta without incurring extra io costs."""

import io
import sys
from typing import Optional, Union

import git

import git_theta
from git_theta import checkpoints, filters, git_utils, metadata, utils


def save_to_git(
    state_dict,
    path: str,
    commit_msg: str,
    tag: Optional[str] = None,
    checkpoint_type: str = "pytorch",
    checkout: bool = False,
) -> git.Commit:
    """Save a model using git-theta without needing to write it to the working tree.

    Args:
      state_dict: The model weights in the framework-native format.
      path: The path where the model will be saved.
      commit_msg: The message to include in the new commit.
      tag: If provided, a tag to add to the new commit.
      checkpoint_type: The checkpoint format name, used to get the checkpoint plugin.
      checkout: If true, the new commit will be checked out (This incurs extra
        compute and I/O cost as the model will be moved from git-storage to the
        working tree).

    Returns:
      The GitPython object representing the commit made with this save. Includes
      information like the sha.
    """
    repo = git_utils.get_git_repo()
    # Convert the deep learning framework native state dict into our checkpoint format.
    checkpoint_handler = checkpoints.get_checkpoint_handler(checkpoint_type)
    ckpt = checkpoint_handler.from_framework(state_dict)
    # Convert the checkpoint into the cleaned metadata file.
    metadata = filters.clean(ckpt, repo, path)
    # Capture metadata writing into a string.
    metadata = str(metadata)
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
    if checkout:
        repo.git.checkout(sha)
    if tag is not None:
        repo.create_tag(tag, ref=sha)
    return sha


def load_from_git(
    sha_or_tag: Union[str, git.Commit],
    path: str,
    checkpoint_type: str = "pytorch",
    checkout: bool = False,
):
    """Load a model from git-theta without having it checked out.

    Args:
      sha_or_tag: A reference to the commit to load the model from. It can be the
        sha1 or a tag.
      path: The path to where the model was saved in the working tree.
      checkpoint_type: The checkpoint format name, used to get the checkpoint plugin.
      checkout: If true, the commit is also checked out, keeping the on disk model
        in sync with the in-memory model (This incurs extra compute and I/O cost
        as the model will be moved from git-storage to the working tree).

    Returns:
      The loaded model in the checkpoint native format.
    """
    repo = git_utils.get_git_repo()
    # Set the checkpoint type env variable so that it respects the user input.
    utils.EnvVarConstants.CHECKPOINT_TYPE = checkpoint_type
    # Look up the metadata for this checkpoint in git.
    metadata_blob = git_utils.get_file_version(repo, path, sha_or_tag)
    # Build the metadata object from the blob data.
    metadata_obj = metadata.Metadata.from_file(metadata_blob.data_stream)
    # Convert the metadata into a checkpoint with weights.
    ckpt = filters.smudge(metadata_obj, repo, path)
    # Checkout the commit we are loading from so the state on disk matches the
    # state in memory
    if checkout:
        repo.git.checkout(sha_or_tag)
    # Convert the checkpoint to the native state dict.
    return ckpt.to_framework()
