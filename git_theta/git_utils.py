"""Utilities for manipulating git."""

import copy
import dataclasses
import filecmp
import fnmatch
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Union

import git
import gitdb

# TODO(bdlester): importlib.resources doesn't have the `.files` API until python
# version `3.9` so use the backport even if using a python version that has
# `importlib.resources`.
if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from file_or_name import file_or_name

from git_theta import async_utils

# These are the git attributes that git-theta currently uses to manage checked-in
# files. Defined as a variable in case extra functionality ever requires more
# attributes.
THETA_ATTRIBUTES = ("filter", "merge", "diff")
THETA_CONFIG_KEYS = (
    "filter.theta.clean=",
    "filter.theta.smudge=",
    "merge.theta.name=",
    "merge.theta.driver=",
    "diff.theta.command=",
)


def get_git_repo():
    """
    Create a git.Repo object for this repository

    Returns
    -------
    git.Repo
        Repo object for the current git repository
    """
    return git.Repo(os.getcwd(), search_parent_directories=True)


def set_hooks():
    repo = get_git_repo()
    hooks_dir = os.path.join(repo.git_dir, "hooks")
    package = importlib_resources.files("git_theta")
    for hook in ["pre-push", "post-commit"]:
        with importlib_resources.as_file(package.joinpath("hooks", hook)) as hook_src:
            hook_dst = os.path.join(hooks_dir, hook)
            if not (os.path.exists(hook_dst) and filecmp.cmp(hook_src, hook_dst)):
                shutil.copy(hook_src, hook_dst)


def get_relative_path_from_root(repo, path):
    """
    Get relative path from repo root

    Parameters
    ----------
    repo : git.Repo
        Repo object for the current git repository

    path : str
        any path

    Returns
    -------
    str
        path relative from the root
    """

    relative_path = os.path.relpath(os.path.abspath(path), repo.working_dir)
    return relative_path


def get_absolute_path(repo: git.Repo, relative_path: str) -> str:
    """Get the absolute path of a repo relative path.

    Parameters
    ----------
    repo
        Repo object for the current git repository.
    path
        The relative path to a file from the root of the git repo.

    Returns
    -------
    str
        The absolute path to the file.
    """
    return os.path.abspath(os.path.join(repo.working_dir, relative_path))


def get_gitattributes_file(repo):
    """
    Get path to this repo's .gitattributes file

    Parameters
    ----------
    repo : git.Repo
        Repo object for the current git repository

    Returns
    -------
    str
        path to $git_root/.gitattributes
    """
    return os.path.join(repo.working_dir, ".gitattributes")


@dataclasses.dataclass
class GitAttributes:
    """Git attributes for a file that matches pattern."""

    pattern: str
    attributes: Dict[str, str]
    raw: Optional[str] = None

    def __str__(self):
        if self.raw:
            return self.raw
        attrs = " ".join(f"{k}={v}" if v else k for k, v in self.attributes.items())
        return f"{self.pattern} {attrs}"

    def __eq__(self, o):
        raw_eq = self.raw == o.raw if self.raw and o.raw else True
        return self.pattern == o.pattern and self.attributes == o.attributes and raw_eq


def read_gitattributes(gitattributes_file) -> List[GitAttributes]:
    """
    Read contents of this repo's .gitattributes file

    Parameters
    ----------
    gitattributes_file : str
        Path to this repo's .gitattributes file

    Returns
    -------
    List[str]
        lines in .gitattributes file
    """
    if os.path.exists(gitattributes_file):
        with open(gitattributes_file, "r") as f:
            return [parse_gitattributes(line.rstrip("\n")) for line in f]
    else:
        return []


def parse_gitattributes(gitattributes: str) -> GitAttributes:
    # TODO: Fix for escaped patterns
    pattern, *attributes = gitattributes.split(" ")
    attrs = {}
    # Overwrite as we go to get the LAST attribute behavior
    for attribute in attributes:
        if "=" in attribute:
            key, value = attribute.split("=")
        # TODO: Update to handle unsetting attributes like "-diff". Currently we
        # just copy then as keys for printing but don't check their semantics,
        # for example a file with an unset diff does currently throw an error
        # when adding git-theta tracking.
        else:
            key = attribute
            value = None
        attrs[key] = value
    return GitAttributes(pattern, attrs, gitattributes)


@file_or_name(gitattributes_file="w")
def write_gitattributes(
    gitattributes_file: Union[str, io.FileIO], attributes: List[GitAttributes]
):
    """
    Write list of attributes to this repo's .gitattributes file

    Parameters
    ----------
    gitattributes_file:
        Path to this repo's .gitattributes file

    attributes:
        Attributes to write to .gitattributes
    """
    gitattributes_file.write("\n".join(map(str, attributes)))
    # End file with newline.
    gitattributes_file.write("\n")


def add_theta_to_gitattributes(
    gitattributes: List[GitAttributes],
    path: str,
    theta_attributes: Sequence[str] = THETA_ATTRIBUTES,
) -> List[GitAttributes]:
    """Add git attributes required by git-theta for path.

    If there is a pattern that covers the current file that applies the git-theta
    attributes, no new pattern is added. If there is a pattern that covers the
    current file and sets attributes used by git-theta an error is raised. If
    there is a pattern that sets non-overlapping attributes they are copied into
    a new path-specific pattern. If there is no match, a new path-specific
    pattern is always created.

    Parameters
    ----------
        gitattributes: A list of parsed git attribute entries.
        path: The path to the model we are adding a filter to.

    Raises
    ------
    ValueError
        `path` is covered by an active git attributes entry that sets merge,
        filter, or diff to a value other than "theta".

    Returns
    -------
    List[GitAttributes]
        The git attributes write to the new gitattribute file with a (possibly)
        new (filter|merge|diff)=theta added that covers `path`.
    """
    previous_attribute = None
    # Find if an active gitattribute entry applies to path
    for gitattribute in gitattributes[::-1]:
        if fnmatch.fnmatchcase(path, gitattribute.pattern):
            previous_attribute = gitattribute
            break
    # If path is already managed by a git attributes entry.
    if previous_attribute:
        # If all of the theta attributes are set, we don't do anything.
        if all(
            previous_attribute.attributes.get(attr) == "theta"
            for attr in theta_attributes
        ):
            return gitattributes
        # If any of the attributes theta uses is set to something else, error out.
        if any(
            attr in previous_attribute.attributes
            and previous_attribute.attributes[attr] != "theta"
            for attr in theta_attributes
        ):
            raise ValueError(
                f"Git Attributes used by git-theta are already set for {path}. "
                f"Found filter={previous_attribute.attributes.get('filter')}, "
                f"diff={previous_attribute.attributes.get('diff')}, "
                f"merge={previous_attribute.attributes.get('merge')}."
            )
    # If the old entry set other attributes, make sure they are preserved.
    attributes = (
        copy.deepcopy(previous_attribute.attributes) if previous_attribute else {}
    )
    for attr in theta_attributes:
        attributes[attr] = "theta"
    new_attribute = GitAttributes(path, attributes)
    gitattributes.append(new_attribute)
    return gitattributes


def get_gitattributes_tracked_patterns(
    gitattributes_file, theta_attributes: Sequence[str] = THETA_ATTRIBUTES
):
    gitattributes = read_gitattributes(gitattributes_file)
    theta_attributes = [
        attr
        for attr in gitattributes
        if attr.attributes.get(a) == "theta"
        for a in theta_attributes
    ]
    return [attr.pattern for attr in theta_attributes]
    # TODO: Correctly handle patterns with escaped spaces in them
    patterns = [attribute.split(" ")[0] for attribute in theta_attributes]
    return patterns


def is_theta_tracked(
    path: str,
    gitattributes: List[GitAttributes],
    theta_attributes: Sequence[str] = THETA_ATTRIBUTES,
) -> bool:
    """Check if `path` is tracked by git-theta based on `.gitattributes`.

    Note: The last line that matches in .gitattributes is the active one so
      start from the end. If the first match (really last) does not have the
      theta filter active then the file is not tracked by Git-Theta.
    """
    for attr in gitattributes[::-1]:
        if fnmatch.fnmatchcase(path, attr.pattern):
            return all(attr.attributes.get(a) == "theta" for a in theta_attributes)
    return False


def add_file(f, repo):
    """
    Add file to git staging area

    Parameters
    ----------
    f : str
        path to file
    repo : git.Repo
        Repo object for current git repository
    """
    logger = logging.getLogger("git_theta")
    logger.debug(f"Adding {f} to staging area")
    repo.git.add(f)


def remove_file(f, repo):
    """
    Remove file or directory and add change to staging area

    Parameters
    ----------
    f : str
        path to file or directory
    repo : git.Repo
        Repo object for current git repository
    """
    logger = logging.getLogger("git_theta")
    logger.debug(f"Removing {f}")
    if os.path.isdir(f):
        repo.git.rm("-r", f)
    else:
        repo.git.rm(f)


def get_file_version(repo, path: str, commit_hash_or_tag: Union[str, git.Commit]):
    """Get a specific version of a file.

    Args:
      repo: The git repository we are working with.
      path: The path to the file we are fetching.
      commit_hash_or_tag: The sha1 for the commit that has the file version we
        want. It can also be a tag for the commit.
    """
    path = get_relative_path_from_root(repo, path)
    try:
        # GitPython can take commit sha1's or tags (or commit objects) here and
        # it gives the same results.
        tree = repo.commit(commit_hash_or_tag).tree
        return tree[path]
    except (git.BadName, KeyError):
        return None


def get_head(repo):
    try:
        head = repo.commit("HEAD")
        return head.hexsha
    except git.BadName:
        return None


async def git_lfs_clean(file_contents: bytes) -> str:
    out = await async_utils.subprocess_run(
        ["git", "lfs", "clean"], input=file_contents, capture_output=True
    )
    return out.stdout.decode("utf-8")


async def git_lfs_smudge(pointer_file: str) -> bytes:
    out = await async_utils.subprocess_run(
        ["git", "lfs", "smudge"],
        input=pointer_file.encode("utf-8"),
        capture_output=True,
    )
    return out.stdout


async def git_lfs_push_oids(remote_name: str, oids: Sequence[str]) -> int:
    if oids:
        out = await async_utils.subprocess_run(
            ["git", "lfs", "push", "--object-id", re.escape(remote_name)] + list(oids),
        )
        return out.returncode
    return 0


def parse_pre_push_args(lines):
    lines_parsed = [
        re.match(
            r"^(?P<local_ref>[^\s]+)\s+(?P<local_sha1>[a-f0-9]{40})\s+(?P<remote_ref>[^\s]+)\s+(?P<remote_sha1>[a-f0-9]{40})\s+$",
            l,
        )
        for l in lines
    ]
    return lines_parsed


def is_git_lfs_installed():
    """
    Helper function that checks if git-lfs is installed to prevent future errors with git-theta
    """
    try:
        results = subprocess.run(
            ["git", "lfs", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return results.returncode == 0
    except:
        return False


def is_git_theta_installed(
    repo: Optional = None, git_theta_config_keys: Sequence[str] = THETA_CONFIG_KEYS
) -> bool:
    """Check if git-theta is installed.

    By checking `git config --list` we see all configuration options reguardless
    of if they setup git-theta up in ~/.gitconfig, ${repo}/.git/config, etc.

    Note:
      This check requires you to be in the repo, but this is fine for our use
      cases.
    """
    repo = get_git_repo() if repo is None else repo
    config = repo.git.config("--list")
    for config_key in git_theta_config_keys:
        if config_key not in config:
            return False
    return True


def make_blob(repo, contents: str, path: str):
    contents = contents.encode("utf-8")
    mode = 33188  # The mode magic number used for blobs in git python
    istream = repo.odb.store(
        gitdb.IStream(git.Blob.type, len(contents), io.BytesIO(contents))
    )
    # 0 is how far into staging the blob is, not sure what different stages do but 0 works for us.
    return git.BaseIndexEntry((mode, istream.binsha, 0, path))
