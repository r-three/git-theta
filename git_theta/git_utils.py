"""Utilities for manipulating git."""

import fnmatch
import git
import os
import json
import logging
import io
import re
import torch
from typing import List, Union
from collections import OrderedDict
import subprocess

from file_or_name import file_or_name


def get_git_repo():
    """
    Create a git.Repo object for this repository

    Returns
    -------
    git.Repo
        Repo object for the current git repository
    """
    return git.Repo(os.getcwd(), search_parent_directories=True)


def get_git_theta(repo):
    """
    If create argument is true, create $git_root/.git_theta and return path
    Otherwise return $git_root/.git_theta path

    Parameters
    ----------
    git_root : str
        path to git repository's root directory
    create : bool
        argument to create the directory

    Returns
    -------
    str
        path to $git_root/.git_theta directory
    """
    git_theta = os.path.join(repo.git_dir, "theta")
    os.makedirs(os.path.join(git_theta, "tmp"), exist_ok=True)
    return git_theta


def get_git_theta_model_dir(repo, model_path, create=False):
    """
    If create is true, create directory under $git_root/.git_theta/ to store a model and return path
    Otherwise just return path that stores a model

    Parameters
    ----------
    repo : git.Repo
        Repo object for the current git repository

    model_path : str
        path to model file being saved

    create : bool
        argument to create the directory
    Returns
    -------
    str
        path to $git_root/.git_theta/$model_path directory
    """
    git_theta = get_git_theta(repo)
    git_theta_model_dir = os.path.join(git_theta, model_path)

    if not os.path.exists(git_theta_model_dir) and create:
        logging.debug(f"Creating model directory {git_theta_model_dir}")
        os.makedirs(git_theta_model_dir)

    return git_theta_model_dir


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


def read_gitattributes(gitattributes_file):
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
            return [line.rstrip("\n") for line in f]
    else:
        return []


@file_or_name(gitattributes_file="w")
def write_gitattributes(
    gitattributes_file: Union[str, io.FileIO], attributes: List[str]
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
    gitattributes_file.write("\n".join(attributes))
    # End file with newline.
    gitattributes_file.write("\n")


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
    logging.debug(f"Adding {f} to staging area")
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
    logging.debug(f"Removing {f}")
    if os.path.isdir(f):
        repo.git.rm("-r", f)
    else:
        repo.git.rm(f)


def git_lfs_install():
    """
    Run the `git lfs install` command

    Returns
    -------
    int
        Return code of `git lfs install`
    """
    out = subprocess.run(["git", "lfs", "install"])
    return out.returncode


def git_lfs_track(repo, directory):
    """
    Run the `git lfs track` command to track the files under `directory`

    Parameters
    ----------
    repo : git.Repo
        Repo object for current git repository
    directory : str
        Track all files under this directory

    Returns
    -------
    int
        Return code of `git lfs track`
    """
    track_glob = os.path.relpath(
        os.path.join(directory, "**", "params", "[0-9]*"), repo.working_dir
    )
    out = subprocess.run(
        ["git", "lfs", "track", f'"{track_glob}"'], cwd=repo.working_dir
    )
    return out.returncode


@file_or_name(f="rb")
def git_lfs_clean(f):
    out = subprocess.run(
        ["git", "lfs", "clean"], input=f.read(), capture_output=True
    ).stdout
    out = re.match(
        "^version (?P<lfs_version>[^\s]*)\s*oid sha256:(?P<oid>[^\s]*)\s*size (?P<size>[0-9]*)$",
        out.decode("utf-8"),
    )
    return OrderedDict(
        {
            "lfs_version": out.group("lfs_version"),
            "oid": out.group("oid"),
            "size": out.group("size"),
        }
    )


@file_or_name(f="rb")
def git_lfs_smudge(f):
    out = subprocess.run(
        ["git", "lfs", "smudge"], input=f.read(), capture_output=True
    ).stdout
    return out


def git_lfs_push_oids(remote_name, oids):
    out = subprocess.run(
        ["git", "lfs", "push", "--object-id", remote_name] + list(oids)
    )
    return out


def add_filter_theta_to_gitattributes(gitattributes: List[str], path: str) -> str:
    """Add a filter=theta that covers file_name.

    Parameters
    ----------
        gitattributes: A list of the lines from the gitattribute files.
        path: The path to the model we are adding a filter to.

    Returns
    -------
    List[str]
        The lines to write to the new gitattribute file with a (possibly) new
        filter=theta added that covers the given file.
    """
    pattern_found = False
    new_gitattributes = []
    for line in gitattributes:
        # TODO(bdlester): Revisit this regex to see if it when the pattern
        # is escaped due to having spaces in it.
        match = re.match("^\s*(?P<pattern>[^\s]+)\s+(?P<attributes>.*)$", line)
        if match:
            # If there is already a pattern that covers the file, add the filter
            # to that.
            if fnmatch.fnmatchcase(path, match.group("pattern")):
                pattern_found = True
                if not "filter=theta" in match.group("attributes"):
                    line = f"{line.rstrip()} filter=theta"
        new_gitattributes.append(line)
    # If we don't find a matching pattern, add a new line that covers just this
    # specific file.
    if not pattern_found:
        new_gitattributes.append(f"{path} filter=theta")
    return new_gitattributes
