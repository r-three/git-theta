import git
import os
import json
import logging
import io
import torch
import subprocess


def get_git_repo():
    """
    Create a git.Repo object for this repository

    Returns
    -------
    git.Repo
        Repo object for the current git repository
    """
    return git.Repo(os.getcwd(), search_parent_directories=True)


def create_git_cml(repo):
    """
    If not already created, create $git_root/.git_cml and return path

    Parameters
    ----------
    git_root : str
        path to git repository's root directory

    Returns
    -------
    str
        path to $git_root/.git_cml directory
    """
    git_cml = os.path.join(repo.working_dir, ".git_cml")
    if not os.path.exists(git_cml):
        logging.debug(f"Creating git cml directory {git_cml}")
        os.makedirs(git_cml)
    return git_cml


def create_git_cml_model_dir(repo, model_path):
    """
    If not already created, create directory under $git_root/.git_cml/ to store a model and return path

    Parameters
    ----------
    repo : git.Repo
        Repo object for the current git repository

    model_path : str
        path to model file being saved

    Returns
    -------
    str
        path to $git_root/.git_cml/$model_name directory
    """
    git_cml = create_git_cml(repo)
    model_file = os.path.basename(model_path)
    git_cml_model = os.path.join(git_cml, model_file)

    if not os.path.exists(git_cml_model):
        logging.debug(f"Creating model directory {git_cml_model}")
        os.makedirs(git_cml_model)
    return git_cml_model


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
            return f.readlines()
    else:
        return []


def write_gitattributes(gitattributes_file, attributes):
    """
    Write list of attributes to this repo's .gitattributes file

    Parameters
    ----------
    gitattributes_file : str
        Path to this repo's .gitattributes file

    attributes : List[str]
        Attributes to write to .gitattributes
    """
    with open(gitattributes_file, "w") as f:
        f.writelines(attributes)


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
    # N.b. this will track all files starting with a number under .git_cml/<model name>/<parameter group>/params/.
    # If we need more fine-grained control, we should modify the code to run `git lfs track` for each file we want to track
    track_glob = os.path.relpath(
        os.path.join(directory, "**", "params", "[0-9]*"), repo.working_dir
    )
    out = subprocess.run(["git", "lfs", "track", f'"{track_glob}"'])
    return out.returncode
