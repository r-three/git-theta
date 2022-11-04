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


def get_git_cml(repo, create=False):
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
    if not os.path.exists(git_cml) and create:
        logging.debug(f"Creating git cml directory {git_cml}")
        os.makedirs(git_cml)
    return git_cml
    
def get_git_cml_model_dir(repo, model_path , create=False):
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
    git_cml = get_git_cml(repo)
    # model_file = os.path.basename(model_path)
    # model_relative_path = get_relative_path_to_root(repo, model_path)
    # git_cml_model = os.path.join(git_cml, model_file)
    git_cml_model_dir = os.path.join(git_cml, model_path)

    if not os.path.exists(git_cml_model_dir) and create:
        logging.debug(f"Creating model directory {git_cml_model_dir}")
        os.makedirs(git_cml_model_dir)
    else:
        # need to raise an error?
        pass
    return git_cml_model_dir

def get_relative_path_from_root(repo, path):
    """
    Get relative path from repo root
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
        repo.git.rm('-r', f)
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
    out = subprocess.run(["git", "lfs", "track", f'"{track_glob}"'], cwd=repo.working_dir)
    return out.returncode
