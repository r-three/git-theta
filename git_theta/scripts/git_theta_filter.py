"""Clean and Smudge filters for version controlling machine learning models."""

import argparse
import logging
import os
import sys
import tempfile

import git_theta
from git_theta import checkpoints, git_utils, metadata
from git_theta.filters import clean, smudge
from git_theta.utils import EnvVarConstants

git_theta.scripts.configure_logging("git-theta-filter")


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
