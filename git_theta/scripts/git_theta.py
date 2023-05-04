"""Installation and .git manipulation scripts."""

import argparse
import logging
import re
import sys

import git

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from git_theta import async_utils, config, git_utils, metadata, theta, utils

logging.basicConfig(
    level=logging.DEBUG,
    format="git-theta: [%(asctime)s] [%(funcName)s] %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="git-theta filter program")
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    subparsers.required = True

    post_commit_parser = subparsers.add_parser(
        "post-commit",
        help="post-commit command that records parameter groups changed in a commit",
    )
    post_commit_parser.set_defaults(func=post_commit)

    pre_push_parser = subparsers.add_parser(
        "pre-push",
        help="pre-push command used to send parameter groups to an LFS store. Should only be called internally by git push.",
    )
    pre_push_parser.add_argument(
        "remote_name", help="Name of the remote being pushed to"
    )
    pre_push_parser.add_argument(
        "remote_location", help="Location of the remote being pushed to"
    )
    pre_push_parser.set_defaults(func=pre_push)

    install_parser = subparsers.add_parser(
        "install", help="install command used to setup global .gitconfig"
    )
    install_parser.set_defaults(func=install)

    track_parser = subparsers.add_parser(
        "track",
        help="track command used to identify model checkpoint for git-theta to track",
    )
    track_parser.add_argument("pattern", help="model checkpoint file pattern to track")
    track_parser.add_argument(
        "--checkpoint_format",
        choices=[e.name for e in entry_points(group="git_theta.plugins.checkpoints")],
        default="pytorch",
        help="Checkpoint format",
    )
    track_parser.set_defaults(func=track)

    add_parser = subparsers.add_parser("add", help="add command used to stage files.")
    add_parser.add_argument("file", help="The file we are git adding.")
    add_parser.add_argument(
        "--update-type",
        choices=[e.name for e in entry_points(group="git_theta.plugins.updates")],
        help="Type of update being applied",
    )
    add_parser.add_argument("--update-data", help="Where update data is stored.")
    add_parser.set_defaults(func=add)

    args = parser.parse_known_args()
    return args


def post_commit(args):
    """
    Post-commit git hook that records the LFS objects that were created (equivalent to parameter groups that were modified) in this commit
    """
    repo = git_utils.get_git_repo()
    theta_commits = theta.ThetaCommits(repo)

    gitattributes = config.GitAttributesFile(repo)

    oids = set()
    commit = repo.commit("HEAD")
    for path in commit.stats.files.keys():
        if gitattributes.is_theta_tracked(path):
            curr_metadata = metadata.Metadata.from_file(commit.tree[path].data_stream)
            prev_metadata = metadata.Metadata.from_commit(repo, path, "HEAD~1")

            added, removed, modified = curr_metadata.diff(prev_metadata)
            oids.update([param.lfs_metadata.oid for param in added.flatten().values()])
            oids.update(
                [param.lfs_metadata.oid for param in modified.flatten().values()]
            )

    commit_info = theta.CommitInfo(oids)
    theta_commits.write_commit_info(commit.hexsha, commit_info)


def pre_push(args):
    """
    Pre-push git hook for sending objects to the LFS server
    """
    repo = git_utils.get_git_repo()
    theta_commits = theta.ThetaCommits(repo)

    # Read lines of the form <local ref> <local sha1> <remote ref> <remote sha1> LF
    lines = sys.stdin.readlines()
    lines_parsed = git_utils.parse_pre_push_args(lines)
    commit_ranges = [
        (l.group("remote_sha1"), l.group("local_sha1")) for l in lines_parsed
    ]
    oids = theta_commits.get_commit_oids_ranges(*commit_ranges)
    async_utils.run(git_utils.git_lfs_push_oids(args.remote_name, oids))


def install(args):
    """
    Install git-lfs and initialize the git-theta filter driver
    """
    # check if git-lfs is installed and abort if not
    if not git_utils.is_git_lfs_installed():
        print(
            f"git-theta depends on git-lfs and it does not appear to be installed. See installation directions at https://github.com/r-three/git-theta/blob/main/README.md#git-lfs-installation"
        )
        sys.exit(1)

    config_writer = git.GitConfigParser(
        git.config.get_config_path("global"), config_level="global", read_only=False
    )
    config_writer.set_value('filter "theta"', "clean", "git-theta-filter clean %f")
    config_writer.set_value('filter "theta"', "smudge", "git-theta-filter smudge %f")
    config_writer.set_value('filter "theta"', "required", "true")
    config_writer.set_value('merge "theta"', "name", "Merge Models with Git-Theta")
    config_writer.set_value('merge "theta"', "driver", "git-theta-merge %O %A %B %P")
    config_writer.set_value('diff "theta"', "command", "git-theta-diff")
    config_writer.release()


def track(args):
    """
    Track a particular model checkpoint file with git-theta
    """
    repo = git_utils.get_git_repo()

    gitattributes = config.GitAttributesFile(repo)
    gitattributes.add_theta(args.pattern)
    gitattributes.write()
    git_utils.add_file(gitattributes.file, repo)

    thetaconfig = config.ThetaConfigFile(repo)
    thetaconfig.add_pattern(vars(args))
    thetaconfig.write()
    git_utils.add_file(thetaconfig.file, repo)


def add(args, unparsed_args):
    repo = git_utils.get_git_repo()
    env_vars = {
        "GIT_THETA_UPDATE_TYPE": args.update_type,
        "GIT_THETA_UPDATE_DATA_PATH": args.update_data,
    }
    # The most common use for `git theta add` is when you have side-loaded
    # information and thus the main checkpoint file has not been modified. This
    # results in git not running the add command as the modification time has
    # not changed. We touch the file so it will actually get added.
    utils.touch(args.file)
    with repo.git.custom_environment(**env_vars):
        repo.git.add(args.file, *unparsed_args)


def main():
    args, unparsed_args = parse_args()
    if not args.func == install:
        git_utils.set_hooks()
    if args.func == add:
        args.func(args, unparsed_args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
