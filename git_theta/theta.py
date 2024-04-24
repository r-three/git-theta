"""Module for reading and writing to .git/theta"""

import functools
import json
import logging
import os
import re

from file_or_name import file_or_name

from git_theta import utils


class CommitInfo:
    def __init__(self, oids):
        self.oids = set(oids) if oids else set()
        if not all(map(utils.is_valid_oid, self.oids)):
            invalid_oids = filter(lambda x: not utils.is_valid_oid(x), self.oids)
            raise ValueError(f"Invalid LFS object-ids {list(invalid_oids)}")

    def __eq__(self, other):
        return self.oids == other.oids

    @classmethod
    @file_or_name(f="r")
    def from_file(cls, f):
        commit_info_dict = json.load(f)
        return cls(commit_info_dict.get("oids"))

    @file_or_name(f="w")
    def write(self, f):
        commit_info_dict = {"oids": list(self.oids)}
        json.dump(commit_info_dict, f, indent=4)


class ThetaCommits:
    def __init__(self, repo):
        self.repo = repo
        self.path = os.path.abspath(os.path.join(repo.git_dir, "theta", "commits"))
        os.makedirs(self.path, exist_ok=True)
        self.logger = logging.getLogger("git_theta")

    @staticmethod
    def combine_oid_sets(oid_sets):
        return functools.reduce(lambda a, b: a.union(b), oid_sets, set())

    def get_commit_path(self, commit_hash):
        if not utils.is_valid_commit_hash(commit_hash):
            raise ValueError(f"Invalid commit hash {commit_hash}")
        return os.path.join(self.path, commit_hash)

    def get_commit_info(self, commit_hash):
        path = self.get_commit_path(commit_hash)
        if not (os.path.exists(path) and os.path.isfile(path)):
            raise ValueError(f"commit {commit_hash} is not found in {self.path}")
        commit = CommitInfo.from_file(path)
        return commit

    def get_commit_info_range(self, start_hash, end_hash):
        self.logger.debug(f"Getting commits from {start_hash}..{end_hash}")
        # N.b. the all-zero hash is used by git to indicate a non-existent start hash
        # For example, a git pre-push hook will receive the all-zero hash if the remote ref does not have any commit history
        if re.match("^0{40}$", start_hash):
            commits = list(self.repo.iter_commits(end_hash))
        else:
            commits = list(self.repo.iter_commits(f"{start_hash}..{end_hash}"))

        self.logger.debug(f"Found commits {commits}")
        commit_infos = [self.get_commit_info(commit.hexsha) for commit in commits]
        return commit_infos

    def get_commit_oids(self, commit_hash):
        self.logger.debug(f"Getting oids from commit {commit_hash}")
        commit_info = self.get_commit_info(commit_hash)
        oids = commit_info.oids
        self.logger.debug(f"Found oids {oids}")
        return oids

    def get_commit_oids_range(self, start_hash, end_hash):
        self.logger.debug(f"Getting oids from commit range {start_hash}..{end_hash}")
        commit_infos = self.get_commit_info_range(start_hash, end_hash)
        oids = ThetaCommits.combine_oid_sets(
            [commit_info.oids for commit_info in commit_infos]
        )
        self.logger.debug(f"Found oids {oids}")
        return oids

    def get_commit_oids_ranges(self, *ranges):
        oids = [
            self.get_commit_oids_range(start_hash, end_hash)
            for start_hash, end_hash in ranges
        ]
        return ThetaCommits.combine_oid_sets(oids)

    def write_commit_info(self, commit_hash, commit_info):
        self.logger.debug(f"Writing commit_info to commit {commit_hash}")
        if not utils.is_valid_commit_hash(commit_hash):
            raise ValueError(f"Cannot write commit info for invalid hash {commit_hash}")
        path = self.get_commit_path(commit_hash)
        if os.path.exists(path):
            raise ValueError(
                f"Cannot duplicate commit info at {path}. Something is wrong!"
            )
        commit_info.write(path)
