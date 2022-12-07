import os
import json
import re
import logging
import functools
from file_or_name import file_or_name


class ThetaCommits:
    def __init__(self, repo):
        self.repo = repo
        self.path = os.path.abspath(os.path.join(repo.git_dir, "theta", "commits"))
        os.makedirs(self.path, exist_ok=True)

    @staticmethod
    @file_or_name(f="r")
    def load_commit_file(f):
        return json.load(f)

    @staticmethod
    @file_or_name(f="w")
    def write_commit_file(f, oids):
        return json.dump(list(oids), f)

    @staticmethod
    def get_oids(commit):
        return set(commit)

    @staticmethod
    def combine_oid_sets(oids):
        return functools.reduce(lambda a, b: a.union(b), oids, set())

    def get_commit_path(self, commit_hash):
        return os.path.join(self.path, commit_hash)

    def get_commit(self, commit_hash):
        path = self.get_commit_path(commit_hash)
        if not (os.path.exists(path) and os.path.isfile(path)):
            raise ValueError(f"commit {commit_hash} is not found in {self.path}")
        commit = self.load_commit_file(path)
        return commit

    def get_commit_range(self, start_commit, end_commit):
        if re.match("^0+$", start_commit):
            commits = self.repo.iter_commits(end_commit)
        else:
            commits = self.repo.iter_commits(start_commit, end_commit)

        commits = [commit.hexsha for commit in commits]
        return commits

    def get_commit_oids(self, commit_hash):
        logging.debug(f"Getting oids from commit {commit_hash}")
        commit = self.get_commit(commit_hash)
        oids = self.get_oids(commit)
        logging.debug(f"Found oids {oids}")
        return oids

    def get_commit_oids_range(self, start_commit, end_commit):
        logging.debug(f"Getting oids from commit range {start_commit}..{end_commit}")
        commits = self.get_commit_range(start_commit, end_commit)
        oids = [self.get_commit_oids(commit) for commit in commits]
        return self.combine_oid_sets(oids)

    def get_commit_oids_ranges(self, *ranges):
        oids = [
            self.get_commit_oids_range(start_commit, end_commit)
            for start_commit, end_commit in ranges
        ]
        return self.combine_oid_sets(oids)

    def write_commit_oids(self, commit_hash, oids):
        logging.debug(f"Writing oids {oids} to commit {commit_hash}")
        path = self.get_commit_path(commit_hash)
        if os.path.exists(path):
            raise ValueError(f"Cannot duplicate commit at {path}. Something is wrong!")

        self.write_commit_file(path, oids)


class ThetaDirectory:
    def __init__(self, repo):
        self.path = os.path.abspath(os.path.join(repo.git_dir, "theta"))
        self.tmp = os.path.join(self.path, "tmp")
        os.makedirs(self.tmp, exist_ok=True)
        self.commits = ThetaCommits(repo)
