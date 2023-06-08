"""Shared fixtures for running tests"""

import os
import random
import shutil
import string

import git
import numpy as np
import pytest

from git_theta import metadata, theta


class DataGenerator:
    @staticmethod
    def random_oid():
        return "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])

    @staticmethod
    def random_commit_hash():
        return "".join([random.choice(string.hexdigits.lower()) for _ in range(40)])

    @staticmethod
    def random_lfs_metadata():
        version = random.choice(["lfs_version1", "my_version", "version1"])
        oid = "".join([random.choice(string.hexdigits.lower()) for _ in range(64)])
        size = str(random.randint(0, 10000))
        return metadata.LfsMetadata(version=version, oid=oid, size=size)

    @staticmethod
    def random_tensor_metadata():
        ndims = random.choice(range(1, 6))
        shape = tuple([random.choice(range(1, 50)) for _ in range(ndims)])
        tensor = np.random.rand(*shape)
        return metadata.TensorMetadata.from_tensor(tensor)

    @staticmethod
    def random_theta_metadata():
        update_type = random.choice(["dense", "sparse"])
        last_commit = "".join(
            [random.choice(string.hexdigits.lower()) for _ in range(40)]
        )
        return metadata.ThetaMetadata(update_type=update_type, last_commit=last_commit)

    @staticmethod
    def random_param_metadata():
        tensor_metadata = DataGenerator.random_tensor_metadata()
        lfs_metadata = DataGenerator.random_lfs_metadata()
        theta_metadata = DataGenerator.random_theta_metadata()
        return metadata.ParamMetadata(
            tensor_metadata=tensor_metadata,
            lfs_metadata=lfs_metadata,
            theta_metadata=theta_metadata,
        )

    @staticmethod
    def random_nested_dict(
        allowed_keys=list(string.ascii_letters), allowed_values=list(range(100))
    ):
        """Generate random nested dicts for testing."""
        result = {}
        prev = [result]
        curr = result
        for _ in range(random.randint(20, 50)):
            # Pick a key
            key = random.choice(allowed_keys)
            # 50/50, do we make a new nest level?
            if random.choice([True, False]):
                curr[key] = {}
                prev.append(curr)
                curr = curr[key]
                continue
            # Otherwise, add a leaf value
            value = random.choice(allowed_values)
            curr[key] = value
            # 50/50 are we done adding values to this node?
            if random.choice([True, False]):
                curr = prev.pop()
            # If we have tried to to up the tree from the root, stop generating.
            if not prev:
                break
        return result

    @staticmethod
    def random_metadata():
        values = [DataGenerator.random_param_metadata() for _ in range(100)]
        random_metadata_dict = DataGenerator.random_nested_dict(allowed_values=values)
        return metadata.Metadata(random_metadata_dict)

    @staticmethod
    def random_commit_info():
        oids = [DataGenerator.random_oid() for _ in range(random.randint(5, 20))]
        return theta.CommitInfo(oids)


@pytest.fixture
def data_generator():
    return DataGenerator


@pytest.fixture
def git_repo_with_commits():
    commit_infos = [
        DataGenerator.random_commit_info() for _ in range(random.randint(5, 20))
    ]
    commit_hashes = []

    repo_dir = ".delete-me"
    os.mkdir(repo_dir)
    try:
        repo = git.Repo.init(repo_dir)

        config_writer = repo.config_writer(config_level="repository")
        config_writer.set_value("user", "name", "myusername")
        config_writer.set_value("user", "email", "myemail")
        config_writer.release()

        theta_commits = theta.ThetaCommits(repo)

        # Write a bunch of empty commits and random ThetaCommits entries
        for commit_info in commit_infos:
            repo.git.commit("--allow-empty", "-m", "empty commit")
            commit_hash = repo.commit("HEAD").hexsha
            theta_commits.write_commit_info(commit_hash, commit_info)
            commit_hashes.append(commit_hash)

        yield repo, commit_hashes, commit_infos
    finally:
        repo.close()
        git.rmtree(repo_dir)
