"""Tests for theta.py"""

import pytest
import os
import tempfile
import git
import random
import string

from git_theta import theta, git_utils
from tests import testing_utils


@pytest.fixture
def git_repo_with_commits():
    commit_infos = [random_commit_info() for _ in range(random.randint(5, 20))]
    commit_hashes = []

    with tempfile.TemporaryDirectory() as repo_dir:
        repo = git.Repo.init(repo_dir)
        theta_commits = theta.ThetaCommits(repo)
        # Write a bunch of empty commits and random ThetaCommits entries
        for commit_info in commit_infos:
            repo.git.commit("--allow-empty", "-m", "empty commit")
            commit_hash = repo.commit("HEAD").hexsha
            theta_commits.write_commit_info(commit_hash, commit_info)
            commit_hashes.append(commit_hash)

        yield repo, commit_hashes, commit_infos


def random_commit_info():
    oids = [testing_utils.random_oid() for _ in range(random.randint(5, 20))]
    return theta.CommitInfo(oids)


def test_commit_info_serialization():
    """
    Test that CommitInfo objects serialize/deserialize to/from files correctly
    """
    commit_info = random_commit_info()
    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        commit_info.write(tmpfile)
        tmpfile.flush()
        commit_info_read = theta.CommitInfo.from_file(tmpfile.name)
    assert commit_info == commit_info_read


def test_get_commit_info(git_repo_with_commits):
    """
    Test getting the correct CommitInfo object for a certain commit hash using a ThetaCommits object
    """
    repo, commit_hashes, commit_infos = git_repo_with_commits
    theta_commits = theta.ThetaCommits(repo)

    for commit_hash, commit_info in zip(commit_hashes, commit_infos):
        assert theta_commits.get_commit_info(commit_hash) == commit_info


def test_get_commit_info_range(git_repo_with_commits):
    """
    Test getting the correct CommitInfo objects for a certain commit hash range using a ThetaCommits object
    """
    repo, commit_hashes, commit_infos = git_repo_with_commits
    theta_commits = theta.ThetaCommits(repo)
    for _ in range(100):
        start = random.randint(0, len(commit_hashes) - 1)  # Exclusive
        end = random.randint(start, len(commit_hashes) - 1)  # Inclusive
        # Returned in reverse chronological order so reverse the result so it matches the order of commit_infos
        commit_info_range = list(
            reversed(
                theta_commits.get_commit_info_range(
                    commit_hashes[start], commit_hashes[end]
                )
            )
        )
        assert len(commit_info_range) == (end - start)
        for idx, commit_info in enumerate(commit_info_range):
            assert commit_info == commit_infos[start + idx + 1]


def test_get_commit_oids(git_repo_with_commits):
    """
    Test getting the correct object-ids for a certain commit hash using a ThetaCommits object
    """
    repo, commit_hashes, commit_infos = git_repo_with_commits
    theta_commits = theta.ThetaCommits(repo)
    for commit_hash, commit_info in zip(commit_hashes, commit_infos):
        assert theta_commits.get_commit_oids(commit_hash) == commit_info.oids


def test_combine_oid_sets():
    """
    Test combining multiple object-id sets into a single set
    """
    oid_sets = [set([1, 2, 3, 4]), set([1, 2]), set([1, 2, 6])]
    combined_set = set([1, 2, 3, 4, 6])
    assert theta.ThetaCommits.combine_oid_sets(oid_sets) == combined_set
