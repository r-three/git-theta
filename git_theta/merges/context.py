#!/usr/bin/env python3

import os

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML

from git_theta import git_utils
from git_theta.merges import Merge
from git_theta.types import ParamName
from git_theta.utils import TEXT_STYLE, DiffState, NoResult


def get_other_commit_in_merge() -> str:
    git_heads = [e for e in os.environ.keys() if e.startswith("GITHEAD_")]
    if not git_heads:
        return None
    return git_heads[0].split("_", maxsplit=1)[-1]


def trim_log(log: str, limit: int = 80) -> str:
    if len(log) >= limit - 3:
        return f"{log[:limit - 3]}..."
    return f"{log}"


class Context(Merge):
    DESCRIPTION = f"Show extra information about what {TEXT_STYLE.format_who('us')} vs {TEXT_STYLE.format_who('them')} means."
    NAME = "context"
    SHORT_CUT = "c"
    INACTIVE_STATES = frozenset()

    def merge(self, *args, **kwargs):
        repo = git_utils.get_git_repo()
        other_hash = get_other_commit_in_merge()
        other_commit = repo.commit(other_hash)
        other_branch = repo.git.branch("--contains", other_hash).strip()
        other_log = other_commit.summary

        my_commit = repo.commit("HEAD")
        my_hash = my_commit.hexsha
        my_branch = repo.active_branch
        my_log = my_commit.summary

        print_formatted_text(
            HTML(
                "Merge Context:\n"
                f"\t{TEXT_STYLE.format_who('us')},   {my_hash[:6]} ({my_branch}): {trim_log(my_log)}\n"
                f"\t{TEXT_STYLE.format_who('them')}, {other_hash[:6]} ({other_branch}): {trim_log(other_log)}"
            )
        )

        return NoResult
