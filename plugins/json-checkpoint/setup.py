#!/usr/bin/env python3

from setuptools import setup


setup(
    name="git_theta_json_checkpoint",
    description="Demo plugin for the git theta VCS.",
    install_requires=[
        "git_theta",
    ],
    packages=["git_theta_json_checkpoint"],
    entry_points={
        "git_theta.plugins.checkpoints": [
            "json = git_theta_json_checkpoint.checkpoints:JSONCheckpoint",
        ]
    },
)
