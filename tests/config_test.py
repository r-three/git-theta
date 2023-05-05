"""Tests for config.py"""

import os

import git
import pytest

from git_theta import config


@pytest.fixture
def git_repo():
    cwd = os.getcwd()
    repo_dir = os.path.abspath(".delete-me")
    os.mkdir(repo_dir)
    try:
        os.chdir(repo_dir)
        repo = git.Repo.init(repo_dir)
        yield repo
    finally:
        os.chdir(cwd)
        repo.close()
        git.rmtree(repo_dir)


@pytest.fixture
def gitattributes():
    return [
        "*.pt filter=theta merge=theta diff=theta",
        "*.png filter=lfs",
        "really-big-file filter=lfs",
        "something else, who knows how cool it could be",
    ]


def test_add_theta_gitattributes_empty_file(git_repo):
    gitattributes = config.GitAttributesFile(git_repo)
    model_path = "example"
    gitattributes.add_theta(model_path)
    assert (
        gitattributes.serialize() == f"{model_path} filter=theta merge=theta diff=theta"
    )


def test_add_theta_gitattributes_no_match(git_repo):
    with open(os.path.join(git_repo.working_dir, ".gitattributes"), "w") as f:
        f.write(
            "\n".join(
                [
                    "Some-other-path filter=lfs",
                    "*-cool-models.pt filter=theta merge=theta diff=theta",
                ]
            )
        )
    gitattributes = config.GitAttributesFile(git_repo)

    model_path = "path/to/my/model.pt"
    gitattributes.add_theta(model_path)
    assert (
        gitattributes.data[-1].serialize()
        == f"{model_path} filter=theta merge=theta diff=theta"
    )


def test_add_theta_gitattributes_exact_match(git_repo):
    model_path = "really/cool/model/yall.ckpt"

    with open(os.path.join(git_repo.working_dir, ".gitattributes"), "w") as f:
        f.write(f"{model_path} ident")
    gitattributes = config.GitAttributesFile(git_repo)

    gitattributes.add_theta(model_path)

    assert (
        len(gitattributes.data) == 1
        and gitattributes.data[0].serialize()
        == f"{model_path} ident filter=theta merge=theta diff=theta"
    )


def test_add_theta_gitattributes_rest_unchanged(git_repo):
    model_path = "model-v3.pt"

    atts = [
        "some-other-path filter=theta merge=theta diff=theta",
        "really-reaaaally-big-files filter=lfs",
        "another filter=theta merge=theta diff=theta",
    ]

    with open(os.path.join(git_repo.working_dir, ".gitattributes"), "w") as f:
        f.write("\n".join(atts))
    gitattributes = config.GitAttributesFile(git_repo)

    gitattributes.add_theta(model_path)

    for i, att in enumerate(atts):
        assert att == gitattributes.data[i].serialize()


def test_read_gitattributes(gitattributes, tmp_path):
    file = tmp_path / ".gitattributes"
    with open(file, "w") as f:
        f.write("\n".join(gitattributes))

    attrs = config.GitAttributesFile.read(file)
    for true_attr, attr in zip(gitattributes, attrs):
        assert attr.serialize() == true_attr


def test_read_gitattributes_missing_file(tmp_path):
    """Test that gitattributes file missing returns an empty list."""
    missing_file = tmp_path / ".gitattributes"
    assert not os.path.exists(missing_file)
    read_attributes = config.GitAttributesFile.read(missing_file)
    assert read_attributes == []


def test_read_gitattributes_empty_file(tmp_path):
    """Test that gitattributes file being empty returns an empty list."""
    empty_file = tmp_path / ".gitattributes"
    empty_file.touch()
    assert os.path.exists(empty_file)
    read_attributes = config.GitAttributesFile.read(empty_file)
    assert read_attributes == []


def test_read_gitattributes_empty_lines(gitattributes, tmp_path):
    """Test that a gitattributes file with empty lines mixed in is read correctly"""
    file = tmp_path / ".gitattributes"
    with open(file, "w") as f:
        f.write("\n\n".join(gitattributes))

    attrs = config.GitAttributesFile.read(file)
    for true_attr, attr in zip(gitattributes, attrs):
        assert attr.serialize() == true_attr


def test_write_gitattributes(git_repo, gitattributes):
    """Test that attributes are written to file unchanged"""
    with open(".gitattributes", "w") as f:
        f.write("\n".join(gitattributes))

    ga = config.GitAttributesFile(git_repo)
    os.remove(".gitattributes")
    ga.write()

    with open(".gitattributes", "r") as f:
        written_gitattributes = f.readlines()

    for attr, written_attr in zip(gitattributes, written_gitattributes):
        assert attr == written_attr.rstrip()


def test_write_gitattributes_creates_file(git_repo):
    """Make sure writing the git attributes can create the missing file before writing."""
    gitattributes_path = ".gitattributes"
    assert not os.path.exists(gitattributes_path)
    ga = config.GitAttributesFile(git_repo)
    ga.add_theta("my_model")
    ga.write()
    assert os.path.exists(gitattributes_path)


def test_read_write_gitattributes_write_read_round_trip(git_repo, gitattributes):
    """Test that we can write attributes, then read them back and they will match."""
    ga = config.GitAttributesFile(git_repo)
    for line in gitattributes:
        ga.data.append(config.PatternAttributes.from_line(line))
    ga.write()

    ga = config.GitAttributesFile(git_repo)
    assert ga.serialize() == "\n".join(gitattributes)


def test_read_write_gitattributes_read_write_round_trip(git_repo, gitattributes):
    """Test reading attrs from file, writing to new file and verify file contents match."""
    with open(".gitattributes", "w") as f:
        f.write("\n".join(gitattributes))

    ga = config.GitAttributesFile(git_repo)
    os.remove(".gitattributes")
    assert not os.path.exists(".gitattributes")
    ga.write()

    with open(".gitattributes", "r") as f:
        written_gitattributes = f.read().split("\n")

    for attr, written_attr in zip(gitattributes, written_gitattributes):
        assert attr == written_attr
