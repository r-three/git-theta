"""Tests for git_utils.py"""

import os
import pytest

from git_theta import git_utils


def test_add_theta_gitattributes_empty_file():
    assert git_utils.add_theta_to_gitattributes([], "example") == [
        "example filter=theta merge=theta"
    ]


def test_add_theta_gitattributes_no_match():
    atts = [
        "Some-other-path filter=lfs",
        "*-cool-models.pt filter=theta merge=theta",
    ]
    model_path = "path/to/my/model.pt"
    assert (
        git_utils.add_theta_to_gitattributes(atts, model_path)[-1]
        == f"{model_path} filter=theta merge=theta"
    )


def test_add_theta_gitattributes_exact_match():
    model_path = "really/cool/model/yall.ckpt"
    atts = [f"{model_path} filter=lfs"]
    assert (
        git_utils.add_theta_to_gitattributes(atts, model_path)[-1]
        == f"{model_path} filter=lfs filter=theta merge=theta"
    )


def test_add_theta_gitattributes_pattern_match():
    model_path = "literal-the-best-checkpoint.pt"
    atts = ["*.pt thing"]
    assert (
        git_utils.add_theta_to_gitattributes(atts, model_path)[-1]
        == f"*.pt thing filter=theta merge=theta"
    )


def test_add_theta_gitattributes_multiple_matches():
    model_path = "100-on-mnist.npy"
    atts = ["*.npy other-filter", f"{model_path} other-filter"]
    assert git_utils.add_theta_to_gitattributes(atts, model_path) == [
        f"{attr} filter=theta merge=theta" for attr in atts
    ]


def test_add_theta_gitattributes_match_with_theta_already():
    model_path = "my-bad-model.chkp"
    atts = ["my-*-model.chkp filter=theta merge=theta"]
    assert git_utils.add_theta_to_gitattributes(atts, model_path) == atts


def test_add_theta_gitattributes_rest_unchanged():
    model_path = "model-v3.pt"
    atts = [
        "some-other-path filter=theta merge=theta",
        "really-reaaaally-big-files filter=lfs",
        r"model-v\d.pt filter",
        "another filter=theta merge=theta",
    ]
    results = git_utils.add_theta_to_gitattributes(atts, model_path)
    for i, (a, r) in enumerate(zip(atts, results)):
        if i == 2:
            continue
        assert a == r


@pytest.fixture
def gitattributes():
    return [
        "*.pt filter=theta merge=theta",
        "*.png filter=lfs",
        "really-big-file filter=lfs",
        "something else, who knows how cool it could be",
    ]


def test_read_gitattributes(gitattributes, tmp_path):
    """Test that reading gitattributes removes newlines."""
    gitattributes_file = tmp_path / ".gitattributes"
    with open(gitattributes_file, "w") as wf:
        wf.write("\n".join(gitattributes))
    read_attributes = git_utils.read_gitattributes(gitattributes_file)
    for attr in read_attributes:
        assert not attr.endswith("\n")


def test_read_gitattributes_missing_file(tmp_path):
    """Test that gitattributes file missing returns an empty list."""
    missing_file = tmp_path / ".gitattributes"
    assert not os.path.exists(missing_file)
    read_attributes = git_utils.read_gitattributes(missing_file)
    assert read_attributes == []


def test_read_gitattributes_empty_file(tmp_path):
    """Test that gitattributes file being empty returns an empty list."""
    empty_file = tmp_path / ".gitattributes"
    empty_file.touch()
    assert os.path.exists(empty_file)
    read_attributes = git_utils.read_gitattributes(empty_file)
    assert read_attributes == []


def test_write_gitattributes(gitattributes, tmp_path):
    """Test that attributes are written to file unchanged and include newlines."""
    attr_file = tmp_path / ".gitattributes"
    for attr in gitattributes:
        assert not attr.endswith("\n")
    git_utils.write_gitattributes(attr_file, gitattributes)
    with open(attr_file) as wf:
        written_attrs = wf.readlines()
    # Check for the newlines which I purposely left on with my reading code.
    for written_attr, attr in zip(written_attrs, gitattributes):
        assert written_attr == f"{attr}\n"


def test_write_gitattributes_ends_in_newline(gitattributes, tmp_path):
    """Make sure we have a final newline when writing out file."""
    attr_file = tmp_path / ".gitattributes"
    git_utils.write_gitattributes(attr_file, gitattributes)
    with open(attr_file) as f:
        attrs = f.read()
    assert attrs[-1] == "\n"


def test_write_gitattributes_creates_file(gitattributes, tmp_path):
    """Make sure writing the git attributes can create the missing file before writing."""
    attr_file = tmp_path / ".gitattributes"
    assert not os.path.exists(attr_file)
    git_utils.write_gitattributes(attr_file, gitattributes)
    assert os.path.exists(attr_file)


def test_read_write_gitattributes_write_read_round_trip(gitattributes, tmp_path):
    """Test that we can write attributes, then read them back and they will match."""
    attr_file = tmp_path / ".gitattributes"
    git_utils.write_gitattributes(attr_file, gitattributes)
    read_attrs = git_utils.read_gitattributes(attr_file)
    assert read_attrs == gitattributes


def test_read_write_gitattributes_read_write_round_trip(gitattributes, tmp_path):
    """Test reading attrs from file, writing to new file and verify file contents match."""
    attr_file = tmp_path / ".gitattributes"
    with open(attr_file, "w") as wf:
        wf.writelines([f"{attr}\n" for attr in gitattributes])

    new_attr_file = tmp_path / ".gitattributes-2"
    read_attrs = git_utils.read_gitattributes(attr_file)
    git_utils.write_gitattributes(new_attr_file, read_attrs)

    with open(attr_file) as old_f:
        old_atts = old_f.read()
    with open(new_attr_file) as new_f:
        new_atts = new_f.read()

    assert old_atts == new_atts
