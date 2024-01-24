"""Tests for git_utils.py"""

import os

import pytest

from git_theta import git_utils


def test_add_theta_gitattributes_empty_file():
    assert list(map(str, git_utils.add_theta_to_gitattributes([], "example"))) == [
        "example filter=theta merge=theta diff=theta"
    ]


def test_is_theta_tracked_with_override():
    attrs = [
        git_utils.parse_gitattributes(a)
        for a in (
            "mymodel.pt filter=theta merge=theta diff=theta",
            "*.pt filter=theta merge=theta diff=theta",
        )
    ]
    print(attrs)
    assert git_utils.is_theta_tracked("mymodel.pt", attrs)


def test_is_theta_tracked_with_override_false():
    attrs = [
        git_utils.parse_gitattributes(a)
        for a in (
            "mymodel.pt filter=theta merge=theta diff=theta",
            "*.pt filter=lfs merge=theta diff=lfs",
        )
    ]
    assert git_utils.is_theta_tracked("mymodel.pt", attrs) == False


def test_is_theta_tracked_no_lines():
    assert git_utils.is_theta_tracked("mymodel.pt", []) == False


def test_is_theta_tracked_no_attrs():
    assert (
        git_utils.is_theta_tracked(
            "mymodel.pt", [git_utils.parse_gitattributes("mymodel.pt")]
        )
        == False
    )


def test_is_theta_tracked_with_following_filter():
    attrs = [
        git_utils.parse_gitattributes(a)
        for a in (
            "mymodel.pt filter=theta merge=theta diff=theta",
            "*.pt filter=theta filter=lfs merge=lfs diff=lfs",
        )
    ]
    assert git_utils.is_theta_tracked("mymodel.pt", attrs) == False


def test_parse_gitattributes_uses_last():
    attr = git_utils.parse_gitattributes("example.txt merge=theta merge=wrong")
    assert attr.attributes["merge"] == "wrong"


def test_parse_gitattributes_no_equal():
    s = "example.txt thing"
    attr = git_utils.parse_gitattributes(s)
    assert attr.attributes == {"thing": None}
    assert str(attr) == s


def test_parse_gitattributes_raw_string():
    og_string = "example.txt merge=theta merge=wrong"
    attr = git_utils.parse_gitattributes(og_string)
    assert str(attr) == og_string
    attr.raw = None
    assert str(attr) == "example.txt merge=wrong"


def test_add_theta_gitattributes_no_match():
    # Should add a new path
    atts = [
        git_utils.parse_gitattributes(a)
        for a in (
            "Some-other-path filter=lfs",
            "*-cool-models.pt filter=theta merge=theta diff=theta",
        )
    ]
    model_path = "path/to/my/model.pt"
    assert (
        str(git_utils.add_theta_to_gitattributes(atts, model_path)[-1])
        == f"{model_path} filter=theta merge=theta diff=theta"
    )


def test_add_theta_gitattributes_exact_match_with_conflicting_attributes():
    model_path = "really/cool/model/yall.ckpt"
    atts = [git_utils.parse_gitattributes(f"{model_path} filter=lfs")]
    with pytest.raises(ValueError):
        new_attributes = git_utils.add_theta_to_gitattributes(atts, model_path)


def test_add_theta_gitattributes_pattern_match_with_conflicting_attributes():
    model_path = "literal-the-best-checkpoint.pt"
    atts = [git_utils.parse_gitattributes("*.pt thing merge=lfs")]
    with pytest.raises(ValueError):
        new_attributes = git_utils.add_theta_to_gitattributes(atts, model_path)


def test_add_theta_gitattributes_exact_match_disjoint_attributes():
    # Should create a new attribute with values copied over
    model_path = "my-test_model"
    atts = [
        git_utils.parse_gitattributes(a)
        for a in ("my-test_model merge=theta diff=theta banana=fruit",)
    ]
    new_att = git_utils.add_theta_to_gitattributes(atts, model_path)[-1]
    assert new_att.attributes["banana"] == "fruit"
    assert new_att.attributes["filter"] == "theta"


def test_add_theta_gitattributes_pattern_disjoint_attributes():
    # Should create a new attribute with values copied over
    model_path = "my-test_model"
    atts = [
        git_utils.parse_gitattributes(a)
        for a in ("my-test* merge=theta diff=theta banana=fruit",)
    ]
    new_att = git_utils.add_theta_to_gitattributes(atts, model_path)[-1]
    assert new_att.pattern == model_path
    assert new_att.attributes["banana"] == "fruit"
    assert new_att.attributes["filter"] == "theta"


def test_add_theta_gitattributes_disjoint_attributes_multiple_matches():
    # Should create a new attribute with values copied over
    model_path = "100-on-mnist.npy"
    atts = [
        git_utils.parse_gitattributes(a)
        for a in ("*.npy other-filter", f"{model_path} target-filter")
    ]
    new_attributes = git_utils.add_theta_to_gitattributes(atts, model_path)
    # Note: target-filter is expected rather than other-filter because the *last*
    # filter in the file is the active one.
    assert (
        str(new_attributes[-1])
        == f"{model_path} target-filter filter=theta merge=theta diff=theta"
    )


def test_add_theta_gitattributes_match_with_theta_already():
    # Should be a no-op
    model_path = "my-bad-model.chkp"
    atts = [
        git_utils.parse_gitattributes(a)
        for a in (
            "my-*-model.chkp filter=theta merge=theta diff=theta",
            "example.txt thing",
        )
    ]
    new_attributes = git_utils.add_theta_to_gitattributes(atts, model_path)
    assert new_attributes == atts


# This should fail until unsetting attributes are handled.
@pytest.mark.xfail
def test_add_theta_gitattributes_unset_diff():
    # The attribute represention (the dict) my change when unsetting is implemented.
    attr = git_utils.GitAttributes("example.pt", {"-diff": None})
    with pytest.raises(ValueError):
        new_attributes = git_utils.add_theta_to_gitattributes([attr], "example.pt")


def test_add_theta_gitattributes_rest_unchanged():
    model_path = "model-v3.pt"
    atts = [
        git_utils.parse_gitattributes(a)
        for a in (
            "some-other-path filter=theta merge=theta diff=theta",
            "really-reaaaally-big-files filter=lfs",
            r"model-v\d.pt filter",
            "another filter=theta merge=theta diff=theta",
        )
    ]
    results = git_utils.add_theta_to_gitattributes(atts, model_path)
    for i, (a, r) in enumerate(zip(atts, results)):
        if i == 2:
            continue
        assert a == r


@pytest.fixture
def gitattributes():
    return [
        "*.pt filter=theta merge=theta diff=theta",
        "*.png filter=lfs",
        "really-big-file filter=lfs",
        "something else",
    ], [
        git_utils.GitAttributes(
            "*.pt", {"filter": "theta", "merge": "theta", "diff": "theta"}
        ),
        git_utils.GitAttributes("*.png", {"filter": "lfs"}),
        git_utils.GitAttributes("really-big-file", {"filter": "lfs"}),
        git_utils.GitAttributes("something", {"else": None}),
    ]


def test_read_gitattributes(gitattributes, tmp_path):
    """Test that reading gitattributes removes newlines."""
    attributes_text, gitattributes = gitattributes
    gitattributes_file = tmp_path / ".gitattributes"
    with open(gitattributes_file, "w") as wf:
        wf.write("\n".join(attributes_text))
    read_attributes = git_utils.read_gitattributes(gitattributes_file)
    assert read_attributes == gitattributes


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
    gitattributes = gitattributes[0]
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
    attributes_text, gitattributes = gitattributes
    attr_file = tmp_path / ".gitattributes"
    git_utils.write_gitattributes(attr_file, attributes_text)
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
