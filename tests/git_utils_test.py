#!/usr/bin/env python3

from git_theta import git_utils


def test_add_filter_gitattributes_empty_file():
    assert git_utils.add_filter_theta_to_gitattributes([], "example") == [
        "example filter=theta\n"
    ]


def test_add_filter_gitattributes_no_match():
    atts = [
        "Some-other-path filter=lfs\n",
        "*-cool-models.pt filter=theta\n",
    ]
    model_path = "path/to/my/model.pt"
    assert (
        git_utils.add_filter_theta_to_gitattributes(atts, model_path)[-1]
        == f"{model_path} filter=theta\n"
    )


def test_add_filter_gitattributes_exact_match():
    model_path = "really/cool/model/yall.ckpt"
    atts = [f"{model_path} filter=lfs\n"]
    assert (
        git_utils.add_filter_theta_to_gitattributes(atts, model_path)[-1]
        == f"{model_path} filter=lfs filter=theta\n"
    )


def test_add_filter_gitattributes_pattern_match():
    model_path = "literal-the-best-checkpoint.pt"
    atts = ["*.pt thing\n"]
    assert (
        git_utils.add_filter_theta_to_gitattributes(atts, model_path)[-1]
        == f"*.pt thing filter=theta\n"
    )


def test_add_filter_gitattributes_multiple_matches():
    model_path = "100-on-mnist.npy"
    atts = ["*.npy\n", f"{model_path}\n"]
    assert git_utils.add_filter_theta_to_gitattributes(atts, model_path) == [
        "*.npy filter=theta\n",
        f"{model_path} filter=theta\n",
    ]


def test_add_filter_gitattributes_match_with_theta_already():
    model_path = "my-bad-model.chkp"
    atts = ["my-*-model.chkp filter=theta"]
    assert git_utils.add_filter_theta_to_gitattributes(atts, model_path) == atts


def test_add_filter_gitattributes_rest_unchanged():
    model_path = "model-v3.pt"
    atts = [
        "some-other-path filter=theta\n",
        "really-reaaaally-big-files filter=lfs\n",
        r"model-v\d.pt\n",
        "another filter=theta\n",
    ]
    results = git_utils.add_filter_theta_to_gitattributes(atts, model_path)
    for i, (a, r) in enumerate(zip(atts, results)):
        if i == 2:
            continue
        assert a == r


def test_add_filter_gitattributes_all_newlines():
    atts = [f"{x}\n" for x in list("abcdef")]
    for gitattr in git_utils.add_filter_theta_to_gitattributes(atts, "b"):
        assert gitattr.endswith("\n")
