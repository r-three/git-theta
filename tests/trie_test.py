"""Test for the Trie (O(n) prefix lookup)."""

from git_theta import utils


def test_trie_insert():
    word = "homework"
    t = utils.Trie()
    assert word not in t
    t.insert(word)
    assert word in t


def test_trie_insert_then_prefix():
    word = "homework"
    t = utils.Trie()
    t.insert(word)
    assert t.prefix("home")


def test_trie_contians_miss():
    t = utils.Trie()
    t.insert("homework")
    t.insert("television")
    t.insert("homeish")
    assert "homelab" not in t


def test_trie_contains_prefix():
    t = utils.Trie()
    t.insert("homework")
    assert "homework" in t
    assert t.prefix("home")
    assert "home" not in t


def test_trie_word_not_prefix():
    base = "home"
    full = f"{base}work"
    t = utils.Trie()
    t.insert(base)
    assert base in t
    assert not t.prefix(base)
    assert full not in t
    t.insert(full)
    assert full in t
    assert t.prefix(base)


def test_trie_iterable():
    added_words = [
        "apple",
        "banana",
        "table",
    ]
    missing_words = ["laptop", "cable"]
    t = utils.Trie.from_iterable(added_words)
    for added in added_words:
        assert added in t
    for miss in missing_words:
        assert miss not in t
