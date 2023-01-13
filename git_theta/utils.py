"""Utilities for git theta."""

import dataclasses
from enum import Enum
from types import MethodType
from typing import Dict, Any, Tuple, Union, Callable, Iterable, Optional
import re
import functools

import os
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Tuple, Union, Callable


def _format(self, value, tag):
    """Wrap `value` in HTML like <tag>s."""
    return f"<{getattr(self, tag)}>{value}</{getattr(self, tag)}>"


# TODO: Make configurable
@dataclasses.dataclass
class TextStyle:
    param: str = "purple"
    model: str = "cyan"
    who: str = "u"
    changed: str = "yellow"
    added: str = "green"
    deleted: str = "red"

    # TODO: Move this to classlevel code gen?
    def __post_init__(self):
        # Add format_field() methods to the object for each field.
        for field in dataclasses.fields(self):
            field = field.name
            method = MethodType(functools.partial(_format, tag=field), self)
            setattr(self, f"format_{field}", method)


TEXT_STYLE = TextStyle()


# TODO: Defer the creation of enum text so we don't need an instantiated
# TextStyle at import time.
class DiffState(Enum):
    EQUAL = "All parameter values are equal."
    CHANGED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    CHANGED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    CHANGED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    DELETED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    DELETED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    DELETED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    ADDED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_added('added')}</b> this parameter."
    ADDED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_added('added')}</b> this parameter."
    ADDED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_added('added')}</b> this parameter."


class EnvVarConstants:
    CHECKPOINT_TYPE: str = "GIT_THETA_CHECKPOINT_TYPE"
    UPDATE_TYPE: str = "GIT_THETA_UPDATE_TYPE"


def flatten(
    d: Dict[str, Any],
    is_leaf: Callable[[Any], bool] = lambda v: not isinstance(v, dict),
) -> Dict[Tuple[str, ...], Any]:
    """Flatten a nested dictionary.

    Parameters
    ----------
    d:
        The nested dictionary to flatten.

    Returns
    -------
    Dict[Tuple[str, ...], Any]
        The flattened version of the dictionary where the key is now a tuple
        of keys representing the path of keys to reach the value in the nested
        dictionary.
    """

    def _flatten(d, prefix: Tuple[str] = ()):
        flat = type(d)({})
        for k, v in d.items():
            if not is_leaf(v):
                flat.update(_flatten(v, prefix=prefix + (k,)))
            else:
                flat[prefix + (k,)] = v
        return flat

    return _flatten(d)


def unflatten(d: Dict[Tuple[str, ...], Any]) -> Dict[str, Union[Dict[str, Any], Any]]:
    """Unflatten a dict into a nested one.

    Parameters
    ----------
    d:
        The dictionary to unflatten. Each key should be a tuple of keys the
        represent the nesting.

    Returns
    Dict
        The nested version of the dictionary.
    """
    nested = type(d)({})
    for ks, v in d.items():
        curr = nested
        for k in ks[:-1]:
            curr = curr.setdefault(k, {})
        curr[ks[-1]] = v
    return nested


class ModelRepresentation(OrderedDict, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def is_leaf(l) -> bool:
        """
        Return whether the passed value is a leaf node in the model representation
        """

    @staticmethod
    @abstractmethod
    def leaves_equal(l1, l2) -> bool:
        """
        Return whether the two passed leaves in the model representation are equal (for the purposes of diff-ing)
        """

    def flatten(self):
        """
        Flatten a nested dictionary representing a model
        """
        return flatten(self, is_leaf=self.is_leaf)

    def unflatten(self):
        """
        Unflatten a flattened model representation
        """
        return unflatten(self)

    @classmethod
    def diff(cls, m1, m2):
        m1_flat = m1.flatten()
        m2_flat = m2.flatten()
        added = cls(
            {k: m1_flat[k] for k in m1_flat.keys() - m2_flat.keys()}
        ).unflatten()
        removed = cls(
            {k: m2_flat[k] for k in m2_flat.keys() - m1_flat.keys()}
        ).unflatten()
        modified = cls()
        for param_keys in set(m1_flat.keys()).intersection(m2_flat.keys()):
            if not cls.leaves_equal(m1_flat[param_keys], m2_flat[param_keys]):
                modified[param_keys] = m1_flat[param_keys]

        modified = modified.unflatten()
        return added, removed, modified


def is_valid_oid(oid: str) -> bool:
    """Check if an LFS object-id is valid

    Parameters
    ----------
    oid:
        LFS object-id

    Returns
    bool
        Whether this object-id is valid
    """
    return re.match("^[0-9a-f]{64}$", oid) is not None


def is_valid_commit_hash(commit_hash: str) -> bool:
    """Check if a git commit hash is valid

    Parameters
    ----------
    commit_hash
        Git commit hash

    Returns
    bool
        Whether this commit hash is valid
    """
    return re.match("^[0-9a-f]{40}$", commit_hash) is not None


class Trie:
    """Data structure for O(n) prefix existence checking."""

    def __init__(self, char: Optional[str] = None):
        self.char = char  # Really only for debugging.
        self.next: Dict[str, Trie] = {}
        self.is_word = False

    def insert(self, word: str):
        # If there are no more letters to add we must be a full word.
        if not word:
            self.is_word = True
            return
        # Get the node for the next character.
        first_char = word[0]
        # Explicit checks over something like ".get" to ensure we only make a
        # node if we have too. Also avoid defaultdict so we don't accidentally
        # create a node during search.
        if first_char not in self.next:
            node = self.__class__(first_char)
            self.next[first_char] = node
        else:
            node = self.next[first_char]
        suffix = word[1:]
        # Recurse
        node.insert(suffix)

    def _query(self, word: str) -> "Trie":
        """Find the Node associated with `word`."""
        # If there are no more characters we are the end of the line.
        if not word:
            return self
        # Get the next node, raise key error if prefix + char is not one we have
        # seen before.
        return self.next[word[0]]._query(word[1:])

    def prefix(self, word: str) -> bool:
        # Note, a full word needs to be part of a longer word to be a prefix.
        # i.e. It has characters that come after it.
        try:
            node = self._query(word)
            # Does the node have continuations?
            return bool(node.next)
        # We didn't make it until the end of the word.
        except KeyError:
            return False

    def __contains__(self, word: str) -> bool:
        # Note, this is O(n) so if there is a set of words it would be faster to
        # check that.
        try:
            node = self._query(word)
            # Does the node represent a word?
            return node.is_word
        # We didn't make it until the end of the word.
        except KeyError:
            return False

    @classmethod
    def from_iterable(cls, words: Iterable[str]) -> "Trie":
        """Build a trie for some collection of words."""
        root = cls()
        for word in words:
            root.insert(word)
        return root

    def __str__(self):
        return f"{self.__class__.__name__}(char={self.char}, is_word={self.is_word}, next={self.next.keys()})"
