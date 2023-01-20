"""Merge operations that select one version or another."""

from git_theta import metadata
from git_theta.merges import Merge
from git_theta.utils import DiffState, TEXT_STYLE
from git_theta.types import ParamName


class TakeUs(Merge):
    DESCRIPTION = f"Use {TEXT_STYLE.format_who('our')} change to the parameter."
    NAME = "take_us"
    SHORT_CUT = "tu"
    # If only they made a change take "us" doesn't make sense.
    INACTIVE_STATES = frozenset(
        {DiffState.CHANGED_B, DiffState.ADDED_B, DiffState.DELETED_B}
    )

    def merge(
        self,
        param_name: ParamName,
        paramA: metadata.ParamMetadata,
        paramB: metadata.ParamMetadata,
        paramO: metadata.ParamMetadata,
        *args,
        **kwargs,
    ) -> metadata.ParamMetadata:
        """Grab the changes from branch A (current)."""
        return paramA


class TakeThem(Merge):
    DESCRIPTION = f"Use {TEXT_STYLE.format_who('their')} change to the parameter."
    NAME = "take_them"
    SHORT_CUT = "tt"
    # If only we made a change take "them" doesn't make sense.
    INACTIVE_STATES = frozenset(
        {
            DiffState.CHANGED_A,
            DiffState.ADDED_A,
            DiffState.DELETED_A,
        }
    )

    def merge(
        self,
        param_name: ParamName,
        paramA: metadata.ParamMetadata,
        paramB: metadata.ParamMetadata,
        paramO: metadata.ParamMetadata,
        *args,
        **kwargs,
    ) -> metadata.ParamMetadata:
        """Grab the changes from branch B (other)."""
        return paramB


class TakeOriginal(Merge):
    DESCRIPTION = f"Use the {TEXT_STYLE.format_who('original')} parameter."
    NAME = "take_original"
    SHORT_CUT = "to"
    INACTIVE_STATES = frozenset({})

    def merge(
        self,
        param_name: ParamName,
        paramA: metadata.ParamMetadata,
        paramB: metadata.ParamMetadata,
        paramO: metadata.ParamMetadata,
        *args,
        **kwargs,
    ) -> metadata.ParamMetadata:
        """Grab the changes from the ancestor."""
        return paramO
