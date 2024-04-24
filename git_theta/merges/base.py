"""Plugins for Model Merging.

Note:
  In order to dynamically create the menu of possible actions that describe what
  each plug-in does, the plugins get imported at the start of the merge tool.
  Therefore, plug-ins must not have slow side-effects that happen at import-time.
"""


import logging
import sys
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type, Union

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from git_theta import metadata, utils

ParamName = Tuple[str, ...]
Parameter = Any
PartialModel = Dict[ParamName, Parameter]


@dataclass
class MergeArgument:
    """Metadata for how to describe and validate a user-specified merge-strategy-specific argument"""

    name: str
    description: str
    type: Type
    range: Optional[Tuple[Union[float, int], Union[float, int]]]

    @property
    def validator(self):
        """Returns a function checking whether a given string is a valid input for this argument"""

        def is_valid(x):
            # TODO: May need to support non-numeric types at some point
            try:
                x = self.type(x)
                if self.range:
                    return x >= self.range[0] and x <= self.range[1]
                return False
            except:
                return False

        return is_valid


class PrintableABCMeta(ABCMeta):
    """Add custom `str` to /classes/, not objects."""

    def __str__(cls):
        return f"{cls.NAME}: {cls.DESCRIPTION}"


@utils.abstract_classattributes("DESCRIPTION", "NAME", "SHORT_CUT", "INACTIVE_STATES")
class Merge(metaclass=PrintableABCMeta):
    """A Plug-in that handles parameter merging.

    Note:
      Informational string about the plugin can contain `prompt_toolkit`
      supported HTML markup for styling and coloring text.
    """

    DESCRIPTION: str = NotImplemented  # Description of Merge Action, shown in menu.
    NAME: str = NotImplemented  # Unique name of the merge, to look up the plugin with.
    SHORT_CUT: str = (
        NotImplemented  # A Request keyboard shortcut to use during merging.
    )
    INACTIVE_STATES: FrozenSet[
        utils.DiffState
    ] = frozenset()  # States where this action will not appear in the menu.

    def __call__(self, param_name, *args, **kwargs):
        logger = logging.getLogger("git_theta")
        logger.info(f"Running {self.NAME} merge on parameter {'/'.join(param_name,)}")
        return self.merge(param_name, *args, **kwargs)

    @abstractmethod
    def merge(
        self,
        param_name: ParamName,
        paramA: metadata.ParamMetadata,
        paramB: metadata.ParamMetadata,
        paramO: metadata.ParamMetadata,
        metadataA: metadata.Metadata,
        metadataB: metadata.Metadata,
        metadataO: metadata.Metadata,
        modelA: PartialModel,
        modelB: PartialModel,
        modelO: PartialModel,
        path: str,
        **kwargs,
    ) -> metadata.ParamMetadata:
        """Merge parameters parameters.

        Parameters
        ----------
            param_name: The name of the parameter we are looking at.
            paramA: The parameter metadata from branch A (current).
            paramB: The parameter metadata from branch B (other).
            paramO: The parameter metadata from the ancestor.
            metadataA: The full model metadata from branch A (current).
            metadataB: The full model metadata from branch B (other).
            metadataO: The full model metadata from the ancestor.
            modelA: A partially filled in model of real parameter values from
                branch A (current). Allows caching and reuse for any sort of
                "full model" merging method.
            modelB: A partially filled in model of real parameter values from
                branch B (other). Allows caching and reuse for any sort of
                "full model" merging method.
            modelO: A partially filled in model of real parameter values from
                the ancestor. Allows caching and reuse for any sort of
                "full model" merging method.
            path: The path to where the model actually lives.
            kwargs: Merge-strategy-specific arguments.
        """

    @classmethod
    def merge_arguments(self) -> List[MergeArgument]:
        """Returns a list of `MergeArgument`s that provide information about the arguments specific to each merge strategy
        Each `MergeArgument` contains:
            1. The name of the merge argument
            2. A text description of what the argument does
            3. The type of the argument
        """
        return []


def all_merge_handlers() -> Dict[str, Merge]:
    """Enumerate and Load (import) all merge plugins."""
    discovered_plugins = entry_points(group="git_theta.plugins.merges")
    loaded_plugins = {ep.name: ep.load() for ep in discovered_plugins}
    return loaded_plugins
