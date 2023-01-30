"""Merge parameters by averaging them."""

from typing import Dict, Any, Sequence
import numpy as np

from git_theta.models import Metadata, ParamMetadata, TensorMetadata, ThetaMetadata
from git_theta.merges import Merge
from git_theta.utils import DiffState, TEXT_STYLE
from git_theta.types import ParamName
from git_theta import updates
from git_theta import params
from git_theta import async_utils
from git_theta import git_utils


PartialModel = Dict[ParamName, Any]
Parameter = Any


# TODO: Add configurable interpolation parameters.
# TODO: Control this class explosion, Average Merge can have it's own menu?
# TODO: Move from a set of Inactive States to Active States?
class Average(Merge):
    DESCRIPTION = f"Average {TEXT_STYLE.format_who('our')} change and {TEXT_STYLE.format_who('their')} change together."
    NAME = "average-ours-theirs"
    SHORT_CUT = "avg-ab"
    INACTIVE_STATES = frozenset(
        {
            # Averaging with the original parameter doesn't make sense?
            DiffState.CHANGED_A,
            DiffState.CHANGED_B,
            # Averaging with only one added parameter doesn't make sense.
            DiffState.ADDED_A,
            DiffState.ADDED_B,
            # Averaging with a deleted parameter doesn't make sense
            DiffState.DELETED_A,
            DiffState.DELETED_B,
            DiffState.DELETED_BOTH,
        }
    )

    def average(self, *params: Sequence[Parameter]) -> Parameter:
        # Will convert to [#params, *param.shape] so sum over the leading dim.
        return np.sum(params, axis=0) / len(params)

    def read_parameter(
        self, param: ParamMetadata, param_name: ParamName, path: str
    ) -> Parameter:
        update_handler = updates.get_update_handler(param.theta_metadata.update_type)(
            params.get_update_serializer()
        )
        return async_utils.run(
            update_handler.apply(param, param_name, git_utils.get_git_repo(), path)
        )

    def write_merged(self, averaged: Parameter, param_name: ParamName):
        tensor_metadata = TensorMetadata.from_tensor(averaged)
        update_handler = updates.get_update_handler("dense")(
            params.get_update_serializer()
        )
        theta_metadata = ThetaMetadata("dense", None)
        # Dense only needs these two...
        lfs_metadata = async_utils.run(update_handler.write(averaged, param_name))
        return ParamMetadata(
            lfs_metadata=lfs_metadata,
            tensor_metadata=tensor_metadata,
            theta_metadata=theta_metadata,
        )

    def merge(
        self,
        param_name: ParamName,
        paramA: ParamMetadata,
        paramB: ParamMetadata,
        paramO: ParamMetadata,
        metadataA: Metadata,
        metadataB: Metadata,
        metadataO: Metadata,
        modelA: PartialModel,
        modelB: PartialModel,
        modelO: PartialModel,
        path: str,
    ) -> ParamMetadata:
        # Load the current parameter
        paramA = self.read_parameter(paramA, param_name, path)
        # Load the other parameter
        paramB = self.read_parameter(paramB, param_name, path)
        result = self.average(paramA, paramB)
        return self.write_merged(result, param_name)


class AverageAll(Average):
    DESCRIPTION = f"Average {TEXT_STYLE.format_who('our')} change, {TEXT_STYLE.format_who('their')} change, and the {TEXT_STYLE.format_who('original')} parameter together."
    NAME = "average-all"
    SHORT_CUT = "avg-all"
    INACTIVE_STATES = frozenset(
        {
            # Need values for all 3 parameters.
            DiffState.CHANGED_A,
            DiffState.CHANGED_B,
            # Averaging with only one added parameter doesn't make sense.
            DiffState.ADDED_A,
            DiffState.ADDED_B,
            # Averaging with a deleted parameter doesn't make sense
            DiffState.DELETED_A,
            DiffState.DELETED_B,
            DiffState.DELETED_BOTH,
        }
    )

    def merge(
        self,
        param_name: ParamName,
        paramA: ParamMetadata,
        paramB: ParamMetadata,
        paramO: ParamMetadata,
        metadataA: Metadata,
        metadataB: Metadata,
        metadataO: Metadata,
        modelA: PartialModel,
        modelB: PartialModel,
        modelO: PartialModel,
        path: str,
    ) -> ParamMetadata:
        # Load the current parameter
        paramA = self.read_parameter(paramA, param_name, path)
        # Load the other parameter
        paramB = self.read_parameter(paramB, param_name, path)
        # Load the original parameter
        paramO = self.read_parameter(paramO, param_name, path)
        result = self.average(paramA, paramB, paramO)
        return self.write_merged(result, param_name)


class AverageOursOriginal(Average):
    DESCRIPTION = f"Average {TEXT_STYLE.format_who('our')} change and the {TEXT_STYLE.format_who('original')} parameter together."
    NAME = "average-ours-original"
    SHORT_CUT = "avg-ao"
    INACTIVE_STATES = frozenset(
        {
            # Need a change in A
            DiffState.CHANGED_B,
            # Can average when a parameter wasn't there
            DiffState.ADDED_A,
            DiffState.ADDED_B,
            DiffState.ADDED_BOTH,
            # Averaging with a deleted parameter doesn't make sense
            DiffState.DELETED_A,
            DiffState.DELETED_B,
            DiffState.DELETED_BOTH,
        }
    )

    def merge(
        self,
        param_name: ParamName,
        paramA: ParamMetadata,
        paramB: ParamMetadata,
        paramO: ParamMetadata,
        metadataA: Metadata,
        metadataB: Metadata,
        metadataO: Metadata,
        modelA: PartialModel,
        modelB: PartialModel,
        modelO: PartialModel,
        path: str,
    ) -> ParamMetadata:
        # Load the current parameter
        paramA = self.read_parameter(paramA, param_name, path)
        # Load the original parameter
        paramO = self.read_parameter(paramO, param_name, path)
        result = self.average(paramA, paramO)
        return self.write_merged(result, param_name)


class AverageTheirsOriginal(Average):
    DESCRIPTION = f"Average {TEXT_STYLE.format_who('their')} change and the {TEXT_STYLE.format_who('original')} parameter together."
    NAME = "average-theirs-original"
    SHORT_CUT = "avg-to"
    INACTIVE_STATES = frozenset(
        {
            # Need a change in B
            DiffState.CHANGED_A,
            # Can't average when a parameter wasn't there
            DiffState.ADDED_A,
            DiffState.ADDED_B,
            DiffState.ADDED_BOTH,
            # Averaging with a deleted parameter doesn't make sense
            DiffState.DELETED_A,
            DiffState.DELETED_B,
            DiffState.DELETED_BOTH,
        }
    )

    def merge(
        self,
        param_name: ParamName,
        paramA: ParamMetadata,
        paramB: ParamMetadata,
        paramO: ParamMetadata,
        metadataA: Metadata,
        metadataB: Metadata,
        metadataO: Metadata,
        modelA: PartialModel,
        modelB: PartialModel,
        modelO: PartialModel,
        path: str,
    ) -> ParamMetadata:
        # Load the other parameter
        paramB = self.read_parameter(paramB, param_name, path)
        # Load the original parameter
        paramO = self.read_parameter(paramO, param_name, path)
        result = self.average(paramB, paramO)
        return self.write_merged(result, param_name)
