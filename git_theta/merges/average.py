"""Merge parameters by averaging them."""

from typing import Dict, Any
from git_theta.models import Metadata, ParamMetadata, TensorMetadata, ThetaMetadata
from git_theta.merges import Merge
from git_theta.utils import DiffState
from git_theta.types import ParamName
from git_theta import updates
from git_theta import params
from git_theta import async_utils
from git_theta import git_utils


PartialModel = Dict[ParamName, Any]
Parameter = Any


# TODO: Add a configurable interpolation parameter.
class Average(Merge):
    DESCRIPTION = "Average our change and their change together."
    NAME = "average"
    SHORT_CUT = "avg"
    INACTIVE_STATES = {
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

    def average(self, a: Parameter, b: Parameter) -> Parameter:
        return a + b / 2

    def read_parameter(
        self, param: ParamMetadata, param_name: ParamName, path: str
    ) -> Parameter:
        update_handler = updates.get_update_handler(param.theta_metadata.update_type)(
            params.get_update_serializer()
        )
        return async_utils.run(
            update_handler.apply(param, param_name, git_utils.get_git_repo(), path)
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

        tensor_metadata = TensorMetadata.from_tensor(result)
        update_handler = updates.get_update_handler("dense")(
            params.get_update_serializer()
        )
        theta_metadata = ThetaMetadata("dense", None)
        # Dense only needs these two...
        lfs_metadata = async_utils.run(update_handler.write(result, param_name))
        return ParamMetadata(
            lfs_metadata=lfs_metadata,
            tensor_metadata=tensor_metadata,
            theta_metadata=theta_metadata,
        )
