"""Classes representing checkpoint metadata files"""

from __future__ import annotations
import hashlib
import dataclasses
from collections import OrderedDict
import re
import json
from typing import ClassVar
import numpy as np
import git
from typing import Union, TextIO, Dict, Tuple, Any
import asyncio

from git_theta import git_utils, utils, lsh
from file_or_name import file_or_name


@dataclasses.dataclass(eq=True)
class MetadataField:
    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self, dict_factory=OrderedDict)


@dataclasses.dataclass(eq=True)
class LfsMetadata(MetadataField):
    version: str
    oid: str
    size: str
    name: ClassVar[str] = "lfs_metadata"

    @property
    def lfs_pointer(self) -> str:
        return f"version {self.version}\noid sha256:{self.oid}\nsize {self.size}\n"

    @classmethod
    def from_pointer(cls, pointer_contents: str) -> LfsMetadata:
        match = re.match(
            r"^version (?P<version>[^\s]+)\noid sha256:(?P<oid>[0-9a-f]{64})\nsize (?P<size>[0-9]+)\n$",
            pointer_contents,
        )
        if match is None:
            raise ValueError(f"Failed to parse pointer file {pointer_contents}")
        return cls(
            version=match.group("version"),
            oid=match.group("oid"),
            size=match.group("size"),
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> LfsMetadata:
        return cls.from_pointer(git_utils.git_lfs_clean(b))


@dataclasses.dataclass(eq=True)
class TensorMetadata(MetadataField):
    shape: str
    dtype: str
    hash: np.ndarray
    name: ClassVar[str] = "tensor_metadata"

    def __post_init__(self):
        self.hash = np.array(self.hash)

    def __eq__(self, other):
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and np.array_equal(self.hash, other.hash)
        )

    @classmethod
    def from_tensor(cls, tensor: np.ndarray) -> TensorMetadata:
        shape = str(tensor.shape)
        dtype = str(tensor.dtype)
        hash = lsh.get_lsh().hash(tensor)
        return cls(shape=shape, dtype=dtype, hash=hash)


@dataclasses.dataclass(eq=True)
class ThetaMetadata(MetadataField):
    update_type: str
    last_commit: str
    name: ClassVar[str] = "theta_metadata"


@dataclasses.dataclass(eq=True)
class ParamMetadata(MetadataField):
    tensor_metadata: TensorMetadata
    lfs_metadata: LfsMetadata
    theta_metadata: ThetaMetadata

    @classmethod
    def from_metadata_dict(cls, d: Dict[str, Any]) -> ParamMetadata:
        tensor_metadata = TensorMetadata(**d[TensorMetadata.name])
        lfs_metadata = LfsMetadata(**d[LfsMetadata.name])
        theta_metadata = ThetaMetadata(**d[ThetaMetadata.name])
        return cls(tensor_metadata, lfs_metadata, theta_metadata)


class Metadata(OrderedDict):
    @classmethod
    def from_metadata_dict(cls, d: Dict[str, Any]) -> Metadata:
        flattened = utils.flatten(d, is_leaf=lambda v: LfsMetadata.name in v)
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = ParamMetadata.from_metadata_dict(param_metadata)
        metadata = utils.unflatten(flattened)
        return cls(metadata)

    @classmethod
    @file_or_name(file="r")
    def from_file(cls, file: TextIO) -> Metadata:
        metadata_dict = json.load(file)
        return cls.from_metadata_dict(metadata_dict)

    @classmethod
    def from_commit(cls, repo: git.Repo, path: str, commit_hash: str) -> Metadata:
        obj = git_utils.get_file_version(repo, path, commit_hash)
        if obj is None:
            return cls()
        else:
            return cls.from_file(obj.data_stream)

    @file_or_name(file="w")
    def write(self, file: TextIO):
        metadata_dict = self.serialize()
        json.dump(metadata_dict, file, indent=4, cls=MetadataEncoder)

    def flatten(self) -> Metadata:
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, ParamMetadata))

    def unflatten(self) -> Metadata:
        return utils.unflatten(self)

    def diff(self, other: Metadata) -> Tuple[Metadata, Metadata, Metadata]:
        self_flattened = self.flatten()
        other_flattened = other.flatten()
        added = Metadata(
            {
                k: self_flattened[k]
                for k in self_flattened.keys() - other_flattened.keys()
            }
        ).unflatten()
        removed = Metadata(
            {
                k: other_flattened[k]
                for k in other_flattened.keys() - self_flattened.keys()
            }
        ).unflatten()
        modified = Metadata()
        for param_keys in set(self_flattened.keys()).intersection(
            other_flattened.keys()
        ):
            if (
                self_flattened[param_keys].lfs_metadata
                != other_flattened[param_keys].lfs_metadata
            ):
                modified[param_keys] = self_flattened[param_keys]

        modified = modified.unflatten()
        return added, removed, modified

    def serialize(self) -> Dict[str, Any]:
        flattened = self.flatten()
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = param_metadata.serialize()
        return flattened.unflatten()


class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


class LazyParam:
    def __init__(
        self,
        tensor: np.ndarray = None,
        tensor_metadata: TensorMetadata = None,
        lfs_metadata: LfsMetadata = None,
    ):
        self._tensor = tensor
        self.tensor_metadata = tensor_metadata
        self.lfs_metadata = lfs_metadata

    @classmethod
    def from_tensor(cls, tensor: np.ndarray):
        tensor_metadata = TensorMetadata.from_tensor(tensor)
        return cls(tensor=tensor, tensor_metadata=tensor_metadata)

    @classmethod
    def from_param_metadata(cls, param_metadata: ParamMetadata):
        if param_metadata is None:
            return None
        return cls(
            lfs_metadata=param_metadata.lfs_metadata,
            tensor_metadata=param_metadata.tensor_metadata,
        )

    def tensor_loaded(self):
        return self._tensor is not None

    @property
    def tensor(self):
        if self._tensor is None:
            self._tensor = asyncio.run(
                git_utils.git_lfs_smudge(self.lfs_metadata.lfs_pointer)
            )
        return self._tensor

    def __eq__(self, other):
        if not isinstance(other, LazyParam):
            return False

        if self.tensor_loaded() and other.tensor_loaded():
            return np.allclose(
                self.tensor,
                other.tensor,
                rtol=utils.EnvVarConstants.PARAMETER_RTOL,
                atol=utils.EnvVarConstants.PARAMETER_ATOL,
            )

        hash_distance = lsh.get_lsh().distance(
            self.tensor_metadata.hash, other.tensor_metadata.hash
        )
        if hash_distance < utils.EnvVarConstants.PARAMETER_ATOL:
            return True
        elif hash_distance < utils.EnvVarConstants.LSH_THRESHOLD:
            return np.allclose(
                self.tensor,
                other.tensor,
                rtol=utils.EnvVarConstants.PARAMETER_RTOL,
                atol=utils.EnvVarConstants.PARAMETER_ATOL,
            )
        else:
            return False
