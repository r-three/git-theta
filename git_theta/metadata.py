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
    hash: str
    name: ClassVar[str] = "tensor_metadata"

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
        json.dump(metadata_dict, file, indent=4)

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
                self_flattened[param_keys].tensor_metadata
                != other_flattened[param_keys].tensor_metadata
            ):
                modified[param_keys] = self_flattened[param_keys]

        modified = modified.unflatten()
        return added, removed, modified

    def serialize(self) -> Dict[str, Any]:
        flattened = self.flatten()
        for param_keys, param_metadata in flattened.items():
            flattened[param_keys] = param_metadata.serialize()
        return flattened.unflatten()
